"""LTX-2.3 영상 워커 — 몸통 (런포드에 의존하지 않는 부분)

이 파일은 런포드를 import 하지 않는다. 로컬에서도 그대로 쓴다.
런포드 진입점은 handler_ltx.py 에 따로 있다.

  ltx_core.py     ⬅ 이 파일. 파일 받기 / ComfyUI 띄우기 / 워크플로 조립 / 실행 / 결과 꺼내기
  handler_ltx.py  ⬅ 껍데기. runpod.serverless.start() 만 있다

설계 문서: logb/2.3영상워커_설계.md
배선 근거: unsloth 가 mp4 에 심어 배포한 검증 워크플로 (노드 39개 전수 판독)
           logb/2.3영상모델_설계.md 10-2절

⚠️ 이 파일에 검사문(assert)을 함부로 붙이지 말 것.
   크레아2 빌드 실패 3건이 전부 나중에 덧붙인 검사문에서 났다.
"""

import base64
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse

import requests
from huggingface_hub import hf_hub_download

# ──────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────

COMFY_HOST = "127.0.0.1"
COMFY_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"

# ComfyUI 가 사진·영상·소리를 읽어가는 폴더.
# 워커와 ComfyUI 가 같은 컨테이너 안에 있으므로 파일로 직접 놓으면 된다
# (HTTP 업로드 /upload/image 를 쓸 필요가 없다).
COMFY_INPUT_DIR = os.environ.get("COMFYUI_INPUT_DIR", "/comfyui/input")

# ⭐ 노드 결과 캐시 방식. 기본을 none 으로 둔 이유는 두 가지다.
#
# ① RAM 이 터진다
#    ComfyUI 는 컨테이너 제한을 모르고 호스트 전체 RAM 을 기준으로 캐시를 쟁여둔다.
#    2026-08-09 실측: 워커가 "total RAM 515597 MB"(503GB, 호스트 전체)로 인식했는데
#    컨테이너 몫은 46.57GiB 뿐이었고, RAM 이 99% 까지 찼다.
#    ComfyUI 기본값이 "inactive = 시스템 RAM 의 100%(최대 128GB)" 라서
#    128GB 까지 써도 된다고 믿고 계속 쌓는다. 46.57GB 에서 컨테이너가 죽인다.
#    (ComfyUI Issue #7465 — 런포드 서버리스에서 같은 증상이 보고됨)
#
# ② 측정이 오염된다
#    같은 조건을 두 번 던지면 두 번째는 계산을 건너뛰어 가짜로 빠른 기록이 나온다.
#    2026-08-09 에 LTX_03 이 19.1초로 찍혔다가 무효 처리된 것이 이것 때문이다.
#
# 값 형식 (템플릿에서 바꾸면 재빌드 없이 되돌릴 수 있다)
#   none        --cache-none          (기본)
#   ram         --cache-ram           ComfyUI 기본 임계값
#   ram:8,32    --cache-ram 8 32      남겨둘 여유를 GB 로 지정 (headroom 이다. 쓸 양이 아니다)
#   classic     --cache-classic
#   lru:10      --cache-lru 10
#   default     아무 옵션도 주지 않는다
COMFY_CACHE_MODE = os.environ.get("COMFY_CACHE_MODE", "none").strip().lower()
MODELS_DIR = os.environ.get("COMFYUI_MODELS_DIR", "/comfyui/models")

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_ACCESS_TOKEN")


def log(msg):
    print(msg, flush=True)


# ⚠️ log 는 반드시 여기(파일 위쪽)에 있어야 한다.
#    아래 LORA_CATALOG 를 만드는 줄이 모듈을 불러오는 순간 실행되는데,
#    카탈로그에 잘못된 줄이 있으면 그 안에서 log 를 부른다.
#    log 가 아래쪽에 있으면 그 시점에 아직 정의되지 않아 NameError 로 워커가 즉사한다.


def _env(name, default=""):
    """환경변수를 읽는다. 앞뒤 공백은 버린다.

    ⚠️ 런포드는 빈 문자열·공백만 있는 환경변수를 컨테이너에 안 넣는다.
       크레아2 에서 LORA_FILE=" " 로 끄려다 실제로 당했다.
       그래서 "끄기"는 빈 값이 아니라 none/off/- 같은 명시적 값으로 한다.
    """
    return os.environ.get(name, default).strip()


def _off(v):
    """이 값이 '끄기'를 뜻하는가."""
    return v.lower() in ("", "none", "off", "-", "0", "false")


# ── 부품 (전부 환경변수. 템플릿만 고치면 어떤 조합이든 된다) ────────────
#
# 부품마다 (저장소, 저장소 안 경로, ComfyUI 폴더) 세 쌍이다.
# ComfyUI 폴더는 어느 로더 노드가 그 파일을 보느냐로 정해진다 — 바꾸면 안 된다.

MODEL_REPO = _env("MODEL_REPO", "QuantStack/LTX-2.3-GGUF")
MODEL_FILE = _env("MODEL_FILE",
                  "LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q6_K.gguf")

# 젬마. DualCLIPLoaderGGUF 의 첫 칸.
TE_REPO = _env("TE_REPO", "unsloth/gemma-3-12b-it-qat-GGUF")
TE_FILE = _env("TE_FILE", "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf")

# 임베딩 커넥터(텍스트 프로젝션). DualCLIPLoaderGGUF 의 둘째 칸.
# ⚠️ 본체가 distilled 면 커넥터도 distilled 용을 받아야 한다.
CONN_REPO = _env("CONN_REPO", "unsloth/LTX-2.3-GGUF")
CONN_FILE = _env("CONN_FILE",
                 "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors")

VAE_REPO = _env("VAE_REPO", "unsloth/LTX-2.3-GGUF")
VAE_FILE = _env("VAE_FILE", "vae/ltx-2.3-22b-distilled_video_vae.safetensors")

# ⚠️ 오디오 VAE 는 models/vae 가 아니라 models/checkpoints 에 놓는다.
#    코어 LTXVAudioVAELoader 가 checkpoints 폴더를 보기 때문이다
#    (comfy_extras/nodes_lt_audio.py 19행). 파일 자체는 unsloth 것 그대로다.
AVAE_REPO = _env("AVAE_REPO", "unsloth/LTX-2.3-GGUF")
AVAE_FILE = _env("AVAE_FILE", "vae/ltx-2.3-22b-distilled_audio_vae.safetensors")

UPS_SPATIAL_REPO = _env("UPSCALER_SPATIAL_REPO", "Lightricks/LTX-2.3")
UPS_SPATIAL_FILE = _env("UPSCALER_SPATIAL_FILE",
                        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
UPS_TEMPORAL_REPO = _env("UPSCALER_TEMPORAL_REPO", "Lightricks/LTX-2.3")
UPS_TEMPORAL_FILE = _env("UPSCALER_TEMPORAL_FILE",
                         "ltx-2.3-temporal-upscaler-x2-1.0.safetensors")

# DualCLIPLoaderGGUF 의 type 칸. 크레아의 CLIPLoader type="krea2" 와 같은 자리다.
CLIP_TYPE = _env("CLIP_TYPE", "ltxv")

# ⭐ 부품을 더 받고 싶을 때 쓰는 칸 (v2, 2026-08-08 추가)
#
#   EXTRA_FILES = 저장소|저장소안경로|ComfyUI폴더 ; 저장소|저장소안경로|ComfyUI폴더
#
#   예) 젬마 멀티모달(mmproj)이 나중에 필요해지면
#       unsloth/gemma-3-12b-it-qat-GGUF|mmproj-BF16.gguf|text_encoders
#
# 왜 만들었나: 부품 목록이 코드에 박혀 있으면 하나 추가할 때마다 재빌드를 해야 한다.
# 보스 요구가 "고칠 일이 생기면 템플릿만 고쳐서 끝" 이므로, 그 범위를 부품 추가까지 넓힌다.
EXTRA_FILES = _env("EXTRA_FILES")

# ── 기본값 ────────────────────────────────────────────────────────────
#
# ⚠️ 해상도·프레임 기본값은 아직 실측 근거가 없다. "작게 시작한다"는 뜻이지 최적값이 아니다.
#    4단계에서 VRAM 한계선을 재고 확정한다.
DEF_WIDTH = int(_env("WIDTH", "704") or 704)
DEF_HEIGHT = int(_env("HEIGHT", "480") or 480)
DEF_FRAMES = int(_env("FRAMES", "97") or 97)
DEF_FPS = float(_env("FPS", "25") or 25)

# 8스텝 / cfg 1.0 은 모델 카드 근거다 —
#   "ltx-2.3-22b-distilled | The distilled version of the full model, 8 steps, CFG=1"
DEF_STEPS = int(_env("STEPS", "8") or 8)
DEF_CFG = float(_env("CFG", "1.0") or 1.0)
DEF_SAMPLER = _env("SAMPLER", "euler_ancestral")

# LTXVScheduler 값. unsloth 검증 워크플로에서 그대로 가져왔다.
DEF_MAX_SHIFT = float(_env("MAX_SHIFT", "2.05") or 2.05)
DEF_BASE_SHIFT = float(_env("BASE_SHIFT", "0.95") or 0.95)
DEF_TERMINAL = float(_env("TERMINAL", "0.1") or 0.1)

# 2단계(업스케일 후 정제)에서 쓸 시그마.
# ⚠️ unsloth 값(3스텝)을 그대로 가져왔다. 그들은 dev 본체 + distilled 로라였고
#    우리는 distilled 본체 단독이다. 최적값은 실측으로 정해야 한다.
DEF_STAGE2_SIGMAS = _env("STAGE2_SIGMAS", "0.909375, 0.725, 0.421875, 0.0")
DEF_STAGE2_CFG = float(_env("STAGE2_CFG", "1.0") or 1.0)

# VAEDecodeTiled 값. 메모리에 맞춰 조절한다(타일이 적을수록 빠르고 메모리를 더 쓴다).
DEF_TILE = int(_env("TILE_SIZE", "512") or 512)
DEF_OVERLAP = int(_env("TILE_OVERLAP", "64") or 64)
DEF_TEMPORAL_SIZE = int(_env("TEMPORAL_SIZE", "4096") or 4096)
DEF_TEMPORAL_OVERLAP = int(_env("TEMPORAL_OVERLAP", "8") or 8)

DEF_NEGATIVE = _env(
    "NEGATIVE_PROMPT",
    "blurry, low quality, still frame, frames, watermark, overlay, titles, "
    "has blurbox, has subtitles")

DEF_AUDIO = not _off(_env("AUDIO", "on"))
DEF_UPSCALE_SPATIAL = not _off(_env("UPSCALE_SPATIAL", "on"))

# ── 결과 반환 ─────────────────────────────────────────────────────────
#
# 런포드 응답 크기 제한 (docs.runpod.io operation-reference 41·172행)
#   /runsync  20 MB   /run  10 MB
# base64 는 원본의 약 1.33배가 된다.
RETURN_MODE = _env("RETURN_MODE", "base64").lower()
MAX_RETURN_MB = float(_env("MAX_RETURN_MB", "14") or 14)

# ──────────────────────────────────────────────────────────────────────
# ⭐ 로라 카탈로그 — 줄 하나가 버튼 하나다
#
# LORA_CATALOG 형식 (줄바꿈 또는 세미콜론으로 구분):
#   이름|저장소|파일경로|종류|기본강도|설명
#
#   종류  plain   그냥 붙인다. 스위치 + 강도 슬라이더로 끝     (9개)
#         iclora  참고 영상/사진이 필요하다                    (21개)
#
# 예)
#   camera-dolly-in|Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In|ltx-2-19b-lora-camera-control-dolly-in.safetensors|plain|1.0|카메라가 다가감
#
# ⭐ 여기 한 줄 추가하면 capabilities 응답에 항목이 하나 늘고,
#    UI 를 안 고쳐도 화면에 스위치가 저절로 하나 생긴다.
# ──────────────────────────────────────────────────────────────────────

def _parse_catalog(raw):
    """LORA_CATALOG 를 읽어 목록으로 만든다. 잘못된 줄은 건너뛰고 로그에 남긴다."""
    out = []
    if not raw:
        return out
    for line in raw.replace(";", "\n").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            log(f"[catalog] ⚠️ 칸이 모자라 건너뜀: {line!r}")
            continue
        name, repo, path = parts[0], parts[1], parts[2]
        kind = (parts[3].lower() if len(parts) > 3 and parts[3] else "plain")
        try:
            strength = float(parts[4]) if len(parts) > 4 and parts[4] else 1.0
        except ValueError:
            strength = 1.0
        label = parts[5] if len(parts) > 5 else name
        if kind not in ("plain", "iclora"):
            log(f"[catalog] ⚠️ 모르는 종류 '{kind}' → plain 으로 본다: {name}")
            kind = "plain"
        out.append({
            "name": name, "repo": repo, "path": path,
            "kind": kind, "default_strength": strength, "label": label,
        })
    return out


LORA_CATALOG = _parse_catalog(_env("LORA_CATALOG"))

# 워커가 뜰 때 미리 받아둘 로라. 쉼표로 나열한다.
#   비움  아무것도 미리 안 받는다 (요청에서 켜는 순간 받는다)
#   이름  그것만 미리 받는다
#   *     카탈로그 전부   ⬅ 로컬용
LORA_PRELOAD = _env("LORA_PRELOAD")

# 지금 워커에 실제로 받아져 있는 로라 이름들
_lora_ready = set()

# 콜드스타트 실측용
T_PROCESS_START = time.time()
_boot_stats = {}

COMFY_PROC = None


# ──────────────────────────────────────────────────────────────────────
# 1) 파일 받기
# ──────────────────────────────────────────────────────────────────────

def _download_one(repo_id, filename, comfy_folder):
    """받아서 <MODELS_DIR>/<comfy_folder>/<파일명> 에 놓는다.

    ⚠️ 저장소가 어떤 폴더 구조를 쓰든 여기서 평평하게 만든다.
       크레아2 에서 realrebelai 의 TURBO/ 하위폴더 때문에 ComfyUI 가 파일을 거부한 적이 있다.
       (Value not in list: 'x.gguf' not in ['TURBO/x.gguf'])

    ⚠️ 이미 있으면 다시 받지 않는다.
       서버리스에선 매번 컨테이너가 사라지므로 영향이 없고,
       로컬에선 한 번만 받고 계속 쓰게 된다. 같은 코드가 양쪽에서 다르게 동작한다.
    """
    dest_dir = os.path.join(MODELS_DIR, comfy_folder)
    os.makedirs(dest_dir, exist_ok=True)
    want = os.path.join(dest_dir, os.path.basename(filename))

    if os.path.exists(want) and os.path.getsize(want) > 0:
        size = os.path.getsize(want)
        log(f"[download] 이미 있음, 건너뜀: {comfy_folder}/{os.path.basename(filename)} "
            f"({size / 1024**3:.2f} GiB)")
        return size

    t0 = time.time()
    path = hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=dest_dir, token=HF_TOKEN,
    )
    if os.path.abspath(path) != os.path.abspath(want):
        os.replace(path, want)
        path = want

    size = os.path.getsize(path)
    sec = time.time() - t0
    speed = (size / 1024**2) / sec if sec > 0 else 0
    log(f"[download] {repo_id}/{filename} → {comfy_folder}/{os.path.basename(filename)}  "
        f"{size / 1024**3:.2f} GiB  {sec:.1f}초  ({speed:.0f} MB/s)")
    return size


def _base_jobs():
    """항상 받아야 하는 부품 목록.

    (저장소, 저장소안 경로, ComfyUI 폴더)

    ⚠️ 폴더는 어느 로더가 그 파일을 보느냐로 정해진다. 근거는 아래와 같다.
       unet                   city96 UnetLoaderGGUF
       text_encoders          city96 DualCLIPLoaderGGUF (젬마 + 커넥터)
       vae                    코어 VAELoader (nodes.py 763·769행)
       checkpoints            코어 LTXVAudioVAELoader (nodes_lt_audio.py 19행)
       latent_upscale_models  코어 LatentUpscaleModelLoader (nodes_hunyuan.py 187행)
    """
    jobs = [
        (MODEL_REPO, MODEL_FILE, "unet"),
        (TE_REPO, TE_FILE, "text_encoders"),
        (CONN_REPO, CONN_FILE, "text_encoders"),
        (VAE_REPO, VAE_FILE, "vae"),
        (AVAE_REPO, AVAE_FILE, "checkpoints"),
    ]
    if not _off(UPS_SPATIAL_FILE):
        jobs.append((UPS_SPATIAL_REPO, UPS_SPATIAL_FILE, "latent_upscale_models"))
    if not _off(UPS_TEMPORAL_FILE):
        jobs.append((UPS_TEMPORAL_REPO, UPS_TEMPORAL_FILE, "latent_upscale_models"))
    jobs.extend(_parse_extra(EXTRA_FILES))
    return jobs


def _parse_extra(raw):
    """EXTRA_FILES 를 읽어 다운로드 목록으로 만든다.

    형식은 카탈로그와 같게 맞췄다 — 파이프(|)로 칸을 나누고 세미콜론/줄바꿈으로 항목을 나눈다.
        저장소|저장소안경로|ComfyUI폴더

    ⚠️ 잘못된 줄은 건너뛰고 로그에 남긴다. 그것 때문에 워커가 죽지는 않는다.
    """
    out = []
    if _off(raw):
        return out
    for line in raw.replace(";", "\n").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3 or not all(parts[:3]):
            log(f"[extra] ⚠️ 칸이 모자라 건너뜀 (저장소|경로|폴더 형식이어야 한다): {line!r}")
            continue
        out.append((parts[0], parts[1], parts[2]))
        log(f"[extra] 추가 부품: {parts[0]}/{parts[1]} → {parts[2]}/")
    return out


def _preload_names():
    """미리 받을 로라 이름 목록."""
    raw = LORA_PRELOAD
    if _off(raw):
        return []
    if raw.strip() == "*":
        return [c["name"] for c in LORA_CATALOG]
    return [x.strip() for x in raw.split(",") if x.strip()]


def ensure_lora(name):
    """로라가 없으면 그때 받는다. 이미 있으면 아무것도 안 한다.

    반환값은 (파일명, 오류메시지). 오류가 있으면 파일명은 None 이다.
    """
    spec = find_lora(name)
    if spec is None:
        have = ", ".join(c["name"] for c in LORA_CATALOG) or "(카탈로그가 비어 있다)"
        return None, f"로라를 못 찾았다: '{name}'. 카탈로그에 있는 것: {have}"

    fname = os.path.basename(spec["path"])
    if spec["name"] in _lora_ready:
        return fname, None

    try:
        _download_one(spec["repo"], spec["path"], "loras")
        _lora_ready.add(spec["name"])
        return fname, None
    except Exception as e:
        return None, (f"로라 '{name}' 를 못 받았다: {type(e).__name__}: {e}. "
                      f"게이팅 저장소면 HF_TOKEN 에 약관 동의가 되어 있어야 한다")


def find_lora(name):
    """카탈로그에서 이름으로 찾는다. 부분 일치도 허용한다."""
    key = (name or "").strip().lower()
    if not key:
        return None
    for c in LORA_CATALOG:
        if c["name"].lower() == key:
            return c
    for c in LORA_CATALOG:
        if key in c["name"].lower():
            return c
    return None


def download_base_models():
    jobs = _base_jobs()
    log(f"[download] 기본 부품 {len(jobs)}개 받기 시작 "
        f"(model={os.path.basename(MODEL_FILE)}, te={os.path.basename(TE_FILE)})")

    t0 = time.time()
    total = 0
    # ⚠️ 순차로 받으면 한 연결 속도에 묶인다. 크레아2 실측으로 동시 4개가 초당 800MB 였다.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_download_one, r, f, d) for (r, f, d) in jobs]
        for fut in futures:
            total += fut.result()

    # 미리 받을 로라
    names = _preload_names()
    if names:
        log(f"[download] 로라 미리 받기 {len(names)}개: {', '.join(names)}")
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {n: pool.submit(ensure_lora, n) for n in names}
            for n, fut in futs.items():
                fname, err = fut.result()
                if err:
                    # ⚠️ 미리 받기 실패로 워커를 죽이지 않는다. 그 로라만 못 쓰게 둔다.
                    log(f"[download] ⚠️ 미리 받기 실패({n}): {err}")

    sec = time.time() - t0
    log(f"[download] 완료: {total / 1024**3:.2f} GiB / {sec:.1f}초 "
        f"({(total / 1024**2) / max(sec, 0.001):.0f} MB/s 평균)")
    _boot_stats["download_sec"] = round(sec, 1)
    _boot_stats["download_gib"] = round(total / 1024**3, 2)


# ──────────────────────────────────────────────────────────────────────
# 2) ComfyUI 띄우기
# ──────────────────────────────────────────────────────────────────────

def cache_args():
    """COMFY_CACHE_MODE 를 ComfyUI 명령줄 옵션으로 바꾼다.

    ⚠️ 모르는 값이 오면 조용히 넘기지 않고 경고를 찍은 뒤 기본(none)으로 간다.
       오타 하나 때문에 RAM 이 터지는 것을 로그에서 바로 알아채기 위해서다.
    """
    m = COMFY_CACHE_MODE
    if m in ("", "default"):
        return []
    if m == "none":
        return ["--cache-none"]
    if m == "classic":
        return ["--cache-classic"]
    if m == "ram":
        return ["--cache-ram"]
    if m.startswith("ram:"):
        # ram:8  또는  ram:8,32   (앞이 active, 뒤가 inactive/pin 임계값. 단위 GB)
        vals = [v.strip() for v in m[4:].split(",") if v.strip()]
        return ["--cache-ram"] + vals
    if m.startswith("lru:"):
        return ["--cache-lru", m[4:].strip()]
    log(f"[comfyui] ⚠️ COMFY_CACHE_MODE='{m}' 를 모르겠다. 기본값 none 으로 간다.")
    return ["--cache-none"]


def start_comfyui():
    cmd = [
        sys.executable, "-u", "/comfyui/main.py",
        "--listen", COMFY_HOST,
        "--port", str(COMFY_PORT),
        "--disable-auto-launch",
        "--disable-metadata",
        "--disable-api-nodes",
        "--log-stdout",
    ] + cache_args()
    log(f"[comfyui] 실행: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd="/comfyui")


def wait_for_comfyui(proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        # ⚠️ 죽었으면 기다려봐야 소용없다. 바로 알린다.
        if proc.poll() is not None:
            raise RuntimeError(
                f"ComfyUI 가 시작하다 죽었다 (exit code {proc.returncode}). 위 로그를 볼 것.")
        try:
            r = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
            if r.status_code == 200:
                sec = time.time() - t0
                log(f"[comfyui] 준비 완료 ({sec:.1f}초)")
                _boot_stats["comfyui_boot_sec"] = round(sec, 1)
                try:
                    log(f"[comfyui] system_stats: "
                        f"{json.dumps(r.json(), ensure_ascii=False)}")
                except Exception:
                    pass
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"ComfyUI 가 {timeout}초 안에 안 떴다.")


# ──────────────────────────────────────────────────────────────────────
# VRAM 감시 — 24GB 한계선을 재기 위한 것
# ──────────────────────────────────────────────────────────────────────

class VramWatcher(threading.Thread):
    """생성하는 동안 2초마다 남은 VRAM 을 봐서 가장 적었던 값을 기록한다.

    정확한 최대치가 아니라 밖에서 뜬 표본의 근사값이다.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.min_free = None
        self.total = None
        self.samples = 0

    def run(self):
        while not self.stop_flag.is_set():
            try:
                d = requests.get(f"{COMFY_URL}/system_stats", timeout=5).json()
                dev = (d.get("devices") or [{}])[0]
                free = dev.get("vram_free")
                if free is not None:
                    self.samples += 1
                    self.total = dev.get("vram_total")
                    if self.min_free is None or free < self.min_free:
                        self.min_free = free
            except Exception:
                pass
            self.stop_flag.wait(2)

    def result(self):
        if self.min_free is None or not self.total:
            return {}
        used = self.total - self.min_free
        return {
            "vram_total_gb": round(self.total / 1024**3, 2),
            "vram_peak_used_gb": round(used / 1024**3, 2),
            "vram_min_free_gb": round(self.min_free / 1024**3, 2),
            "vram_samples": self.samples,
        }


# ──────────────────────────────────────────────────────────────────────
# 3) 워크플로 조립
#
# 배선은 unsloth 검증 워크플로(노드 39개)를 그대로 따르되, 부품을 우리 것으로 바꿨다.
#   unsloth   VAELoaderKJ ×2                → 코어 VAELoader + LTXVAudioVAELoader
#   unsloth   dev 본체 + distilled 로라      → distilled 본체 단독 (로라 불필요)
#
# 흐름
#   1단계  목표 해상도의 절반으로 생성
#   업스케일  LTXVLatentUpsampler 로 2배
#   2단계  적은 스텝으로 정제
#   디코드  영상 + 소리 → mp4
# ──────────────────────────────────────────────────────────────────────

def build_workflow(p):
    """요청값(p)으로 ComfyUI API 형식 워크플로를 만든다."""
    wf = {}

    # ── 로더 ──────────────────────────────────────────────────────
    wf["1"] = {"class_type": "LTXVAudioVAELoader",
               "inputs": {"ckpt_name": os.path.basename(AVAE_FILE)}}
    wf["2"] = {"class_type": "VAELoader",
               "inputs": {"vae_name": os.path.basename(VAE_FILE)}}
    wf["3"] = {"class_type": "UnetLoaderGGUF",
               "inputs": {"unet_name": os.path.basename(MODEL_FILE)}}
    wf["4"] = {"class_type": "DualCLIPLoaderGGUF",
               "inputs": {"clip_name1": os.path.basename(TE_FILE),
                          "clip_name2": os.path.basename(CONN_FILE),
                          "type": CLIP_TYPE}}

    # ── 로라 사슬 ─────────────────────────────────────────────────
    # 본체("3")에서 시작해 하나씩 이어 붙인다. 마지막 노드가 최종 모델이 된다.
    # 노드 번호는 60번대를 쓴다(기존 번호와 안 부딪히게).
    #
    # ⚠️ unsloth 워크플로와 일부러 다르게 한 곳이다.
    #    그들은 1단계에 본체를 직접 걸고 2단계에만 로라를 걸었다.
    #    그 로라가 "distilled 로라"(정제 전용)였기 때문이다.
    #    우리 로라는 카메라 움직임·효과음 같은 것이라 1단계부터 걸려야 효과가 난다.
    #    → 1·2단계 모두 로라를 태운다.
    #    ⚠️ 이건 추론이다. 실측에서 결과가 이상하면 여기를 먼저 의심할 것.
    model_ref = ["3", 0]
    for i, spec in enumerate(p["loras"]):
        nid = str(60 + i)
        wf[nid] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": model_ref,
                              "lora_name": spec["file"],
                              "strength_model": spec["strength"]}}
        model_ref = [nid, 0]

    # ── 프롬프트 ──────────────────────────────────────────────────
    wf["5"] = {"class_type": "CLIPTextEncode",
               "inputs": {"text": p["prompt"], "clip": ["4", 0]}}
    wf["6"] = {"class_type": "CLIPTextEncode",
               "inputs": {"text": p["negative"], "clip": ["4", 0]}}
    wf["13"] = {"class_type": "LTXVConditioning",
                "inputs": {"frame_rate": p["fps"],
                           "positive": ["5", 0], "negative": ["6", 0]}}

    # ── 잠재 ──────────────────────────────────────────────────────
    # ⚠️ 업스케일을 켜면 1단계는 목표의 절반 해상도로 돈다.
    #    unsloth 워크플로가 ImageScaleBy 0.5 로 하는 일을 여기서 숫자로 직접 한다.
    if p["upscale_spatial"]:
        w1 = max(64, (p["width"] // 2 // 32) * 32)
        h1 = max(64, (p["height"] // 2 // 32) * 32)
    else:
        w1, h1 = p["width"], p["height"]

    wf["10"] = {"class_type": "LTXVEmptyLatentAudio",
                "inputs": {"frames_number": p["frames"],
                           "frame_rate": p["fps"],
                           "batch_size": 1,
                           "audio_vae": ["1", 0]}}
    wf["11"] = {"class_type": "EmptyLTXVLatentVideo",
                "inputs": {"width": w1, "height": h1,
                           "length": p["frames"], "batch_size": 1}}
    # ── 사진 → 영상 (i2v) ─────────────────────────────────────────
    # 빈 잠재("11")에 사진을 얹어 첫 프레임을 고정한다.
    #
    # 노드 규격은 실물로 확인했다 (지어낸 이름이 아니다):
    #   LoadImage                     코어 nodes.py 1721행
    #     입력 image=<파일 이름>  /  출력 0=IMAGE, 1=MASK
    #   LTXVImgToVideoConditionOnly   ComfyUI-LTXVideo latents.py 499행
    #     입력 vae / image / latent / strength(0~1)  /  출력 LATENT
    #     ⚠️ 이 노드는 v2 부터 쓸 수 있다. v1 에서는 kornia 오류로
    #        ComfyUI-LTXVideo 묶음이 통째로 import 실패였다.
    #
    # 사진 크기는 노드가 알아서 잠재 크기에 맞춘다("Automatically resizes image").
    # 그래서 업스케일을 켜서 1단계가 절반 해상도가 되어도 따로 손댈 것이 없다.
    video_latent_ref = ["11", 0]
    if p["mode"] == "i2v":
        wf["70"] = {"class_type": "LoadImage",
                    "inputs": {"image": p["image"]}}
        wf["71"] = {"class_type": "LTXVImgToVideoConditionOnly",
                    "inputs": {"vae": ["2", 0],
                               "image": ["70", 0],
                               "latent": ["11", 0],
                               "strength": p["image_strength"]}}
        video_latent_ref = ["71", 0]

    wf["12"] = {"class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": video_latent_ref,
                           "audio_latent": ["10", 0]}}

    # ── 1단계 샘플링 ──────────────────────────────────────────────
    wf["20"] = {"class_type": "RandomNoise",
                "inputs": {"noise_seed": p["seed"]}}
    wf["21"] = {"class_type": "KSamplerSelect",
                "inputs": {"sampler_name": p["sampler"]}}
    wf["22"] = {"class_type": "LTXVScheduler",
                "inputs": {"steps": p["steps"],
                           "max_shift": p["max_shift"],
                           "base_shift": p["base_shift"],
                           "stretch": True,
                           "terminal": p["terminal"],
                           "latent": ["12", 0]}}
    wf["23"] = {"class_type": "CFGGuider",
                "inputs": {"cfg": p["cfg"], "model": model_ref,
                           "positive": ["13", 0], "negative": ["13", 1]}}
    wf["24"] = {"class_type": "SamplerCustomAdvanced",
                "inputs": {"noise": ["20", 0], "guider": ["23", 0],
                           "sampler": ["21", 0], "sigmas": ["22", 0],
                           "latent_image": ["12", 0]}}
    wf["25"] = {"class_type": "LTXVSeparateAVLatent",
                "inputs": {"av_latent": ["24", 0]}}
    wf["26"] = {"class_type": "LTXVCropGuides",
                "inputs": {"positive": ["13", 0], "negative": ["13", 1],
                           "latent": ["25", 0]}}

    if p["upscale_spatial"]:
        # ── 업스케일 + 2단계 정제 ─────────────────────────────────
        wf["30"] = {"class_type": "LatentUpscaleModelLoader",
                    "inputs": {"model_name": os.path.basename(UPS_SPATIAL_FILE)}}
        wf["31"] = {"class_type": "LTXVLatentUpsampler",
                    "inputs": {"samples": ["26", 2],
                               "upscale_model": ["30", 0], "vae": ["2", 0]}}
        wf["32"] = {"class_type": "LTXVConcatAVLatent",
                    "inputs": {"video_latent": ["31", 0],
                               "audio_latent": ["25", 1]}}
        wf["40"] = {"class_type": "RandomNoise",
                    "inputs": {"noise_seed": p["seed"]}}
        wf["41"] = {"class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": p["sampler"]}}
        wf["42"] = {"class_type": "ManualSigmas",
                    "inputs": {"sigmas": p["stage2_sigmas"]}}
        wf["43"] = {"class_type": "CFGGuider",
                    "inputs": {"cfg": p["stage2_cfg"], "model": model_ref,
                               "positive": ["26", 0], "negative": ["26", 1]}}
        wf["44"] = {"class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["40", 0], "guider": ["43", 0],
                               "sampler": ["41", 0], "sigmas": ["42", 0],
                               "latent_image": ["32", 0]}}
        wf["45"] = {"class_type": "LTXVSeparateAVLatent",
                    "inputs": {"av_latent": ["44", 1]}}
        video_latent, audio_latent = ["45", 0], ["45", 1]
    else:
        # 업스케일을 끄면 1단계 결과를 그대로 디코드한다.
        # ⚠️ 이때 LTXVCropGuides("26")를 건너뛴다.
        #    그 노드는 키프레임(사진→영상 등에서 넣는 가이드 프레임)을 잘라내는 일을 하는데,
        #    키프레임이 0개면 그냥 통과시킨다 (nodes_lt.py 522행 num_keyframes == 0 분기).
        #    우리는 지금 글자→영상만 하므로 건너뛰어도 결과가 같다.
        #    ⚠️ 나중에 사진→영상을 붙이면 이 경로에도 CropGuides 가 필요해진다.
        video_latent, audio_latent = ["25", 0], ["25", 1]

    # ── 디코드 · 저장 ─────────────────────────────────────────────
    wf["50"] = {"class_type": "VAEDecodeTiled",
                "inputs": {"tile_size": p["tile_size"],
                           "overlap": p["tile_overlap"],
                           "temporal_size": p["temporal_size"],
                           "temporal_overlap": p["temporal_overlap"],
                           "samples": video_latent, "vae": ["2", 0]}}

    create_inputs = {"fps": p["fps"], "images": ["50", 0]}
    if p["audio"]:
        wf["51"] = {"class_type": "LTXVAudioVAEDecode",
                    "inputs": {"samples": audio_latent, "audio_vae": ["1", 0]}}
        create_inputs["audio"] = ["51", 0]

    wf["52"] = {"class_type": "CreateVideo", "inputs": create_inputs}
    wf["53"] = {"class_type": "SaveVideo",
                "inputs": {"filename_prefix": "video/ltx",
                           "format": "mp4", "codec": "auto",
                           "video": ["52", 0]}}
    return wf


# ──────────────────────────────────────────────────────────────────────
# 4) ComfyUI 에 던지고 결과 받기
# ──────────────────────────────────────────────────────────────────────

def queue_workflow(wf):
    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": wf}, timeout=60)
    if r.status_code != 200:
        # ⚠️ ComfyUI 는 거부 이유를 본문에 적어 보낸다. 그대로 올려야 원인을 안다.
        raise RuntimeError(
            f"ComfyUI 가 워크플로를 거부했다 (HTTP {r.status_code}): {r.text[:2000]}")
    return r.json()["prompt_id"]


def wait_for_result(prompt_id, proc, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"생성 도중 ComfyUI 가 죽었다 (exit code {proc.returncode}). "
                f"메모리 부족일 가능성이 높다.")
        try:
            h = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10).json()
        except requests.RequestException:
            time.sleep(1)
            continue

        entry = h.get(prompt_id)
        if entry:
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                raise RuntimeError(
                    f"ComfyUI 실행 오류: "
                    f"{json.dumps(status, ensure_ascii=False)[:2000]}")
            if status.get("completed") or entry.get("outputs"):
                return entry
        time.sleep(1)
    raise RuntimeError(f"{timeout}초 안에 생성이 안 끝났다.")


def fetch_video(entry):
    """결과에서 영상 파일을 꺼낸다.

    ⚠️ SaveVideo 의 출력 키 이름을 우리가 실제로 본 적이 없다.
       그래서 images/gifs/videos 를 모두 뒤지고, 그래도 없으면
       어떤 키가 왔는지 오류에 담아 알려준다. 조용히 실패하지 않는다.
    """
    outputs = entry.get("outputs") or {}
    for node_out in outputs.values():
        for key in ("videos", "gifs", "images", "files"):
            for item in node_out.get(key) or []:
                if not isinstance(item, dict) or "filename" not in item:
                    continue
                q = urllib.parse.urlencode({
                    "filename": item["filename"],
                    "subfolder": item.get("subfolder", ""),
                    "type": item.get("type", "output"),
                })
                r = requests.get(f"{COMFY_URL}/view?{q}", timeout=600)
                r.raise_for_status()
                return r.content, item["filename"]
    raise RuntimeError(
        "결과에서 영상 파일을 못 찾았다. 돌아온 출력 키: "
        f"{json.dumps({k: list(v.keys()) for k, v in outputs.items()}, ensure_ascii=False)[:1000]}")


# ──────────────────────────────────────────────────────────────────────
# 5) 요청 해석
# ──────────────────────────────────────────────────────────────────────

def save_media(job):
    """요청에 실려온 사진·영상·소리를 ComfyUI 가 읽는 폴더에 파일로 놓는다.

    ⭐ 이 함수 하나가 image- / video- / audio- 로 시작하는 기능 전부의 전제조건이다.
       LTX-2.3 은 11가지 입출력 조합을 지원하는데, 그중 9가지가 "재료를 넣는" 것이라
       재료를 넣을 통로가 없으면 배선을 아무리 잘 그려도 쓸 수 없다.

    요청 형식:
      "media": [{"name": "first.png", "data": "(base64)"}, ...]

    워커와 ComfyUI 가 같은 컨테이너 안에 있으므로 파일로 직접 놓으면 된다.
    배선에서는 LoadImage 등에 파일 이름만 적으면 그대로 읽힌다.

    반환: (놓인 파일 이름 목록, 오류메시지)
    """
    items = job.get("media")
    if not items:
        return [], None
    if not isinstance(items, list):
        return None, "media 는 목록이어야 한다."

    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
    saved = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            return None, f"media[{i}] 가 잘못됐다: {item!r}"
        raw_name = str(item.get("name") or "").strip()
        data = item.get("data")
        if not raw_name or not data:
            return None, f"media[{i}] 에 name 또는 data 가 없다."

        # ⚠️ 경로 탈출 방지. "../../etc/passwd" 같은 이름이 와도 파일명만 남긴다.
        name = os.path.basename(raw_name)
        if not name or name in (".", ".."):
            return None, f"media[{i}] 의 name 이 파일 이름이 아니다: {raw_name!r}"

        try:
            blob = base64.b64decode(data, validate=True)
        except Exception as e:
            return None, f"media[{i}]({name}) 의 data 가 base64 가 아니다: {e}"

        path = os.path.join(COMFY_INPUT_DIR, name)
        with open(path, "wb") as f:
            f.write(blob)
        log(f"[media] 저장 {name} ({len(blob)/1048576:.2f} MB) → {path}")
        saved.append(name)

    return saved, None


def resolve_loras(job):
    """요청의 loras 목록을 카탈로그와 맞추고, 없으면 그때 받는다.

    반환값은 (목록, 오류메시지).

    stock(순정) 이면 아무것도 안 건다.
    loras 를 아예 안 보내면 역시 아무것도 안 건다 — 켠 것만 걸리는 게 기본이다.

    ⚠️ 못 찾거나 못 받으면 조용히 넘어가지 않고 오류로 돌려준다.
       로라는 안 붙어도 ComfyUI 가 오류를 안 내기 때문에(로라 문서 2절 ①),
       여기서 막지 않으면 "붙은 줄 알았는데 안 붙은" 결과가 나온다.
    """
    if job.get("stock"):
        return [], None

    req = job.get("loras")
    if not req:
        return [], None
    if not isinstance(req, list):
        return None, "loras 는 목록이어야 한다."

    out = []
    for item in req:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            return None, f"loras 항목이 잘못됐다: {item!r}"

        name = str(item.get("name") or item.get("file") or "").strip()
        spec = find_lora(name)
        if spec is None:
            have = ", ".join(c["name"] for c in LORA_CATALOG) or "(카탈로그가 비어 있다)"
            return None, f"로라를 못 찾았다: '{name}'. 카탈로그에 있는 것: {have}"

        # ⚠️ IC-LoRA 는 참고 영상/사진을 넣는 별도 노드가 필요하다.
        #    GGUF 본체에 붙인 검증 사례를 아직 못 봤으므로, 되는 척하지 않고 막는다.
        if spec["kind"] == "iclora":
            return None, (
                f"'{spec['name']}' 는 IC-LoRA 라 참고 영상/사진이 필요하다. "
                f"아직 지원 안 한다 (설계문서 8절 미확인 2번). "
                f"지금은 종류가 plain 인 로라만 쓸 수 있다")

        fname, err = ensure_lora(spec["name"])
        if err:
            return None, err

        strength = item.get("strength")
        try:
            strength = float(strength) if strength is not None else spec["default_strength"]
        except (TypeError, ValueError):
            return None, f"'{name}' 의 강도가 숫자가 아니다: {strength!r}"

        out.append({"name": spec["name"], "file": fname, "strength": strength})
    return out, None


def parse_params(job):
    """요청을 기본값과 합쳐 실제로 쓸 값으로 만든다."""
    stock = bool(job.get("stock"))

    seed = job.get("seed", None)
    if seed in (None, -1, "random"):
        seed = random.randint(0, 2**31 - 1)

    ups = job.get("upscale") or {}
    if stock:
        upscale_spatial = False
    elif isinstance(ups, dict) and "spatial" in ups:
        sp = ups["spatial"]
        upscale_spatial = bool(sp.get("on", True)) if isinstance(sp, dict) else bool(sp)
    else:
        upscale_spatial = DEF_UPSCALE_SPATIAL

    # ⭐ 어떤 배선을 그릴지. 안 적으면 image 가 있는지로 알아서 고른다.
    #    t2v  글자 → 영상        (기본)
    #    i2v  사진 → 영상        첫 프레임을 사진으로 고정한다
    #    그 밖의 조합은 workflow 를 통째로 던지는 길로 쓴다 (generate 참고)
    image = str(job.get("image") or "").strip()
    mode = str(job.get("mode") or "").strip().lower()
    if not mode:
        mode = "i2v" if image else "t2v"

    return {
        "mode": mode,
        "image": os.path.basename(image) if image else "",
        # 1.0 이면 첫 프레임을 사진 그대로 고정, 낮추면 사진에서 더 자유로워진다
        "image_strength": float(job.get("image_strength", 1.0)),
        "prompt": str(job.get("prompt") or "").strip(),
        "negative": str(job.get("negative_prompt", DEF_NEGATIVE)),
        "width": int(job.get("width", DEF_WIDTH)),
        "height": int(job.get("height", DEF_HEIGHT)),
        "frames": int(job.get("frames", DEF_FRAMES)),
        "fps": float(job.get("fps", DEF_FPS)),
        "steps": int(job.get("steps", DEF_STEPS)),
        "cfg": float(job.get("cfg", DEF_CFG)),
        "sampler": str(job.get("sampler", DEF_SAMPLER)),
        "seed": int(seed),
        "audio": bool(job.get("audio", DEF_AUDIO)),
        "upscale_spatial": upscale_spatial,
        "max_shift": float(job.get("max_shift", DEF_MAX_SHIFT)),
        "base_shift": float(job.get("base_shift", DEF_BASE_SHIFT)),
        "terminal": float(job.get("terminal", DEF_TERMINAL)),
        "stage2_sigmas": str(job.get("stage2_sigmas", DEF_STAGE2_SIGMAS)),
        "stage2_cfg": float(job.get("stage2_cfg", DEF_STAGE2_CFG)),
        "tile_size": int(job.get("tile_size", DEF_TILE)),
        "tile_overlap": int(job.get("tile_overlap", DEF_OVERLAP)),
        "temporal_size": int(job.get("temporal_size", DEF_TEMPORAL_SIZE)),
        "temporal_overlap": int(job.get("temporal_overlap", DEF_TEMPORAL_OVERLAP)),
        "stock": stock,
        "loras": [],
    }


# ──────────────────────────────────────────────────────────────────────
# 6) ⭐ 버튼 목록 — UI 가 이걸 받아 화면을 스스로 그린다
# ──────────────────────────────────────────────────────────────────────

def capabilities():
    return {
        "model": {
            "name": os.path.basename(MODEL_FILE),
            "repo": MODEL_REPO,
            "backend": "gguf",
            "text_encoder": os.path.basename(TE_FILE),
            "clip_type": CLIP_TYPE,
        },
        "features": [
            {"id": "audio", "label": "소리 생성", "type": "switch",
             "default": DEF_AUDIO},
            {"id": "steps", "label": "스텝", "type": "int",
             "default": DEF_STEPS, "min": 1, "max": 60},
            {"id": "cfg", "label": "프롬프트 따름 강도", "type": "float",
             "default": DEF_CFG, "min": 1.0, "max": 10.0},
            {"id": "width", "label": "가로", "type": "int", "default": DEF_WIDTH},
            {"id": "height", "label": "세로", "type": "int", "default": DEF_HEIGHT},
            {"id": "frames", "label": "프레임 수", "type": "int", "default": DEF_FRAMES},
            {"id": "fps", "label": "초당 프레임", "type": "float", "default": DEF_FPS},
            {"id": "seed", "label": "시드", "type": "int", "default": -1,
             "note": "-1 이면 매번 무작위"},
            {"id": "sampler", "label": "샘플러", "type": "text", "default": DEF_SAMPLER},
            {"id": "stock", "label": "순정으로 뽑기", "type": "switch", "default": False,
             "note": "로라·따로 붙인 업스케일을 전부 끈다. 로라가 실제로 붙었는지 대조할 때 쓴다"},
        ],
        "upscale": [
            {"id": "spatial", "label": "해상도 올리기 (2단계 정제)", "type": "switch",
             "default": DEF_UPSCALE_SPATIAL,
             "note": "켜면 절반 해상도로 뽑아 2배로 올린다. 끄면 절반 해상도로 한 번에 끝낸다"},
        ],
        "loras": [
            {
                "name": c["name"],
                "label": c["label"],
                "kind": c["kind"],
                "default_strength": c["default_strength"],
                "needs_reference": c["kind"] == "iclora",
                "supported": c["kind"] == "plain",
                "ready": c["name"] in _lora_ready,
            }
            for c in LORA_CATALOG
        ],
        # ⭐ 어떤 방식으로 시킬 수 있는지. UI 는 이 목록으로 탭이나 버튼을 만들면 된다.
        "modes": [
            {"id": "t2v", "label": "글자 → 영상", "needs": [],
             "note": "기본값. image 를 안 주면 이걸로 간다"},
            {"id": "i2v", "label": "사진 → 영상", "needs": ["image"],
             "note": "첫 프레임을 사진으로 고정한다. media 로 사진을 함께 보내야 한다"},
        ],
        # ⭐ 재료 넣는 통로. 이것이 있어야 사진·영상·소리를 쓰는 기능들이 열린다.
        "media": {
            "how": "요청에 media 를 실으면 워커가 컨테이너 안 입력 폴더에 파일로 놓는다",
            "format": [{"name": "파일 이름 (예: first.png)", "data": "base64 문자열"}],
            "then": "놓은 뒤 image 에 그 파일 이름을 적거나, workflow 의 노드에서 그 이름으로 부른다",
            "dir": COMFY_INPUT_DIR,
        },
        # ⭐ 위 두 가지로 안 되는 조합은 배선을 통째로 던지면 된다.
        "workflow_passthrough": {
            "how": "ComfyUI API 형식 JSON 을 workflow 에 통째로 넣으면 그대로 실행한다",
            "note": "media 와 함께 쓰면 모델이 지원하는 입출력 조합을 전부 쓸 수 있다. "
                    "코드를 고치거나 도커를 다시 만들 필요가 없다",
            "model_supports": [
                "text-to-video", "image-to-video", "video-to-video",
                "image-text-to-video", "audio-to-video", "text-to-audio",
                "video-to-audio", "audio-to-audio", "text-to-audio-video",
                "image-to-audio-video", "image-text-to-audio-video",
            ],
        },
        "return": {"mode": RETURN_MODE, "max_mb": MAX_RETURN_MB},
        "cache": {"mode": COMFY_CACHE_MODE, "args": cache_args(),
                  "note": "환경변수 COMFY_CACHE_MODE 로 바꾼다. 기본 none — "
                          "컨테이너 RAM 이 터지는 것과 같은 조건 재실행이 "
                          "가짜로 빨라지는 것을 둘 다 막는다"},
        "notes": [
            "IC-LoRA(참고 영상·사진이 필요한 것)는 아직 배선이 없다. "
            "노드는 v2 부터 쓸 수 있다 (LTXICLoRALoaderModelOnly / LTXAddVideoICLoRAGuide)",
            "해상도·프레임 기본값은 실측 전 잠정값이다",
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# 7) 기동 · 생성
# ──────────────────────────────────────────────────────────────────────

def boot():
    """워커가 뜰 때 한 번 돈다."""
    global COMFY_PROC
    if not HF_TOKEN:
        log("[init] ⚠️ HF_TOKEN 이 없다. 비게이트 저장소만 받을 수 있다.")
    if not LORA_CATALOG:
        log("[init] ⚠️ LORA_CATALOG 가 비어 있다. 로라 없이 기본 생성만 된다.")

    download_base_models()
    COMFY_PROC = start_comfyui()
    wait_for_comfyui(COMFY_PROC)

    _boot_stats["total_cold_start_sec"] = round(time.time() - T_PROCESS_START, 1)
    log(f"[init] 콜드스타트 총 {_boot_stats['total_cold_start_sec']}초 "
        f"(다운로드 {_boot_stats.get('download_sec')}초 + "
        f"ComfyUI 기동 {_boot_stats.get('comfyui_boot_sec')}초)")


def generate(job):
    """요청 하나를 처리한다. 이 함수가 몸통의 입구다."""
    # 버튼 목록 요청
    if str(job.get("action", "")).lower() == "capabilities":
        return capabilities()

    # ⭐ 재료(사진·영상·소리)를 먼저 컨테이너 안에 놓는다.
    #    workflow 를 통째로 던질 때도 필요하므로 분기 앞에서 처리한다.
    saved, err = save_media(job)
    if err:
        return {"error": err}

    # 워크플로 통째로 던지기 (공식 ComfyUI 화면에서 만든 것을 그대로 실행)
    raw_wf = job.get("workflow")

    if raw_wf is None:
        p = parse_params(job)
        if not p["prompt"]:
            return {"error": "prompt 가 비어 있다."}

        # ⚠️ 사진이 없는데 i2v 로 돌면 ComfyUI 가 엉뚱한 오류를 내거나
        #    조용히 다른 그림을 그린다. 여기서 막아 원인을 분명히 한다.
        if p["mode"] == "i2v" and not p["image"]:
            return {"error": "mode 가 i2v 인데 image 가 없다. "
                             "media 로 사진을 보내고 image 에 그 파일 이름을 적어라."}
        if p["image"]:
            here = os.path.join(COMFY_INPUT_DIR, p["image"])
            if p["image"] not in saved and not os.path.exists(here):
                return {"error": f"image '{p['image']}' 를 찾을 수 없다. "
                                 f"media 에 같이 실어 보냈는지 확인해라. "
                                 f"이번 요청에 실려온 것: {saved or '없음'}"}

        loras, err = resolve_loras(job)
        if err:
            return {"error": err}
        p["loras"] = loras
        wf = build_workflow(p)
    else:
        if not isinstance(raw_wf, dict):
            return {"error": "workflow 는 ComfyUI API 형식 JSON 이어야 한다."}
        p = None
        wf = raw_wf

    watcher = VramWatcher()
    watcher.start()
    t0 = time.time()
    try:
        prompt_id = queue_workflow(wf)
        if p is not None:
            desc = ", ".join(f"{s['name']}@{s['strength']}" for s in p["loras"]) or "없음"
            src = f" 사진={p['image']}@{p['image_strength']}" if p["mode"] == "i2v" else ""
            log(f"[job] 큐 등록 {prompt_id} — [{p['mode']}]{src} "
                f"{p['width']}x{p['height']} "
                f"{p['frames']}프레임 {p['fps']}fps {p['steps']}스텝 cfg{p['cfg']} "
                f"seed{p['seed']} 업스케일={p['upscale_spatial']} 소리={p['audio']} "
                f"로라=[{desc}]{' 순정' if p['stock'] else ''}")
        else:
            log(f"[job] 큐 등록 {prompt_id} — 통짜 워크플로 ({len(wf)}개 노드)")

        entry = wait_for_result(prompt_id, COMFY_PROC)
        payload, filename = fetch_video(entry)
    except Exception as e:
        watcher.stop_flag.set()
        log(f"[job] ⚠️ 생성 실패: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        out = {"error": f"생성 실패: {type(e).__name__}: {e}"}
        out.update(watcher.result())
        return out
    finally:
        watcher.stop_flag.set()

    elapsed = time.time() - t0
    size_mb = len(payload) / 1024**2

    result = {
        "format": "mp4",
        "bytes": len(payload),
        "size_mb": round(size_mb, 2),
        "filename": filename,
        "generation_sec": round(elapsed, 1),
    }

    if p is not None:
        result.update({
            # ⭐ 어떤 배선으로 돌았는지. 사진이 실제로 물렸는지 확인하는 단서다.
            #    로라와 같은 이유로 넣는다 — 사진이 안 물려도 영상은 나오기 때문이다.
            "mode": p["mode"],
            "image": p["image"] or None,
            "image_strength": p["image_strength"] if p["mode"] == "i2v" else None,
            "width": p["width"], "height": p["height"],
            "frames": p["frames"], "fps": p["fps"],
            "steps": p["steps"], "cfg": p["cfg"],
            "seed": p["seed"], "sampler": p["sampler"],
            "audio": p["audio"],
            "upscale_spatial": p["upscale_spatial"],
            "stock": p["stock"],
            # ⭐ 무엇이 실제로 걸렸는지 담는다.
            #    로라는 안 붙어도 오류가 안 나므로 이 값이 유일한 단서다.
            "loras_applied": [
                {"name": s["name"], "file": s["file"], "strength": s["strength"]}
                for s in p["loras"]
            ] or None,
        })
        result["model"] = f"{MODEL_REPO}/{MODEL_FILE}"
        result["text_encoder"] = f"{TE_REPO}/{TE_FILE}"

    # ⚠️ base64 는 원본의 약 1.33배가 된다.
    #    런포드 제한은 /runsync 20MB, /run 10MB 다.
    if RETURN_MODE == "base64":
        if size_mb > MAX_RETURN_MB:
            result["error"] = (
                f"영상이 {size_mb:.1f} MB 라 응답에 담지 못한다 "
                f"(한계 {MAX_RETURN_MB} MB). "
                f"해상도·길이를 줄이거나 RETURN_MODE 를 바꿔야 한다. "
                f"영상 자체는 워커 안에 만들어져 있다: {filename}")
        else:
            result["video_base64"] = base64.b64encode(payload).decode("utf-8")
    else:
        result["error"] = f"RETURN_MODE '{RETURN_MODE}' 는 아직 구현 안 됐다"

    if _boot_stats:
        result["boot"] = dict(_boot_stats)
        _boot_stats.clear()
    result.update(watcher.result())

    log(f"[job] 완료 {elapsed:.1f}초 / {size_mb:.1f} MB "
        f"/ VRAM 최대사용 {result.get('vram_peak_used_gb')} GB")
    return result
