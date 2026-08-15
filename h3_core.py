"""미니맥스 H3 영상+소리 워커 — 몸통 (런포드에 의존하지 않는 부분)

이 파일은 런포드를 import 하지 않는다. 로컬에서도 그대로 쓴다.
런포드 진입점은 handler_h3.py 에 따로 있다.

  h3_core.py     ⬅ 이 파일. 파일 받기 / ComfyUI 띄우기 / 워크플로 조립 / 실행 / 결과 꺼내기
  handler_h3.py  ⬅ 껍데기. runpod.serverless.start() 만 있다

명세: E:\\미니맥스H3_분석\\명세_워커h3.html
배선 근거: Comfy-Org 공식 워크플로 템플릿(video_minimax_h3_t2v/i2v/r2v.json)의
           definitions.subgraphs 안 노드·연결 30개를 전수 판독했다 (2026-08-14).
           노드 이름과 입력 이름은 전부 거기서 그대로 옮긴 것이다. 지어낸 것이 없다.

⚠️ 이 파일에 검사문(assert)을 함부로 붙이지 말 것.
   크레아2 빌드 실패 4건이 전부 나중에 덧붙인 검사문에서 났다.
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

COMFY_INPUT_DIR = os.environ.get("COMFYUI_INPUT_DIR", "/comfyui/input")
MODELS_DIR = os.environ.get("COMFYUI_MODELS_DIR", "/comfyui/models")

# 노드 결과 캐시. LTX 와 같은 이유로 기본 none 이다 —
#   ① ComfyUI 가 컨테이너 제한을 모르고 호스트 RAM 기준으로 캐시를 쌓아 RAM 이 터진다
#   ② 같은 조건을 두 번 던지면 두 번째가 가짜로 빨라져 측정이 오염된다
COMFY_CACHE_MODE = os.environ.get("COMFY_CACHE_MODE", "none").strip().lower()

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_ACCESS_TOKEN")


def log(msg):
    print(msg, flush=True)


# ⚠️ log 는 반드시 여기(파일 위쪽)에 있어야 한다.
#    아래 LORA_CATALOG 를 만드는 줄이 모듈을 불러오는 순간 실행되는데,
#    카탈로그에 잘못된 줄이 있으면 그 안에서 log 를 부른다.
#    log 가 아래쪽에 있으면 NameError 로 워커가 즉사한다. (LTX 에서 그대로 가져온 교훈)


def _env(name, default=""):
    """환경변수를 읽는다. 앞뒤 공백은 버린다.

    ⚠️ 런포드는 빈 문자열·공백만 있는 환경변수를 컨테이너에 안 넣는다.
       그래서 "끄기"는 빈 값이 아니라 none/off/- 같은 명시적 값으로 한다.
    """
    return os.environ.get(name, default).strip()


def _off(v):
    """이 값이 '끄기'를 뜻하는가."""
    return v.lower() in ("", "none", "off", "-", "0", "false")


# ── 부품 4개 ──────────────────────────────────────────────────────────
#
# 전부 Comfy-Org/MiniMax-H3 한 곳에서 받는다. 크기는 2026-08-14 허깅페이스 API 실측이다.
#
#   본체      19.5 GB   → models/diffusion_models  (코어 UNETLoader)
#   글 이해기 14.6 GB   → models/text_encoders     (코어 CLIPLoader, type=minimax)
#   영상 VAE   4.9 GB   → models/vae               (코어 VAELoader)
#   소리 VAE   0.6 GB   → models/vae               (코어 VAELoader)  ⬅ LTX 와 다르다
#
# ⚠️ LTX 는 소리 VAE 를 models/checkpoints 에 놓아야 했다(LTXVAudioVAELoader 가 거기를 봐서).
#    H3 는 둘 다 models/vae 다. 공식 모델카드와 공식 워크플로 양쪽에서 확인했다.

MODEL_REPO = _env("MODEL_REPO", "Comfy-Org/MiniMax-H3")
MODEL_FILE = _env("MODEL_FILE",
                  "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors")

TE_REPO = _env("TE_REPO", "Comfy-Org/MiniMax-H3")
TE_FILE = _env("TE_FILE",
               "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")

VAE_REPO = _env("VAE_REPO", "Comfy-Org/MiniMax-H3")
VAE_FILE = _env("VAE_FILE", "vae/minimax_h3_video_vae_fp16.safetensors")

AVAE_REPO = _env("AVAE_REPO", "Comfy-Org/MiniMax-H3")
AVAE_FILE = _env("AVAE_FILE", "vae/minimax_h3_audio_vae_fp32.safetensors")

# CLIPLoader 의 type 칸. 공식 워크플로 위젯값이 'minimax' 다.
CLIP_TYPE = _env("CLIP_TYPE", "minimax")

# 부품을 더 받고 싶을 때 (LTX 와 같은 형식)
#   EXTRA_FILES = 저장소|저장소안경로|ComfyUI폴더 ; ...
# 참조모드(ref2va) 본체를 나중에 붙일 때 이 칸으로 받으면 재빌드가 필요 없다.
EXTRA_FILES = _env("EXTRA_FILES")

# ── 기본값 ────────────────────────────────────────────────────────────
#
# 공식 워크플로 템플릿의 값을 그대로 가져왔다.
#   해상도 864x480 = 0.4MP 16:9 (공식 해상도표의 기본 줄)
#   스텝 20 / 스케줄러 simple / denoise 1 → BasicScheduler 위젯 ['simple', 20, 1]
#   샘플러 res_multistep → KSamplerSelect 위젯
#   fps 24 → CreateVideo 위젯
DEF_WIDTH = int(_env("WIDTH", "864") or 864)
DEF_HEIGHT = int(_env("HEIGHT", "480") or 480)
DEF_DURATION = float(_env("DURATION", "5") or 5)
DEF_FPS = int(_env("FPS", "24") or 24)
DEF_STEPS = int(_env("STEPS", "20") or 20)
DEF_DENOISE = float(_env("DENOISE", "1.0") or 1.0)
DEF_SAMPLER = _env("SAMPLER", "res_multistep")
DEF_SCHEDULER = _env("SCHEDULER", "simple")

# ⭐ 시그마 시프트 — 기본은 끔
#    켜면 코어 노드 MiniMaxH3SigmaShift 를 모델 사슬에 끼운다.
#    터보 로라 제작자(Abiray)가 권장한 값이 영상 12 / 소리 6 이다.
#    ComfyUI 노드 자체의 기본값은 영상 12 / 소리 3 이라 소리 쪽만 다르다.
#    ⚠️ 공식 기본 템플릿에는 이 노드가 없다. 끄면 공식과 똑같은 배선이 된다.
DEF_SIGMA_SHIFT = not _off(_env("SIGMA_SHIFT", "off"))
DEF_SHIFT_VIDEO = float(_env("SHIFT_VIDEO", "12.0") or 12.0)
DEF_SHIFT_AUDIO = float(_env("SHIFT_AUDIO", "6.0") or 6.0)

# H3 의 화폭 한계. 짧은 변 768 이 기본이고 1344x768(0.98MP)이 실용 상한이다.
# 전부 32 의 배수여야 한다. (공식 워크플로 설명 노트)
CANVAS_MULTIPLE = 32
CANVAS_MAX_PIXELS = 1344 * 768

# ⭐ ComfyUI 실행 플래그 — 이게 24GB 안에 들어가게 하는 핵심이다
#
#   --disable-pinned-memory
#     ComfyUI 는 리눅스에서 시스템 RAM 의 90% 를 page-lock 한다
#     (comfy/model_management.py: MAX_PINNED_MEMORY = ram * 0.90).
#     잠긴 페이지는 커널이 회수할 수 없어서, 메모리가 빠듯해지면 커널이
#     프로세스를 SIGKILL 한다. 스왑을 늘려도 소용없다.
#     공개 실측(tonyd2wild/minimax-h3-local): 같은 조건에서
#       기본값                    호스트 RAM 29,866 MB → 커널이 죽임
#       --disable-pinned-memory   호스트 RAM  7,508 MB → 정상 완료 (15초 영상)
#
#   --fp16-intermediates
#     노드 사이를 오가는 텐서를 절반 크기로 만든다. 원저자 표현으로 "싼 보험".
#
# ⚠️ 둘 다 환경변수로 끌 수 있게 해뒀다. 문제가 생기면 재빌드 없이 되돌린다.
DISABLE_PINNED_MEMORY = not _off(_env("DISABLE_PINNED_MEMORY", "on"))
FP16_INTERMEDIATES = not _off(_env("FP16_INTERMEDIATES", "on"))

# ── 결과 반환 ─────────────────────────────────────────────────────────
#
# 런포드 응답 크기 제한: /runsync 20 MB, /run 10 MB. base64 는 원본의 약 1.33배다.
RETURN_MODE = _env("RETURN_MODE", "base64").lower()
MAX_RETURN_MB = float(_env("MAX_RETURN_MB", "14") or 14)

# ──────────────────────────────────────────────────────────────────────
# ⭐ 로라 카탈로그 — 줄 하나가 버튼 하나다
#
# LORA_CATALOG 형식 (줄바꿈 또는 세미콜론으로 구분):
#   이름|저장소|파일경로|종류|기본강도|설명
#
#   종류  plain  그냥 붙인다 (지금은 이것만 있다)
#
# 예)
#   turbo-6|SanDiegoDude/H3-Turbo-6-Step-LoRA-Comfy|minimax_h3_turbo_6step_ema_fl2va_pruned.safetensors|plain|1.0|터보 6스텝
#
# ⭐ 여기 한 줄 추가하면 capabilities 응답에 항목이 하나 늘고,
#    UI 를 안 고쳐도 화면에 스위치가 저절로 하나 생긴다.
# ⭐ 카탈로그에 없는 로라는 요청에서 repo/file 을 직접 적어 쓸 수 있다(resolve_loras 참고).
#    LTX 에는 없는 기능이고 크레아2 에는 있다. H3 는 크레아2 쪽을 따른다.
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
        if len(parts) < 3 or not all(parts[:3]):
            log(f"[catalog] ⚠️ 칸이 모자라 건너뜀: {line!r}")
            continue
        name, repo, path = parts[0], parts[1], parts[2]
        kind = (parts[3] if len(parts) > 3 and parts[3] else "plain").lower()
        if kind != "plain":
            log(f"[catalog] ⚠️ 모르는 종류 '{kind}' → plain 으로 본다: {name}")
            kind = "plain"
        try:
            strength = float(parts[4]) if len(parts) > 4 and parts[4] else 1.0
        except ValueError:
            log(f"[catalog] ⚠️ 강도가 숫자가 아니라 1.0 으로 본다: {name}")
            strength = 1.0
        label = parts[5] if len(parts) > 5 else name
        out.append({
            "name": name, "repo": repo, "path": path,
            "kind": kind, "default_strength": strength, "label": label,
        })
    return out


LORA_CATALOG = _parse_catalog(_env("LORA_CATALOG"))

# 워커가 뜰 때 미리 받아둘 로라. 쉼표로 나열한다.
#   비움  아무것도 미리 안 받는다 (요청에서 켜는 순간 받는다)  ⬅ 기본
#   이름  그것만 미리 받는다
#   *     카탈로그 전부
LORA_PRELOAD = _env("LORA_PRELOAD")

_lora_ready = set()

T_PROCESS_START = time.time()
_boot_stats = {}

COMFY_PROC = None


# ──────────────────────────────────────────────────────────────────────
# 1) 파일 받기
# ──────────────────────────────────────────────────────────────────────

def _download_one(repo_id, filename, comfy_folder):
    """받아서 <MODELS_DIR>/<comfy_folder>/<파일명> 에 놓는다.

    ⚠️ 저장소가 어떤 폴더 구조를 쓰든 여기서 평평하게 만든다.
       H3 저장소는 diffusion_models/ vae/ 같은 하위폴더를 쓰는데,
       ComfyUI 로더는 파일 이름만 본다.

    ⚠️ 이미 있으면 다시 받지 않는다.
       서버리스에선 매번 컨테이너가 사라지므로 영향이 없고,
       로컬에선 한 번만 받고 계속 쓰게 된다.
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


def _parse_extra(raw):
    """EXTRA_FILES 를 읽어 다운로드 목록으로 만든다. 형식: 저장소|경로|폴더"""
    out = []
    if _off(raw):
        return out
    for line in raw.replace(";", "\n").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3 or not all(parts[:3]):
            log(f"[extra] ⚠️ 칸이 모자라 건너뜀 (저장소|경로|폴더): {line!r}")
            continue
        out.append((parts[0], parts[1], parts[2]))
        log(f"[extra] 추가 부품: {parts[0]}/{parts[1]} → {parts[2]}/")
    return out


def _base_jobs():
    """항상 받아야 하는 부품 4개. (저장소, 저장소안 경로, ComfyUI 폴더)

    폴더는 어느 로더가 그 파일을 보느냐로 정해진다 — 바꾸면 안 된다.
      diffusion_models  코어 UNETLoader
      text_encoders     코어 CLIPLoader
      vae               코어 VAELoader (영상·소리 둘 다)
    """
    jobs = [
        (MODEL_REPO, MODEL_FILE, "diffusion_models"),
        (TE_REPO, TE_FILE, "text_encoders"),
        (VAE_REPO, VAE_FILE, "vae"),
        (AVAE_REPO, AVAE_FILE, "vae"),
    ]
    jobs.extend(_parse_extra(EXTRA_FILES))
    return jobs


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


def ensure_lora_file(repo, path):
    """로라 파일이 없으면 그때 받는다. 반환값은 (파일명, 오류메시지)."""
    key = f"{repo}/{path}"
    fname = os.path.basename(path)
    if key in _lora_ready:
        return fname, None
    try:
        _download_one(repo, path, "loras")
        _lora_ready.add(key)
        return fname, None
    except Exception as e:
        return None, (f"로라를 못 받았다: {repo}/{path} — {type(e).__name__}: {e}. "
                      f"게이팅 저장소면 HF_TOKEN 에 약관 동의가 되어 있어야 한다")


def _preload_names():
    raw = LORA_PRELOAD
    if _off(raw):
        return []
    if raw.strip() == "*":
        return [c["name"] for c in LORA_CATALOG]
    return [x.strip() for x in raw.split(",") if x.strip()]


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

    names = _preload_names()
    if names:
        log(f"[download] 로라 미리 받기 {len(names)}개: {', '.join(names)}")
        for n in names:
            spec = find_lora(n)
            if spec is None:
                log(f"[download] ⚠️ 카탈로그에 없어 건너뜀: {n}")
                continue
            _, err = ensure_lora_file(spec["repo"], spec["path"])
            if err:
                # ⚠️ 미리 받기 실패로 워커를 죽이지 않는다. 그 로라만 못 쓰게 둔다.
                log(f"[download] ⚠️ {err}")

    sec = time.time() - t0
    log(f"[download] 완료 — 총 {total / 1024**3:.2f} GiB, {sec:.1f}초")
    _boot_stats["download_sec"] = round(sec, 1)
    _boot_stats["download_gib"] = round(total / 1024**3, 2)


# ──────────────────────────────────────────────────────────────────────
# 2) ComfyUI 띄우기
# ──────────────────────────────────────────────────────────────────────

def cache_args():
    """COMFY_CACHE_MODE 를 ComfyUI 명령줄 옵션으로 바꾼다."""
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
        vals = [v.strip() for v in m[4:].split(",") if v.strip()]
        return ["--cache-ram"] + vals
    if m.startswith("lru:"):
        return ["--cache-lru", m[4:].strip()]
    log(f"[comfyui] ⚠️ COMFY_CACHE_MODE='{m}' 를 모르겠다. 기본값 none 으로 간다.")
    return ["--cache-none"]


def memory_args():
    """24GB 안에 들어가게 하는 플래그. 위 DISABLE_PINNED_MEMORY 주석 참고."""
    args = []
    if DISABLE_PINNED_MEMORY:
        args.append("--disable-pinned-memory")
    if FP16_INTERMEDIATES:
        args.append("--fp16-intermediates")
    return args


def start_comfyui():
    cmd = [
        sys.executable, "-u", "/comfyui/main.py",
        "--listen", COMFY_HOST,
        "--port", str(COMFY_PORT),
        "--disable-auto-launch",
        "--disable-metadata",
        "--disable-api-nodes",
        "--log-stdout",
    ] + memory_args() + cache_args()
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
# 3) 길이·해상도 맞추기
# ──────────────────────────────────────────────────────────────────────

def frames_for(seconds, fps=24):
    """초를 H3 가 받는 프레임 수로 바꾼다.

    H3 는 17 의 배수 + 5 인 값만 받는다. 공식 워크플로의 ComfyMathExpression 노드에
    적힌 식을 그대로 옮긴 것이다:

        max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17

    5초  → 124 프레임
    15초 → 362 프레임  (훈련된 최대 길이)
    """
    n = max(5, int(round(float(seconds) * fps)))
    return n + (5 - (n % 17)) % 17


def snap_canvas(w, h):
    """32 의 배수로 맞추고 화폭 상한을 넘지 않게 한다.

    공식 워크플로 설명: "H3's native canvas is a 768px short edge,
    capped at 768x1344 pixels, rounded to a multiple of 32"

    ⚠️ 상한을 넘으면 오류로 막지 않고 비율을 지키며 줄인다.
       막아버리면 왜 안 되는지 모른 채 요청이 실패하기 때문이다. 줄인 값은 응답에 담는다.
    """
    w = max(CANVAS_MULTIPLE, int(w))
    h = max(CANVAS_MULTIPLE, int(h))
    if w * h > CANVAS_MAX_PIXELS:
        scale = (CANVAS_MAX_PIXELS / float(w * h)) ** 0.5
        w = int(w * scale)
        h = int(h * scale)
    w = max(CANVAS_MULTIPLE, (w // CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    h = max(CANVAS_MULTIPLE, (h // CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return w, h


# ──────────────────────────────────────────────────────────────────────
# 4) 워크플로 조립
#
# ⭐ 아래 노드 이름·입력 이름은 전부 Comfy-Org 공식 템플릿에서 그대로 옮긴 것이다.
#    (video_minimax_h3_t2v.json 의 definitions.subgraphs, 노드 15개 · 연결 30개 판독)
#
#    UNETLoader ──┬────────────────► BasicGuider ──┐
#                 └► BasicScheduler ──► SIGMAS ────┤
#    CLIPLoader(minimax) ─┐                        ├─► SamplerCustomAdvanced
#    VAELoader(영상) ─────┼─► MiniMaxH3ImageToVideo┘        │
#                         │      0: CONDITIONING            ├─► VAEDecode ──────┐
#                         │      1: LATENT                  └─► VAEDecodeAudio ─┤
#    VAELoader(소리) ─────────────────────────────────────────────┘             │
#    RandomNoise ─► NOISE                                    CreateVideo ◄──────┘
#    KSamplerSelect(res_multistep) ─► SAMPLER                     │
#                                                              SaveVideo
#
# ⚠️ 확인된 사실 — MiniMaxH3ImageToVideo 는 vae 를 하나만 받는다(영상 VAE).
#    소리 VAE 는 VAEDecodeAudio 에만 물린다. 그림만 보고 둘 다 물린다고 짐작하면 틀린다.
# ──────────────────────────────────────────────────────────────────────

def build_workflow(p):
    wf = {}

    # ── 로더 ──────────────────────────────────────────────────────────
    wf["1"] = {"class_type": "UNETLoader",
               "inputs": {"unet_name": os.path.basename(MODEL_FILE),
                          "weight_dtype": "default"}}
    wf["2"] = {"class_type": "CLIPLoader",
               "inputs": {"clip_name": os.path.basename(TE_FILE),
                          "type": CLIP_TYPE,
                          "device": "default"}}
    wf["3"] = {"class_type": "VAELoader",
               "inputs": {"vae_name": os.path.basename(VAE_FILE)}}
    wf["4"] = {"class_type": "VAELoader",
               "inputs": {"vae_name": os.path.basename(AVAE_FILE)}}

    # ── 로라 사슬 (모델에만 붙인다) ────────────────────────────────────
    #    ⚠️ 본체는 BasicGuider 와 BasicScheduler 두 곳에 물린다.
    #       사슬 끝을 양쪽에 똑같이 넣어야 로라가 실제로 반영된다.
    model_ref = ["1", 0]
    nid = 10
    for spec in p["loras"]:
        wf[str(nid)] = {"class_type": "LoraLoaderModelOnly",
                        "inputs": {"model": model_ref,
                                   "lora_name": spec["file"],
                                   "strength_model": spec["strength"]}}
        model_ref = [str(nid), 0]
        nid += 1

    # ── 시그마 시프트 (선택) ──────────────────────────────────────────
    #    코어 노드 MiniMaxH3SigmaShift (comfy_extras/nodes_minimax_h3.py 283행).
    #    영상과 소리의 잡음 일정을 따로 조절한다. 기본값은 shift_video 12 / shift_audio 3.
    #
    #    ⭐ 왜 넣었나 — 터보 로라 제작자(Abiray)가 권장 설정으로
    #       "Video Sigma Shift 12 / Audio Sigma Shift 6" 을 명시했다.
    #       이 노드가 없으면 그 설정을 줄 방법이 없다.
    #    ⚠️ 공식 기본 템플릿에는 이 노드가 없다. 그래서 기본은 끔(안 넣음)이고,
    #       요청이나 환경변수로 켤 때만 사슬에 끼운다. 순정 동작은 안 바뀐다.
    if p["sigma_shift"]:
        wf[str(nid)] = {"class_type": "MiniMaxH3SigmaShift",
                        "inputs": {"model": model_ref,
                                   "shift_video": p["shift_video"],
                                   "shift_audio": p["shift_audio"]}}
        model_ref = [str(nid), 0]
        nid += 1

    # ── 입력 사진 (선택) ──────────────────────────────────────────────
    h3_inputs = {
        "clip": ["2", 0],
        "vae": ["3", 0],
        "prompt": p["prompt"],
        "width": p["width"],
        "height": p["height"],
        "length": p["length"],
    }
    if p["image"]:
        wf["5"] = {"class_type": "LoadImage", "inputs": {"image": p["image"]}}
        h3_inputs["first_frame"] = ["5", 0]
    if p["last_image"]:
        wf["6"] = {"class_type": "LoadImage", "inputs": {"image": p["last_image"]}}
        h3_inputs["last_frame"] = ["6", 0]

    # ── H3 본 노드 — 출력 0=CONDITIONING, 1=LATENT ────────────────────
    wf["20"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": h3_inputs}

    # ── 샘플링 ────────────────────────────────────────────────────────
    wf["21"] = {"class_type": "BasicGuider",
                "inputs": {"model": model_ref, "conditioning": ["20", 0]}}
    wf["22"] = {"class_type": "BasicScheduler",
                "inputs": {"model": model_ref,
                           "scheduler": p["scheduler"],
                           "steps": p["steps"],
                           "denoise": p["denoise"]}}
    wf["23"] = {"class_type": "KSamplerSelect",
                "inputs": {"sampler_name": p["sampler"]}}
    wf["24"] = {"class_type": "RandomNoise",
                "inputs": {"noise_seed": p["seed"]}}
    wf["25"] = {"class_type": "SamplerCustomAdvanced",
                "inputs": {"noise": ["24", 0],
                           "guider": ["21", 0],
                           "sampler": ["23", 0],
                           "sigmas": ["22", 0],
                           "latent_image": ["20", 1]}}

    # ── 꺼내기 — 같은 잠재에서 영상과 소리를 따로 꺼낸다 ───────────────
    wf["26"] = {"class_type": "VAEDecode",
                "inputs": {"samples": ["25", 0], "vae": ["3", 0]}}
    wf["27"] = {"class_type": "VAEDecodeAudio",
                "inputs": {"samples": ["25", 0], "vae": ["4", 0]}}

    # ── 합치고 저장 ───────────────────────────────────────────────────
    # bit_depth 는 선택 항목이고 기본이 8 이다. 공식 템플릿 위젯값도 8 이라 명시해 둔다.
    wf["28"] = {"class_type": "CreateVideo",
                "inputs": {"images": ["26", 0],
                           "fps": p["fps"],
                           "audio": ["27", 0],
                           "bit_depth": 8}}
    wf["29"] = {"class_type": "SaveVideo",
                "inputs": {"video": ["28", 0],
                           "filename_prefix": "video/MiniMax_H3",
                           "format": "auto",
                           "codec": "auto"}}
    return wf


# ──────────────────────────────────────────────────────────────────────
# 5) 실행 · 결과 꺼내기   (LTX 워커에서 검증된 부분을 그대로 쓴다)
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
                f"메모리 부족일 가능성이 높다. "
                f"--disable-pinned-memory 가 켜져 있는지 로그에서 확인할 것.")
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
       그래서 videos/gifs/images/files 를 모두 뒤지고, 그래도 없으면
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
# 6) 요청 해석
# ──────────────────────────────────────────────────────────────────────

def save_media(job):
    """요청에 실려온 사진을 ComfyUI 가 읽는 폴더에 파일로 놓는다."""
    media = job.get("media")
    if not media:
        return [], None
    if not isinstance(media, list):
        return None, "media 는 목록이어야 한다."

    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
    saved = []
    for i, item in enumerate(media):
        if not isinstance(item, dict):
            return None, f"media[{i}] 가 잘못됐다: {item!r}"
        name = os.path.basename(str(item.get("name") or "").strip())
        data = item.get("data")
        if not name or not data:
            return None, f"media[{i}] 에 name 또는 data 가 없다."

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
    """요청의 loras 목록을 실제 파일로 바꾸고, 없으면 그때 받는다.

    두 가지 방식을 다 받는다 —
      ① 카탈로그 이름   {"name": "turbo-6", "strength": 1.0}
      ② 주소로 직접     {"repo": "...", "file": "...", "strength": 1.0}

    ⭐ ②가 있어서 카탈로그에 없는 로라도 재빌드·재시작 없이 바로 시험할 수 있다.
       LTX 워커에는 없는 기능이고 크레아2 에는 있다.

    ⚠️ 못 찾거나 못 받으면 조용히 넘어가지 않고 오류로 돌려준다.
       로라는 안 붙어도 ComfyUI 가 오류를 안 내기 때문에,
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

        repo = str(item.get("repo") or "").strip()
        path = str(item.get("file") or "").strip()
        strength = item.get("strength")

        if repo and path:
            # ② 주소로 직접
            name = f"{repo}/{os.path.basename(path)}"
            default_strength = 1.0
        else:
            # ① 카탈로그 이름
            key = str(item.get("name") or path or "").strip()
            spec = find_lora(key)
            if spec is None:
                have = ", ".join(c["name"] for c in LORA_CATALOG) or "(카탈로그가 비어 있다)"
                return None, (
                    f"로라를 못 찾았다: '{key}'. 카탈로그에 있는 것: {have}. "
                    f"카탈로그에 없는 것을 쓰려면 repo 와 file 을 직접 적어라 "
                    f"(예: {{\"repo\":\"fal/MiniMax-H3-Realism-People-LoRA\","
                    f"\"file\":\"h3-realism-people-t2v-i2v-r2v.safetensors\"}})")
            repo, path = spec["repo"], spec["path"]
            name = spec["name"]
            default_strength = spec["default_strength"]

        fname, err = ensure_lora_file(repo, path)
        if err:
            return None, err

        try:
            strength = float(strength) if strength is not None else default_strength
        except (TypeError, ValueError):
            return None, f"'{name}' 의 강도가 숫자가 아니다: {strength!r}"

        out.append({"name": name, "repo": repo, "file": fname, "strength": strength})
    return out, None


def parse_params(job):
    """요청을 기본값과 합쳐 실제로 쓸 값으로 만든다."""
    stock = bool(job.get("stock"))

    seed = job.get("seed", None)
    if seed in (None, -1, "random"):
        seed = random.randint(0, 2**31 - 1)

    image = str(job.get("image") or "").strip()
    last_image = str(job.get("last_image") or "").strip()
    mode = str(job.get("mode") or "").strip().lower()
    if not mode:
        mode = "i2v" if image else "t2v"

    fps = int(job.get("fps", DEF_FPS))
    duration = float(job.get("duration", DEF_DURATION))

    # ⭐ frames 를 직접 준 경우에도 17n+5 격자에 맞춘다. 안 맞으면 ComfyUI 가 거부한다.
    if job.get("frames") is not None:
        raw = int(job["frames"])
        length = raw + (5 - (raw % 17)) % 17
    else:
        length = frames_for(duration, fps)

    req_w = int(job.get("width", DEF_WIDTH))
    req_h = int(job.get("height", DEF_HEIGHT))
    width, height = snap_canvas(req_w, req_h)

    return {
        "mode": mode,
        "image": os.path.basename(image) if image else "",
        "last_image": os.path.basename(last_image) if last_image else "",
        "prompt": str(job.get("prompt") or "").strip(),
        "width": width, "height": height,
        "requested_width": req_w, "requested_height": req_h,
        "length": length,
        "duration": round(length / float(fps), 2),
        "fps": fps,
        "steps": int(job.get("steps", DEF_STEPS)),
        "denoise": float(job.get("denoise", DEF_DENOISE)),
        "sampler": str(job.get("sampler", DEF_SAMPLER)),
        "scheduler": str(job.get("scheduler", DEF_SCHEDULER)),
        "seed": int(seed),
        "stock": stock,
        # 시그마 시프트 — 안 켜면 공식 템플릿과 완전히 같은 배선이 된다
        "sigma_shift": bool(job.get("sigma_shift", DEF_SIGMA_SHIFT)),
        "shift_video": float(job.get("shift_video", DEF_SHIFT_VIDEO)),
        "shift_audio": float(job.get("shift_audio", DEF_SHIFT_AUDIO)),
        "loras": [],
    }


# ──────────────────────────────────────────────────────────────────────
# 7) ⭐ 버튼 목록 — UI 가 이걸 받아 화면을 스스로 그린다
# ──────────────────────────────────────────────────────────────────────

def capabilities():
    return {
        "model": {
            "name": os.path.basename(MODEL_FILE),
            "repo": MODEL_REPO,
            "backend": "safetensors",
            "text_encoder": os.path.basename(TE_FILE),
            "clip_type": CLIP_TYPE,
            "note": "영상과 소리를 한 번에 만든다. 32kHz 스테레오가 영상에 같이 담긴다",
        },
        "features": [
            {"id": "duration", "label": "길이(초)", "type": "float",
             "default": DEF_DURATION, "min": 0.2, "max": 15.1,
             "note": "17프레임 격자에 맞춰 자동 보정된다. 5초=124프레임, 15초=362프레임(훈련 상한)"},
            {"id": "steps", "label": "스텝", "type": "int",
             "default": DEF_STEPS, "min": 1, "max": 60,
             "note": "터보 로라를 쓰면 6~8 로 줄인다. 4 까지 내리면 소리가 망가진다는 보고가 있다"},
            {"id": "width", "label": "가로", "type": "int", "default": DEF_WIDTH},
            {"id": "height", "label": "세로", "type": "int", "default": DEF_HEIGHT,
             "note": "짧은 변 768 이 기본 화폭. 1344x768 이 실용 상한이고 32의 배수여야 한다"},
            {"id": "fps", "label": "초당 프레임", "type": "int", "default": DEF_FPS},
            {"id": "seed", "label": "시드", "type": "int", "default": -1,
             "note": "-1 이면 매번 무작위"},
            {"id": "sampler", "label": "샘플러", "type": "text", "default": DEF_SAMPLER},
            {"id": "scheduler", "label": "스케줄러", "type": "text", "default": DEF_SCHEDULER},
            {"id": "stock", "label": "순정으로 뽑기", "type": "switch", "default": False,
             "note": "로라를 전부 끈다. 로라가 실제로 붙었는지 대조할 때 쓴다"},
            {"id": "sigma_shift", "label": "시그마 시프트 쓰기", "type": "switch",
             "default": DEF_SIGMA_SHIFT,
             "note": "터보 로라를 쓸 때 켠다. 끄면 공식 템플릿과 똑같은 배선이 된다"},
            {"id": "shift_video", "label": "영상 시그마 시프트", "type": "float",
             "default": DEF_SHIFT_VIDEO, "min": 0.01, "max": 100.0},
            {"id": "shift_audio", "label": "소리 시그마 시프트", "type": "float",
             "default": DEF_SHIFT_AUDIO, "min": 0.01, "max": 100.0},
        ],
        "loras": [
            {
                "name": c["name"],
                "label": c["label"],
                "kind": c["kind"],
                "default_strength": c["default_strength"],
                "supported": True,
                "ready": f"{c['repo']}/{c['path']}" in _lora_ready,
            }
            for c in LORA_CATALOG
        ],
        "lora_direct": {
            "how": "카탈로그에 없는 로라는 repo 와 file 을 직접 적으면 바로 쓸 수 있다",
            "format": {"repo": "fal/MiniMax-H3-Realism-People-LoRA",
                       "file": "h3-realism-people-t2v-i2v-r2v.safetensors",
                       "strength": 1.0},
            "note": "재빌드도 워커 재시작도 필요 없다. 전체 목록은 H3_로라목록.html 참고",
        },
        "modes": [
            {"id": "t2v", "label": "글자 → 영상", "needs": [],
             "note": "기본값. image 를 안 주면 이걸로 간다"},
            {"id": "i2v", "label": "사진 → 영상", "needs": ["image"],
             "note": "첫 프레임을 사진으로 고정한다. last_image 를 주면 끝 프레임도 고정된다"},
        ],
        "media": {
            "how": "요청에 media 를 실으면 워커가 컨테이너 안 입력 폴더에 파일로 놓는다",
            "format": [{"name": "파일 이름 (예: first.png)", "data": "base64 문자열"}],
            "then": "놓은 뒤 image / last_image 에 그 파일 이름을 적는다",
            "dir": COMFY_INPUT_DIR,
        },
        "workflow_passthrough": {
            "how": "ComfyUI API 형식 JSON 을 workflow 에 통째로 넣으면 그대로 실행한다",
            "note": "참조모드(MiniMaxH3ReferenceToVideo)처럼 아직 배선을 안 만든 것도 "
                    "이 통로로 쓸 수 있다. 다만 ref2va 본체를 EXTRA_FILES 로 먼저 받아야 한다",
        },
        "memory": {
            "disable_pinned_memory": DISABLE_PINNED_MEMORY,
            "fp16_intermediates": FP16_INTERMEDIATES,
            "note": "--disable-pinned-memory 가 24GB 안에 들어가게 하는 핵심이다. "
                    "끄면 호스트 RAM 이 4배로 늘어 커널이 워커를 죽일 수 있다",
        },
        "return": {"mode": RETURN_MODE, "max_mb": MAX_RETURN_MB},
        "cache": {"mode": COMFY_CACHE_MODE, "args": cache_args()},
        "notes": [
            "우리가 직접 돌려본 성능 수치가 아직 없다. VRAM·속도·화질 전부 미측정이다",
            "참조모드(ref2va)는 본체를 하나 더(19.5GB) 받아야 해서 1차에서 뺐다",
            "2K 재생성 모듈은 미니맥스가 공개하지 않았다. 768p 가 천장이다",
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# 8) 기동 · 생성
# ──────────────────────────────────────────────────────────────────────

def boot():
    """워커가 뜰 때 한 번 돈다."""
    global COMFY_PROC
    if not HF_TOKEN:
        log("[init] ⚠️ HF_TOKEN 이 없다. 비게이트 저장소만 받을 수 있다.")
    if not LORA_CATALOG:
        log("[init] ⚠️ LORA_CATALOG 가 비어 있다. 로라는 repo/file 직접 지정으로만 쓸 수 있다.")

    download_base_models()
    COMFY_PROC = start_comfyui()
    wait_for_comfyui(COMFY_PROC)

    _boot_stats["total_cold_start_sec"] = round(time.time() - T_PROCESS_START, 1)
    log(f"[init] 콜드스타트 총 {_boot_stats['total_cold_start_sec']}초 "
        f"(다운로드 {_boot_stats.get('download_sec')}초 + "
        f"ComfyUI 기동 {_boot_stats.get('comfyui_boot_sec')}초)")


def generate(job):
    """요청 하나를 처리한다. 이 함수가 몸통의 입구다."""
    if str(job.get("action", "")).lower() == "capabilities":
        return capabilities()

    # ⭐ 재료(사진)를 먼저 컨테이너 안에 놓는다.
    #    workflow 를 통째로 던질 때도 필요하므로 분기 앞에서 처리한다.
    saved, err = save_media(job)
    if err:
        return {"error": err}

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
        for key in ("image", "last_image"):
            name = p[key]
            if name:
                here = os.path.join(COMFY_INPUT_DIR, name)
                if name not in saved and not os.path.exists(here):
                    return {"error": f"{key} '{name}' 를 찾을 수 없다. "
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
            src = f" 사진={p['image']}" if p["image"] else ""
            src += f" 끝사진={p['last_image']}" if p["last_image"] else ""
            log(f"[job] 큐 등록 {prompt_id} — [{p['mode']}]{src} "
                f"{p['width']}x{p['height']} "
                f"{p['length']}프레임({p['duration']}초) {p['fps']}fps "
                f"{p['steps']}스텝 {p['sampler']}/{p['scheduler']} "
                f"seed{p['seed']} 로라=[{desc}]{' 순정' if p['stock'] else ''}")
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
            "mode": p["mode"],
            "image": p["image"] or None,
            "last_image": p["last_image"] or None,
            "width": p["width"], "height": p["height"],
            "length": p["length"], "duration": p["duration"], "fps": p["fps"],
            "steps": p["steps"], "denoise": p["denoise"],
            "sampler": p["sampler"], "scheduler": p["scheduler"],
            "seed": p["seed"],
            "stock": p["stock"],
            "sigma_shift": p["sigma_shift"],
            "shift_video": p["shift_video"] if p["sigma_shift"] else None,
            "shift_audio": p["shift_audio"] if p["sigma_shift"] else None,
            # ⭐ 무엇이 실제로 걸렸는지 담는다.
            #    로라는 안 붙어도 오류가 안 나므로 이 값이 유일한 단서다.
            "loras_applied": [
                {"name": s["name"], "repo": s["repo"],
                 "file": s["file"], "strength": s["strength"]}
                for s in p["loras"]
            ] or None,
        })
        # ⚠️ 요청한 크기를 우리가 줄였으면 알려준다. 조용히 바꾸지 않는다.
        if (p["requested_width"], p["requested_height"]) != (p["width"], p["height"]):
            result["canvas_adjusted"] = {
                "requested": f"{p['requested_width']}x{p['requested_height']}",
                "used": f"{p['width']}x{p['height']}",
                "why": "32의 배수 + 화폭 상한(1344x768) 에 맞췄다",
            }
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
