"""Krea 2 이미지 워커 — 몸통 (v9)

⭐ 이 파일은 런포드를 import 하지 않는다.
   나중에 로컬 컴퓨터로 옮길 때 이 파일을 그대로 쓴다. 껍데기(handler_krea2.py)만 버린다.
   영상 워커(ltx_core.py)와 같은 구조다. 보스 지시 — "나중에 로컬에 설치해서 구동시킬 거야".

v8(handler_krea2.py 811줄 한 덩어리)에서 옮겨오면서 바뀐 것
  1) 껍데기·몸통 분리          runpod 를 안 부른다. boot() 를 껍데기가 호출한다
  2) LORA_CATALOG             이름으로 로라를 부른다. 없으면 그때 받는다
  3) 요청에 저장소·파일 직접 지정  카탈로그에 없는 로라도 주소만 알면 쓴다
     ⚠️ 영상 워커에는 없는 길이다. 영상은 로라가 30개뿐이라 카탈로그로 충분했지만
        이미지는 1,895개라 다 등록할 수 없다(약 199KB). 그래서 주소 통로를 연다
  4) action: capabilities     버튼 목록을 돌려준다
  5) workflow 통째로 던지기     ComfyUI API JSON 을 그대로 실행한다
  6) stock: true              로라를 전부 끄고 순정으로 뽑는다 (검증 기준점)

v7·v8 기능은 그대로 살아 있다
  v7  로라 여러 개 사슬 / lora_start_step 스텝 분기
  v8  LORA_SKIP_KEYS 로 로라의 특정 층 빼기 (검열 대응 실험용)
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
MODELS_DIR = "/comfyui/models"
# 요청에 실려온 사진이 놓이는 곳. ComfyUI 의 LoadImage 가 여기를 본다.
COMFY_INPUT_DIR = os.environ.get("COMFYUI_INPUT_DIR", "/comfyui/input")

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

# ComfyUI 자식 프로세스. boot() 에서 채운다.
COMFY_PROC = None

# ──────────────────────────────────────────────────────────────────────
# 부품 — 전부 환경변수 (v8 에서 그대로 옮김)
#
#   MODEL_BACKEND 가 하는 일은 "어느 로더를 쓰고 어느 폴더에 놓느냐" 뿐이다.
#     fp8   → UNETLoader      / models/diffusion_models/
#     gguf  → UnetLoaderGGUF  / models/unet/
# ──────────────────────────────────────────────────────────────────────
COMFY_REPO = "Comfy-Org/Krea-2"

MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "fp8").strip().lower()
MODEL_REPO = os.environ.get("MODEL_REPO", COMFY_REPO).strip()
MODEL_FILE = os.environ.get(
    "MODEL_FILE", "diffusion_models/krea2_raw_fp8_scaled.safetensors").strip()

TE_REPO = os.environ.get("TE_REPO", COMFY_REPO).strip()
TE_FILE = os.environ.get("TE_FILE", "text_encoders/qwen3vl_4b_bf16.safetensors").strip()

VAE_REPO = os.environ.get("VAE_REPO", COMFY_REPO).strip()
VAE_FILE = os.environ.get("VAE_FILE", "vae/qwen_image_vae.safetensors").strip()

# ⚠️ LoRA 를 끄려면 LORA_FILE 에 none / off / - 중 하나를 넣는다.
#    빈 문자열이나 공백은 쓰면 안 된다. 런포드가 빈 값을 버려서 기본값이 되살아난다.
#    2026-08-05 에 실제로 당했다.
LORA_REPO_RAW = os.environ.get("LORA_REPO", COMFY_REPO).strip()
LORA_FILE_RAW = os.environ.get(
    "LORA_FILE", "loras/krea2_turbo_lora_rank_64_bf16.safetensors").strip()
LORA_STRENGTH_RAW = os.environ.get("LORA_STRENGTH", "0.6").strip()

USE_LORA = LORA_FILE_RAW.lower() not in ("", "none", "off", "-", "0", "false")


def _split_csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def _pick(lst, i, fallback):
    """i 번째 값. 목록이 짧으면 마지막 값을 쓴다."""
    if not lst:
        return fallback
    return lst[i] if i < len(lst) else lst[-1]


_LORA_FILES = _split_csv(LORA_FILE_RAW) if USE_LORA else []
_LORA_REPOS = _split_csv(LORA_REPO_RAW)
_LORA_STRENGTHS = [float(x) for x in _split_csv(LORA_STRENGTH_RAW)]

# 부팅 때 받아둘 로라. 요청에 loras 를 안 보내면 이게 전부 걸린다 (v8 과 같은 동작).
LORA_SPECS = [
    {
        "repo": _pick(_LORA_REPOS, i, COMFY_REPO),
        "file": f,
        "strength": _pick(_LORA_STRENGTHS, i, 0.6),
    }
    for i, f in enumerate(_LORA_FILES)
]

# ──────────────────────────────────────────────────────────────────────
# ⭐ v9 새 기능 — 로라 카탈로그
#
# 형식 (줄바꿈 또는 세미콜론으로 구분):
#   이름|저장소|저장소안 파일경로|기본강도|설명
#
# 예:
#   실사|RudySen/Krea2-realism-V2|krea2-realism-v2.safetensors|0.8|사진처럼
#   때깔|mgwr/M87|M87.safetensors|1.0|전체 완성도를 올린다
#
# ⚠️ 여기에 1,895개를 다 넣으면 안 된다. 약 199KB 라 환경변수로 감당이 안 되고,
#    버튼이 1,895개면 고르지도 못한다. 자주 쓰는 것만 넣는다.
#    나머지는 요청에 repo/file 을 직접 적어 쓴다 (아래 resolve_loras 참고).
# ──────────────────────────────────────────────────────────────────────

def _parse_catalog(raw):
    """LORA_CATALOG 를 읽어 목록으로 만든다. 잘못된 줄은 건너뛰고 로그에 남긴다."""
    out = []
    if not raw or raw.strip().lower() in ("none", "off", "-"):
        return out
    for line in raw.replace(";", "\n").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            log(f"[catalog] ⚠️ 칸이 모자라 건너뛴다: {line[:80]}")
            continue
        name, repo, path = parts[0], parts[1], parts[2]
        try:
            strength = float(parts[3]) if len(parts) > 3 and parts[3] else 1.0
        except ValueError:
            strength = 1.0
        out.append({
            "name": name,
            "repo": repo,
            "path": path,
            "file": os.path.basename(path),
            "default_strength": strength,
            "note": parts[4] if len(parts) > 4 else "",
        })
    return out


LORA_CATALOG = _parse_catalog(os.environ.get("LORA_CATALOG", ""))

# 워커가 뜰 때 미리 받아둘 카탈로그 로라. 쉼표로 나열. "*" 는 전부.
LORA_PRELOAD = os.environ.get("LORA_PRELOAD", "").strip()

# 지금 워커에 실제로 받아져 있는 로라 파일명
_lora_ready = set()

# ⭐ v8: 로라에서 특정 층을 빼고 쓴다.
#    체형 LoRA 를 얹으면 검열 처리가 그려지는 문제 때문에 만든 것이다.
#    이 LoRA 는 512개 텐서 중 64개가 txtfusion(프롬프트 해석부)을 건드린다.
#    ⚠️ "txtfusion 이 원인" 은 아직 추정이다. 그래서 무엇을 뺄지 고를 수 있게 했다.
#    ⚠️ 함정: "blocks" 로 쓰면 txtfusion.refiner_blocks 까지 걸려 LoRA 가 통째로 사라진다.
#            본체만 빼려면 점을 붙여 "blocks." 로 쓸 것.
LORA_SKIP_KEYS = [
    x.strip().lower()
    for x in os.environ.get("LORA_SKIP_KEYS", "").split(",")
    if x.strip()
]

# ⭐ 보스가 정한 값 (2026-08-04). 원출처는 Civitai 실사 워크플로 krea2_simple_v1.
DEFAULT_LORA_STRENGTH = LORA_SPECS[0]["strength"] if LORA_SPECS else 0.6
DEFAULT_STEPS = int(os.environ.get("STEPS", "12"))
# CFG 1.0 = 네거티브가 작동하지 않는다. 1 을 넘기면 생성 시간이 2배가 된다.
DEFAULT_CFG = 1.0
# v7: 로라를 몇 번째 스텝부터 걸지. 0 이면 처음부터.
DEFAULT_LORA_START_STEP = int(os.environ.get("LORA_START_STEP", "0"))
# v7: er_sde — 체형 로라 제작자 권장값. 실측에서 화질이 뚜렷이 좋았다 (2026-08-06).
DEFAULT_SAMPLER = os.environ.get("SAMPLER", "er_sde").strip()
DEFAULT_SCHEDULER = os.environ.get("SCHEDULER", "simple").strip()
DEFAULT_WIDTH = int(os.environ.get("WIDTH", "928"))
DEFAULT_HEIGHT = int(os.environ.get("HEIGHT", "1664"))
DEFAULT_SEED = 42

T_PROCESS_START = time.time()
_boot_stats = {}


def log(msg):
    print(msg, flush=True)


# ──────────────────────────────────────────────────────────────────────
# 1) 모델 받기
# ──────────────────────────────────────────────────────────────────────

def _model_folder():
    """본체가 들어갈 폴더. gguf → unet / fp8 → diffusion_models"""
    return "unet" if MODEL_BACKEND == "gguf" else "diffusion_models"


def _plan_downloads():
    """(저장소, 저장소안 경로, ComfyUI 폴더) 목록.

    ⚠️ 목적지를 명시해야 한다. 예전에 realrebelai 의 TURBO/ 폴더가 그대로 따라와서
       models/unet/TURBO/... 로 들어갔고 ComfyUI 가 거부했다 (2026-08-05).
    """
    jobs = [
        (MODEL_REPO, MODEL_FILE, _model_folder()),
        (TE_REPO, TE_FILE, "text_encoders"),
        (VAE_REPO, VAE_FILE, "vae"),
    ]
    for spec in LORA_SPECS:
        jobs.append((spec["repo"], spec["file"], "loras"))
    return jobs


def _download_one(repo_id, filename, comfy_folder):
    """받아서 /comfyui/models/<폴더>/<파일명> 에 놓는다. 저장소 구조는 평평하게 만든다."""
    t0 = time.time()
    dest_dir = os.path.join(MODELS_DIR, comfy_folder)
    os.makedirs(dest_dir, exist_ok=True)

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest_dir,
        token=HF_TOKEN,
    )

    want = os.path.join(dest_dir, os.path.basename(filename))
    if os.path.abspath(path) != os.path.abspath(want):
        os.replace(path, want)
        log(f"[download] 위치 정리: {path} → {want}")
        path = want

    size = os.path.getsize(path)
    sec = time.time() - t0
    speed = (size / 1024**2) / sec if sec > 0 else 0
    log(f"[download] {repo_id}/{filename}  →  {comfy_folder}/{os.path.basename(filename)}  "
        f"{size / 1024**3:.2f} GiB  {sec:.1f}초  ({speed:.0f} MB/s)")
    return size


def strip_lora_keys(path):
    """로라에서 LORA_SKIP_KEYS 에 걸리는 텐서를 빼고 덮어쓴다. (v8)

    ⚠️ 실패하면 원본을 그대로 둔다. 검열이 남더라도 워커가 죽는 것보다 낫다.
    """
    if not LORA_SKIP_KEYS:
        return
    try:
        from safetensors.torch import load_file, save_file

        tensors = load_file(path)
        before = len(tensors)

        kept, dropped = {}, []
        for name, t in tensors.items():
            if any(k in name.lower() for k in LORA_SKIP_KEYS):
                dropped.append(name)
            else:
                kept[name] = t

        if not dropped:
            log(f"[lora] ⚠️ SKIP_KEYS={LORA_SKIP_KEYS} 에 걸린 텐서가 없다. "
                f"이름을 잘못 적었을 수 있다. 원본 그대로 쓴다 ({before}개)")
            return
        if not kept:
            log(f"[lora] ⚠️ SKIP_KEYS 가 전부를 걸러냈다({before}개). 원본 그대로 쓴다. "
                f"'blocks' 대신 'blocks.' 처럼 점을 붙여볼 것")
            return

        save_file(kept, path)
        log(f"[lora] {os.path.basename(path)} — {before}개 → {len(kept)}개 "
            f"({len(dropped)}개 제외, keys={LORA_SKIP_KEYS})")
        log(f"[lora] 제외한 것 예시: {dropped[:3]}")
    except Exception as e:
        log(f"[lora] ⚠️ 층 제외 실패, 원본 그대로 쓴다: {type(e).__name__}: {e}")


def _preload_names():
    raw = LORA_PRELOAD
    if not raw or raw.lower() in ("none", "off", "-"):
        return []
    if raw.strip() == "*":
        return [c["name"] for c in LORA_CATALOG]
    return [x.strip() for x in raw.split(",") if x.strip()]


def find_lora(name):
    """카탈로그에서 이름으로 찾는다. 부분 일치도 허용한다."""
    key = (name or "").strip().lower()
    if not key:
        return None
    for c in LORA_CATALOG:
        if c["name"].lower() == key:
            return c
    for c in LORA_CATALOG:
        if key in c["name"].lower() or key in c["file"].lower():
            return c
    return None


def ensure_lora(repo, path, tag=""):
    """로라가 없으면 그때 받는다. 이미 있으면 아무것도 안 한다.

    반환값은 (파일명, 오류메시지).
    ⚠️ 조용히 넘어가지 않는다. 로라는 안 붙어도 ComfyUI 가 오류를 안 내기 때문에,
       여기서 막지 않으면 "붙은 줄 알았는데 안 붙은" 결과가 나온다.
    """
    fname = os.path.basename(path)
    if fname in _lora_ready:
        return fname, None
    dest = os.path.join(MODELS_DIR, "loras", fname)
    if os.path.exists(dest):
        _lora_ready.add(fname)
        return fname, None
    try:
        _download_one(repo, path, "loras")
        strip_lora_keys(dest)
        _lora_ready.add(fname)
        return fname, None
    except Exception as e:
        return None, (f"로라 '{tag or fname}' 를 못 받았다: {type(e).__name__}: {e}. "
                      f"저장소와 파일경로가 맞는지, 게이팅 저장소면 HF_TOKEN 에 "
                      f"약관 동의가 되어 있는지 확인할 것")


def download_models():
    jobs = _plan_downloads()
    _names = ", ".join(os.path.basename(s["file"]) for s in LORA_SPECS) or "없음"
    log(f"[download] {len(jobs)}개 파일 받기 시작 "
        f"(backend={MODEL_BACKEND}, model={MODEL_FILE}, lora={_names}, steps={DEFAULT_STEPS})")

    t0 = time.time()
    total = 0
    # ⚠️ 순차로 받으면 한 연결 속도에 묶인다. 동시에 받아야 초당 800MB 가 나온다.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_download_one, r, f, d) for (r, f, d) in jobs]
        for fut in futures:
            total += fut.result()

    # v8: 받아둔 로라에서 지정한 층을 뺀다. SKIP_KEYS 가 비어 있으면 아무것도 안 한다.
    for spec in LORA_SPECS:
        fname = os.path.basename(spec["file"])
        strip_lora_keys(os.path.join(MODELS_DIR, "loras", fname))
        _lora_ready.add(fname)

    # v9: 카탈로그에서 미리 받을 것
    names = _preload_names()
    if names:
        log(f"[download] 카탈로그 미리 받기 {len(names)}개: {', '.join(names)}")
        for n in names:
            spec = find_lora(n)
            if spec is None:
                log(f"[download] ⚠️ 카탈로그에 없다: {n}")
                continue
            _, err = ensure_lora(spec["repo"], spec["path"], spec["name"])
            if err:
                # ⚠️ 미리 받기 실패로 워커를 죽이지 않는다. 그 로라만 못 쓰게 둔다.
                log(f"[download] ⚠️ 미리 받기 실패({n}): {err}")

    sec = time.time() - t0
    log(f"[download] 전부 완료: {total / 1024**3:.2f} GiB / {sec:.1f}초 "
        f"({(total / 1024**2) / max(sec, 0.001):.0f} MB/s 평균)")
    _boot_stats["download_sec"] = round(sec, 1)
    _boot_stats["download_gib"] = round(total / 1024**3, 2)


# ──────────────────────────────────────────────────────────────────────
# 2) ComfyUI 띄우기
# ──────────────────────────────────────────────────────────────────────

def start_comfyui():
    cmd = [
        sys.executable, "-u", "/comfyui/main.py",
        "--listen", COMFY_HOST,
        "--port", str(COMFY_PORT),
        "--disable-auto-launch",
        "--disable-metadata",
        "--disable-api-nodes",
        "--log-stdout",
    ]
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
                    log(f"[comfyui] system_stats: {json.dumps(r.json(), ensure_ascii=False)}")
                except Exception:
                    pass
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"ComfyUI 가 {timeout}초 안에 안 떴다.")


# ──────────────────────────────────────────────────────────────────────
# VRAM 감시
# ──────────────────────────────────────────────────────────────────────

def _vram_snapshot():
    try:
        d = requests.get(f"{COMFY_URL}/system_stats", timeout=5).json()
        dev = (d.get("devices") or [{}])[0]
        return {
            "vram_total": dev.get("vram_total"),
            "vram_free": dev.get("vram_free"),
        }
    except Exception:
        return None


class VramWatcher(threading.Thread):
    """생성하는 동안 2초마다 남은 VRAM 을 본다. 정확한 최대치가 아니라 근사값이다."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.min_free = None
        self.total = None
        self.samples = 0

    def run(self):
        while not self.stop_flag.is_set():
            s = _vram_snapshot()
            if s and s.get("vram_free") is not None:
                self.samples += 1
                self.total = s.get("vram_total")
                if self.min_free is None or s["vram_free"] < self.min_free:
                    self.min_free = s["vram_free"]
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
# 3) 워크플로 만들기
# ──────────────────────────────────────────────────────────────────────

def build_workflow(prompt, negative, width, height, steps, cfg, seed,
                   loras, start_step, sampler, scheduler, backend, num_images=1):
    """ComfyUI API 형식 워크플로를 만든다.

    노드 구성은 ComfyUI 공식 템플릿(image_krea2_turbo_t2i.json)과 같다:
      UNETLoader → (LoraLoaderModelOnly …) → KSampler → VAEDecode → SaveImage
      CLIPLoader(type=krea2) → CLIPTextEncode → (ConditioningZeroOut)
      VAELoader / EmptyLatentImage

    v7 두 가지
      1) 로라를 여러 개 사슬처럼 잇는다 (노드 40, 41, 42 …)
      2) start_step 이 0 보다 크면 KSampler 를 둘로 나눈다.
         앞구간은 로라 없이 돌려 구도를 잡고, 뒷구간만 로라를 태운다.
    """
    if backend == "gguf":
        loader = {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": os.path.basename(MODEL_FILE)},
        }
    else:
        loader = {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": os.path.basename(MODEL_FILE),
                # ⚠️ "default" 여야 한다. fp8_scaled 는 배율값을 자기가 들고 있다.
                #    fp8_e4m3fn 을 고르면 이중으로 깎인다.
                "weight_dtype": "default",
            },
        }

    wf = {
        "1": loader,
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": os.path.basename(TE_FILE),
                # ⭐ 반드시 krea2. 이게 있어야 12층 추출이 발동한다.
                #    잘못되면 조건값 크기가 30720 이 아니라 2560 으로 나온다.
                "type": "krea2",
                "device": "default",
            },
        },
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": os.path.basename(VAE_FILE)}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": prompt}},
        # ⭐ v9: batch_size 가 곧 num_images 다. 공식 sample() 의 num_images 와 같은 것으로,
        #    한 프롬프트로 여러 장을 한 번에 뽑는다. 시안을 여러 개 뽑아 고를 때 쓴다.
        #    ⚠️ 공식은 시드가 seed, seed+1 … 로 올라가지만 ComfyUI 배치는 한 시드에서
        #       배치 차원으로 뽑는다. 결과가 같지 않을 수 있다. 실측으로 확인해야 한다.
        "7": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": num_images}},
        "10": {"class_type": "SaveImage",
               "inputs": {"images": ["9", 0], "filename_prefix": "krea2"}},
    }

    # ⚠️ CFG 가 1.0 이면 네거티브가 계산에 안 들어간다. 빈 조건을 넣는 게 정석이다.
    if cfg <= 1.0 or not negative:
        wf["6"] = {"class_type": "ConditioningZeroOut",
                   "inputs": {"conditioning": ["5", 0]}}
    else:
        wf["6"] = {"class_type": "CLIPTextEncode",
                   "inputs": {"clip": ["2", 0], "text": negative}}

    # ── 로라 사슬 ────────────────────────────────────────────────
    model_ref = ["1", 0]
    for i, spec in enumerate(loras):
        nid = str(40 + i)
        wf[nid] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": os.path.basename(spec["file"]),
                "strength_model": spec["strength"],
            },
        }
        model_ref = [nid, 0]

    # ── 샘플링 ───────────────────────────────────────────────────
    if loras and 0 < start_step < steps:
        # 앞구간: 로라를 안 거친 원본("1")으로. 여기서 구도가 정해진다.
        wf["8"] = {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0],
                "latent_image": ["7", 0], "add_noise": "enable", "noise_seed": seed,
                "steps": steps, "cfg": cfg, "sampler_name": sampler,
                "scheduler": scheduler, "start_at_step": 0,
                "end_at_step": start_step, "return_with_leftover_noise": "enable",
            },
        }
        # 뒷구간: 로라 사슬을 태운다.
        # ⚠️ add_noise=disable 이어야 앞구간 결과를 이어받는다. enable 이면 그림이 망가진다.
        wf["81"] = {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": model_ref, "positive": ["5", 0], "negative": ["6", 0],
                "latent_image": ["8", 0], "add_noise": "disable", "noise_seed": seed,
                "steps": steps, "cfg": cfg, "sampler_name": sampler,
                "scheduler": scheduler, "start_at_step": start_step,
                "end_at_step": steps, "return_with_leftover_noise": "disable",
            },
        }
        latent_out = ["81", 0]
    else:
        wf["8"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": model_ref, "positive": ["5", 0], "negative": ["6", 0],
                "latent_image": ["7", 0], "seed": seed, "steps": steps, "cfg": cfg,
                "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0,
            },
        }
        latent_out = ["8", 0]

    wf["9"] = {"class_type": "VAEDecode",
               "inputs": {"samples": latent_out, "vae": ["3", 0]}}
    return wf


# ──────────────────────────────────────────────────────────────────────
# 4) ComfyUI 에 던지고 결과 받기
# ──────────────────────────────────────────────────────────────────────

def queue_workflow(wf):
    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": wf}, timeout=60)
    if r.status_code != 200:
        # ⚠️ ComfyUI 는 거부 이유를 본문에 적어 보낸다. 그대로 올려야 원인을 안다.
        raise RuntimeError(f"ComfyUI 가 워크플로를 거부했다 (HTTP {r.status_code}): {r.text[:2000]}")
    return r.json()["prompt_id"]


def wait_for_result(prompt_id, proc, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"생성 도중 ComfyUI 가 죽었다 (exit code {proc.returncode}). "
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
                    f"ComfyUI 실행 오류: {json.dumps(status, ensure_ascii=False)[:2000]}")
            if status.get("completed") or entry.get("outputs"):
                return entry
        time.sleep(1)
    raise RuntimeError(f"{timeout}초 안에 생성이 안 끝났다.")


def fetch_images(entry):
    """결과 이미지를 전부 받는다. (v9 — 여러 장 뽑기 때문에 전부로 바꿨다)

    반환: [(바이트, 파일명), ...]
    """
    out = []
    for node_out in (entry.get("outputs") or {}).values():
        for img in node_out.get("images") or []:
            q = urllib.parse.urlencode({
                "filename": img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output"),
            })
            r = requests.get(f"{COMFY_URL}/view?{q}", timeout=120)
            r.raise_for_status()
            out.append((r.content, img["filename"]))
    if not out:
        raise RuntimeError("결과에 이미지가 없다.")
    return out


# ──────────────────────────────────────────────────────────────────────
# 5) 요청 처리
# ──────────────────────────────────────────────────────────────────────

def save_media(job):
    """요청에 실려온 사진을 ComfyUI 가 읽는 폴더에 파일로 놓는다. (v9)

    ⭐ 이 함수가 "참조 이미지로 만들기"의 전제조건이다.
       크레아2 순정 기능 중 스타일 전이는 사진을 넣어야 쓸 수 있는데,
       사진을 넣을 통로가 없으면 배선을 아무리 잘 그려도 못 쓴다.
       영상 워커 v3 의 같은 함수를 옮긴 것이다.

    요청 형식:
      "media": [{"name": "ref.png", "data": "(base64)"}, ...]

    워커와 ComfyUI 가 같은 컨테이너 안에 있으므로 파일로 직접 놓으면 된다.
    배선(workflow)에서는 LoadImage 에 파일 이름만 적으면 그대로 읽힌다.

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
    """요청의 로라 지정을 실제로 걸 목록으로 바꾼다. 반환값은 (목록, 오류메시지).

    길이 네 개다.

      ① stock: true          아무것도 안 건다. 순정 기준점 (v9 신규)
      ② loras 를 안 보냄      부팅 때 받아둔 LORA_SPECS 를 전부 건다 (v8 과 동일)
                             lora_strength 를 보내면 그 강도를 전부에 적용
      ③ {"name": "실사"}      카탈로그에서 이름으로 찾는다. 없으면 그때 받는다 (v9 신규)
      ④ {"repo": "...", "file": "..."}
                             카탈로그에 없어도 주소로 직접 받는다 (v9 신규)
                             ⚠️ 로라가 1,895개라 카탈로그에 다 못 넣기 때문에 낸 길이다
      ⑤ {"file": "Radiance"}  받아둔 것 중에서 파일명 일부로 찾는다 (v8 과 동일)

    ⚠️ 못 찾거나 못 받으면 조용히 넘어가지 않고 오류로 돌려준다.
       로라는 안 붙어도 ComfyUI 가 오류를 안 낸다 (로라 문서 2절 ①).
    """
    if job.get("stock"):
        return [], None

    req = job.get("loras")

    if req is None:
        if not LORA_SPECS:
            return [], None
        s = job.get("lora_strength")
        if s is None:
            return [dict(x) for x in LORA_SPECS], None
        return [dict(x, strength=float(s)) for x in LORA_SPECS], None

    if not isinstance(req, list):
        return None, "loras 는 목록이어야 한다."

    out = []
    for item in req:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            return None, f"loras 항목이 잘못됐다: {item!r}"

        name = str(item.get("name") or "").strip()
        repo = str(item.get("repo") or "").strip()
        path = str(item.get("file") or item.get("path") or "").strip()
        strength = item.get("strength")

        # ④ 주소를 직접 적은 경우 — 카탈로그 밖 로라
        if repo and path:
            fname, err = ensure_lora(repo, path, f"{repo}/{path}")
            if err:
                return None, err
            out.append({"repo": repo, "file": fname,
                        "strength": float(strength) if strength is not None else 1.0})
            continue

        # ③ 카탈로그에서 이름으로
        spec = find_lora(name or path)
        if spec is not None:
            fname, err = ensure_lora(spec["repo"], spec["path"], spec["name"])
            if err:
                return None, err
            out.append({"repo": spec["repo"], "file": fname,
                        "strength": float(strength) if strength is not None
                        else spec["default_strength"]})
            continue

        # ⑤ 받아둔 것 중에서 파일명 일부로 (v8 방식)
        key = (path or name).lower()
        matched = None
        for s in LORA_SPECS:
            if key and key in os.path.basename(s["file"]).lower():
                matched = s
                break
        if matched is None:
            have_cat = ", ".join(c["name"] for c in LORA_CATALOG) or "(카탈로그 비어 있음)"
            have_pre = ", ".join(os.path.basename(s["file"]) for s in LORA_SPECS) or "없음"
            return None, (f"로라를 못 찾았다: '{name or path}'. "
                          f"카탈로그: {have_cat} / 받아둔 것: {have_pre}. "
                          f"카탈로그에 없으면 repo 와 file 을 직접 적어라")
        d = dict(matched)
        if strength is not None:
            d["strength"] = float(strength)
        out.append(d)

    return out, None


def capabilities():
    """UI 가 화면을 스스로 그릴 수 있게 스위치 목록을 돌려준다. (v9 신규)

    ⚠️ 로라 카탈로그에 없는 것도 쓸 수 있다. 그건 목록으로 안 돌려준다 —
       1,895개를 응답에 담을 수 없기 때문이다. lora_by_address 를 볼 것.
    """
    return {
        "model": {
            "repo": MODEL_REPO, "file": MODEL_FILE, "backend": MODEL_BACKEND,
            "text_encoder": f"{TE_REPO}/{TE_FILE}", "vae": f"{VAE_REPO}/{VAE_FILE}",
        },
        "features": [
            {"id": "width", "label": "가로", "type": "int", "default": DEFAULT_WIDTH},
            {"id": "height", "label": "세로", "type": "int", "default": DEFAULT_HEIGHT},
            {"id": "steps", "label": "스텝", "type": "int", "default": DEFAULT_STEPS,
             "min": 1, "max": 60},
            {"id": "cfg", "label": "따름 강도", "type": "float", "default": DEFAULT_CFG,
             "min": 1.0, "max": 10.0},
            {"id": "sampler", "label": "샘플러", "type": "text", "default": DEFAULT_SAMPLER},
            {"id": "scheduler", "label": "스케줄러", "type": "text", "default": DEFAULT_SCHEDULER},
            {"id": "seed", "label": "시드", "type": "int", "default": DEFAULT_SEED},
            {"id": "num_images", "label": "한 번에 뽑을 장수", "type": "int",
             "default": 1, "min": 1, "max": 8,
             "note": "시안을 여러 장 뽑아 고를 때. ⚠️ 장수를 늘리면 응답이 커진다"},
            {"id": "stock", "label": "순정으로 뽑기", "type": "switch", "default": False,
             "note": "로라를 전부 끈다. 로라가 실제로 붙었는지 대조하는 기준점"},
            {"id": "lora_start_step", "label": "로라를 몇 스텝부터 걸까",
             "type": "int", "default": DEFAULT_LORA_START_STEP,
             "note": "0 이면 처음부터. 올리면 앞구간은 로라 없이 돌아 구도가 살아난다"},
        ],
        "loras": [
            {"name": c["name"], "label": c["note"] or c["name"],
             "repo": c["repo"], "file": c["file"],
             "default_strength": c["default_strength"],
             "ready": c["file"] in _lora_ready}
            for c in LORA_CATALOG
        ],
        "lora_by_address": {
            "how": "카탈로그에 없는 로라는 저장소와 파일경로를 직접 적으면 그때 받아 쓴다",
            "example": {"loras": [{"repo": "RudySen/Krea2-realism-V2",
                                   "file": "krea2-realism-v2.safetensors",
                                   "strength": 0.8}]},
            "why": "크레아2 로라가 1,895개라 카탈로그에 다 넣을 수 없다(약 199KB)",
        },
        "media": {
            "how": "media 에 [{name, data(base64)}] 를 실으면 사진을 컨테이너에 파일로 놓는다",
            "then": "workflow 의 LoadImage 에 그 파일 이름을 적으면 읽힌다",
            "dir": COMFY_INPUT_DIR,
            "why": "크레아2 순정 기능 중 '참조 이미지로 만들기(스타일 전이)'의 전제조건",
        },
        "workflow_passthrough": {
            "how": "workflow 에 ComfyUI API 형식 JSON 을 넣으면 그대로 실행한다",
            "why": "코드를 안 고치고 새 배선을 시험하기 위한 통로",
            "style_reference": {
                "상태": "아직 안 해봤다",
                "필요한 로라": "Comfy-Org/Krea-2 loras/krea2_style_reference.safetensors",
                "노드": ["LoadImage", "TextEncodeQwenImageEditPlus",
                       "FluxKontextMultiReferenceLatentMethod", "ModelSamplingFlux"],
                "⚠️": "공식 템플릿은 int8 본체 기준이다. 우리 GGUF 조합은 미확인",
            },
        },
        "native_features": {
            "글로 만들기": "된다",
            "참조 이미지로 만들기(스타일 전이)": "media + workflow 로 시험 가능. 미확인",
            "여러 장 뽑기": "된다 (num_images)",
            "mu 1.15": "ComfyUI 가 자동 적용 (supported_models.py shift=1.15)",
            "무드보드": "불가 — 크레아 웹앱에서만 만든다",
            "슬라이더(intensity/complexity/movement)": "불가 — 로라가 공개 안 됨",
            "creativity": "워커 밖 — 프롬프트를 늘리는 단계에서 다룬다",
        },
        "lora_skip_keys": LORA_SKIP_KEYS or None,
        "preloaded": sorted(_lora_ready),
    }


def generate(job):
    """요청 하나를 처리한다. 이 함수가 몸통의 입구다."""
    if str(job.get("action", "")).lower() == "capabilities":
        return capabilities()

    # ⭐ v9: 사진을 먼저 컨테이너 안에 놓는다.
    #    workflow 를 통째로 던질 때도 필요하므로 분기 앞에서 처리한다.
    saved, err = save_media(job)
    if err:
        return {"error": err}

    # ⭐ v9: 워크플로 통째로 던지기
    raw_wf = job.get("workflow")
    wf_loras = []
    if raw_wf is not None:
        if not isinstance(raw_wf, dict):
            return {"error": "workflow 는 ComfyUI API 형식 JSON 이어야 한다."}
        # ⭐ v10: 배선을 통째로 던질 때도 로라를 미리 받아둔다.
        #    ⚠️ 배선은 손대지 않는다. 파일을 컨테이너에 내려놓기만 하고,
        #       어느 노드에 어떤 강도로 걸지는 배선의 lora_name 이 정한다.
        #    ⚠️ v9 은 이 분기가 resolve_loras 를 건너뛰었다. 그래서 배선에 로라를 적어도
        #       파일이 없어 ComfyUI 가 실패했다. 스타일 전이가 정확히 여기서 막혔다.
        #    ⚠️ loras 를 안 적으면 아무것도 받지 않는다. 여기서 resolve_loras 를 무조건
        #       부르면 요청에 없어도 템플릿 로라(체형)를 받아버린다. 그건 배선과 무관하다.
        if job.get("loras") is not None:
            wf_loras, err = resolve_loras(job)
            if err:
                return {"error": err}
        wf = raw_wf
        p = None
    else:
        prompt = job.get("prompt")
        if not prompt or not str(prompt).strip():
            return {"error": "prompt 가 비어 있다."}

        loras, err = resolve_loras(job)
        if err:
            return {"error": err}

        p = {
            "prompt": str(prompt),
            "negative": str(job.get("negative_prompt") or ""),
            "width": int(job.get("width", DEFAULT_WIDTH)),
            "height": int(job.get("height", DEFAULT_HEIGHT)),
            "steps": int(job.get("steps", DEFAULT_STEPS)),
            "cfg": float(job.get("cfg", DEFAULT_CFG)),
            "sampler": str(job.get("sampler", DEFAULT_SAMPLER)),
            "scheduler": str(job.get("scheduler", DEFAULT_SCHEDULER)),
            "start_step": int(job.get("lora_start_step", DEFAULT_LORA_START_STEP)),
            "stock": bool(job.get("stock")),
            "loras": loras,
            # ⭐ v9: 한 번에 여러 장. 공식 sample() 의 num_images 와 같은 것
            "num_images": max(1, min(int(job.get("num_images", 1)), 8)),
        }
        seed = job.get("seed", DEFAULT_SEED)
        p["seed"] = (random.randint(0, 2**31 - 1)
                     if seed in (None, -1, "random") else int(seed))

        wf = build_workflow(p["prompt"], p["negative"], p["width"], p["height"],
                            p["steps"], p["cfg"], p["seed"], p["loras"],
                            p["start_step"], p["sampler"], p["scheduler"],
                            MODEL_BACKEND, p["num_images"])

    watcher = VramWatcher()
    watcher.start()
    t0 = time.time()
    try:
        prompt_id = queue_workflow(wf)
        if p is not None:
            desc = ", ".join(f"{os.path.basename(s['file'])}@{s['strength']}"
                             for s in p["loras"]) or "없음"
            log(f"[job] 큐 등록 {prompt_id} — {p['width']}x{p['height']} {p['steps']}스텝 "
                f"cfg{p['cfg']} seed{p['seed']} backend={MODEL_BACKEND} "
                f"lora=[{desc}]{' 순정' if p['stock'] else ''} "
                f"start_step={p['start_step']} sampler={p['sampler']} "
                f"장수={p['num_images']}{f' 사진={saved}' if saved else ''}")
        else:
            wf_desc = ", ".join(os.path.basename(s["file"]) for s in wf_loras) or "없음"
            log(f"[job] 큐 등록 {prompt_id} — workflow 통째로 받음 (노드 {len(wf)}개)"
                f" 미리받은로라=[{wf_desc}]"
                f"{f' 사진={saved}' if saved else ''}")
        entry = wait_for_result(prompt_id, COMFY_PROC)
        images = fetch_images(entry)
        payload, filename = images[0]
    except Exception as e:
        # ⚠️ 응답만 돌려보내면 로그에 흔적이 안 남는다. 나중에 추적이 불가능해진다.
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
    result = {
        # 첫 장. 한 장만 뽑았으면 v8 과 형식이 같다
        "image_base64": base64.b64encode(payload).decode("utf-8"),
        "format": "png",
        "bytes": len(payload),
        "generation_sec": round(elapsed, 1),
        "filename": filename,
        "media_saved": saved or None,
        "backend": MODEL_BACKEND,
        "model": f"{MODEL_REPO}/{MODEL_FILE}",
        "text_encoder": f"{TE_REPO}/{TE_FILE}",
        "lora_skip_keys": LORA_SKIP_KEYS or None,
    }
    # ⭐ v9: 여러 장을 뽑았으면 나머지도 담는다.
    #    ⚠️ 런포드 응답 크기 제한이 있다(/run 10MB, /runsync 20MB, base64 는 1.33배).
    #       장수를 늘리면 걸릴 수 있다. 그래서 개수와 총 크기를 응답에 같이 담아
    #       무엇 때문에 큰지 알 수 있게 한다.
    if len(images) > 1:
        result["images_base64"] = [
            base64.b64encode(b).decode("utf-8") for b, _ in images]
        result["filenames"] = [n for _, n in images]
        result["images_count"] = len(images)
        result["total_bytes"] = sum(len(b) for b, _ in images)

    if p is not None:
        result.update({
            "width": p["width"], "height": p["height"], "seed": p["seed"],
            "steps": p["steps"], "cfg": p["cfg"], "num_images": p["num_images"],
            "sampler": p["sampler"], "scheduler": p["scheduler"],
            "stock": p["stock"],
            # ⭐ 무엇이 실제로 걸렸는지. 로라는 조용히 무시되므로 이게 유일한 확인 수단이다
            "loras_applied": [
                {"file": os.path.basename(s["file"]), "strength": s["strength"]}
                for s in p["loras"]
            ] or None,
            "lora_start_step": p["start_step"] if p["loras"] else None,
            "lora_strength": p["loras"][0]["strength"] if p["loras"] else None,
            "lora": (f"{p['loras'][0].get('repo','')}/{p['loras'][0]['file']}"
                     if p["loras"] else None),
            "negative_prompt_used": bool(p["negative"]) and p["cfg"] > 1.0,
        })
    else:
        result["workflow_passthrough"] = True
        # ⭐ v10: 배선 모드에서 어떤 로라 파일을 미리 받아뒀는지.
        #    ⚠️ "받아놨다"는 뜻이지 "걸렸다"는 뜻이 아니다. 실제로 걸렸는지는 배선이 정한다.
        result["loras_downloaded"] = [
            {"file": os.path.basename(s["file"]), "repo": s.get("repo", "")}
            for s in wf_loras
        ] or None

    if _boot_stats:
        result["boot"] = dict(_boot_stats)
        _boot_stats.clear()
    result.update(watcher.result())

    log(f"[job] 완료 {elapsed:.1f}초 / {len(payload) / 1024**2:.1f} MB "
        f"/ VRAM 최대사용 {result.get('vram_peak_used_gb')} GB")
    return result


# ──────────────────────────────────────────────────────────────────────
# 6) 기동
# ──────────────────────────────────────────────────────────────────────

def boot():
    """워커가 뜰 때 한 번 돈다. 껍데기가 부른다."""
    global COMFY_PROC
    if not HF_TOKEN:
        log("[init] ⚠️ HF_TOKEN 이 없다. 비게이트 저장소만 받을 수 있다.")
    if not LORA_CATALOG:
        log("[init] LORA_CATALOG 가 비어 있다. 카탈로그 없이도 "
            "요청에 repo/file 을 적으면 로라를 쓸 수 있다.")
    else:
        log(f"[init] 로라 카탈로그 {len(LORA_CATALOG)}개: "
            f"{', '.join(c['name'] for c in LORA_CATALOG)}")

    download_models()
    COMFY_PROC = start_comfyui()
    wait_for_comfyui(COMFY_PROC)

    _boot_stats["total_cold_start_sec"] = round(time.time() - T_PROCESS_START, 1)
    log(f"[init] 콜드스타트 총 {_boot_stats['total_cold_start_sec']}초 "
        f"(다운로드 {_boot_stats.get('download_sec')}초 + "
        f"ComfyUI 기동 {_boot_stats.get('comfyui_boot_sec')}초)")
