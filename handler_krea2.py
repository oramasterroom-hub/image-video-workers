"""Krea 2 Raw + Turbo LoRA — 런포드 서버리스 handler (ComfyUI 방식)

이 파일 하나가 전부 한다:
  1) 워커가 뜰 때 허깅페이스에서 모델을 직접 받아 ComfyUI models/ 에 놓는다
  2) ComfyUI 를 자식 프로세스로 띄운다
  3) 요청이 오면 워크플로 JSON 을 만들어 ComfyUI 에 던지고 결과 이미지를 돌려준다

⚠️ 입출력 형식을 Qwen 워커(handler_qwen.py)와 일부러 맞췄다.
   같은 프롬프트로 나란히 뽑아 비교하는 게 이 워커의 목적이기 때문이다.
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
import runpod
from huggingface_hub import hf_hub_download

# ──────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────

COMFY_HOST = "127.0.0.1"
COMFY_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
MODELS_DIR = "/comfyui/models"

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

# ──────────────────────────────────────────────────────────────────────
# ⭐ 부품을 전부 환경변수로 뺐다 (2026-08-05)
#
# 이전 판은 파일 이름이 코드에 박혀 있어서, 다른 모델을 시험하려면 매번
# 재빌드를 해야 했다. 이제 템플릿 환경변수만 바꾸면 어떤 조합이든 돌아간다.
#
# MODEL_BACKEND 가 하는 일은 "어느 ComfyUI 로더를 쓰고 어느 폴더에 놓느냐" 뿐이다.
#   fp8   → UNETLoader        / models/diffusion_models/   ComfyUI 기본 노드
#   gguf  → UnetLoaderGGUF    / models/unet/               molbal 커스텀 노드 필요
#
# LORA_FILE 이 비어 있으면 LoRA 를 아예 안 얹는다.
#   ⚠️ Turbo 체크포인트에는 증류가 이미 들어 있으므로 LoRA 를 얹으면 안 된다.
#
# 설정 예시
#   Raw + Turbo LoRA (기존)
#     MODEL_FILE=diffusion_models/krea2_raw_fp8_scaled.safetensors
#     LORA_FILE=loras/krea2_turbo_lora_rank_64_bf16.safetensors  LORA_STRENGTH=0.6  STEPS=12
#   Turbo 단독 Q8
#     MODEL_REPO=realrebelai/KREA-2_GGUFs
#     MODEL_FILE=TURBO/Krea-2-Turbo-Q8_0.gguf   MODEL_BACKEND=gguf
#     LORA_FILE=(비움)                          STEPS=8
# ──────────────────────────────────────────────────────────────────────
COMFY_REPO = "Comfy-Org/Krea-2"

MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "fp8").strip().lower()
MODEL_REPO = os.environ.get("MODEL_REPO", COMFY_REPO).strip()
MODEL_FILE = os.environ.get(
    "MODEL_FILE", "diffusion_models/krea2_raw_fp8_scaled.safetensors").strip()

TE_REPO = os.environ.get("TE_REPO", COMFY_REPO).strip()
TE_FILE = os.environ.get(
    "TE_FILE", "text_encoders/qwen3vl_4b_bf16.safetensors").strip()

VAE_REPO = os.environ.get("VAE_REPO", COMFY_REPO).strip()
VAE_FILE = os.environ.get("VAE_FILE", "vae/qwen_image_vae.safetensors").strip()

# 비어 있으면 LoRA 를 안 쓴다
LORA_REPO = os.environ.get("LORA_REPO", COMFY_REPO).strip()
LORA_FILE = os.environ.get(
    "LORA_FILE", "loras/krea2_turbo_lora_rank_64_bf16.safetensors").strip()
USE_LORA = bool(LORA_FILE)

# ⭐ 보스가 정한 값 (2026-08-04). 원출처는 Civitai 실사 워크플로 krea2_simple_v1.
#    ⚠️ Turbo 단독으로 쓸 때는 STEPS=8 로 내려야 한다 (Krea 공식 권장).
DEFAULT_LORA_STRENGTH = float(os.environ.get("LORA_STRENGTH", "0.6"))
DEFAULT_STEPS = int(os.environ.get("STEPS", "12"))
# CFG 1.0 = 네거티브가 작동하지 않는다. 1 을 넘기면 생성 시간이 2배가 된다.
DEFAULT_CFG = 1.0
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"
# Qwen 의 9:16 값과 같게 맞췄다. 나란히 비교하기 위해서다.
DEFAULT_WIDTH = 928
DEFAULT_HEIGHT = 1664
DEFAULT_SEED = 42

# 콜드스타트 실측용. 첫 응답에 담아 보낸다.
T_PROCESS_START = time.time()
_boot_stats = {}


def log(msg):
    print(msg, flush=True)


# ──────────────────────────────────────────────────────────────────────
# 1) 모델 받기
# ──────────────────────────────────────────────────────────────────────

def _model_dir():
    """본체를 어느 폴더에 놓을지. ComfyUI 가 로더별로 다른 폴더를 본다.

    gguf  → models/unet/               UnetLoaderGGUF 가 여기를 본다
    fp8   → models/diffusion_models/   UNETLoader 가 여기를 본다
    """
    return f"{MODELS_DIR}/unet" if MODEL_BACKEND == "gguf" else MODELS_DIR


def _plan_downloads():
    """(repo_id, 저장소안 경로, 내려받을 폴더) 목록을 만든다.

    ⭐ Comfy-Org/Krea-2 저장소의 폴더 구조가 ComfyUI models/ 구조와 똑같다.
       그래서 local_dir=/comfyui/models 로 받으면 알아서 제자리에 놓인다.
       ⚠️ 다른 저장소(예: realrebelai GGUF)는 그 구조가 아니라서
          _model_dir() 로 목적지를 따로 정해준다.

    ⚠️ 2026-08-05: 안 쓰는 폴백을 받던 것을 없앴다.
       Raw GGUF 12.76GiB 를 매번 받고 한 번도 안 썼다. 콜드스타트에서 25초를 버렸다.
       이제 부품을 전부 환경변수로 정하므로, 갈아탈 때는 템플릿 값만 바꾸면 된다.
    """
    jobs = [
        (MODEL_REPO, MODEL_FILE, _model_dir()),
        (TE_REPO, TE_FILE, MODELS_DIR),
        (VAE_REPO, VAE_FILE, MODELS_DIR),
    ]
    if USE_LORA:
        jobs.append((LORA_REPO, LORA_FILE, MODELS_DIR))
    return jobs


def _download_one(repo_id, filename, local_dir):
    t0 = time.time()
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        token=HF_TOKEN,
    )
    size = os.path.getsize(path)
    sec = time.time() - t0
    speed = (size / 1024**2) / sec if sec > 0 else 0
    log(f"[download] {repo_id}/{filename}  {size / 1024**3:.2f} GiB  "
        f"{sec:.1f}초  ({speed:.0f} MB/s)")
    return size


def download_models():
    jobs = _plan_downloads()
    log(f"[download] {len(jobs)}개 파일 받기 시작 "
        f"(backend={MODEL_BACKEND}, model={MODEL_FILE}, "
        f"lora={LORA_FILE or '없음'}, steps={DEFAULT_STEPS})")

    t0 = time.time()
    total = 0
    # ⚠️ 순차로 받으면 한 연결의 속도에 묶인다. 동시에 받아야 초당 800MB 가 나온다.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_download_one, r, f, d) for (r, f, d) in jobs]
        for fut in futures:
            total += fut.result()

    sec = time.time() - t0
    log(f"[download] 전부 완료: {total / 1024**3:.2f} GiB / {sec:.1f}초 "
        f"({(total / 1024**2) / sec:.0f} MB/s 평균)")
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
        # 우리가 안 쓰는 것들. 켜둘 이유가 없다.
        "--disable-api-nodes",
        "--log-stdout",
    ]
    log(f"[comfyui] 실행: {' '.join(cmd)}")
    # 자식 프로세스 로그가 그대로 워커 로그로 나오게 둔다. 원인 추적에 필요하다.
    return subprocess.Popen(cmd, cwd="/comfyui")


def wait_for_comfyui(proc, timeout=900):
    """ComfyUI 가 요청을 받을 수 있을 때까지 기다린다."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        # ⚠️ 죽었으면 기다려봐야 소용없다. 바로 알린다.
        if proc.poll() is not None:
            raise RuntimeError(
                f"ComfyUI 가 시작하다 죽었다 (exit code {proc.returncode}). 위 로그를 볼 것."
            )
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
# VRAM 감시 — ⑤ "24GB 에서 되는가" 를 재기 위한 것
# ──────────────────────────────────────────────────────────────────────

def _vram_snapshot():
    try:
        d = requests.get(f"{COMFY_URL}/system_stats", timeout=5).json()
        dev = (d.get("devices") or [{}])[0]
        return {
            "vram_total": dev.get("vram_total"),
            "vram_free": dev.get("vram_free"),
            "torch_vram_total": dev.get("torch_vram_total"),
            "torch_vram_free": dev.get("torch_vram_free"),
        }
    except Exception:
        return None


class VramWatcher(threading.Thread):
    """생성하는 동안 2초마다 남은 VRAM 을 봐서 가장 적었던 값을 기록한다.

    ComfyUI 는 diffusers 처럼 max_memory_allocated() 를 우리가 직접 못 읽는다.
    그래서 밖에서 표본을 뜨는 방식으로 잰다. 정확한 최대치는 아니고 근사값이다.
    """

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
                   lora_strength, sampler, scheduler, backend):
    """ComfyUI API 형식 워크플로를 만든다.

    노드 구성은 ComfyUI 공식 템플릿(image_krea2_turbo_t2i.json)에서 확인한 것과 같다:
      UNETLoader → (LoraLoaderModelOnly) → KSampler → VAEDecode → SaveImage
      CLIPLoader(type=krea2) → CLIPTextEncode → (ConditioningZeroOut)
      VAELoader / EmptyLatentImage

    ⚠️ LoRA 노드는 LORA_FILE 이 있을 때만 끼운다.
       Turbo 체크포인트는 증류가 이미 들어 있어서 LoRA 를 얹으면 안 된다.
    """
    if backend == "gguf":
        # ⚠️ molbal 커스텀 노드가 있어야 동작한다. 이 파일은 models/unet/ 에 있다.
        loader = {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": os.path.basename(MODEL_FILE)},
        }
    else:
        loader = {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": os.path.basename(MODEL_FILE),
                # ⚠️ "default" 여야 한다. fp8_scaled 파일은 배율값(weight_scale)을
                #    자기가 들고 있어서 ComfyUI 가 알아서 처리한다.
                #    여기서 fp8_e4m3fn 을 고르면 이중으로 깎인다.
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
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": os.path.basename(VAE_FILE)},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": prompt},
        },
        "7": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                # LoRA 를 쓰면 "4"(LoRA 노드), 안 쓰면 "1"(본체)에서 바로 받는다
                "model": ["4" if USE_LORA else "1", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": "krea2"},
        },
    }

    # LoRA 는 쓸 때만 노드를 끼운다
    if USE_LORA:
        wf["4"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": os.path.basename(LORA_FILE),
                "strength_model": lora_strength,
            },
        }

    # ⚠️ CFG 가 1.0 이면 네거티브가 계산에 안 들어간다. 빈 조건을 넣는 게 정석이다.
    #    괜히 글을 넣으면 인코딩 시간만 쓰고 결과에는 아무 영향이 없다.
    if cfg <= 1.0 or not negative:
        wf["6"] = {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["5", 0]},
        }
    else:
        wf["6"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": negative},
        }

    return wf


# ──────────────────────────────────────────────────────────────────────
# 4) ComfyUI 에 던지고 결과 받기
# ──────────────────────────────────────────────────────────────────────

def queue_workflow(wf):
    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": wf}, timeout=60)
    if r.status_code != 200:
        # ⚠️ ComfyUI 는 검증 실패 이유를 본문에 적어 보낸다. 그대로 올려야 원인을 안다.
        raise RuntimeError(f"ComfyUI 가 워크플로를 거부했다 (HTTP {r.status_code}): {r.text[:2000]}")
    return r.json()["prompt_id"]


def wait_for_result(prompt_id, proc, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
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
                raise RuntimeError(f"ComfyUI 실행 오류: {json.dumps(status, ensure_ascii=False)[:2000]}")
            if status.get("completed") or entry.get("outputs"):
                return entry
        time.sleep(1)
    raise RuntimeError(f"{timeout}초 안에 생성이 안 끝났다.")


def fetch_image(entry):
    for node_out in (entry.get("outputs") or {}).values():
        for img in node_out.get("images") or []:
            q = urllib.parse.urlencode({
                "filename": img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output"),
            })
            r = requests.get(f"{COMFY_URL}/view?{q}", timeout=120)
            r.raise_for_status()
            return r.content, img["filename"]
    raise RuntimeError("결과에 이미지가 없다.")


# ──────────────────────────────────────────────────────────────────────
# 요청마다
# ──────────────────────────────────────────────────────────────────────

def handler(event):
    job = event.get("input") or {}

    prompt = job.get("prompt")
    if not prompt or not str(prompt).strip():
        return {"error": "prompt 가 비어 있다."}

    negative = job.get("negative_prompt") or ""
    width = int(job.get("width", DEFAULT_WIDTH))
    height = int(job.get("height", DEFAULT_HEIGHT))
    steps = int(job.get("steps", DEFAULT_STEPS))
    cfg = float(job.get("cfg", DEFAULT_CFG))
    lora_strength = float(job.get("lora_strength", DEFAULT_LORA_STRENGTH))
    sampler = str(job.get("sampler", DEFAULT_SAMPLER))
    scheduler = str(job.get("scheduler", DEFAULT_SCHEDULER))
    # ⚠️ backend 는 요청마다 못 바꾼다. 받아둔 파일이 그 하나뿐이기 때문이다.
    #    바꾸려면 템플릿 환경변수(MODEL_BACKEND / MODEL_FILE)를 고쳐야 한다.
    backend = MODEL_BACKEND

    seed = job.get("seed", DEFAULT_SEED)
    seed = random.randint(0, 2**31 - 1) if seed in (None, -1, "random") else int(seed)

    wf = build_workflow(str(prompt), str(negative), width, height, steps, cfg,
                        seed, lora_strength, sampler, scheduler, backend)

    watcher = VramWatcher()
    watcher.start()
    t0 = time.time()
    try:
        prompt_id = queue_workflow(wf)
        log(f"[job] 큐 등록 {prompt_id} — {width}x{height} {steps}스텝 "
            f"cfg{cfg} lora{lora_strength} seed{seed} backend={backend}")
        entry = wait_for_result(prompt_id, COMFY_PROC)
        payload, filename = fetch_image(entry)
    # ⚠️ except 에서 응답만 돌려보내면 로그에 흔적이 안 남는다.
    #    나중에 자동으로 돌릴 때 추적이 불가능해지므로 로그에도 찍는다.
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

    result = {
        "image_base64": base64.b64encode(payload).decode("utf-8"),
        "format": "png",
        "bytes": len(payload),
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "lora_strength": lora_strength if USE_LORA else None,
        "sampler": sampler,
        "scheduler": scheduler,
        "backend": backend,
        "model": f"{MODEL_REPO}/{MODEL_FILE}",
        "text_encoder": f"{TE_REPO}/{TE_FILE}",
        "lora": f"{LORA_REPO}/{LORA_FILE}" if USE_LORA else None,
        "negative_prompt_used": bool(negative) and cfg > 1.0,
        "generation_sec": round(elapsed, 1),
        "filename": filename,
    }
    # ⭐ ③콜드스타트 ⑤메모리 — 이 숫자를 보려고 만든 것이다. 첫 응답에만 담긴다.
    if _boot_stats:
        result["boot"] = dict(_boot_stats)
        _boot_stats.clear()
    result.update(watcher.result())

    log(f"[job] 완료 {elapsed:.1f}초 / {len(payload) / 1024**2:.1f} MB "
        f"/ VRAM 최대사용 {result.get('vram_peak_used_gb')} GB")
    return result


# ──────────────────────────────────────────────────────────────────────
# 기동 (모듈을 불러오는 시점 = 워커가 뜨는 시점에 한 번만 돈다)
# ──────────────────────────────────────────────────────────────────────

if not HF_TOKEN:
    # 우리가 받는 파일은 전부 비게이트 저장소에 있어서 토큰 없이도 받힌다.
    # 그래도 없으면 알려준다. 나중에 게이트 저장소를 쓰게 될 때를 대비해서다.
    log("[init] ⚠️ HF_TOKEN 이 없다. 비게이트 저장소만 받을 수 있다.")

download_models()
COMFY_PROC = start_comfyui()
wait_for_comfyui(COMFY_PROC)

_boot_stats["total_cold_start_sec"] = round(time.time() - T_PROCESS_START, 1)
log(f"[init] 콜드스타트 총 {_boot_stats['total_cold_start_sec']}초 "
    f"(다운로드 {_boot_stats.get('download_sec')}초 + "
    f"ComfyUI 기동 {_boot_stats.get('comfyui_boot_sec')}초)")

runpod.serverless.start({"handler": handler})
