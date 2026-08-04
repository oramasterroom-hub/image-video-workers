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

# 1안(fp8) / 폴백(gguf) 전환. 재빌드 없이 환경변수로 바꾼다.
MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "fp8").strip().lower()
# 텍스트 인코더 bf16 / fp8 전환. 24GB 에 안 들어가면 fp8 로 내린다.
TE_VARIANT = os.environ.get("TE_VARIANT", "bf16").strip().lower()
# 폴백 부품도 같이 받아둘지. 받는 건 공짜라 기본 켬 (보스 지시 2026-08-04)
DOWNLOAD_FALLBACKS = os.environ.get("DOWNLOAD_FALLBACKS", "1") != "0"

COMFY_REPO = "Comfy-Org/Krea-2"
GGUF_REPO = "vantagewithai/Krea-2-Raw-GGUF"

FILE_FP8_MODEL = "diffusion_models/krea2_raw_fp8_scaled.safetensors"
FILE_TE_BF16 = "text_encoders/qwen3vl_4b_bf16.safetensors"
FILE_TE_FP8 = "text_encoders/qwen3vl_4b_fp8_scaled.safetensors"
FILE_VAE = "vae/qwen_image_vae.safetensors"
FILE_LORA = "loras/krea2_turbo_lora_rank_64_bf16.safetensors"
FILE_GGUF_MODEL = "krea2_raw-Q8_0.gguf"

# ⭐ 보스가 정한 값 (2026-08-04). 원출처는 Civitai 실사 워크플로 krea2_simple_v1.
DEFAULT_LORA_STRENGTH = 0.6
DEFAULT_STEPS = 12
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

def _plan_downloads():
    """(repo_id, 저장소안 경로, 내려받을 폴더) 목록을 만든다.

    ⭐ Comfy-Org/Krea-2 저장소의 폴더 구조가 ComfyUI models/ 구조와 똑같다.
       그래서 local_dir=/comfyui/models 로 받으면 알아서 제자리에 놓인다.
    """
    jobs = [
        (COMFY_REPO, FILE_VAE, MODELS_DIR),
        (COMFY_REPO, FILE_LORA, MODELS_DIR),
    ]

    # 본체
    if MODEL_BACKEND == "gguf":
        jobs.append((GGUF_REPO, FILE_GGUF_MODEL, f"{MODELS_DIR}/unet"))
    else:
        jobs.append((COMFY_REPO, FILE_FP8_MODEL, MODELS_DIR))

    # 텍스트 인코더
    jobs.append((COMFY_REPO, FILE_TE_FP8 if TE_VARIANT == "fp8" else FILE_TE_BF16, MODELS_DIR))

    # 폴백도 같이 받아둔다. 갈아탈 때 재빌드도 재다운로드도 안 하기 위해서다.
    if DOWNLOAD_FALLBACKS:
        if MODEL_BACKEND == "gguf":
            jobs.append((COMFY_REPO, FILE_FP8_MODEL, MODELS_DIR))
        else:
            jobs.append((GGUF_REPO, FILE_GGUF_MODEL, f"{MODELS_DIR}/unet"))
        jobs.append((COMFY_REPO, FILE_TE_BF16 if TE_VARIANT == "fp8" else FILE_TE_FP8, MODELS_DIR))

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
        f"(backend={MODEL_BACKEND}, te={TE_VARIANT}, fallbacks={DOWNLOAD_FALLBACKS})")

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
                   lora_strength, sampler, scheduler, backend, te_variant):
    """ComfyUI API 형식 워크플로를 만든다.

    노드 구성은 ComfyUI 공식 템플릿(image_krea2_turbo_t2i.json)에서 확인한 것과 같다:
      UNETLoader → LoraLoaderModelOnly → KSampler → VAEDecode → SaveImage
      CLIPLoader(type=krea2) → CLIPTextEncode → (ConditioningZeroOut)
      VAELoader / EmptyLatentImage
    """
    if backend == "gguf":
        # ⚠️ 폴백 경로. molbal 커스텀 노드가 있어야 동작한다.
        loader = {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": FILE_GGUF_MODEL},
        }
    else:
        loader = {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": os.path.basename(FILE_FP8_MODEL),
                # ⚠️ "default" 여야 한다. 이 파일은 배율값(weight_scale)을 자기가 들고 있어서
                #    ComfyUI 가 알아서 처리한다. 여기서 fp8_e4m3fn 을 고르면 이중으로 깎인다.
                "weight_dtype": "default",
            },
        }

    te_file = FILE_TE_FP8 if te_variant == "fp8" else FILE_TE_BF16

    wf = {
        "1": loader,
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": os.path.basename(te_file),
                # ⭐ 반드시 krea2. 이게 있어야 12층 추출이 발동한다.
                #    잘못되면 조건값 크기가 30720 이 아니라 2560 으로 나온다.
                "type": "krea2",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": os.path.basename(FILE_VAE)},
        },
        "4": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": os.path.basename(FILE_LORA),
                "strength_model": lora_strength,
            },
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
                "model": ["4", 0],
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
    backend = str(job.get("backend", MODEL_BACKEND)).lower()
    te_variant = str(job.get("te_variant", TE_VARIANT)).lower()

    seed = job.get("seed", DEFAULT_SEED)
    seed = random.randint(0, 2**31 - 1) if seed in (None, -1, "random") else int(seed)

    wf = build_workflow(str(prompt), str(negative), width, height, steps, cfg,
                        seed, lora_strength, sampler, scheduler, backend, te_variant)

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
        "lora_strength": lora_strength,
        "sampler": sampler,
        "scheduler": scheduler,
        "backend": backend,
        "model": (f"{GGUF_REPO}/{FILE_GGUF_MODEL}" if backend == "gguf"
                  else f"{COMFY_REPO}/{FILE_FP8_MODEL}"),
        "text_encoder": f"{COMFY_REPO}/{FILE_TE_FP8 if te_variant == 'fp8' else FILE_TE_BF16}",
        "lora": f"{COMFY_REPO}/{FILE_LORA}",
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
