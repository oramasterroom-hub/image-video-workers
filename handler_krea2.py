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

# ⚠️ LoRA 를 끄려면 LORA_FILE 에 none / off / - 중 하나를 넣는다.
#    2026-08-05: 빈 문자열이나 공백은 쓰면 안 된다.
#    런포드가 빈 값을 버려서 handler 의 기본값(Turbo LoRA)이 되살아난다.
#    API 로 조회하면 " " 가 저장돼 보이는데도 컨테이너에는 안 들어온다. 실제로 당했다.
#
# ⭐ v7 (2026-08-06): LoRA 를 여러 개 쓸 수 있다. 쉼표로 나열한다.
#    LORA_FILE     = body.safetensors,face.safetensors
#    LORA_REPO     = repoA,repoB      (부족하면 마지막 값을 나머지에 적용)
#    LORA_STRENGTH = 0.6,0.8          (부족하면 마지막 값을 나머지에 적용)
#    ⚠️ 쉼표가 없으면 v6 과 100% 동일하게 동작한다.
LORA_REPO_RAW = os.environ.get("LORA_REPO", COMFY_REPO).strip()
LORA_FILE_RAW = os.environ.get(
    "LORA_FILE", "loras/krea2_turbo_lora_rank_64_bf16.safetensors").strip()
LORA_STRENGTH_RAW = os.environ.get("LORA_STRENGTH", "0.6").strip()

USE_LORA = LORA_FILE_RAW.lower() not in ("", "none", "off", "-", "0", "false")


def _split_csv(s):
    """쉼표로 나눈다. 빈 조각은 버린다."""
    return [x.strip() for x in s.split(",") if x.strip()]


def _pick(lst, i, fallback):
    """i 번째 값을 준다. 목록이 짧으면 마지막 값을 쓴다.

    저장소나 강도를 하나만 적어도 모든 LoRA 에 적용되게 하기 위한 것이다.
    """
    if not lst:
        return fallback
    return lst[i] if i < len(lst) else lst[-1]


_LORA_FILES = _split_csv(LORA_FILE_RAW) if USE_LORA else []
_LORA_REPOS = _split_csv(LORA_REPO_RAW)
_LORA_STRENGTHS = [float(x) for x in _split_csv(LORA_STRENGTH_RAW)]

# 받아둘 LoRA 목록. 워커가 뜰 때 이 순서대로 받고, 이 순서대로 사슬에 건다.
LORA_SPECS = [
    {
        "repo": _pick(_LORA_REPOS, i, COMFY_REPO),
        "file": f,
        "strength": _pick(_LORA_STRENGTHS, i, 0.6),
    }
    for i, f in enumerate(_LORA_FILES)
]

# ⭐ v8 (2026-08-06): LoRA 에서 특정 층을 빼고 쓴다.
#
#    왜: 체형 LoRA 를 얹으면 특정 부위에 검열 처리(블러·모자이크·착의)가 그려진다.
#        실측으로 LoRA 없으면 안 걸리고, 강도가 셀수록 심해지는 것을 확인했다.
#        그리고 이 LoRA 는 512개 텐서 중 64개가 txtfusion(프롬프트 해석부)을 건드린다.
#        프롬프트·cfg·강도·스텝으로는 통제가 안 됐다 — 해석 통로 자체가 개조되기 때문으로 추정.
#        ⚠️ "txtfusion 이 원인"은 아직 추정이다. blocks 쪽일 수도 있다.
#           그래서 무엇을 뺄지 고를 수 있게 만들었다. 빼고 뽑아보면 그 자리에서 갈린다.
#
#    쓰는 법: 텐서 이름에 들어갈 조각을 쉼표로 나열한다. 대소문자 무시.
#      LORA_SKIP_KEYS=txtfusion       → 64개 제외, 448개(체형) 적용
#      LORA_SKIP_KEYS=blocks.         → 448개 제외, 64개만 적용
#      LORA_SKIP_KEYS=(비움)          → 아무것도 안 뺌 = v7 과 완전히 동일
#
#    ⚠️ 함정: "blocks" 로 쓰면 txtfusion.refiner_blocks / layerwise_blocks 까지 걸려서
#            LoRA 가 통째로 사라진다. 진짜 본체만 빼려면 점을 붙여 "blocks." 로 쓸 것.
LORA_SKIP_KEYS = [
    x.strip().lower()
    for x in os.environ.get("LORA_SKIP_KEYS", "").split(",")
    if x.strip()
]

# ⭐ 보스가 정한 값 (2026-08-04). 원출처는 Civitai 실사 워크플로 krea2_simple_v1.
#    ⚠️ Turbo 단독으로 쓸 때는 STEPS=8 로 내려야 한다 (Krea 공식 권장).
DEFAULT_LORA_STRENGTH = LORA_SPECS[0]["strength"] if LORA_SPECS else 0.6
DEFAULT_STEPS = int(os.environ.get("STEPS", "12"))
# CFG 1.0 = 네거티브가 작동하지 않는다. 1 을 넘기면 생성 시간이 2배가 된다.
DEFAULT_CFG = 1.0
# ⭐ v7: LoRA 를 몇 번째 스텝부터 걸지. 0 이면 처음부터 = v6 과 같다.
#    확산 모델은 구도가 초반 스텝에서 정해지므로, 앞구간을 LoRA 없이 돌리면
#    구도는 원본이 잡고 체형·질감만 LoRA 가 입힌다. 구도 잘림 대책이다.
DEFAULT_LORA_START_STEP = int(os.environ.get("LORA_START_STEP", "0"))
# ⭐ v7: er_sde 로 바꿨다. 체형 LoRA 제작자 권장값이고, 실측에서 화질이 뚜렷이 좋았다
#    (필름 입자감·모공이 살아나고 속도 손해도 없었다. 2026-08-06).
DEFAULT_SAMPLER = os.environ.get("SAMPLER", "er_sde").strip()
DEFAULT_SCHEDULER = os.environ.get("SCHEDULER", "simple").strip()
# ⭐ v7: 환경변수로 뺐다. 원래 Qwen 의 9:16 값을 그대로 쓰던 것이라
#    Krea 2 최적값을 찾으면 템플릿만 고쳐서 바꿀 수 있게 했다.
DEFAULT_WIDTH = int(os.environ.get("WIDTH", "928"))
DEFAULT_HEIGHT = int(os.environ.get("HEIGHT", "1664"))
DEFAULT_SEED = 42

# 콜드스타트 실측용. 첫 응답에 담아 보낸다.
T_PROCESS_START = time.time()
_boot_stats = {}


def log(msg):
    print(msg, flush=True)


# ──────────────────────────────────────────────────────────────────────
# 1) 모델 받기
# ──────────────────────────────────────────────────────────────────────

def _model_folder():
    """본체가 들어갈 ComfyUI 폴더 이름. 로더마다 보는 곳이 다르다.

    gguf  → models/unet/               UnetLoaderGGUF 가 여기를 본다
    fp8   → models/diffusion_models/   UNETLoader 가 여기를 본다
    """
    return "unet" if MODEL_BACKEND == "gguf" else "diffusion_models"


def _plan_downloads():
    """(repo_id, 저장소안 경로, ComfyUI 폴더 이름) 목록을 만든다.

    ⚠️ 2026-08-05 실패로 배운 것 — 목적지를 명시해야 한다.
       예전에는 local_dir=/comfyui/models 로 받아서 저장소의 폴더 구조를 그대로 썼다.
       Comfy-Org/Krea-2 는 폴더 이름이 마침 ComfyUI 와 같아서(vae/ loras/) 잘 됐는데,
       realrebelai 는 TURBO/ 라는 자기 폴더를 써서 파일이
       models/unet/TURBO/Krea-2-Turbo-Q8_0.gguf 로 들어갔다.
       그런데 워크플로에는 파일명만 적어서 ComfyUI 가 거부했다:
         Value not in list: 'Krea-2-Turbo-Q8_0.gguf'
                     not in ['TURBO/Krea-2-Turbo-Q8_0.gguf']
       → 이제 저장소 구조와 무관하게 우리가 정한 폴더에 파일명만으로 놓는다.

    ⚠️ 안 쓰는 폴백은 받지 않는다. 예전에 Raw GGUF 12.76GiB 를 매번 받고
       한 번도 안 썼다. 부품이 환경변수라 갈아탈 때 템플릿만 바꾸면 된다.
    """
    jobs = [
        (MODEL_REPO, MODEL_FILE, _model_folder()),
        (TE_REPO, TE_FILE, "text_encoders"),
        (VAE_REPO, VAE_FILE, "vae"),
    ]
    # v7: 나열된 LoRA 를 전부 받는다. 하나만 적었으면 하나만 받는다.
    for spec in LORA_SPECS:
        jobs.append((spec["repo"], spec["file"], "loras"))
    return jobs


def _download_one(repo_id, filename, comfy_folder):
    """받아서 /comfyui/models/<comfy_folder>/<파일명> 에 놓는다.

    ⚠️ 저장소가 어떤 폴더 구조를 쓰든 여기서 평평하게 만든다.
       그래야 워크플로에서 파일명만으로 가리킬 수 있다.
    """
    t0 = time.time()
    dest_dir = os.path.join(MODELS_DIR, comfy_folder)
    os.makedirs(dest_dir, exist_ok=True)

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest_dir,
        token=HF_TOKEN,
    )

    # 저장소 안에 하위 폴더가 있었으면 파일이 그만큼 깊이 들어간다. 끌어올린다.
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
    """LoRA 에서 LORA_SKIP_KEYS 에 걸리는 텐서를 빼고 덮어쓴다. (v8)

    워커가 뜰 때 한 번만 돈다. LORA_SKIP_KEYS 가 비어 있으면 아무것도 안 한다.

    ⚠️ 실패하면 원본을 그대로 둔다. 검열이 남더라도 워커가 죽는 것보다 낫다.
       (safetensors 를 다시 쓰다 깨지면 ComfyUI 가 로드 자체를 못 한다)
    """
    if not LORA_SKIP_KEYS:
        return

    try:
        from safetensors.torch import load_file, save_file

        tensors = load_file(path)
        before = len(tensors)

        kept = {}
        dropped = []
        for name, t in tensors.items():
            low = name.lower()
            if any(k in low for k in LORA_SKIP_KEYS):
                dropped.append(name)
            else:
                kept[name] = t

        if not dropped:
            log(f"[lora] ⚠️ SKIP_KEYS={LORA_SKIP_KEYS} 에 걸린 텐서가 없다. "
                f"이름을 잘못 적었을 수 있다. 원본 그대로 쓴다 ({before}개)")
            return

        if not kept:
            log(f"[lora] ⚠️ SKIP_KEYS={LORA_SKIP_KEYS} 가 전부를 걸러냈다({before}개). "
                f"LoRA 가 통째로 사라지므로 원본 그대로 쓴다. "
                f"'blocks' 대신 'blocks.' 처럼 점을 붙여볼 것")
            return

        save_file(kept, path)
        log(f"[lora] {os.path.basename(path)} — {before}개 → {len(kept)}개 "
            f"({len(dropped)}개 제외, keys={LORA_SKIP_KEYS})")
        log(f"[lora] 제외한 것 예시: {dropped[:3]}")

    except Exception as e:
        log(f"[lora] ⚠️ 층 제외 실패, 원본 그대로 쓴다: {type(e).__name__}: {e}")


def download_models():
    jobs = _plan_downloads()
    _lora_names = ", ".join(os.path.basename(s["file"]) for s in LORA_SPECS) or "없음"
    log(f"[download] {len(jobs)}개 파일 받기 시작 "
        f"(backend={MODEL_BACKEND}, model={MODEL_FILE}, "
        f"lora={_lora_names}, steps={DEFAULT_STEPS})")

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

    # v8: 받아둔 LoRA 에서 지정한 층을 뺀다. SKIP_KEYS 가 비어 있으면 아무것도 안 한다.
    for spec in LORA_SPECS:
        strip_lora_keys(os.path.join(MODELS_DIR, "loras",
                                     os.path.basename(spec["file"])))


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
                   loras, start_step, sampler, scheduler, backend):
    """ComfyUI API 형식 워크플로를 만든다.

    노드 구성은 ComfyUI 공식 템플릿(image_krea2_turbo_t2i.json)에서 확인한 것과 같다:
      UNETLoader → (LoraLoaderModelOnly …) → KSampler → VAEDecode → SaveImage
      CLIPLoader(type=krea2) → CLIPTextEncode → (ConditioningZeroOut)
      VAELoader / EmptyLatentImage

    ⚠️ LoRA 노드는 LORA_FILE 이 있을 때만 끼운다.
       Turbo 체크포인트는 증류가 이미 들어 있어서 증류 LoRA 를 또 얹으면 안 된다.
       (체형·얼굴 LoRA 처럼 목적이 다른 것은 얹어도 된다 — 공식 권장 사용법이다)

    ⭐ v7 두 가지가 늘었다
      1) LoRA 를 여러 개 사슬처럼 잇는다 (노드 40, 41, 42 …)
      2) start_step 이 0 보다 크면 KSampler 를 둘로 나눈다
         앞구간은 LoRA 를 안 거친 원본 모델로 돌려 구도를 잡게 하고,
         뒷구간만 LoRA 를 태워 체형·질감을 입힌다.
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

    # ── LoRA 사슬 ────────────────────────────────────────────────
    # 본체("1")에서 시작해 LoRA 를 하나씩 이어 붙인다. 마지막 노드가 최종 모델이 된다.
    # 노드 번호는 40번대를 쓴다. 기존 번호와 부딪히지 않게 하기 위해서다.
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
        # 앞구간: LoRA 를 안 거친 원본("1")으로 돌린다. 여기서 구도가 정해진다.
        # ⚠️ return_with_leftover_noise=enable 로 노이즈를 남겨 뒷구간에 넘긴다.
        wf["8"] = {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["1", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "add_noise": "enable",
                "noise_seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "start_at_step": 0,
                "end_at_step": start_step,
                "return_with_leftover_noise": "enable",
            },
        }
        # 뒷구간: LoRA 사슬을 태운다. 체형·질감이 여기서 입혀진다.
        # ⚠️ add_noise=disable 이어야 앞구간 결과를 그대로 이어받는다.
        #    enable 로 두면 노이즈가 두 번 들어가 그림이 망가진다.
        wf["81"] = {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": model_ref,
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["8", 0],
                "add_noise": "disable",
                "noise_seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "start_at_step": start_step,
                "end_at_step": steps,
                "return_with_leftover_noise": "disable",
            },
        }
        latent_out = ["81", 0]
    else:
        # 분기를 안 쓰면 v6 과 똑같이 KSampler 하나로 끝낸다.
        wf["8"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": model_ref,
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
        }
        latent_out = ["8", 0]

    wf["9"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": latent_out, "vae": ["3", 0]},
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

def resolve_loras(job):
    """요청에 적힌 LoRA 지정을 받아둔 목록(LORA_SPECS)과 맞춘다.

    반환값은 (목록, 오류메시지) 다. 오류가 있으면 목록은 None 이다.

    방식 A (기존) — loras 를 안 보내면 받아둔 것을 전부 쓴다.
        lora_strength 를 보내면 모든 LoRA 에 그 강도를 적용한다.
        LoRA 가 하나뿐이던 v6 과 결과가 같다.

    방식 B (신규) — loras 목록으로 골라 쓴다.
        {"loras": [{"file": "Radiance", "strength": 0.6}]}
        file 은 파일명 일부만 적어도 된다. 긴 이름을 다 안 쓰게 하려는 것이다.

    ⚠️ 못 찾으면 조용히 넘어가지 않고 오류로 돌려준다.
       예전에 LORA_FILE 이 조용히 무시돼서 원인을 못 찾은 적이 있다.
    """
    if not LORA_SPECS:
        return [], None

    req = job.get("loras")

    if req is None:
        s = job.get("lora_strength")
        if s is None:
            return [dict(x) for x in LORA_SPECS], None
        out = []
        for spec in LORA_SPECS:
            d = dict(spec)
            d["strength"] = float(s)
            out.append(d)
        return out, None

    if not isinstance(req, list):
        return None, "loras 는 목록이어야 한다."

    have = [os.path.basename(s["file"]) for s in LORA_SPECS]
    out = []
    for item in req:
        if not isinstance(item, dict):
            return None, f"loras 항목이 잘못됐다: {item!r}"
        key = str(item.get("file", "")).strip()
        matched = None
        for spec in LORA_SPECS:
            if key and key.lower() in os.path.basename(spec["file"]).lower():
                matched = spec
                break
        if matched is None:
            return None, f"LoRA 를 못 찾았다: '{key}' (워커가 받아둔 것: {have})"
        d = dict(matched)
        if item.get("strength") is not None:
            d["strength"] = float(item["strength"])
        out.append(d)
    return out, None


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
    sampler = str(job.get("sampler", DEFAULT_SAMPLER))
    scheduler = str(job.get("scheduler", DEFAULT_SCHEDULER))

    loras, lora_err = resolve_loras(job)
    if lora_err:
        return {"error": lora_err}
    start_step = int(job.get("lora_start_step", DEFAULT_LORA_START_STEP))
    # ⚠️ backend 는 요청마다 못 바꾼다. 받아둔 파일이 그 하나뿐이기 때문이다.
    #    바꾸려면 템플릿 환경변수(MODEL_BACKEND / MODEL_FILE)를 고쳐야 한다.
    backend = MODEL_BACKEND

    seed = job.get("seed", DEFAULT_SEED)
    seed = random.randint(0, 2**31 - 1) if seed in (None, -1, "random") else int(seed)

    wf = build_workflow(str(prompt), str(negative), width, height, steps, cfg,
                        seed, loras, start_step, sampler, scheduler, backend)

    watcher = VramWatcher()
    watcher.start()
    t0 = time.time()
    try:
        prompt_id = queue_workflow(wf)
        _lora_desc = ", ".join(
            f"{os.path.basename(s['file'])}@{s['strength']}" for s in loras) or "없음"
        log(f"[job] 큐 등록 {prompt_id} — {width}x{height} {steps}스텝 "
            f"cfg{cfg} seed{seed} backend={backend} "
            f"lora=[{_lora_desc}] start_step={start_step} sampler={sampler}")
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
        # ⭐ v7: 무엇이 실제로 적용됐는지 눈으로 확인할 수 있게 담는다
        "loras_applied": [
            {"file": os.path.basename(s["file"]), "strength": s["strength"]}
            for s in loras
        ] or None,
        "lora_start_step": start_step if loras else None,
        # v8: 어떤 층을 뺐는지. 비었으면 아무것도 안 뺀 것(v7 과 동일 동작)
        "lora_skip_keys": LORA_SKIP_KEYS or None,
        # 아래 둘은 v6 과 형식을 맞추기 위해 남겨둔다 (첫 번째 LoRA 기준)
        "lora_strength": loras[0]["strength"] if loras else None,
        "sampler": sampler,
        "scheduler": scheduler,
        "backend": backend,
        "model": f"{MODEL_REPO}/{MODEL_FILE}",
        "text_encoder": f"{TE_REPO}/{TE_FILE}",
        "lora": f"{loras[0]['repo']}/{loras[0]['file']}" if loras else None,
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
