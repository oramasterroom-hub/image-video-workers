"""
Qwen-Image-2512 (GGUF Q8_0) 런포드 서버리스 handler

⭐ 이번 작업의 목적은 그림 품질 판정이 아니다.
   24GB GPU 에서 도는지, 그리는 데 메모리를 얼마나 쓰는지(gpu_peak_gb) 재는 것이다.

모델 구성 (2026-08-04 파일 목록으로 확인한 실제 크기)
  그림 그리는 부분   qwen-image-2512-Q8_0.gguf   21.76 GB   (unsloth)
  글 읽는 부분       text_encoder 4조각          16.58 GB   (Qwen 원본)
  VAE + 설정         vae/tokenizer/scheduler      0.26 GB   (Qwen 원본)
                                              ──────────
                                                38.60 GB

⚠️ 런포드 모델 캐싱을 쓰지 않는다.
   모델 캐싱은 저장소를 통째로 받는데, GGUF 저장소는 양자화 15종이 들어 있어 256GB 다.
   우리가 쓸 건 그중 21.76GB 짜리 파일 하나뿐이라, 파일을 지정해서 직접 받는다.

⚠️ 그래서 HF_HUB_OFFLINE 을 켜지 않는다. (이전 모델들과 다른 점)
   이전 handler 들은 캐시에서 읽기만 해서 오프라인이 맞았지만, 여기서는 직접 받아야 한다.
"""

import base64
import io
import os
import sys
import time
import traceback

print(f"[init] python: {sys.executable}", flush=True)

# 허깅페이스에서 받는 속도를 올린다. 38.6GB 를 받아야 해서 켠다.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# 메모리 파편화 대응. 여유가 3GiB 대로 빠듯해서 반드시 필요하다.
# ⚠️ torch 를 import 하기 전에 설정해야 적용된다.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 불러오기를 감싸서, 죽더라도 어디서 죽었는지 반드시 로그에 남게 한다.
try:
    import runpod

    print("[init] runpod 불러오기 OK", flush=True)

    import torch

    print(
        f"[init] torch {torch.__version__} "
        f"cuda_build={torch.version.cuda} "
        f"cuda_available={torch.cuda.is_available()}",
        flush=True,
    )

    from huggingface_hub import hf_hub_download, snapshot_download
    from diffusers import (
        QwenImagePipeline,
        QwenImageTransformer2DModel,
        GGUFQuantizationConfig,
    )
    from accelerate import cpu_offload

    print("[init] diffusers/huggingface_hub 불러오기 OK", flush=True)
except Exception:
    print("[init] ⚠️ 불러오기 단계에서 실패했다", flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise


GGUF_REPO = os.environ.get("GGUF_REPO", "unsloth/Qwen-Image-2512-GGUF")
GGUF_FILE = os.environ.get("GGUF_FILE", "qwen-image-2512-Q8_0.gguf")
BASE_REPO = os.environ.get("BASE_REPO", "Qwen/Qwen-Image-2512")

TORCH_DTYPE = getattr(torch, os.environ.get("TORCH_DTYPE", "bfloat16"))

# ⭐ 공식 모델 카드 기준값 (2026-08-04 README 원문 확인)
#    ⚠️ guidance_scale 이 아니라 true_cfg_scale 이다. 이름이 이전 모델들과 다르다.
DEFAULT_STEPS = 50
DEFAULT_TRUE_CFG = 4.0
DEFAULT_SEED = 32
# ⚠️ diffusers 안에서 기본값이 서로 다르다.
#    __call__ 은 512, encode_prompt 는 1024. 헷갈리지 않게 명시해서 넘긴다.
DEFAULT_MAX_SEQ = 512

# ⭐ 공식이 지정한 해상도 7개. 이 목록 밖 크기는 모델이 학습하지 않아 결과가 나빠진다.
OFFICIAL_SIZES = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}
# 세로 숏폼용. 우리 기본값.
DEFAULT_WIDTH, DEFAULT_HEIGHT = OFFICIAL_SIZES["9:16"]

# 공식 모델 카드의 네거티브 프롬프트 원문(중국어).
# 글 읽는 부분이 Qwen2.5-VL 이라 중국어를 잘 읽는다. 공식이 쓴 그대로 쓴다.
#   저해상도·저화질·사지기형·손가락기형·과포화·밀랍인형느낌·얼굴디테일없음
#   ·과도하게매끄러움·AI느낌·구도혼란·글자흐림/왜곡
DEFAULT_NEGATIVE = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，"
    "人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
)


# ---- 워커가 켜질 때 한 번만 ----

if not torch.cuda.is_available():
    print("[init] ⚠️ CUDA를 못 찾았다. GPU 워커가 맞는지 확인할 것.", flush=True)
    raise RuntimeError("CUDA unavailable")

_t0 = time.time()

try:
    # ① 그림 그리는 부분 — GGUF 파일 하나만 받는다 (21.76GB)
    print(f"[init] GGUF 받는 중: {GGUF_REPO}/{GGUF_FILE}", flush=True)
    gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
    print(
        f"[init] GGUF 받기 완료 ({time.time() - _t0:.0f}초): {gguf_path}", flush=True
    )

    # ② 나머지 부품 — 원본 저장소에서 필요한 것만 받는다
    #    ⚠️ transformer 폴더의 가중치(40.86GB)는 받지 않는다. GGUF 로 대체하기 때문이다.
    #       다만 transformer/config.json 은 GGUF 를 해석하는 데 필요해서 받는다.
    _t1 = time.time()
    print(f"[init] 나머지 부품 받는 중: {BASE_REPO}", flush=True)
    base_dir = snapshot_download(
        repo_id=BASE_REPO,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "vae/*",
            "transformer/config.json",
        ],
    )
    print(f"[init] 나머지 받기 완료 ({time.time() - _t1:.0f}초): {base_dir}", flush=True)
except Exception:
    print("[init] ⚠️ 모델 파일 받는 단계에서 실패했다", flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    raise

try:
    # ③ GGUF 를 트랜스포머로 불러온다
    #    ⚠️ 파이프라인 통째로는 GGUF 를 못 부른다. 모델 단위로만 된다(공식 문서 명시).
    #    ⚠️ config/subfolder 를 넘길 때 나던 버그(이슈 #12891)는
    #       PR #12894 로 2026-01-02 고쳐졌고 0.39.0 에 들어 있다.
    print("[init] GGUF 트랜스포머 불러오는 중...", flush=True)
    transformer = QwenImageTransformer2DModel.from_single_file(
        gguf_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=TORCH_DTYPE),
        config=base_dir,
        subfolder="transformer",
        torch_dtype=TORCH_DTYPE,
    )
    print("[init] GGUF 트랜스포머 OK", flush=True)

    # ④ 나머지를 붙여 파이프라인을 만든다
    PIPE = QwenImagePipeline.from_pretrained(
        base_dir,
        transformer=transformer,
        torch_dtype=TORCH_DTYPE,
    )
    print("[init] 파이프라인 생성 완료", flush=True)

    # ⑤ ⭐ 메모리 전략
    #
    # ⚠️ 2026-08-04 1차 시도 실패 기록.
    #    처음에는 PIPE.enable_model_cpu_offload() 를 썼는데, 그건 세 덩어리를
    #    전부 CPU 메모리에 대기시킨다. 그래서 CPU 쪽에서 터졌다.
    #      트랜스포머 20.27 + 글읽는부분 15.44 + VAE 0.24 + 오버헤드 = 37.33 GiB
    #      컨테이너 RAM 한도 46.57 GiB → 80% 를 쓰고 있다가 생성 시작하며 OOM(exit 137)
    #    ⚠️ 워커 로그의 "503GB available" 은 기계 전체 값이지 우리 컨테이너 몫이 아니다.
    #       우리 몫은 46.57 GiB 다. 이걸 잘못 읽어서 설계가 틀렸다.
    #
    # 그래서 배치를 바꾼다.
    #   트랜스포머 + VAE → GPU 상주    (20.51 GiB / 24GB)
    #   글 읽는 부분     → CPU 상주    (15.44 GiB / 46.57GiB)
    #
    # 이러면 CPU 메모리가 37 → 17 GiB 로 줄어 여유가 넉넉해지고,
    # 트랜스포머가 GPU 에 상주하므로 50스텝 생성이 빠르다.
    #
    # ⚠️ 글 읽는 부분(8.3B)을 통째로 GPU 에 올리면 20.27+15.44=35.7 로 24GB 를 넘는다.
    #    그래서 cpu_offload 로 레이어를 하나씩만 올렸다 내린다.
    #    대가: 프롬프트 읽는 게 느려진다. 다만 50스텝 생성은 영향 없다.
    PIPE.transformer.to("cuda")
    PIPE.vae.to("cuda")
    print("[init] 트랜스포머·VAE 를 GPU 에 올림", flush=True)

    cpu_offload(PIPE.text_encoder, execution_device=torch.device("cuda"))
    print("[init] 글 읽는 부분은 CPU 에 두고 레이어 단위로만 GPU 사용", flush=True)

    # ⑥ 마지막 펼치는 단계를 조각내서 처리한다. 화질을 깎는 설정이 아니다.
    #    ⚠️ 2026-08-03 확인: 이 기능은 파이프라인이 아니라 VAE 쪽에 있다.
    #       PIPE.enable_vae_tiling() 이라는 함수는 없다.
    for name, fn in (("타일", "enable_tiling"), ("슬라이스", "enable_slicing")):
        try:
            getattr(PIPE.vae, fn)()
            print(f"[init] VAE {name} 처리 켬", flush=True)
        except Exception as e:
            print(f"[init] ⚠️ VAE {name} 처리를 못 켰다: {e}", flush=True)
except Exception:
    print("[init] ⚠️ 모델 로딩 단계에서 실패했다", flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    raise

print(f"[init] 모델 로딩 완료 (총 {time.time() - _t0:.0f}초)", flush=True)
# ⚠️ 1차 시도 때는 전부 CPU 에 있어서 이 값이 0.00 이라 아무 정보가 없었다.
#    이제 트랜스포머가 GPU 에 상주하므로 실제 점유량이 찍힌다.
print(
    f"[init] GPU 점유: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB "
    f"(예약 {torch.cuda.memory_reserved() / 1024**3:.2f} GiB)",
    flush=True,
)


# ---- 요청마다 ----


def handler(event):
    job = event.get("input") or {}

    prompt = job.get("prompt")
    if not prompt or not str(prompt).strip():
        return {"error": "prompt 가 비어 있다."}

    negative_prompt = job.get("negative_prompt")
    if negative_prompt is None:
        negative_prompt = DEFAULT_NEGATIVE
    negative_prompt = str(negative_prompt)

    # 공식 해상도 목록에서 비율로 고를 수 있게 한다.
    ratio = job.get("aspect_ratio")
    if ratio:
        if ratio not in OFFICIAL_SIZES:
            return {
                "error": f"공식 해상도 목록에 없는 비율이다: {ratio}. "
                         f"쓸 수 있는 것: {list(OFFICIAL_SIZES)}"
            }
        width, height = OFFICIAL_SIZES[ratio]
    else:
        width = int(job.get("width", DEFAULT_WIDTH))
        height = int(job.get("height", DEFAULT_HEIGHT))

    steps = int(job.get("steps", DEFAULT_STEPS))
    true_cfg = float(job.get("true_cfg_scale", DEFAULT_TRUE_CFG))
    seed = int(job.get("seed", DEFAULT_SEED))
    max_seq = int(job.get("max_sequence_length", DEFAULT_MAX_SEQ))

    # 공식 목록 밖 크기면 알려준다. 막지는 않는다 — 실측이 목적이라 실험 여지를 남긴다.
    off_list = (width, height) not in OFFICIAL_SIZES.values()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t_start = time.time()
    try:
        image = PIPE(
            prompt=str(prompt),
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            true_cfg_scale=true_cfg,
            max_sequence_length=max_seq,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
    # ⚠️ except 에서 응답만 돌려보내면 로그에 흔적이 안 남는다.
    #    나중에 자동으로 돌릴 때 추적이 불가능해지므로 로그에도 찍는다.
    except torch.cuda.OutOfMemoryError as e:
        peak = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.empty_cache()
        print(
            f"[job] ⚠️ GPU 메모리 부족: {width}x{height} {steps}스텝 / 최대 {peak:.2f} GiB",
            flush=True,
        )
        traceback.print_exc()
        sys.stdout.flush()
        return {
            "error": f"GPU 메모리 부족: {width}x{height}. ({e})",
            "gpu_peak_gb": round(peak, 2),
        }
    except Exception as e:
        print(f"[job] ⚠️ 이미지 생성 실패: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        return {"error": f"이미지 생성 실패: {type(e).__name__}: {e}"}

    elapsed = time.time() - t_start

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    payload = buf.getvalue()

    out_width, out_height = image.size

    result = {
        "image_base64": base64.b64encode(payload).decode("utf-8"),
        "format": "png",
        "bytes": len(payload),
        "width": out_width,
        "height": out_height,
        "seed": seed,
        "steps": steps,
        "true_cfg_scale": true_cfg,
        "max_sequence_length": max_seq,
        "negative_prompt_used": bool(negative_prompt),
        "model": f"{GGUF_REPO}/{GGUF_FILE}",
        "dtype": str(TORCH_DTYPE).replace("torch.", ""),
        # ⭐ 이번 작업의 목적. 이 숫자 하나를 보려고 만든 것이다.
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "generation_sec": round(elapsed, 1),
    }

    if off_list:
        result["note"] = (
            f"{width}x{height} 는 공식 해상도 목록에 없다. "
            f"결과가 나빠질 수 있다."
        )
    if (out_width, out_height) != (width, height):
        result["requested_size"] = f"{width}x{height}"

    return result


runpod.serverless.start({"handler": handler})
