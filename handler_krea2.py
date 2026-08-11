"""Krea 2 이미지 워커 — 런포드 껍데기 (v9)

이 파일이 하는 일은 하나뿐이다: 런포드 요청을 받아 몸통(krea2_core)에 넘긴다.

  krea2_core.py     ⬅ 몸통. 실제 일은 전부 여기서 한다. 런포드를 모른다
  handler_krea2.py  ⬅ 이 파일. 로컬로 옮기면 버리는 부분

⭐ 왜 갈랐나
   보스가 나중에 컴퓨터를 바꿔 로컬에서 돌릴 예정이다.
   그때 이 파일만 떼고 krea2_core.py 는 그대로 쓴다. 다시 만들지 않는다.
   영상 워커(handler_ltx.py + ltx_core.py)와 같은 구조다.

⚠️ v8 까지는 이 파일 하나가 811줄이었다.
   불러오는 순간 다운로드·ComfyUI 기동이 함께 돌고 마지막 줄에
   runpod.serverless.start() 가 박혀 있어서 떼어낼 수가 없었다.
   2026-08-11 에 갈랐다. 기능은 그대로 옮겼고 v7·v8 기능도 살아 있다.
"""

import runpod

import krea2_core


def handler(event):
    """런포드가 요청마다 부른다."""
    job = event.get("input") or {}
    return krea2_core.generate(job)


# 워커가 뜨는 시점에 한 번 — 파일 받기 + ComfyUI 기동
krea2_core.boot()

runpod.serverless.start({"handler": handler})
