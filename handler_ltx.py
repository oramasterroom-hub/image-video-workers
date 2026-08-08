"""LTX-2.3 영상 워커 — 런포드 껍데기

이 파일이 하는 일은 하나뿐이다: 런포드 요청을 받아 몸통(ltx_core)에 넘긴다.

  ltx_core.py     ⬅ 몸통. 실제 일은 전부 여기서 한다. 런포드를 모른다
  handler_ltx.py  ⬅ 이 파일. 로컬로 옮기면 버리는 부분

⭐ 왜 갈랐나 (설계 조건 5번)
   보스가 나중에 컴퓨터를 바꿔 로컬에서 돌릴 예정이다.
   그때 이 파일만 떼고 ltx_core.py 는 그대로 쓴다. 다시 만들지 않는다.

⚠️ 크레아2 워커(handler_krea2.py)는 이 둘이 한 덩어리였다.
   모듈을 불러오는 순간 다운로드·ComfyUI 기동이 함께 돌고,
   마지막 줄에 runpod.serverless.start() 가 박혀 있다. 그래서 떼어낼 수가 없다.
"""

import runpod

import ltx_core


def handler(event):
    """런포드가 요청마다 부른다."""
    job = event.get("input") or {}
    return ltx_core.generate(job)


# 워커가 뜨는 시점에 한 번 — 파일 받기 + ComfyUI 기동
ltx_core.boot()

runpod.serverless.start({"handler": handler})
