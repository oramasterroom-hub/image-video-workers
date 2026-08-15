"""미니맥스 H3 영상+소리 워커 — 런포드 껍데기

이 파일이 하는 일은 하나뿐이다: 런포드 요청을 받아 몸통(h3_core)에 넘긴다.

  h3_core.py     ⬅ 몸통. 실제 일은 전부 여기서 한다. 런포드를 모른다
  handler_h3.py  ⬅ 이 파일. 로컬로 옮기면 버리는 부분

⭐ LTX 워커(handler_ltx.py)와 같은 구조다. 보스가 나중에 로컬로 옮길 때
   이 파일만 떼고 h3_core.py 는 그대로 쓴다.
"""

import runpod

import h3_core


def handler(event):
    """런포드가 요청마다 부른다."""
    job = event.get("input") or {}
    return h3_core.generate(job)


# 워커가 뜨는 시점에 한 번 — 파일 받기 + ComfyUI 기동
h3_core.boot()

runpod.serverless.start({"handler": handler})
