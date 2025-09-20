from functools import lru_cache
import os
import json
import traceback
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Path
from fastapi.responses import JSONResponse
from pathlib import Path as PPath
from fastapi import Query

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://43.202.177.63:8080")
S3_URL = os.getenv("S3_URL", "https://s3-eyedia.s3.ap-northeast-2.amazonaws.com")

app = FastAPI()

# 메타데이터 로드 및 인덱싱
@lru_cache(maxsize=1)
def _load_text_meta_index():
    with PPath("data/faiss/met_text_meta.json").open("r", encoding="utf-8") as f:
        items = json.load(f)
    index = {}
    for it in items:
        if "id" in it:
            index[str(it["id"])] = it
        if "objectID" in it:
            index[str(it["objectID"])] = it
    return index

# 인덱싱한 메타데이터 검색
def _lookup_meta(painting_id: int | str):
    idx = _load_text_meta_index()
    return idx.get(str(painting_id))

# 백엔드: 메타데이터 저장 API 호출
def send_metadata_to_backend(painting_id : int):
    item = _lookup_meta(painting_id)
    if not item:
        raise ValueError(f"[send_metadata_to_backend] met_text_meta.json에서 ID {painting_id}를 찾을 수 없습니다.")

    payload = {
        "objectId": int(item.get("objectID") or item.get("id")),   # 백엔드가 camelCase(objectId) 기대한다고 가정
        "title": item.get("title"),
        "artist": item.get("artist"),
        "description": item.get("summary"),
        "exhibition": 1, # The_Met id, 하드코딩
        "imageUrl": f"{S3_URL}/1/{painting_id}/{painting_id}"
    }
    
    url = f"{BACKEND_URL}/api/v1/paintings/save"
    print(f"[POST] {url}\nPayload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return payload

# 백엔드: WebSocket Push용 엔드포인트 호출 - 그림 인식
def push_painting_detected(painting_id : int):
    # Todo: q존재 여부에 따른 백엔드 api 호출 변경
    # 백엔드로 전송해야 하는 데이터: artId, q, 해당 q에 해당하는 설명?
    try:
        res = requests.post(f"{BACKEND_URL}/api/v1/events/detect", json=painting_id)
        res.raise_for_status()
        print("[✅] WebSocket push 성공")
    except Exception as e:
        print(f"[WARN] WebSocket push 실패: {e}")

# 백엔드: WebSocket Push용 엔드포인트 호출 - 영역 인식
def push_painting_area_detected(painting_id: int, q: str = None):
    try:
        # 영역 정보 포함 → JSON 객체 전송
        # Todo: 백엔드와 맞춰야 함. 백엔드 dto 수정 필요
        payload = {
            "artId": painting_id,
            "q": q
        }
        url = f"{BACKEND_URL}/api/v1/events/detect-area"

        res = requests.post(url, json=payload)
        res.raise_for_status()
        print(f"[✅] WebSocket push 성공: {url}")

    except Exception as e:
        print(f"[WARN] WebSocket push 실패: {e}")

# FastAPI 엔드포인트
@app.post("/process-image")
async def process_uploaded_image(
    painting_id: int = Query(..., alias="art_id"),
    q: str | None = Query(None, alias="q")
):
    try:
        # 메타데이터 백엔드 저장
        backend_payload = send_metadata_to_backend(painting_id)

        # WebSocket Push
        if(q):
            push_painting_area_detected(painting_id, q) # q1받아오도록 수정. 정해진 api 있음

        else:
            push_painting_detected(painting_id) # q1받아오도록 수정. 정해진 api 있음

        # 최종 FastAPI 응답
        return JSONResponse(content={**backend_payload, "result": "success"}, status_code=200)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": f"서버 오류: {e}"}, status_code=500)