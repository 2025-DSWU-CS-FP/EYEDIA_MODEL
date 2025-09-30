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
from typing import Dict, Any, List

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://43.202.177.63:8080")
S3_URL = os.getenv("S3_URL", "https://s3-eyedia.s3.ap-northeast-2.amazonaws.com")
FAISS_JSON_PATH = PPath("./data/faiss/met_structured_with_objects.json")  # 실제 경로로 조정

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
def send_metadata_to_backend(art_id : int):
    item = _lookup_meta(art_id)
    if not item:
        raise ValueError(f"[send_metadata_to_backend] met_text_meta.json에서 ID {art_id}를 찾을 수 없습니다.")

    payload = {
        "artId": int(item.get("artId") or item.get("id")),   # 백엔드가 camelCase(objectId) 기대한다고 가정
        "title": item.get("title"),
        "artist": item.get("artist"),
        "description": item.get("summary"),
        "exhibition": 1, # The_Met id, 하드코딩
        "imageUrl": f"{S3_URL}/1/{art_id}/{art_id}.jpg"
    }
    
    url = f"{BACKEND_URL}/api/v1/paintings/save"
    print(f"[POST] {url}\nPayload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    print("✅ 백엔드 서버 DB 저장 성공")
VALID_QUADS = {"Q1", "Q2", "Q3", "Q4"}

@lru_cache(maxsize=1)
def _load_faiss_json() -> Dict[int, Dict[str, Any]]:
    """
    JSON을 로드하고 full_image_id(=painting_id)로 바로 접근 가능하도록 인덱싱.
    반환 형태: { full_image_id: { ..원본 json.. }, ... }
    """
    if not FAISS_JSON_PATH.exists():
        raise FileNotFoundError(f"FAISS JSON not found: {FAISS_JSON_PATH}")

    with FAISS_JSON_PATH.open("r", encoding="utf-8") as f:
        # 파일이 단일 객체 하나가 아니라 여러 항목 리스트일 수도 있어 대비
        data = json.load(f)

    # 케이스 A) 파일 구조가 "하나의 작품"만 담긴 단일 객체
    if isinstance(data, dict) and "full_image_id" in data:
        return {int(data["full_image_id"]): data}

    # 케이스 B) 여러 작품을 담은 리스트
    if isinstance(data, list):
        indexed = {}
        for item in data:
            if isinstance(item, dict) and "full_image_id" in item:
                indexed[int(item["full_image_id"])] = item
        return indexed

    raise ValueError("Unexpected FAISS JSON structure. Expect dict with full_image_id or list of such dicts.")


def _extract_quadrant_crops(painting_id: int, q: str) -> List[Dict[str, Any]]:
    """
    해당 painting의 crops 중 quadrant_ratios[q] > 0 인 것만 추출하여
    ratio 내림차순으로 정렬해 반환.
    """
    q = (q or "").upper().strip()
    if q not in VALID_QUADS:
        raise ValueError(f"q must be one of {sorted(VALID_QUADS)} (got: {q!r})")

    db = _load_faiss_json()
    item = db.get(int(painting_id))
    if not item:
        # painting이 없으면 빈 리스트
        return []

    crops = item.get("crops", []) or []
    results = []
    for c in crops:
        ratios = (c.get("quadrant_ratios") or {})
        ratio = float(ratios.get(q, 0.0) or 0.0)
        if ratio > 0.0:
            results.append({
                "cropId": c.get("crop_id"),
                "cropPath": c.get("crop_path"),
                "cropDescription": c.get("crop_description"),
                "primaryQuadrant": c.get("primary_quadrant"),
                "quadrant": q,
                "ratio": ratio
            })

    # 비율 내림차순
    results.sort(key=lambda x: x["ratio"], reverse=True)
    return results

# 백엔드: WebSocket Push용 엔드포인트 호출 - 그림 인식
def push_painting_detected(painting_id : int):
    # Todo: q존재 여부에 따른 백엔드 api 호출 변경
    # 백엔드로 전송해야 하는 데이터: artId, q, 해당 q에 해당하는 설명?
    try:
        res = requests.post(f"{BACKEND_URL}/api/v1/events/detect", json=painting_id)
        res.raise_for_status()
        print("[✅] WebSocket push 성공")
        return res.json()
    except Exception as e:
        print(f"[WARN] WebSocket push 실패: {e}")
        return res.json()

# 백엔드: WebSocket Push용 엔드포인트 호출 - 영역 인식
def push_painting_area_detected(painting_id: int, q: str | list[str] = None, top_k: int | None = None) -> bool:
    """
    painting_id와 여러 사분면(q)을 받아 해당 crop 리스트를 구성해 백엔드로 POST.
    q는 문자열 하나("Q1") 또는 문자열 리스트(["Q1","Q2"]) 모두 가능.
    """
    try:
        # q를 리스트로 정규화
        if isinstance(q, str):
            q_list = [q.upper().strip()]
        elif isinstance(q, list):
            q_list = [str(x).upper().strip() for x in q if x]
        else:
            q_list = []

        # 유효한 Q만 필터링
        q_list = [x for x in q_list if x in VALID_QUADS]

        crops_all = []
        for q_item in q_list:
            crops = _extract_quadrant_crops(painting_id, q_item)
            if top_k is not None and top_k > 0:
                crops = crops[:top_k]

            # 각 crop에 현재 q 추가 (겹치는 경우도 그대로 허용)
            for c in crops:
                crops_all.append({**c, "quadrant": q_item})

        payload = {
            "artId": int(painting_id),
            "q": q_list,
            "list": crops_all
        }
        
        url = f"{BACKEND_URL}/api/v1/events/detect-area"
        print(f"[Test] 응답 결과 확인용: {url} (artId={painting_id}, q={q_list}, count={len(crops_all)}, data={payload})")
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        print(f"[✅] WebSocket push 성공: {url} (artId={painting_id}, q={q_list}, count={len(crops_all)})")
        return res.json()

    except Exception as e:
        print(f"[WARN] WebSocket push 실패: {e}")
        return res.json()

# FastAPI 엔드포인트
@app.post("/process-image")
async def process_uploaded_image(
    painting_id: int = Query(..., alias="art_id"),
    q: str | None = Query(None, alias="q")
):
    try:
        # WebSocket Push
        if(q):
            backend_payload = push_painting_area_detected(painting_id, q) # q1받아오도록 수정. 정해진 api 있음

        else:
            # 메타데이터 백엔드 저장
            send_metadata_to_backend(painting_id)
            backend_payload = push_painting_detected(painting_id) # q1받아오도록 수정. 정해진 api 있음

        # 최종 FastAPI 응답
        return JSONResponse(backend_payload)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": f"서버 오류: {e}"}, status_code=500)