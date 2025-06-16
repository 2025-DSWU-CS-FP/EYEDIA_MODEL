from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import json
import re

app = FastAPI()

BACKEND_RESPONSE_URL = "http://localhost:8080/api/vi/ai/object-description"
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMSIsImlhdCI6MTc0ODc3Mzk2NCwiZXhwIjoxNzQ4Nzc3NTY0fQ.7b78D6DdUXAs6LIwO-vnSSa87B743nfP1Cnx45H42ac"
STRUCTURE_PATH = "data/faiss/met_structured_with_objects.json"

class AnalyzeClickRequest(BaseModel):
    image_id: str

def extract_object_id(crop_id: str) -> int:
    match = re.search(r'_crop(\d+)\.jpg$', crop_id)
    return int(match.group(1)) if match else -1

@app.post("/analyze/click")
def analyze_click(req: AnalyzeClickRequest):
    image_filename = f"image_{req.image_id}.jpg"

    try:
        with open(STRUCTURE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON 로딩 실패: {str(e)}")

    matched = next((item for item in data if str(item["full_image_id"]) == req.image_id), None)
    if not matched:
        raise HTTPException(status_code=404, detail="해당 image_id에 대한 crop 정보 없음")

    for crop in matched.get("crops", []):
        crop_id = crop.get("crop_id")
        description = crop.get("crop_description", "설명 없음")
        object_id = extract_object_id(crop_id)
        paintingId = int(req.image_id)

        payload = {
            "paintingId": paintingId,
            "objectId": object_id,
            "description": description,
            "sendingType": "AI"
        }
        headers = {
            "Authorization": f"Bearer {ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }
        try:
            res = requests.post(BACKEND_RESPONSE_URL, json=payload, headers=headers)
            if res.status_code != 200:
                raise HTTPException(status_code=res.status_code, detail=f"Spring 전송 실패: {res.text}")
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Spring 요청 중 오류 발생: {str(e)}")

    return {
        "status": "ok",
        "message": "모든 객체 설명 전송 완료",
        "image_url": f"/images/{image_filename}"
    }
