from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import subprocess
import os
import json

app = FastAPI()

# Spring 서버 주소 (예: Spring Boot Controller)
BACKEND_VALIDATION_URL = "http://localhost:8080/api/vi/ai/painting-id"
BACKEND_RESPONSE_URL = "http://localhost:8080/api/vi/ai/object-description"  # 실제 Spring 엔드포인트
ACCESS_TOKEN = ""  # 실제 토큰으로 대체

# 최초 스크립트 실행 여부 저장
initialized_images = set()

# JSON 구조 데이터 경로
STRUCTURE_PATH = "data/met_structure_with_objects.json"

# 요청 DTO
class ValidateRequest(BaseModel):
    paintingId: int

class AnalyzeClickRequest(BaseModel):
    image_id: str
    click_index: int = 0

# paintingId 유효성 확인
@app.post("/validate/painting-id")
def validate_painting_id(req: ValidateRequest):
    try:
        response = requests.post(
            BACKEND_VALIDATION_URL,
            json={"paintingId": req.paintingId},
            headers={"Authorization": ACCESS_TOKEN}
        )
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=" Spring 서버에서 paintingId 유효하지 않음")
        return {"status": "valid"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f" Spring 요청 실패: {e}")

# 클릭 분석 및 crop 설명 추출
@app.post("/analyze/click")
def analyze_click(req: AnalyzeClickRequest):
    full_image_id = f"image_{req.image_id}"
    image_path = f"data/met_images/{full_image_id}.jpg"
    crop_id = f"{full_image_id}_crop{req.click_index}.jpg"
    crop_path = f"data/cropped_images/{crop_id}"

    # 원본 이미지 확인
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"원본 이미지 없음: {image_path}")

    #  최초 요청: 스크립트 실행
    if req.image_id not in initialized_images:
        try:
            subprocess.run(["python", "scripts/fetch_text_and_build_faiss.py"], check=True)
            subprocess.run(["python", "scripts/click_and_find_faiss_seg.py", image_path], check=True)
            subprocess.run(["python", "scripts/crop_and_sav_each_description.py", image_path], check=True)
            initialized_images.add(req.image_id)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"❌ 스크립트 실행 실패: {e}")

    # crop 이미지 존재 확인
    if not os.path.exists(crop_path):
        raise HTTPException(status_code=404, detail=f" Crop 이미지 없음: {crop_path}")

    # crop_description 로딩
    try:
        with open(STRUCTURE_PATH, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        matched_description = "설명 없음"
        for item in all_data:
            if str(item["full_image_id"]) == req.image_id:
                for crop in item.get("crops", []):
                    if crop["crop_id"] == crop_id:
                        matched_description = crop.get("crop_description", "설명 없음")
                        break
                break
    except Exception as e:
        raise HTTPException(status_code=500, detail=f" 설명 추출 실패: {e}")

    #  Spring 서버로 전송
    result_payload = {
        "paintingId": int(req.image_id),  # Spring에서 paintingId로 받는다고 가정
        "cropId": crop_id,
        "description": matched_description
    }
    headers = {
        "Authorization": ACCESS_TOKEN,
        "Content-Type": "application/json"
    }
    response = requests.post(BACKEND_RESPONSE_URL, json=result_payload, headers=headers)

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f" Spring 전송 실패: {response.status_code} {response.text}"
        )

    #  최종 응답
    return {
        "status": "ok",
        "crop_id": crop_id,
        "description": matched_description,
        "backend_response": response.json()
    }