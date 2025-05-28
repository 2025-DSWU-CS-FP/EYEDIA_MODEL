from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import subprocess
import os
import json

app = FastAPI()
BACKEND_VALIDATION_URL = "http://localhost:8080/api/vi/ai/painting-id"
BACKEND_RESPONSE_URL = "http://localhost:8080/api/model/response"
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJnaWxkb25nMTIzNDU2IiwiaWF0IjoxNzQ4NDIwMTUxLCJleHAiOjE3NDg0MjM3NTF9.45I5bPXOpPOOcBlMCr5q4vldkGdbpLW-qu5lLhLzfNI"
initialized_images = set()
STRUCTURE_PATH = "data/met_structure_with_objects.json"

class ValidateRequest(BaseModel):
    paintingId: int


class AnalyzeClickRequest(BaseModel):
    image_id: str
    click_index: int = 0

@app.post("/validate/painting-id")
def validate_painting_id(req: ValidateRequest):
    try:
        response = requests.post(
            BACKEND_VALIDATION_URL,
            json={"paintingId": req.paintingId},
            headers={"Authorization": ACCESS_TOKEN}
        )
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Spring 서버에서 paintingId 유효하지 않음")
        return {"status": "valid"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spring 요청 실패: {e}")


@app.post("/analyze/click")
def analyze_click(req: AnalyzeClickRequest):
    full_image_id =f"image_{req.image_id}"
    image_path = f"data/met_images/{full_image_id}.jpg"
    crop_id = f"{full_image_id}_crop{req.click_index}.jpg"
    crop_path = f"data/cropped_images/{crop_id}"

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"❗ 원본 이미지 없음: {image_path}")

    # 최초 스크립트 실행
    if req.image_id not in initialized_images:
        try:
            subprocess.run(["python", "scripts/fetch_text_and_build_faiss.py"], check=True)
            subprocess.run(["python", "scripts/click_and_find_faiss_seg.py", image_path], check=True)
            subprocess.run(["python", "scripts/crop_and_sav_each_description.py", image_path], check=True)
            initialized_images.add(full_image_id)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"❌ 스크립트 실행 실패: {e}")

    # Crop ID 구성 및 이미지 존재 확인
    crop_id = f"{full_image_id}_crop{req.click_index}.jpg"
    crop_path = f"data/cropped_images/{crop_id}"

    if not os.path.exists(crop_path):
        raise HTTPException(status_code=404, detail=f"❗ Crop 이미지 없음: {crop_path}")

    # crop_description 로딩
    try:
        with open(STRUCTURE_PATH, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        matched_description = None
        for item in all_data:
            if str(item["full_image_id"]) == req.image_id:
                for crop in item.get("crops", []):
                    if crop["crop_id"] == crop_id:
                        matched_description = crop.get("crop_description", "")
                        break
                break

        if not matched_description:
            matched_description = "설명 없음"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ 설명 추출 실패: {e}")

    # Spring 서버로 전송
    result_payload = {
        "full_image_id": full_image_id,
        "crop_id": crop_id,
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
            detail=f"❌ Spring 전송 실패: {response.status_code} {response.text}"
        )

    return {
        "status": "ok",
        "crop_id": crop_id,
        "description": matched_description,
        "backend_response": response.json()
    }
