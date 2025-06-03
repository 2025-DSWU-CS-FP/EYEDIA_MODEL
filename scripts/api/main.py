import os
import json
import subprocess
import re
from typing import Set

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # .env에서 환경변수 로딩
app = FastAPI()

# Spring 서버 주소
BACKEND_VALIDATION_URL = "http://localhost:8080/api/vi/ai/painting-id"
BACKEND_RESPONSE_URL = "http://localhost:8080/api/vi/ai/object-description"
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN") or ""  # 실제 토큰
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")

genai.configure(api_key=GEMINI_API_KEY)
initialized_images: Set[str] = set()
STRUCTURE_PATH = "data/met_structure_with_objects.json"

# 요청 DTO
class ValidateRequest(BaseModel):
    paintingId: int

class AnalyzeClickRequest(BaseModel):
    image_id: str
    click_index: int = 0

class DescriptionRequest(BaseModel):
    crop_description: str

def get_docent_description_from_text(crop_description: str) -> str:
    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        prompt = (
            "당신은 미술관의 도슨트입니다. 아래 설명을 바탕으로 관람객에게 친절하게 설명해주세요. "
            "너무 딱딱하거나 기술적이지 않게 풀어서 말해주세요.\n\n"
            f"[작품 설명]: {crop_description}\n\n"
            "→ 도슨트 스타일로 설명해주세요:"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini 도슨트 설명 생성 실패: {e}")

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
    full_image_id = f"image_{req.image_id}"
    image_path = f"data/met_images/{full_image_id}.jpg"
    crop_id = f"{full_image_id}_crop{req.click_index}.jpg"
    crop_path = f"data/cropped_images/{crop_id}"

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"원본 이미지 없음: {image_path}")

    if req.image_id not in initialized_images:
        try:
            subprocess.run(["python", "scripts/fetch_text_and_build_faiss.py"], check=True)
            subprocess.run(["python", "scripts/click_and_find_faiss_seg.py", image_path], check=True)
            subprocess.run(["python", "scripts/crop_and_sav_each_description.py", image_path], check=True)
            initialized_images.add(req.image_id)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"❌ 스크립트 실행 실패: {e}")

    if not os.path.exists(crop_path):
        raise HTTPException(status_code=404, detail=f"Crop 이미지 없음: {crop_path}")

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
        raise HTTPException(status_code=500, detail=f"설명 추출 실패: {e}")

    # ⚠️ 안전하게 paintingId 변환
    try:
        painting_id = int(re.sub(r"\D", "", req.image_id))  # "435638"
    except:
        raise HTTPException(status_code=400, detail="image_id에서 paintingId 추출 실패")

    result_payload = {
        "paintingId": painting_id,
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
            detail=f"Spring 전송 실패: {response.status_code} {response.text}"
        )

    return {
        "status": "ok",
        "crop_id": crop_id,
        "description": matched_description,
        "backend_response": response.json()
    }

# ✅ Gemini 설명 생성 테스트 엔드포인트 -- 테스트용이므로 잘 개발되면 이 코드 지워주세요
@app.post("/test/gemini")
def test_gemini(req: DescriptionRequest):
    result = get_docent_description_from_text(req.crop_description)
    return {"docent_description": result}