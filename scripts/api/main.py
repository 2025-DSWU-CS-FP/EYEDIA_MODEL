from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import subprocess
import os
import json

app = FastAPI()

# 🔗 백엔드 API 주소
BACKEND_CONFIRM_URL = "http://localhost:8080/api/v1/ai/painting-id"
BACKEND_RESPONSE_URL = "http://localhost:8080/api/model/response"
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJnaWxkb25nMTIzIiwiaWF0IjoxNzQ4MzY5MzcwLCJleHAiOjE3NDgzNzI5NzB9.VoJseC7omB1YOLQ8iIkk_H45KK4n0jn0wlc4vK17lPc"

# 📦 요청 모델
class AnalyzeClickRequest(BaseModel):
    image_id: str
    click_x: int = 0
    click_y: int = 0

@app.post("/analyze/click")
def analyze_click(req: AnalyzeClickRequest):
    try:
        full_image_id = f"image_{req.image_id}"
        image_path = f"data/met_images/{full_image_id}.jpg"

        # ✅ 1. 이미지 존재 확인
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"이미지 없음: {image_path}")


        click_x, click_y = req.click_x, req.click_y
        # ✅ 4. YOLO 실행 (bbox 생성)
        subprocess.run(["python", "scripts/click_and_find_faiss_seg.py", image_path], check=True)

        # ✅ 5. bbox 파일 확인 및 클릭된 객체 찾기
        bbox_path = f"data/bboxes/{full_image_id}.json"
        if not os.path.exists(bbox_path):
            raise HTTPException(status_code=404, detail=f"BBox 파일 없음: {bbox_path}")
        with open(bbox_path, "r") as f:
            bboxes = json.load(f)

        click_index = next(
            (idx for idx, box in enumerate(bboxes)
             if box["x1"] <= req.click_x <= box["x2"] and box["y1"] <= req.click_y <= box["y2"]),
            -1
        )
        if click_index == -1:
            raise HTTPException(status_code=404, detail="클릭 위치에 해당하는 객체가 없습니다")

        # ✅ 6. Crop 이미지 생성 및 설명 추론
        crop_id = f"{full_image_id}_crop{click_index}.jpg"
        crop_path = f"data/cropped_images/{crop_id}"
        if not os.path.exists(crop_path):
            raise HTTPException(status_code=404, detail=f"크롭 이미지 없음: {crop_path}")

        result_path = f"output/{crop_id}_result.json"
        if not os.path.exists(result_path):
            subprocess.run(["python", "scripts/crop_and_sav_each_description.py", crop_path, result_path], check=True)

        with open(result_path, "r", encoding="utf-8") as f:
            description = json.load(f).get("description", "")

        # ✅ 7. 결과 Spring 서버로 전송
        result_payload = {
            "full_image_id": full_image_id,
            "crop_id": crop_id,
            "description": description
        }
        headers = {
            "Authorization": f"Bearer {ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }
        response = requests.post(BACKEND_RESPONSE_URL, json=result_payload, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Spring 서버로 설명 전송 실패")

        return {
            "status": "ok",
            "description": description,
            "backend_response": response.json()
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"스크립트 실행 오류: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"기타 오류: {str(e)}")
