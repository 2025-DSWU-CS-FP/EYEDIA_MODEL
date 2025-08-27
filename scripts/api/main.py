import os
import json
import traceback
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

BACKEND_URL = os.getenv("BACKEND_URL", "http://43.202.177.63:8080/api/v1/")
S3_URL = os.getenv("S3_URL", "https://s3-eyedia.s3.ap-northeast-2.amazonaws.com/")

app = FastAPI()

# CLIP 모델 초기화
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# 메타데이터 로드
def load_artworks():
    with open("data/faiss/met_structured_with_objects.json", "r", encoding="utf-8") as f:
        structured_data = json.load(f)
    with open("data/faiss/met_text_meta.json", "r", encoding="utf-8") as f:
        text_meta_data = json.load(f)
    return structured_data, text_meta_data

# 백엔드: 메타데이터 저장 API 호출
def send_metadata_to_backend(image_url, artwork):
    payload = {
        "objectId": artwork["objectId"],
        "title": artwork["title"],
        "artist": artwork["artist"],
        "description": artwork["description"],
        "exhibition": artwork["exhibition"],
        "imageUrl": image_url
    }
    print(f"[POST] {BACKEND_URL}/save\nPayload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    res = requests.post(f"{BACKEND_URL}/save", json=payload)
    res.raise_for_status()
    return payload

# 백엔드: WebSocket Push용 엔드포인트 호출
def push_painting_detected(payload):
    try:
        res = requests.post(f"http://43.202.177.63:8080/paintings-push", json=payload)
        res.raise_for_status()
        print("[✅] WebSocket push 성공")
    except Exception as e:
        print(f"[WARN] WebSocket push 실패: {e}")

# FastAPI 엔드포인트
@app.post("/process-image")
async def process_uploaded_image(file: UploadFile):
    try:

        # 메타데이터 백엔드 저장
        backend_payload = send_metadata_to_backend(image_url, best_match)

        # WebSocket Push
        push_painting_detected({**backend_payload, "result": "success"})

        # 최종 FastAPI 응답
        return JSONResponse(content={**backend_payload, "result": "success"}, status_code=200)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": f"서버 오류: {e}"}, status_code=500)