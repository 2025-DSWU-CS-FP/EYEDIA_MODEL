import os
import json
import traceback
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

# ✅ 환경 변수 로드
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080/api/v1/paintings")

app = FastAPI()

# ✅ CLIP 모델 초기화
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# ✅ 메타데이터 로드
def load_artworks():
    with open("data/faiss/met_structured_with_objects.json", "r", encoding="utf-8") as f:
        structured_data = json.load(f)
    with open("data/faiss/met_text_meta.json", "r", encoding="utf-8") as f:
        text_meta_data = json.load(f)
    return structured_data, text_meta_data

# ✅ 이미지 임베딩
def embed_image(img: Image.Image):
    inputs = clip_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32").squeeze()

# ✅ 가장 유사한 작품 찾기
def find_most_similar(uploaded_img: Image.Image, structured_data, text_meta_data):
    uploaded_vec = embed_image(uploaded_img)
    best_match = None
    max_score = -float("inf")

    for art in structured_data:
        art_img_path = art["image_path"]
        try:
            art_img = Image.open(art_img_path)
            art_vec = embed_image(art_img)
            score = float(np.dot(uploaded_vec, art_vec))
            if score > max_score:
                max_score = score
                best_match = art
        except Exception as e:
            print(f"❗ 이미지 로드 실패: {art_img_path} - {e}")

    if not best_match:
        return None

    # text_meta_data에서 추가 정보 매칭
    object_id = best_match["full_image_id"]
    meta_info = next((item for item in text_meta_data if item["objectID"] == object_id), None)

    # 메타데이터가 없으면 null로 대체
    result = {
        "objectId": object_id,
        "title": meta_info["title"] if meta_info else None,
        "artist": meta_info["artist"] if meta_info else None,
        "description": meta_info["summary"] if meta_info else None,
        "imagePath": meta_info["image_path"] if meta_info else best_match["image_path"],
        "exhibition": "The_Met"
    }
    return result

# ✅ 백엔드: 이미지 업로드 API 호출
def send_image_to_backend(image_file_path, exhibition, title):
    with open(image_file_path, "rb") as img_file:
        files = {"image": img_file}
        data = {
            "exhibition": exhibition,
            "title": title
        }
        res = requests.post(f"{BACKEND_URL}/upload", files=files, data=data)
        res.raise_for_status()
        return res.json()["result"] # ApiResponse 구조 기준

# ✅ 백엔드: 메타데이터 저장 API 호출
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
    return res.json()

# ✅ FastAPI 엔드포인트
@app.post("/process-image")
async def process_uploaded_image(file: UploadFile):
    try:
        # 1️⃣ 업로드 이미지 저장
        save_path = f"temp/{file.filename}"
        with open(save_path, "wb") as f:
            f.write(await file.read())

        # 2️⃣ 유사 작품 찾기
        structured_data, text_meta_data = load_artworks()
        best_match = find_most_similar(Image.open(save_path).convert("RGB"), structured_data, text_meta_data)

        if not best_match:
            return JSONResponse(content={"error": "유사한 작품을 찾지 못했습니다."}, status_code=404)

        exhibition = best_match["exhibition"]
        title = best_match["title"]

        # 3️⃣ 이미지 S3에 업로드
        image_url = send_image_to_backend(save_path, exhibition, title)

        # 4️⃣ 메타데이터 백엔드로 전송
        send_metadata_to_backend(image_url, best_match)

        return JSONResponse(content={
            "result": "success",
            "objectId": best_match["objectId"],
            "title": title,
            "artist": best_match["artist"],
            "description": best_match["description"],
            "exhibition": exhibition,
            "imageUrl": image_url
        }, status_code=200)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)