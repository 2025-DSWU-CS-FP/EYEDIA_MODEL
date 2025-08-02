import os
import json
import traceback
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import requests
import openai
from dotenv import load_dotenv
from urllib.parse import quote


# === 환경 변수 로드 ===
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080/api/v1/paintings")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# === FastAPI 초기화 ===
app = FastAPI()

# === CLIP 모델 초기화 ===
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)

# === GPT 도슨트 생성 ===
def gpt_docent_ko(crop_description: str) -> str:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "당신은 미술관의 도슨트입니다. "
        "아래 설명을 바탕으로 관람객에게 친절하고 감성적으로 설명해주세요. "
        "기술적인 내용은 피하고, 그림에 없는 내용은 말하지 마세요.\n\n"
        f"[설명]\n{crop_description}\n\n"
        "→ 도슨트 스타일로 설명:"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# === 메타데이터 로드 ===
def load_artworks():
    with open("data/faiss/met_structured_with_objects.json", "r", encoding="utf-8") as f:
        structured_data = json.load(f)
    with open("data/faiss/met_text_meta.json", "r", encoding="utf-8") as f:
        text_meta_data = json.load(f)
    return structured_data, text_meta_data

# === 이미지 임베딩 ===
def embed_image(img: Image.Image):
    inputs = clip_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32").squeeze()

# === Crop padding (유사도 향상) ===
def pad_crop(image: Image.Image, padding: int = 50):
    w, h = image.size
    new_img = Image.new("RGB", (w + 2 * padding, h + 2 * padding), (0, 0, 0))
    new_img.paste(image, (padding, padding))
    return new_img

# === 유사 작품 찾기 ===
def find_most_similar(uploaded_img: Image.Image, structured_data, text_meta_data, threshold=0.05):
    uploaded_vec = embed_image(uploaded_img)
    best_match = None
    max_score = -float("inf")

    for art in structured_data:
        art_img_path = art["image_path"]
        try:
            art_img = Image.open(art_img_path).convert("RGB")
            art_vec = embed_image(art_img)
            score = float(np.dot(uploaded_vec, art_vec))
            if score > max_score:
                max_score = score
                best_match = art
        except Exception as e:
            print(f"이미지 로드 실패: {art_img_path} - {e}")

    if max_score < threshold:
        return None

    object_id = best_match["full_image_id"]
    # object_id = best_match["crop_id"]
    meta_info = next((item for item in text_meta_data if item["objectID"] == object_id), None)

    return {
        "objectId": object_id,
        "title": meta_info["title"] if meta_info else None,
        "artist": meta_info["artist"] if meta_info else None,
        "description": meta_info["summary"] if meta_info else None,
        "imagePath": meta_info["image_path"] if meta_info else best_match["image_path"],
        "exhibition": "The_Met"
    }

def find_most_object_imilar(uploaded_img: Image.Image, structured_data, text_meta_data, threshold=0.05):
    uploaded_vec = embed_image(uploaded_img)
    best_match = None
    max_score = -float("inf")

    for art in structured_data:
        art_img_path = art["image_path"]
        try:
            art_img = Image.open(art_img_path).convert("RGB")
            art_vec = embed_image(art_img)
            score = float(np.dot(uploaded_vec, art_vec))
            if score > max_score:
                max_score = score
                best_match = art
        except Exception as e:
            print(f"이미지 로드 실패: {art_img_path} - {e}")

    if max_score < threshold:
        return None

    object_id = best_match["full_image_id"]
    object_id = best_match["crop_id"]
    meta_info = next((item for item in text_meta_data if item["objectID"] == object_id), None)

    return {
        "objectId": object_id,
        "title": meta_info["title"] if meta_info else None,
        "artist": meta_info["artist"] if meta_info else None,
        "description": meta_info["summary"] if meta_info else None,
        "imagePath": meta_info["image_path"] if meta_info else best_match["image_path"],
        "exhibition": "The_Met"
    }

# 백엔드: 이미지 업로드 API 호출
def send_image_to_backend(image_file_path, exhibition, title):
    with open(image_file_path, "rb") as img_file:
        files = {"image": img_file}
        data = {"exhibition": exhibition, "title": title}
        res = requests.post(f"{BACKEND_URL}/upload", files=files, data=data)
    res.raise_for_status()
    return res.json()["result"]

def send_object_image_to_backend(crop_path, painting_id, description, exhibition, title):
    files = {"image": open(crop_path, "rb")}
    data = {
        "paintingId": str(painting_id),
        "description": description or "",
        "exhibition": exhibition,
        "title": title
    }
    res = requests.post(f"{BACKEND_URL}/objects/upload", files=files, data=data)
    res.raise_for_status()
    return res.json().get("result")


# === 백엔드 전송 ===
def send_metadata_to_backend(image_url, artwork, is_crop=False, painting_id=None):
    if is_crop:
        payload = {
            "description": artwork.get("description"),
            "imageUrl": image_url,
            "paintingId": painting_id,
            "title": artwork["title"],
            "exhibition": artwork["exhibition"]
        }
        endpoint = f"{BACKEND_URL}/objects/save"
    else:
        payload = {
            "objectId": artwork["objectId"],
            "title": artwork["title"],
            "artist": artwork["artist"],
            "description": artwork["description"],
            "exhibition": artwork["exhibition"],
            "imageUrl": image_url
        }
        endpoint = f"{BACKEND_URL}/save"

    print(f"[POST] {endpoint}\nPayload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    res = requests.post(endpoint, json=payload)
    res.raise_for_status()
    return res.json()

# === 전체 그림 title → paintingId 조회 ===
def get_painting_id_by_title(title: str, artist: str = None):
    try:
        res = requests.get(f"{BACKEND_URL}/find-id", params={"title": title, "artist": artist})
        res.raise_for_status()
        return res.json().get("paintingId")
    except Exception as e:
        print(f"paintingId 조회 실패: {e}")
        return None

# === /process-image === (전체 그림)
@app.post("/process-image")
async def process_uploaded_image(file: UploadFile):
    try:
        os.makedirs("temp", exist_ok=True)
        save_path = f"temp/{file.filename}"
        with open(save_path, "wb") as f:
            f.write(await file.read())

        structured_data, text_meta_data = load_artworks()
        best_match = find_most_similar(Image.open(save_path).convert("RGB"), structured_data, text_meta_data, threshold=0.1)
        if not best_match:
            return JSONResponse(content={"error": "유사한 작품을 찾지 못했습니다."}, status_code=404)
        
        exhibition = best_match["exhibition"]
        title = best_match["title"]

        # GPT 설명 증강
        if best_match["description"]:
            best_match["description"] = gpt_docent_ko(best_match["description"])

        # S3 업로드 대신 로컬 URL 사용
        image_url = send_image_to_backend(save_path, exhibition, title)
        send_metadata_to_backend(image_url, best_match, is_crop=False)

        return JSONResponse(content={"result": "success", **best_match, "imageUrl": image_url}, status_code=200)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

# === /process-crop === (객체 크롭)
@app.post("/process-crop")
async def process_crop_image(file: UploadFile):
    try:
        os.makedirs("temp", exist_ok=True)
        save_path = f"temp/{file.filename}"
        with open(save_path, "wb") as f:
            f.write(await file.read())

        img = Image.open(save_path).convert("RGB")
        img_padded = pad_crop(img)

        structured_data, text_meta_data = load_artworks()
        best_match = find_most_similar(img_padded, structured_data, text_meta_data, threshold=0.05)
        if not best_match:
            return JSONResponse(content={"error": "유사한 객체를 찾지 못했습니다."}, status_code=404)

        # GPT 설명
        if best_match["description"]:
            best_match["description"] = gpt_docent_ko(best_match["description"])

        painting_title = best_match.get("title")  
        painting_artist = best_match.get("artist")
        exhibition = best_match["exhibition"]

        painting_id = get_painting_id_by_title(painting_title, painting_artist)
        if not painting_id:
            return JSONResponse(content={"error": "해당 그림의 ID를 찾을 수 없습니다."}, status_code=404)

        image_url = send_object_image_to_backend(save_path, painting_id, best_match["description"], exhibition, painting_title)
        send_metadata_to_backend(image_url, best_match, is_crop=True, painting_id=painting_id)
        print(painting_id)

        return JSONResponse(content={"result": "success", **best_match, "imageUrl": image_url}, status_code=200)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
