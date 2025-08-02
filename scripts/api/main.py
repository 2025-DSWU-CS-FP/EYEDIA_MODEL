from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import os, json, traceback
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
import openai
from dotenv import load_dotenv

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
        "아래 crop 설명을 바탕으로 관람객에게 친절하고 감성적으로 설명해주세요. "
        "기술적인 내용은 피하고, 그림에 없는 내용은 말하지 마세요.\n\n"
        f"[Crop 설명]\n{crop_description}\n\n"
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
def find_most_similar(uploaded_img: Image.Image, structured_data, text_meta_data):
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

    # 임계값 설정 (crop은 0.05~0.1로 완화)
    if max_score < 0.05:
        return None

    # 메타데이터 결합
    object_id = best_match["full_image_id"]
    meta_info = next((item for item in text_meta_data if item["objectID"] == object_id), None)
    result = {
        "objectId": object_id,
        "title": meta_info["title"] if meta_info else None,
        "artist": meta_info["artist"] if meta_info else None,
        "description": meta_info["summary"] if meta_info else None,
        "imagePath": meta_info["image_path"] if meta_info else best_match["image_path"],
        "exhibition": "The_Met"
    }
    return result

# === 백엔드 전송 ===
def send_metadata_to_backend(image_url, artwork):
    payload = {
        "objectId": artwork["objectId"],
        "title": artwork["title"],
        "artist": artwork["artist"],
        "description": artwork["description"],
        "exhibition": artwork["exhibition"],
        "imageUrl": image_url
    }
    res = requests.post(f"{BACKEND_URL}/save", json=payload)
    res.raise_for_status()
    return res.json()

# === /process-crop 엔드포인트 ===
@app.post("/process-crop")
async def process_crop_image(file: UploadFile):
    try:
        os.makedirs("temp", exist_ok=True)
        save_path = f"temp/{file.filename}"
        with open(save_path, "wb") as f:
            f.write(await file.read())

        # Crop padding 적용
        img = Image.open(save_path).convert("RGB")
        img_padded = pad_crop(img)

        # 유사 작품 검색
        structured_data, text_meta_data = load_artworks()
        best_match = find_most_similar(img_padded, structured_data, text_meta_data)
        if not best_match:
            return JSONResponse(content={"error": "유사한 객체를 찾지 못했습니다."}, status_code=404)

        # GPT 도슨트 설명
        if best_match["description"]:
            docent_desc = gpt_docent_ko(best_match["description"])
            best_match["description"] = docent_desc

        # 메타데이터 전송 (이미지 업로드 대신 URL만 저장)
        image_url = f"http://localhost:8000/{save_path}"  # 임시 URL
        send_metadata_to_backend(image_url, best_match)

        return JSONResponse(content={
            "result": "success",
            "objectId": best_match["objectId"],
            "title": best_match["title"],
            "artist": best_match["artist"],
            "description": best_match["description"],
            "exhibition": best_match["exhibition"],
            "imageUrl": image_url
        }, status_code=200)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
