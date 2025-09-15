from functools import lru_cache
import os
import json
import traceback
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Path
from fastapi.responses import JSONResponse
from pathlib import Path as PPath

load_dotenv()
<<<<<<< HEAD
print("[✅] OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080/api/v1/paintings")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
=======

BACKEND_URL = os.getenv("BACKEND_URL", "http://43.202.177.63:8080")
S3_URL = os.getenv("S3_URL", "https://s3-eyedia.s3.ap-northeast-2.amazonaws.com")
>>>>>>> bef946602615501d21df3445c9e4727da0508065

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

<<<<<<< HEAD
# GPT 도슨트 설명 생성 함수
def gpt_docent_ko(crop_description: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "당신은 미술관의 도슨트입니다. 아래 설명을 바탕으로 관람객에게 친절하게 설명해주세요. "
        "너무 딱딱하거나 기술적이지 않게 풀어서 말해주세요.\n\n"
        f"[작품 설명]: {crop_description}\n\n"
        "→ 도슨트 스타일로 설명해주세요:"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# 메타데이터 로드
def load_artworks():
    with open("data/faiss/met_structured_with_objects.json", "r", encoding="utf-8") as f:
        structured_data = json.load(f)
    with open("data/faiss/met_text_meta.json", "r", encoding="utf-8") as f:
        text_meta_data = json.load(f)
    return structured_data, text_meta_data

# 이미지 임베딩
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

# 가장 유사한 작품 찾기
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
            print(f"이미지 로드 실패: {art_img_path} - {e}")

    if not best_match:
        return None

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

# 백엔드: 이미지 업로드 API 호출
def send_image_to_backend(image_file_path, exhibition, title):
    with open(image_file_path, "rb") as img_file:
        files = {"image": img_file}
        data = {"exhibition": exhibition, "title": title}
        res = requests.post(f"{BACKEND_URL}/upload", files=files, data=data)
    res.raise_for_status()
    return res.json()["result"]
=======
# 인덱싱한 메타데이터 검색
def _lookup_meta(painting_id: int | str):
    idx = _load_text_meta_index()
    return idx.get(str(painting_id))
>>>>>>> bef946602615501d21df3445c9e4727da0508065

# 백엔드: 메타데이터 저장 API 호출
def send_metadata_to_backend(painting_id : int):
    item = _lookup_meta(painting_id)
    if not item:
        raise ValueError(f"[send_metadata_to_backend] met_text_meta.json에서 ID {painting_id}를 찾을 수 없습니다.")

    payload = {
        "objectId": int(item.get("objectID") or item.get("id")),   # 백엔드가 camelCase(objectId) 기대한다고 가정
        "title": item.get("title"),
        "artist": item.get("artist"),
        "description": item.get("summary"),
        "exhibition": 1, # The_Met id, 하드코딩
        "imageUrl": f"{S3_URL}/1/{painting_id}/{painting_id}"
    }
    
    url = f"{BACKEND_URL}/api/v1/paintings/save"
    print(f"[POST] {url}\nPayload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return payload

# 백엔드: WebSocket Push용 엔드포인트 호출
def push_painting_detected(painting_id : int):
    try:
        res = requests.post(f"{BACKEND_URL}/events/detect", json=painting_id)
        res.raise_for_status()
        print("[✅] WebSocket push 성공")
    except Exception as e:
        print(f"[WARN] WebSocket push 실패: {e}")

# FastAPI 엔드포인트
@app.post("/process-image/{painting_id}")
async def process_uploaded_image(painting_id : int = Path(..., ge=1)):
    try:
<<<<<<< HEAD
        # os.makedirs("temp", exist_ok=True)
        # 업로드 이미지 저장
        save_path = f"temp/{file.filename}"
        with open(save_path, "wb") as f:
            f.write(await file.read())
=======
        # 메타데이터 백엔드 저장
        backend_payload = send_metadata_to_backend(painting_id)
>>>>>>> bef946602615501d21df3445c9e4727da0508065

        # WebSocket Push
        push_painting_detected(painting_id)

        # 최종 FastAPI 응답
        return JSONResponse(content={**backend_payload, "result": "success"}, status_code=200)

    except Exception as e:
        traceback.print_exc()
<<<<<<< HEAD
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
=======
        return JSONResponse(content={"error": f"서버 오류: {e}"}, status_code=500)
>>>>>>> bef946602615501d21df3445c9e4727da0508065
