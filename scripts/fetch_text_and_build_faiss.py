import requests, json, faiss, os
from sentence_transformers import SentenceTransformer
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from pathlib import Path
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

def fetch_met_data():
    os.makedirs("./data/met_images", exist_ok=True)
    os.makedirs("./data/faiss", exist_ok=True)

    department_id = 11  # European Paintings
    res = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects?departmentIds={department_id}")
    object_ids = res.json().get("objectIDs", [])[:10]

    embedder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

    texts, text_meta, structured_data = [], [], []
    image_embeddings, image_meta, image_data_json = [], [], []

    for i, object_id in enumerate(object_ids):
        obj = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}").json()
        if not obj.get("primaryImageSmall"):
            continue  # 이미지 없는 경우 스킵

        title = obj.get("title", "")
        artist = obj.get("artistDisplayName", "")
        date = obj.get("objectDate", "")
        medium = obj.get("medium", "")
        culture = obj.get("culture", "")
        img_url = obj["primaryImageSmall"]
        image_path = f"data/met_images/image_{object_id}.jpg"

        try:
            with open(image_path, "wb") as f:
                f.write(requests.get(img_url).content)
        except:
            continue

        summary = f"{title}. {artist}. {date}. {medium}. {culture}".strip()[:300]
        texts.append(summary)

        # 텍스트 메타
        text_meta.append({
            "id": f"{object_id}",
            "objectID": object_id,
            "title": title,
            "artist": artist,
            "summary": summary,
            "image_path": image_path
        })

        # 이미지 메타
        image_meta.append({
            "id": f"{object_id}",
            "title": title,
            "artist": artist,
            "image_path": image_path
        })

        # 이미지 데이터
        image_data_json.append({
            "image_id": object_id,
            "file_path": image_path,
            "title": title,
            "artist": artist
        })

        # 이미지 임베딩
        try:
            pil_img = Image.open(image_path).convert("RGB")
            inputs = clip_processor(images=pil_img, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = clip_model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            image_embeddings.append(emb.cpu().numpy().astype("float32")[0])
        except Exception as e:
            print(f"❗ 이미지 임베딩 실패: {object_id}, 에러: {e}")
            continue

        # 구조화된 JSON
        structured_data.append({
            "full_image_id": object_id,
            "full_image_description": summary,
            "crops": []  # 후처리에서 crop 추가 예정
        })

        print(f"✅ [{i+1}] {title} (objectID: {object_id})")

    # 텍스트 FAISS 인덱스
    text_embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
    text_index.add(text_embeddings)
    faiss.write_index(text_index, "./data/faiss/met_text.index")

    # 이미지 FAISS 인덱스
    image_index = faiss.IndexFlatIP(len(image_embeddings[0]))
    image_index.add(np.array(image_embeddings))
    faiss.write_index(image_index, "./data/faiss/met_image.index")

    # JSON 저장
    with open("./data/faiss/met_text_meta.json", "w", encoding="utf-8") as f:
        json.dump(text_meta, f, indent=2, ensure_ascii=False)

    with open("./data/faiss/met_structured_with_objects.json", "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    with open("./data/faiss/met_image_meta.json", "w", encoding="utf-8") as f:
        json.dump(image_meta, f, indent=2, ensure_ascii=False)

    with open("./data/faiss/image_data.json", "w", encoding="utf-8") as f:
        json.dump(image_data_json, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    fetch_met_data()
