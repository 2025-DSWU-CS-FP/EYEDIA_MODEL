import os
import json
import faiss
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# 기존 Met JSON 경로
META_PATH = "data/faiss/met_text_meta.json"
STRUCT_PATH = "data/faiss/met_structured_with_objects.json"
INDEX_PATH = "data/faiss/met_text.index"

# Custom 추가할 이미지
custom_images = [
    {
        "objectID": 200001,
        "title": "The Dance Class (무용수업)",
        "artist": "Edgar Degas",
        "date": "1874",
        "medium": "Oil on Canvas",
        "culture": "French",
        "image_path": "data/met_images/image_20001.jpg"
    },
    {
        "objectID": 200002,
        "title": "Woman with a Parasol (양산을 쓴 여인)",
        "artist": "Claude Monet",
        "date": "1875",
        "medium": "Oil on Canvas",
        "culture": "French",
        "image_path": "data/met_images/image_20002.jpg"
    },
    {
        "objectID": 200003,
        "title": "The Card Sharp with the Ace of Diamonds (다이아몬드 에이스를 가진 사기꾼)",
        "artist": "Georges de La Tour",
        "date": "1630",
        "medium": "Oil on Canvas",
        "culture": "French",
        "image_path": "data/met_images/image_20003.jpg"
    }
]

def append_custom_to_met(custom_images):
    # === 기존 JSON 로드 ===
    with open(META_PATH, "r", encoding="utf-8") as f:
        faiss_meta = json.load(f)
    with open(STRUCT_PATH, "r", encoding="utf-8") as f:
        structured_data = json.load(f)

    # === 기존 FAISS 로드 ===
    index = faiss.read_index(INDEX_PATH)

    # === CLIP 준비 ===
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)

    def embed_text(text):
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = clip_model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype("float32")

    # === Custom 이미지 처리 ===
    new_embeddings = []

    for item in custom_images:
        summary = f"{item['title']}. {item['artist']}. {item['date']}. {item['medium']}. {item['culture']}"[:300]

        # FAISS 메타
        faiss_meta.append({
            "id": str(item["objectID"]),
            "objectID": item["objectID"],
            "title": item["title"],
            "artist": item["artist"],
            "summary": summary,
            "image_path": item["image_path"]
        })

        # 구조화 JSON
        structured_data.append({
            "full_image_id": item["objectID"],
            "full_image_description": summary,
            "image_path": item["image_path"],
            "crops": []
        })

        # CLIP 임베딩
        emb = embed_text(summary)[0]
        new_embeddings.append(emb)

        print(f"✅ 추가 완료: {item['title']} ({item['objectID']})")

    # === FAISS에 벡터 추가 ===
    new_embeddings = np.stack(new_embeddings).astype("float32")
    index.add(new_embeddings)

    # === 저장 ===
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(faiss_meta, f, indent=2, ensure_ascii=False)
    with open(STRUCT_PATH, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    print("✅ Custom 이미지가 기존 Met 데이터에 병합 완료!")

if __name__ == "__main__":
    append_custom_to_met(custom_images)
