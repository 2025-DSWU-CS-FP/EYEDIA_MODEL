import requests, json, faiss, os
from sentence_transformers import SentenceTransformer

def fetch_met_data():
    os.makedirs("./data/met_images", exist_ok=True)
    os.makedirs("./data/faiss", exist_ok=True)

    department_id = 11  # European Paintings
    res = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects?departmentIds={department_id}")
    object_ids = res.json().get("objectIDs", [])[:10]

    embedder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    texts, faiss_meta, structured_data = [], [], []

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

        # 전체 설명 (FAISS 임베딩용)
        summary = f"{title}. {artist}. {date}. {medium}. {culture}".strip()[:300]
        texts.append(summary)

        # FAISS 메타 저장용
        faiss_meta.append({
            "id": f"{object_id}",
            "objectID": object_id,
            "title": title,
            "artist": artist,
            "summary": summary,
            "image_path": image_path
        })

        # 구조화된 JSON
        structured_data.append({
            "full_image_id": object_id,
            "full_image_description": summary,
            "crops": []  # 후처리에서 crop_id, crop_description 추가 예정
        })

        print(f"✅ [{i+1}] {title} (objectID: {object_id})")

    # FAISS 인덱스 저장
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "./data/faiss/met_text.index")

    # 메타 정보 저장 
    with open("./data/faiss/met_text_meta.json", "w", encoding="utf-8") as f:
        json.dump(faiss_meta, f, indent=2, ensure_ascii=False)

    with open("./data/faiss/met_structured_with_objects.json", "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    fetch_met_data()
