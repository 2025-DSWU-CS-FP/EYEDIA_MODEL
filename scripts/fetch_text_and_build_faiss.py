import requests, json, faiss, os
from sentence_transformers import SentenceTransformer

def fetch_met_data():
    os.makedirs("data/met_images", exist_ok=True)
    os.makedirs("data/faiss", exist_ok=True)

    department_id = 11  # European Paintings
    res = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects?departmentIds={department_id}")
    object_ids = res.json().get("objectIDs", [])[:10]

    embedder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    texts, meta = [], []

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
        image_path = f"data/met_images/image_{i}.jpg"

        try:
            with open(image_path, "wb") as f:
                f.write(requests.get(img_url).content)
        except:
            continue

        summary = f"{title}. {artist}. {date}. {medium}. {culture}".strip()[:300]
        texts.append(summary)
        meta.append({
            "id": f"item_{i}",
            "objectID": object_id,
            "title": title,
            "artist": artist,
            "summary": summary,
            "image_path": image_path
        })
        print(f"✅ [{i+1}] {title}")

    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "data/faiss/met_text.index")

    with open("data/faiss/met_text_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    fetch_met_data()
