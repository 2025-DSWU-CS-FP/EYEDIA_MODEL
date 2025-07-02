import requests, json, faiss, os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

def safe_text(s):
    return s.encode("ascii", "ignore").decode("ascii")  # ASCII 이외 문자 제거


# def enrich_description(description):
#     prompt = f"이 문장을 바탕으로 예술 작품 객체에 대한 자세한 한국어 설명을 생성해줘:\n'{description}'"
#     try:
#         response = requests.post(LLM_API_URL, json={"prompt": prompt, "max_tokens": 100})
#         response.raise_for_status()
#         return response.json().get("output", description)  # 실패 시 원본 유지
#     except Exception as e:
#         print(f"❗ LLM 요청 실패: {e}")
#         return description



def fetch_crop_based_faiss():
    os.makedirs("./data/faiss", exist_ok=True)

    # 🔄 CLIP 모델 로딩
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)

    # 📄 crop이 들어 있는 구조화 데이터 로딩
    structured_path = "./data/faiss/met_structured_with_objects.json"
    if not os.path.exists(structured_path):
        print("❗ 구조화 JSON이 존재하지 않습니다.")
        return

    with open(structured_path, "r", encoding="utf-8") as f:
        structured_data = json.load(f)

    crop_texts = []
    crop_meta = []

    for item in structured_data:
        full_image_id = item["full_image_id"]
        for crop in item.get("crops", []):
            crop_id = crop["crop_id"]
            crop_description = crop["crop_description"]
            crop_texts.append(crop_description)
            crop_meta.append({
                "crop_id": crop_id,
                "full_image_id": full_image_id,
                "crop_description": crop_description
            })

    if not crop_texts:
        print("❗ crop_description 데이터가 없습니다.")
        return

    # 🧠 텍스트 임베딩 수행
    with torch.no_grad():
        inputs = processor(text=crop_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        embeddings = clip_model.get_text_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        embeddings_np = embeddings.cpu().numpy().astype("float32")

    print(f"✅ crop 기반 임베딩 차원: {embeddings_np.shape[1]}")
    print(f"📦 총 crop 개수: {len(crop_meta)}")

    # ✅ FAISS 인덱스 생성 및 저장
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    faiss.write_index(index, "./data/faiss/met_crop.index")

    # 메타 정보 저장
    with open("./data/faiss/met_crop_meta.json", "w", encoding="utf-8") as f:
        json.dump(crop_meta, f, indent=2, ensure_ascii=False)

    print("🎉 crop 기반 FAISS 인덱스 및 메타 저장 완료")

if __name__ == "__main__":
    fetch_crop_based_faiss()
