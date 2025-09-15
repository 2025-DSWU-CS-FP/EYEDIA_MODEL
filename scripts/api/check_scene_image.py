import torch
import faiss
import json
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ✅ 모델 초기화
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# ✅ FAISS 및 메타데이터 로드
faiss_index_path = Path("./data/faiss/met_text.index")
meta_path = Path("./data/faiss/met_structured_with_objects.json")

faiss_index = faiss.read_index(str(faiss_index_path))

with open(meta_path, "r", encoding="utf-8") as f:
    met_image_meta = json.load(f)

def check_scene_image(image_path: str, threshold: float = 0.35) -> dict:
    """
    주어진 scene 이미지가 벡터 DB에 존재하는지 확인
    """
    image = Image.open(image_path).convert("RGB")
    image = preprocess_scene_image(image)

    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    vec = embedding.cpu().numpy().astype("float32")

    D, I = faiss_index.search(vec, k=1)
    distance = float(D[0][0])
    matched_idx = int(I[0][0])

    if distance < threshold:
        match_info = met_image_meta[matched_idx]
        return {
            "exists": True,
            "matched_image_id": match_info["full_image_id"],
            "matched_image_path": match_info["image_path"],
            "matched_description": match_info["full_image_description"],
            "distance": distance
        }
    else:
        return {"exists": False, "distance": distance}
def preprocess_scene_image(image: Image.Image) -> Image.Image:
    w, h = image.size
    crop_ratio = 0.8
    left = int(w * (1 - crop_ratio) / 2)
    top = int(h * (1 - crop_ratio) / 2)
    right = int(w * (1 + crop_ratio) / 2)
    bottom = int(h * (1 + crop_ratio) / 2)
    return image.crop((left, top, right, bottom))

# ✅ 테스트
if __name__ == "__main__":
    
    result = check_scene_image("./data/scene_images/scene_436240.JPG")
    print(result)