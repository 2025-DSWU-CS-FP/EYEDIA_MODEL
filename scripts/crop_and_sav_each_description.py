import os
import json
import cv2
import faiss
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

# 경로 설정
DATA_PATH = "./data/faiss/met_structured_with_objects.json"
CROP_DIR = "./data/crops"
FAISS_INDEX_PATH = "./data/faiss/met_text.index"
META_PATH = "./data/faiss/met_text_meta.json"

# 디렉토리 준비
os.makedirs(CROP_DIR, exist_ok=True)

# 모델 로딩
print("YOLO 로딩 중...")
yolo = YOLO("yolov8n-seg.pt")
print("CLIP 로딩 중...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)
print(f"모델 준비 완료 (device: {device})")

# summary(객체 설명) 메타데이터 및 FAISS index 로드
print("FAISS 및 메타데이터 로딩 중...")
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
summaries = [entry["summary"] for entry in meta]
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
print("준비 완료.\n")

def embed_image(img_path):
    """이미지 임베딩(512-dim 벡터) 생성"""
    pil = Image.open(img_path).convert("RGB")
    inputs = clip_processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")

def crop_and_describe_objects(item):
    """한 이미지에서 crop 및 설명 생성"""
    image_path = item.get("image_path", "")
    if not os.path.exists(image_path):
        print(f"  [이미지 없음] {image_path}")
        return []

    image = cv2.imread(image_path)
    if image is None:
        print(f"  [이미지 읽기 실패] {image_path}")
        return []

    results = yolo(image)[0]
    n_boxes = len(results.boxes)
    print(f"  [YOLO] 객체 {n_boxes}개 감지됨")
    crops = []

    if n_boxes == 0:
        print("    [!] 객체가 감지되지 않음 (crop 생성 안됨)")
        return []

    for i, box in enumerate(results.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        crop_img = image[y1:y2, x1:x2]
        crop_id = f"{item['full_image_id']}_crop{i}"
        crop_path = os.path.join(CROP_DIR, f"{crop_id}.jpg")
        cv2.imwrite(crop_path, crop_img)

        # crop 임베딩 → 유사 summary 검색
        emb = embed_image(crop_path)
        D, I = faiss_index.search(emb, 1)
        summary = summaries[I[0][0]]

        print(f"    [{i}] crop 저장: {crop_path} | crop_description: {summary[:40]}...")
        crops.append({
            "crop_id": crop_id,
            "crop_path": crop_path,
            "crop_description": summary
        })
    return crops

def run():
    # 전체 데이터 로드 및 crop 생성
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"총 {len(data)}개 이미지에서 crop 및 설명 생성 시작!\n")
    for idx, item in enumerate(data):
        print(f"[{idx+1}/{len(data)}] 이미지: {item.get('image_path', '')}")
        item["crops"] = crop_and_describe_objects(item)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("\n✅ crop 생성 및 유사 객체 기반 설명 자동 생성 완료")

if __name__ == "__main__":
    run()
