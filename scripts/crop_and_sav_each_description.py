import os, json, cv2, numpy as np
from PIL import Image
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer, util

def crop_and_update_structured():
    # 경로 설정
    image_dir = Path("./data/met_images")
    crop_dir = Path("./data/cropped_images")
    faiss_dir = Path("./data/faiss")
    structured_path = faiss_dir / "met_structured.json"
    output_path = faiss_dir / "met_structured_with_objects.json"

    crop_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로딩
    yolo = YOLO("yolov8n-seg.pt")
    sbert = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

    # 메타 데이터 로딩
    with open(structured_path, "r", encoding="utf-8") as f:
        structured_data = json.load(f)

    for item in structured_data:
        full_image_id = item["full_image_id"]
        img_path = image_dir / f"image_{full_image_id}.jpg"
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"⚠️ 이미지 로드 실패: {img_path}")
            continue

        full_desc_text = item.get("full_image_description", "")
        if not full_desc_text:
            print(f"❗ 설명 없음: {full_image_id}")
            continue

        description_sentences = [s.strip() for s in full_desc_text.replace("**", "").split(".") if s.strip()]
        sentence_embeddings = sbert.encode(description_sentences, convert_to_tensor=True)

        img_resized = cv2.resize(img, (1280, 720))
        results = yolo(img_resized, conf=0.3)[0]

        if not results.masks:
            continue

        item["crops"] = []
        class_names = results.names
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        labels = list(set([class_names[idx] for idx in classes]))
        description_map = {}
        for label in labels:
            query_embedding = sbert.encode(label, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
            top_indices = torch.topk(cos_scores, k=min(2, len(cos_scores))).indices.tolist()
            description_map[label] = " ".join([description_sentences[i] for i in top_indices])

        label_count = {}
        for i, (mask, cls_idx) in enumerate(zip(masks, classes)):
            bin_mask = (mask > 0.5).astype(np.uint8)
            ys, xs = np.where(bin_mask == 1)
            if ys.size == 0 or xs.size == 0:
                continue

            cropped = img_resized[np.min(ys):np.max(ys), np.min(xs):np.max(xs)]
            label = class_names[cls_idx]
            label_count[label] = label_count.get(label, 0) + 1
            crop_name = f"image_{full_image_id}_crop{label_count[label]}.jpg"
            crop_path = crop_dir / crop_name
            cv2.imwrite(str(crop_path), cropped)

            description = description_map[label]

            item["crops"].append({
                "crop_id": crop_name,
                "crop_description": description
            })

    # 결과 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 객체 정보가 {output_path.name} 에 저장되었습니다.")
    print(f"✅ 총 {sum(len(i['crops']) for i in structured_data)}개의 crop 객체가 생성되었습니다.")

if __name__ == "__main__":
    crop_and_update_structured()
