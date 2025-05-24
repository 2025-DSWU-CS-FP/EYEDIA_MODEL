from ultralytics import YOLO
from sentence_transformers import SentenceTransformer, util
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
import cv2
from PIL import Image
import os
import json
import faiss
from pathlib import Path

# set
image_path = "./data/raw_images/image-7.jpg"
file_path = "./data/faiss/image_description_en.json"
image_filename = os.path.basename(image_path)


# 2. 모델 로딩
yolo_model = YOLO("yolov8n-seg.pt")
sbert = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 3. image_data.json에서 설명 불러오기
with open(file_path, "r", encoding="utf-8") as f:
    loaded_data = json.load(f)

import cv2
image = cv2.imread(image_path)

full_desc_text = loaded_data[0]["full_image_description"]
description_sentences = [s.strip() for s in full_desc_text.replace("**", "").split(".") if s.strip()]
sentence_embeddings = sbert.encode(description_sentences, convert_to_tensor=True)

# 4. 이미지 로딩 및 YOLO 감지
image = cv2.imread(image_path)
results = yolo_model(image, conf=0.3)[0]
os.makedirs("crops", exist_ok=True)

# 5. 라벨별 설명 추출
labels = list(set([yolo_model.names[int(cls)] for cls in results.boxes.cls]))
description_map = {}
for label in labels:
    query_embedding = sbert.encode(label, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
    top_indices = torch.topk(cos_scores, k=min(2, len(cos_scores))).indices.tolist()
    description_map[label] = " ".join([description_sentences[i] for i in top_indices])

# 6. crop 처리 및 저장
crop_list = []
clip_vectors, clip_ids, clip_descriptions = [], [], []
label_count = {}

if results.masks is not None:
    masks = results.masks.data.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    for i, mask in enumerate(masks):
        cls_idx = classes[i]
        label = yolo_model.names[cls_idx]
        label_count[label] = label_count.get(label, 0) + 1

        binary_mask = (mask > 0.5).astype(np.uint8)
        binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))
        binary_mask_3ch = np.stack([binary_mask]*3, axis=-1)
        masked_image = np.where(binary_mask_3ch == 1, image, 255)

        x_indices, y_indices = np.where(binary_mask == 1)
        if x_indices.size == 0 or y_indices.size == 0:
            continue

        x_min, x_max = np.min(y_indices), np.max(y_indices)
        y_min, y_max = np.min(x_indices), np.max(x_indices)
        cropped_object = masked_image[y_min:y_max, x_min:x_max]

        if cropped_object.size == 0:
            continue

        crop_filename = f"crops/{label}_{label_count[label]}.jpg"
        cv2.imwrite(crop_filename, cropped_object)

        crop_description = description_map[label]

        crop_list.append({
            "crop_id": crop_filename,
            "crop_description": crop_description
        })

        try:
            pil_image = Image.open(crop_filename).convert("RGB")
            inputs = clip_processor(images=pil_image, return_tensors="pt").to("cpu")
            with torch.no_grad():
                emb = clip_model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            clip_vectors.append(emb.cpu().numpy().astype("float32"))
            clip_ids.append(crop_filename)
            clip_descriptions.append(crop_description)
        except Exception as e:
            print(f"❗ {crop_filename} 처리 오류: {e}")
else:
    print("❗ Segmentation 결과 없음")

# 7. image_data.json 저장 (리스트 형태)
structured_data = [{
    "full_image_id": image_filename,
    "full_image_description": full_desc_text,
    "crops": crop_list
}]

with open("image_data.json", "w", encoding="utf-8") as f:
    json.dump(structured_data, f, indent=2, ensure_ascii=False)
print("✅ image_data.json 저장 완료")

# 8. FAISS 인덱스 및 image_meta.json 저장 (리스트 구조)
os.makedirs("faiss", exist_ok=True)
index_save_path = "./faiss/image_clip_v2.index"
meta_save_path = "./faiss/image_meta_v2.json"

if clip_vectors:
    matrix = np.vstack(clip_vectors)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, index_save_path)
    print(f"✅ FAISS 인덱스 저장 완료: {index_save_path}")

    structured_meta = [{
        "full_image_id": image_filename,
        "full_image_description": full_desc_text,
        "crops": [
            {
                "crop_id": cid,
                "crop_description": cdesc
            }
            for cid, cdesc in zip(clip_ids, clip_descriptions)
        ]
    }]
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(structured_meta, f, indent=2, ensure_ascii=False)
    print(f"✅ image_meta.json 저장 완료")
else:
    print("❗ 저장할 CLIP 벡터가 없습니다.")
