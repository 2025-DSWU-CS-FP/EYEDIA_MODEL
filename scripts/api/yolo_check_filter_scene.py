import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 충돌 방지

import cv2
import numpy as np
import torch
import faiss
import json
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

# ✅ YOLO 모델 로드
yolo_model = YOLO("yolov8n.pt")  # yolov8n-seg.pt 도 가능

# ✅ CLIP 모델 로드
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ✅ FAISS 인덱스 및 메타데이터 로드
index = faiss.read_index("./data/faiss/met_text.index")
with open("./data/faiss/met_structured_with_objects.json", "r", encoding="utf-8") as f:
    image_meta = json.load(f)

# ✅ COCO 클래스 ID → 이름 매핑
COCO_CLASSES = yolo_model.model.names

# ✅ 그림 유사 클래스 목록 정의 (필요시 수정)
ART_CLASSES = ["tv", "book", "laptop", "cell phone", "remote", "keyboard", "monitor"]

def embed_crop(image: np.ndarray):
    pil_image = Image.fromarray(image)
    inputs = clip_processor(images=pil_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings[0].numpy()

# ✅ 실시간 카메라 처리 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = COCO_CLASSES[cls_id]

        # 🔍 그림 유사한 클래스만 필터링
        if label not in ART_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue

        query_vec = embed_crop(crop).reshape(1, -1)
        D, I = index.search(query_vec, k=1)

        matched_id = image_meta[I[0][0]]["full_image_id"]
        match_label = f"{label}: {matched_id} ({D[0][0]:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, match_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("🎨 Art-Like Detection + FAISS", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
