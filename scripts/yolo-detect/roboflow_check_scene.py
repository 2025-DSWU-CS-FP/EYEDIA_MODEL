import os
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import faiss
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 충돌 방지
# ✅ Roboflow client 설정
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")  # 여기에 자신의 API 키를 입력하세요
)

# ✅ CLIP 모델 로드
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ✅ FAISS index 및 메타데이터 로드
index = faiss.read_index("./data/faiss/met_text.index")
with open("./data/faiss/met_structured_with_objects.json", "r", encoding="utf-8") as f:
    image_meta = json.load(f)

def embed_crop(image: np.ndarray):
    pil_image = Image.fromarray(image)
    inputs = clip_processor(images=pil_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings[0].numpy()

# ✅ 실시간 처리 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 임시 저장 후 Roboflow API 요청
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, frame)
    result = client.infer(temp_path, model_id="artwork-set/9")

    for pred in result["predictions"]:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        # Crop 및 임베딩
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue

        query_vec = embed_crop(crop).reshape(1, -1)
        D, I = index.search(query_vec, k=1)

        matched_id = image_meta[I[0][0]]["full_image_id"]
        label = f"Match: {matched_id} ({D[0][0]:.2f})"

        # 박스 및 매칭 결과 시각화
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Live Detection + FAISS", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
