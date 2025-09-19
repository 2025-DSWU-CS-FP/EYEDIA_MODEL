import os
import cv2
import numpy as np
import torch
import faiss
import json
import requests
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ✅ YOLO 모델 로드
yolo_model = YOLO("yolov8n.pt")

# ✅ CLIP 모델 로드
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ✅ FAISS 인덱스 및 메타데이터 로드
index = faiss.read_index("./data/faiss/met_text.index")
with open("./data/faiss/met_structured_with_objects.json", "r", encoding="utf-8") as f:
    image_meta = json.load(f)

# ✅ COCO 클래스 및 필터 대상
COCO_CLASSES = yolo_model.model.names
ART_CLASSES = ["tv", "book", "laptop", "cell phone", "remote", "keyboard", "monitor"]

# ✅ 전역 변수
latest_painting_id = None
latest_coords = None
command_executed = False
selected_q = "Q1" # 사용자가 선택한 영역

def embed_crop(image: np.ndarray):
    pil_image = Image.fromarray(image)
    inputs = clip_processor(images=pil_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings[0].numpy()

# ✅ 명령어 입력 먼저 받기
command = input("먼저 명령어를 입력하세요 (/detect-art 또는 /detect-area): ").strip().lower()

if command.startswith("/detect-art"):
    mode = "art"
elif command.startswith("/detect-area"):
    mode = "area"
else:
    print("❌ 잘못된 명령입니다. 프로그램을 종료합니다.")
    exit()

# ✅ 카메라 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)[0]
    found = False  # 감지 여부

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = COCO_CLASSES[cls_id]

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

        latest_painting_id = matched_id
        latest_coords = (x1, y1, x2, y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, match_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        found = True
        break  # 하나만 감지되면 바로 중단

    cv2.imshow("🎨 Art-Like Detection + FAISS", frame)
# /process-image?art_id=200001&q=Q1
    if found:
        if mode == "art":
            if latest_painting_id:
                print(f"🖼️ 감지된 그림 ID: {latest_painting_id}")
                url = f"http://3.34.240.201:8000/process-image?art_id={latest_painting_id}"
                res = requests.get(url)
                print(f"🎯 그림 인식 전송 완료: {res.status_code}")
                print(f"모델로 push by /detect-art : {url}")
                command_executed = True
                break


        elif mode == "area":
            if latest_painting_id and selected_q:
                url = f"http://3.34.240.201:8000/process-image?art_id={latest_painting_id}&q={selected_q}"
                res = requests.get(url)
                # Todo: 응시 영역 호출 함수 추가
                print(f"🗺️ {selected_q} 영역 전송 완료: {res.status_code}")
                print(f"모델로 push by /detect-area : {url}")
                command_executed = True
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

# ✅ 종료 처리
cap.release()
cv2.destroyAllWindows()

if command_executed:
    print("🎉 명령 실행 후 프로그램을 종료합니다.")
else:
    print("👋 프로그램을 정상 종료했습니다.")
