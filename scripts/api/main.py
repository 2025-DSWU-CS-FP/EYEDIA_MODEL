import os
import cv2
import numpy as np
import torch
import faiss
import json
import threading
import time
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import JSONResponse
import uvicorn
import io

# --- 1. 모델 및 DB 로드 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("모델 및 데이터를 로드하는 중입니다...")
yolo_model = YOLO("yolov8n.pt")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

try:
    index = faiss.read_index("./data/faiss/met_text.index")
    with open("./data/faiss/met_structured_with_objects.json", "r", encoding="utf-8") as f:
        image_meta = json.load(f)
except FileNotFoundError as e:
    print(f"오류: FAISS 인덱스 또는 메타데이터 파일을 찾을 수 없습니다. 경로를 확인하세요: {e}")
    exit()

print("로드 완료.")

COCO_CLASSES = yolo_model.model.names
ART_CLASSES = ["tv", "book", "laptop", "cell phone", "remote", "keyboard", "monitor", "picture frame"]

# --- 2. 스레드 간 데이터 공유를 위한 변수 ---
latest_frames = {
    "front": None,
    "eye": None,
    "processed_frame": np.zeros((720, 1280, 3), dtype=np.uint8),
    "new_frame_received": False,  # 이벤트 플래그
    "lock": threading.Lock()
}

# --- 3. FastAPI 서버 설정 (수신부) ---
app = FastAPI()

@app.post("/api/v1/ingest")
async def ingest_images(front: UploadFile = File(...), eye: UploadFile = File(...)):
    # 라즈베리파이에서 이미지를 받으면...
    front_bytes = await front.read()
    eye_bytes = await eye.read()
    front_img = cv2.imdecode(np.frombuffer(front_bytes, np.uint8), cv2.IMREAD_COLOR)
    eye_img = cv2.imdecode(np.frombuffer(eye_bytes, np.uint8), cv2.IMREAD_COLOR)

    # 공유 변수에 저장하고, 플래그를 올린다!
    with latest_frames["lock"]:
        latest_frames["front"] = front_img
        latest_frames["eye"] = eye_img
        latest_frames["new_frame_received"] = True
            
    return JSONResponse(content={"status": "success"}, status_code=200)

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="warning")

def embed_crop(image: np.ndarray):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(images=pil_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings[0].cpu().numpy()

# --- 4. 메인 처리 루프 (분석 및 시각화) ---
def main_processing_loop():
    """새로운 프레임이 도착했을 때만 분석을 실행하고, 터미널에 유사 이미지 ID만 출력하는 메인 루프"""
    WINDOW_NAME = "🎨 FAISS Similarity Search (from Pi) [q: quit]"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.putText(latest_frames["processed_frame"], "Waiting for Raspberry Pi capture...", (50, 360), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    while True:
        process_this_frame = False
        current_front_frame = None

        with latest_frames["lock"]:
            if latest_frames["new_frame_received"]:
                process_this_frame = True
                latest_frames["new_frame_received"] = False
                current_front_frame = latest_frames["front"].copy()

        if process_this_frame and current_front_frame is not None:
            print("\n📸 새로운 이미지를 수신하여 분석을 시작합니다...")
            frame = current_front_frame
            
            results = yolo_model(frame)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = COCO_CLASSES.get(cls_id, "Unknown")
                if label not in ART_CLASSES:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                try:
                    query_vec = embed_crop(crop).reshape(1, -1)
                    D, I = index.search(query_vec, k=1)
                    
                    matched_meta = image_meta[I[0][0]]
                    match_label = f"{label}: {matched_meta['full_image_id']} ({D[0][0]:.2f})"
                    
                    # ✅ [수정] 터미널에 유사 이미지 ID만 출력합니다.
                    print(f"유사 이미지: {matched_meta['full_image_id']}")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, match_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception as e:
                    print(f"[FAISS/CLIP Error] {e}")

            with latest_frames["lock"]:
                latest_frames["processed_frame"] = frame

        cv2.imshow(WINDOW_NAME, latest_frames["processed_frame"])
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
# --- 5. 프로그램 실행 ---
if __name__ == "__main__":
    # 서버 스레드와 처리 루프를 동시에 실행
    print("🚀 FastAPI 서버 스레드를 시작합니다. Host: 0.0.0.0, Port: 8008")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # 메인 스레드에서 OpenCV 창 및 처리 루프 실행
    main_processing_loop()
    
    print("프로그램을 종료합니다.")