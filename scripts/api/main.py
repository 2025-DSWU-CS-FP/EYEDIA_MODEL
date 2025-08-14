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

# --- 설정 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 충돌 방지

# ✅ YOLO 모델 로드
yolo_model = YOLO("yolov8n.pt")

# ✅ CLIP 모델 로드
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ✅ FAISS 인덱스 및 메타데이터 로드
try:
    index = faiss.read_index("./data/faiss/met_text.index")
    with open("./data/faiss/met_structured_with_objects.json", "r", encoding="utf-8") as f:
        image_meta = json.load(f)
except FileNotFoundError as e:
    print(f"오류: FAISS 인덱스 또는 메타데이터 파일을 찾을 수 없습니다. 경로를 확인하세요: {e}")
    exit()


# ✅ COCO 클래스 ID → 이름 매핑
COCO_CLASSES = yolo_model.model.names

# ✅ 그림 유사 클래스 목록 정의
ART_CLASSES = ["tv", "book", "laptop", "cell phone", "remote", "keyboard", "monitor", "picture frame"]

# --- 실시간 프레임 저장을 위한 공유 변수 ---
# 스레드 간의 안전한 데이터 공유를 위해 lock을 사용합니다.
latest_frames = {
    "front": None,
    "eye": None,
    "lock": threading.Lock()
}

# --- FastAPI 앱 설정 ---
app = FastAPI()

@app.post("/api/v1/ingest")
async def ingest_images(front: UploadFile = File(...), eye: UploadFile = File(...)):
    """라즈베리파이에서 보낸 전면, 눈 이미지를 받아 전역 변수에 저장합니다."""
    try:
        # 비동기적으로 파일 내용을 읽어옴
        front_bytes = await front.read()
        eye_bytes = await eye.read()

        # OpenCV가 읽을 수 있는 numpy 배열로 변환
        front_np = np.frombuffer(front_bytes, np.uint8)
        eye_np = np.frombuffer(eye_bytes, np.uint8)
        
        front_img = cv2.imdecode(front_np, cv2.IMREAD_COLOR)
        eye_img = cv2.imdecode(eye_np, cv2.IMREAD_COLOR)

        if front_img is None or eye_img is None:
            raise ValueError("이미지 디코딩에 실패했습니다.")

        # 스레드 잠금을 사용하여 안전하게 프레임 업데이트
        with latest_frames["lock"]:
            latest_frames["front"] = front_img
            latest_frames["eye"] = eye_img
            
        return JSONResponse(content={"status": "success"}, status_code=200)
    
    except Exception as e:
        print(f"[INGEST ERROR] {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


def run_server():
    """Uvicorn 서버를 실행하는 함수"""
    print("🚀 FastAPI 서버를 시작합니다. Host: 0.0.0.0, Port: 8008")
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="warning")


def embed_crop(image: np.ndarray):
    """잘라낸 이미지를 CLIP 모델로 임베딩합니다."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(images=pil_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings[0].cpu().numpy()


# --- 메인 처리 루프 ---
def main_processing_loop():
    """라즈베리파이에서 받은 프레임을 처리하고 화면에 표시하는 메인 루프"""
    WINDOW_NAME = "🎨 Art-Like Detection (from Pi) + FAISS [q: quit]"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        # 프레임 복사본 가져오기
        with latest_frames["lock"]:
            front_frame = latest_frames["front"]
            eye_frame = latest_frames["eye"]

        # 아직 프레임이 수신되지 않았으면 잠시 대기
        if front_frame is None:
            # 대기 메시지를 표시할 검은 화면 생성
            placeholder_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(placeholder_frame, "Waiting for Raspberry Pi stream...", (50, 360), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, placeholder_frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            continue

        frame = front_frame.copy() # 처리를 위해 프레임 복사
        
        # YOLO로 객체 탐지
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
            
            # FAISS 검색
            try:
                query_vec = embed_crop(crop).reshape(1, -1)
                D, I = index.search(query_vec, k=1)
                
                matched_meta = image_meta[I[0][0]]
                match_label = f"{label}: {matched_meta['full_image_id']} ({D[0][0]:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, match_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                print(f"[FAISS/CLIP Error] {e}")


        # 눈동자 이미지를 좌상단에 오버레이
        if eye_frame is not None:
            try:
                h, w = frame.shape[:2]
                eye_small = cv2.resize(eye_frame, (w // 4, h // 4)) # 크기 조절
                sh, sw = eye_small.shape[:2]
                frame[10:10+sh, 10:10+sw] = eye_small
            except Exception as e:
                # 크기 불일치 등 오류가 발생해도 계속 진행
                pass

        cv2.imshow(WINDOW_NAME, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # FastAPI 서버를 별도의 스레드에서 실행 (데몬 스레드로 설정하여 메인이 끝나면 같이 종료)
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # 메인 스레드에서 OpenCV 처리 루프 실행
    main_processing_loop()

    print("프로그램을 종료합니다.")