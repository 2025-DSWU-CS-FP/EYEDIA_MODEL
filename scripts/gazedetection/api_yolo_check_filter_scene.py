# eyedia_pipeline_post.py
import os
import cv2
import numpy as np
import torch
import faiss
import json
import requests
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------[ 설정 ]-----------------
YOLO_WEIGHTS = "yolov8n.pt"
CLIP_ID = "openai/clip-vit-base-patch32"
FAISS_INDEX = "./data/faiss/met_text.index"
FAISS_META  = "./data/faiss/met_structured_with_objects.json"
REQUEST_TIMEOUT = 5

# 관심 클래스
ART_CLASSES = ["tv", "book", "laptop", "cell phone", "remote", "keyboard", "monitor"]

# gaze 모듈 (옵션)
try:
    import gazedetection as gd
    HAS_GAZE = True
except Exception:
    HAS_GAZE = False

# -----------------[ 모델 로드 ]-----------------
yolo_model = YOLO(YOLO_WEIGHTS)
clip_model = CLIPModel.from_pretrained(CLIP_ID)
clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)

index = faiss.read_index(FAISS_INDEX)
with open(FAISS_META, "r", encoding="utf-8") as f:
    image_meta = json.load(f)

COCO_CLASSES = yolo_model.model.names

# -----------------[ 헬퍼 ]-----------------
def embed_crop(image_bgr: np.ndarray) -> np.ndarray:
    """CLIP 임베딩(L2 정규화, float32)"""
    if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = image_bgr
    pil = Image.fromarray(rgb)
    inputs = clip_processor(images=pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb[0].cpu().numpy().astype("float32")

def get_gaze_q(frame_bgr: np.ndarray, fallback_q: str) -> str:
    """가이즈 가능 시 자동 Q 업데이트"""
    if not HAS_GAZE:
        return fallback_q
    try:
        zone = gd.predict_zone(frame_bgr)   # 1..4 또는 None
        if zone in (1,2,3,4):
            return f"Q{zone}"
    except Exception as e:
        print(f"[WARN] gaze 예측 실패: {e}")
    return fallback_q

def draw_box(frame, box, color, text):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, text, (x1, max(0,y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def detect_first_art(frame_bgr):
    """관심 클래스 첫 객체만 사용하여 art_id 반환"""
    results = yolo_model(frame_bgr)[0]
    for b in results.boxes:
        cls_id = int(b.cls[0])
        label = COCO_CLASSES[cls_id]
        if label not in ART_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        qv = embed_crop(crop).reshape(1, -1)
        D, I = index.search(qv, k=1)
        idx = int(I[0][0])
        art_id = image_meta[idx]["full_image_id"]
        score = float(D[0][0])
        return art_id, (x1,y1,x2,y2), label, score
    return None, None, None, None

# -----------------[ 모드별 실행 ]-----------------
def run_art_mode():
    print("✅ ART 모드 시작 (/detect-art)")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            art_id, box, label, score = detect_first_art(frame)
            if art_id:
                draw_box(frame, box, (255,0,0), f"{label}: {art_id} ({score:.2f})")
                cv2.putText(frame, "Mode: ART", (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                cv2.imshow("EYEDIA - ART", frame)

                print(f"🖼️ 감지된 그림 ID: {art_id}")
                url = f"http://3.34.240.201:8000/process-image?art_id={art_id}"
                try:
                    res = requests.post(url, timeout=REQUEST_TIMEOUT)
                    print(f"🎯 그림 인식 전송 완료: {res.status_code}")
                    print(f"모델로 push by /detect-art : {url}")
                except requests.RequestException as e:
                    print(f"[WARN] 요청 실패: {e}")
                break
            else:
                cv2.putText(frame, "Mode: ART (검색중...)", (10,24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                cv2.imshow("EYEDIA - ART", frame)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                print("👋 사용자 종료(Q).")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("[DONE] ART mode 종료")

def run_area_mode():
    print("✅ AREA 모드 시작 (/detect-area)")
    selected_q = "Q1"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            new_q = get_gaze_q(frame, selected_q)
            if new_q != selected_q:
                selected_q = new_q

            art_id, box, label, score = detect_first_art(frame)
            if art_id:
                draw_box(frame, box, (0,255,255), f"{label}: {art_id} ({score:.2f})")
                cv2.putText(frame, f"Mode: AREA  Q:{selected_q}", (10,24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.imshow("EYEDIA - AREA", frame)

                print(f"🖼️ 감지된 그림 ID: {art_id}")
                url = f"http://3.34.240.201:8000/process-image?art_id={art_id}&q={selected_q}"
                try:
                    res = requests.post(url, timeout=REQUEST_TIMEOUT)
                    print(f"🗺️ {selected_q} 영역 전송 완료: {res.status_code}")
                    print(f"모델로 push by /detect-area : {url}")
                except requests.RequestException as e:
                    print(f"[WARN] 요청 실패: {e}")
                break
            else:
                cv2.putText(frame, f"Mode: AREA  Q:{selected_q} (검색중...)", (10,24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.imshow("EYEDIA - AREA", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 사용자 종료(Q).")
                break
            if key in (ord('1'), ord('2'), ord('3'), ord('4')):
                selected_q = f"Q{chr(key)}"
                print(f"[MANUAL] Q → {selected_q}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("[DONE] AREA mode 종료")

# -----------------[ 엔트리포인트 ]-----------------
if __name__ == "__main__":
    cmd = input("먼저 명령어를 입력하세요 (/detect-art 또는 /detect-area): ").strip().lower()
    if cmd.startswith("/detect-art"):
        run_art_mode()
    elif cmd.startswith("/detect-area"):
        run_area_mode()
    else:
        print("[ERROR] 알 수 없는 명령입니다. /detect-art 또는 /detect-area 중 하나를 입력하세요.")
