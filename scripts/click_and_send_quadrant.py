import os
import json
import numpy as np
import torch
import cv2
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
from dotenv import load_dotenv
import requests
import faiss

# 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BACKEND_OBJECT_DESC_URL = os.getenv("BACKEND_OBJECT_DESC_URL", "http://localhost:8080/api/v1/ai/object-description")
client = OpenAI(api_key=OPENAI_API_KEY)

# === CLIP & FAISS 초기화 ===
device = "cuda" if torch.cuda.is_available() else "cpu"
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

INDEX_PATH = "data/faiss/met_text.index"
META_PATH = "data/faiss/met_text_meta.json"

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    faiss_meta = json.load(f)

# === 유틸 함수 ===
def embed_image(img: Image.Image) -> np.ndarray:
    """CLIP 임베딩 추출"""
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32").squeeze()

def gpt_docent_ko(description_list, quadrant, painting_name):
    combined = "\n".join(f"- {desc}" for desc in description_list)
    prompt = (
        f"당신은 미술관의 도슨트입니다. 아래는 {painting_name}그림의 {quadrant} 분면의 후보 객체 설명입니다.\n\n"
        f"{combined}\n\n"
        "→ 이 내용을 바탕으로 관람객에게 친절하고 감성적으로 설명해주세요. "
        "너무 기술적이거나 딱딱하지 않게 풀어주세요."
        "그림에 없는 내용은 설명하지 마세요. 그림에 대한 내용만 설명하세요."
        "해당 그림 외의 다른 그림에 대한 내용은 언급하지 말아주세요"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def get_quadrant(x, y, w, h):
    if x < w // 2 and y < h // 2:
        return "Q1"
    elif x >= w // 2 and y < h // 2:
        return "Q2"
    elif x < w // 2 and y >= h // 2:
        return "Q3"
    else:
        return "Q4"

def detect_and_search(image_path):
    if not os.path.exists(image_path):
        print(f"❗ 파일 없음: {image_path}")
        return

    image_bgr = cv2.imread(image_path)
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # YOLO 객체 탐지
    yolo = YOLO("yolov8n-seg.pt")
    results = yolo(image_pil)[0]

    if results.boxes is None or len(results.boxes) == 0:
        print("❗ 객체가 감지되지 않음")
        return

    # === 1️⃣ 대표 작품 결정 ===
    whole_emb = embed_image(image_pil).reshape(1, -1)
    distances, indices = index.search(whole_emb, 1)  # top-1만 사용
    rep_idx = indices[0][0]
    rep_meta = faiss_meta[rep_idx]
    rep_title = rep_meta.get("title", "Unknown Artwork")
    rep_artist = rep_meta.get("artist", "Unknown Artist")
    print(f"🎨 대표 작품 결정: {rep_title} by {rep_artist}")

    click_pos = {"x": -1, "y": -1}

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos["x"], click_pos["y"] = x, y

    cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("YOLO", on_click)

    while True:
        vis = image_bgr.copy()
        # 사분면 그리기
        cv2.line(vis, (w//2, 0), (w//2, h), (255,255,255), 2)
        cv2.line(vis, (0, h//2), (w, h//2), (255,255,255), 2)
        # YOLO 박스 표시
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("YOLO", vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

        if click_pos["x"] != -1:
            cx, cy = click_pos["x"], click_pos["y"]
            quadrant = get_quadrant(cx, cy, w, h)
            print(f"✅ 클릭된 분면: {quadrant}")

            # === 2️⃣ 클릭된 분면 crop 검색 (보조) ===
            selected_crops = []
            for box in results.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                if get_quadrant((x1+x2)//2, (y1+y2)//2, w, h) == quadrant:
                    crop_img = image_pil.crop((x1, y1, x2, y2))
                    crop_emb = embed_image(crop_img).reshape(1, -1)
                    distances, indices = index.search(crop_emb, 3)  # top-3 검색

                    for idx, dist in zip(indices[0], distances[0]):
                        meta = faiss_meta[idx]
                        selected_crops.append(
                            f"{meta['title']} by {meta['artist']} (score={dist:.3f})"
                        )

            # crop이 없으면 그냥 빈 설명으로
            if not selected_crops:
                selected_crops = [f"{rep_title}의 세부 객체"]

            # === 3️⃣ GPT 설명 ===
            docent_description = gpt_docent_ko(selected_crops, quadrant, rep_title)
            print(rep_title)
            print(f"\n[GPT 설명 - {quadrant}]\n{docent_description}\n")

            # === 4️⃣ 백엔드 전송 ===
            payload = {
                "quadrant": quadrant,
                "description": docent_description,
                "imageurl": image_path,
                "title": rep_title,
                "artist": rep_artist
            }
            try:
                res = requests.post(BACKEND_OBJECT_DESC_URL, json=payload)
                print(f"[POST] 전송 완료: {res.status_code} - {res.text}")
            except Exception as e:
                print(f"[ERROR] 백엔드 전송 실패: {e}")

            click_pos["x"] = -1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_search("data/met_images/monet_woman_parasol.jpg")

