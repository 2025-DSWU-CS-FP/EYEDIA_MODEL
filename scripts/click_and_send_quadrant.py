import os
import json
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import cv2
import requests
from openai import OpenAI
from dotenv import load_dotenv

# 환경 변수 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BACKEND_OBJECT_DESC_URL = os.getenv("BACKEND_OBJECT_DESC_URL", "http://localhost:8080/api/v1/ai/object-description")
client = OpenAI(api_key=OPENAI_API_KEY)

# GPT 설명 생성 함수
def gpt_docent_ko(description_list, quadrant):
    combined = "\n".join(f"- {desc}" for desc in description_list)
    prompt = (
        f"당신은 미술관의 도슨트입니다. 아래는 {quadrant} 분면에 있는 객체 설명들입니다.\n\n"
        f"{combined}\n\n"
        "→ 이 내용을 바탕으로 관람객에게 친절하고 감성적으로 설명해주세요. 너무 기술적이거나 딱딱하지 않게 풀어주세요."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    docent_text = response.choices[0].message.content.strip()
    return docent_text

# 분면 판별 함수
def get_quadrant(x, y, w, h):
    if x < w // 2 and y < h // 2:
        return "Q1"  # top-left
    elif x >= w // 2 and y < h // 2:
        return "Q2"  # top-right
    elif x < w // 2 and y >= h // 2:
        return "Q3"  # bottom-left
    else:
        return "Q4"  # bottom-right

# crop 메타데이터 로드
def load_crop_meta(path="data/faiss/met_structured_with_objects.json"):
    with open(path, "r", encoding="utf-8") as f:
        structured = json.load(f)
    meta = []
    for item in structured:
        for crop in item.get("crops", []):
            if crop.get("crop_id") and crop.get("crop_description"):
                cx = crop.get("center_x")
                cy = crop.get("center_y")
                if cx is not None and cy is not None:
                    meta.append({
                        "crop_id": crop["crop_id"],
                        "crop_description": crop["crop_description"],
                        "center_x": cx,
                        "center_y": cy,
                        "paintingId": item.get("full_image_id", ""),
                        "title": item.get("full_image_title", ""),
                        "artist": item.get("full_image_artist", ""),
                    })
    return meta

# 메인 실행 함수
def detect_and_send_quadrant(image_path):
    if not os.path.exists(image_path):
        print(f"❗ 파일 없음: {image_path}")
        return

    # 이미지 로딩
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print("❗ 이미지 로드 실패")
        return
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # YOLO 모델 로드
    yolo = YOLO("yolov8n-seg.pt")
    results = yolo(image_pil)[0]

    if results.boxes is None or len(results.boxes) == 0:
        print("❗ 객체가 감지되지 않음")
        return

    # 중심선 그리기
    def draw_quadrants(img):
        center_x, center_y = w // 2, h // 2
        cv2.line(img, (center_x, 0), (center_x, h), (255, 255, 255), 2)
        cv2.line(img, (0, center_y), (w, center_y), (255, 255, 255), 2)
        return img

    # crop 메타 로드
    crop_meta = load_crop_meta()

    # 클릭 이벤트 처리
    click_pos = {"x": -1, "y": -1}
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos["x"], click_pos["y"] = x, y

    cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("YOLO", on_click)

    while True:
        vis = image_bgr.copy()
        vis = draw_quadrants(vis)
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("YOLO", vis)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

        if click_pos["x"] != -1:
            # 클릭된 분면 처리
            cx, cy = click_pos["x"], click_pos["y"]
            quadrant = get_quadrant(cx, cy, w, h)
            print(f"✅ 클릭된 분면: {quadrant}")

            selected_crops = [
                crop for crop in crop_meta
                if get_quadrant(crop["center_x"], crop["center_y"], w, h) == quadrant
            ]

            if not selected_crops:
                print("❌ 해당 분면에 객체 설명이 없습니다.")
                click_pos["x"] = -1
                continue

            descriptions = [c["crop_description"] for c in selected_crops]
            docent_description = gpt_docent_ko(descriptions, quadrant)
            print(f"\n[GPT 설명 - {quadrant}]\n{docent_description}\n")

            rep_crop = selected_crops[0]
            payload = {
                "quadrant": quadrant,
                "description": docent_description,
                "imageurl": image_path,
                "title": rep_crop["title"],
                "artist": rep_crop["artist"],
                "paintingId": rep_crop["paintingId"]
            }

            try:
                res = requests.post(BACKEND_OBJECT_DESC_URL, json=payload)
                print(f"[POST] 전송 완료: {res.status_code} - {res.text}")
            except Exception as e:
                print(f"[ERROR] 백엔드 전송 실패: {e}")

            # 클릭 초기화
            click_pos["x"] = -1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_send_quadrant("data/met_images/image_436244.jpg")
