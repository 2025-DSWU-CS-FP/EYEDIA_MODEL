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
import openai
from dotenv import load_dotenv

# 환경 변수 및 GPT API 세팅
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BACKEND_OBJECT_DESC_URL = os.getenv("BACKEND_OBJECT_DESC_URL", "http://localhost:8080/api/v1/ai/object-description")

def load_meta(structured_path="data/faiss/met_structured_with_objects.json"):
    path = Path(structured_path)
    if not path.exists():
        raise FileNotFoundError(f"❗ 메타 데이터 파일이 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        structured = json.load(f)
    meta = []
    for item in structured:
        for crop in item.get("crops", []):
            if crop.get("crop_id") and crop.get("crop_description"):
                meta.append({
                    "crop_id": crop["crop_id"],
                    "crop_description": crop["crop_description"],
                    "title": item.get("full_image_title", ""),
                    "artist": item.get("full_image_artist", ""),
                    "paintingId": item.get("full_image_id", ""),
                })
    return meta

def find_clicked_object(masks, x, y, img_shape, mask_shape):
    scale_x = mask_shape[1] / img_shape[1]
    scale_y = mask_shape[0] / img_shape[0]
    mx = int(x * scale_x)
    my = int(y * scale_y)
    if 0 <= my < mask_shape[0] and 0 <= mx < mask_shape[1]:
        for idx, mask in enumerate(masks):
            if mask[my][mx] > 0:
                return idx
    return -1

def gpt_docent_ko(crop_description: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    openai.api_key = OPENAI_API_KEY
    prompt = (
        "당신은 미술관의 도슨트입니다. 아래 설명을 바탕으로 관람객에게 친절하게 설명해주세요. "
        "너무 딱딱하거나 기술적이지 않게 풀어서 말해주세요.\n\n"
        f"[작품 설명]: {crop_description}\n\n"
        "→ 도슨트 스타일로 설명해주세요:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()

def detect_and_send_crop(image_path):
    print(f"▶ 이미지 경로: {image_path}")
    if not os.path.exists(image_path):
        print("❗ 이미지 경로가 존재하지 않음")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("❗ 이미지 로드 실패")
        return
    image = cv2.resize(image, (1280, 720))

    yolo = YOLO("yolov8n-seg.pt")
    results = yolo(image, conf=0.3)[0]

    if results.masks is None or results.boxes is None:
        print("❗ 감지된 객체가 없습니다.")
        return

    masks_np = results.masks.data.cpu().numpy()
    mask_shape = masks_np[0].shape

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip.to(device)

    crop_meta = load_meta()
    def embed(img):
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype("float32").squeeze()

    selected_idx = -1
    crop_result = {}

    def on_click(event, x, y, flags, param):
        nonlocal selected_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            img_shape = (vis.shape[0], vis.shape[1])
            idx = find_clicked_object(masks_np, x, y, img_shape, mask_shape)
            if idx >= 0:
                selected_idx = idx

    cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO", 1280, 720)
    cv2.setMouseCallback("YOLO", on_click)

    while True:
        vis = image.copy()
        for i, mask_np in enumerate(masks_np):
            mask_resized = cv2.resize(mask_np, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
            binary_mask = mask_resized.astype(bool)
            color = (0, 255, 0) if i == selected_idx else (0, 0, 255)
            vis[binary_mask] = color
        cv2.imshow("YOLO", vis)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("🛑 종료 요청")
            cv2.destroyAllWindows()
            return

        if selected_idx >= 0:
            x1, y1, x2, y2 = results.boxes.xyxy[selected_idx].int().tolist()
            cropped = image[y1:y2, x1:x2]
            pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            vec = embed(pil_crop)

            crops_dir = Path("data/crops/")
            crop_files = list(crops_dir.glob("*.jpg"))
            best_match = None
            max_score = -float('inf')
            for meta in crop_meta:
                crop_img_path = f"data/crops/{meta['crop_id']}.jpg"
                if not os.path.exists(crop_img_path):
                    continue
                try:
                    crop_img = Image.open(crop_img_path)
                    desc_embedding = embed(crop_img)
                    if desc_embedding.shape != vec.shape:
                        continue
                    score = float(np.dot(vec, desc_embedding))
                    if score > max_score:
                        max_score = score
                        best_match = meta
                except Exception:
                    continue

            if best_match is None or max_score < 0.0:
                print("❌ 유사한 crop을 찾지 못했습니다.")
                selected_idx = -1
                continue

            # 1️⃣ GPT 증강
            docent_description = gpt_docent_ko(best_match["crop_description"])
            print(f"[GPT 도슨트 설명]\n{docent_description}")

            # 2️⃣ 백엔드로 전송
            payload = {
                "objectId": best_match["crop_id"],
                "description": docent_description,
                "imageurl": image_path,
                "title": best_match.get("title", ""),
                "artist": best_match.get("artist", ""),
                "paintingId": best_match["paintingId"]
            }
            print(f"[POST] {BACKEND_OBJECT_DESC_URL}\n{json.dumps(payload, ensure_ascii=False, indent=2)}")
            try:
                res = requests.post(BACKEND_OBJECT_DESC_URL, json=payload)
                print(f"[INFO] 백엔드 응답: {res.status_code} - {res.text}")
            except Exception as e:
                print(f"[ERROR] 백엔드 요청 실패: {e}")

            cv2.destroyAllWindows()
            break

    return

if __name__ == "__main__":
    detect_and_send_crop("data/met_images/image_436244.jpg")
