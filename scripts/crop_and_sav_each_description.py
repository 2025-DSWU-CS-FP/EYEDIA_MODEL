import os
import json
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv
from openai import OpenAI

# ========== 환경 설정 ==========
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❗ OPENAI_API_KEY가 설정되지 않았습니다.")
client = OpenAI(api_key=api_key)

# ========== 모델 및 경로 ==========
yolo = YOLO("yolov8n-seg.pt")
IMAGE_DIR = "data/met_images"
JSON_PATH = "data/faiss/met_structured_with_objects.json"

# ========== GPT 도슨트 설명 생성 ==========
def generate_docent_description(label: str, image_description: str) -> str:
    prompt = f"""
당신은 예술작품을 설명하는 한국어 도슨트입니다.

아래 이미지는 '{label}' 라는 객체를 포함하고 있으며, 전체 그림 설명은 다음과 같습니다:
"{image_description}"

이 객체에 대해 직관적이고 감성적인 도슨트 설명을 작성해 주세요.
말투는 구어체로, 쉬운 단어와 비유를 사용하고 너무 짧지 않게 설명해 주세요.
'객체'나 '레이블'같은 표현은 쓰지 마세요. 무조건 한국어로 해주세요.

도슨트 설명:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❗ GPT 오류: {e}")
        return "설명 생성 실패"

# ========== 전체 이미지 처리 ==========
def process_all_images():
    if not os.path.exists(JSON_PATH):
        print(f"❌ JSON 파일이 존재하지 않습니다: {JSON_PATH}")
        return

    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ JSON 로딩 실패: {e}")
        return

    for item in data:
        image_id = str(item.get("full_image_id"))
        image_path = os.path.join(IMAGE_DIR, f"image_{image_id}.jpg")
        full_desc = item.get("full_image_description", "")

        if not os.path.exists(image_path):
            print(f"❌ 이미지 없음: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 이미지 로딩 실패: {image_path}")
            continue

        results = yolo(image, conf=0.3)[0]
        if not results.masks or not results.boxes or results.boxes.cls is None:
            print(f"⚠️ 객체 인식 실패 또는 마스크 없음: {image_id}")
            continue

        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        names = yolo.names

        item["crops"] = []
        for idx, mask in enumerate(masks):
            if idx >= len(classes):
                continue
            label = names[classes[idx]]

            print(f"🖼️ crop_id: {image_id}_{idx} → GPT 설명 생성 중...")
            gpt_desc = generate_docent_description(label, full_desc)

            item["crops"].append({
                "crop_id": f"{image_id}_{idx}",
                "label": label,
                "crop_description": gpt_desc
            })
            print(f"✅ 생성 완료: {gpt_desc[:40]}...")

    try:
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n📁 전체 crop 설명 저장 완료: {JSON_PATH}")
    except Exception as e:
        print(f"❌ JSON 저장 실패: {e}")

# ========== 실행 ==========
if __name__ == "__main__":
    process_all_images()
