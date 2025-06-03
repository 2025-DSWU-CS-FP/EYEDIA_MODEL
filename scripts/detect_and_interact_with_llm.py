import cv2
import numpy as np
import torch
import faiss
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
import json
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

# ===========================
# 환경 변수에서 Gemini API 키 불러오기
# ===========================
load_dotenv()  # .env 파일에서 환경변수 불러오기
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❗ 환경 변수 'GEMINI_API_KEY'가 설정되지 않았습니다.")

genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

# ===========================
# YOLO 및 CLIP 모델 로드
# ===========================
yolo_model = YOLO("yolov8n-seg.pt")
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name).to("cuda")
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# ===========================
# FAISS 인덱스 및 메타정보 로드
# ===========================
index = faiss.read_index("data/faiss/image_clip.index")

with open("data/faiss/image_meta.json", "r", encoding="utf-8") as f:
    image_meta = json.load(f)

crop_id_list = []
crop_id_to_description = {}

for entry in image_meta:
    for crop in entry["crops"]:
        crop_id_list.append(crop["crop_id"])
        crop_id_to_description[crop["crop_id"]] = crop["crop_description"]

# ===========================
# 도슨트 설명 생성 함수 (Gemini 기반)
# ===========================
def generate_description(prompt: str) -> str:
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini API 오류: {e}"

# ===========================
# 이미지 임베딩 함수 (GPU)
# ===========================
def image_embedding_from_pil(pil_img: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=pil_img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")

# ===========================
# 유사 이미지 검색 함수
# ===========================
def search_similar_image(embedding: np.ndarray, index: faiss.Index, top_k=1):
    distances, indices = index.search(embedding, top_k)
    return distances, indices

# ===========================
# 객체 탐지 및 클릭 인터랙션
# ===========================
def detect_and_interact(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

    screen_width, screen_height = 1280, 720
    image = cv2.resize(image, (screen_width, screen_height))

    results = yolo_model(image, conf=0.3)[0]
    class_names = yolo_model.names

    if results.masks is None:
        print("❗ 객체가 감지되지 않았습니다.")
        return

    masks = results.masks.data.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)

    seg_image = image.copy()
    object_regions = []

    for i, mask in enumerate(masks):
        class_id = classes[i]
        color = colors[class_id]
        binary_mask = (mask > 0.5).astype(np.uint8)
        binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))

        colored_mask = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            colored_mask[:, :, c] = binary_mask * color[c]

        seg_image = cv2.addWeighted(seg_image, 1.0, colored_mask, 0.5, 0)
        object_regions.append((binary_mask, class_names[class_id]))

    os.makedirs("cropped_objects", exist_ok=True)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for mask, label in object_regions:
                if mask[y, x] == 1:
                    print(f"🖱️ ({x}, {y}) → '{label}' 객체 클릭")

                    binary_mask = mask
                    binary_mask_3ch = np.stack([binary_mask]*3, axis=-1)
                    masked_image = np.where(binary_mask_3ch == 1, image, 255)

                    x_indices, y_indices = np.where(binary_mask == 1)
                    x_min, x_max = np.min(y_indices), np.max(y_indices)
                    y_min, y_max = np.min(x_indices), np.max(x_indices)
                    cropped_object = masked_image[y_min:y_max, x_min:x_max]

                    if cropped_object.size == 0:
                        print("❗ 클릭한 객체가 너무 작습니다.")
                        return

                    save_path = f"cropped_objects/cropped_{x}_{y}.png"
                    cv2.imwrite(save_path, cropped_object)
                    print(f"📷 크롭된 객체 저장 완료: {save_path}")

                    pil_cropped = Image.fromarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB))
                    embedding = image_embedding_from_pil(pil_cropped)
                    distances, indices = search_similar_image(embedding, index)

                    print(f"\n🔍 가장 유사한 이미지 인덱스: {indices[0][0]}")
                    print(f"🔎 거리(유사도 점수): {distances[0][0]}")

                    if indices[0][0] < len(crop_id_list):
                        matched_crop_id = crop_id_list[indices[0][0]]
                        description = crop_id_to_description.get(matched_crop_id, "설명 없음")

                        print(f"📄 파일명: {matched_crop_id}")
                        print(f"📝 설명: {description}")

                        try:
                            print("\n🤖 도슨트 설명:")

                            prompt = f"""
                            당신은 예술작품을 설명하는 도슨트입니다.

                            주어진 정보를 바탕으로 관람객에게 마치 전시장에서 직접 설명하듯, 따뜻하고 자연스럽게 말해 주세요.  
                            말투는 구어체이며, 비유나 쉬운 단어를 사용해 주세요.

                            [정보]
                            - 객체 이름: {label}
                            - 객체 설명: {description}

                            도슨트 설명:
                            """

                            start_time = time.time()
                            output_text = generate_description(prompt)
                            end_time = time.time()

                            print("\n" + output_text + "\n")
                            print(f"⏱️ 응답 소요 시간: {end_time - start_time:.2f}초")

                        except Exception as e:
                            print("❗ Gemini 실행 중 오류 발생:", e)

                    else:
                        print("❗ 인덱스 범위를 벗어났습니다.")
                    return
            print(f"🖱️ ({x}, {y}) → 객체 없음")

    cv2.namedWindow("YOLOv8 Segmentation Click")
    cv2.setMouseCallback("YOLOv8 Segmentation Click", on_mouse)

    while True:
        cv2.imshow("YOLOv8 Segmentation Click", seg_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

# ===========================
# 실행
# ===========================
if __name__ == "__main__":
    detect_and_interact("data/raw_images/image-7.jpg")