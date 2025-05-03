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

# ===========================
# 모델 로드
# ===========================
yolo_model = YOLO("yolov8n-seg.pt")
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# FAISS 인덱스 로드
index = faiss.read_index("../data/faiss/image_clip.index")

# ===========================
# image_meta.json 로드
# ===========================
with open("../data/faiss/image_meta.json", "r") as f:
    image_meta = json.load(f)

crop_id_list = []
crop_id_to_description = {}

for entry in image_meta:
    for crop in entry["crops"]:
        crop_id_list.append(crop["crop_id"])
        crop_id_to_description[crop["crop_id"]] = crop["crop_description"]

# ===========================
# 이미지 임베딩 함수 (PIL 이미지 입력)
# ===========================
def image_embedding_from_pil(pil_img: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 정규화
    return emb.cpu().numpy().astype("float32")

# ===========================
# 유사 이미지 검색 함수
# ===========================
def search_similar_image(embedding: np.ndarray, index: faiss.Index, top_k=1):
    distances, indices = index.search(embedding, top_k)
    return distances, indices

# ===========================
# 객체 탐지 및 클릭 이벤트 함수
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

                    # 1. 마스크를 3채널로 변환
                    binary_mask_3ch = np.stack([binary_mask]*3, axis=-1)

                    # 2. 이미지와 곱해서 객체 부분만 남기고 배경은 흰색으로 채우기
                    masked_image = np.where(binary_mask_3ch == 1, image, 255)

                    # 3. 마스크 부분만 자르기
                    x_indices, y_indices = np.where(binary_mask == 1)
                    x_min, x_max = np.min(y_indices), np.max(y_indices)
                    y_min, y_max = np.min(x_indices), np.max(x_indices)
                    cropped_object = masked_image[y_min:y_max, x_min:x_max]

                    if cropped_object.size == 0:
                        print("❗ 클릭한 객체가 너무 작습니다.")
                        return

                    # 크롭한 객체 저장
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
# 실행 부분 -> 테스트 할 이미지 지정
# ===========================
if __name__ == "__main__":
    detect_and_interact("../data/raw_images/image-7.jpg")