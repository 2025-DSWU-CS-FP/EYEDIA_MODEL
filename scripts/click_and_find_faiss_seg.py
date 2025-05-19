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
import openai  # 수정됨

# ===========================
# 모델 로드
# ===========================
yolo_model = YOLO("yolov8n-seg.pt")
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# FAISS 인덱스 로드
index = faiss.read_index("data/faiss/image_clip.index")

# crop_id 설명 로드
meta_paths = [
    Path("data/faiss/image_meta.json"),
    Path("data/faiss/image_en_meta.json")
]
crop_id_list = []
crop_id_to_description = {}
for path in meta_paths:
    with open(path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
        for entry in meta_data:
            for crop in entry["crops"]:
                crop_id = crop["crop_id"]
                crop_id_list.append(crop_id)
                crop_id_to_description[crop_id] = crop["crop_description"]

# ===========================
# 이미지 임베딩 함수
# ===========================
def image_embedding_from_pil(pil_img: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=pil_img, return_tensors="pt")
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
# OpenAI 호환 Ollama 클라이언트 연결
# ===========================
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="nokeyneeded"
)

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
                    binary_mask_3ch = np.stack([binary_mask] * 3, axis=-1)
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

                    matched_crop_id = crop_id_list[indices[0][0]]
                    description = crop_id_to_description.get(matched_crop_id, "설명 없음")
                    print(f"📄 crop_id: {matched_crop_id}")
                    print(f"📝 원문 설명: {description}")

                    try:
                        phi_response = client.chat.completions.create(
                            model="phi3:latest",
                            temperature=0.7,
                            messages=[
                                {"role": "system", "content": "당신은 예술작품 설명을 도와주는 도슨트입니다. 한국어로 이 객체에 대해 설명해주세요."},
                                {"role": "user", "content": f"{description}"}
                            ]
                        )
                        refined_text = phi_response.choices[0].message.content
                        print(f"🎨 정제된 설명 (phi3):\n{refined_text.strip()}")

                    except Exception as e:
                        print("❗ phi3 응답 실패:", e)

                    return  # 객체를 찾고 나면 종료

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
