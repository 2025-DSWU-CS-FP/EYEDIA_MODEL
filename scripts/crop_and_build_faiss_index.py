import os, json, cv2, numpy as np, faiss
from PIL import Image
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

def crop_and_update_structured():
    # 경로 설정
    image_dir = Path("data/met_images")
    crop_dir = Path("data/cropped_images")
    faiss_dir = Path("data/faiss")
    structured_path = faiss_dir / "met_structured.json"
    output_path = faiss_dir / "met_structured_with_objects.json"

    crop_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로딩
    yolo = YOLO("yolov8n-seg.pt")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 기존 structured 데이터 불러오기
    with open(structured_path, "r", encoding="utf-8") as f:
        structured_data = json.load(f)

    vectors = []

    def embed_image(pil_image):
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip.get_image_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)

    for item in structured_data:
        img_path = Path(item["img"])
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ 이미지 로드 실패: {img_path}")
            continue

        img_resized = cv2.resize(img, (1280, 720))
        results = yolo(img_resized, conf=0.3)[0]

        if not results.masks:
            continue

        item["객체들"] = []
        class_names = results.names
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for i, (mask, cls_idx) in enumerate(zip(masks, classes)):
            bin_mask = (mask > 0.5).astype(np.uint8)
            ys, xs = np.where(bin_mask == 1)
            if ys.size == 0 or xs.size == 0:
                continue

            cropped = img_resized[np.min(ys):np.max(ys), np.min(xs):np.max(xs)]
            crop_name = f"{img_path.stem}_crop{i}.jpg"
            crop_path = crop_dir / crop_name
            cv2.imwrite(str(crop_path), cropped)

            pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            emb = embed_image(pil_crop).cpu().numpy().astype("float32")
            vectors.append(emb)

            class_name = class_names[int(cls_idx)]
            description = f"이 객체는 '{class_name}'으로 감지되었습니다."

            item["객체들"].append({
                "객체이미지": str(crop_path),
                "객체설명": description
            })

    # 최종 JSON 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 객체 정보가 met_structured_with_objects.json에 저장되었습니다.")
    print(f"✅ 총 {len(vectors)}개의 crop 객체가 생성되었습니다.")

if __name__ == "__main__":
    crop_and_update_structured()
