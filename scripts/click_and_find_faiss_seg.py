import cv2, json, numpy as np, torch, faiss, os
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import requests
import time

OBJECT_DESC_URL = "http://localhost:8080/api/v1/ai/object-description"

def load_meta():
    structured_path = Path("./data/faiss/met_structured_with_objects.json")
    meta_path = Path("./data/faiss/met_text_meta.json")

    if not structured_path.exists() or not meta_path.exists():
        raise FileNotFoundError("❗ 메타 데이터 파일이 없습니다.")

    with open(structured_path, "r", encoding="utf-8") as f1:
        structured = json.load(f1)
    with open(meta_path, "r", encoding="utf-8") as f2:
        text_meta = json.load(f2)

    meta_lookup = {str(entry["id"]): entry for entry in text_meta}
    crop_meta = []

    for item in structured:
        full_id = str(item["full_image_id"])
        matched_meta = meta_lookup.get(full_id, {})
        for crop in item.get("crops", []):
            if "crop_description" not in crop:
                continue
            crop_meta.append({
                "crop_id": crop["crop_id"],
                "crop_description": crop["crop_description"],
                "full_image_id": full_id,
                "artist": matched_meta.get("artist", "unknown"),
                "title": matched_meta.get("title", "untitled"),
                "paintingId": int(full_id)
            })
    return crop_meta

def run(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❗ 이미지 없음: {image_path}")

    yolo = YOLO("yolov8n-seg.pt")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip.to(device)

    index_path = "./data/faiss/met_crop.index"
    if not os.path.exists(index_path):
        raise FileNotFoundError("❗ met_crop.index 없음")
    index = faiss.read_index(index_path)
    crop_meta = load_meta()

    def embed(img):
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype("float32")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("❗ 이미지 로딩 실패")
    image = cv2.resize(image, (1280, 720))

    results = yolo(image, conf=0.3)[0]
    masks = results.masks.data.cpu().numpy() if results.masks else []

    seg_image = image.copy()
    resized_masks = []

    for mask in masks:
        bin_mask = (mask > 0.1).astype(np.uint8)
        bin_mask = cv2.resize(bin_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        resized_masks.append(bin_mask)

        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        overlay = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            overlay[:, :, c] = bin_mask * color[c]
        seg_image = cv2.addWeighted(seg_image, 1.0, overlay, 0.4, 0)

    last_click_time = 0
    click_delay = 0.5

    def on_touch(event, x, y, flags, param):
        nonlocal last_click_time

        current_time = time.time()
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if current_time - last_click_time < click_delay:
            return
        last_click_time = current_time

        print(f"\n🖱 클릭 위치: ({x}, {y})")
        patch_size = 10
        h, w = seg_image.shape[:2]

        for bin_mask in resized_masks:
            x_min, x_max = max(x - patch_size, 0), min(x + patch_size, w)
            y_min, y_max = max(y - patch_size, 0), min(y + patch_size, h)
            patch = bin_mask[y_min:y_max, x_min:x_max]

            if np.any(patch == 1):
                ys, xs = np.where(bin_mask == 1)
                if ys.size == 0 or xs.size == 0:
                    return

                crop = image[np.min(ys):np.max(ys), np.min(xs):np.max(xs)]
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                emb = embed(pil)

                _, idx = index.search(emb, 1)
                if idx[0][0] < 0 or idx[0][0] >= len(crop_meta):
                    print("❗ 검색 결과 무효")
                    return

                matched = crop_meta[idx[0][0]]
                payload = {
                    "objectId": matched["crop_id"],
                    "description": matched["crop_description"],
                    "imageurl": "http://example.com/image.jpg",
                    "title": matched["title"],
                    "artist": matched["artist"],
                    "paintingId": matched["paintingId"]
                }

                print("📦 보내는 JSON:", json.dumps(payload, indent=2, ensure_ascii=False))
                try:
                    res2 = requests.post(OBJECT_DESC_URL, json=payload)
                    print(f"📤 설명 등록 응답: {res2.status_code} {res2.text}")
                except Exception as e:
                    print(f"❌ 설명 전송 실패: {e}")
                return
        print("❌ 클릭한 위치에 객체 없음")

    cv2.namedWindow("met viewer")
    cv2.setMouseCallback("met viewer", on_touch)

    while True:
        cv2.imshow("met viewer", seg_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image_id = "436419"
    image_path = f"./data/met_images/image_{test_image_id}.jpg"
    run(image_path)
