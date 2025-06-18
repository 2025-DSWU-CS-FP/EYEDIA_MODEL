import cv2, json, numpy as np, torch, faiss, os, re
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import requests

BACKEND_URL = "http://localhost:8080/api/v1/ai/object-description"

def generate_met_image_meta_from_structured():
    structured_path = Path("./data/faiss/met_structured_with_objects.json")
    image_index_path = Path("./data/faiss/met_crop.index")
    output_path = Path("./data/faiss/met_crop_meta.json")

    if not structured_path.exists():
        raise FileNotFoundError(f"met_structured_with_objects.json 사용 불가: {structured_path}")

    with open(structured_path, "r", encoding="utf-8") as f:
        structured_data = json.load(f)

    meta = []
    for item in structured_data:
        full_id = str(item["full_image_id"])
        match = re.search(r'(\d+)', full_id)
        image_id = match.group(1) if match else full_id
        for crop in item.get("crops", []):
            meta.append({
                "crop_id": crop["crop_id"],
                "crop_description": crop["crop_description"],
                "full_image_id": image_id
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"met_crop_meta.json 생성 완료 ({len(meta)}개 객체)")

    if not image_index_path.exists():
        print("⚠️ met_crop.index 없음. 빈 인덱스 생성 중...")
        dummy_index = faiss.IndexFlatIP(512)
        faiss.write_index(dummy_index, str(image_index_path))
        print("빈 met_crop.index 생성 완료")

def run(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일 없음: {image_path}")

    yolo = YOLO("yolov8n-seg.pt")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip.to(device)

    base_dir = "./data/faiss"
    image_meta_path = f"{base_dir}/met_crop_meta.json"
    index_path = f"{base_dir}/met_crop.index"

    for path in [image_meta_path, index_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"필수 파일 없음: {path}")

    index = faiss.read_index(index_path)
    with open(image_meta_path, "r", encoding="utf-8") as f:
        crop_meta = json.load(f)

    def embed(img):
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        emb_np = emb.cpu().numpy().astype("float32")
        if emb_np.ndim == 1:
            emb_np = emb_np.reshape(1, -1)
        elif emb_np.shape[1] != index.d:
            raise ValueError(f"임베딩 차원 {emb_np.shape[1]} ≠ 인덱스 차원 {index.d}")
        return emb_np

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지 로드 실패: {image_path}")
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

    def on_touch(event, x, y, flags, param):
        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP]:
            print(f"\n🖱️ 클릭 위치: ({x}, {y})")
            h, w = seg_image.shape[:2]
            patch_size = 10

            for i, bin_mask in enumerate(resized_masks):
                x_min = max(x - patch_size, 0)
                x_max = min(x + patch_size, w)
                y_min = max(y - patch_size, 0)
                y_max = min(y + patch_size, h)
                patch = bin_mask[y_min:y_max, x_min:x_max]

                if np.any(patch == 1):
                    ys, xs = np.where(bin_mask == 1)
                    if ys.size == 0 or xs.size == 0:
                        print("마스크 비어 있음")
                        return

                    print("✅ 객체 클릭 감지됨")
                    cv2.circle(seg_image, (x, y), 10, (0, 255, 0), 2)
                    cv2.putText(seg_image, "객체 인식됨", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    crop = image[np.min(ys):np.max(ys), np.min(xs):np.max(xs)]
                    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    emb = embed(pil)

                    _, idx = index.search(emb, 1)
                    if not idx.any() or idx[0][0] < 0 or idx[0][0] >= len(crop_meta):
                        print("❗ 유효하지 않은 검색 결과")
                        return

                    matched_crop = crop_meta[idx[0][0]]
                    description = matched_crop.get("crop_description", "설명 없음")
                    matched_crop_id = matched_crop.get("crop_id", "crop_id 없음")
                  
                    print(f"crop_id (objectId): {matched_crop_id}")
                    print(f"설명: {description}")


                    try:
                        payload = {
                            "objectId": matched_crop_id,
                            "description": description,
                            "imageUrl": "http://example.com/image.jpg",  
                        }
                        res = requests.post(BACKEND_URL, json=payload,  headers={"Content-Type": "application/json"})
                        print("📦 보내는 JSON:", json.dumps(payload, indent=2)) 

                        print(f"백엔드 전송 완료: {res.status_code} {res.text}")
                    except Exception as e:
                        print(f"백엔드 전송 오류: {e}")
                    return
            print("❌ 클릭한 위치에 감지된 객체 없음")

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
