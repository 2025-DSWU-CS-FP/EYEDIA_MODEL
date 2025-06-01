import cv2, json, numpy as np, torch, faiss, os, re
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import requests

BACKEND_URL = "http://localhost:8080/api/model/response"  # Spring Boot 백엔드 URL

def generate_met_image_meta_from_structured():
    structured_path = Path("./data/faiss/met_structured_with_objects.json")
    image_index_path = Path("./data/faiss/met_image.index")
    output_path = Path("./data/faiss/met_image_meta.json")

    if not structured_path.exists():
        raise FileNotFoundError(f"❗ met_structured_with_objects.json 파일이 없습니다: {structured_path}")

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
                "id": f"item_{image_id}"
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"✅ met_image_meta.json 생성 완료 ({len(meta)}개 객체)")

    if not image_index_path.exists():
        print("⚠️ met_image.index가 없어 빈 인덱스를 생성합니다.")
        dummy_index = faiss.IndexFlatIP(512)
        faiss.write_index(dummy_index, str(image_index_path))
        print("✅ 빈 met_image.index 생성 완료")

def run(image_path):
    assert os.path.exists(image_path), "❗ 이미지 파일이 존재하지 않습니다"

    yolo = YOLO("yolov8n-seg.pt")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

    base_dir = "./data/faiss"
    image_meta_path = f"{base_dir}/met_image_meta.json"
    index_path = f"{base_dir}/met_image.index"

    for path in [image_meta_path, index_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❗ 필수 파일이 없습니다: {path}")

    index = faiss.read_index(index_path)
    with open(image_meta_path, "r", encoding="utf-8") as f:
        crop_meta = json.load(f)

    def embed(img):
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            emb = clip.get_image_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❗ 이미지 로드 실패: {image_path}")
    image = cv2.resize(image, (1280, 720))

    results = yolo(image, conf=0.3)[0]
    masks = results.masks.data.cpu().numpy() if results.masks else []

    seg_image = image.copy()
    resized_masks = []

    for i, mask in enumerate(masks):
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
            print(f"📱 터치 위치: ({x}, {y})")
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
                        print("❗ 마스크 비어 있음")
                        return

                    print("✅ 객체 클릭 인식됨")
                    cv2.circle(seg_image, (x, y), 10, (0, 255, 0), 2)
                    cv2.putText(seg_image, "✅ 객체 인식됨", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    crop = image[np.min(ys):np.max(ys), np.min(xs):np.max(xs)]
                    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    emb = embed(pil).cpu().numpy().astype("float32")
                    _, idx = index.search(emb, 1)

                    matched_crop_id = crop_meta[idx[0][0]]['crop_id']
                    matched_crop = next((c for c in crop_meta if c['crop_id'] == matched_crop_id), None)
                    description = matched_crop.get("crop_description", "설명 없음") if matched_crop else "crop_id 매칭 실패"

                    print(f"\n🎯 crop_id: {matched_crop_id}")
                    print(f"📄 설명:\n{description}")

                    try:
                        payload = {
                            "full_image_id": Path(image_path).stem,
                            "crop_id": matched_crop_id,
                            "description": description
                        }
                        res = requests.post(BACKEND_URL, json=payload)
                        print(f"✅ 백엔드 전송 완료: {res.status_code} {res.text}")
                    except Exception as e:
                        print(f"❗ 백엔드 전송 오류: {e}")
                    return
            print("❌ 터치한 위치에 감지된 객체가 없습니다.")

    cv2.namedWindow("met viewer")
    cv2.setMouseCallback("met viewer", on_touch)
    while True:
        cv2.imshow("met viewer", seg_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_met_image_meta_from_structured()
    run("data/met_images/image_435638.jpg")
