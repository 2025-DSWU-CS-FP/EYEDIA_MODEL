import cv2, json, numpy as np, torch, faiss, os
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
import openai
import re

def run(image_path):
    assert os.path.exists(image_path), "❗ 이미지 파일이 존재하지 않습니다"
    
    # 모델 로딩
    yolo = YOLO("yolov8n-seg.pt")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="nokeyneeded")

    # 경로 설정
    base_dir = "C:/Users/dorot/EYEDIA_MODEL/data/faiss"
    image_meta_path = f"{base_dir}/met_image_meta.json"
    text_meta_path = f"{base_dir}/met_text_meta.json"
    index_path = f"{base_dir}/met_image.index"

    for path in [image_meta_path, text_meta_path, index_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❗ 필수 파일이 없습니다: {path}")

    index = faiss.read_index(index_path)
    with open(image_meta_path, "r", encoding="utf-8") as f:
        crop_meta = json.load(f)
    with open(text_meta_path, "r", encoding="utf-8") as f:
        text_meta = json.load(f)
    id_to_summary = {m["id"]: m["summary"] for m in text_meta}

    def embed(img):
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            emb = clip.get_image_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)

    def crop_to_item_id(crop_id):
        match = re.search(r"image_(\d+)_crop", crop_id)
        return f"item_{match.group(1)}" if match else None

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❗ 이미지 로드 실패: {image_path}")
    image = cv2.resize(image, (1280, 720))

    results = yolo(image, conf=0.3)[0]
    masks = results.masks.data.cpu().numpy() if results.masks else []

    seg_image = image.copy()
    resized_masks = []

    # 🎨 마스크 시각화 및 resize
    for i, mask in enumerate(masks):
        bin_mask = (mask > 0.5).astype(np.uint8)
        bin_mask = cv2.resize(bin_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        resized_masks.append(bin_mask)

        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        overlay = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            overlay[:, :, c] = bin_mask * color[c]

        seg_image = cv2.addWeighted(seg_image, 1.0, overlay, 0.4, 0)

    # 🖱️ 클릭 이벤트
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"🖱️ 클릭 위치: ({x}, {y})")
            for i, bin_mask in enumerate(resized_masks):
                if bin_mask[y, x] == 1:
                    ys, xs = np.where(bin_mask == 1)
                    if ys.size == 0 or xs.size == 0:
                        print("❗ 마스크 비어 있음")
                        return

                    crop = image[np.min(ys):np.max(ys), np.min(xs):np.max(xs)]
                    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    emb = embed(pil).cpu().numpy().astype("float32")
                    _, idx = index.search(emb, 1)

                    matched_crop_id = crop_meta[idx[0][0]]["crop_id"]
                    item_id = crop_to_item_id(matched_crop_id)
                    summary = id_to_summary.get(item_id, "요약 없음")
                    print(f"✅ 객체 {i}번 클릭됨")

                    print(f"\n🎯 item_id: {item_id}")
                    print(f"📄 요약:\n{summary}")

                    try:
                        res = client.chat.completions.create(
                            model="phi3:latest",
                            messages=[
                                {"role": "system", "content": "당신은 한국어 도슨트입니다. 아래 내용을 관람객에게 설명해 주세요."},
                                {"role": "user", "content": summary}
                            ]
                        )
                        print(f"\n🗣️ 도슨트 설명:\n{res.choices[0].message.content.strip()}")
                    except Exception as e:
                        print(f"❗ phi3 오류: {e}")
                    return
            print("❌ 클릭한 위치에 감지된 객체가 없습니다.")

    # 실행
    cv2.namedWindow("MET 작품 도슨트")
    cv2.setMouseCallback("MET 작품 도슨트", on_mouse)
    while True:
        cv2.imshow("MET 작품 도슨트", seg_image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 종료
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run("data/met_images/image_9.jpg")
