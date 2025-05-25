# scripts/utils.py
import cv2, torch, faiss, json, numpy as np, re
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
import openai

# 모델 및 리소스 초기화 (전역)
yolo = YOLO("yolov8n-seg.pt")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="nokeyneeded")

index = faiss.read_index("./data/faiss/met_image.index")
with open("./data/faiss/met_image_meta.json", "r", encoding="utf-8") as f:
    crop_meta = json.load(f)
with open("./data/faiss/met_text_meta.json", "r", encoding="utf-8") as f:
    text_meta = json.load(f)
id_to_summary = {m["id"]: m["summary"] for m in text_meta}


def embed_image(pil_img):
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


def detect_objects(image):
    image = cv2.resize(image, (1280, 720))
    results = yolo(image, conf=0.3)[0]
    masks = results.masks.data.cpu().numpy() if results.masks else []

    resized_masks = []
    for mask in masks:
        bin_mask = (mask > 0.5).astype(np.uint8)
        bin_mask = cv2.resize(bin_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        resized_masks.append(bin_mask)
    return resized_masks


def extract_crop(image, masks, x, y):
    for i, mask in enumerate(masks):
        if mask[y, x] == 1:
            ys, xs = np.where(mask == 1)
            if ys.size == 0 or xs.size == 0:
                return None, None, None, i
            crop = image[np.min(ys):np.max(ys), np.min(xs):np.max(xs)]
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            vec = embed_image(pil_crop)
            _, idx = index.search(vec, 1)
            crop_id = crop_meta[idx[0][0]]["crop_id"]

            match = re.search(r"image_(\d+)_crop", crop_id)
            if not match:
                return crop_id, None, vec, i

            item_id = f"item_{match.group(1)}"
            return crop_id, item_id, vec, i

    return None, None, None, -1


def get_dosun_text(item_id):
    summary = id_to_summary.get(item_id, "요약 없음")
    try:
        res = client.chat.completions.create(
            model="phi3:latest",
            messages=[
                {"role": "system", "content": "당신은 한국어 도슨트입니다. 아래 내용을 관람객에게 설명해 주세요."},
                {"role": "user", "content": summary}
            ]
        )
        return summary, res.choices[0].message.content.strip()
    except Exception as e:
        return summary, f"phi3 오류: {e}"
