import os
import json
import cv2
import faiss
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

# 경로 설정
DATA_PATH = "./data/faiss/met_structured_with_objects.json"
CROP_DIR = "./data/crops"
FAISS_INDEX_PATH = "./data/faiss/met_text.index"
META_PATH = "./data/faiss/met_text_meta.json"

# 디렉토리 준비
os.makedirs(CROP_DIR, exist_ok=True)

# 모델 로딩
print("YOLO 로딩 중...")
yolo = YOLO("yolov8n-seg.pt")
print("CLIP 로딩 중...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)
print(f"모델 준비 완료 (device: {device})")

# 사분면 계산 유틸리티
def _intersect_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)

def get_quadrants_for_bbox(x1, y1, x2, y2, w, h, min_ratio=0.05, min_pixels=1):
    """
    bbox가 겹치는 모든 사분면을 반환
    - min_ratio: (교차면적 / bbox면적) 임계값. 작게 두면 스친 것도 포함
    - min_pixels: 최소 교차 픽셀 수
    반환: (hit_quads(list), ratios(dict))
      hit_quads 예: ["Q4","Q2"]  (겹침 비율 큰 순)
      ratios 예: {"Q1":0.0,"Q2":0.12,"Q3":0.0,"Q4":0.68}
    """
    # 경계 보정
    x1 = max(0, min(x1, w)); x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h)); y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return [], {}

    mx, my = w // 2, h // 2
    quads = {
        "Q1": (0,  0,  mx, my),   # 좌상
        "Q2": (mx, 0,  w,  my),   # 우상
        "Q3": (0,  my, mx, h),    # 좌하
        "Q4": (mx, my, w,  h),    # 우하
    }

    box = (x1, y1, x2, y2)
    box_area = (x2 - x1) * (y2 - y1)
    ratios = {}
    hits = []
    for q, rect in quads.items():
        ia = _intersect_area(box, rect)
        r = ia / box_area if box_area > 0 else 0.0
        ratios[q] = r
        if ia >= min_pixels and r >= min_ratio:
            hits.append(q)

    # 겹침 비율 큰 순으로 정렬
    hits.sort(key=lambda q: ratios[q], reverse=True)
    return hits, ratios


# summary(객체 설명) 메타데이터 및 FAISS index 로드
print("FAISS 및 메타데이터 로딩 중...")
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
summaries = [entry["summary"] for entry in meta]
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
print("준비 완료.\n")

def embed_image(img_path):
    """이미지 임베딩(512-dim 벡터) 생성"""
    pil = Image.open(img_path).convert("RGB")
    inputs = clip_processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")

def crop_and_describe_objects(item):
    """한 이미지에서 crop 및 설명 생성"""
    image_path = item.get("image_path", "")
    if not os.path.exists(image_path):
        print(f"  [이미지 없음] {image_path}")
        return []
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [이미지 읽기 실패] {image_path}")
        return []
    
    h, w, _ = image.shape
    print(f"[INFO] 원본 이미지 크기: {w}x{h}")

    results = yolo(image)[0]
    n_boxes = len(results.boxes)
    print(f"  [YOLO] 객체 {n_boxes}개 감지됨")
    crops = []

    if n_boxes == 0:
        print("    [!] 객체가 감지되지 않음 (crop 생성 안됨)")
        return []
    
    h, w, _ = image.shape

    MIN_RATIO = 0.05   # bbox 면적의 5% 이상 겹치면 포함 (상황에 따라 0.0~0.15로 조정)
    MIN_PIXELS = 1

    for i, box in enumerate(results.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)

        # 유효한 crop 영역 보정
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            print(f"[SKIP] 잘못된 crop 영역: {x1}, {y1}, {x2}, {y2}")
            continue

        crop_img = image[y1:y2, x1:x2]
        crop_id = f"{item['full_image_id']}_crop{i}"
        crop_path = os.path.join(CROP_DIR, f"{crop_id}.jpg")
        cv2.imwrite(crop_path, crop_img)

        # crop 임베딩 → summary 검색
        emb = embed_image(crop_path)                       # (1,d)
        D, I = faiss_index.search(emb, 1)
        summary = summaries[I[0][0]]

        # ✅ 여러 사분면 계산
        quads, ratios = get_quadrants_for_bbox(
            x1, y1, x2, y2, w, h,
            min_ratio=MIN_RATIO,
            min_pixels=MIN_PIXELS
        )
        # 주 사분면(겹침 비율 최대) — 겹침이 전혀 없으면 None
        primary = None
        if ratios:
            primary = max(ratios, key=ratios.get)
            if ratios[primary] == 0.0:
                primary = None

        print(
            f"    [{i}] crop 저장: {crop_path} | primary: {primary} | "
            f"quads: {quads} | crop_description: {summary[:40]}..."
        )

        crops.append({
            "crop_id": crop_id,
            "crop_path": crop_path,
            "crop_description": summary,
            "primary_quadrant": primary,     # 새 필드
            "quadrants": quads,              # 새 필드 (여러 개 가능)
            "quadrant_ratios": ratios        # 새 필드 (디버그/후처리용)
        })

    return crops

def run():
    # 전체 데이터 로드 및 crop 생성
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"총 {len(data)}개 이미지에서 crop 및 설명 생성 시작!\n")
    for idx, item in enumerate(data):
        print(f"[{idx+1}/{len(data)}] 이미지: {item.get('image_path', '')}")
        item["crops"] = crop_and_describe_objects(item)

        # ✅ 각 crop의 quadrant 확인 (다중 사분면 로직 대응 + 후방 호환)
    for crop in item["crops"]:
        primary = crop.get("primary_quadrant") or crop.get("quadrant")  # 예전 필드 호환
        quads   = crop.get("quadrants") or ([crop["quadrant"]] if "quadrant" in crop else [])
        ratios  = crop.get("quadrant_ratios", {})

        # ratios를 깔끔하게(0이 아닌 것만, 내림차순, 소수 2자리) 표현
        if ratios:
            nonzero = {k: v for k, v in ratios.items() if v > 0}
            ratios_str = ", ".join(
                f"{k}:{nonzero[k]:.2f}" for k in sorted(nonzero, key=nonzero.get, reverse=True)
            ) if nonzero else "-"
        else:
            ratios_str = "-"

        print(
            f"    - crop_id: {crop['crop_id']} | primary: {primary} | "
            f"quads: {quads} | ratios: {ratios_str}"
        )

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("\n✅ crop 생성 및 유사 객체 기반 설명 자동 생성 완료")

if __name__ == "__main__":
    run()
