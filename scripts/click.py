import os
import json
import numpy as np
import torch
import cv2
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
from dotenv import load_dotenv
import requests
import faiss

# ── ENV ────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BACKEND_OBJECT_DESC_URL = os.getenv("BACKEND_OBJECT_DESC_URL", "http://localhost:8080/api/v1/ai/object-description")
client = OpenAI(api_key=OPENAI_API_KEY)

# ── CLIP & FAISS ──────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

INDEX_PATH = "data/faiss/met_text.index"
META_PATH  = "data/faiss/met_text_meta.json"  # ← 여기엔 objectID가 있어야 함
STRUCTURED_PATH = "data/faiss/met_structured_with_objects.json"  # ← 새로 사용

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    faiss_meta = json.load(f)

# ── structured JSON 로드 & 인덱싱: full_image_id → record ─────────────
def _load_structured_by_id(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 리스트/딕셔너리 모두 대응
    by_id = {}
    if isinstance(data, dict):
        # 이미 id-keyed 이면 그대로
        if "full_image_id" in data:
            by_id[data["full_image_id"]] = data
        else:
            # nested dict일 수 있으니 모든 value 스캔
            def _collect(obj):
                if isinstance(obj, dict):
                    if "full_image_id" in obj:
                        by_id[obj["full_image_id"]] = obj
                    else:
                        for v in obj.values():
                            _collect(v)
                elif isinstance(obj, list):
                    for v in obj:
                        _collect(v)
            _collect(data)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "full_image_id" in item:
                by_id[item["full_image_id"]] = item
    return by_id

structured_by_id = _load_structured_by_id(STRUCTURED_PATH)

# ── 유틸 ───────────────────────────────────────────────────────────────
def embed_image(img: Image.Image) -> np.ndarray:
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32").squeeze()

def gpt_docent_ko(description_list, quadrant, painting_name):
    combined = "\n".join(f"- {desc}" for desc in description_list)
    prompt = (
        f"당신은 미술관의 도슨트입니다. 아래는 {painting_name}그림의 {quadrant} 분면의 후보 객체 설명입니다.\n\n"
        f"{combined}\n\n"
        "→ 이 내용을 바탕으로 관람객에게 친절하고 감성적으로 설명해주세요. "
        "너무 기술적이거나 딱딱하지 않게 풀어주세요."
        "그림에 없는 내용은 설명하지 마세요. 그림에 대한 내용만 설명하세요."
        "해당 그림 외의 다른 그림에 대한 내용은 언급하지 말아주세요"
        "그림에 대한 전체 설명보다는 각 객체에 대한 설명을 중심으로 작성해주세요."
    )
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip()

def get_quadrant(x, y, w, h):
    if x < w // 2 and y < h // 2:
        return "Q1"
    elif x >= w // 2 and y < h // 2:
        return "Q2"
    elif x < w // 2 and y >= h // 2:
        return "Q3"
    else:
        return "Q4"

# 기존 YOLO-분면 유틸(필요시 유지)
def _intersect_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)

def get_quadrants_for_bbox(x1, y1, x2, y2, w, h, min_ratio=0.05, min_pixels=1):
    x1 = max(0, min(x1, w)); x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h)); y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return [], {}

    mx, my = w // 2, h // 2
    quads = {"Q1": (0,0,mx,my), "Q2": (mx,0,w,my), "Q3": (0,my,mx,h), "Q4": (mx,my,w,h)}
    box = (x1,y1,x2,y2)
    area = (x2-x1)*(y2-y1)
    ratios = {}
    hit = []
    for q, rect in quads.items():
        ia = _intersect_area(box, rect)
        r = (ia/area) if area>0 else 0.0
        ratios[q] = r
        if ia >= min_pixels and r >= min_ratio:
            hit.append(q)
    hit.sort(key=lambda q: ratios[q], reverse=True)
    return hit, ratios

# ── 새로 추가: structured에서 분면별 crop_description 수집 ───────────────
def get_crop_descriptions_from_structured(structured_rec: dict, quadrant: str, top_k: int = 5):
    """
    structured_rec["crops"]에서 클릭 분면(quadrant)을 포함하는 crop만 뽑아
    quadrant_ratios[quadrant] 기준 내림차순으로 정렬하고 crop_description 리스트 반환
    """
    if not structured_rec:
        return []
    crops = structured_rec.get("crops", []) or []
    cand = []
    for c in crops:
        qs = c.get("quadrants", []) or []
        if quadrant in qs:
            ratios = c.get("quadrant_ratios", {}) or {}
            score = float(ratios.get(quadrant, 0.0))
            desc = (c.get("crop_description") or "").strip()
            if desc:
                cand.append((score, desc))
    # 분면 비율 기준 정렬
    cand.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in cand[:top_k]]

# ── FAISS/메타 sanity check ────────────────────────────────────────────
ntotal = index.ntotal
if ntotal == 0:
    print("[ERROR] FAISS index is empty (ntotal=0). 검색 불가.")
if len(faiss_meta) != ntotal:
    print(f"[WARN] faiss_meta length ({len(faiss_meta)}) != index.ntotal ({ntotal}). 매핑 어긋남 가능.")

K = min(3, ntotal)

# ── 메타 → structured 매핑 함수 (이름 비교 금지) ─────────────────────────
def find_structured_record_for_rep(rep_meta: dict) -> dict | None:
    """
    met_text_meta.json의 rep_meta에서 objectID(또는 id, full_image_id)를 읽어
    met_structured_with_objects.json의 full_image_id로 매칭한다.
    """
    # 선호: objectID → full_image_id
    rep_id = None
    for key in ("objectID", "full_image_id"):
        if key in rep_meta:
            rep_id = rep_meta[key]
            break
    if rep_id is None:
        print("[WARN] rep_meta에 objectID/full_image_id가 없습니다. (이름 매칭은 사용하지 않도록 요청받음)")
        return None
    # int/str 혼용 대비
    try:
        rep_id_int = int(rep_id)
    except Exception:
        rep_id_int = rep_id  # 그대로 키로 시도
    rec = structured_by_id.get(rep_id_int) or structured_by_id.get(rep_id)
    if rec is None:
        print(f"[WARN] structured_by_id에서 full_image_id={rep_id} 레코드를 찾지 못했습니다.")
    return rec

# ── 메인 루프 ──────────────────────────────────────────────────────────
def detect_and_search(image_path):
    if not os.path.exists(image_path):
        print(f"❗ 파일 없음: {image_path}")
        return

    image_bgr = cv2.imread(image_path)
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # YOLO 탐지(원하면 유지)
    yolo = YOLO("yolov8n-seg.pt")
    results = yolo(image_pil)[0]
    if results.boxes is None or len(results.boxes) == 0:
        print("❗ 객체가 감지되지 않음")
        return

    # 1) 대표 작품 결정 (CLIP→FAISS)
    whole_emb = embed_image(image_pil).reshape(1, -1)
    distances, indices = index.search(whole_emb, 1)
    rep_idx = int(indices[0][0])
    rep_meta = faiss_meta[rep_idx] if 0 <= rep_idx < len(faiss_meta) else {}
    rep_title  = rep_meta.get("title", "Unknown Artwork")
    rep_artist = rep_meta.get("artist", "Unknown Artist")
    print(f"🎨 대표 작품 결정: {rep_title} by {rep_artist}")

    # 2) structured에서 이 대표작 레코드 찾기 (id 매칭)
    structured_rec = find_structured_record_for_rep(rep_meta)

    click_pos = {"x": -1, "y": -1}
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos["x"], click_pos["y"] = x, y

    cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("YOLO", on_click)

    while True:
        vis = image_bgr.copy()
        cv2.line(vis, (w//2, 0), (w//2, h), (255,255,255), 2)
        cv2.line(vis, (0, h//2), (w, h//2), (255,255,255), 2)
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("YOLO", vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

        if click_pos["x"] != -1:
            cx, cy = click_pos["x"], click_pos["y"]
            quadrant = get_quadrant(cx, cy, w, h)
            print(f"✅ 클릭된 분면: {quadrant}")

            # 3) (중요) structured에서 분면별 crop_description 뽑기
            desc_list = get_crop_descriptions_from_structured(structured_rec, quadrant, top_k=5)

            # 만약 해당 분면 crop이 전혀 없다면, 최소한 대표작 맥락을 넣어준다.
            if not desc_list:
                print(f"[INFO] structured 내부에 {quadrant} 분면 crop이 없습니다. 간단한 대체 설명으로 진행합니다.")
                desc_list = [f"{rep_title}의 {quadrant} 분면에 위치한 주요 대상에 대해 설명해 주세요."]

            # 4) GPT 설명
            docent_description = gpt_docent_ko(desc_list, quadrant, rep_title)
            print(rep_title)
            print(f"\n[GPT 설명 - {quadrant}]\n{docent_description}\n")

            # 5) 백엔드 전송
            payload = {
                "quadrant": quadrant,
                "description": docent_description,
                "imageurl": image_path,
                "title": rep_title,
                "artist": rep_artist
            }
            try:
                res = requests.post(BACKEND_OBJECT_DESC_URL, json=payload)
                print(f"[POST] 전송 완료: {res.status_code} - {res.text}")
            except Exception as e:
                print(f"[ERROR] 백엔드 전송 실패: {e}")

            click_pos["x"] = -1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_search("data/met_images/image_20003.jpg")
