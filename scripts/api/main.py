"""
  python main.py --show \
    --faiss-index ./data/faiss/met_text.index \
    --meta ./data/faiss/met_structured_with_objects.json \
    --exhibition "summer-2025" \
    --title "unknown-frame" \
    --front 0 --eye 1

단축키
- q : 종료
- a : 자동 업로드 토글
- s : 수동 업로드(현재 best 크롭)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 로드 충돌 허용(경고 억제)

import argparse
import base64
import time
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv

import faiss
from transformers import CLIPProcessor, CLIPModel


# ── ENV ────────────────────────────────────────────────────────────────
load_dotenv()
BACKEND_URL = (os.getenv("BACKEND_URL") or "").rstrip("/")
if not BACKEND_URL:
    raise RuntimeError("환경변수 BACKEND_URL 이 비어 있습니다. 예) BACKEND_URL=http://<host>:8080")

UPLOAD_ENDPOINT = f"{BACKEND_URL}/api/v1/paintings/upload"

# ── 기본 파라미터 ─────────────────────────────────────────────────────
WIN = "Front(overlay) + Eye(inset)"
DEFAULT_ART_CLASSES = ["tv", "laptop", "book", "cell phone", "remote", "keyboard", "monitor"]


# ── 업로드 유틸 (Base64 JSON) ─────────────────────────────────────────
def upload_front_crop_b64(crop_bgr: np.ndarray, exhibition: str, title: str, timeout: float = 30.0):
    """
    전면 카메라 크롭 이미지를 JPEG→Base64로 인코딩해
    POST /api/v1/paintings/upload 로 전송합니다.

    Query:
      exhibition, title
    Body(JSON):
      {"image": "<base64>"}
    """
    ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return None, {"error": "jpeg encode failed"}

    img_b64 = base64.b64encode(buf).decode("utf-8")

    params = {"exhibition": exhibition, "title": title}
    payload = {"image": img_b64}

    try:
        resp = requests.post(UPLOAD_ENDPOINT, params=params, json=payload, timeout=timeout)
        try:
            data = resp.json()
        except Exception:
            data = {"text": resp.text[:400]}
        return resp.status_code, data
    except Exception as e:
        return None, {"error": f"request failed: {e}"}


# ── 트랙바 유틸 ───────────────────────────────────────────────────────
def _noop(v): ...
def create_trackbars():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Conf x100", WIN, 40, 100, _noop)  # 기본 0.40
    cv2.createTrackbar("Area %",   WIN, 4,   20,  _noop)  # 기본 4%

def get_thresholds(show_enabled: bool, default_conf=0.40, default_area=0.04):
    if not show_enabled:
        return default_conf, default_area
    c = cv2.getTrackbarPos("Conf x100", WIN) / 100.0
    a = cv2.getTrackbarPos("Area %",    WIN) / 100.0
    return max(0.0, min(1.0, c)), max(0.0, min(0.5, a))


# ── 카메라/레코더 ────────────────────────────────────────────────────
def open_cam(index, w=1280, h=720, fps=30):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS,          fps)
            print(f"[INFO] camera {index} opened with backend={backend}")
            return cap
    raise RuntimeError(f"Camera open failed: index={index}")

def make_writer(path: Path, w, h, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (w, h))


# ── CLIP/FAISS 매처 ──────────────────────────────────────────────────
class ClipFaissMatcher:
    def __init__(self, clip_model_name: str, faiss_index_path: str, meta_path: str, device: str):
        self.enabled = False
        self.device = device
        self.model = None
        self.proc = None
        self.index = None
        self.meta = None
        self.dim = None

        if faiss_index_path and meta_path and Path(faiss_index_path).exists() and Path(meta_path).exists():
            print(f"[INFO] Loading CLIP({clip_model_name}) + FAISS index...")
            self.model = CLIPModel.from_pretrained(clip_model_name)
            self.proc  = CLIPProcessor.from_pretrained(clip_model_name)
            self.model.to(self.device)
            self.model.eval()

            self.index = faiss.read_index(faiss_index_path)
            self.dim = self.index.d
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            self.enabled = True
            print(f"[INFO] FAISS ready. dim={self.dim}, meta={len(self.meta)}")
        else:
            print("[INFO] CLIP/FAISS 비활성화 (인덱스/메타 경로 확인)")

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self.proc(images=pil, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        return feats[0].detach().float().cpu().numpy()

    def search_top1(self, crop_bgr: np.ndarray):
        if not self.enabled or crop_bgr is None:
            return None, None
        vec = self.embed(crop_bgr).reshape(1, -1)
        if vec.shape[1] != self.dim:
            print(f"[WARN] CLIP dim({vec.shape[1]}) != FAISS dim({self.dim})")
            return None, None
        D, I = self.index.search(vec, k=1)
        idx = int(I[0][0])
        if idx < 0 or idx >= len(self.meta):
            return None, None
        return self.meta[idx], float(D[0][0])


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    # 카메라/출력
    ap.add_argument("--front", type=int, default=0, help="전면 카메라 index")
    ap.add_argument("--eye",   type=int, default=1, help="눈 카메라 index")
    ap.add_argument("--fps",   type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height",type=int, default=720)
    ap.add_argument("--outdir", default="recordings")
    ap.add_argument("--show", action="store_true", help="미리보기/트랙바 표시")
    ap.add_argument("--no-eye-record", action="store_true", help="동공 영상 녹화 비활성화")
    # 업로드
    ap.add_argument("--no-auto", action="store_true", help="자동 업로드 비활성화")
    ap.add_argument("--cooldown", type=float, default=3.0, help="자동 업로드 최소 간격(sec)")
    ap.add_argument("--exhibition", default="default-exhibition", help="업로드 쿼리 파라미터: exhibition")
    ap.add_argument("--title", default="untitled", help="업로드 쿼리 파라미터: title")
    # 모델/경로
    ap.add_argument("--yolo-model", default="yolov8n.pt")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--faiss-index", default="", help="FAISS 인덱스 경로")
    ap.add_argument("--meta", default="", help="FAISS 메타데이터(JSON) 경로")
    # 클래스
    ap.add_argument("--art-classes", default=",".join(DEFAULT_ART_CLASSES),
                    help="감지 허용 클래스 콤마구분 (COCO 라벨 기준)")
    args = ap.parse_args()

    # 출력 폴더
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    front_mp4 = outdir / f"{ts}_front.mp4"
    eye_mp4   = outdir / f"{ts}_eye.mp4"
    print(f"[INFO] q: 종료, a: 자동 업로드 토글, s: 수동 업로드")

    # 디바이스
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    # YOLO
    yolo = YOLO(args.yolo_model)
    names = yolo.model.names

    # CLIP/FAISS (선택)
    matcher = ClipFaissMatcher(args.clip_model, args.faiss_index, args.meta, device)

    # 카메라
    print("[DEBUG] opening cams…")
    cap_front = open_cam(args.front, args.width, args.height, args.fps)
    cap_eye   = open_cam(args.eye,   args.width, args.height, args.fps)

    okf, f0 = cap_front.read()
    oke, e0 = cap_eye.read()
    if not (okf and oke):
        raise RuntimeError("첫 프레임을 읽지 못했습니다.")

    H, W = f0.shape[:2]
    writer_front = make_writer(front_mp4, W, H, args.fps)
    writer_eye   = None if args.no_eye_record else make_writer(eye_mp4, e0.shape[1], e0.shape[0], args.fps)

    print(f"[INFO] Recording → {front_mp4}")
    if writer_eye is None:
        print("[INFO] Eye recording disabled (--no-eye-record)")
    else:
        print(f"[INFO]            {eye_mp4}")

    autosend = (not args.no_auto)
    last_sent = 0.0
    art_like = set(x.strip() for x in args.art_classes.split(",") if x.strip())

    if args.show:
        create_trackbars()

    best_crop = None  # 업로드 대상
    while True:
        okf, f = cap_front.read()
        oke, e = cap_eye.read()
        if not (okf and oke):
            time.sleep(0.005); continue

        # 항상 전면 녹화
        writer_front.write(f)
        # 동공 녹화는 옵션
        if writer_eye is not None:
            writer_eye.write(e)

        # 임계치
        conf_th, area_min = get_thresholds(args.show, default_conf=0.40, default_area=0.04)

        # YOLO 감지 → 가장 큰 ART_LIKE 1개
        best = None; best_area = -1.0
        try:
            res = yolo(f, verbose=False)[0]
        except Exception as ex:
            print("[WARN] YOLO inference failed:", ex)
            res = None

        if res is not None:
            for b in res.boxes:
                cls_id = int(b.cls[0])
                label  = names.get(cls_id, str(cls_id))
                conf   = float(b.conf[0]) if b.conf is not None else 0.0
                if label not in art_like or conf < conf_th:
                    continue
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                w,h = x2-x1, y2-y1
                if w<=0 or h<=0: continue
                area = (w*h)/(W*H)
                if area < area_min: continue
                if area > best_area:
                    best_area = area
                    best = (x1,y1,x2,y2,label,conf)

        overlay = f.copy()
        best_crop = None
        match_text = None

        if best is not None:
            x1,y1,x2,y2,label,conf = best
            best_crop = f[y1:y2, x1:x2].copy()

            # CLIP+FAISS 매칭 표시(선택)
            matched_meta, distance = matcher.search_top1(best_crop)
            if matched_meta is not None:
                full_id = matched_meta.get("full_image_id") or matched_meta.get("id") or "N/A"
                title_m = matched_meta.get("title") or ""
                match_text = f"{label}/{conf:.2f}  →  match:{full_id}  dist:{distance:.2f}"
                if title_m:
                    match_text += f"  {title_m}"

            cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
            txt = match_text if match_text else f"{label}/{conf:.2f}"
            cv2.putText(overlay, txt, (x1, max(20,y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)

        # 미리보기
        if args.show:
            eye_small = cv2.resize(e, (overlay.shape[1]//3, overlay.shape[0]//3))
            overlay[0:eye_small.shape[0], 0:eye_small.shape[1]] = eye_small
            info = f"conf_th={conf_th:.2f}  area_min={area_min*100:.0f}%  auto={autosend}"
            cv2.putText(overlay, info, (10, overlay.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow(WIN, overlay)

        # 자동 업로드 (전면 크롭만)
        now = time.time()
        if autosend and best_crop is not None and (now - last_sent > args.cooldown):
            code, resp = upload_front_crop_b64(best_crop, args.exhibition, args.title)
            print(f"[AUTO UPLOAD] status={code} resp={resp}")
            last_sent = now

        # 키 입력
        k = cv2.waitKey(1) & 0xFF if args.show else 255
        if k == ord('q'):
            break
        elif k == ord('a'):
            autosend = not autosend
            print("[AUTO]", autosend)
        elif k == ord('s') and best_crop is not None:
            code, resp = upload_front_crop_b64(best_crop, args.exhibition, args.title)
            print(f"[MANUAL UPLOAD] status={code} resp={resp}")

    # 정리
    cap_front.release(); cap_eye.release()
    writer_front.release()
    if writer_eye is not None:
        writer_eye.release()
    if args.show:
        cv2.destroyAllWindows()
    print("[INFO] done.")


if __name__ == "__main__":
    main()
