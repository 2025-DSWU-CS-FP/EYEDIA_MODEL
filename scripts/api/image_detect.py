# filename: image_detect.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse, time, cv2
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import requests

import torch
from ultralytics import YOLO

# ── env ────────────────────────────────────────────────────────────────────────
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
PROCESS_ENDPOINT = f"{BACKEND_URL}/process-image"

ART_LIKE = {"tv", "laptop", "book", "cell phone", "remote", "keyboard"}

# ── model ──────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = YOLO("yolov8n.pt")

# ── upload ─────────────────────────────────────────────────────────────────────
def send_to_fastapi(img_path: Path):
    with open(img_path, "rb") as f:
        files = {"file": (img_path.name, f, "image/jpeg")}
        r = requests.post(PROCESS_ENDPOINT, files=files, timeout=60)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"text": r.text[:300]}

# ── ui(slider) ─────────────────────────────────────────────────────────────────
WIN = "Front(overlay) + Eye(inset)"
def _noop(v): pass

def create_trackbars():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Conf x100", WIN, 40, 100, _noop)  # 0.40
    cv2.createTrackbar("Area %",   WIN, 4,  20,  _noop)   # 4%

def get_thresholds():
    c = cv2.getTrackbarPos("Conf x100", WIN) / 100.0
    a = cv2.getTrackbarPos("Area %",    WIN) / 100.0
    return max(0.0, min(1.0, c)), max(0.0, min(0.5, a))

# ── camera ─────────────────────────────────────────────────────────────────────
def open_cam(index, w=1280, h=720, fps=30):
    # 백엔드 fallback(DShow → MSMF → ANY)
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

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", type=int, default=0)
    ap.add_argument("--eye",   type=int, default=1)
    ap.add_argument("--fps",   type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height",type=int, default=720)
    ap.add_argument("--outdir", default="recordings")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-auto", action="store_true", help="자동 업로드 비활성화")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    front_mp4 = outdir / f"{ts}_front.mp4"
    eye_mp4   = outdir / f"{ts}_eye.mp4"
    tmpdir = Path("temp"); tmpdir.mkdir(exist_ok=True)

    print("[DEBUG] opening cams…")
    cap_front = open_cam(args.front, args.width, args.height, args.fps)
    cap_eye   = open_cam(args.eye,   args.width, args.height, args.fps)

    okf, f0 = cap_front.read()
    oke, e0 = cap_eye.read()
    if not (okf and oke):
        raise RuntimeError("첫 프레임을 읽지 못했습니다.")
    H, W = f0.shape[:2]
    writer_front = make_writer(front_mp4, W, H, args.fps)
    writer_eye   = make_writer(eye_mp4,   e0.shape[1], e0.shape[0], args.fps)

    print(f"[INFO] Recording → {front_mp4}\n                 {eye_mp4}")
    print("[INFO] q: 종료, a: 자동 업로드 토글, s: 수동 업로드(현재 전면 크롭)")

    autosend = (not args.no_auto)
    cooldown = 3.0
    last_sent = 0.0
    names = yolo.model.names

    if args.show:
        create_trackbars()

    best_crop = None

    while True:
        okf, f = cap_front.read()
        oke, e = cap_eye.read()
        if not (okf and oke):
            time.sleep(0.005); continue

        # 항상 녹화
        writer_front.write(f)
        writer_eye.write(e)

        # 슬라이더 값
        conf_th, area_min = (0.40, 0.04)
        if args.show:
            conf_th, area_min = get_thresholds()

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
                if label not in ART_LIKE or conf < conf_th:
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
        crop = None
        if best is not None:
            x1,y1,x2,y2,label,conf = best
            crop = f[y1:y2, x1:x2].copy()
            best_crop = crop
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
            text = f"{label}/{conf:.2f}"
            cv2.putText(overlay, text, (x1, max(20,y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)

        # 미리보기
        if args.show:
            eye_small = cv2.resize(e, (overlay.shape[1]//3, overlay.shape[0]//3))
            overlay[0:eye_small.shape[0], 0:eye_small.shape[1]] = eye_small
            info = f"conf_th={conf_th:.2f}  area_min={area_min*100:.0f}%  auto={autosend}"
            cv2.putText(overlay, info, (10, overlay.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow(WIN, overlay)

        # 자동 업로드: 전면 크롭만 서버에 전달
        now = time.time()
        if autosend and best_crop is not None and (now - last_sent > cooldown):
            jpg = tmpdir / f"crop_{int(now)}.jpg"
            cv2.imwrite(str(jpg), best_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            code, resp = send_to_fastapi(jpg)
            print(f"[AUTO SEND] {code} -> {resp}")
            last_sent = now

        # 키 이벤트
        k = cv2.waitKey(1) & 0xFF if args.show else 255
        if k == ord('q'):
            break
        elif k == ord('a'):
            autosend = not autosend
            print("[AUTO]", autosend)
        elif k == ord('s') and best_crop is not None:
            jpg = tmpdir / f"crop_{int(time.time())}.jpg"
            cv2.imwrite(str(jpg), best_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            code, resp = send_to_fastapi(jpg)
            print(f"[SEND] {code} -> {resp}")
            last_sent = time.time()

    # 정리
    cap_front.release(); cap_eye.release()
    writer_front.release(); writer_eye.release()
    if args.show: cv2.destroyAllWindows()
    print("[INFO] done.")

if __name__ == "__main__":
    main()
