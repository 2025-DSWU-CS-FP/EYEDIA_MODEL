# eyedia_pipeline.py
import os
import cv2
import math
import json
import faiss
import torch
import requests
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

# (옵션) 시선 예측 의존성
try:
    import dlib
    import joblib
    HAS_GAZE_DEPS = True
except Exception:
    HAS_GAZE_DEPS = False

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===============================
# Config
# ===============================
YOLO_WEIGHTS = "yolov8n.pt"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# 환경변수로 서버 바꾸기: EYEDIA_API_BASE
API_BASE = os.getenv("EYEDIA_API_BASE", "http://3.34.240.201:8000")

YOLO_CONF = 0.30
YOLO_IOU  = 0.45
REQUEST_TIMEOUT = 5

# COCO 중 "작품 비슷"하게 간주할 대상
ART_CLASSES = {"tv", "book", "laptop", "cell phone", "remote", "keyboard", "monitor"}

# (옵션) 시선 예측 파일 이름
PREDICTOR_FILENAME = "shape_predictor_68_face_landmarks.dat"
GAZE_MODEL_FILENAME = "gaze_model.pkl"
ZONE_TO_Q = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}

# ===============================
# 경로 자동 탐색
# ===============================
def resolve_faiss_paths() -> Tuple[Path, Path]:
    """
    data/faiss 또는 eyedia_model/data/faiss 에 있는
    met_text.index, met_structured_with_objects.json 탐색.
    """
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    candidates = [
        cwd / "data" / "faiss",
        cwd / "eyedia_model" / "data" / "faiss",
        script_dir / "data" / "faiss",
        script_dir.parent / "data" / "faiss",
        script_dir.parent / "eyedia_model" / "data" / "faiss",
        script_dir.parent.parent / "data" / "faiss",
        script_dir.parent.parent / "eyedia_model" / "data" / "faiss",
    ]
    tried = []
    for base in candidates:
        idx = base / "met_text.index"
        meta = base / "met_structured_with_objects.json"
        tried.append(str(base))
        if idx.is_file() and meta.is_file():
            print(f"✅ FAISS 경로 사용: {base}")
            return idx, meta
    raise FileNotFoundError(
        "❌ data/faiss 파일을 찾지 못했습니다.\n- 확인한 위치들:\n"
        + "\n".join(f"  - {p}" for p in tried)
    )

def resolve_gaze_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """
    gaze 파일 탐색. predictor는 필수(없으면 gaze 비활성화), model은 선택.
    """
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    candidates = [
        cwd,
        cwd / "eyedia_model" / "data" / "gaze",
        script_dir,
        script_dir / "data" / "gaze",
        script_dir.parent / "data" / "gaze",
        script_dir.parent / "eyedia_model" / "data" / "gaze",
        script_dir.parent.parent / "eyedia_model" / "data" / "gaze",
    ]
    tried = []
    for base in candidates:
        sp = base / PREDICTOR_FILENAME
        gm = base / GAZE_MODEL_FILENAME
        tried.append(str(base))
        if sp.is_file():
            return sp, (gm if gm.is_file() else None)
    print("ℹ️ Gaze predictor 미발견 → 수동 Q 사용. 확인한 위치:\n" + "\n".join(" - "+p for p in tried))
    return None, None

# ===============================
# 카메라 오픈 (Windows 안정화: DirectShow 우선)
# ===============================
def open_camera(cam_index=0):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
    if cap.isOpened():
        return cap
    cap = cv2.VideoCapture(cam_index)
    if cap.isOpened():
        return cap
    return None

# ===============================
# Context (모델과 상태 보관)
# ===============================
class DetectContext:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.yolo = None
        self.clip_model = None
        self.clip_proc = None
        self.index = None
        self.meta = None
        self.coco_names = None

        # gaze
        self.gaze_ready = False
        self.detector = None
        self.predictor = None
        self.gaze_model = None

        # state
        self.selected_q = "Q1"

# ===============================
# Init
# ===============================
def init_models(ctx: DetectContext):
    # YOLO
    ctx.yolo = YOLO(YOLO_WEIGHTS)
    ctx.yolo.to(ctx.device)
    ctx.yolo.overrides["conf"] = YOLO_CONF
    ctx.yolo.overrides["iou"]  = YOLO_IOU

    # CLIP
    ctx.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(ctx.device).eval()
    ctx.clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    # FAISS & META
    faiss_idx_path, meta_json_path = resolve_faiss_paths()
    ctx.index = faiss.read_index(str(faiss_idx_path))
    with open(str(meta_json_path), "r", encoding="utf-8") as f:
        ctx.meta = json.load(f)

    ctx.coco_names = ctx.yolo.model.names if hasattr(ctx.yolo, "model") else ctx.yolo.names

    # Gaze (optional)
    if HAS_GAZE_DEPS:
        sp_path, gm_path = resolve_gaze_paths()
        if sp_path is not None:
            try:
                ctx.detector = dlib.get_frontal_face_detector()
                ctx.predictor = dlib.shape_predictor(str(sp_path))
                if gm_path and gm_path.is_file():
                    ctx.gaze_model = joblib.load(str(gm_path))
                    ctx.gaze_ready = True
                    print(f"✅ Gaze OK: predictor={sp_path.name}, model={gm_path.name}")
                else:
                    ctx.gaze_ready = False
                    print(f"ℹ️ predictor는 찾았지만 {GAZE_MODEL_FILENAME}은 없어 수동 Q로 폴백합니다.")
            except Exception as e:
                ctx.gaze_ready = False
                print(f"⚠️ Gaze disabled (fallback to manual Q). Reason: {e}")
        else:
            ctx.gaze_ready = False
            # 위 resolve_gaze_paths에서 안내 메시지 출력됨
    else:
        print("ℹ️ dlib/joblib 미설치 → gaze 비활성화(수동 Q 사용).")

    # Sanity: CLIP dim vs FAISS dim
    try:
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        d = embed_crop(ctx, dummy).reshape(1, -1).shape[1]
        if d != ctx.index.d:
            print(f"⚠️ FAISS dim({ctx.index.d}) != CLIP dim({d}). "
                  f"인덱스를 동일 차원/모달리티로 재구축해야 정확히 동작합니다.")
    except Exception as e:
        print(f"⚠️ 임베딩/인덱스 확인 중 문제: {e}")

    # YOLO 워밍업
    _ = ctx.yolo.predict(source=np.zeros((640,640,3), dtype=np.uint8), verbose=False, device=ctx.device)

# ===============================
# Core helpers
# ===============================
def embed_crop(ctx: DetectContext, image: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(image[:, :, ::-1]) if image.ndim == 3 and image.shape[2] == 3 else Image.fromarray(image)
    inputs = ctx.clip_proc(images=pil, return_tensors="pt", padding=True).to(ctx.device)
    with torch.no_grad():
        feats = ctx.clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats[0].detach().cpu().numpy()

def detect_frame(ctx: DetectContext, frame: np.ndarray):
    """
    YOLO → 첫 매치만 사용 → CLIP 임베딩 → FAISS top1
    Returns: (art_id, (x1,y1,x2,y2), label, score) or (None, None, None, None)
    """
    res = ctx.yolo.predict(source=frame, verbose=False, device=ctx.device)[0]
    for box in res.boxes:
        cls_id = int(box.cls[0])
        label = ctx.coco_names[cls_id] if isinstance(ctx.coco_names, dict) else ctx.coco_names[cls_id]
        if label not in ART_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        qv = embed_crop(ctx, crop).reshape(1, -1).astype(np.float32)
        D, I = ctx.index.search(qv, k=1)
        idx = int(I[0][0])
        art_id = ctx.meta[idx]["full_image_id"]
        score = float(D[0][0])
        return art_id, (x1, y1, x2, y2), label, score
    return None, None, None, None

def draw_box_and_label(frame: np.ndarray, box, text: str, color=(0,255,0)):
    if box is None:
        return frame
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return frame

# ===============================
# 스마트 백엔드 전송 (GET/POST + 경로/파라미터 조합 자동 시도)
# ===============================
def send_to_backend_smart(art_id, q: Optional[str] = None, base: Optional[str] = None, timeout: int = REQUEST_TIMEOUT):
    base = (base or API_BASE).strip().rstrip("/")
    trials = []

    # 후보 엔드포인트 & 메서드 & 파라미터 조합
    paths = ["/process-image", "/process_image", "/api/process-image", "/api/process_image"]
    # GET 쿼리식 파라미터 조합
    get_params = [
        {"art_id": art_id, **({"q": q} if q else {})},
        {"id": art_id, **({"q": q} if q else {})},
        {"object_id": art_id, **({"q": q} if q else {})},
    ]
    # POST 바디 조합
    post_jsons = [
        {"art_id": art_id, **({"q": q} if q else {})},
        {"id": art_id, **({"q": q} if q else {})},
        {"object_id": art_id, **({"q": q} if q else {})},
    ]

    # 1) 우선 GET /process-image?art_id=...
    trials.append(("GET", f"{base}/process-image", get_params[0], None))

    # 2) GET 다른 파라미터 이름들
    for p in get_params[1:]:
        trials.append(("GET", f"{base}/process-image", p, None))

    # 3) 경로 변형 + GET
    for path in paths[1:]:
        for p in get_params:
            trials.append(("GET", f"{base}{path}", p, None))

    # 4) POST JSON 시도
    for path in paths:
        for j in post_jsons:
            trials.append(("POST", f"{base}{path}", None, j))

    last_log = (None, None, None)
    for method, url, params, json_body in trials:
        try:
            if method == "GET":
                r = requests.get(url, params=params, timeout=timeout)
            else:
                r = requests.post(url, json=json_body, timeout=timeout)
            body = r.text if r.text else None
            if 200 <= r.status_code < 300:
                print(f"✅ Backend OK {r.status_code} {method} {r.url}")
                return r.status_code, r.url, (body[:500] if body else None)
            else:
                print(f"⚠️ Backend {r.status_code} {method} {r.url}")
                if body:
                    print(f"↳ Body: {body[:300]}")
                last_log = (r.status_code, r.url, (body[:500] if body else None))
        except requests.RequestException as e:
            print(f"⚠️ Request error: {method} {url} -> {e}")
            last_log = (None, url, None)

    return last_log  # (status, url, body)

# ===============================
# (옵션) Gaze helpers
# ===============================
def _get_eye_keypoints(shape, gray, idxs):
    pts = np.array([(shape.part(i).x, shape.part(i).y) for i in idxs], dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    if w == 0 or h == 0:
        return None, None, None, None
    eye_roi = gray[y:y+h, x:x+w]

    inner_corner = (shape.part(idxs[3]).x, shape.part(idxs[3]).y)
    outer_corner = (shape.part(idxs[0]).x, shape.part(idxs[0]).y)

    clahe = cv2.createCLAHE(2.0, (8,8))
    eye_roi = clahe.apply(eye_roi)
    thr = cv2.adaptiveThreshold(eye_roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    contours,_ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    pupil = None
    best = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area == 0: continue
        peri = cv2.arcLength(c, True)
        if peri == 0: continue
        circ = 4*math.pi*(area/(peri*peri))
        if 0.7 < circ < 1.2 and 15 < area < 400 and circ > best:
            best = circ
            pupil = c

    pupil_center=None
    if pupil is not None:
        M=cv2.moments(pupil)
        if M['m00']!=0:
            cx=int(M['m10']/M['m00'])+x
            cy=int(M['m01']/M['m00'])+y
            pupil_center=(cx,cy)

    glint_center=None
    if eye_roi.size>0:
        _, max_val, _, max_loc = cv2.minMaxLoc(eye_roi)
        if max_val>180:
            glint_center=(max_loc[0]+x, max_loc[1]+y)

    return inner_corner, outer_corner, pupil_center, glint_center

def _calc_features(left_eye, right_eye):
    if not left_eye or not right_eye:
        return None
    if not all(p is not None for eye in [left_eye, right_eye] for p in eye):
        return None

    l_inner, _, l_pupil, l_glint = left_eye
    r_inner, _, r_pupil, r_glint = right_eye

    l_pupil, l_glint, l_inner = np.array(l_pupil), np.array(l_glint), np.array(l_inner)
    r_pupil, r_glint, r_inner = np.array(r_pupil), np.array(r_glint), np.array(r_inner)

    vl_pg = l_pupil - l_glint
    vr_pg = r_pupil - r_glint
    vl_pc = l_pupil - l_inner
    vr_pc = r_pupil - r_inner
    vl_gc = l_glint - l_inner
    vr_gc = r_glint - r_inner
    vcc   = l_inner - r_inner
    dist_cc = np.linalg.norm(vcc)

    def ang(a,b):
        return np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-6), -1.0, 1.0))

    theta_l = ang(vl_pg, vl_gc)
    theta_r = ang(vr_pg, vr_gc)
    diff_cc = np.arctan2(vcc[1], vcc[0])

    feat = np.concatenate([
        vl_pg,[np.linalg.norm(vl_pg)], vr_pg,[np.linalg.norm(vr_pg)],
        vl_pc,[np.linalg.norm(vl_pc)], vr_pc,[np.linalg.norm(vr_pc)],
        vl_gc,[np.linalg.norm(vl_gc)], vr_gc,[np.linalg.norm(vr_gc)],
        [dist_cc, theta_l, theta_r, diff_cc]
    ])
    return feat

def predict_gaze_selected_q(ctx: DetectContext, frame_bgr: np.ndarray) -> Optional[str]:
    """
    얼굴/눈 키포인트 기반 1~4구역 예측 → Q1~Q4 반환 (실패 시 None)
    """
    if not ctx.gaze_ready:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = ctx.detector(gray)
    if len(faces) == 0:
        return None
    face = faces[0]
    landmarks = ctx.predictor(gray, face)
    left_idx  = list(range(36, 42))
    right_idx = list(range(42, 48))
    left_eye  = _get_eye_keypoints(landmarks, gray, left_idx)
    right_eye = _get_eye_keypoints(landmarks, gray, right_idx)
    feats = _calc_features(left_eye, right_eye)
    if feats is None:
        return None
    try:
        zone = int(ctx.gaze_model.predict([feats])[0])  # 1..4
        return ZONE_TO_Q.get(zone)
    except Exception:
        return None

# ===============================
# API-like helpers
# ===============================
def detect_once(ctx: DetectContext, frame_bgr: np.ndarray, mode: str = "art"):
    """
    한 프레임 계산만 수행(전송X)
    Returns dict: {'art_id':..., 'selected_q': 'Qx', 'box':..., 'label':..., 'score':...}
    """
    sel_q = ctx.selected_q
    if mode == "area":
        q_pred = predict_gaze_selected_q(ctx, frame_bgr)
        if q_pred is not None:
            sel_q = q_pred
            ctx.selected_q = sel_q

    art_id, box, label, score = detect_frame(ctx, frame_bgr)
    return {"art_id": art_id, "selected_q": sel_q, "box": box, "label": label, "score": score}

def detect_and_send_once(ctx: DetectContext, frame_bgr: np.ndarray, mode: str = "art"):
    """
    한 프레임 계산 + 전송까지
    Returns dict: {'art_id', 'selected_q', 'status', 'url', 'response_body'}
    """
    out = detect_once(ctx, frame_bgr, mode)
    art_id = out["art_id"]
    sel_q  = out["selected_q"]
    if art_id is None:
        out.update({"status": None, "url": None, "response_body": None})
        return out

    if mode == "art":
        status, url, body = send_to_backend_smart(art_id, None)
    else:
        status, url, body = send_to_backend_smart(art_id, sel_q)
    out.update({"status": status, "url": url, "response_body": (body[:500] if body else None)})
    return out

# ===============================
# Loop runners (camera)
# ===============================
def run_loop(ctx: DetectContext, mode: str = "art", show_window: bool = True, debounce_frames: int = 4, cam_index: int = 0):
    """
    카메라 루프: 안정적으로 art_id가 연속 N프레임 잡히면 1회 전송 후 종료.
    """
    cap = open_camera(cam_index)
    if cap is None or not cap.isOpened():
        print("❌ 카메라 오픈 실패. 다른 앱 점유/권한(설정>개인정보 보호 및 보안>카메라)을 확인하세요.")
        return None

    last_id = None
    streak = 0
    sent_result = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 프레임을 읽지 못했습니다.")
                break

            if mode == "area":
                q_pred = predict_gaze_selected_q(ctx, frame)
                if q_pred is not None:
                    ctx.selected_q = q_pred

            art_id, box, label, score = detect_frame(ctx, frame)

            if art_id:
                streak = streak + 1 if art_id == last_id else 1
                last_id = art_id
                text = f"{label}: {art_id} ({score:.2f})  streak {streak}/{debounce_frames}"
                draw_box_and_label(frame, box, text)

            if show_window:
                info = f"Mode: {mode}"
                if mode == "area":
                    info += f"  Q: {ctx.selected_q}"
                cv2.putText(frame, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.imshow("EYEDIA Detect", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                # 수동 Q 오버라이드(1~4)
                if mode == "area" and key in (ord('1'), ord('2'), ord('3'), ord('4')):
                    ctx.selected_q = f"Q{chr(key)}"
                    print(f"🔧 Manual Q override: {ctx.selected_q}")

            if art_id and streak >= debounce_frames:
                if mode == "art":
                    status, url, body = send_to_backend_smart(art_id, None)
                    sent_result = {"art_id": art_id, "selected_q": None, "status": status, "url": url, "response_body": (body[:500] if body else None)}
                else:
                    status, url, body = send_to_backend_smart(art_id, ctx.selected_q)
                    sent_result = {"art_id": art_id, "selected_q": ctx.selected_q, "status": status, "url": url, "response_body": (body[:500] if body else None)}
                print("➡️ sent:", sent_result)
                break

        return sent_result
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ===============================
# CLI
# ===============================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["art", "area"], default="art")
    parser.add_argument("--show", action="store_true", help="윈도우 표시")
    parser.add_argument("--debounce", type=int, default=4, help="연속 감지 프레임 수")
    parser.add_argument("--once", action="store_true", help="카메라 한 프레임만 처리 후 종료(전송X)")
    parser.add_argument("--send-once", action="store_true", help="카메라 한 프레임 처리하고 전송까지 한번")
    parser.add_argument("--cam", type=int, default=0, help="카메라 인덱스 (0=기본)")
    parser.add_argument("--q", choices=["Q1","Q2","Q3","Q4"], default="Q1", help="area 모드 초기 Q")
    args = parser.parse_args()

    ctx = DetectContext()
    ctx.selected_q = args.q
    init_models(ctx)

    if args.once or args.send_once:
        cap = open_camera(args.cam)
        if cap is None or not cap.isOpened():
            raise RuntimeError("Cannot open camera")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Cannot read frame")

        if args.send_once:
            res = detect_and_send_once(ctx, frame, mode=args.mode)
            print(res)
        else:
            res = detect_once(ctx, frame, mode=args.mode)
            print(res)
    else:
        res = run_loop(ctx, mode=args.mode, show_window=args.show, debounce_frames=args.debounce, cam_index=args.cam)
        print(res)
