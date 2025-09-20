# gazedetection.py
import cv2
import dlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
from typing import Optional

# ── 기본 설정
SCREEN_ZONES = {1: "Top-Left", 2: "Top-Right", 3: "Bot-Left", 4: "Bot-Right"}
SAMPLES_PER_ZONE = 20

# ── 내부 전역 (지연 초기화용)
_G = {
    "detector": None,
    "predictor": None,
    "model": None,
    "warned": False,
}

def _resolve_path(filename: str) -> Optional[Path]:
    """
    모델 파일을 찾을 수 있는 흔한 경로를 순회.
    """
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    candidates = [
        cwd / filename,
        here / filename,
        cwd / "eyedia_model" / "data" / "gaze" / filename,
        here / "data" / "gaze" / filename,
        here.parent / "data" / "gaze" / filename,
        here.parent / "eyedia_model" / "data" / "gaze" / filename,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None

def _lazy_init_for_predict() -> bool:
    """
    predict_zone()에서 최초 1회만 호출되어 dlib predictor와 학습 모델 로드.
    """
    if _G["detector"] is not None and _G["predictor"] is not None and _G["model"] is not None:
        return True

    # dlib
    _G["detector"] = dlib.get_frontal_face_detector()

    sp_path = _resolve_path("shape_predictor_68_face_landmarks.dat")
    if sp_path is None:
        if not _G["warned"]:
            print("ℹ️ Gaze predictor(.dat) 파일을 찾지 못했습니다 → gaze 비활성화 (predict_zone -> None).")
            _G["warned"] = True
        return False
    _G["predictor"] = dlib.shape_predictor(str(sp_path))

    # 학습 모델
    gm_path = _resolve_path("gaze_model.pkl")
    if gm_path is None:
        if not _G["warned"]:
            print("ℹ️ gaze_model.pkl을 찾지 못했습니다 → gaze 비활성화 (predict_zone -> None).")
            _G["warned"] = True
        return False
    _G["model"] = joblib.load(str(gm_path))
    return True

def _get_eye_keypoints(shape, gray_frame, eye_points_indices):
    eye_points = np.array([(shape.part(i).x, shape.part(i).y) for i in eye_points_indices], dtype=np.int32)
    x, y, w, h = cv2.boundingRect(eye_points)
    if w == 0 or h == 0:
        return None, None, None, None
    eye_roi = gray_frame[y:y+h, x:x+w]

    inner_corner = (shape.part(eye_points_indices[3]).x, shape.part(eye_points_indices[3]).y)
    outer_corner = (shape.part(eye_points_indices[0]).x, shape.part(eye_points_indices[0]).y)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eye_roi = clahe.apply(eye_roi)

    thr = cv2.adaptiveThreshold(eye_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    pupil_contour = None
    max_circ = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area == 0:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circ = 4 * np.pi * (area / (peri * peri))
        if 0.7 < circ < 1.2 and 15 < area < 400:
            if circ > max_circ:
                max_circ = circ
                pupil_contour = c

    pupil_center = None
    if pupil_contour is not None:
        M = cv2.moments(pupil_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"]) + x
            cy = int(M["m01"] / M["m00"]) + y
            pupil_center = (cx, cy)

    glint_center = None
    if eye_roi.size > 0:
        _, max_val, _, max_loc = cv2.minMaxLoc(eye_roi)
        if max_val > 180:
            glint_center = (max_loc[0] + x, max_loc[1] + y)

    return inner_corner, outer_corner, pupil_center, glint_center

def _calc_features(left_eye, right_eye):
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
    vcc = l_inner - r_inner

    def L(v): return np.linalg.norm(v)
    def ang(a, b): return np.arccos(
        np.clip(np.dot(a, b) / (L(a) * L(b) + 1e-6), -1.0, 1.0)
    )

    dist_cc = L(vcc)
    theta_l = ang(vl_pg, vl_gc)
    theta_r = ang(vr_pg, vr_gc)
    diff_cc = np.arctan2(vcc[1], vcc[0])

    feats = np.concatenate([
        vl_pg, [L(vl_pg)], vr_pg, [L(vr_pg)],
        vl_pc, [L(vl_pc)], vr_pc, [L(vr_pc)],
        vl_gc, [L(vl_gc)], vr_gc, [L(vr_gc)],
        [dist_cc, theta_l, theta_r, diff_cc]
    ])
    return feats

def predict_zone(frame_bgr: np.ndarray) -> Optional[int]:
    """
    [핵심] 프레임 한 장에서 시선 구역 예측.
      - 성공: 1|2|3|4
      - 실패: None (모델/예측 불가/얼굴 없음 등)
    """
    if not _lazy_init_for_predict():
        return None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _G["detector"](gray)
    if len(faces) == 0:
        return None
    face = faces[0]
    landmarks = _G["predictor"](gray, face)

    left_idx = list(range(36, 42))
    right_idx = list(range(42, 48))
    left_eye = _get_eye_keypoints(landmarks, gray, left_idx)
    right_eye = _get_eye_keypoints(landmarks, gray, right_idx)

    feats = _calc_features(left_eye, right_eye)
    if feats is None:
        return None

    try:
        zone = int(_G["model"].predict([feats])[0])  # 1..4
        if zone in (1, 2, 3, 4):
            return zone
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────
# 아래는 '수집/학습/시연'을 위한 간단한 CLI (원하면 사용)
# ──────────────────────────────────────────────────────────────
def _run_collect_and_train():
    print("--- 데이터 수집 모드 (자동 진행) ---")
    print(f"각 구역을 응시한 상태에서 '스페이스바'를 눌러 데이터를 {SAMPLES_PER_ZONE}개씩 수집합니다.")
    print("수집 완료 시 's'를 눌러 학습/저장합니다.")

    detector = dlib.get_frontal_face_detector()
    sp_path = _resolve_path("shape_predictor_68_face_landmarks.dat")
    if sp_path is None:
        print("❌ predictor(.dat) 파일을 찾을 수 없습니다.")
        return
    predictor = dlib.shape_predictor(str(sp_path))

    features_data, labels_data = [], []
    current_zone_to_collect = 1
    collected_counts = {i: 0 for i in range(1, 5)}

    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            features = None
            for face in faces:
                landmarks = predictor(gray, face)
                left_eye_keypoints = _get_eye_keypoints(landmarks, gray, list(range(36, 42)))
                right_eye_keypoints = _get_eye_keypoints(landmarks, gray, list(range(42, 48)))
                features = _calc_features(left_eye_keypoints, right_eye_keypoints)

            disp = np.zeros((720, 1280, 3), dtype=np.uint8)
            if current_zone_to_collect <= 4:
                cnt = collected_counts[current_zone_to_collect]
                text = f"Look at Zone [{current_zone_to_collect}] ({cnt}/{SAMPLES_PER_ZONE}). Press SPACE."
            else:
                text = "Collection Complete! Press 's' to train and save."
            cv2.putText(disp, text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.imshow("collect", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' ') and features is not None and current_zone_to_collect <= 4:
                if collected_counts[current_zone_to_collect] < SAMPLES_PER_ZONE:
                    features_data.append(features)
                    labels_data.append(current_zone_to_collect)
                    collected_counts[current_zone_to_collect] += 1
                    print(f"Zone {current_zone_to_collect} ({collected_counts[current_zone_to_collect]}/{SAMPLES_PER_ZONE})")
                if collected_counts[current_zone_to_collect] == SAMPLES_PER_ZONE:
                    current_zone_to_collect += 1

            if key == ord('s') and all(v == SAMPLES_PER_ZONE for v in collected_counts.values()):
                print("\n--- Training Model ---")
                X = np.array(features_data)
                y = np.array(labels_data)
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X, y)
                joblib.dump(model, "gaze_model.pkl")
                print("✅ Model saved: gaze_model.pkl")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def _run_predict_demo():
    print("--- 실시간 예측 데모 (predict_zone 사용) ---")
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            zone = predict_zone(frame)
            text = f"Gaze Prediction: Zone {zone}" if zone else "Gaze Prediction: (None)"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            cv2.imshow("gaze demo", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["collect", "demo"], default="demo")
    args = p.parse_args()

    if args.mode == "collect":
        _run_collect_and_train()
    else:
        _run_predict_demo()
