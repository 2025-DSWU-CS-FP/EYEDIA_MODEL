import cv2
import requests
import time
from datetime import datetime

INGEST_URL = "http://172.20.0.1:8008/api/v1/ingest"  
FRONT_IDX = 2   # 전면 카메라 인덱스
EYE_IDX   = 0   # 눈동자 카메라 인덱스
WIDTH, HEIGHT, FPS = 640, 480, 15 # 해상도 및 FPS 설정

WINDOW = "Pi Preview  [c: capture&send | s: swap cams | q: quit]"


def open_cam(idx, w=1280, h=720, fps=30):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    if not cap.isOpened():
        raise RuntimeError(f"Camera open failed: {idx}")
    return cap

def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("frame read failed")
    return frame

def to_jpeg_bytes(frame, quality=92):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()

def send_pair(front_jpg: bytes, eye_jpg: bytes):
    ts = datetime.utcnow().isoformat()
    files = {
        "front": ("front.jpg", front_jpg, "image/jpeg"),
        "eye":   ("eye.jpg",   eye_jpg,   "image/jpeg"),
    }
    r = requests.post(INGEST_URL, files=files, timeout=10)
    print("[POST]", r.status_code, r.text[:200])


def main():
    # 카메라 열기
    cap_front = open_cam(FRONT_IDX, WIDTH, HEIGHT, FPS)
    cap_eye   = open_cam(EYE_IDX,   WIDTH, HEIGHT, FPS)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("키 조작:  c=캡처·전송  s=카메라 스왑  q=종료")

    last_info = ""
    while True:
        try:
            f = read_frame(cap_front)
            e = read_frame(cap_eye)
        except Exception as ex:
            print("카메라 읽기 오류:", ex)
            time.sleep(0.2)
            continue

        # 미리보기: 전면과 눈 프레임을 좌우로 합치기
        try:
            eye_small = cv2.resize(e, (f.shape[1] // 3, f.shape[0] // 3))
            preview = f.copy()
            # 좌상단에 눈 프레임 inset
            h, w = eye_small.shape[:2]
            preview[0:h, 0:w] = eye_small
            # 안내 텍스트
            cv2.putText(preview, last_info, (10, preview.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception:
            preview = f  # 실패해도 전면만 표시

        cv2.imshow(WINDOW, preview)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break

        elif k == ord('s'):
            # 전/후면(눈) 카메라 스왑
            cap_front, cap_eye = cap_eye, cap_front
            last_info = "swapped cams"

        elif k == ord('c'):
            # 캡처 & 업로드
            try:
                front_jpg = to_jpeg_bytes(f)
                eye_jpg   = to_jpeg_bytes(e)
                send_pair(front_jpg, eye_jpg)
                last_info = "captured & sent"
            except Exception as ex:
                last_info = f"send error: {ex}"
                print("ERR:", ex)

    cap_front.release()
    cap_eye.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
