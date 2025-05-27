from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess, os, json, requests
import cv2
from ultralytics import YOLO

app = FastAPI()
BACKEND_URL = "http://localhost:8080/api/model/response"  # SpringBoot 수신 주소

class AnalyzeClickRequest(BaseModel):
    image_id: str        # 예: "435638"
    click_x: int         # 클릭 x 좌표
    click_y: int         # 클릭 y 좌표

@app.post("/analyze/click")
def analyze_click(req: AnalyzeClickRequest):
    try:
        print("✅ 클릭 기반 요청 수신")
        full_image_id = f"image_{req.image_id}"
        image_path = f"data/met_images/{full_image_id}.jpg"
        assert os.path.exists(image_path), f"❌ 전체 이미지 없음: {image_path}"

        # 1. YOLO 객체 감지 및 크롭 실행
        subprocess.run(["python", "scripts/click_and_find_faiss_seg.py", image_path], check=True)

        # 2. YOLO 결과의 bbox 로드 (예: json으로 저장되어 있다고 가정)
        bbox_path = f"data/bboxes/{full_image_id}.json"
        assert os.path.exists(bbox_path), f"❌ BBox 파일 없음: {bbox_path}"
        with open(bbox_path, "r", encoding="utf-8") as f:
            bboxes = json.load(f)  # [{"x1":..., "y1":..., "x2":..., "y2":...}, ...]

        click_index = -1
        for idx, box in enumerate(bboxes):
            if box["x1"] <= req.click_x <= box["x2"] and box["y1"] <= req.click_y <= box["y2"]:
                click_index = idx
                break

        if click_index == -1:
            raise ValueError("❌ 클릭된 위치에 해당하는 객체가 없습니다.")

        crop_id = f"{full_image_id}_crop{click_index}.jpg"
        crop_path = f"data/cropped_images/{crop_id}"
        assert os.path.exists(crop_path), f"❌ 크롭 이미지 없음: {crop_path}"

        # 3. crop 설명 결과 경로
        result_path = f"output/{crop_id}_result.json"
        if not os.path.exists(result_path):
            subprocess.run(["python", "scripts/crop_and_save_each_description.py", crop_path, result_path], check=True)

        # 4. 설명 로딩
        with open(result_path, "r", encoding="utf-8") as f:
            description = json.load(f)["description"]

        # 5. 백엔드 전송
        result = {
            "full_image_id": full_image_id,
            "crop_id": crop_id,
            "description": description
        }

        res = requests.post(BACKEND_URL, json=result)
        return {"status": "ok", "backend_response": res.json()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
