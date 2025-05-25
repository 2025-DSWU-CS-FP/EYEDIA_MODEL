from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

from scripts.api.utils import detect_objects, extract_crop, get_dosun_text

app = FastAPI()

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...), x: int = Form(...), y: int = Form(...)):
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(status_code=400, content={"error": "이미지를 읽을 수 없습니다."})

    masks = detect_objects(image)
    crop_id, item_id, _, index = extract_crop(image, masks, x, y)

    if not crop_id:
        return JSONResponse(status_code=404, content={"error": "해당 위치에 객체 없음"})

    summary, explanation = get_dosun_text(item_id)

    return {
        "object_index": index,
        "crop_id": crop_id,
        "item_id": item_id,
        "summary": summary,
        "dosun": explanation
    }
