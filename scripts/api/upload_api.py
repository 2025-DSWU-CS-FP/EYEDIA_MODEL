from fastapi import FastAPI,File, UploadFile
from fastapi.responses import JSONResponse
import os
from datetime import datetime

app =FastAPI()

UPLOAD_DIR="data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-files")
async def upload_files(eye_video: UploadFile= File(...), front_image:UploadFile=File(...)):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{timestamp}_eye.mp4"
    image_filename = f"{timestamp}_front.jpg"

    video_path = os.path.join(UPLOAD_DIR, video_filename)
    image_path = os.path.join(UPLOAD_DIR, image_filename)

    with open(video_path, "wb") as vf: 
        vf.write(await eye_video.read())
    with open(image_path, "wb") as imf:
        imf.write(await front_image.read())

    print(f"업로드 완료: {video_path} ,{image_path}")

    return JSONResponse(content={
        "message": "파일 업로드 성공",
        "eye_video" : video_filename,
        "front_image": image_filename
    })