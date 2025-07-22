from inference_sdk import InferenceHTTPClient
import os

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=""
)

img_path = os.path.abspath("./data/scene_images/scene_435627.jpg")
assert os.path.exists(img_path), f"❗ 이미지 경로가 존재하지 않음: {img_path}"

result = client.infer(img_path, model_id="artwork-set/9")
print(result)

import cv2

img_path = ".//data/scene_images/scene_435627.jpg"
img = cv2.imread(img_path)

# 박스 정보
x, y, w, h = 1446, 2132, 1760, 1022
x1 = int(x - w / 2)
y1 = int(y - h / 2)
x2 = int(x + w / 2)
y2 = int(y + h / 2)

# # 박스 시각화
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
cv2.imshow("Detected Picture", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cropped = img[y1:y2, x1:x2]
cv2.imwrite("cropped_detected.jpg", cropped)
