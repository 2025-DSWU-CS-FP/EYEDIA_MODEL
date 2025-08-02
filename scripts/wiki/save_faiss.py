# import requests
# from bs4 import BeautifulSoup
# import os
# import json
# import faiss
# import torch
# import numpy as np
# from PIL import Image, ImageFile
# from io import BytesIO
# from transformers import CLIPProcessor, CLIPModel

# # PIL 디코딩 충돌 방지
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# # 🔹 CLIP 모델 초기화 (CPU 사용 강제)
# device = "cpu"
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# clip_model.to(device)
# clip_model.eval()

# # 🔹 썸네일 URL → 원본 이미지 URL 변환
# def convert_to_original_image_url(thumbnail_url: str) -> str:
#     if "/thumb/" in thumbnail_url:
#         base_url, thumb_path = thumbnail_url.split("/thumb/")
#         parts = thumb_path.split("/")
#         original_path = "/".join(parts[:-1])
#         return f"{base_url}/{original_path}"
#     return thumbnail_url

# # 🔹 이미지 URL에서 PIL 이미지 로딩
# def load_image_from_url(url: str) -> Image.Image:
#     headers = {
#         "User-Agent": "EyediaBot/1.0 (Eyedia script; contact: kangchaewon@example.com)"
#     }
#     response = requests.get(url, headers=headers)
#     response.raise_for_status()
#     img = Image.open(BytesIO(response.content)).convert("RGB")
#     img.load()
#     return img

# # 🔹 이미지 → CLIP 임베딩 (세그폴트 방지 버전)
# def embed_image_from_url(url: str):
#     print(f"[🔍] 이미지 로딩 시도: {url}")
#     image = load_image_from_url(url).resize((224, 224))
#     print("[✅] 이미지 로딩 성공")

#     inputs = clip_processor(images=image, return_tensors="pt")
#     pixel_values = inputs["pixel_values"]

#     print(f"[ℹ️] pixel_values dtype: {pixel_values.dtype}, shape: {pixel_values.shape}")
#     if pixel_values.dtype != torch.float32:
#         pixel_values = pixel_values.to(dtype=torch.float32)

#     pixel_values = pixel_values.to(device)

#     with torch.no_grad():
#         image_features = clip_model.get_image_features(pixel_values=pixel_values)

#     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#     return image_features.cpu().numpy().astype("float32")

# # 🔹 위키백과에서 제목, 이미지(원본), 설명 추출
# def fetch_wikipedia_info(title: str, section_id: str = "설명", lang: str = "ko"):
#     url = f"https://{lang}.wikipedia.org/wiki/{title}"
#     headers = {
#         "User-Agent": "EyediaBot/1.0 (Eyedia script; contact: kangchaewon@example.com)"
#     }
#     response = requests.get(url, headers=headers)
#     if response.status_code != 200:
#         print(f"[❌] '{title}' 요청 실패")
#         return None

#     soup = BeautifulSoup(response.text, "html.parser")
#     clean_title = title.replace("_", " ")

#     # infobox 내 이미지 URL
#     image_url = None
#     infobox = soup.find("table", class_="infobox")
#     if infobox:
#         img = infobox.find("img")
#         if img:
#             raw_url = "https:" + img["src"]
#             image_url = convert_to_original_image_url(raw_url)

#     # 설명 세트션 파싱
#     paragraphs = []
#     for h2 in soup.find_all("h2"):
#         span = h2.find("span", {"id": section_id})
#         if span:
#             for sibling in h2.find_next_siblings():
#                 if sibling.name == "h2":
#                     break
#                 if sibling.name == "p":
#                     paragraphs.append(sibling.get_text(strip=True))
#             break

#     return {
#         "title": clean_title,
#         "image_url": image_url,
#         "description": "\n\n".join(paragraphs) if paragraphs else "[설명 없음]"
#     }

# # 🔹 FAISS 인덱스 + 메타정보 저장
# def save_image_to_faiss(info, index_path, meta_path):
#     if not info["image_url"]:
#         print(f"[❌] 이미지 없음: {info['title']}")
#         return
#     try:
#         emb = embed_image_from_url(info["image_url"])[0]
#     except Exception as e:
#         print(f"[❌] 이미지 임베딩 실패: {info['title']} → {e}")
#         return

#     # FAISS 인덱스 생성
#     dim = emb.shape[0]
#     if os.path.exists(index_path):
#         index = faiss.read_index(index_path)
#     else:
#         index = faiss.IndexFlatIP(dim)
#     index.add(np.array([emb]))
#     faiss.write_index(index, index_path)

#     # 메타정보 JSON 저장
#     meta = []
#     if os.path.exists(meta_path):
#         with open(meta_path, "r", encoding="utf-8") as f:
#             meta = json.load(f)

#     meta.append({
#         "title": info["title"],
#         "image_url": info["image_url"],
#         "description": info["description"]
#     })

#     with open(meta_path, "w", encoding="utf-8") as f:
#         json.dump(meta, f, indent=2, ensure_ascii=False)

#     print(f"✅ '{info['title']}' 이미지 임베딩 저장 완료")

# # ✅ 실행 예시
# titles = [
#     "소크라테스의_죽음",
#     "민중을_이끝는_자유의_여심",
#     "진주_귀거리를_한_소녀",
#     "그랑드_자트섬의_일요일_오후",
#     "라스_메니나스"
# ]

# for title in titles:
#     print(f"📦 '{title.replace('_', ' ')}' 임베딩 시도 중...")
#     info = fetch_wikipedia_info(title)
#     if info:
#         save_image_to_faiss(
#             info,
#             index_path="./data/faiss/wiki_text.index",
#             meta_path="./data/faiss/wiki_text_meta.json"
#         )
import requests
from bs4 import BeautifulSoup
import os
import json
import faiss
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# 🔹 CLIP 모델 초기화 (텍스트 전용)
device = "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.to(device)
clip_model.eval()

# 🔹 썸네일 URL → 원본 이미지 URL 변환
def convert_to_original_image_url(thumbnail_url: str) -> str:
    if "/thumb/" in thumbnail_url:
        base_url, thumb_path = thumbnail_url.split("/thumb/")
        parts = thumb_path.split("/")
        original_path = "/".join(parts[:-1])
        return f"{base_url}/{original_path}"
    return thumbnail_url

# 🔹 위키백과에서 제목, 이미지(원본), 설명 추출
def fetch_wikipedia_info(title: str, section_id: str = "설명", lang: str = "ko"):
    url = f"https://{lang}.wikipedia.org/wiki/{title}"
    headers = {
        "User-Agent": "EyediaBot/1.0 (Eyedia script; contact: kangchaewon@example.com)"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"[❌] '{title}' 요청 실패 (status {response.status_code})")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    clean_title = title.replace("_", " ")

    # infobox 내 이미지 URL
    image_url = None
    infobox = soup.find("table", class_="infobox")
    if infobox:
        img = infobox.find("img")
        if img:
            raw_url = "https:" + img["src"]
            image_url = convert_to_original_image_url(raw_url)

    # 설명 섹션 파싱 (향상된 방식)
    paragraphs = fetch_wikipedia_section_by_id(title, section_id, lang)

    # 설명이 없을 경우 첫 번째 <p> 2~3개 추출하여 대체
    if not paragraphs:
        all_paragraphs = soup.find_all("p")
        for p in all_paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 50:
                paragraphs.append(text)
            if len(paragraphs) >= 3:
                break

    return {
        "title": clean_title,
        "image_url": image_url,
        "description": "\n\n".join(paragraphs) if paragraphs else "[설명 없음]"
    }

# 🔹 특정 섹션의 문단 추출
def fetch_wikipedia_section_by_id(title: str, section_id: str = "설명", lang: str = "ko") -> list:
    url = f"https://{lang}.wikipedia.org/wiki/{title}"
    headers = {
        "User-Agent": "EyediaBot/1.0 (Eyedia script; contact: kangchaewon@example.com)"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    target_h2 = soup.find("span", {"id": section_id})
    if not target_h2:
        return []

    section_paragraphs = []
    for sibling in target_h2.parent.find_next_siblings():
        if sibling.name == "h2":
            break
        if sibling.name == "p":
            section_paragraphs.append(sibling.get_text(strip=True))
    return section_paragraphs

# 🔹 텍스트 임베딩 함수
def embed_text(text: str):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")

# 🔹 FAISS 인덱스 + 메타 저장
def save_text_to_faiss(info, index_path, meta_path):
    if not info["description"] or info["description"] == "[설명 없음]":
        print(f"[❌] 설명 없음: {info['title']}")
        return
    try:
        emb = embed_text(info["description"])[0]
    except Exception as e:
        print(f"[❌] 텍스트 임베딩 실패: {info['title']} → {e}")
        return

    # FAISS 인덱스
    dim = emb.shape[0]
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(np.array([emb]))
    faiss.write_index(index, index_path)

    # 메타 정보 저장
    meta = []
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    meta.append({
        "title": info["title"],
        "image_url": info["image_url"],
        "description": info["description"]
    })

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"✅ '{info['title']}' 텍스트 임베딩 저장 완료")

# ✅ 실행 예시
titles = [
    "소크라테스의_죽음",
    "민중을_이끄는_자유의_여신",
    "진주_귀걸이를_한_소녀",
    "그랑드_자트섬의_일요일_오후",
    "라스_메니나스"
]

for title in titles:
    print(f"📦 '{title.replace('_', ' ')}' 임베딩 시도 중...")
    info = fetch_wikipedia_info(title)
    if info:
        save_text_to_faiss(
            info,
            index_path="./data/faiss/wiki_text.index",
            meta_path="./data/faiss/wiki_text_meta.json"
        )