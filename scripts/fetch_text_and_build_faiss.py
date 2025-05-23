import requests
import xml.etree.ElementTree as ET
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

#=============================
#설정 
#=============================

## SERVICE_KEY="f7a04215-b2c7-4470-8f43-865f0cfd8755"
URL = "http://api.kcisa.kr/openapi/service/rest/meta10/get20150041"
PARAMS = {
    "serviceKey": SERVICE_KEY,
    "numOfRows": "100",
    "pageNo": "1"
}

FAISS_INDEX_PATH = "phi3_subject.index"
META_JSON_PATH = "phi3_subject_meta.json"



#==============================
# 모델 설정
#==============================

client = OpenAI(
    base_url ="http://localhost:11434/v1",
    api_key="ollama"
    

)

embedder =SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# ===========================
# API 호출 및 XML 파싱
# =========================== 

response = requests.get(URL, params =PARAMS)
response.encoding ="utf-8"
root =ET.fromstring(response.text)
items = root.findall(".//item")

texts=[]
meta =[]

for i, item in enumerate (items):
    title = item.findtext("title","")
    keyword = item.findtext("subjectKeywor","")
    desc = item.findtext("description", "")
    abstract = item.findtext("abstract", "")
    

    raw_text =f"{keyword}.{desc} {abstract}".strip()
    if not raw_text:
        continue

    try:
        res = client.chat.completions.create(
            model="phi3:latest",
            messages=[
                {"role": "system", "content": "당신은 한국어 정보를 요약하는 도우미입니다."},
                {"role": "user", "content": f"다음 내용을 핵심적으로 요약해줘:\n\n{raw_text}"}
            ]
        )
        summary = res.choices[0].message.content.strip()
    except Exception as e:
        print(f"❗ phi3 요약 실패 ({i}):", e)
        summary = raw_text[:200]

    print(f"✅ [{i+1}] 요약 완료: {summary[:50]}...")
    texts.append(summary)
    meta.append({
        "id": f"item_{i}",
        "title": title,
        "subjectKeyword": keyword,
        "summary": summary
    })

# ===========================
# 벡터 임베딩 + FAISS 저장
# ===========================


embeddings = embedder.encoder(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
index = faiss.IndexFlatIP(embeddings.shape[1])

index.add(embeddings)
faiss.write_index(index, FAISS_INDEX_PATH)
with open(META_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"\n✅ 총 {len(texts)}개 항목 저장 완료 → {FAISS_INDEX_PATH}, {META_JSON_PATH}")