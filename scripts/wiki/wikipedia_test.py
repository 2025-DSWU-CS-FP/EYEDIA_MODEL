import requests
from bs4 import BeautifulSoup

def fetch_wikipedia_section_by_id(title: str, section_id: str = "설명", lang: str = "ko") -> list:
    """
    위키피디아 문서에서 <h2 id=section_id>가 있는 섹션 아래 문단(<p>)들을 리스트로 반환.
    다음 섹션(<div class="mw-heading2">)이 나오면 중단.
    """
    url = f"https://{lang}.wikipedia.org/wiki/{title}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"[❌] '{title}' 페이지 요청 실패 (status {response.status_code})")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    # 대상 h2 태그 (id="설명") 탐색
    target_h2 = soup.find("h2", id=section_id)
    if not target_h2:
        print(f"[❌] 섹션 ID '{section_id}'를 찾을 수 없습니다.")
        return []

    # 해당 <h2>를 감싸고 있는 <div class="mw-heading2"> 상위 섹션 블록
    section_div = target_h2.find_parent("div", class_="mw-heading2")
    if not section_div:
        print(f"[❌] 섹션 DIV를 찾을 수 없습니다.")
        return []

    # 형제 노드 중 다음 heading 전까지의 <p> 태그 추출
    section_paragraphs = []
    for sibling in section_div.find_next_siblings():
        if sibling.name == "div" and "mw-heading2" in sibling.get("class", []):
            break
        if sibling.name == "p":
            section_paragraphs.append(sibling.get_text(strip=True))

    return section_paragraphs

# ✅ 사용 예시
title = "소크라테스의_죽음"
paragraphs = fetch_wikipedia_section_by_id(title, section_id="설명")

print(f"✅ '{title}' 문서의 '설명' 섹션 내용:\n")
for i, para in enumerate(paragraphs):
    print(f"[문단 {i+1}]\n{para}\n")
