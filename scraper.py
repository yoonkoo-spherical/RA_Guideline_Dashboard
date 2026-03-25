import os
import time
import requests
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import tempfile
import re
from supabase import create_client, Client
from dotenv import load_dotenv
import json
from google import genai
from google.genai import types
from curl_cffi import requests as curl_requests  # 최신 WAF 우회 라이브러리 추가

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Supabase Client Error: {e}")
    exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)
FILTER_MODEL = "gemini-3.1-flash-lite-preview"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def is_valid_text(text):
    """추출된 텍스트가 의미 있는 문자열인지 판별합니다."""
    if not text.strip():
        return False
    alphanumeric_count = len(re.findall(r'[a-zA-Z0-9가-힣]', text))
    total_length = len(text.replace(" ", "").replace("\n", ""))

    if total_length == 0:
        return False
    return (alphanumeric_count / total_length) > 0.4

def extract_text_with_ocr(file_path):
    """임시 저장된 PDF 파일에서 페이지 단위로 텍스트를 추출하거나 OCR을 수행합니다."""
    try:
        text = ""
        doc = fitz.open(file_path)
        for page in doc:
            page_text = page.get_text()

            if is_valid_text(page_text):
                text += page_text + "\n"
            else:
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img, lang='eng+kor')
                text += ocr_text + "\n"

        doc.close()
        return text if text.strip() else "추출 불가: 유효 텍스트 없음"
    except Exception as e:
        return f"추출 불가: {e}"

def search_agency_guidelines(agency, site_domain):
    """Serper API를 사용하여 규제 기관의 가이드라인(PDF)을 검색합니다."""
    if not SERPER_API_KEY:
        print("Serper API 환경 변수가 설정되지 않았습니다.")
        return []

    query = f'site:{site_domain} "biosimilar" OR "monoclonal antibody" filetype:pdf'
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
        "num": 10
    })

    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=30)
        response.raise_for_status()

        search_results = response.json().get('organic', [])

        extracted_links = []
        for item in search_results:
            extracted_links.append({
                "agency": agency,
                "title": item.get('title', ''),
                "snippet": item.get('snippet', ''),
                "url": item.get('link', '')
            })
        return extracted_links
    except requests.exceptions.HTTPError as e:
        print(f"{agency} 검색 API HTTP 오류 발생: {e.response.status_code}")
        print(f"상세 오류 메시지: {e.response.text}")
        return []
    except Exception as e:
        print(f"{agency} 검색 중 일반 오류 발생: {e}")
        return []

def filter_links_with_llm(links):
    if not links:
        return []

    valid_links = []
    for link in links:
        prompt = f"""
        다음 문서는 웹 검색을 통해 수집된 PDF입니다. 이 문서가 '바이오시밀러(Biosimilar)' 또는 '단일클론항체(Monoclonal Antibody)'의 개발, 제조, 임상, 또는 인허가와 관련된 공식 규제 가이드라인입니까?
        단순 뉴스 기사, 일반 제품 설명서, 마케팅 자료인 경우 false를 반환하십시오.
        
        문서 제목: {link['title']}
        문서 요약: {link['snippet']}
        URL: {link['url']}
        
        오직 JSON 형식으로만 응답하십시오: {{"is_relevant": true 또는 false}}
        """

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=FILTER_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
                )
                result = json.loads(response.text)
                if result.get("is_relevant"):
                    valid_links.append(link)
                break

            except Exception as e:
                if "503" in str(e) and attempt < max_retries - 1:
                    print(f"서버 과부하(503) 발생. {retry_delay}초 후 다시 시도합니다... (시도 {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"LLM 필터링 중 오류 발생({link['url']}): {e} -> 목록에서 제외됨")
                    break

        time.sleep(1.0)

    return valid_links

def download_and_save_pdf(doc_info):
    url = doc_info['url']

    existing = supabase.table("guidelines").select("url").eq("url", url).execute()
    if existing.data:
        print(f" - 이미 존재하는 문서: {url}")
        return

    print(f" - 다운로드 시도: {url}")
    temp_pdf_path = None
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # curl_cffi를 사용하여 크롬 110 버전의 지문 모방 및 타임아웃 60초 지정
            res = curl_requests.get(url, impersonate="chrome110", timeout=60)
            res.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(res.content)
                temp_pdf_path = temp_pdf.name

            raw_text = extract_text_with_ocr(temp_pdf_path)

            supabase.table("guidelines").insert({
                "title": doc_info['title'],
                "agency": doc_info['agency'],
                "category": "Biosimilar/mAb",
                "url": url,
                "raw_text": raw_text
            }).execute()
            print(f" - DB 저장 완료: {doc_info['title']}")
            break  # 성공 시 재시도 루프 탈출

        except Exception as e:
            print(f" - 다운로드 실패 (시도 {attempt + 1}/{max_retries}) ({url}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # 실패 시 5초 대기 후 재시도
        finally:
            # 매 시도마다 임시 파일이 생성되었다면 삭제하여 용량 관리
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
                temp_pdf_path = None

def run_scraper():
    print("--- 지능형 가이드라인 수집기 시작 ---")

    agencies = [
        {"name": "FDA", "domain": "fda.gov"},
        {"name": "EMA", "domain": "ema.europa.eu"},
        {"name": "MHRA", "domain": "gov.uk"},
        {"name": "Health Canada", "domain": "canada.ca"}
    ]

    all_raw_links = []

    for agency in agencies:
        print(f"\n[{agency['name']}] 검색 중...")
        links = search_agency_guidelines(agency['name'], agency['domain'])
        print(f" -> {len(links)}개의 원시 링크 발견")
        all_raw_links.extend(links)
        time.sleep(1)

    print("\n--- LLM 필터링 시작 ---")
    filtered_links = filter_links_with_llm(all_raw_links)
    print(f" -> 필터링 완료: 총 {len(all_raw_links)}개 중 {len(filtered_links)}개 유효 문서 판별됨")

    print("\n--- 문서 다운로드 및 DB 저장 시작 ---")
    for doc in filtered_links:
        download_and_save_pdf(doc)
        time.sleep(1)

    print("\n--- 수집기 작동 종료 ---")

if __name__ == "__main__":
    run_scraper()
