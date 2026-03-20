import os
import time
import requests
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_bytes
from supabase import create_client, Client
from dotenv import load_dotenv
import json
from google import genai
from google.genai import types

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Supabase Client Error: {e}")
    exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)
FILTER_MODEL = "gemini-3.1-flash"

# 규제 기관 봇 차단 방어용 헤더
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def extract_text_with_ocr(pdf_bytes):
    try:
        text = ""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
        
        if text.strip():
            return text
            
        images = convert_from_bytes(pdf_bytes)
        text = "".join(pytesseract.image_to_string(img, lang='eng+kor') for img in images)
        return text if text.strip() else "추출 불가: OCR 실패"
    except Exception:
        return "추출 불가"

def search_agency_guidelines(agency, site_domain):
    print(f"현재 로드된 API KEY 앞 5자리: {GOOGLE_SEARCH_API_KEY[:5]}***") # 추가
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_CX:
        print("구글 검색 API 환경 변수가 설정되지 않았습니다.")
        return []

    query = f'site:{site_domain} "biosimilar" OR "monoclonal antibody" filetype:pdf'
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': GOOGLE_SEARCH_API_KEY,
        'cx': GOOGLE_SEARCH_CX,
        'q': query,
        'num': 10
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        search_results = response.json().get('items', [])
        
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
        # 구글 API 403 에러의 상세 원인(JSON)을 출력하도록 개선
        error_details = e.response.text
        print(f"{agency} 검색 API HTTP 오류 발생: {e.response.status_code}")
        print(f"상세 오류 메시지: {error_details}")
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
        try:
            response = client.models.generate_content(
                model=FILTER_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
            )
            result = json.loads(response.text)
            if result.get("is_relevant"):
                valid_links.append(link)
        except Exception as e:
            print(f"LLM 필터링 중 오류 발생({link['url']}): {e}")
            valid_links.append(link)
            
        time.sleep(0.5)

    return valid_links

def download_and_save_pdf(doc_info):
    url = doc_info['url']
    
    existing = supabase.table("guidelines").select("url").eq("url", url).execute()
    if existing.data:
        print(f" - 이미 존재하는 문서: {url}")
        return

    print(f" - 다운로드 시도: {url}")
    try:
        # 다운로드 시 User-Agent 헤더를 포함하여 기관 서버의 봇 차단 방어
        res = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        res.raise_for_status()
        pdf_bytes = res.content
        
        raw_text = extract_text_with_ocr(pdf_bytes)
        
        supabase.table("guidelines").insert({
            "title": doc_info['title'],
            "agency": doc_info['agency'],
            "category": "Biosimilar/mAb",
            "url": url,
            "raw_text": raw_text
        }).execute()
        print(f" - DB 저장 완료: {doc_info['title']}")
        
    except requests.exceptions.HTTPError as e:
        print(f" - 다운로드 거부(HTTP Error) ({url}): {e.response.status_code}")
    except Exception as e:
        print(f" - 다운로드 또는 저장 실패 ({url}): {e}")

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
