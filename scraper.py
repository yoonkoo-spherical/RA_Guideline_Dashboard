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

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GOOGLE_SEARCH_API_KEY = "AIzaSyCxrgLsD13-gSyHUPm7yaBW7VTqdGhj75w"
GOOGLE_SEARCH_CX = "75a7d5b8e12dc47af"

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Supabase Client Error: {e}")
    exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)
FILTER_MODEL = "gemini-3.1-flash"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def is_valid_text(text):
    """추출된 텍스트가 의미 있는 문자열인지 판별합니다."""
    if not text.strip():
        return False
    # 영문자, 숫자, 한글의 개수를 측정
    alphanumeric_count = len(re.findall(r'[a-zA-Z0-9가-힣]', text))
    total_length = len(text.replace(" ", "").replace("\n", ""))
    
    if total_length == 0:
        return False
    # 유효 문자 비율이 40% 이상인지 확인
    return (alphanumeric_count / total_length) > 0.4

def extract_text_with_ocr(file_path):
    """임시 저장된 PDF 파일에서 페이지 단위로 텍스트를 추출하거나 OCR을 수행합니다."""
    try:
        text = ""
        doc = fitz.open(file_path)
        for page in doc:
            page_text = page.get_text()
            
            # 텍스트가 정상적으로 추출되었는지 확인
            if is_valid_text(page_text):
                text += page_text + "\n"
            else:
                # 텍스트가 없거나 깨진 경우 해당 페이지만 OCR 적용
                pix = page.get_pixmap(dpi=150) # 해상도 지정으로 OCR 품질 확보
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img, lang='eng+kor')
                text += ocr_text + "\n"
                
        doc.close()
        return text if text.strip() else "추출 불가: 유효 텍스트 없음"
    except Exception as e:
        return f"추출 불가: {e}"

def search_agency_guidelines(agency, site_domain):
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_CX:
        print("구글 검색 API 환경 변수가 설정되지 않았습니다.")
        return []
    
    # API 키 검증 오류 방지
    print(f"현재 로드된 API KEY 앞뒤 식별: {GOOGLE_SEARCH_API_KEY[:5]}***{GOOGLE_SEARCH_API_KEY[-4:]}")

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
            # 예외 발생 시 Fail-closed 적용 (다운로드 목록에서 제외)
            print(f"LLM 필터링 중 오류 발생({link['url']}): {e} -> 목록에서 제외됨")
            
        time.sleep(0.5)

    return valid_links

def download_and_save_pdf(doc_info):
    url = doc_info['url']
    
    existing = supabase.table("guidelines").select("url").eq("url", url).execute()
    if existing.data:
        print(f" - 이미 존재하는 문서: {url}")
        return

    print(f" - 다운로드 시도: {url}")
    temp_pdf_path = None
    try:
        # 스트리밍 방식으로 파일 다운로드 (메모리 최적화)
        res = requests.get(url, headers=REQUEST_HEADERS, stream=True, timeout=30)
        res.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            for chunk in res.iter_content(chunk_size=8192):
                if chunk:
                    temp_pdf.write(chunk)
            temp_pdf_path = temp_pdf.name
        
        # 임시 파일에서 텍스트 추출
        raw_text = extract_text_with_ocr(temp_pdf_path)
        
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
    finally:
        # 작업 완료 후 임시 파일 삭제
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

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
