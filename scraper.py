import os
import time
import json
import re
import tempfile
import random
from urllib.parse import urljoin

import requests
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
from supabase import create_client, Client
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# 환경 변수 로드
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

# 고도화된 브라우저 헤더 (429 및 차단 방지)
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive"
}

def is_valid_text(text):
    """추출된 텍스트의 유효성 판별"""
    if not text.strip():
        return False
    alphanumeric_count = len(re.findall(r'[a-zA-Z0-9가-힣]', text))
    total_length = len(text.replace(" ", "").replace("\n", ""))
    if total_length == 0:
        return False
    return (alphanumeric_count / total_length) > 0.4

def extract_text_with_ocr(file_path):
    """PDF에서 텍스트 추출 또는 OCR 수행"""
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

def request_with_retry(url, max_retries=3):
    """지수 백오프 및 랜덤 지연을 적용한 HTTP 요청"""
    wait_time = 5
    for attempt in range(max_retries):
        try:
            # 요청 전 랜덤 지연 (2~5초)
            time.sleep(random.uniform(2.0, 5.0))
            
            res = curl_requests.get(
                url, 
                impersonate="chrome110", 
                headers=BROWSER_HEADERS, 
                timeout=60
            )
            
            if res.status_code == 429:
                print(f"   [!] 429 Too Many Requests. {wait_time}초 대기 후 재시도 ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
                wait_time *= 2
                continue
            
            res.raise_for_status()
            return res
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"   [!] 최종 요청 실패: {url} ({e})")
                return None
            time.sleep(wait_time)
            wait_time *= 2
    return None

def search_agency_guidelines(agency, site_domain):
    """Serper API를 통한 문서 검색 (단순화된 쿼리 적용)"""
    if not SERPER_API_KEY:
        print(f"[{agency}] Serper API Key missing.")
        return []

    # 400 에러 방지를 위한 단순화된 쿼리
    query = f"site:{site_domain} biosimilar guidance draft submission requirement information"
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": 10})
    headers = {
        'X-API-KEY': str(SERPER_API_KEY).strip(),
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=30)
        if response.status_code != 200:
            print(f"[{agency}] API 오류 ({response.status_code}): {response.text}")
            return []
        
        search_results = response.json().get('organic', [])
        return [{
            "agency": agency,
            "title": item.get('title', ''),
            "snippet": item.get('snippet', ''),
            "url": item.get('link', '')
        } for item in search_results]
    except Exception as e:
        print(f"[{agency}] 검색 실행 중 예외 발생: {e}")
        return []

def filter_links_with_llm(links):
    """LLM을 통한 문서 유효성 검증"""
    if not links: return []
    valid_links = []
    for link in links:
        prompt = f"""
        이 문서가 바이오시밀러 또는 mAb의 개발, 인허가, 제출 요건(Submission Requirements), 또는 공식 초안(Draft) 가이드라인입니까?
        규제 기관의 공식 정보라면 true, 뉴스나 단순 홍보물이라면 false를 반환하세요.
        제목: {link['title']}
        요약: {link['snippet']}
        JSON 응답: {{"is_relevant": true/false}}
        """
        try:
            response = client.models.generate_content(
                model=FILTER_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
            )
            if json.loads(response.text).get("is_relevant"):
                valid_links.append(link)
        except Exception:
            continue
        time.sleep(0.5)
    return valid_links

def download_and_save(pdf_url, doc_info):
    """PDF 다운로드 및 DB 저장"""
    try:
        existing = supabase.table("guidelines").select("url").eq("url", pdf_url).execute()
        if existing.data:
            return

        res = request_with_retry(pdf_url)
        if not res: return

        print(f"   -> 다운로드 성공: {pdf_url}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(res.content)
            temp_path = temp_pdf.name

        raw_text = extract_text_with_ocr(temp_path)
        supabase.table("guidelines").insert({
            "title": doc_info['title'],
            "agency": doc_info['agency'],
            "category": "Biosimilar/mAb",
            "url": pdf_url,
            "raw_text": raw_text
        }).execute()
        
        print(f"   -> DB 저장 완료")
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"   -> 처리 오류: {e}")

def process_document(link):
    """PDF 여부 판별 및 HTML 내 PDF 탐색"""
    url = link['url']
    if url.lower().endswith('.pdf'):
        download_and_save(url, link)
    else:
        res = request_with_retry(url)
        if not res: return
        
        try:
            soup = BeautifulSoup(res.text, 'html.parser')
            pdf_links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.lower().endswith('.pdf'):
                    pdf_links.append(urljoin(url, href))
            
            # 발견된 PDF 중 중복 제거 후 상위 3개 처리
            unique_pdfs = list(set(pdf_links))
            if unique_pdfs:
                print(f"   -> HTML 내 {len(unique_pdfs)}개의 PDF 발견")
                for pdf_url in unique_pdfs[:3]:
                    download_and_save(pdf_url, link)
            else:
                print(f"   -> HTML 내 PDF 링크 없음")
        except Exception as e:
            print(f"   -> HTML 파싱 실패: {e}")

def run_scraper():
    print("--- 지능형 규제 가이드라인 수집기 가동 ---")
    
    agencies = [
        {"name": "FDA", "domain": "fda.gov"},
        {"name": "EMA", "domain": "ema.europa.eu"},
        {"name": "MHRA", "domain": "gov.uk"},
        {"name": "Health Canada", "domain": "canada.ca"},
        {"name": "Health Canada (Sub)", "domain": "hc-sc.gc.ca"},
    ]

    all_links = []
    for agency in agencies:
        print(f"\n[{agency['name']}] 검색 중...")
        raw_results = search_agency_guidelines(agency['name'], agency['domain'])
        if raw_results:
            print(f" -> {len(raw_results)}개 발견. 필터링 중...")
            filtered = filter_links_with_llm(raw_results)
            print(f" -> {len(filtered)}개 유효 판별.")
            all_links.extend(filtered)
        time.sleep(2)

    print("\n--- 문서 수집 및 분석 시작 ---")
    for link in all_links:
        print(f"\n[대상] {link['title']}")
        process_document(link)
        time.sleep(random.uniform(3, 6))

    print("\n--- 모든 수집 작업 종료 ---")

if __name__ == "__main__":
    run_scraper()
