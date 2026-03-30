import os
import time
import json
import re
import tempfile
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

def search_agency_guidelines(agency, site_domain):
    """검색 쿼리 및 결과 수를 확장하여 검색 수행"""
    if not SERPER_API_KEY:
        print("Serper API Key missing.")
        return []

    # 쿼리 강화: filetype:pdf 제거하여 HTML 결과도 수집
    query = f'site:{site_domain} "biosimilar" (guidance OR guideline OR draft OR submission OR requirement OR information)'
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": 30})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=30)
        response.raise_for_status()
        return [{
            "agency": agency,
            "title": item.get('title', ''),
            "snippet": item.get('snippet', ''),
            "url": item.get('link', '')
        } for item in response.json().get('organic', [])]
    except Exception as e:
        print(f"[{agency}] 검색 오류: {e}")
        return []

def filter_links_with_llm(links):
    """LLM을 통한 문서 유효성 검증 (Draft 및 규제 정보 포함)"""
    if not links: return []
    valid_links = []
    for link in links:
        prompt = f"""
        이 문서가 바이오시밀러 또는 mAb의 개발, 인허가, 제출 요건(Submission Requirements), 또는 공식 초안(Draft) 가이드라인입니까?
        규제 기관의 공식 정보라면 true, 뉴스나 단순 홍보물이라면 false를 반환하세요.
        문서 제목: {link['title']}
        문서 요약: {link['snippet']}
        URL: {link['url']}
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
    """실제 PDF 다운로드 및 DB 저장 공통 로직"""
    try:
        # 중복 체크
        existing = supabase.table("guidelines").select("url").eq("url", pdf_url).execute()
        if existing.data:
            return

        print(f"   -> 다운로드: {pdf_url}")
        res = curl_requests.get(pdf_url, impersonate="chrome110", timeout=60)
        res.raise_for_status()

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
    """URL이 PDF이면 바로 저장, HTML이면 내부 PDF 탐색 후 저장"""
    url = link['url']
    if url.lower().endswith('.pdf'):
        download_and_save(url, link)
    else:
        # HTML 페이지에서 PDF 링크 탐색
        try:
            res = curl_requests.get(url, impersonate="chrome110", timeout=30)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, 'html.parser')
            # 'biosimilar' 관련 텍스트가 포함되거나 'pdf' 확장자를 가진 링크 추출
            pdf_links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.lower().endswith('.pdf'):
                    pdf_links.append(urljoin(url, href))
            
            # 중복 제거 후 저장 (주요 문서 1~2개 우선 처리)
            for pdf_url in list(set(pdf_links))[:3]:
                download_and_save(pdf_url, link)
        except Exception as e:
            print(f"   -> HTML 파싱 실패: {url} ({e})")

def run_scraper():
    print("--- 전방위 규제 가이드라인 수집기 가동 ---")
    
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
        print(f" -> {len(raw_results)}개 결과 발견. 필터링 시작...")
        filtered = filter_links_with_llm(raw_results)
        print(f" -> {len(filtered)}개 유효 문서 판별됨.")
        all_links.extend(filtered)
        time.sleep(1)

    print("\n--- 문서 수집 및 분석 시작 ---")
    for link in all_links:
        print(f"\n[대상] {link['title']}")
        process_document(link)
        time.sleep(1)

    print("\n--- 수집 완료 ---")

if __name__ == "__main__":
    run_scraper()
