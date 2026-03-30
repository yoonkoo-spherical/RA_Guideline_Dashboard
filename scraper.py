import os
import time
import json
import re
import tempfile
import random
import urllib3
from urllib.parse import urljoin, urlparse

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

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# 환경 변수 로드
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
PROXY_URL = os.getenv("PROXY_URL") # 예: http://user:pass@proxy.server.com:port

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Supabase Client Error: {e}")
    exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)
FILTER_MODEL = "gemini-3.1-flash-lite-preview"

# 무작위 핑거프린트 프로필 목록
IMPERSONATE_PROFILES = ["chrome110", "chrome116", "edge101", "safari15_5"]

def get_random_headers():
    return {
        "User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(110, 122)}.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

def is_valid_text(text):
    if not text.strip():
        return False
    alphanumeric_count = len(re.findall(r'[a-zA-Z0-9가-힣]', text))
    total_length = len(text.replace(" ", "").replace("\n", ""))
    if total_length == 0:
        return False
    return (alphanumeric_count / total_length) > 0.4

def extract_text_with_ocr(file_path):
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
    wait_time = 5
    # 세션을 유지하여 봇 검증 통과 시 발급되는 쿠키 활용
    session = curl_requests.Session()
    proxies = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else None

    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(3.0, 7.0))
            profile = random.choice(IMPERSONATE_PROFILES)
            
            res = session.get(
                url, 
                impersonate=profile, 
                headers=get_random_headers(), 
                timeout=60,
                proxies=proxies
            )

            if res.status_code in [403, 429]:
                print(f"   [!] {res.status_code} 차단 감지. {wait_time}초 대기 후 프로필 변경 재시도 ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
                wait_time *= 2
                continue

            res.raise_for_status()
            return res
            
        except Exception as e:
            error_msg = str(e)
            if "INTERNAL_ERROR" in error_msg or "time" in error_msg.lower():
                try:
                    time.sleep(3)
                    res_fallback = requests.get(
                        url, 
                        headers=get_random_headers(), 
                        timeout=60,
                        verify=False,
                        proxies=proxies
                    )
                    res_fallback.raise_for_status()
                    return res_fallback
                except Exception as fallback_e:
                    error_msg = f"세션 및 폴백 모두 실패: {fallback_e}"

            if attempt == max_retries - 1:
                return None
            
            time.sleep(wait_time)
            wait_time *= 2
    return None

def search_agency_guidelines(agency, site_domain):
    if not SERPER_API_KEY:
        return []

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
            return []

        search_results = response.json().get('organic', [])
        return [{
            "agency": agency,
            "title": item.get('title', ''),
            "snippet": item.get('snippet', ''),
            "url": item.get('link', '')
        } for item in search_results]
    except Exception:
        return []

def search_alternative_pdf_via_serper(title, original_domain):
    """대상 도메인 접속 불가 시 Serper API를 사용하여 다른 도메인에 업로드된 동일 PDF 탐색"""
    if not SERPER_API_KEY:
        return None

    print(f"   [!] 원본 서버 접근 불가. Serper API로 대체 PDF 링크 검색 시도...")
    # 원본 도메인을 제외하고 해당 제목의 PDF 파일 검색
    query = f'"{title}" filetype:pdf -site:{original_domain}'
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": 3})
    headers = {
        'X-API-KEY': str(SERPER_API_KEY).strip(),
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=30)
        if response.status_code == 200:
            results = response.json().get('organic', [])
            for item in results:
                link = item.get('link', '')
                if link.lower().endswith('.pdf'):
                    return link
    except Exception as e:
        print(f"   [!] 대체 검색 중 예외 발생: {e}")
    return None

def filter_links_with_llm(links):
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

def process_and_save_pdf(pdf_url, doc_info):
    try:
        existing = supabase.table("guidelines").select("url").eq("url", pdf_url).execute()
        if existing.data:
            return True # 이미 존재함

        res = request_with_retry(pdf_url)
        if not res: 
            return False # 다운로드 실패

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
        return True
    except Exception as e:
        print(f"   -> 처리 오류: {e}")
        return False

def download_and_save(pdf_url, doc_info):
    success = process_and_save_pdf(pdf_url, doc_info)
    
    # 다운로드 실패 시 Serper API를 활용한 폴백 로직 실행
    if not success:
        domain = urlparse(pdf_url).netloc
        alt_url = search_alternative_pdf_via_serper(doc_info['title'], domain)
        if alt_url:
            print(f"   -> 대체 PDF 링크 발견: {alt_url}")
            process_and_save_pdf(alt_url, doc_info)
        else:
            print("   -> 대체 PDF 링크를 찾지 못했습니다.")

def process_document(link):
    url = link['url']
    if url.lower().endswith('.pdf'):
        download_and_save(url, link)
    else:
        res = request_with_retry(url)
        
        # HTML 페이지 자체 접근 실패 시 문서 제목으로 PDF 직접 검색 (Fallback)
        if not res:
            domain = urlparse(url).netloc
            alt_url = search_alternative_pdf_via_serper(link['title'], domain)
            if alt_url:
                print(f"   -> HTML 접근 불가. 대체 PDF 링크 발견: {alt_url}")
                process_and_save_pdf(alt_url, link)
            else:
                print(f"   -> 최종 요청 실패 및 대체 탐색 불가: {url}")
            return

        try:
            soup = BeautifulSoup(res.text, 'html.parser')
            pdf_links = []
            
            for a in soup.find_all('a', href=True):
                href = a['href'].lower()
                text = a.get_text().lower()
                
                if '.pdf' in href:
                    pdf_links.append(urljoin(url, a['href']))
                elif '/media/' in href and '/download' in href:
                    pdf_links.append(urljoin(url, a['href']))
                elif 'pdf' in text and not href.startswith(('javascript:', '#', 'mailto:')):
                    pdf_links.append(urljoin(url, a['href']))

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
