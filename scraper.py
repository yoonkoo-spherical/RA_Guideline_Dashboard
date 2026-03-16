import os
import time
import requests
import re
import fitz
import pytesseract
from pdf2image import convert_from_bytes
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Supabase Client Error: {e}")
    exit(1)

def extract_text_with_ocr(pdf_bytes):
    try:
        images = convert_from_bytes(pdf_bytes)
        text = "".join(pytesseract.image_to_string(img, lang='eng+kor') for img in images)
        return text if text.strip() else "추출 불가: OCR 실패"
    except Exception:
        return "추출 불가"

def fetch_html_with_scraperapi(url, render='false'):
    if not SCRAPER_API_KEY: return None
    payload = {'api_key': SCRAPER_API_KEY, 'url': url, 'render': render}
    for _ in range(3):
        try:
            res = requests.get('https://api.scraperapi.com/', params=payload, timeout=60)
            if res.status_code == 200: return res.text
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return None

def fetch_binary_with_scraperapi(url):
    if not SCRAPER_API_KEY: return None
    payload = {'api_key': SCRAPER_API_KEY, 'url': url}
    for _ in range(3):
        try:
            res = requests.get('https://api.scraperapi.com/', params=payload, timeout=60)
            if res.status_code == 200 and res.content.startswith(b"%PDF"): return res.content
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return None

def extract_content_robust(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    raw_text = None
    
    try:
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        is_pdf_content_type = "application/pdf" in response.headers.get("Content-Type", "").lower()
        url_lower = url.lower()
        
        is_likely_pdf = url_lower.endswith(".pdf") or "download" in url_lower or is_pdf_content_type
        
        if is_likely_pdf:
            pdf_content = None
            if response.status_code == 200 and response.content.startswith(b"%PDF"):
                pdf_content = response.content
            else:
                pdf_content = fetch_binary_with_scraperapi(url)
                
            if pdf_content and pdf_content.startswith(b"%PDF"):
                doc = fitz.open(stream=pdf_content, filetype="pdf")
                text = "".join(page.get_text() for page in doc)
                raw_text = text if len(text.strip()) >= 50 else extract_text_with_ocr(pdf_content)

        if not raw_text:
            html_content = response.text if response.status_code == 200 else fetch_html_with_scraperapi(url, render='false')
            if not html_content or len(html_content) < 1000: 
                html_content = fetch_html_with_scraperapi(url, render='true')

            if not html_content: 
                raw_text = "추출 불가: 웹페이지 접근 실패"
            else:
                soup = BeautifulSoup(html_content, 'html.parser')
                pdf_links = []
                for a in soup.find_all("a", href=True):
                    href_lower = a['href'].lower()
                    if href_lower.startswith(('mailto:', 'javascript:', 'tel:', '#')): continue
                    if "acrobat" in href_lower or "get.adobe" in href_lower: continue 
                    if ".pdf" in href_lower or "download" in href_lower or "attachment" in href_lower or "/media/" in href_lower:
                        pdf_links.append(urljoin(url, a['href']))

                for pdf_url in pdf_links:
                    try:
                        pdf_res = requests.get(pdf_url, headers=headers, timeout=30)
                        pdf_content = None
                        if pdf_res.status_code == 200 and pdf_res.content.startswith(b"%PDF"):
                            pdf_content = pdf_res.content
                        else:
                            pdf_content = fetch_binary_with_scraperapi(pdf_url)

                        if pdf_content and pdf_content.startswith(b"%PDF"):
                            doc = fitz.open(stream=pdf_content, filetype="pdf")
                            text = "".join(page.get_text() for page in doc)
                            if len(text.strip()) > 50: 
                                raw_text = text
                                break
                            ocr_text = extract_text_with_ocr(pdf_content)
                            if not ocr_text.startswith("추출 불가"): 
                                raw_text = ocr_text
                                break
                    except Exception:
                        continue 

                if not raw_text:
                    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
                        tag.extract()
                    
                    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'govspeak|main-content|content|mws-body|page-body|container'))
                    html_text = main_content.get_text(separator='\n', strip=True) if main_content else soup.body.get_text(separator='\n', strip=True) if soup.body else ""
                    
                    if len(html_text.strip()) > 100: 
                        raw_text = html_text
                    else:
                        raw_text = "추출 불가: HTML 본문 부족"
                        
    except Exception as e:
        raw_text = f"추출 불가: 예외 발생 ({e})"
        
    # [핵심 수정] PostgreSQL DB 저장 전 Null Byte 및 제어 문자 강제 제거
    if raw_text:
        raw_text = raw_text.replace('\x00', '').replace('\u0000', '')
    
    return raw_text

def get_crawler_configs():
    try:
        response = supabase.table("crawler_configs").select("*").execute()
        return response.data
    except Exception as e:
        print(f"설정 로드 실패: {e}")
        return []

def get_manual_upload_keywords():
    try:
        response = supabase.table("guidelines").select("agency, category").ilike("url", "%guidelines_pdf%").execute()
        manual_docs = response.data
        keywords_by_agency = {}
        for doc in manual_docs:
            agency = doc.get("agency")
            category = doc.get("category")
            if agency and category:
                agency_upper = agency.strip().upper()
                if agency_upper not in keywords_by_agency:
                    keywords_by_agency[agency_upper] = set()
                keywords_by_agency[agency_upper].add(category.strip().lower())
        return keywords_by_agency
    except Exception as e:
        return {}

def get_base_domain(url):
    domain = urlparse(url).netloc
    parts = domain.split('.')
    return '.'.join(parts[-2:]) if len(parts) > 1 else domain

def discover_guidelines(agency, start_url, keywords, current_depth=0, max_depth=2, visited=None, base_domain=None, parent_is_relevant=False):
    if visited is None: visited = set()
    if base_domain is None: base_domain = get_base_domain(start_url)

    normalized_url = start_url.split('#')[0].rstrip('/')
    if normalized_url in visited or current_depth > max_depth: return []
    
    visited.add(normalized_url)
    print(f"[{agency}] 탐색 중 (Depth {current_depth}): {normalized_url}")
    
    html = fetch_html_with_scraperapi(normalized_url, render='false')
    if not html or len(html) < 500:
        html = fetch_html_with_scraperapi(normalized_url, render='true')
    
    if not html: return []

    soup = BeautifulSoup(html, 'html.parser')
    found_guidelines = []
    page_title = soup.title.string if soup.title else ""
    current_page_is_relevant = parent_is_relevant or any(kw.lower() in page_title.lower() or kw.lower() in normalized_url.lower() for kw in keywords)
    
    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get('href', '').strip()
        if not href or href.startswith(('mailto:', 'tel:', 'javascript:')): continue
            
        title = a_tag.get_text(separator=' ', strip=True) 
        full_url = urljoin(start_url, href).split('#')[0]
        
        is_doc = any(ext in full_url.lower() for ext in [".pdf", "download", "guidance-documents", "/file/"])
        is_relevant = any(kw.lower() in title.lower() or kw.lower() in full_url.lower() for kw in keywords)
        link_is_relevant = is_relevant or current_page_is_relevant

        if is_doc and link_is_relevant:
            found_guidelines.append({
                "agency": agency,
                "title": title if title else full_url.split('/')[-1],
                "url": full_url,
                "status": "Final",
                "published_date": "N/A", 
                "category": keywords[0] if keywords else "general"
            })
            continue

        if is_relevant and not is_doc and current_depth < max_depth:
            link_domain = get_base_domain(full_url)
            if base_domain in link_domain:
                found_guidelines.extend(discover_guidelines(
                    agency, full_url, keywords, current_depth + 1, max_depth, visited, base_domain, is_relevant
                ))
    return found_guidelines

def backfill_missing_texts():
    print("\n--- 기존 DB 문서 중 raw_text 누락 건 텍스트 추출 작업 ---")
    missing_docs = supabase.table("guidelines").select("url").is_("raw_text", "null").execute().data
    if not missing_docs:
        print("누락된 텍스트가 없습니다.")
        return
    
    print(f"총 {len(missing_docs)}건의 텍스트를 보완 추출합니다.")
    for doc in missing_docs:
        url = doc['url']
        print(f" -> 텍스트 추출 시도: {url}")
        raw_text = extract_content_robust(url)
        try:
            supabase.table("guidelines").update({"raw_text": raw_text}).eq("url", url).execute()
        except Exception as e:
            print(f"    - DB 업데이트 오류: {e}")
        time.sleep(1)

def save_to_supabase(guidelines):
    if not guidelines: return
    unique_docs = list({doc['url']: doc for doc in guidelines}.values())
    
    try:
        existing_records = supabase.table("guidelines").select("url").execute()
        existing_urls = {record['url'] for record in existing_records.data}
    except Exception as e:
        existing_urls = set()

    new_docs = [doc for doc in unique_docs if doc['url'] not in existing_urls]
    print(f"\n총 신규 탐색 문서: {len(new_docs)}건")

    if not new_docs: return
    
    success_count = 0
    for doc in new_docs:
        try:
            print(f" -> 신규 문서 텍스트 추출 중: {doc['url']}")
            doc['raw_text'] = extract_content_robust(doc['url'])
            supabase.table("guidelines").upsert(doc, on_conflict="url").execute()
            success_count += 1
            time.sleep(1)
        except Exception as e:
            print(f"저장 에러 ({doc['url']}): {e}")
            
    print(f"최종 {success_count} 건 신규 저장 완료.")

if __name__ == "__main__":
    backfill_missing_texts()

    configs = get_crawler_configs()
    if not configs:
        configs = [
            {"agency": "FDA", "base_url": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents", "keywords": "biosimilar,monoclonal"},
            {"agency": "EMA", "base_url": "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/scientific-guidelines", "keywords": "biosimilar,monoclonal"},
            {"agency": "MHRA", "base_url": "https://www.gov.uk/government/collections/mhra-guidance-on-biosimilar-products", "keywords": "biosimilar"},
            {"agency": "Health Canada", "base_url": "https://www.canada.ca/en/health-canada/services/drugs-health-products/biologics-radiopharmaceuticals-genetic-therapies/applications-submissions/guidance-documents.html", "keywords": "biosimilar,biologic"}
        ]

    manual_keywords = get_manual_upload_keywords()
    for config in configs:
        agency = config['agency']
        agency_upper = agency.strip().upper()
        existing_keywords = set(k.strip() for k in config.get('keywords', 'biosimilar').split(','))
        if agency_upper in manual_keywords:
            existing_keywords.update(manual_keywords[agency_upper])
        config['keywords'] = ",".join(list(existing_keywords))

    all_collected_docs = []
    for config in configs:
        agency = config['agency']
        start_url = config['base_url']
        keywords = [k.strip() for k in config.get('keywords', '').split(',') if k.strip()]
        docs = discover_guidelines(agency, start_url, keywords)
        all_collected_docs.extend(docs)

    save_to_supabase(all_collected_docs)
