import os
import time
import requests
import re
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

def fetch_with_scraperapi(url, render='false'):
    payload = {
        'api_key': SCRAPER_API_KEY,
        'url': url,
        'render': render,
        'keep_headers': 'true'
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get('https://api.scraperapi.com/', params=payload, timeout=90)
            if response.status_code == 200:
                return response.text
            elif response.status_code in [429, 500, 502, 503]:
                time.sleep(5)
            else:
                print(f" -> API Error {response.status_code} for URL: {url}")
                return None
        except Exception as e:
            print(f" -> Connection Error for {url} (Attempt {attempt+1}): {e}")
            time.sleep(5)
    return None

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
        print(f"수동 업로드 키워드 조회 실패: {e}")
        return {}

def get_base_domain(url):
    domain = urlparse(url).netloc
    parts = domain.split('.')
    return '.'.join(parts[-2:]) if len(parts) > 1 else domain

def discover_guidelines(agency, start_url, keywords, current_depth=0, max_depth=2, visited=None, base_domain=None, parent_is_relevant=False):
    if visited is None:
        visited = set()
        
    if base_domain is None:
        base_domain = get_base_domain(start_url)

    normalized_url = start_url.split('#')[0].rstrip('/')
    if normalized_url in visited or current_depth > max_depth:
        return []
    
    visited.add(normalized_url)
    print(f"[{agency}] 탐색 중 (Depth {current_depth}): {normalized_url}")
    
    # 1차 시도: 1 크레딧 소모 (render='false')
    html = fetch_with_scraperapi(normalized_url, render='false')
    
    # 실패 시 2차 시도: 5 크레딧 소모 (render='true')
    if not html or len(html) < 500:
        html = fetch_with_scraperapi(normalized_url, render='true')
    
    if not html:
        return []

    soup = BeautifulSoup(html, 'html.parser')
    found_guidelines = []

    page_title = soup.title.string if soup.title else ""
    current_page_is_relevant = parent_is_relevant or any(kw.lower() in page_title.lower() or kw.lower() in normalized_url.lower() for kw in keywords)
    
    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get('href', '').strip()
        if not href or href.startswith(('mailto:', 'tel:', 'javascript:')):
            continue
            
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

def save_to_supabase(guidelines):
    if not guidelines:
        print("수집된 문서가 없습니다.")
        return
        
    unique_docs = list({doc['url']: doc for doc in guidelines}.values())
    
    try:
        existing_records = supabase.table("guidelines").select("url").execute()
        existing_urls = {record['url'] for record in existing_records.data}
    except Exception as e:
        print(f"기존 DB URL 조회 실패: {e}")
        existing_urls = set()

    new_docs = [doc for doc in unique_docs if doc['url'] not in existing_urls]

    print(f"\n--- 수집 결과 요약 ---")
    print(f"총 수집 문서: {len(unique_docs)}건")
    print(f"기존 DB 중복 문서: {len(unique_docs) - len(new_docs)}건 (저장 생략)")
    print(f"신규 저장 대상 문서: {len(new_docs)}건")

    if not new_docs:
        print("새로 추가할 문서가 없으므로 저장 단계를 종료합니다.")
        return
    
    success_count = 0
    for doc in new_docs:
        try:
            supabase.table("guidelines").upsert(doc, on_conflict="url").execute()
            success_count += 1
        except Exception as e:
            print(f"저장 에러 ({doc['url']}): {e}")
            
    print(f"최종 {success_count} 건 신규 저장 완료.")

if __name__ == "__main__":
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
        print(f"[{agency}] 최종 스크래핑 키워드 적용: {config['keywords']}")

    all_collected_docs = []

    for config in configs:
        agency = config['agency']
        start_url = config['base_url']
        keywords = [k.strip() for k in config.get('keywords', '').split(',') if k.strip()]
        
        docs = discover_guidelines(agency, start_url, keywords)
        all_collected_docs.extend(docs)

    save_to_supabase(all_collected_docs)
