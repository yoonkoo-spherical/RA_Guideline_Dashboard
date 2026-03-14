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
    """ScraperAPI 프록시를 통한 공통 요청 처리 함수"""
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
                time.sleep(5) # Rate limit 또는 서버 오류 시 대기 후 재시도
            else:
                print(f" -> API Error {response.status_code} for URL: {url}")
                return None
        except Exception as e:
            print(f" -> Connection Error for {url} (Attempt {attempt+1}): {e}")
            time.sleep(5)
    return None

def get_crawler_configs():
    """DB에서 수집 대상 기관 및 베이스 URL 설정을 가져옴"""
    try:
        response = supabase.table("crawler_configs").select("*").execute()
        return response.data
    except Exception as e:
        print(f"설정 로드 실패: {e}")
        return []

def get_base_domain(url):
    """URL에서 베이스 도메인만 추출 (예: www.fda.gov -> fda.gov)"""
    domain = urlparse(url).netloc
    parts = domain.split('.')
    return '.'.join(parts[-2:]) if len(parts) > 1 else domain

def discover_guidelines(agency, start_url, keywords, current_depth=0, max_depth=2, visited=None, base_domain=None):
    """페이지 내 링크를 탐색하여 가이드라인과 상세 지침을 재귀적으로 수집"""
    if visited is None:
        visited = set()
        
    if base_domain is None:
        base_domain = get_base_domain(start_url)

    # 정규화된 URL로 방문 여부 체크 (무한 루프 방지)
    normalized_url = start_url.split('#')[0].rstrip('/')
    if normalized_url in visited or current_depth > max_depth:
        return []
    
    visited.add(normalized_url)
    print(f"[{agency}] 탐색 중 (Depth {current_depth}): {normalized_url}")
    
    html = fetch_with_scraperapi(normalized_url, render='true')
    
    if not html:
        return []

    soup = BeautifulSoup(html, 'html.parser')
    found_guidelines = []
    
    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get('href', '').strip()
        if not href or href.startswith(('mailto:', 'tel:', 'javascript:')):
            continue
            
        title = a_tag.get_text(separator=' ', strip=True) 
        full_url = urljoin(start_url, href).split('#')[0]
        
        is_doc = any(ext in full_url.lower() for ext in [".pdf", "download", "guidance-documents"])
        is_relevant = any(kw.lower() in title.lower() or kw.lower() in full_url.lower() for kw in keywords)

        if is_doc and is_relevant:
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
                    agency, full_url, keywords, current_depth + 1, max_depth, visited, base_domain
                ))

    return found_guidelines

def save_to_supabase(guidelines):
    """수집된 데이터를 검증하고 DB에 존재하지 않는 신규 문서만 저장"""
    if not guidelines:
        print("수집된 문서가 없습니다.")
        return
        
    # 1. 스크립트 실행 중 중복 수집된 문서 제거 (URL 기준)
    unique_docs = list({doc['url']: doc for doc in guidelines}.values())
    
    # 2. Supabase DB에 기존에 저장된 URL 목록 가져오기 (비교용)
    try:
        existing_records = supabase.table("guidelines").select("url").execute()
        existing_urls = {record['url'] for record in existing_records.data}
    except Exception as e:
        print(f"기존 DB URL 조회 실패: {e}")
        existing_urls = set()

    # 3. DB에 없는 새로운 문서만 필터링
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
            # 중복 검증을 마쳤으므로 바로 insert 처리 (에러 방지용으로 upsert 유지)
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

    all_collected_docs = []

    for config in configs:
        agency = config['agency']
        start_url = config['base_url']
        keywords = [k.strip() for k in config.get('keywords', 'biosimilar').split(',')]
        
        docs = discover_guidelines(agency, start_url, keywords)
        all_collected_docs.extend(docs)

    save_to_supabase(all_collected_docs)
