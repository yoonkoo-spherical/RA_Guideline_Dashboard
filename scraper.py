import os
import time
import requests
from bs4 import BeautifulSoup
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
        'render': render
    }
    try:
        response = requests.get('https://api.scraperapi.com/', params=payload, timeout=60)
        if response.status_code == 200:
            return response.text
        else:
            print(f" -> API Error {response.status_code} for URL: {url}")
            return None
    except Exception as e:
        print(f" -> Connection Error for {url}: {e}")
        return None

def fetch_fda_guidelines():
    keywords = ["biosimilar", "monoclonal antibody"] 
    guidelines = []
    print("--- Starting FDA Scraping via ScraperAPI ---")
    
    for keyword in keywords:
        target_url = f"https://www.fda.gov/regulatory-information/search-fda-guidance-documents?keys={keyword}"
        html = fetch_with_scraperapi(target_url, render='true')
        
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            table_rows = soup.select("table tbody tr")
            count = 0
            for row in table_rows:
                cols = row.find_all("td")
                if len(cols) >= 4:
                    title_tag = cols[0].find("a")
                    if not title_tag: continue
                    
                    title = title_tag.text.strip()
                    href = title_tag.get('href', '')
                    link = "https://www.fda.gov" + href if href.startswith("/") else href
                    guidelines.append({
                        "agency": "FDA", "title": title, "url": link,
                        "status": cols[1].text.strip(), "published_date": cols[3].text.strip(),
                        "category": keyword
                    })
                    count += 1
            print(f" -> Found {count} rows for keyword '{keyword}'")
    return guidelines

def fetch_ema_biosimilar_guidelines():
    print("\n--- Starting EMA Scraping via ScraperAPI ---")
    target_url = "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/scientific-guidelines/multidisciplinary-guidelines/multidisciplinary-guidelines-biosimilar"
    html = fetch_with_scraperapi(target_url)
    
    guidelines = []
    if html:
        soup = BeautifulSoup(html, 'html.parser')
        for a_tag in soup.find_all("a", href=True):
            href = a_tag.get('href', '')
            title = a_tag.text.strip()
            if not title or "#" in href or "multidisciplinary-guidelines" in href.lower(): continue
            
            if "guideline" in href.lower() or href.lower().endswith(".pdf"):
                full_url = href if href.startswith("http") else "https://www.ema.europa.eu" + href
                guidelines.append({
                    "agency": "EMA", "title": title, "url": full_url,
                    "status": "Final", "published_date": "N/A", "category": "biosimilar"
                })
    return list({doc['url']: doc for doc in guidelines}.values())

def fetch_mhra_guidelines():
    print("\n--- Starting UK MHRA Scraping via ScraperAPI ---")
    # 1. 바이오시밀러 메인 지침 페이지
    main_url = "https://www.gov.uk/government/publications/guidance-on-the-licensing-of-biosimilar-products/guidance-on-the-licensing-of-biosimilar-products"
    guidelines = []
    
    # 2. 검색 포털 (Biosimilar 검색 결과)
    search_url = "https://www.gov.uk/search/all?organisations%5B%5D=medicines-and-healthcare-products-regulatory-agency&order=updated-newest&keywords=biosimilar"
    html = fetch_with_scraperapi(search_url)
    
    if html:
        soup = BeautifulSoup(html, 'html.parser')
        results = soup.select(".gem-c-document-list__item")
        for item in results:
            link_tag = item.find("a")
            if link_tag:
                title = link_tag.text.strip()
                url = "https://www.gov.uk" + link_tag['href'] if link_tag['href'].startswith("/") else link_tag['href']
                date = item.find("time").get("datetime")[:10] if item.find("time") else "N/A"
                guidelines.append({
                    "agency": "MHRA", "title": title, "url": url,
                    "status": "Final", "published_date": date, "category": "biosimilar"
                })
    return guidelines

def fetch_health_canada_guidelines():
    print("\n--- Starting Health Canada Scraping via ScraperAPI ---")
    # 가이드라인 목록 페이지
    target_url = "https://www.canada.ca/en/health-canada/services/drugs-health-products/biologics-radiopharmaceuticals-genetic-therapies/applications-submissions/guidance-documents.html"
    html = fetch_with_scraperapi(target_url)
    
    guidelines = []
    if html:
        soup = BeautifulSoup(html, 'html.parser')
        # 테이블 기반 구조 파싱
        rows = soup.select("table tbody tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 2:
                link_tag = cols[0].find("a")
                if link_tag:
                    title = link_tag.text.strip()
                    url = "https://www.canada.ca" + link_tag['href'] if link_tag['href'].startswith("/") else link_tag['href']
                    # 바이오시밀러 관련 문서만 필터링
                    if "biosimilar" in title.lower() or "biologic" in title.lower():
                        guidelines.append({
                            "agency": "Health Canada", "title": title, "url": url,
                            "status": "Final", "published_date": "N/A", "category": "biosimilar"
                        })
    return guidelines

def save_to_supabase(guidelines):
    print(f"\n--- Saving {len(guidelines)} items to Supabase ---")
    if not guidelines: return
    success_count = 0
    for doc in guidelines:
        try:
            supabase.table("guidelines").upsert(doc, on_conflict="url").execute()
            success_count += 1
        except Exception as e:
            print(f"Supabase Insert Error: {e}")
    print(f"Successfully processed {success_count} documents.")

if __name__ == "__main__":
    fda = fetch_fda_guidelines()
    ema = fetch_ema_biosimilar_guidelines()
    mhra = fetch_mhra_guidelines()
    canada = fetch_health_canada_guidelines()
    
    all_docs = fda + ema + mhra + canada
    save_to_supabase(all_docs)
