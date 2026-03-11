import os
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

def fetch_fda_guidelines():
    keywords = ["biosimilar", "monoclonal antibody"] 
    guidelines = []
    print("--- Starting FDA Scraping via ScraperAPI ---")
    
    for keyword in keywords:
        target_url = f"https://www.fda.gov/regulatory-information/search-fda-guidance-documents?keys={keyword}"
        
        # ScraperAPI 파라미터 구성
        payload = {
            'api_key': SCRAPER_API_KEY,
            'url': target_url,
            'render': 'true' # JS 동적 렌더링 활성화
        }
        
        try:
            # 타임아웃을 60초로 설정 (프록시 렌더링 소요 시간 감안)
            response = requests.get('https://api.scraperapi.com/', params=payload, timeout=60)
            print(f"FDA ({keyword}) Response Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                table_rows = soup.select("table tbody tr")
                
                count = 0
                for row in table_rows:
                    cols = row.find_all("td")
                    if len(cols) >= 4:
                        title_tag = cols[0].find("a")
                        if not title_tag:
                            continue
                        
                        title = title_tag.text.strip()
                        href = title_tag.get('href', '')
                        link = "https://www.fda.gov" + href if href.startswith("/") else href
                        status = cols[1].text.strip()
                        date = cols[3].text.strip()
                        
                        guidelines.append({
                            "agency": "FDA",
                            "title": title,
                            "url": link,
                            "status": status,
                            "published_date": date,
                            "category": keyword
                        })
                        count += 1
                print(f" -> Found {count} rows for keyword '{keyword}'")
            else:
                print(f" -> Failed. Status code: {response.status_code}")
        except Exception as e:
            print(f" -> Error during FDA scraping for '{keyword}': {e}")
            
    return guidelines

def fetch_ema_biosimilar_guidelines():
    print("\n--- Starting EMA Scraping via ScraperAPI ---")
    target_url = "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/scientific-guidelines/multidisciplinary-guidelines/multidisciplinary-guidelines-biosimilar"
    
    payload = {
        'api_key': SCRAPER_API_KEY,
        'url': target_url
        # EMA는 정적 링크 구조이므로 render='true' 생략 가능
    }
    
    guidelines = []
    try:
        response = requests.get('https://api.scraperapi.com/', params=payload, timeout=60)
        print(f"EMA Response Status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for a_tag in soup.find_all("a", href=True):
                href = a_tag['href']
                title = a_tag.text.strip()
                
                if not title:
                    continue
                    
                href_lower = href.lower()
                if "guideline" in href_lower or "reflection-paper" in href_lower or "position-statement" in href_lower or href_lower.endswith(".pdf"):
                    if href == target_url or href == target_url.replace("https://www.ema.europa.eu", ""):
                        continue

                    full_url = href if href.startswith("http") else "https://www.ema.europa.eu" + href
                    
                    guidelines.append({
                        "agency": "EMA",
                        "title": title,
                        "url": full_url,
                        "status": "Final",
                        "published_date": "N/A", 
                        "category": "biosimilar"
                    })
            
            unique_guidelines = {doc['url']: doc for doc in guidelines}.values()
            guidelines = list(unique_guidelines)
            print(f" -> Found {len(guidelines)} unique documents from EMA")

    except Exception as e:
         print(f" -> Error during EMA scraping: {e}")
         
    return guidelines

def save_to_supabase(guidelines):
    print(f"\n--- Saving {len(guidelines)} items to Supabase ---")
    if not guidelines:
        print("No guidelines to save. Skipping Supabase insert.")
        return
        
    success_count = 0
    for doc in guidelines:
        try:
            supabase.table("guidelines").upsert(doc, on_conflict="url").execute()
            success_count += 1
        except Exception as e:
             print(f"Supabase Insert Error on URL '{doc['url']}': {e}")
             
    print(f"Successfully processed {success_count} documents into Supabase.")

if __name__ == "__main__":
    fda_docs = fetch_fda_guidelines()
    ema_docs = fetch_ema_biosimilar_guidelines()
    
    all_docs = fda_docs + ema_docs
    save_to_supabase(all_docs)
