import os
import cloudscraper
from bs4 import BeautifulSoup
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Supabase Client Error: {e}")
    exit(1)

# Cloudscraper 초기화 (일반 브라우저처럼 위장하여 보안 우회)
scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'desktop': True
    }
)

def fetch_fda_guidelines():
    keywords = ["biosimilar", "monoclonal antibody"] 
    guidelines = []
    print("--- Starting FDA Scraping ---")
    
    # FDA 검색 페이지 URL
    url = "https://www.fda.gov/regulatory-information/search-fda-guidance-documents"
    
    for keyword in keywords:
        params = {"keys": keyword}
        try:
            response = scraper.get(url, params=params, timeout=20)
            print(f"FDA ({keyword}) Response Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                table_rows = soup.select("table.views-table tbody tr")
                print(f" -> Found {len(table_rows)} rows for keyword '{keyword}'")
                
                for row in table_rows:
                    cols = row.find_all("td")
                    if len(cols) >= 4:
                        title_tag = cols[0].find("a")
                        title = title_tag.text.strip() if title_tag else "N/A"
                        link = "https://www.fda.gov" + title_tag['href'] if title_tag else ""
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
            else:
                print(f" -> Failed. Status code: {response.status_code}")
        except Exception as e:
            print(f" -> Error during FDA scraping: {e}")
            
    return guidelines

def fetch_ema_biosimilar_guidelines():
    print("\n--- Starting EMA Scraping ---")
    # 변경된 EMA 바이오시밀러 가이드라인 URL
    url = "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/scientific-guidelines/multidisciplinary-guidelines/multidisciplinary-guidelines-biosimilar"
    
    try:
        response = scraper.get(url, timeout=20)
        print(f"EMA Response Status: {response.status_code}")
        
        guidelines = []
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            document_items = soup.select(".ecl-file") 
            print(f" -> Found {len(document_items)} documents from EMA")
            
            for item in document_items:
                title_tag = item.select_one(".ecl-file__title")
                title = title_tag.text.strip() if title_tag else "N/A"
                link_tag = item.select_one("a")
                link = link_tag['href'] if link_tag else ""
                
                guidelines.append({
                    "agency": "EMA",
                    "title": title,
                    "url": link,
                    "status": "Final",
                    "published_date": "N/A", 
                    "category": "biosimilar"
                })
        else:
            print(f" -> Failed. Status code: {response.status_code}")
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
