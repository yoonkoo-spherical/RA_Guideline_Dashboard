import os
import requests
from bs4 import BeautifulSoup
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Supabase 클라이언트 초기화 에러 방지
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Supabase Client Error: {e}")
    exit(1)

# 실제 브라우저처럼 보이기 위한 헤더 강화
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

def fetch_fda_guidelines():
    keywords = ["biosimilar", "monoclonal antibody"] 
    guidelines = []
    print("--- Starting FDA Scraping ---")
    
    for keyword in keywords:
        url = "https://www.fda.gov/regulatory-information/search-fda-guidance-documents"
        params = {"keys": keyword}
        
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=15)
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
                print(f" -> Failed to fetch FDA page. Status code: {response.status_code}")
        except Exception as e:
            print(f" -> Error during FDA scraping: {e}")
            
    return guidelines

def fetch_ema_biosimilar_guidelines():
    print("\n--- Starting EMA Scraping ---")
    url = "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/scientific-guidelines/multidisciplinary/multidisciplinary-biosimilar"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
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
            print(f" -> Failed to fetch EMA page. Status code: {response.status_code}")
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
