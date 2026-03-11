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

def fetch_fda_guidelines():
    keywords = ["biosimilar", "monoclonal antibody"] 
    guidelines = []
    print("--- Starting FDA Scraping via ScraperAPI ---")
    
    for keyword in keywords:
        target_url = f"https://www.fda.gov/regulatory-information/search-fda-guidance-documents?keys={keyword}"
        payload = {
            'api_key': SCRAPER_API_KEY,
            'url': target_url,
            'render': 'true'
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"FDA ({keyword}) - Attempt {attempt + 1}/{max_retries}")
                response = requests.get('https://api.scraperapi.com/', params=payload, timeout=60)
                
                if response.status_code == 200:
                    print(f" -> Success (Status 200)")
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
                    break
                
                elif response.status_code >= 500:
                    print(f" -> Server Error {response.status_code}. Retrying in 5 seconds...")
                    time.sleep(5)
                
                else:
                    print(f" -> Failed. Status code: {response.status_code}")
                    break
                    
            except Exception as e:
                print(f" -> Network/Timeout Error: {e}. Retrying in 5 seconds...")
                time.sleep(5)
            
    return guidelines

def fetch_ema_biosimilar_guidelines():
    print("\n--- Starting EMA Scraping via ScraperAPI ---")
    target_url = "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/scientific-guidelines/multidisciplinary-guidelines/multidisciplinary-guidelines-biosimilar"
    
    payload = {
        'api_key': SCRAPER_API_KEY,
        'url': target_url
    }
    
    guidelines = []
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"EMA - Attempt {attempt + 1}/{max_retries}")
            response = requests.get('https://api.scraperapi.com/', params=payload, timeout=60)
            
            if response.status_code == 200:
                print(f" -> Success (Status 200)")
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag.get('href', '')
                    title = a_tag.text.strip()
                    
                    if not title or not href:
                        continue
                        
                    if "#" in href:
                        continue
                    
                    if href.endswith("multidisciplinary-guidelines") or href.endswith("scientific-guidelines"):
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
                break
                
            elif response.status_code >= 500:
                print(f" -> Server Error {response.status_code}. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f" -> Failed. Status code: {response.status_code}")
                break
                
        except Exception as e:
             print(f" -> Network/Timeout Error: {e}. Retrying in 5 seconds...")
             time.sleep(5)
             
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
