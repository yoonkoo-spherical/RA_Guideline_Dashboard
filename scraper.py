import os
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
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

def fetch_fda_guidelines(page):
    keywords = ["biosimilar", "monoclonal antibody"] 
    guidelines = []
    print("--- Starting FDA Scraping ---")
    
    for keyword in keywords:
        url = f"https://www.fda.gov/regulatory-information/search-fda-guidance-documents?keys={keyword}"
        try:
            # 타임아웃 30초로 증가
            page.goto(url, wait_until="networkidle", timeout=30000)
            print(f" -> [DEBUG] FDA '{keyword}' Page Title: {page.title()}")
            
            # 테이블 대기 시간 30초로 증가
            page.wait_for_selector("table tbody tr", timeout=30000)
            
            html = page.content()
            soup = BeautifulSoup(html, 'html.parser')
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
        except Exception as e:
            print(f" -> Error during FDA scraping for '{keyword}': {e}")
            
    return guidelines

def fetch_ema_biosimilar_guidelines(page):
    print("\n--- Starting EMA Scraping ---")
    url = "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/scientific-guidelines/multidisciplinary-guidelines/multidisciplinary-guidelines-biosimilar"
    
    guidelines = []
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
        html = page.content()
        soup = BeautifulSoup(html, 'html.parser')
        
        for a_tag in soup.find_all("a", href=True):
            href = a_tag['href']
            title = a_tag.text.strip()
            
            if not title:
                continue
                
            href_lower = href.lower()
            if "guideline" in href_lower or "reflection-paper" in href_lower or "position-statement" in href_lower or href_lower.endswith(".pdf"):
                if href == url or href == url.replace("https://www.ema.europa.eu", ""):
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
    with sync_playwright() as p:
        # 봇 탐지 우회를 위한 브라우저 실행 인자 추가
        browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        
        # 일반 크롬 브라우저와 동일한 User-Agent 설정
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        fda_docs = fetch_fda_guidelines(page)
        ema_docs = fetch_ema_biosimilar_guidelines(page)
        
        browser.close()
        
        all_docs = fda_docs + ema_docs
        save_to_supabase(all_docs)
