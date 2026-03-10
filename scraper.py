import os
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
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

def fetch_fda_guidelines():
    keywords = ["biosimilar", "monoclonal antibody"] 
    guidelines = []
    print("--- Starting FDA Scraping ---")
    
    url = "https://www.fda.gov/regulatory-information/search-fda-guidance-documents"
    
    for keyword in keywords:
        params = {"keys": keyword}
        try:
            response = curl_requests.get(url, params=params, impersonate="chrome110", timeout=30)
            print(f"FDA ({keyword}) Response Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # [디버깅 추가] 페이지 제목과 본문 앞부분을 출력하여 실제 수신된 페이지 확인
                page_title = soup.title.text.strip() if soup.title else "No Title"
                print(f" -> [DEBUG] Page Title: {page_title}")
                
                # 만약 차단 페이지라면 "Just a moment..." 또는 "Access Denied" 등이 출력됩니다.
                if "Just a moment" in page_title or "Access Denied" in page_title:
                    print(" -> [DEBUG] Blocked by CAPTCHA/Security challenge.")
                    continue
                
                table_rows = soup.select("table tbody tr")
                print(f" -> [DEBUG] Number of <tr> tags found: {len(table_rows)}")
                
                # 기존 파싱 로직 생략 (원인 파악이 우선이므로)
                
            else:
                print(f" -> Failed. Status code: {response.status_code}")
        except Exception as e:
            print(f" -> Error during FDA scraping: {e}")
            
    return guidelines

def fetch_ema_biosimilar_guidelines():
    print("\n--- Starting EMA Scraping ---")
    url = "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/scientific-guidelines/multidisciplinary-guidelines/multidisciplinary-guidelines-biosimilar"
    
    try:
        response = curl_requests.get(url, impersonate="chrome110", timeout=30)
        print(f"EMA Response Status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # [디버깅 추가] EMA 페이지 제목 확인
            page_title = soup.title.text.strip() if soup.title else "No Title"
            print(f" -> [DEBUG] Page Title: {page_title}")
            
            # 모든 <a> 태그 개수 확인
            a_tags = soup.find_all("a")
            print(f" -> [DEBUG] Total <a> tags found on page: {len(a_tags)}")
            
        else:
            print(f" -> Failed. Status code: {response.status_code}")
    except Exception as e:
         print(f" -> Error during EMA scraping: {e}")
         
    return [] # 테스트 목적이므로 빈 리스트 반환

def save_to_supabase(guidelines):
    print(f"\n--- Saving {len(guidelines)} items to Supabase ---")
    pass # 테스트 목적이므로 DB 저장 단계 생략

if __name__ == "__main__":
    fda_docs = fetch_fda_guidelines()
    ema_docs = fetch_ema_biosimilar_guidelines()
