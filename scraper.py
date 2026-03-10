import os
import requests
from bs4 import BeautifulSoup
from supabase import create_client, Client
from dotenv import load_dotenv

# 환경 변수 로드 (Supabase 접속 정보)
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_fda_guidelines():
    """
    FDA 웹사이트에서 지정된 키워드 목록을 순회하며 가이드라인을 수집합니다.
    """
    # 검색을 원하는 키워드 리스트 (필요시 이 배열에 추가/수정)
    keywords = ["biosimilar", "monoclonal antibody"] 
    guidelines = []
    
    for keyword in keywords:
        url = "https://www.fda.gov/regulatory-information/search-fda-guidance-documents"
        params = {"keys": keyword}
        headers = {"User-Agent": "Mozilla/5.0"}
        
        response = requests.get(url, params=params, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        table_rows = soup.select("table.views-table tbody tr")
        
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
                    "category": keyword # 검색에 사용된 키워드를 카테고리로 저장
                })
    return guidelines

def fetch_ema_biosimilar_guidelines():
    """
    EMA 웹사이트의 Multidisciplinary: Biosimilar 섹션에서 문서를 수집합니다.
    """
    url = "https://www.ema.europa.eu/en/human-regulatory-overview/research-development/scientific-guidelines/multidisciplinary/multidisciplinary-biosimilar"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    guidelines = []
    document_items = soup.select(".ecl-file") 
    
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
            "category": "biosimilar" # 고정 카테고리 지정
        })
    return guidelines

def save_to_supabase(guidelines):
    """
    수집된 가이드라인 데이터를 Supabase DB에 저장 (URL 기준 중복 방지)
    """
    if not guidelines:
        return
        
    for doc in guidelines:
        response = supabase.table("guidelines").upsert(
            doc, on_conflict="url"
        ).execute()
        
    print(f"Total {len(guidelines)} documents processed into Supabase.")

if __name__ == "__main__":
    fda_docs = fetch_fda_guidelines()
    ema_docs = fetch_ema_biosimilar_guidelines()
    
    all_docs = fda_docs + ema_docs
    save_to_supabase(all_docs)
