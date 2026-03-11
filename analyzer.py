import os
import time
import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from google import genai
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

def extract_text_from_url(url):
    try:
        # 방화벽 차단 방지를 위한 기본 브라우저 헤더 추가
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            return None
        
        # 1. URL 자체가 PDF인 경우
        if b"%PDF" in response.content[:5]:
            doc = fitz.open(stream=response.content, filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            return text
        
        # 2. URL이 웹페이지(HTML)인 경우 내부에서 PDF 다운로드 링크 탐색
        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_link = None
        
        for a_tag in soup.find_all("a", href=True):
            if a_tag['href'].lower().endswith(".pdf"):
                pdf_link = a_tag['href']
                break
                
        if pdf_link:
            # 상대 경로(예: /documents/...)를 절대 경로(https://...)로 변환
            full_pdf_url = urljoin(url, pdf_link)
            
            pdf_response = requests.get(full_pdf_url, headers=headers, timeout=30)
            if pdf_response.status_code == 200 and b"%PDF" in pdf_response.content[:5]:
                doc = fitz.open(stream=pdf_response.content, filetype="pdf")
                text = "".join(page.get_text() for page in doc)
                return text

        return None
    except Exception as e:
        print(f"Text extraction failed for {url}: {e}")
        return None

def generate_summary(text):
    if not text or len(text) < 100:
        return "본문 텍스트를 충분히 추출하지 못했습니다."
    
    truncated_text = text[:10000]
    
    prompt = f"""
    당신은 RA(Regulatory Affairs) 전문가입니다. 아래의 규제 가이드라인 원문 텍스트를 읽고,
    바이오시밀러 및 단일클론항체 개발 실무자를 위해 핵심 요구사항 3가지를 한국어로 명확하게 요약하십시오.
    
    [가이드라인 텍스트]
    {truncated_text}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "AI 요약 생성 중 오류가 발생했습니다."

def process_unsummarized_docs():
    print("--- Starting AI Summarization ---")
    
    response = supabase.table("guidelines").select("*").is_("ai_summary", "null").limit(5).execute()
    docs = response.data
    
    if not docs:
        print("요약이 필요한 새 문서가 없습니다.")
        return
        
    for doc in docs:
        print(f"Processing: {doc['title']}")
        
        text = extract_text_from_url(doc['url'])
        if text:
            summary = generate_summary(text)
            
            supabase.table("guidelines").update({"ai_summary": summary}).eq("url", doc["url"]).execute()
            print(" -> 요약 완료 및 DB 업데이트 성공")
        else:
            print(" -> 텍스트 추출 실패 (PDF 링크를 찾을 수 없거나 접근 불가)")
            supabase.table("guidelines").update({"ai_summary": "PDF 텍스트 추출 불가"}).eq("url", doc["url"]).execute()
            
        time.sleep(10)

if __name__ == "__main__":
    process_unsummarized_docs()
