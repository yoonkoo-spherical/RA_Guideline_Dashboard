import os
import time
import requests
import fitz  # PyMuPDF
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from google import genai
from google.genai import types
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
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200: return None
        if b"%PDF" in response.content[:5]:
            doc = fitz.open(stream=response.content, filetype="pdf")
            return "".join(page.get_text() for page in doc)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_link = next((a['href'] for a in soup.find_all("a", href=True) if a['href'].lower().endswith(".pdf")), None)
        if pdf_link:
            full_pdf_url = urljoin(url, pdf_link)
            pdf_response = requests.get(full_pdf_url, headers=headers, timeout=30)
            if pdf_response.status_code == 200 and b"%PDF" in pdf_response.content[:5]:
                doc = fitz.open(stream=pdf_response.content, filetype="pdf")
                return "".join(page.get_text() for page in doc)
        return None
    except Exception as e:
        print(f"Extraction failed for {url}: {e}")
        return None

def analyze_document(text):
    if not text or len(text) < 100:
        return {"ref_number": "N/A", "summary": "텍스트 추출 실패"}
    
    # Step 8: 식별자 추출 프롬프트 고도화
    prompt = f"""
    당신은 RA 전문가입니다. 가이드라인 원문을 읽고 두 항목을 추출하십시오.
    
    1. ref_number: 고유 식별자. 
       - FDA 문서인 경우 'Docket Number' (예: FDA-2019-D-5390)를 찾으십시오.
       - EMA 문서인 경우 'EMEA/CHMP/...', 'EMA/...', 'CPMP/...' 형태의 참조 번호를 찾으십시오.
       - 명확하지 않으면 반드시 "N/A"로 기재하십시오.
    2. summary: 바이오시밀러 개발 실무자를 위한 핵심 요구사항 3가지 (한국어)
    
    [가이드라인 텍스트]
    {text[:10000]}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "ref_number": {"type": "STRING"},
                        "summary": {"type": "STRING"}
                    },
                    "required": ["ref_number", "summary"]
                }
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Analysis Error: {e}")
        return {"ref_number": "N/A", "summary": "요약 생성 오류"}

def compare_documents(old_text, new_text):
    # Step 8: 비교 리포트 출력 형식 규격화 (표 형태 강제)
    prompt = f"""
    당신은 RA 전문가입니다. 구버전과 신버전 텍스트를 비교하여 실무적으로 변경된 사항을 분석하십시오.
    비교 결과는 반드시 아래의 마크다운 표(Table) 형식을 포함하여 작성하십시오.
    
    | 변경 항목 | 구버전 (기존 기준) | 신버전 (변경된 기준) | 실무 영향도 (High/Med/Low) |
    |---|---|---|---|
    | ... | ... | ... | ... |
    
    표 작성 후, 하단에 2~3문장으로 종합적인 결론을 덧붙이십시오.
    
    [구버전 텍스트 일부]
    {old_text[:8000]}
    
    [신규 텍스트 일부]
    {new_text[:8000]}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"비교 분석 오류: {e}"

def process_unsummarized_docs():
    print("--- Starting Summarization & Version Tracking ---")
    response = supabase.table("guidelines").select("*").is_("ai_summary", "null").limit(5).execute()
    docs = response.data
    
    if not docs:
        print("요약이 필요한 문서가 없습니다.")
        return
        
    for doc in docs:
        print(f"\nProcessing: {doc['title']}")
        text = extract_text_from_url(doc['url'])
        
        if text:
            analysis_result = analyze_document(text)
            ref_num = analysis_result.get("ref_number", "N/A")
            summary = analysis_result.get("summary", "")
            print(f" -> 식별자: {ref_num}")
            
            if ref_num != "N/A":
                existing_docs = supabase.table("guidelines").select("url").eq("ref_number", ref_num).neq("url", doc["url"]).execute().data
                if existing_docs:
                    old_url = existing_docs[0]['url']
                    print(f" -> 구버전 감지({old_url}). 비교 분석 시작.")
                    old_text = extract_text_from_url(old_url)
                    
                    if old_text:
                        comparison_text = compare_documents(old_text, text)
                        supabase.table("version_comparisons").insert({
                            "ref_number": ref_num,
                            "old_url": old_url,
                            "new_url": doc["url"],
                            "comparison_text": comparison_text
                        }).execute()
            
            supabase.table("guidelines").update({
                "ai_summary": summary,
                "ref_number": ref_num
            }).eq("url", doc["url"]).execute()
        else:
            supabase.table("guidelines").update({"ai_summary": "텍스트 추출 불가"}).eq("url", doc["url"]).execute()
        time.sleep(10)

if __name__ == "__main__":
    process_unsummarized_docs()
