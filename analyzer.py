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
    
    # 8년차 RA 전문가 수준의 데이터 추출 프롬프트
    prompt = f"""
    당신은 8년 이상 경력의 글로벌 바이오시밀러 RA(Regulatory Affairs) 전문가입니다.
    원문을 분석하여 다음 두 항목을 추출하십시오.
    
    1. ref_number: 고유 식별자 (FDA Docket Number 또는 EMA 참조 번호). 불명확 시 "N/A".
    2. summary: CTD(공통기술문서) 작성 시 실무자가 즉각적으로 참고해야 할 핵심 규제 요건을 요약하십시오. 
       - 단순 나열을 피하고, CMC(품질), 비임상, 임상(PK/PD, 면역원성), 적응증 외삽(Extrapolation), 상호교환성(Interchangeability) 중 해당 문서가 중점적으로 다루는 허용 한계(Margin)와 평가 지표를 구체적으로 명시하십시오.
    
    [가이드라인 텍스트]
    {text[:10000]}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite', # 요약/추출 작업은 속도와 비용을 고려해 플래시 모델 유지
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
        return {"ref_number": "N/A", "summary": "요약 생성 오류"}

def compare_documents(old_text, new_text):
    # 인허가 전략 관점의 변경점 비교 프롬프트
    prompt = f"""
    당신은 8년 이상 경력의 글로벌 바이오시밀러 RA 전문가입니다. 구버전과 신버전 텍스트를 비교 분석하십시오.
    규제 당국의 심사 기조가 어떻게 변화했는지, 허가 지연 리스크를 방지하기 위해 실무적으로 어떤 조치를 취해야 하는지 명확히 작성하십시오.
    
    결과는 반드시 아래 마크다운 표 형식으로 작성하십시오.
    
    | 규제 변경 항목 (CMC/임상/통계 등) | 구버전 허가 기준 | 신버전 허가 기준 | CTD 작성 및 인허가 전략 시 실무 영향 (Action Item) |
    |---|---|---|---|
    | ... | ... | ... | ... |
    
    [구버전 텍스트 일부]
    {old_text[:8000]}
    
    [신규 텍스트 일부]
    {new_text[:8000]}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro', # 복잡한 비교 분석에는 상위 모델 적용
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
        return
        
    for doc in docs:
        text = extract_text_from_url(doc['url'])
        if text:
            analysis_result = analyze_document(text)
            ref_num = analysis_result.get("ref_number", "N/A")
            summary = analysis_result.get("summary", "")
            
            if ref_num != "N/A":
                existing_docs = supabase.table("guidelines").select("url").eq("ref_number", ref_num).neq("url", doc["url"]).execute().data
                if existing_docs:
                    old_url = existing_docs[0]['url']
                    old_text = extract_text_from_url(old_url)
                    if old_text:
                        comparison_text = compare_documents(old_text, text)
                        supabase.table("version_comparisons").insert({
                            "ref_number": ref_num, "old_url": old_url, "new_url": doc["url"], "comparison_text": comparison_text
                        }).execute()
            
            supabase.table("guidelines").update({"ai_summary": summary, "ref_number": ref_num}).eq("url", doc["url"]).execute()
        else:
            supabase.table("guidelines").update({"ai_summary": "텍스트 추출 불가"}).eq("url", doc["url"]).execute()
        time.sleep(10)

if __name__ == "__main__":
    process_unsummarized_docs()
