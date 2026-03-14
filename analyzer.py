import os
import smtplib
from email.mime.text import MIMEText
import requests
import fitz
import json
import pytesseract
from pdf2image import convert_from_bytes
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from google import genai
from supabase import create_client, Client

# 1. 환경 변수 및 설정
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

SMTP_EMAIL = os.environ.get("SMTP_EMAIL")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

def send_alert_email(subject, content):
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        return

    msg = MIMEText(content)
    msg['Subject'] = subject
    msg['From'] = SMTP_EMAIL
    msg['To'] = SMTP_EMAIL

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"이메일 발송 실패: {e}")

def extract_text_with_ocr(pdf_bytes):
    try:
        images = convert_from_bytes(pdf_bytes)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img, lang='eng+kor')
        return text if text.strip() else "추출 불가: OCR 텍스트 인식 실패"
    except Exception as e:
        return f"추출 불가: OCR 처리 에러 ({e})"

def extract_text_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        if response.status_code != 200:
            return f"추출 불가: 접근 권한 에러 (HTTP {response.status_code})"

        if "application/pdf" in response.headers.get("Content-Type", "").lower() or url.lower().endswith(".pdf"):
            doc = fitz.open(stream=response.content, filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            if len(text.strip()) < 50:
                return extract_text_with_ocr(response.content)
            return text

        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_links = [urljoin(url, a['href']) for a in soup.find_all("a", href=True) if ".pdf" in a['href'].lower() or "download" in a['href'].lower()]

        if pdf_links:
            pdf_res = requests.get(pdf_links[0], headers=headers, timeout=30)
            if pdf_res.status_code == 200:
                doc = fitz.open(stream=pdf_res.content, filetype="pdf")
                text = "".join(page.get_text() for page in doc)
                if len(text.strip()) < 50:
                    return extract_text_with_ocr(pdf_res.content)
                return text
            else:
                return "추출 불가: PDF 링크 다운로드 실패"

        return "추출 불가: 웹페이지 내 PDF 링크 누락"
    except Exception as e:
        return f"추출 불가: 예외 발생 ({e})"

def analyze_document(text):
    prompt = f"""
    당신은 RA 전문가입니다. 다음 가이드라인 원문을 분석하여 JSON 형식으로만 응답하십시오.
    1. "summary": 핵심 규제 내용 요약 (한국어, 300자 이내)
    2. "ref_number": 문서의 공식 식별 번호 (예: FDA-2023-D-1234, EMA/CHMP/123). 문서 내에 없다면 "N/A"로 표기할 것.

    원문:
    {text[:20000]}
    """
    try:
        response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
        result_text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(result_text)
    except Exception:
        return {"summary": "요약 실패: LLM 분석 에러", "ref_number": "N/A"}

def compare_documents(old_text, new_text):
    prompt = f"""
    구버전과 신버전 가이드라인의 주요 변경점을 분석하여 한국어로 요약하십시오.
    
    [구버전 핵심 내용]
    {old_text[:15000]}

    [신버전 핵심 내용]
    {new_text[:15000]}
    """
    try:
        response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
        return response.text
    except Exception as e:
        return f"비교 분석 실패: {e}"

def embed_and_store_chunks(url, text):
    """추출된 텍스트를 분할하여 벡터 데이터베이스에 저장"""
    if len(text.strip()) < 50:
        return False
        
    # 동일 URL에 대한 기존 임베딩 삭제 (재처리 시 중복 방지)
    supabase.table("document_chunks").delete().eq("url", url).execute()
    
    chunk_size = 1000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    for index, chunk in enumerate(chunks):
        try:
            response = client.models.embed_content(model=EMBEDDING_MODEL, contents=chunk)
            embedding_vector = response.embeddings[0].values
            
            supabase.table("document_chunks").insert({
                "url": url,
                "chunk_index": index,
                "content": chunk,
                "embedding": embedding_vector
            }).execute()
        except Exception as e:
            print(f" -> Chunk {index} 임베딩 에러: {e}")
            return False
    return True

def process_unsummarized_docs():
    print("통합 문서 분석 및 임베딩 파이프라인을 시작합니다.")
    
    # 처리 대상: 요약이 없거나, 추출 실패 기록이 있는 문서 (최대 10개 제한)
    response = supabase.table("guidelines").select("*").or_("ai_summary.is.null,ai_summary.ilike.*추출 불가*").limit(10).execute()
    docs = response.data
    
    if not docs:
        print("대기 중이거나 처리할 문서가 없습니다.")
        return

    for doc in docs:
        doc_url = doc['url']
        print(f"\n처리 중: {doc['title']} ({doc_url})")
        
        # 1. 텍스트 단일 추출 (비용/시간 절감의 핵심)
        text = extract_text_from_url(doc_url)
        
        if "추출 불가" in text:
            print(f" -> {text}")
            supabase.table("guidelines").update({"ai_summary": text}).eq("url", doc_url).execute()
            continue
            
        # 2. 요약 및 식별자 추출
        analysis_result = analyze_document(text)
        summary = analysis_result.get("summary", "N/A")
        ref_num = analysis_result.get("ref_number", "N/A")
        print(f" -> AI 요약 및 식별자({ref_num}) 추출 완료")
        
        # 3. 신/구버전 자동 비교
        if ref_num != "N/A" and doc.get('ref_number') != ref_num:
            existing_docs = supabase.table("guidelines").select("url").eq("ref_number", ref_num).neq("url", doc_url).execute().data
            if existing_docs:
                old_url = existing_docs[0]['url']
                old_text = extract_text_from_url(old_url)
                if "추출 불가" not in old_text:
                    comparison_text = compare_documents(old_text, text)
                    supabase.table("version_comparisons").insert({
                        "ref_number": ref_num, "old_url": old_url, "new_url": doc_url, "comparison_text": comparison_text
                    }).execute()
                    send_alert_email(f"[RA 시스템] 가이드라인 업데이트 감지: {ref_num}", comparison_text)
                    print(" -> 버전 비교 완료 및 알림 이메일 발송")
                
        # 4. 임베딩 및 DB 적재
        is_embedded = embed_and_store_chunks(doc_url, text)
        if is_embedded:
            print(" -> 벡터 임베딩 및 DB 적재 완료")
        else:
            print(" -> 벡터 임베딩 실패")
            summary = "요약 성공하였으나 임베딩 중단됨"

        # 5. 상태 최종 업데이트
        supabase.table("guidelines").update({
            "ai_summary": summary,
            "ref_number": ref_num
        }).eq("url", doc_url).execute()

if __name__ == "__main__":
    process_unsummarized_docs()
