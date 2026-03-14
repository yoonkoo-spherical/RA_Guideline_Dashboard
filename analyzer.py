import os
import smtplib
import re
import time
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
SCRAPER_API_KEY = os.environ.get("SCRAPER_API_KEY")

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

def fetch_html_with_scraperapi(url):
    """ScraperAPI 호출 및 3회 재시도 로직 적용"""
    if not SCRAPER_API_KEY: return None
    payload = {'api_key': SCRAPER_API_KEY, 'url': url, 'render': 'true'}
    for _ in range(3):
        try:
            res = requests.get('https://api.scraperapi.com/', params=payload, timeout=60)
            if res.status_code == 200: return res.text
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return None

def fetch_binary_with_scraperapi(url):
    """PDF 우회 다운로드 및 3회 재시도 로직 적용"""
    if not SCRAPER_API_KEY: return None
    payload = {'api_key': SCRAPER_API_KEY, 'url': url}
    for _ in range(3):
        try:
            res = requests.get('https://api.scraperapi.com/', params=payload, timeout=60)
            if res.status_code == 200 and b"%PDF" in res.content[:5]: return res.content
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return None

def extract_text_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        is_pdf = "application/pdf" in response.headers.get("Content-Type", "").lower() or url.lower().endswith(".pdf")
        
        if response.status_code == 200 and is_pdf:
            doc = fitz.open(stream=response.content, filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            return text if len(text.strip()) >= 50 else extract_text_with_ocr(response.content)

        html_content = response.text if response.status_code == 200 else fetch_html_with_scraperapi(url)
        if html_content and len(html_content) < 1000: 
            html_content = fetch_html_with_scraperapi(url)

        if not html_content: return f"추출 불가: 웹페이지 접근 실패 (HTTP {response.status_code})"

        soup = BeautifulSoup(html_content, 'html.parser')
        pdf_links = []
        for a in soup.find_all("a", href=True):
            href_lower = a['href'].lower()
            if href_lower.startswith(('mailto:', 'javascript:', 'tel:', '#')): continue
            if "acrobat" in href_lower or "get.adobe" in href_lower: continue 
            
            if ".pdf" in href_lower or "download" in href_lower or "attachment" in href_lower or "/media/" in href_lower:
                pdf_links.append(urljoin(url, a['href']))

        for pdf_url in pdf_links:
            try:
                pdf_res = requests.get(pdf_url, headers=headers, timeout=30)
                pdf_content = None
                if pdf_res.status_code == 200 and b"%PDF" in pdf_res.content[:5]:
                    pdf_content = pdf_res.content
                else:
                    pdf_content = fetch_binary_with_scraperapi(pdf_url)

                if pdf_content:
                    doc = fitz.open(stream=pdf_content, filetype="pdf")
                    text = "".join(page.get_text() for page in doc)
                    if len(text.strip()) > 50: return text
                    
                    ocr_text = extract_text_with_ocr(pdf_content)
                    if not ocr_text.startswith("추출 불가"): return ocr_text
            except Exception:
                continue 

        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
            tag.extract()
        
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'govspeak|main-content|content'))
        html_text = main_content.get_text(separator='\n', strip=True) if main_content else soup.body.get_text(separator='\n', strip=True) if soup.body else ""
        
        if len(html_text.strip()) > 200: return html_text
        return "추출 불가: PDF 링크 다운로드에 실패하였으며, HTML 본문 텍스트도 부족함"
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
    prompt = f"""구버전과 신버전 가이드라인의 주요 변경점을 분석하여 한국어로 요약하십시오.
    [구버전 핵심 내용]\n{old_text[:15000]}\n[신버전 핵심 내용]\n{new_text[:15000]}"""
    try:
        response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
        return response.text
    except Exception as e:
        return f"비교 분석 실패: {e}"

def embed_and_store_chunks(url, text):
    if len(text.strip()) < 50: return False
    supabase.table("document_chunks").delete().eq("url", url).execute()
    
    chunk_size = 1000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    for index, chunk in enumerate(chunks):
        try:
            response = client.models.embed_content(model=EMBEDDING_MODEL, contents=chunk)
            embedding_vector = response.embeddings[0].values
            supabase.table("document_chunks").insert({
                "url": url, "chunk_index": index, "content": chunk, "embedding": embedding_vector
            }).execute()
        except Exception as e:
            print(f" -> Chunk {index} 임베딩 에러: {e}")
            # [추가됨] 임베딩 중간 실패 시 불완전한 데이터 롤백(삭제)
            supabase.table("document_chunks").delete().eq("url", url).execute()
            print(" -> 부분 임베딩 찌꺼기 삭제 완료")
            return False
    return True

def process_unsummarized_docs():
    print("통합 문서 분석 및 임베딩 파이프라인을 시작합니다.")
    response = supabase.table("guidelines").select("*").or_("ai_summary.is.null,ai_summary.ilike.*추출 불가*").limit(10).execute()
    docs = response.data
    
    if not docs:
        print("대기 중이거나 처리할 문서가 없습니다.")
        return

    for doc in docs:
        doc_url = doc['url']
        print(f"\n처리 중: {doc['title']} ({doc_url})")
        
        text = extract_text_from_url(doc_url)
        
        if "추출 불가" in text:
            print(f" -> {text}")
            supabase.table("guidelines").update({"ai_summary": text}).eq("url", doc_url).execute()
            try:
                supabase.table("document_chunks").delete().eq("url", doc_url).execute()
                print(" -> 기존 잘못된 임베딩(더미 데이터) 클렌징 완료")
            except Exception as e:
                print(f" -> 임베딩 클렌징 에러: {e}")
            continue
            
        analysis_result = analyze_document(text)
        summary = analysis_result.get("summary", "N/A")
        ref_num = analysis_result.get("ref_number", "N/A")
        print(f" -> AI 요약 및 식별자({ref_num}) 추출 완료")
        
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
                
        is_embedded = embed_and_store_chunks(doc_url, text)
        if is_embedded:
            print(" -> 벡터 임베딩 및 DB 적재 완료")
        else:
            print(" -> 벡터 임베딩 실패")
            summary = "요약 성공하였으나 임베딩 중단됨"

        supabase.table("guidelines").update({
            "ai_summary": summary,
            "ref_number": ref_num
        }).eq("url", doc_url).execute()

if __name__ == "__main__":
    process_unsummarized_docs()
