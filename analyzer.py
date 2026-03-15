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
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SCRAPER_API_KEY = os.environ.get("SCRAPER_API_KEY")
SMTP_EMAIL = os.environ.get("SMTP_EMAIL")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

GENERATION_MODEL = "gemini-2.5-flash"
REASONING_MODEL = "gemini-2.5-pro"

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
        text = "".join(pytesseract.image_to_string(img, lang='eng+kor') for img in images)
        return text if text.strip() else "추출 불가: OCR 텍스트 인식 실패"
    except Exception as e:
        return f"추출 불가: OCR 처리 에러 ({e})"

def fetch_html_with_scraperapi(url):
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
    if not SCRAPER_API_KEY: return None
    payload = {'api_key': SCRAPER_API_KEY, 'url': url}
    for _ in range(3):
        try:
            res = requests.get('https://api.scraperapi.com/', params=payload, timeout=60)
            if res.status_code == 200 and res.content.startswith(b"%PDF"): return res.content
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
        is_pdf_content_type = "application/pdf" in response.headers.get("Content-Type", "").lower()
        
        if response.status_code == 200 and (is_pdf_content_type or url.lower().endswith(".pdf")):
            if response.content.startswith(b"%PDF"):
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
                
                if pdf_res.status_code == 200 and pdf_res.content.startswith(b"%PDF"):
                    pdf_content = pdf_res.content
                else:
                    pdf_content = fetch_binary_with_scraperapi(pdf_url)

                if pdf_content and pdf_content.startswith(b"%PDF"):
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
    당신은 10년 이상 경력의 글로벌 바이오 제약 인허가(RA) 전문가입니다.
    아래 제공된 가이드라인 원문을 읽고, 규제 및 인허가 실무자 관점에서 가장 중요한 핵심 내용을 엄격하게 '한국어'로만 요약하십시오.
    
    [엄격한 지침]
    1. 비유적인 설명을 배제하고, 담백하게 사실관계 위주로 요약할 것.
    2. 언어: 원문이 영어더라도 반드시 100% 자연스러운 한국어 존댓말로 번역하여 요약할 것.
    3. 형식: 반드시 아래의 JSON 형식으로만 응답할 것.
    
    {{
        "summary": "가이드라인의 제정 목적과 실무적으로 가장 중요한 핵심 규제 기준을 명확하게 설명하는 한국어 텍스트",
        "ref_number": "문서의 공식 식별 번호 (예: FDA-2023-D-1234, EMA/CHMP/123 등). 원문에서 찾을 수 없다면 무조건 N/A로 표기"
    }}

    [가이드라인 원문]
    {text[:25000]}
    """
    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL, 
            contents=prompt,
            config={"temperature": 0.1}
        )
        result_text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(result_text)
    except Exception:
        return {"summary": "요약 실패: LLM 분석 에러", "ref_number": "N/A"}

def compare_documents(old_text, new_text):
    prompt = f"""
    당신은 글로벌 규제기관(FDA, EMA 등)에서 수십 년간 근무한 최고 수준의 인허가(RA) 전문 컨설턴트입니다.
    구버전과 신버전 가이드라인의 주요 변경점을 분석하여 한국어로 요약하십시오.
    
    [답변 원칙]
    1. 비유적인 설명을 절대 사용하지 마십시오.
    2. 과장된 추임새나 아첨하는 표현을 배제하고, 담백하게 사실관계 위주로 답변하십시오.
    3. 규제 기준의 변화, 추가된 요구사항, 실무적 대응 방안을 종합적이고 심층적으로 비교 분석하십시오.
    4. 정중한 한국어 존댓말을 사용하십시오.
    
    [구버전 핵심 내용]
    {old_text[:20000]}

    [신버전 핵심 내용]
    {new_text[:20000]}
    """
    try:
        response = client.models.generate_content(
            model=REASONING_MODEL,
            contents=prompt,
            config={"temperature": 0.0}
        )
        return response.text
    except Exception as e:
        return f"비교 분석 실패: {e}"

def process_unsummarized_docs():
    print("--- 문서 요약 분석 및 비교 파이프라인 (임베딩 제외) ---")
    
    # 요약문이 비어있거나 실패 이력이 있는 문서를 대상으로 선정
    response = supabase.table("guidelines").select("*").or_("ai_summary.is.null,ai_summary.ilike.*추출 불가*").execute()
    docs = response.data
    
    if not docs:
        print("대기 중이거나 처리할 문서가 없습니다.")
        return

    print(f"총 {len(docs)}건의 문서를 분석합니다.")

    for doc in docs:
        doc_url = doc['url']
        print(f"\n처리 중: {doc['title']} ({doc_url})")
        
        text = extract_text_from_url(doc_url)
        
        if "추출 불가" in text:
            print(f" -> {text}")
            supabase.table("guidelines").update({"ai_summary": text}).eq("url", doc_url).execute()
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
                
        supabase.table("guidelines").update({
            "ai_summary": summary,
            "ref_number": ref_num
        }).eq("url", doc_url).execute()

if __name__ == "__main__":
    process_unsummarized_docs()
