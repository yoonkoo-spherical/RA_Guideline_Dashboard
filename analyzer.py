import os
import smtplib
from email.mime.text import MIMEText
import requests
import fitz  # PyMuPDF
import json
import pytesseract
from pdf2image import convert_from_bytes
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import google.generativeai as genai
from supabase import create_client, Client

# 1. 환경 변수 및 설정
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

SMTP_EMAIL = os.environ.get("SMTP_EMAIL")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# 최적의 가성비를 위해 Flash 모델 사용 (Pro 전환 시 모델명 변경)
GENERATION_MODEL = "gemini-2.5-flash"

def send_alert_email(subject, content):
    """신/구버전 비교 결과 등 주요 변경점 감지 시 이메일 발송"""
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        print("이메일 환경변수가 설정되지 않아 알림을 생략합니다.")
        return

    msg = MIMEText(content)
    msg['Subject'] = subject
    msg['From'] = SMTP_EMAIL
    msg['To'] = SMTP_EMAIL  # 본인에게 발송

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"이메일 발송 성공: {subject}")
    except Exception as e:
        print(f"이메일 발송 실패: {e}")

def extract_text_with_ocr(pdf_bytes):
    """이미지형 PDF에서 OCR을 통해 텍스트 추출"""
    try:
        images = convert_from_bytes(pdf_bytes)
        text = ""
        for img in images:
            # 영어와 한국어 동시 인식 (Ubuntu 환경에 tesseract-ocr-kor 설치 필요)
            text += pytesseract.image_to_string(img, lang='eng+kor')
        return text if text.strip() else "추출 불가: OCR 텍스트 인식 실패"
    except Exception as e:
        return f"추출 불가: OCR 처리 에러 ({e})"

def extract_text_from_url(url):
    """웹 페이지 접근, PDF 다운로드 및 텍스트 추출 (봇 차단 회피 적용)"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        if response.status_code != 200:
            return f"추출 불가: 접근 권한 에러 (HTTP {response.status_code})"

        # 응답이 PDF인 경우
        if "application/pdf" in response.headers.get("Content-Type", "").lower() or url.lower().endswith(".pdf"):
            doc = fitz.open(stream=response.content, filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            if len(text.strip()) < 50:
                print(" -> 텍스트 레이어 없음. OCR 처리를 시도합니다.")
                return extract_text_with_ocr(response.content)
            return text

        # 응답이 HTML인 경우 PDF 링크 탐색
        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_links = []
        for a in soup.find_all("a", href=True):
            href = a['href'].lower()
            if ".pdf" in href or "download" in href:
                pdf_links.append(urljoin(url, a['href']))

        if pdf_links:
            target_pdf = pdf_links[0]
            print(f" -> HTML 내 PDF 링크 발견: {target_pdf}")
            pdf_res = requests.get(target_pdf, headers=headers, timeout=30)
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
    """Gemini API를 활용하여 가이드라인 요약 및 식별자 추출"""
    prompt = f"""
    당신은 RA 전문가입니다. 다음 가이드라인 원문을 분석하여 JSON 형식으로만 응답하십시오.
    1. "summary": 핵심 규제 내용 요약 (한국어, 300자 이내)
    2. "ref_number": 문서의 공식 식별 번호 (예: FDA-2023-D-1234, EMA/CHMP/123). 문서 내에 없다면 "N/A"로 표기할 것.

    원문:
    {text[:20000]}
    """
    model = genai.GenerativeModel(GENERATION_MODEL)
    try:
        response = model.generate_content(prompt)
        result_text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(result_text)
    except Exception as e:
        print(f"LLM 분석 오류: {e}")
        return {"summary": "요약 실패: LLM 분석 에러", "ref_number": "N/A"}

def compare_documents(old_text, new_text):
    """신/구버전 문서 내용 대조 및 변경점 추출"""
    prompt = f"""
    당신은 규제 문서 비교 전문가입니다. 구버전과 신버전 가이드라인의 주요 변경점을 분석하여 한국어로 요약하십시오.
    
    [구버전 핵심 내용 일부]
    {old_text[:15000]}

    [신버전 핵심 내용 일부]
    {new_text[:15000]}
    """
    model = genai.GenerativeModel(GENERATION_MODEL)
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"비교 분석 실패: {e}"

def process_unsummarized_docs():
    """메인 실행 로직"""
    print("문서 요약 및 분석 작업을 시작합니다.")
    
    # ai_summary가 null이거나 '추출 불가' 상태인 문서를 20개 가져옴 (PostgREST 문법)
    response = supabase.table("guidelines").select("*").or_("ai_summary.is.null,ai_summary.ilike.*추출 불가*").limit(20).execute()
    docs = response.data
    
    if not docs:
        print("대기 중이거나 처리할 문서가 없습니다.")
        return

    for doc in docs:
        doc_url = doc['url']
        print(f"\n처리 중: {doc['title']} ({doc_url})")
        
        # 1. 텍스트 추출
        text = extract_text_from_url(doc_url)
        
        # 추출 실패 시 상태 업데이트 후 다음 문서로 넘어감
        if "추출 불가" in text:
            print(f" -> {text}")
            supabase.table("guidelines").update({"ai_summary": text}).eq("url", doc_url).execute()
            continue
            
        # 2. 요약 및 분석
        analysis_result = analyze_document(text)
        summary = analysis_result.get("summary", "N/A")
        ref_num = analysis_result.get("ref_number", "N/A")
        print(f" -> 식별자: {ref_num}")
        
        # 3. 버전 비교 및 이메일 발송 로직
        if ref_num != "N/A" and doc.get('ref_number') != ref_num:
            # 동일한 식별자를 가지되 URL이 다른 기존 문서가 있는지 확인
            existing_docs = supabase.table("guidelines").select("url").eq("ref_number", ref_num).neq("url", doc_url).execute().data
            
            if existing_docs:
                print(" -> 🔄 기존 버전 발견. 비교 분석을 시작합니다.")
                old_url = existing_docs[0]['url']
                old_text = extract_text_from_url(old_url)
                
                if "추출 불가" not in old_text:
                    comparison_text = compare_documents(old_text, text)
                    # 데이터베이스 기록
                    supabase.table("version_comparisons").insert({
                        "ref_number": ref_num,
                        "old_url": old_url,
                        "new_url": doc_url,
                        "comparison_text": comparison_text
                    }).execute()
                    
                    # 이메일 발송
                    send_alert_email(f"[RA 시스템] 가이드라인 업데이트 감지: {ref_num}", comparison_text)
                
        # 4. DB 최종 업데이트
        supabase.table("guidelines").update({
            "ai_summary": summary,
            "ref_number": ref_num
        }).eq("url", doc_url).execute()

if __name__ == "__main__":
    process_unsummarized_docs()
