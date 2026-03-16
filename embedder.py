import os
import time
import re
import json
import requests
import fitz
import pytesseract
from pdf2image import convert_from_bytes
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from google import genai
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = "gemini-embedding-001"

def extract_text_with_ocr(pdf_bytes):
    try:
        images = convert_from_bytes(pdf_bytes)
        text = "".join(pytesseract.image_to_string(img, lang='eng+kor') for img in images)
        return text if text.strip() else "추출 불가: OCR 실패"
    except Exception:
        return "추출 불가"

def fetch_html_with_scraperapi(url, render='false'):
    if not SCRAPER_API_KEY: return None
    payload = {'api_key': SCRAPER_API_KEY, 'url': url, 'render': render}
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

def extract_content_robust(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        is_pdf_content_type = "application/pdf" in response.headers.get("Content-Type", "").lower()
        url_lower = url.lower()
        
        is_likely_pdf = url_lower.endswith(".pdf") or "download" in url_lower or is_pdf_content_type
        
        if is_likely_pdf:
            pdf_content = None
            if response.status_code == 200 and response.content.startswith(b"%PDF"):
                pdf_content = response.content
            else:
                pdf_content = fetch_binary_with_scraperapi(url)
                
            if pdf_content and pdf_content.startswith(b"%PDF"):
                doc = fitz.open(stream=pdf_content, filetype="pdf")
                text = "".join(page.get_text() for page in doc)
                return text if len(text.strip()) >= 50 else extract_text_with_ocr(pdf_content)

        # 1차 시도: 일반 HTTP GET이 차단되었거나 실패한 경우 render='false'로 프록시 호출
        html_content = response.text if response.status_code == 200 else fetch_html_with_scraperapi(url, render='false')
        
        # 2차 시도: 렌더링이 필요한 동적 웹페이지인 경우 render='true'로 재호출
        if not html_content or len(html_content) < 1000: 
            html_content = fetch_html_with_scraperapi(url, render='true')

        if not html_content: return "추출 불가: 웹페이지 접근 실패"

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
        
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'govspeak|main-content|content|mws-body|page-body|container'))
        html_text = main_content.get_text(separator='\n', strip=True) if main_content else soup.body.get_text(separator='\n', strip=True) if soup.body else ""
        
        if len(html_text.strip()) > 100: return html_text
        return "추출 불가: HTML 본문 부족"
    except Exception as e:
        return f"추출 불가: 예외 발생 ({e})"

def clean_and_chunk_text(text, chunk_size=1000, overlap=100):
    text = text.replace('\x00', '').replace('\u0000', '')
    text = re.sub(r'\s+', ' ', text).strip()
    
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks

def process_embeddings():
    print("--- Starting Document Embedding (Unified Extraction & Retry Mode) ---")
    
    all_docs = supabase.table("guidelines").select("url, title").not_.ilike("ai_summary", "%추출 불가%").not_.is_("ai_summary", "null").execute().data
    
    print("--- 기존 임베딩 완료 문서 개별 확인 중 (1000 한계 우회) ---")
    valid_embedded_urls = set()
    for doc in all_docs:
        res = supabase.table("document_chunks").select("url").eq("url", doc["url"]).neq("content", "FAILED").limit(1).execute()
        if res.data:
            valid_embedded_urls.add(doc["url"])
            
    unprocessed_docs = [doc for doc in all_docs if doc['url'] not in valid_embedded_urls]
    
    if not unprocessed_docs:
        print("모든 유효 문서의 임베딩이 완료되었습니다.")
        return

    print(f"총 {len(unprocessed_docs)}건의 미완료 문서 임베딩을 일괄 진행합니다.")

    for target_doc in unprocessed_docs:
        print(f"\nProcessing / Retrying: {target_doc['title']}")
        
        supabase.table("document_chunks").delete().eq("url", target_doc["url"]).execute()
        
        text = extract_content_robust(target_doc['url'])
        
        if not text or text.startswith("추출 불가") or text == "FAILED":
            print(f" -> {text}. 임베딩 생략.")
            continue

        chunks = clean_and_chunk_text(text)
        print(f" -> {len(chunks)} 개의 청크로 분할 완료. 임베딩 시작...")

        success = True
        batch_records = []
        
        for i, chunk in enumerate(chunks):
            embedding_vector = None
            
            for attempt in range(3):
                try:
                    response = client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=chunk
                    )
                    embedding_vector = response.embeddings[0].values
                    break 
                except Exception as e:
                    print(f"   - Chunk {i+1} API 호출 실패 (시도 {attempt+1}/3): {e}")
                    time.sleep(2)
            
            if not embedding_vector:
                print(f"   - Chunk {i+1} 최종 임베딩 실패. 진행 중단 및 문서 롤백.")
                success = False
                break 

            batch_records.append({
                "url": target_doc["url"],
                "chunk_index": i,
                "content": chunk,
                "embedding": embedding_vector
            })
            
            print(f"   - Chunk {i+1}/{len(chunks)} 임베딩 추출 완료")
            time.sleep(0.1)

        if success and batch_records:
            try:
                supabase.table("document_chunks").insert(batch_records).execute()
                print(f" -> {len(batch_records)}개 청크 DB 일괄 저장 완료 🟢")
            except Exception as db_e:
                print(f" -> DB 일괄 저장 중 오류 발생: {db_e}")
                success = False

        if not success:
            print(" -> 임베딩 또는 DB 저장 오류 발생. 불완전한 청크 삭제(롤백) 처리 🔴")
            supabase.table("document_chunks").delete().eq("url", target_doc["url"]).execute()

    print("\n -> 배치 임베딩 작업 종료")

if __name__ == "__main__":
    process_embeddings()
