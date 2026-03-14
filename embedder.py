import os
import time
import re
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
        return text if text.strip() else "FAILED"
    except Exception:
        return "FAILED"

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
            if res.status_code == 200 and b"%PDF" in res.content[:5]: return res.content
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return None

def extract_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}
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
        if not html_content: return None

        soup = BeautifulSoup(html_content, 'html.parser')
        pdf_links = [urljoin(url, a['href']) for a in soup.find_all("a", href=True) 
                     if ".pdf" in a['href'].lower() or "download" in a['href'].lower() or "attachment" in a['href'].lower()]

        for pdf_url in pdf_links:
            try:
                pdf_res = requests.get(pdf_url, headers=headers, timeout=30)
                pdf_content = pdf_res.content if (pdf_res.status_code == 200 and b"%PDF" in pdf_res.content[:5]) else fetch_binary_with_scraperapi(pdf_url)
                if pdf_content:
                    doc = fitz.open(stream=pdf_content, filetype="pdf")
                    text = "".join(page.get_text() for page in doc)
                    if len(text.strip()) > 50: return text
            except Exception:
                continue 

        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]): tag.extract()
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'govspeak|main-content|content'))
        html_text = main_content.get_text(separator='\n', strip=True) if main_content else soup.body.get_text(separator='\n', strip=True) if soup.body else ""
        
        return html_text if len(html_text.strip()) > 200 else None
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None

def clean_and_chunk_text(text, chunk_size=1000, overlap=100):
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks

def process_embeddings():
    print("--- Starting Document Embedding (Batch Tracking & Retry Mode) ---")
    
    all_docs = supabase.table("guidelines").select("url, title").execute().data
    valid_chunks_response = supabase.table("document_chunks").select("url").neq("content", "FAILED").execute().data
    valid_embedded_urls = {item['url'] for item in valid_chunks_response}
    
    unprocessed_docs = [doc for doc in all_docs if doc['url'] not in valid_embedded_urls]
    
    if not unprocessed_docs:
        print("모든 문서의 정상 임베딩이 완료되었습니다.")
        return

    print(f"총 {len(unprocessed_docs)}건의 미완료/실패 문서 임베딩을 일괄 진행합니다.")

    # [:3] 제한 제거: 횟수 제한 없이 전체 문서 순차 처리
    for target_doc in unprocessed_docs:
        print(f"\nProcessing / Retrying: {target_doc['title']}")
        
        supabase.table("document_chunks").delete().eq("url", target_doc["url"]).execute()
        
        text = extract_text(target_doc['url'])
        
        if not text or text.startswith("추출 불가") or text == "FAILED":
            print(" -> 텍스트 추출 실패. 임베딩 생략.")
            continue

        chunks = clean_and_chunk_text(text)
        print(f" -> {len(chunks)} 개의 청크로 분할 완료. 임베딩 시작...")

        success = True
        for i, chunk in enumerate(chunks):
            try:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=chunk
                )
                embedding_vector = response.embeddings[0].values
                
                supabase.table("document_chunks").insert({
                    "url": target_doc["url"],
                    "chunk_index": i,
                    "content": chunk,
                    "embedding": embedding_vector
                }).execute()
                
                print(f"   - Chunk {i+1}/{len(chunks)} 저장 완료")
                time.sleep(1) # 유료 플랜이므로 대기 시간 단축
                
            except Exception as e:
                print(f"   - Chunk {i+1} 임베딩 실패: {e}")
                success = False
                break 

        if not success:
            print(" -> 임베딩 오류 발생. 불완전한 청크 삭제(롤백) 처리.")
            supabase.table("document_chunks").delete().eq("url", target_doc["url"]).execute()

    print("\n -> 배치 임베딩 작업 종료")

if __name__ == "__main__":
    process_embeddings()
