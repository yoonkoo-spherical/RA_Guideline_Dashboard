import os
import time
import re
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

def extract_text(url):
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
        print(f"Extraction failed: {e}")
        return None

def clean_and_chunk_text(text, chunk_size=1000, overlap=100):
    # 1. 텍스트 정제: 연속된 공백 및 줄바꿈을 하나의 공백으로 치환하여 토큰 절약
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. 청크 분할
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:  # 너무 짧은 꼬리 부분 제외
            chunks.append(chunk)
    return chunks

def process_embeddings():
    print("--- Starting Document Embedding ---")
    
    # 전체 가이드라인 목록 조회
    all_docs = supabase.table("guidelines").select("url, title").execute().data
    
    # 이미 임베딩된 문서의 URL 목록 조회
    embedded_urls_response = supabase.table("document_chunks").select("url").execute().data
    embedded_urls = {item['url'] for item in embedded_urls_response}
    
    # 임베딩되지 않은 문서 필터링
    unprocessed_docs = [doc for doc in all_docs if doc['url'] not in embedded_urls]
    
    if not unprocessed_docs:
        print("모든 문서의 임베딩이 완료되었습니다.")
        return

    # API 제한을 고려하여 1회 실행 시 1개의 문서만 처리
    target_doc = unprocessed_docs[0]
    print(f"Processing: {target_doc['title']}")
    
    text = extract_text(target_doc['url'])
    if not text:
        print(" -> 텍스트 추출 실패. 임베딩을 건너뜁니다.")
        # 실패 기록을 남겨 무한 루프 방지 (더미 데이터 삽입)
        supabase.table("document_chunks").insert({
            "url": target_doc["url"], "chunk_index": -1, "content": "EXTRACTION_FAILED", "embedding": [0]*768
        }).execute()
        return

    chunks = clean_and_chunk_text(text)
    print(f" -> {len(chunks)} 개의 청크로 분할 완료. 임베딩 시작...")

    for i, chunk in enumerate(chunks):
        try:
            # 텍스트 임베딩 전용 모델 호출
            response = client.models.embed_content(
                model='text-embedding-004',
                contents=chunk
            )
            embedding_vector = response.embeddings[0].values
            
            # DB 삽입
            supabase.table("document_chunks").insert({
                "url": target_doc["url"],
                "chunk_index": i,
                "content": chunk,
                "embedding": embedding_vector
            }).execute()
            
            print(f"   - Chunk {i+1}/{len(chunks)} 저장 완료")
            time.sleep(2) # 분당 API 호출 제한(RPM) 방지를 위한 대기 시간
            
        except Exception as e:
            print(f"   - Chunk {i+1} 임베딩 실패: {e}")

    print(" -> 문서 임베딩 및 DB 적재 완료")

if __name__ == "__main__":
    process_embeddings()
