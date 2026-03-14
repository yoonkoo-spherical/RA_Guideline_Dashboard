import os
import requests
import fitz
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from google import genai
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = 'gemini-embedding-001'

def extract_text(url):
    """임베딩용 텍스트 추출 (OCR 제외, 순수 텍스트 위주)"""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200: return None
        
        if "application/pdf" in response.headers.get("Content-Type", "").lower() or url.lower().endswith(".pdf"):
            doc = fitz.open(stream=response.content, filetype="pdf")
            return "".join(page.get_text() for page in doc)
            
        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_links = [urljoin(url, a['href']) for a in soup.find_all("a", href=True) if ".pdf" in a['href'].lower()]
        if pdf_links:
            pdf_res = requests.get(pdf_links[0], headers=headers, timeout=30)
            doc = fitz.open(stream=pdf_res.content, filetype="pdf")
            return "".join(page.get_text() for page in doc)
        return soup.get_text(separator=' ', strip=True)
    except Exception:
        return None

def embed_and_store():
    print("임베딩 스크립트를 시작합니다.")
    
    # 임베딩되지 않은 문서를 찾기 위해 전체 문서 목록과 이미 임베딩된 목록 대조
    docs_response = supabase.table("guidelines").select("url, title").execute()
    all_docs = docs_response.data
    
    chunks_response = supabase.table("document_chunks").select("url").execute()
    embedded_urls = set([item['url'] for item in chunks_response.data])
    
    # 아직 임베딩되지 않은 대상만 추출 (최대 10개씩 처리하여 제한 방지)
    target_docs = [doc for doc in all_docs if doc['url'] not in embedded_urls][:10]
    
    if not target_docs:
        print("모든 문서가 임베딩되어 있습니다.")
        return

    for doc in target_docs:
        url = doc['url']
        print(f"임베딩 시도: {doc['title']} ({url})")
        
        text = extract_text(url)
        if not text or len(text.strip()) < 50:
            print(" -> 텍스트 추출 실패 또는 너무 짧음. 건너뜁니다.")
            continue
            
        # Chunk 단위 분할 (1000자 기준)
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
                break # 해당 문서 임베딩 중단
        print(f" -> 임베딩 완료 ({len(chunks)} chunks)")

if __name__ == "__main__":
    embed_and_store()
