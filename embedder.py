import os
import time
import re
from google import genai
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = "gemini-embedding-001"

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
    print("--- Starting Document Embedding (DB Text Reader Mode) ---")
    
    # 메타데이터 추출을 위해 agency, category 추가
    all_docs = supabase.table("guidelines").select("url, title, agency, category, raw_text").not_.is_("raw_text", "null").not_.ilike("raw_text", "%추출 불가%").execute().data
    
    print("--- 기존 임베딩 완료 문서 개별 확인 중 (1000 한계 우회) ---")
    valid_embedded_urls = set()
    page_size = 1000
    for i in range(100):
        chunk_response = supabase.table("document_chunks").select("url").neq("content", "FAILED").range(i * page_size, (i + 1) * page_size - 1).execute()
        if not chunk_response.data:
            break
        valid_embedded_urls.update(item['url'] for item in chunk_response.data)

    unprocessed_docs = [doc for doc in all_docs if doc['url'] not in valid_embedded_urls]
    
    if not unprocessed_docs:
        print(" -> 새로 임베딩할 문서가 없습니다. 🟢")
        return

    print(f" -> 총 {len(unprocessed_docs)}개의 문서를 임베딩합니다.")

    for target_doc in unprocessed_docs:
        print(f"\nProcessing: {target_doc['title']}")
        
        # 메타데이터 헤더 생성
        agency = target_doc.get('agency') or 'N/A'
        title = target_doc.get('title') or 'N/A'
        category = target_doc.get('category') or 'N/A'
        meta_header = f"[기관: {agency}]\n[문서명: {title}]\n[분류: {category}]\n본문: "
        
        supabase.table("document_chunks").delete().eq("url", target_doc["url"]).execute()
        
        text = target_doc.get("raw_text", "")
        if not text.strip() or "추출 불가" in text:
            print(" -> 유효한 텍스트가 없어 임베딩을 생략합니다.")
            continue
            
        chunks = clean_and_chunk_text(text)
        print(f" -> {len(chunks)}개 청크 분할 완료. 임베딩 API 호출 시작...")
        
        batch_records = []
        success = True
        
        for i, chunk in enumerate(chunks):
            chunk_with_meta = meta_header + chunk
            embedding_vector = None
            
            for attempt in range(3):
                try:
                    response = client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=chunk_with_meta
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
                "content": chunk_with_meta,
                "embedding": embedding_vector
            })
            
            print(f"   - Chunk {i+1}/{len(chunks)} 임베딩 완료")
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
