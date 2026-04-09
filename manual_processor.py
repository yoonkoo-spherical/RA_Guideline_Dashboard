import os
import tempfile
import re
from google import genai
from supabase import create_client, Client
from dotenv import load_dotenv
import scraper
import analyzer
import embedder

load_dotenv()

# 환경 변수 및 클라이언트 초기화
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

def process_file_immediately(file_bytes, file_name, agency, category):
    """업로드된 파일을 즉시 분석하고 벡터화하여 DB에 저장합니다."""
    try:
        tmp_path = None
        # 1. 임시 파일 저장 및 텍스트 추출
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            raw_text = scraper.extract_text_with_ocr(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        if not raw_text or "추출 불가" in raw_text:
            return False, "텍스트를 추출할 수 없습니다."

        # 2. AI 요약 및 식별자 추출
        analysis = analyzer.analyze_document(raw_text)
        summary = analysis.get("summary", "N/A")
        ref_num = analysis.get("ref_number", "N/A")

        # 3. Supabase Storage 업로드
        # (기존 app.py 로직을 여기로 통합하거나 app.py에서 처리 후 URL을 넘겨받을 수 있습니다.)
        # 여기서는 파일 중복 에러를 방지하기 위해 upsert 설정을 사용합니다.
        supabase.storage.from_("guidelines_pdf").upload(
            file_name, file_bytes, {"upsert": "true", "content-type": "application/pdf"}
        )
        file_url = supabase.storage.from_("guidelines_pdf").get_public_url(file_name)

        # 4. 가이드라인 레코드 생성/업데이트
        supabase.table("guidelines").upsert({
            "title": file_name,
            "agency": agency,
            "category": category,
            "url": file_url,
            "raw_text": raw_text,
            "ai_summary": summary,
            "ref_number": ref_num
        }).execute()

        # 5. 임베딩 및 청크 저장
        meta_header = f"[기관: {agency}]\n[문서명: {file_name}]\n[분류: {category}]\n본문: "
        chunks = embedder.clean_and_chunk_text(raw_text)
        
        batch_records = []
        for i, chunk in enumerate(chunks):
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=meta_header + chunk
            )
            batch_records.append({
                "url": file_url,
                "chunk_index": i,
                "content": meta_header + chunk,
                "embedding": response.embeddings[0].values
            })

        if batch_records:
            # 기존 청크가 있다면 삭제 후 재삽입
            supabase.table("document_chunks").delete().eq("url", file_url).execute()
            supabase.table("document_chunks").insert(batch_records).execute()

        return True, "파일 처리가 즉시 완료되었습니다."
    except Exception as e:
        return False, str(e)
