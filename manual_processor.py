import os
import tempfile
import re
import time
from google import genai
from supabase import create_client, Client
from dotenv import load_dotenv
import analyzer
import embedder

# 텍스트 추출 고도화를 위한 라이브러리 임포트
import fitz  # PyMuPDF
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

load_dotenv()

# 환경 변수 및 클라이언트 초기화
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

def _extract_text_robust(pdf_path: str) -> str:
    """
    PyMuPDF를 사용하여 텍스트를 우선 추출하고, 
    텍스트 레이어가 없거나 불완전한 경우 OCR을 수행하는 영문 문서 최적화 추출 함수입니다.
    """
    text_parts = []
    try:
        doc = fitz.open(pdf_path)
        for page_idx, page in enumerate(doc):
            page_text = page.get_text("text").strip()
            
            # 페이지에 텍스트가 없거나 지나치게 적은 경우 스캔 이미지 레이어로 판단하여 OCR 수행
            if (not page_text or len(page_text) < 50) and OCR_AVAILABLE:
                try:
                    # 메모리 절약을 위해 해당 페이지 선택적 이미지 변환 (150 DPI)
                    images = convert_from_path(pdf_path, first_page=page_idx+1, last_page=page_idx+1, dpi=150)
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0], lang="eng")
                        page_text = ocr_text.strip()
                except Exception:
                    pass
            
            if page_text:
                text_parts.append(page_text)
        
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"추출 불가: PDF 파싱 중 오류 발생 ({str(e)})"

def process_file_immediately(file_bytes, file_name, agency, category):
    """업로드된 파일을 즉시 분석하고 벡터화하여 DB에 저장합니다."""
    try:
        tmp_path = None
        # 1. 임시 파일 저장 및 자체 고도화 텍스트 추출
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            raw_text = _extract_text_robust(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        if not raw_text or "추출 불가" in raw_text or len(raw_text.strip()) == 0:
            return False, "텍스트를 추출할 수 없습니다. 파일의 손상 여부나 보안 설정을 확인하십시오."

        # 2. AI 요약 및 식별자 추출
        analysis = analyzer.analyze_document(raw_text)
        summary = analysis.get("summary", "N/A")
        ref_num = analysis.get("ref_number", "N/A")

        # 3. Supabase Storage 업로드 (upsert 설정 유지)
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
            # 기존 청크 삭제 후 재삽입 연동
            supabase.table("document_chunks").delete().eq("url", file_url).execute()
            supabase.table("document_chunks").insert(batch_records).execute()

        return True, "파일 처리가 즉시 완료되었습니다."
    except Exception as e:
        return False, str(e)
