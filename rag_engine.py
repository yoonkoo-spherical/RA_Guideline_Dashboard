import os
import streamlit as st
from google import genai
from supabase import create_client, Client
from dotenv import load_dotenv

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError, Exception):
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

ROUTING_MODEL = 'gemini-2.5-flash-lite' 
GENERATION_MODEL = 'gemini-2.5-flash-lite' 
EMBEDDING_MODEL = 'gemini-embedding-001'

def get_query_embedding(query_text):
    response = client.models.embed_content(model=EMBEDDING_MODEL, contents=query_text)
    return response.embeddings[0].values

def retrieve_relevant_chunks(query_embedding, match_threshold=0.3, match_count=5):
    response = supabase.rpc(
        'match_documents',
        {'query_embedding': query_embedding, 'match_threshold': match_threshold, 'match_count': match_count}
    ).execute()
    return response.data

def route_query(query_text):
    prompt = f"사용자의 질문이 제약/바이오 규제, 바이오시밀러, 가이드라인과 관련된 질문인지 판단하십시오. 관련이 있다면 'RAG', 단순 인사나 일반 대화라면 'GENERAL'이라고만 답변하십시오.\n질문: {query_text}"
    try:
        response = client.models.generate_content(model=ROUTING_MODEL, contents=prompt)
        return response.text.strip().upper()
    except Exception:
        return "RAG"

def generate_rag_response(query_text, chunks):
    if not chunks:
        return "현재 데이터베이스 내에서 질문과 관련된 가이드라인 내용을 찾을 수 없습니다.", []

    context_text = "\n\n".join([f"[출처: {c['url']}]\n{c['content']}" for c in chunks])
    prompt = f"당신은 RA 전문가입니다. 아래에 제공된 [참고 문서]만을 기반으로 사용자의 [질문]에 객관적이고 사실적으로 답변하십시오. 문서에 없는 내용은 임의로 추측하지 말고 모른다고 명시하십시오. 답변 시 어떤 출처(URL)를 참고했는지 번호표기([1], [2])를 사용하여 간략히 기재하십시오.\n\n[참고 문서]\n{context_text}\n\n[질문]\n{query_text}"
    
    try:
        response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
        return response.text, chunks # Step 9: 텍스트와 함께 출처 청크 데이터를 반환
    except Exception as e:
        return f"답변 생성 오류: {e}", []

def ask_guideline(query_text):
    route_decision = route_query(query_text)
    if "GENERAL" in route_decision:
        response = client.models.generate_content(model=ROUTING_MODEL, contents=query_text)
        return response.text, []
        
    query_embedding = get_query_embedding(query_text)
    if not query_embedding: return "질문 분석 실패", []
    
    chunks = retrieve_relevant_chunks(query_embedding)
    return generate_rag_response(query_text, chunks)

def compare_multiple_documents(docs_info):
    context_parts = []
    for doc in docs_info:
        title = doc['title']
        url = doc['url']
        response = supabase.table("document_chunks").select("content").eq("url", url).order("chunk_index").execute()
        if response.data:
            full_text = "\n".join([item['content'] for item in response.data])
            context_parts.append(f"--- [문서 제목: {title}] ---\n{full_text[:30000]}")
        else:
            context_parts.append(f"--- [문서 제목: {title}] ---\n(임베딩 텍스트 없음)")

    combined_context = "\n\n".join(context_parts)
    prompt = f"""
    당신은 RA(Regulatory Affairs) 전문가입니다. 아래에 제공된 여러 가이드라인 문서의 텍스트를 대조 분석하십시오.
    
    요구사항:
    1. 선택된 문서들 간의 규제 접근 방식, 평가 기준 등의 공통점과 차이점을 식별하십시오.
    2. 핵심 차이점은 한눈에 파악하기 쉽게 마크다운 표(Table) 형식으로 정리하십시오.
    3. 실무자를 위한 종합적인 결론을 텍스트로 요약하십시오.

    [비교 대상 문서 텍스트]
    {combined_context}
    """
    try:
        response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
        return response.text
    except Exception as e:
        return f"다중 문서 비교 오류: {e}"
