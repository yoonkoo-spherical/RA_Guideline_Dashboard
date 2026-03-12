import os
import streamlit as st
from google import genai
from supabase import create_client, Client
from dotenv import load_dotenv

# --- 로컬 및 Streamlit 환경 동시 지원을 위한 안전한 키 로딩 ---
try:
    # 1. Streamlit Cloud Secrets 우선 확인
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError, Exception):
    # 2. 로컬 테스트 환경 (.env) 대체 로딩
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

# 지능형 라우팅 모델 이원화 (향후 GENERATION_MODEL만 상위 모델로 변경)
ROUTING_MODEL = 'gemini-2.5-flash-lite' 
GENERATION_MODEL = 'gemini-2.5-flash-lite' 
EMBEDDING_MODEL = 'gemini-embedding-001'

def get_query_embedding(query_text):
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query_text
    )
    return response.embeddings[0].values

def retrieve_relevant_chunks(query_embedding, match_threshold=0.3, match_count=5):
    response = supabase.rpc(
        'match_documents',
        {
            'query_embedding': query_embedding,
            'match_threshold': match_threshold,
            'match_count': match_count
        }
    ).execute()
    return response.data

def route_query(query_text):
    prompt = f"""
    사용자의 질문이 제약/바이오 규제, 바이오시밀러, 가이드라인과 관련된 질문인지 판단하십시오.
    관련이 있다면 'RAG', 단순 인사나 일반 대화라면 'GENERAL'이라고만 답변하십시오.
    
    질문: {query_text}
    """
    try:
        response = client.models.generate_content(
            model=ROUTING_MODEL,
            contents=prompt
        )
        return response.text.strip().upper()
    except Exception as e:
        print(f"Routing Error: {e}")
        return "RAG"

def generate_rag_response(query_text, chunks):
    if not chunks:
        return "현재 데이터베이스 내에서 질문과 관련된 가이드라인 내용을 찾을 수 없습니다."

    context_text = "\n\n".join([f"[출처: {c['url']}]\n{c['content']}" for c in chunks])
    
    prompt = f"""
    당신은 RA 전문가입니다. 
    아래에 제공된 [참고 문서]만을 기반으로 사용자의 [질문]에 객관적이고 사실적으로 답변하십시오.
    문서에 없는 내용은 임의로 추측하지 말고 모른다고 명시하십시오.
    답변 시 어떤 출처(URL)를 참고했는지 간략히 기재하십시오.

    [참고 문서]
    {context_text}

    [질문]
    {query_text}
    """
    
    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"답변 생성 오류: {e}"

def ask_guideline(query_text):
    route_decision = route_query(query_text)
    
    if "GENERAL" in route_decision:
        response = client.models.generate_content(model=ROUTING_MODEL, contents=query_text)
        return response.text
        
    query_embedding = get_query_embedding(query_text)
    if not query_embedding:
         return "질문 분석 실패"
    
    chunks = retrieve_relevant_chunks(query_embedding)
    final_answer = generate_rag_response(query_text, chunks)
    
    return final_answer

if __name__ == "__main__":
    # 터미널 테스트 실행
    print("--- RAG 엔진 테스트 ---")
    question = "바이오시밀러 임상 평가 시 가장 중요하게 고려해야 할 사항은 무엇인가요?"
    print(f"Q: {question}")
    answer = ask_guideline(question)
    print(f"\nA: {answer}")
