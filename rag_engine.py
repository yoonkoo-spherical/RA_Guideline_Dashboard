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

# 지능형 라우팅: 의도 파악은 가벼운 모델, 실제 전문가 답변 생성은 상위 모델(1.5 Pro) 사용
ROUTING_MODEL = 'gemini-2.5-flash-lite' 
GENERATION_MODEL = 'gemini-1.5-pro' 
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
    prompt = f"사용자의 질문이 제약/바이오 규제, 바이오시밀러, 가이드라인과 관련된 질문인지 판단하십시오. 관련이 있다면 'RAG', 일반 대화라면 'GENERAL'이라고 답변하십시오.\n질문: {query_text}"
    try:
        response = client.models.generate_content(model=ROUTING_MODEL, contents=prompt)
        return response.text.strip().upper()
    except Exception:
        return "RAG"

def generate_rag_response(query_text, chunks):
    if not chunks:
        return "현재 데이터베이스 내에서 질문과 관련된 규제 가이드라인 내용을 찾을 수 없습니다. 다른 키워드로 검색해 보십시오.", []

    context_text = "\n\n".join([f"[출처 번호: {i+1}] (URL: {c['url']})\n{c['content']}" for i, c in enumerate(chunks)])
    
    # 실무 인허가 문서(CTD) 작성 지원을 위한 고도화 프롬프트
    prompt = f"""
    당신은 8년 이상 경력의 규제과학(Regulatory Science) 전문가 및 Medical Writer입니다.
    제공된 [참고 문서]를 바탕으로, 실제 규제기관 제출용 문서(CTD) 작성이나 인허가 전략 수립에 즉시 활용될 수 있는 최고 수준의 답변을 작성하십시오.

    [지침]
    1. 비유적 표현을 엄격히 배제하고, ICH, FDA, EMA에서 통용되는 공식 규제 용어를 사용하여 객관적이고 사실적으로 작성하십시오.
    2. 참고 문서에 명시된 구체적인 수치, 통계적 허용 한계(Margin), 평가 지표, 규제 당국의 권고 사항을 누락 없이 포함하십시오.
    3. 정보가 불충분할 경우 임의로 추론하지 말고, "해당 가이드라인 원문 조각에는 관련 세부 기준이 명시되어 있지 않습니다"라고 명확히 선을 그으십시오.
    4. 텍스트 내에서 특정 기준을 언급할 때 반드시 해당 내용이 도출된 [출처 번호: N]를 문장 끝에 명시하십시오.

    [참고 문서]
    {context_text}

    [질문]
    {query_text}
    """
    
    try:
        response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
        return response.text, chunks 
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
            context_parts.append(f"--- [문서: {title}] ---\n{full_text[:30000]}")
        else:
            context_parts.append(f"--- [문서: {title}] ---\n(임베딩 텍스트 없음)")

    combined_context = "\n\n".join(context_parts)
    
    # 다중 기관/문서 간 Gap Analysis 프롬프트
    prompt = f"""
    당신은 글로벌 바이오시밀러 인허가 전략을 총괄하는 RA 전문가입니다. 
    제공된 여러 가이드라인 문서를 대조하여 규제적 차이(Gap Analysis)를 수행하십시오.

    [지침]
    1. 품질(CMC), 비임상, 임상(PK/PD, 면역원성), 적응증 외삽 관점에서 문서 간의 규제 접근 방식과 허용 기준의 차이를 객관적으로 도출하십시오.
    2. 지역별(FDA vs EMA 등) 규제 차이로 인해 발생할 수 있는 허가 지연 리스크와 데이터 준비 전략을 결론에 포함하십시오.
    3. 결과는 반드시 아래 표 형식을 사용하여 작성하십시오.

    | 평가 항목 (CMC/면역원성 등) | 문서 A 규제 기준 | 문서 B 규제 기준 | 규제적 차이(Gap) 및 전략적 유의점 |
    |---|---|---|---|
    | ... | ... | ... | ... |

    [비교 대상 문서 텍스트]
    {combined_context}
    """
    try:
        response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
        return response.text
    except Exception as e:
        return f"다중 문서 비교 오류: {e}"
