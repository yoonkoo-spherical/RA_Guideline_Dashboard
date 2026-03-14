import os
from google import genai
from supabase import create_client, Client
from dotenv import load_dotenv

try:
    SUPABASE_URL = os.environ.get("SUPABASE_URL") or st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or st.secrets["SUPABASE_KEY"]
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError, Exception):
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

# 지능형 라우팅 모델 설정
ROUTING_MODEL = 'gemini-2.5-flash-lite' 
GENERATION_MODEL = 'gemini-2.5-flash' 
EMBEDDING_MODEL = 'gemini-embedding-001'

def record_token_usage(model_name, input_tokens, output_tokens):
    """토큰 사용량 DB 기록"""
    try:
        supabase.table("token_usage").insert({
            "model_name": model_name, 
            "input_tokens": input_tokens, 
            "output_tokens": output_tokens
        }).execute()
    except Exception as e:
        print(f"토큰 기록 실패: {e}")

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

    # Reranking: 가장 관련성 높은 상위 5개 청크만 활용하여 노이즈 감소
    top_chunks = chunks[:5]
    context_text = "\n\n".join([f"[출처 번호: {i+1}] (URL: {c['url']})\n{c['content']}" for i, c in enumerate(top_chunks)])
    
    # 할루시네이션 방지 강화 프롬프트
    prompt = f"""
    당신은 8년 이상 경력의 규제과학(Regulatory Science) 전문가 및 Medical Writer입니다.
    제공된 [참고 문서]만을 바탕으로 규제기관 제출용 문서(CTD) 작성이나 인허가 전략 수립에 활용될 수 있는 답변을 작성하십시오.

    [할루시네이션 방지 엄격 지침]
    1. 신뢰도 평가: 답변 상단에 '신뢰도: [높음/중간/낮음]'을 명시하십시오. 참고 문서에 관련 정보가 전혀 없다면 '낮음'으로 표기하고 임의의 답변을 생성하지 마십시오.
    2. 인용구 엄격 제한: 특정 기준, 수치, 통계적 허용 한계(Margin)를 언급할 때 반드시 해당 내용이 도출된 [출처 번호: N]를 문장 끝에 명시하십시오.
    3. 자기 비판(Self-Critique): 최종 출력 전, 작성한 내용이 참고 문서와 논리적 모순이 없는지 검증하십시오. 비유적 표현을 배제하고 공식 규제 용어만 사용하십시오.

    [참고 문서]
    {context_text}

    [질문]
    {query_text}
    """
    
    try:
        response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
        
        # 토큰 기록
        in_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        out_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        record_token_usage(GENERATION_MODEL, in_tokens, out_tokens)
        
        return response.text, top_chunks 
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
    
    prompt = f"""
    당신은 글로벌 바이오시밀러 인허가 전략을 총괄하는 RA 전문가입니다. 
    제공된 여러 가이드라인 문서를 대조하여 규제적 차이(Gap Analysis)를 수행하십시오.

    [시각적 가독성 원칙]
    - 전체 내용은 명확한 제목(###)과 구분선(---)으로 단락을 분리하십시오.
    - 긴 문장은 피하고 글머리 기호(-)를 사용하여 핵심만 간결하게 요약하십시오.
    - 중요한 키워드나 수치는 **굵게** 표시하십시오.

    [작성 지침]
    1. 품질(CMC), 비임상, 임상(PK/PD, 면역원성), 적응증 외삽 관점에서 규제 접근 방식의 차이를 객관적으로 도출하십시오.
    2. 핵심 차이점은 반드시 아래 마크다운 표(Table) 형식을 사용하여 한눈에 보이게 정리하십시오. 표 안의 텍스트는 개조식으로 짧게 쓰십시오.

    | 평가 항목 (CMC/면역원성 등) | 문서 A 규제 기준 | 문서 B 규제 기준 | 규제적 차이(Gap) 및 전략적 유의점 |
    |---|---|---|---|
    | ... | ... | ... | ... |

    3. 지역별 규제 차이로 인해 발생할 수 있는 허가 지연 리스크와 데이터 준비 전략을 하단에 별도 단락으로 요약하십시오.

    [비교 대상 문서 텍스트]
    {combined_context}
    """
    try:
        response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
        
        # 토큰 기록
        in_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        out_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        record_token_usage(GENERATION_MODEL, in_tokens, out_tokens)
        
        return response.text
    except Exception as e:
        return f"다중 문서 비교 오류: {e}"
