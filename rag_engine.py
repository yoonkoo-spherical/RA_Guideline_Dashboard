import os
import json
import time
from google import genai
from google.genai import types
from supabase import create_client, Client
from dotenv import load_dotenv
import streamlit as st

# 환경 변수 로드
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Supabase Client Error in rag_engine: {e}")

client = genai.Client(api_key=GEMINI_API_KEY)

# 모델 설정
REASONING_MODEL = "gemini-2.5-pro"
FAST_MODEL = "gemini-3.1-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

# Tier 1 환경 한도 및 문맥 최적화를 위한 최대 글자 수
MAX_TOTAL_CHARS = 35000 

# ---------------------------------------------------------
# 1. 유틸리티 함수 (로깅, 재시도, 임베딩)
# ---------------------------------------------------------

def log_usage(response, task_name):
    """Gemini API 응답에서 토큰 사용량을 안전하게 추출하여 DB에 기록합니다."""
    try:
        if response and hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            # Pro 모델인지 Flash 모델인지 판별하여 기록
            model_used = REASONING_MODEL if task_name in ["Chatbot", "Multi_Compare"] else FAST_MODEL
            supabase.table("token_usage").insert({
                "model_name": model_used,
                "input_tokens": usage.prompt_token_count,
                "output_tokens": usage.candidates_token_count,
                "task_name": task_name
            }).execute()
    except Exception as e:
        print(f"Usage Logging Error ({task_name}): {e}")

def execute_with_retry(api_call_func, max_retries=3):
    """지수 백오프를 적용한 API 호출 재시도 래퍼"""
    for attempt in range(max_retries):
        try:
            return api_call_func()
        except Exception as e:
            error_msg = str(e)
            if any(code in error_msg for code in ["503", "429", "UNAVAILABLE", "500", "INTERNAL"]):
                if attempt < max_retries - 1:
                    sleep_time = 2 ** (attempt + 1)
                    time.sleep(sleep_time)
                    continue
            raise e

def get_embedding(text: str):
    """텍스트를 벡터로 변환"""
    def _call_embed():
        return client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
    try:
        response = execute_with_retry(_call_embed)
        return response.embeddings[0].values
    except Exception:
        return None

# ---------------------------------------------------------
# 2. 분석 및 검색 로직
# ---------------------------------------------------------

def analyze_intent_advanced(user_query: str) -> dict:
    """의도 분석 및 다각도 검색 파라미터 생성 (개정 이력 및 특정 문서명 포함)"""
    prompt = f"""
    사용자의 규제 질의를 분석하여 DB 검색용 JSON 파라미터를 생성하십시오.
    
    [추출 가이드]
    1. expanded_queries: 한국어 질의를 영어 전문 용어로 확장한 질의문 2-3개.
    2. target_agency: "FDA", "EMA", "ICH", "MHRA", "MFDS" 중 하나로 정규화 (없으면 null).
    3. target_title: 언급된 특정 문서 제목이나 가이드라인 명칭.
    4. history_keyword: 개정 이력, 변경점 관련 질문일 경우 대상 문서 번호(예: Q1A, Q5E).

    {{
        "expanded_queries": [],
        "target_agency": null,
        "target_title": null,
        "history_keyword": null
    }}
    질의: {user_query}
    """
    try:
        response = execute_with_retry(lambda: client.models.generate_content(
            model=FAST_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
        ))
        log_usage(response, "Intent_Analysis")
        return json.loads(response.text)
    except Exception:
        return {"expanded_queries": [user_query], "target_agency": None, "target_title": None, "history_keyword": None}

def rerank_chunks(user_query: str, chunks: list[dict], top_n: int = 15) -> list[dict]:
    """LLM을 이용한 검색 결과 재정렬"""
    if not chunks: return []
    
    input_list = "\n".join([f"ID:{i} | 내용:{c.get('content', '')[:250]}" for i, c in enumerate(chunks)])
    prompt = f"질문: {user_query}\n\n위 질문에 답변하기 위해 가장 핵심적인 규제 조항을 담은 ID 15개를 중요도 순으로 나열하십시오. 결과는 오직 JSON 숫자 배열로만 출력하십시오.\n\n{input_list}"
    
    try:
        res = execute_with_retry(lambda: client.models.generate_content(
            model=FAST_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
        ))
        log_usage(res, "Reranking")
        indices = json.loads(res.text)
        return [chunks[i] for i in indices if i < len(chunks)]
    except Exception:
        return chunks[:top_n]

# ---------------------------------------------------------
# 3. 메인 비즈니스 함수 (Chatbot, Compare)
# ---------------------------------------------------------

def ask_guideline(user_query: str):
    """챗봇 메인 로직 (DB 검색 + 개정 이력 + 웹 검색 보완)"""
    accessed_sources = []
    db_context_parts = []
    all_chunks = {}

    params = analyze_intent_advanced(user_query)
    
    # [기능 Merge] 개정 이력 우선 검색
    h_kw = params.get("history_keyword")
    if h_kw:
        try:
            res = supabase.table("version_comparisons").select("*").or_(f"ref_number.ilike.%{h_kw}%,comparison_text.ilike.%{h_kw}%").execute()
            if res.data:
                history_txt = "\n".join([f"[개정정보: {d['ref_number']}] {d['comparison_text']}" for d in res.data])
                db_context_parts.append(f"### 문서 개정 및 업데이트 이력\n{history_txt}")
        except Exception: pass

    # 벡터 검색 수행
    search_list = params.get("expanded_queries", []) + [user_query]
    for q in search_list:
        embedding = get_embedding(q)
        if embedding:
            try:
                res = supabase.rpc("match_document_chunks_with_filters", {
                    "query_embedding": embedding,
                    "match_threshold": 0.22, 
                    "match_count": 25,
                    "filter_agency": params.get("target_agency"),
                    "filter_title": params.get("target_title")
                }).execute()
                for d in res.data: all_chunks[d['id']] = d
            except Exception: continue

    # 재정렬 및 컨텍스트 구성
    unique_chunks = list(all_chunks.values())
    reranked = rerank_chunks(user_query, unique_chunks, top_n=12)

    for d in reranked:
        accessed_sources.append({"url": d['url'], "title": d.get('title', 'Unknown Source')})
        db_context_parts.append(f"### 원문 출처: {d.get('title', '문서')} (URL: {d['url']})\n{d['content']}")

    db_context_str = "\n\n".join(db_context_parts) if db_context_parts else "내부 DB에 직접적인 관련 데이터가 부족합니다."

    system_instruction = """
    당신은 글로벌 인허가 전략을 수립하는 30년 경력의 RA 수석 컨설턴트입니다.
    제공된 [내부 DB 데이터]를 최우선 근거로 사용하여 사용자의 질문에 대해 심화된 분석 리포트를 작성하십시오.

    [작성 가이드라인]
    1. 전략적 의견: 단순 정보 나열이 아니라, 규제 요건이 실무에 미치는 영향과 대응 전략을 제시하십시오.
    2. 출처 매핑: 답변 중 DB 내용을 인용한 문장 끝에 반드시 해당 문서명을 명시하십시오. (예: ...해야 합니다. [출처: FDA Biosimilar Guidance])
    3. 정보 구분: DB 기반 내용은 **[DB 참조]**, 웹 검색 보완 내용은 **[웹 참조]**로 명확히 분리하십시오.
    4. 톤앤매너: 비유 없이 담백하고 전문적인 경어체를 사용하십시오. 과장이나 아첨은 절대 금지합니다.
    """

    try:
        chat = client.chats.create(
            model=REASONING_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        response = execute_with_retry(lambda: chat.send_message(f"[내부 DB 데이터]\n{db_context_str}\n\n질의: {user_query}"))
        log_usage(response, "Chatbot")
        
        # 중복 소스 제거
        final_sources = []
        seen = set()
        for s in accessed_sources:
            if s['url'] not in seen:
                final_sources.append(s)
                seen.add(s['url'])

        return getattr(response, 'text', "답변을 생성할 수 없습니다."), final_sources
    except Exception as e:
        return f"분석 중 오류가 발생했습니다: {str(e)}", []

def extract_core_content(text: str, query: str, max_length: int) -> str:
    """다중 문서 비교 시 컨텍스트 초과 방지를 위한 핵심 내용 선별 압축"""
    prompt = f"질문({query})과 관련된 핵심 규제 요건 및 기준점만 {max_length}자 내외로 추출하십시오: {text}"
    try:
        res = execute_with_retry(lambda: client.models.generate_content(
            model=FAST_MODEL, contents=prompt, config=types.GenerateContentConfig(temperature=0.0)
        ))
        log_usage(res, "Summarization")
        return getattr(res, 'text', text[:max_length])
    except:
        return text[:max_length]

def compare_multiple_documents(docs_info, user_query: str = "종합 비교 분석 요청"):
    """여러 가이드라인 문서를 객관적으로 대조 분석"""
    try:
        docs_text = ""
        doc_count = len(docs_info)
        chars_per_doc = MAX_TOTAL_CHARS // doc_count if doc_count > 0 else MAX_TOTAL_CHARS

        for i, doc in enumerate(docs_info):
            if not doc.get('url'): continue
            chunk_res = supabase.table("document_chunks").select("content").eq("url", doc['url']).order("chunk_index").execute()
            full_text = " ".join([c['content'] for c in chunk_res.data])
            
            if len(full_text) > chars_per_doc:
                processed_text = extract_core_content(full_text, user_query, chars_per_doc)
            else:
                processed_text = full_text
            
            docs_text += f"\n\n--- [문서 {i+1}: {doc.get('title', 'Unknown')} ({doc.get('agency', 'N/A')})] ---\n{processed_text}"

        system_instruction = """
        당신은 RA 최고 전문가입니다. 제공된 여러 문서들을 객관적으로 대조하여 차이점과 공통점을 분석하십시오.
        마크다운 표를 활용하여 가시성을 높이고, 실무적 관점에서의 시사점을 **[DB 참조]**와 **[웹 참조]**를 구분하여 기술하십시오.
        """
        
        response = execute_with_retry(lambda: client.models.generate_content(
            model=REASONING_MODEL,
            contents=f"요청: {user_query}\n\n[대상 문서군]\n{docs_text}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        ))
        log_usage(response, "Multi_Compare")
        return getattr(response, 'text', "비교 분석 결과를 생성할 수 없습니다.")
    except Exception as e:
        return f"분석 오류: {str(e)}"
