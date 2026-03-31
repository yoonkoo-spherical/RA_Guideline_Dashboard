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

REASONING_MODEL = "gemini-2.5-pro"
FAST_MODEL = "gemini-3.1-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

MAX_TOTAL_CHARS = 35000 

def execute_with_retry(api_call_func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return api_call_func()
        except Exception as e:
            error_msg = str(e)
            if any(code in error_msg for code in ["503", "429", "UNAVAILABLE", "500", "INTERNAL"]):
                if attempt < max_retries - 1:
                    sleep_time = 2 ** (attempt + 1)
                    time.sleep(sleep_time)
                else:
                    raise e
            else:
                raise e

def get_embedding(text: str):
    def _call_embed():
        return client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
    try:
        response = execute_with_retry(_call_embed)
        return response.embeddings[0].values
    except Exception:
        return None

def analyze_intent_advanced(user_query: str) -> dict:
    """사용자의 의도를 심층 분석하여 다각도 검색 파라미터를 생성합니다."""
    prompt = f"""
    사용자의 규제 관련 질의를 분석하여 DB 검색을 위한 심층 파라미터를 생성하십시오.
    결과는 반드시 JSON 형식으로만 출력하십시오.

    [분석 요구사항]
    1. core_topics: 질의의 핵심 주제어를 영문으로 변환하십시오. (예: "설비 변경" -> "Equipment Change")
    2. expanded_queries: 의미적으로 연관된 전문 규제 용어를 포함한 질의문을 3개 이상 생성하십시오. 
       (예: "제조 공정 변경" -> ["post-approval change management protocol", "comparability protocol for manufacturing changes", "regulatory reporting categories for equipment"])
    3. agency_focus: 언급된 기관(FDA, EMA 등)을 정규화하십시오.
    4. document_type: 특정 문서 유형(Guidance, Draft, Requirement 등)이 감지되면 명시하십시오.

    {{
        "expanded_queries": [],
        "core_topics": [],
        "agency_focus": null,
        "document_type": null
    }}

    질의: {user_query}
    """
    def _call_analyze():
        return client.models.generate_content(
            model=FAST_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
        )
    try:
        response = execute_with_retry(_call_analyze)
        return json.loads(response.text)
    except Exception:
        return {"expanded_queries": [user_query], "core_topics": [], "agency_focus": None, "document_type": None}

def ask_guideline(user_query: str):
    accessed_sources = []
    all_chunks = {}

    # 1. 의도 분석 및 검색어 확장
    params = analyze_intent_advanced(user_query)
    
    # 2. 다중 쿼리 벡터 검색 (의미 기반)
    search_list = params.get("expanded_queries", []) + [user_query]
    for q in search_list:
        embedding = get_embedding(q)
        if embedding:
            try:
                # 검색 임계값을 낮추어 의미적 연관성이 있는 후보를 넓게 수집
                res = supabase.rpc("match_document_chunks_with_filters", {
                    "query_embedding": embedding,
                    "match_threshold": 0.20, 
                    "match_count": 25,
                    "filter_agency": params.get("agency_focus")
                }).execute()
                for d in res.data:
                    all_chunks[d['id']] = d
            except Exception:
                continue

    # 3. 데이터 구조화 및 Reranking
    unique_chunks = list(all_chunks.values())
    # LLM을 사용하여 질문과 가장 밀접한 실제 규제 조항 위주로 재정렬
    reranked = rerank_chunks(user_query, unique_chunks, top_n=15)

    db_context_list = []
    for d in reranked:
        # 문서 제목 정보를 포함하여 모델이 출처를 인지하게 함
        doc_info = supabase.table("guidelines").select("title, agency").eq("url", d['url']).single().execute().data
        title = doc_info.get('title', 'Unknown Document')
        agency = doc_info.get('agency', 'N/A')
        
        accessed_sources.append({"url": d['url'], "title": title})
        db_context_list.append(f"### 출처: {title} ({agency})\nURL: {d['url']}\n내용: {d['content']}")

    db_context_str = "\n\n".join(db_context_list) if db_context_list else "검색된 내부 데이터가 없습니다."

    # 4. 전략적 종합 답변 생성
    system_instruction = """
    당신은 글로벌 인허가 전략을 수립하는 30년 경력의 RA 수석 컨설턴트입니다.
    제공된 [내부 DB 데이터]를 기반으로 사용자의 질문에 대해 단순 나열이 아닌 '심화된 분석'과 '전략적 의견'을 제시하십시오.

    [답변 구성 지침]
    1. 사실 기반 분석: 모든 답변은 철저히 [내부 DB 데이터]에 근거해야 합니다. 추측을 배제하고 객관적 사실관계를 먼저 정리하십시오.
    2. 전문가적 통찰: 규제 조항들 사이의 연관성을 파악하여, 해당 규정이 실무(예: 설비 변경 등)에 미치는 영향과 대응 우선순위를 제언하십시오.
    3. 출처 매핑(필수): 답변의 핵심 문장이나 단락 뒤에 반드시 해당 내용의 근거가 된 문서명을 기록하십시오. (예: ~해야 합니다. [출처: FDA Equipment Guidance])
    4. 정보 부족 시 대응: DB에 명확한 답변이 없을 경우, 관련성이 높은 유사 조항을 바탕으로 논리적인 추론을 제시하되 반드시 'DB 내 직접적인 언급은 없으나 유사 사례를 바탕으로 한 제언'임을 밝히십시오.
    5. 서술 톤: 비유를 배제하고 담백하며 전문적인 경어체를 사용하십시오. 과장이나 아첨은 금지합니다.
    """

    final_prompt = f"""
    [사용자 질의]
    {user_query}

    [내부 DB 데이터]
    {db_context_str}

    위 데이터를 바탕으로 심화 분석 리포트를 작성하십시오. 답변 마지막에는 참고한 DB 문서 리스트를 정리하십시오.
    """

    try:
        chat = client.chats.create(
            model=REASONING_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                tools=[types.Tool(google_search=types.GoogleSearch())] # 최신 동향 보완용
            )
        )
        response = execute_with_retry(lambda: chat.send_message(final_prompt))
        
        # 중복 소스 제거
        unique_sources = []
        seen = set()
        for s in accessed_sources:
            if s['url'] not in seen:
                unique_sources.append(s)
                seen.add(s['url'])

        return response.text, unique_sources
    except Exception as e:
        return f"분석 중 오류가 발생했습니다: {str(e)}", []

def rerank_chunks(user_query: str, chunks: list[dict], top_n: int = 15) -> list[dict]:
    if not chunks: return []
    
    # 청크 메타데이터 요약본 생성
    input_list = "\n".join([f"ID:{i} | 내용:{c['content'][:200]}" for i, c in enumerate(chunks)])
    prompt = f"질문: {user_query}\n\n다음 리스트 중 질문의 핵심 규제 요건을 가장 잘 설명하는 항목 15개의 ID를 중요도 순서대로 나열하십시오. 결과는 오직 JSON 숫자 배열로만 출력하십시오.\n\n{input_list}"
    
    try:
        res = client.models.generate_content(
            model=FAST_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
        )
        indices = json.loads(res.text)
        return [chunks[i] for i in indices if i < len(chunks)]
    except Exception:
        return chunks[:top_n]
