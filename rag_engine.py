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

# Gemini SDK 클라이언트 초기화
client = genai.Client(api_key=GEMINI_API_KEY)

REASONING_MODEL = "gemini-2.5-pro"
FAST_MODEL = "gemini-3.1-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

# Tier 1 환경 한도 준수를 위한 최대 전송 글자 수
MAX_TOTAL_CHARS = 35000 

def execute_with_retry(api_call_func, max_retries=3):
    """API 호출 중 에러 발생 시 지수 백오프로 재시도하고, 성공 시 토큰 사용량을 DB에 자동 기록합니다."""
    for attempt in range(max_retries):
        try:
            response = api_call_func()
            
            # API 응답에서 토큰 사용량을 추출하여 DB에 누적
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                try:
                    in_tokens = response.usage_metadata.prompt_token_count
                    out_tokens = response.usage_metadata.candidates_token_count
                    if in_tokens or out_tokens:
                        supabase.table("token_usage").insert({
                            "input_tokens": in_tokens,
                            "output_tokens": out_tokens
                        }).execute()
                except Exception:
                    pass
            
            return response
            
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "429" in error_msg or "UNAVAILABLE" in error_msg or "500" in error_msg or "INTERNAL" in error_msg:
                if attempt < max_retries - 1:
                    sleep_time = 2 ** (attempt + 1)
                    log_message = f"API 서버 지연/에러 발생. {sleep_time}초 후 재시도합니다... (시도: {attempt+1}/{max_retries-1})"

                    print(log_message)

                    try:
                        st.toast(log_message, icon="⏳")
                    except Exception:
                        pass

                    time.sleep(sleep_time)
                else:
                    raise e
            else:
                raise e

def get_embedding(text: str):
    """텍스트를 벡터 임베딩으로 변환합니다."""
    def _call_embed():
        return client.models.embed_content(model=EMBEDDING_MODEL, contents=text)

    try:
        response = execute_with_retry(_call_embed)
        return response.embeddings[0].values
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def analyze_intent_and_extract_params(user_query: str) -> dict:
    """사용자 질문을 분석하여 내부 DB 검색용 파라미터를 JSON으로 추출합니다."""
    prompt = f"""
    사용자 질의를 분석하여 내부 DB 검색을 위한 파라미터를 추출하십시오.
    단순한 단어의 나열(Keyword matching)이 아닌, 문서의 의미(Semantic) 기반으로 검색할 수 있도록 파라미터를 구성하십시오.
    결과는 반드시 아래 JSON 형식으로만 출력하십시오.
    
    [추출 가이드]
    1. search_queries: 규제 가이드라인은 주로 영문이므로, 사용자의 한국어 질의가 의미하는 바를 **자연어 형태의 영문 질의문(Question)**과 **핵심 개념을 포괄하는 영문 구문(Phrase)**으로 2~3개 생성하십시오. 
       (예: "제조 공정 설비 변경 시 FDA 대응" -> ["post-approval manufacturing equipment changes FDA", "reporting categories for process equipment change biosimilar"])
    2. target_agency: 질의에 언급된 규제 기관명을 다음 중 하나로 정확히 정규화하십시오: "FDA", "EMA", "MHRA", "Health Canada", "ICH", "MFDS". 해당하지 않거나 불분명하면 null을 반환하십시오.
    3. target_title: 질의에 특정 문서명이 명시된 경우 이를 추출하십시오. 없으면 null입니다.
    4. history_keyword: 개정 이력, 변경점 관련 질문일 경우 대상 문서 식별자를 추출하십시오. 없으면 null입니다.

    {{
        "search_queries": ["영문 의미 기반 검색어 1", "영문 의미 기반 검색어 2"],
        "target_agency": "정규화된 기관명 또는 null",
        "target_title": "특정 문서명 또는 null",
        "history_keyword": "개정 이력 식별자 또는 null"
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
        params = json.loads(response.text)
        if "search_queries" not in params or not isinstance(params["search_queries"], list):
            params["search_queries"] = []
        return params
    except Exception:
        return {"search_queries": [], "target_agency": None, "target_title": None, "history_keyword": None}

def rerank_chunks(user_query: str, chunks: list[dict], top_n: int = 10) -> list[dict]:
    """검색된 청크들을 연관성 기준으로 재정렬합니다."""
    if not chunks:
        return []

    chunks_text = "\n".join([f"[{i}] {c.get('content', '')[:300]}..." for i, c in enumerate(chunks)])
    prompt = f"""
    질문: {user_query}
    다음 검색된 문서 청크들 중 질문에 답변하는 데 가장 관련성이 높은 청크의 번호를 연관성 순으로 최대 {top_n}개까지 나열하십시오.
    결과는 JSON 배열(숫자 리스트) 형식으로만 출력하십시오.
    청크 목록:
    {chunks_text}
    """

    def _call_rerank():
        return client.models.generate_content(
            model=FAST_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
        )

    try:
        response = execute_with_retry(_call_rerank)
        top_indices = json.loads(response.text)
        return [chunks[i] for i in top_indices if i < len(chunks)]
    except Exception:
        return chunks[:top_n]

def ask_guideline(user_query: str, forced_docs: list = None):
    if forced_docs is None:
        forced_docs = []

    accessed_sources = []
    db_context_parts = []

    # 0단계: 필수 참조 문서 주입
    if forced_docs:
        forced_chunks = []
        for doc in forced_docs:
            try:
                chunk_res = supabase.table("document_chunks").select("content").eq("url", doc['url']).order("chunk_index").execute()
                full_text = " ".join([c['content'] for c in chunk_res.data])
                # 컨텍스트 초과 방지를 위해 길이 제한 적용
                if len(full_text) > 20000:
                    full_text = full_text[:20000] + "...(중략)"
                forced_chunks.append(f"-[필수 참조 문서: {doc.get('title', 'Unknown')} ({doc.get('url', '')})]\n원문 내용: {full_text}")
                accessed_sources.append({"url": doc['url'], "title": doc.get('title', '필수 참조 문서')})
            except Exception as e:
                print(f"Forced doc error: {e}")
        
        if forced_chunks:
            db_context_parts.append("[사용자 지정 필수 참조 문서]\n" + "\n\n".join(forced_chunks))

    # 1단계: 내부 DB 사전 검색 수행
    params = analyze_intent_and_extract_params(user_query)

    # 개정 이력 검색 (필요한 경우)
    history_keyword = params.get("history_keyword")
    if history_keyword:
        try:
            res = supabase.table("version_comparisons").select("*").ilike("ref_number", f"%{history_keyword}%").order("created_at", desc=False).execute()
            docs = res.data
            if not docs:
                res = supabase.table("version_comparisons").select("*").ilike("comparison_text", f"%{history_keyword}%").order("created_at", desc=False).execute()
                docs = res.data

            if docs:
                history_chunks = []
                for d in docs:
                    if d.get('new_url'): accessed_sources.append({"url": d['new_url'], "title": f"개정 이력: {d.get('ref_number', 'N/A')}"})
                    date_str = d.get('created_at', '')[:10]
                    history_chunks.append(f"-[DB 감지일: {date_str}, 식별자: {d.get('ref_number', 'N/A')}]\n변경점 원문: {d.get('comparison_text', '')}")
                db_context_parts.append("[문서 개정 이력 및 타임라인]\n" + "\n\n".join(history_chunks))
        except Exception as e:
            print(f"History search error: {e}")

    # 일반 문서 청크 검색 (의미 기반 다중 쿼리 + 사용자 한국어 원문 쿼리 포함)
    all_docs_map = {} 
    target_agency = params.get("target_agency")
    target_title = params.get("target_title")
    
    search_queries = params.get("search_queries", [])
    if user_query not in search_queries:
        search_queries.append(user_query)

    for q in search_queries:
        query_embedding = get_embedding(q)
        if query_embedding:
            try:
                rpc_params = {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.25,  
                    "match_count": 30,        
                    "filter_agency": target_agency,
                    "filter_title": target_title
                }
                search_res = supabase.rpc("match_document_chunks_with_filters", rpc_params).execute()
                for d in search_res.data:
                    all_docs_map[d['id']] = d
            except Exception as e:
                print(f"Vector Search Error: {e}")

    unique_docs = list(all_docs_map.values())
    reranked_docs = rerank_chunks(user_query, unique_docs, top_n=10)

    if reranked_docs:
        chunk_texts = []
        for d in reranked_docs:
            accessed_sources.append({"url": d['url'], "title": d.get('title', 'DB 검색 문서')})
            chunk_texts.append(f"-[출처: {d.get('url')}]\n원문 내용: {d.get('content', '')}")
        db_context_parts.append("[일반 규정 검색 결과]\n" + "\n\n".join(chunk_texts))

    db_context_str = "\n\n".join(db_context_parts)
    
    if len(db_context_str) > 30000:
        db_context_str = db_context_str[:30000] + "\n...(컨텍스트 토큰 한도 제한으로 중략됨)"
    if not db_context_str.strip():
        db_context_str = "내부 DB에서 관련된 정보를 찾지 못했습니다."

    # 2단계: 메인 모델 (DB 데이터 주입 + 구글 웹 검색 연동)
    system_instruction = """
    당신은 글로벌 규제기관(FDA, EMA, ICH 등)의 규정과 바이오시밀러 인허가에 정통한 30년 경력의 RA 최고 전문가입니다.

    [상황별 답변 및 데이터 활용 원칙]
    1. 최우선 참조 원칙: 프롬프트에 [사용자 지정 필수 참조 문서]가 있는 경우, 반드시 해당 문서를 최우선으로 검토 및 참조하여 답변을 구성하십시오.
    2. 추론 및 보완: 필수 참조 문서 외에 제공된 [일반 규정 검색 결과]는 답변 도출 추론 과정에서 필요한 경우 추가로 참조하십시오.
    3. 원문 포함 서술: 답변 내용의 이해를 명확히 돕기 위해, 참조한 문헌(필수 문서 및 일반 검색 문서 모두)의 핵심 원문 구절이나 내용을 답변 내에 직접 발췌하여 포함하십시오.
    4. 지식 보완 (웹 검색): 주입된 DB 정보가 부족하거나 최신 규제 동향 파악이 필요한 경우, 내장된 구글 검색 도구(`Google Search`)를 호출하여 지식을 보완하십시오.

    [출력 및 서술 원칙]
    1. 비유적인 설명을 철저히 배제하고, 규제 조항과 사실관계에 근거하여 객관적이고 사실적인 설명만을 제공하십시오.
    2. 사용자에게 아첨하거나 과장된 추임새를 절대 사용하지 마십시오. 담백하게 사실관계 위주로 답변하십시오.
    3. 정보 출처 구분: 제공된 DB 데이터 기반 서술은 **[DB 참조]**로 명시하며 원문을 포함하고, 웹 검색을 통해 보완된 데이터는 **[웹 참조]**로 표기하여 시각적으로 분리하십시오.
    4. 출처 링크 제공: 인용한 웹 검색 내용의 출처 URL 및 참조한 내부 DB 문서의 원문 출처 링크는 반드시 답변의 마지막 부분에 목록 형태로 명시하여 즉시 확인 가능하도록 조치하십시오.
    5. [웹 참조]의 출처 URL 링크는 환각 정보를 제공하지 않도록 엄격히 검증한 후 제공하십시오.
    6. 한국어 존댓말을 사용하며, 가독성을 위해 마크다운 표와 글머리 기호를 적절히 활용하십시오.
    """

    final_prompt = f"[내부 DB 정보]\n{db_context_str}\n\n---\n\n[사용자 질의]\n{user_query}"

    try:
        chat = client.chats.create(
            model=REASONING_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1, 
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )

        def _send_chat():
            return chat.send_message(final_prompt)

        response = execute_with_retry(_send_chat)

        unique_sources = []
        seen_urls = set()
        for source in accessed_sources:
            if source['url'] not in seen_urls:
                seen_urls.add(source['url'])
                unique_sources.append(source)

        return response.text, unique_sources

    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {str(e)}", []


def extract_core_content(text: str, query: str, max_length: int) -> str:
    """단순 절삭을 방지하기 위해 FAST_MODEL을 활용하여 핵심 문맥을 선별적으로 압축합니다."""
    prompt = f"""
    다음 문서에서 사용자의 질의와 관련된 핵심 규제 요건, 기준점, 예외 조항 등을 추출하십시오.
    객관적인 사실 관계 위주로 요약하며, 전체 텍스트 분량은 {max_length}자 내외로 구성하십시오.
    
    [사용자 질의]
    {query}
    
    [문서 원문]
    {text}
    """
    def _call_extract():
        return client.models.generate_content(
            model=FAST_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
    try:
        response = execute_with_retry(_call_extract)
        return response.text
    except Exception as e:
        print(f"Extraction failed, falling back to truncation: {e}")
        return text[:max_length]


def compare_multiple_documents(docs_info, user_query: str = "위 문서들을 종합적이고 심층적으로 비교 분석하십시오."):
    """다수의 가이드라인 문서를 객관적으로 대조하고 웹 검색을 통해 심층 분석을 보완합니다."""
    try:
        docs_text = ""
        doc_count = len(docs_info)
        chars_per_doc = MAX_TOTAL_CHARS // doc_count if doc_count > 0 else MAX_TOTAL_CHARS

        for i, doc in enumerate(docs_info):
            chunk_res = supabase.table("document_chunks").select("content").eq("url", doc['url']).order("chunk_index").execute()
            full_text = " ".join([c['content'] for c in chunk_res.data])

            # 원문이 할당된 글자 수를 초과할 경우, 핵심 내용 압축 함수 호출
            if len(full_text) > chars_per_doc:
                processed_text = extract_core_content(full_text, user_query, chars_per_doc)
            else:
                processed_text = full_text

            docs_text += f"\n\n--- [문서 {i+1}: {doc.get('title', 'Unknown')} ({doc.get('agency', 'N/A')})] ---\n{processed_text}" 

        system_instruction = """
        당신은 바이오시밀러 및 제약 인허가에 정통한 30년 경력의 RA 최고 전문가입니다.

        [다중 문서 비교 분석 원칙]
        1. 객관성 및 사실 기반: 제공된 문서에 근거하여 용어 정의, 요구 수준, 기준점 차이를 객관적으로 대조하십시오.
        2. 지식 보완 및 문맥 확장: 
           - 제공된 [분석 대상 문서들]을 통한 대조 결과는 **[DB 참조]**로 명시하고 원문의 내용을 표시하십시오.
           - 규제가 다르게 제정된 정책적 배경, 최신 업데이트 사항, 실무 전략에 미치는 영향을 분석할 때 내장된 구글 검색 도구(`Google Search`)를 활용하여 내용을 보완하고, 이를 **[웹 참조]**로 명확히 분리 서술하십시오.
        
        [출력 및 서술 원칙]
        1. 비유적인 설명을 배제하고, 규제 조항과 사실관계에 근거하여 객관적이고 사실적인 설명만을 제공하십시오.
        2. 사용자에게 아첨하거나 과장된 추임새를 절대 사용하지 마십시오. 담백하게 사실관계 위주로 서술하십시오.
        3. 인용: 특정 내용을 서술할 때 문서명 등 식별 정보를 반드시 포함하십시오. 웹 검색 내용도 출처 URL을 포함하여 문서의 마지막에 배치하십시오. 웹 출처는 반드시 출처별로 링크를 붙여서 즉시 확인이 가능하도록 하십시오.
        4. [웹 참조]의 출처 URL 링크는 환각 정보를 제공하지 않도록 엄격히 검증한 후 제공하십시오.
        5. 한국어 존댓말을 사용하며, 가독성을 위해 마크다운 비교 요약표를 우선적으로 배치하십시오.
        """

        prompt = f"질문/요청: {user_query}\n\n[분석 대상 문서들]\n{docs_text}"

        def _generate_comparison():
            return client.models.generate_content(
                model=REASONING_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0, 
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )

        response = execute_with_retry(_generate_comparison)
        return response.text

    except Exception as e:
        return f"문서 비교 분석 중 오류가 발생했습니다: {str(e)}"
