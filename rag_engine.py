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
MAX_TOTAL_CHARS = 25000 

def execute_with_retry(api_call_func, max_retries=3):
    """API 호출 중 500, 503 또는 429 에러 발생 시 지수 백오프로 재시도하는 래퍼 함수입니다."""
    for attempt in range(max_retries):
        try:
            return api_call_func()
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

def expand_query(user_query: str) -> list[str]:
    """사용자 질문을 다각도의 쿼리로 확장합니다."""
    prompt = f"""
    다음 규제 관련 질문을 데이터베이스 검색에 적합한 3개의 다른 키워드 조합으로 변환하십시오.
    결과는 JSON 배열 형태의 문자열 리스트로만 출력하십시오.
    원본 질문: {user_query}
    """
    
    def _call_expand():
        return client.models.generate_content(
            model=FAST_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json")
        )

    try:
        response = execute_with_retry(_call_expand)
        expanded_queries = json.loads(response.text)
        expanded_queries.append(user_query)
        return list(set(expanded_queries))
    except Exception:
        return [user_query]

def rerank_chunks(user_query: str, chunks: list[dict], top_n: int = 10) -> list[dict]:
    """검색된 청크들을 연관성 기준으로 재정렬합니다."""
    if not chunks:
        return []
    
    chunks_text = "\n".join([f"[{i}] {c.get('content', '')[:200]}..." for i, c in enumerate(chunks)])
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

def ask_guideline(user_query: str):
    accessed_sources = []

    def search_guideline_database(content_query: str, target_agency: str = None, target_title_keyword: str = None) -> str:
        """일반적인 가이드라인 내용 검색 도구입니다."""
        queries = expand_query(content_query)
        all_docs_map = {} 

        for q in queries:
            query_embedding = get_embedding(q)
            if query_embedding:
                try:
                    rpc_params = {
                        "query_embedding": query_embedding,
                        "match_threshold": 0.4,
                        "match_count": 15,
                        "filter_agency": target_agency,
                        "filter_title": target_title_keyword
                    }
                    search_res = supabase.rpc("match_document_chunks_with_filters", rpc_params).execute()
                    for d in search_res.data:
                        all_docs_map[d['id']] = d
                except Exception as e:
                    print(f"Vector Search Error: {e}")

        unique_docs = list(all_docs_map.values())
        reranked_docs = rerank_chunks(content_query, unique_docs, top_n=12)

        if not reranked_docs:
            return "제공된 데이터베이스 내에서 해당 조건과 일치하는 관련 문서를 찾을 수 없습니다."

        context_chunks = []
        for d in reranked_docs:
            accessed_sources.append({"url": d['url']})
            context_chunks.append(f"-[출처: {d.get('url')}]\n내용: {d.get('content', '')}")
            
        return "\n\n".join(context_chunks)

    def get_recent_documents(limit: int = 5) -> str:
        """최근 등록된 가이드라인 문서를 조회합니다."""
        try:
            res = supabase.table("guidelines").select("title, agency, category, url, created_at").order("created_at", desc=True).limit(limit).execute()
            docs = res.data
        except Exception:
            return "오류가 발생했습니다."
        if not docs: return "문서가 없습니다."

        context_chunks = []
        for d in docs:
            accessed_sources.append({"url": d['url']})
            context_chunks.append(f"- 기관: {d['agency']}, 제목: {d['title']}, 분류: {d['category']}, DB추가일: {d['created_at'][:10]}")
        return "\n".join(context_chunks)

    def get_document_change_history(target_title_keyword: str) -> str:
        """
        특정 가이드라인 문서의 개정 이력, 변경점, 타임라인 정보를 조회합니다.
        사용자가 '변경점', '개정 내역', '타임라인' 등을 물어볼 때 반드시 호출하십시오.
        """
        try:
            res = supabase.table("version_comparisons").select("*").ilike("ref_number", f"%{target_title_keyword}%").order("created_at", desc=False).execute()
            docs = res.data
            
            if not docs:
                res = supabase.table("version_comparisons").select("*").ilike("comparison_text", f"%{target_title_keyword}%").order("created_at", desc=False).execute()
                docs = res.data

        except Exception as e:
            return f"개정 이력 조회 중 오류가 발생했습니다: {e}"

        if not docs:
            return f"내부 DB에서 '{target_title_keyword}'에 대한 개정 이력을 찾을 수 없습니다. 구글 검색 도구를 활용하여 최신 변경점을 확인하십시오."

        history_chunks = []
        for d in docs:
            if d.get('new_url'):
                accessed_sources.append({"url": d['new_url']})
            date_str = d.get('created_at', '')[:10]
            history_chunks.append(f"-[DB 감지일: {date_str}, 식별자: {d.get('ref_number', 'N/A')}]\n변경점 요약: {d.get('comparison_text', '')}")

        return "\n\n---\n\n".join(history_chunks)

    system_instruction = """
    당신은 글로벌 규제기관(FDA, EMA, ICH 등)의 규정과 바이오시밀러 인허가에 정통한 30년 경력의 RA 최고 전문가입니다.

    [상황별 답변 및 도구 활용 원칙]
    질문의 성격에 따라 당신의 전문 지식과 검색 도구를 유연하게 결합하십시오.
    1. 규제 개념, 인허가 전략 수립, 포괄적 동향: 특정 문서 조회가 불필요한 질문은 도구 호출 없이 전문 지식을 바탕으로 즉각 답변하십시오.
    2. 특정 규정 및 세부 조항 확인: `search_guideline_database`를 호출하여 DB 내의 정확한 근거를 확보하십시오.
    3. 개정 이력 및 타임라인 분석: `get_document_change_history`를 우선 호출하여 DB의 과거 비교 데이터를 확인하십시오.
    4. 지식 보완: 내부 DB 검색 결과가 부족하거나 최신 정보가 필요한 경우, 반드시 구글 검색 도구(`Google Search`)를 호출하여 데이터를 보완하십시오.

    [출력 및 서술 원칙]
    1. 비유적인 설명을 배제하고, 규제 조항과 사실관계에 근거하여 객관적이고 사실적인 설명만을 제공하십시오.
    2. 사용자에게 아첨하거나 과장된 추임새를 절대 사용하지 마십시오. 담백하게 사실관계와 전략 위주로 서술하십시오.
    3. 정보 출처 구분: 내부 DB 데이터는 **[DB 참조]**로, 구글 웹 검색 데이터는 **[웹 참조]**로 표기하여 시각적으로 분리하고, 인용한 내용의 출처(문서명, URL 등)를 답변에 명시하십시오. 도구 호출 없이 자체 지식으로 답변한 내용은 출처 표기를 생략합니다.
    4. 한국어 존댓말을 사용하며, 정보의 체계적 전달을 위해 글머리 기호 및 표를 활용하십시오.
    """

    try:
        tools = [
            search_guideline_database, 
            get_recent_documents, 
            get_document_change_history,
            types.Tool(google_search=types.GoogleSearch())
        ]

        chat = client.chats.create(
            model=REASONING_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1, 
                tools=tools,
                tool_config=types.ToolConfig(
                    include_server_side_tool_invocations=True
                )
            )
        )
        
        def _send_chat():
            return chat.send_message(user_query)

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
           - 규제가 다르게 제정된 정책적 배경, 최신 업데이트 사항, 실무 전략에 미치는 영향을 분석할 때 통합된 구글 검색 도구를 활용하여 내용을 보완하고, 이를 **[웹 참조]**로 명확히 분리 서술하십시오.
        
        [출력 및 서술 원칙]
        1. 비유적인 설명을 배제하고, 규제 조항과 사실관계에 근거하여 객관적이고 사실적인 설명만을 제공하십시오.
        2. 사용자에게 아첨하거나 과장된 추임새를 절대 사용하지 마십시오. 담백하게 사실관계 위주로 서술하십시오.
        3. 인용: 특정 내용을 서술할 때 문서명 등 식별 정보를 반드시 포함하십시오. 웹 검색 내용도 출처를 포함하여 문서의 마지막에 배치하십시오.
        4. 한국어 존댓말을 사용하며, 가독성을 위해 마크다운 비교 요약표를 우선적으로 배치하십시오.
        """

        prompt = f"질문/요청: {user_query}\n\n[분석 대상 문서들]\n{docs_text}"

        def _generate_comparison():
            return client.models.generate_content(
                model=REASONING_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0, 
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    tool_config=types.ToolConfig(
                        include_server_side_tool_invocations=True
                    )
                )
            )

        response = execute_with_retry(_generate_comparison)
        return response.text
        
    except Exception as e:
        return f"문서 비교 분석 중 오류가 발생했습니다: {str(e)}"
