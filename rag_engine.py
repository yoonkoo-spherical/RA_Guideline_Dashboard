import os
import json
from google import genai
from google.genai import types
from supabase import create_client, Client
from dotenv import load_dotenv

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

REASONING_MODEL = "gemini-3.1-pro-preview"
FAST_MODEL = "gemini-3.1-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

# 비용 최적화를 위한 토큰 관리 상수
MAX_TOTAL_CHARS = 250000 

def get_embedding(text: str):
    """텍스트를 벡터 임베딩으로 변환합니다."""
    try:
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
        return response.embeddings[0].values
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def expand_query(user_query: str) -> list[str]:
    """사용자 질문을 다각도의 쿼리로 확장하여 DB 검색 누락을 방지합니다."""
    prompt = f"""
    다음 바이오시밀러 규제 관련 질문을 데이터베이스 검색에 적합한 3개의 다른 키워드 조합으로 변환하십시오.
    동의어, 전문 용어, 관련 규제 기관(FDA, EMA, MFDS 등)의 영문 약어를 포함하십시오.
    결과는 JSON 배열 형태의 문자열 리스트로만 출력하십시오.
    원본 질문: {user_query}
    """
    try:
        response = client.models.generate_content(
            model=FAST_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json")
        )
        expanded_queries = json.loads(response.text)
        expanded_queries.append(user_query) # 원본 질문 포함
        return list(set(expanded_queries))
    except Exception as e:
        print(f"Query Expansion Error: {e}")
        return [user_query]

def rerank_chunks(user_query: str, chunks: list[dict], top_n: int = 10) -> list[dict]:
    """검색된 청크들을 질문과의 실제 연관성 기준으로 재정렬합니다."""
    if not chunks:
        return []
    
    # 평가를 위한 프롬프트 구성
    chunks_text = "\n".join([f"[{i}] {c.get('content', '')[:200]}..." for i, c in enumerate(chunks)])
    prompt = f"""
    질문: {user_query}
    다음 검색된 문서 청크들 중 질문에 답변하는 데 가장 관련성이 높은 청크의 번호(인덱스)를 연관성 순으로 최대 {top_n}개까지 나열하십시오.
    결과는 JSON 배열(숫자 리스트) 형식으로만 출력하십시오.
    
    청크 목록:
    {chunks_text}
    """
    try:
        response = client.models.generate_content(
            model=FAST_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
        )
        top_indices = json.loads(response.text)
        return [chunks[i] for i in top_indices if i < len(chunks)]
    except Exception as e:
        print(f"Reranking Error: {e}")
        return chunks[:top_n]

def ask_guideline(user_query: str):
    """
    사용자의 질문 의도를 파악하여 하이브리드 검색 및 외부 웹 검색을 수행하고,
    결과를 종합하여 최고 수준의 RA 전문 답변을 생성합니다.
    """
    accessed_sources = []

    def search_guideline_hybrid(query: str) -> str:
        """
        벡터 유사도와 키워드(BM25) 검색을 결합한 하이브리드 검색을 수행하고, 결과를 재정렬합니다.
        """
        queries = expand_query(query)
        all_docs_map = {} # 중복 제거용 딕셔너리

        for q in queries:
            # 1. Vector Search (의미 기반)
            query_embedding = get_embedding(q)
            if query_embedding:
                try:
                    search_res = supabase.rpc("match_document_chunks", {
                        "query_embedding": query_embedding,
                        "match_threshold": 0.4, # 기준을 다소 낮추어 누락 방지
                        "match_count": 15
                    }).execute()
                    
                    for d in search_res.data:
                        all_docs_map[d['id']] = d
                except Exception as e:
                    print(f"Vector Search Error: {e}")

            # 2. Keyword Search (단어/규정 번호 기반 누락 방지)
            # 주의: Supabase에 'content' 컬럼 대상의 FTS(Full Text Search)가 구성되어 있어야 가장 효과적입니다.
            try:
                # 띄어쓰기 기준으로 키워드 추출하여 텍스트 검색 (간이 형태)
                keywords = " | ".join(q.split()) 
                text_res = supabase.table("document_chunks").select("*").textSearch("content", keywords).limit(10).execute()
                for d in text_res.data:
                    all_docs_map[d.get('id', d['url'])] = d
            except Exception as e:
                print(f"Text Search Error: {e}")

        unique_docs = list(all_docs_map.values())
        
        # 3. Context Re-ranking
        reranked_docs = rerank_chunks(query, unique_docs, top_n=12)

        if not reranked_docs:
            return "제공된 내부 데이터베이스 내에서 관련 문서를 찾을 수 없습니다."

        context_chunks = []
        for d in reranked_docs:
            accessed_sources.append({"url": d['url']})
            # Semantic Chunking 메타데이터(section, clause 등)가 DB에 존재할 경우를 고려한 포맷팅
            section_info = f" (섹션: {d.get('section', 'N/A')}, 조항: {d.get('clause', 'N/A')})" if 'section' in d else ""
            context_chunks.append(f"-[출처: {d.get('url')}{section_info}]\n내용: {d.get('content', '')}")
            
        return "\n\n".join(context_chunks)

    def get_recent_documents(limit: int = 5) -> str:
        """최근에 등록된 가이드라인 문서를 조회합니다."""
        try:
            res = supabase.table("guidelines").select("title, agency, category, url, created_at").order("created_at", desc=True).limit(limit).execute()
            docs = res.data
        except Exception as e:
            return "최근 문서 목록을 조회하는 중 오류가 발생했습니다."

        if not docs:
            return "데이터베이스에 등록된 문서가 없습니다."

        context_chunks = []
        for d in docs:
            accessed_sources.append({"url": d['url']})
            context_chunks.append(f"- 기관: {d['agency']}, 제목: {d['title']}, 분류: {d['category']}, DB추가일: {d['created_at'][:10]}")
            
        return "\n".join(context_chunks)

    # RA 30년차 최고 전문가 시스템 프롬프트
    system_instruction = """
    당신은 글로벌 바이오시밀러 제약사에서 30년 이상의 경력을 쌓은 최고 수준의 인허가(RA) 전문 컨설턴트입니다. FDA, EMA, MFDS, ICH 가이드라인과 실무 적용 사례의 미세한 맥락에 완벽히 정통합니다.

    [작업 원칙]
    1. 상황 판단 및 도구 활용: 사용자의 질문을 분석하여 내부 DB 검색(`search_guideline_hybrid`)을 우선 수행하십시오.
    2. 객관성과 사실 기반: 비유, 과장, 아첨, 불필요한 추임새를 엄격히 금지합니다. 규제 조항과 과학적 사실에 근거하여 담백하고 매우 전문적인 수준으로 답변하십시오.
    3. 정보 출처의 엄격한 분리 및 보완:
       - **[DB 참조]**: `search_guideline_hybrid`를 통해 도출된 내부 데이터베이스 정보.
       - **[웹/외부 참조]**: 내부 DB에 정보가 부족하거나 최신 동향 파악이 필요한 경우, 통합된 'Google 웹 검색 도구'를 활용하여 정보를 보완하십시오.
       - 답변 작성 시 위 두 출처를 마크다운 태그를 활용해 시각적으로 명확히 분리하십시오. 출처 충돌 시 내부 DB 문서를 우선 기준으로 삼고, 웹 검색 결과는 실무적 보충 설명으로만 활용하십시오.
    4. 자동 출처 표기 (Citation): DB 정보를 인용할 때, 제공된 검색 결과의 메타데이터(문서명, 섹션, 조항)를 추출하여 문장 끝에 명시하십시오. (예: [문서명, 제X조 제Y항])
    5. 맥락 파악 및 실무적 통찰: 단순 번역이나 요약을 넘어, 조항 간의 모순점, 실제 바이오시밀러 개발/승인 시 마주치는 Risk, 그리고 30년 경력자 수준의 보수적인 권고사항을 객관적으로 분석하십시오.
    6. 언어: 한국어 존댓말을 사용하며, 표와 글머리 기호를 활용하여 구조적으로 서술하십시오.
    """

    try:
        # Google 웹 검색 도구를 설정하여 DB 검색 도구와 동시 활용 가능하도록 구성
        tools = [
            search_guideline_hybrid, 
            get_recent_documents, 
            types.Tool(google_search=types.GoogleSearch())
        ]

        chat = client.chats.create(
            model=REASONING_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1, # 사실 기반 및 보수적 서술을 위해 온도 최소화
                tools=tools,
            )
        )
        
        response = chat.send_message(user_query)
        
        unique_sources = []
        seen_urls = set()
        for source in accessed_sources:
            if source['url'] not in seen_urls:
                seen_urls.add(source['url'])
                unique_sources.append(source)

        return response.text, unique_sources

    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {str(e)}", []


def compare_multiple_documents(docs_info, user_query: str = "위 문서들을 종합적이고 심층적으로 비교 분석하십시오."):
    """
    선택된 다수의 가이드라인 문서를 객관적으로 대조하고 웹 검색을 통해 심층 분석을 보완합니다.
    """
    try:
        docs_text = ""
        doc_count = len(docs_info)

        chars_per_doc = (MAX_TOTAL_CHARS - 10000) // doc_count if doc_count > 0 else MAX_TOTAL_CHARS

        for i, doc in enumerate(docs_info):
            chunk_res = supabase.table("document_chunks").select("content", "section", "clause").eq("url", doc['url']).order("chunk_index").execute()
            
            doc_chunks = []
            for c in chunk_res.data:
                # Semantic Chunking을 고려한 메타데이터 포함
                meta = f"[섹션: {c.get('section', 'N/A')}, 조항: {c.get('clause', 'N/A')}] "
                doc_chunks.append(meta + c['content'])

            full_text = " ".join(doc_chunks)
            truncated_text = full_text[:chars_per_doc]
            docs_text += f"\n\n--- [문서 {i+1}: {doc.get('title', 'Unknown')} ({doc.get('agency', 'N/A')})] ---\n{truncated_text}" 

        system_instruction = """
        당신은 글로벌 바이오시밀러 제약사에서 30년 이상의 경력을 쌓은 최고 수준의 인허가(RA) 전문 컨설턴트입니다.

        [다중 문서 비교 분석 원칙]
        1. 객관성 및 사실 기반: 비유, 과장, 아첨을 엄격히 금지합니다. 객관적이고 사실적인 대조를 우선시하십시오.
        2. 출처 구분 (DB vs 웹): 제공된 [분석 대상 문서들] 기반의 분석을 'DB 참조'로 명시하고, 규제 변화 배경이나 최신 동향 등 보완 설명이 필요할 경우 통합된 구글 검색 도구를 활용하여 '웹 참조'로 명확히 분리하여 서술하십시오.
        3. 문서 기반 대조 및 출처 자동화: 요구 자료의 수준, 용어 정의, 실무적 기준점 차이를 대조하고, 특정 내용 언급 시 해당 문서의 [섹션/조항] 정보를 반드시 표기하십시오.
        4. 전문적 통찰: 공통점과 차이점 도출에 그치지 않고, 가장 엄격한 기준이 무엇인지, 그리고 그러한 규제 차이가 실무 전략에 미치는 영향을 사실관계에 입각해 서술하십시오.
        5. 형식: 한국어 존댓말을 사용하며, 가독성을 위해 비교 요약표 등 마크다운을 체계적으로 활용하십시오.
        """

        prompt = f"질문/요청: {user_query}\n\n[분석 대상 문서들]\n{docs_text}"

        response = client.models.generate_content(
            model=REASONING_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0, # 문서 대조 시 할루시네이션 방지
                tools=[types.Tool(google_search=types.GoogleSearch())] # 비교 시에도 부족한 맥락을 검색하도록 도구 추가
            )
        )
        return response.text
        
    except Exception as e:
        return f"문서 비교 분석 중 오류가 발생했습니다: {str(e)}"
