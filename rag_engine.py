import os
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
EMBEDDING_MODEL = "gemini-embedding-001"

# 비용 최적화를 위한 토큰 관리 상수 (다중 문서 비교 시 동적 절사 용도)
MAX_TOTAL_CHARS = 250000 

def get_embedding(text: str):
    """텍스트를 벡터 임베딩으로 변환합니다."""
    try:
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
        return response.embeddings[0].values
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def ask_guideline(user_query: str):
    """
    사용자의 질문 의도를 파악하여 적절한 검색 도구를 선택하고, 
    결과 및 전문 지식을 종합하여 답변을 생성하는 지능형 에이전트 함수입니다.
    """
    # 에이전트가 도구를 사용할 때마다 참조한 문서의 URL을 수집하는 리스트
    accessed_sources = []

    # 도구 1: 의미 기반 벡터 검색
    def search_guideline_by_keyword(query: str) -> str:
        """
        가이드라인 문서의 구체적인 내용, 규제 기준, 실무 지침 등을 의미(Semantic) 기반으로 검색합니다.
        사용자가 특정 규제 요건, 가이드라인의 세부 내용, 기준의 차이점 등을 물어볼 때 사용합니다.
        """
        query_embedding = get_embedding(query)
        if not query_embedding:
            return "시스템 오류: 쿼리 임베딩 생성에 실패했습니다."

        try:
            # 매치 수를 15로 유지하여 충분한 문맥 확보
            search_res = supabase.rpc("match_document_chunks", {
                "query_embedding": query_embedding,
                "match_threshold": 0.5,
                "match_count": 15
            }).execute()
            docs = search_res.data
        except Exception as rpc_e:
            print(f"RPC Search Error: {rpc_e}")
            return "데이터베이스 검색 중 오류가 발생했습니다."

        if not docs:
            return "제공된 데이터베이스 내에서 해당 키워드와 일치하는 문서 내용을 찾을 수 없습니다."

        context_chunks = []
        for d in docs:
            accessed_sources.append({"url": d['url']})
            context_chunks.append(f"- 내용: {d.get('content', '')}")
            
        return "\n\n".join(context_chunks)

    # 도구 2: 메타데이터 기반 최신 문서 조회
    def get_recent_documents(limit: int = 5) -> str:
        """
        가장 최근에 데이터베이스에 추가되거나 업데이트된 가이드라인 문서 목록을 날짜순으로 조회합니다.
        사용자가 '최근 문서', '최신 가이드라인', '업데이트 내역' 등 시간/현황과 관련된 질문을 할 때 사용합니다.
        """
        try:
            res = supabase.table("guidelines").select("title, agency, category, url, created_at").order("created_at", desc=True).limit(limit).execute()
            docs = res.data
        except Exception as e:
            print(f"SQL Select Error: {e}")
            return "최근 문서 목록을 조회하는 중 오류가 발생했습니다."

        if not docs:
            return "데이터베이스에 등록된 문서가 없습니다."

        context_chunks = []
        for d in docs:
            accessed_sources.append({"url": d['url']})
            context_chunks.append(f"- 기관: {d['agency']}, 제목: {d['title']}, 분류: {d['category']}, DB추가일: {d['created_at'][:10]}")
            
        return "\n".join(context_chunks)

    # 챗봇 에이전트 시스템 프롬프트 설정
    system_instruction = """
    당신은 글로벌 규제기관(FDA, EMA, ICH 등)의 수십년 경력의 최고 수준 인허가(RA) 전문 컨설턴트입니다.

    [작업 원칙]
    1. 상황 판단 및 도구 활용: 사용자의 질문 의도를 분석하여 스스로 판단하고 도구를 호출하십시오.
       - 세부 규제 내용, 기준, 차이점 문의 -> `search_guideline_by_keyword` 호출
       - 시스템에 등록된 최신 문서, 전체 목록 등 현황 문의 -> `get_recent_documents` 호출
    2. 사실관계 위주 작성: 비유적인 설명을 사용하지 말고, 객관적이고 사실적인 설명을 우선시하십시오. 아첨하는 표현이나 과장된 추임새를 엄격히 금지합니다.
    3. 지식의 보완 (Knowledge Bridging): 도구 검색 결과에 명시된 내용이 없거나 부족한 경우, "제공된 데이터베이스에는 해당 내용이 명시되어 있지 않으나, 일반적인 글로벌 규제 동향에 따르면..." 이라고 한계를 객관적으로 밝힌 후, 귀하의 최고 수준 RA 전문 지식을 활용하여 사실에 근거한 상세한 답변을 제공하십시오.
    4. 형식: 한국어 존댓말을 사용하며, 정보의 체계적 전달을 위해 마크다운(글머리 기호, 표 등)을 적절히 활용하십시오.
    """

    try:
        # Function Calling을 지원하는 챗 세션 생성
        chat = client.chats.create(
            model=REASONING_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3, # 사실 기반 서술을 위해 낮은 온도 설정
                tools=[search_guideline_by_keyword, get_recent_documents],
            )
        )
        
        # 모델이 질문을 분석하고, 필요한 도구를 자동 실행한 뒤 최종 답변을 생성
        response = chat.send_message(user_query)
        
        # 에이전트가 도구 실행 중 누적한 URL 리스트의 중복 제거
        unique_sources = []
        seen_urls = set()
        for source in accessed_sources:
            if source['url'] not in seen_urls:
                seen_urls.add(source['url'])
                unique_sources.append(source)

        return response.text, unique_sources

    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {str(e)}", []


def compare_multiple_documents(docs_info):
    """
    선택된 다수의 가이드라인 문서를 객관적으로 대조하고 심층 분석합니다.
    """
    try:
        docs_text = ""
        doc_count = len(docs_info)

        # 문서 개수에 따라 동적으로 할당량을 분할하여 토큰 한도 초과 방지
        chars_per_doc = (MAX_TOTAL_CHARS - 10000) // doc_count if doc_count > 0 else MAX_TOTAL_CHARS

        for i, doc in enumerate(docs_info):
            chunk_res = supabase.table("document_chunks").select("content").eq("url", doc['url']).order("chunk_index").execute()
            full_text = " ".join([c['content'] for c in chunk_res.data])

            truncated_text = full_text[:chars_per_doc]
            docs_text += f"\n\n--- [문서 {i+1}: {doc.get('title', 'Unknown')} ({doc.get('agency', 'N/A')})] ---\n{truncated_text}" 

        # 다중 문서 비교 분석 시스템 프롬프트 설정
        system_instruction = """
        당신은 글로벌 규제기관(FDA, EMA, ICH 등)의 수십년 경력의 최고 수준 인허가(RA) 전문 컨설턴트입니다.

        [다중 문서 비교 분석 원칙]
        1. 객관성 및 사실 기반: 비유적인 설명을 사용하지 말고, 객관적이고 사실적인 대조를 우선시하십시오. 아첨하는 표현이나 과장된 추임새를 엄격히 금지합니다.
        2. 문서 기반 대조: 제공된 [분석 대상 문서들]의 텍스트를 바탕으로 요구 자료의 수준, 용어 정의, 실무적 기준점 차이를 명확하고 담백하게 대조하십시오.
        3. 전문적 통찰: 단순한 차이점 나열에 그치지 않고, 정책적 기조나 규제 진화 배경 등 그러한 차이가 발생한 원인에 대한 전문가적 분석을 사실관계에 근거하여 서술하십시오.
        4. 형식: 한국어 존댓말을 사용하며, 가독성을 위해 비교 요약표 등 마크다운을 체계적으로 활용하십시오.
        """

        prompt = f"[분석 대상 문서들]\n{docs_text}\n\n위 문서들을 종합적이고 심층적으로 비교 분석하십시오."

        response = client.models.generate_content(
            model=REASONING_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1
            )
        )
        return response.text
        
    except Exception as e:
        return f"문서 비교 분석 중 오류가 발생했습니다: {str(e)}"
