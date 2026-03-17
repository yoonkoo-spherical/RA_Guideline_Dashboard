import os
from google import genai
from google.genai import types
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Supabase Client Error in rag_engine: {e}")

client = genai.Client(api_key=GEMINI_API_KEY)

REASONING_MODEL = "gemini-3.1-pro-preview"
EMBEDDING_MODEL = "gemini-embedding-001"

MAX_TOTAL_CHARS = 250000 

def get_embedding(text):
    try:
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
        return response.embeddings[0].values
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def ask_guideline(query):
    # 각 도구에서 접근된 URL 출처를 저장하기 위한 리스트
    accessed_sources = []

    # 도구 1: 내용 기반 벡터 검색
    def search_guideline_by_keyword(search_query: str) -> str:
        """
        가이드라인 문서의 구체적인 내용, 규제 기준, 실무 지침 등을 의미 기반으로 검색합니다.
        """
        query_embedding = get_embedding(search_query)
        if not query_embedding:
            return "시스템 오류: 쿼리 임베딩 생성에 실패했습니다."

        try:
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

    # 도구 2: 날짜 기반 최신 문서 조회
    def get_recent_documents(limit: int = 5) -> str:
        """
        가장 최근에 데이터베이스에 추가되거나 업데이트된 가이드라인 문서 목록을 날짜순으로 조회합니다.
        사용자가 '최근 문서', '최신 가이드라인' 등을 물어볼 때 사용합니다.
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
            context_chunks.append(f"- [{d['agency']}] {d['title']} (분류: {d['category']}, 추가일: {d['created_at'][:10]})")
            
        return "\n".join(context_chunks)

    system_prompt = """
    당신은 글로벌 규제기관(FDA, EMA, ICH 등) 인허가(RA) 전문 컨설턴트입니다.
    
    [답변 원칙 및 엄격한 지침]
    1. [도구 활용 및 메타 인지] 사용자의 질문 의도를 파악하여 제공된 도구를 사용하십시오.
       - 규제의 세부 내용, 기준, 차이점을 물을 때는 `search_guideline_by_keyword` 도구를 호출하십시오.
       - 시스템에 최근 추가된 문서, 최신 업데이트 목록 등을 물을 때는 `get_recent_documents` 도구를 호출하십시오.
    2. [객관성 유지] 비유적인 설명을 사용하지 말고, 객관적이고 사실적인 설명을 우선시하십시오. 감정적인 추임새를 배제하고 담백하게 서술하십시오.
    3. [유연한 지식 확장] 도구 실행 결과, 데이터베이스에서 명확한 답을 찾지 못한 경우 답변을 단순히 거부하지 마십시오. "현재 제공된 데이터베이스에는 명시되어 있지 않으나, 일반적인 글로벌 규제 동향에 따르면..."과 같이 사실관계에 입각한 전문 지식을 활용하여 한계를 밝히고 최대한 답변하십시오.
    4. [형식] 정중한 한국어 존댓말을 사용하며, 정보의 위계 구분을 위해 마크다운(표, 리스트 등)을 적절히 활용하십시오.
    """

    try:
        # Function Calling이 포함된 챗 세션 생성
        chat = client.chats.create(
            model=REASONING_MODEL,
            config=types.GenerateContentConfig(
                tools=[search_guideline_by_keyword, get_recent_documents],
                system_instruction=system_prompt,
                temperature=0.2
            )
        )
        
        # 모델이 도구를 스스로 판단하여 호출하고, 그 결과를 바탕으로 최종 답변을 생성함
        response = chat.send_message(query)
        
        # 도구 실행 중 누적된 출처 URL 중복 제거
        unique_sources = []
        seen_urls = set()
        for source in accessed_sources:
            if source['url'] not in seen_urls:
                seen_urls.add(source['url'])
                unique_sources.append(source)

        return response.text, unique_sources

    except Exception as e:
        return f"답변 생성 오류: {e}", []

def compare_multiple_documents(docs_info):
    try:
        docs_text = ""
        doc_count = len(docs_info)

        chars_per_doc = (MAX_TOTAL_CHARS - 10000) // doc_count if doc_count > 0 else MAX_TOTAL_CHARS

        for i, doc in enumerate(docs_info):
            chunk_res = supabase.table("document_chunks").select("content").eq("url", doc['url']).order("chunk_index").execute()
            full_text = " ".join([c['content'] for c in chunk_res.data])

            truncated_text = full_text[:chars_per_doc]
            docs_text += f"\n\n--- [문서 {i+1}: {doc.get('title', 'Unknown')} ({doc.get('agency', 'N/A')})] ---\n{truncated_text}" 

        system_prompt = """
        당신은 글로벌 규제기관(FDA, EMA, ICH 등) 인허가(RA) 전문 컨설턴트입니다.
        
        [다중 문서 비교 원칙 및 엄격한 지침]
        1. [객관성 및 사실 기반] 비유적인 설명을 사용하지 말고, 객관적이고 사실적인 설명을 우선시하십시오. 감정적인 추임새를 배제하고 담백하게 서술하십시오.
        2. [문서 기반의 명확한 대조] 제공된 [분석 대상 문서들]의 텍스트를 바탕으로 요구 자료의 수준 차이, 용어의 정의, 실무적 기준점 차이를 객관적으로 대조하십시오.
        3. [전문적 통찰] 차이점 나열에 그치지 않고, 왜 이러한 차이가 발생했는지(예: 정책적 기조, 규제 진화 배경 등)에 대한 종합적인 통찰을 포함하십시오.
        4. [형식] 정중한 한국어 존댓말을 사용하며, 비교 요약표 등 마크다운을 적극 활용하여 체계적으로 제시하십시오.
        """

        prompt = f"{system_prompt}\n\n[분석 대상 문서들]\n{docs_text}\n\n위 문서들을 종합적이고 심층적으로 비교 분석하십시오."

        response = client.models.generate_content(
            model=REASONING_MODEL,
            contents=prompt,
            config={"temperature": 0.1}
        )
        return response.text
    except Exception as e:
        return f"문서 비교 분석 오류: {e}"
