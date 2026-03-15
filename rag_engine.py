import os
from google import genai
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

# 최고 성능 모델로 업그레이드 (다중 문서 처리 및 고도화된 추론용)
REASONING_MODEL = "gemini-3.1-pro-preview"
EMBEDDING_MODEL = "gemini-embedding-001"

# 비용 최적화를 위한 토큰 관리 상수 (128k 토큰 초과 시 비용 2배 증가 방지용)
# 보통 1 토큰 = 3~4 글자(영문 기준), 한글은 효율이 다르나 보수적으로 120,000 토큰 = 약 250,000자로 산정
MAX_TOTAL_CHARS = 250000 

def get_embedding(text):
    try:
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
        return response.embeddings[0].values
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def ask_guideline(query):
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:
            return "답변 생성 오류: 쿼리 임베딩 생성에 실패했습니다.", []

        # 검색 청크 수 확대 (8 -> 15): 더 넓은 문맥 확보로 답변 정합성 개선
        try:
            search_res = supabase.rpc("match_document_chunks", {
                "query_embedding": query_embedding,
                "match_threshold": 0.5,
                "match_count": 15
            }).execute()
            docs = search_res.data
        except Exception as rpc_e:
            print(f"RPC Search Error: {rpc_e}")
            docs = []
        
        if not docs:
            context_text = "제공된 데이터베이스 내에서 관련 가이드라인 내용을 찾을 수 없습니다."
            sources = []
        else:
            context_text = "\n\n".join([f"[{i+1}] {d.get('content', '')}" for i, d in enumerate(docs)])
            unique_urls = list({d['url']: d for d in docs}.values())
            sources = [{"url": d['url']} for d in unique_urls]

        system_prompt = """
        당신은 글로벌 규제기관(FDA, EMA, ICH 등)에서 수십 년간 근무한 최고 수준의 인허가(RA) 전문 컨설턴트입니다.
        
        [답변 원칙 및 엄격한 지침]
        1. [객관성 및 사실 기반] 비유적인 설명을 절대 사용하지 말고, 객관적이고 사실적인 설명을 우선시하십시오. 과장된 추임새나 아첨하는 표현을 철저히 배제하고 담백하게 사실관계 위주로만 답변하십시오.
        2. [정보 출처의 명확한 분리] 답변의 핵심 팩트는 반드시 제공된 [검색된 가이드라인 내용]에 근거해야 합니다. 문서에 없는 내용을 명시된 것처럼 지어내지 마십시오(할루시네이션 엄격 금지).
        3. [지식의 확장] 사용자의 질문이 폭넓은 통찰을 요구하는 경우, "일반적인 글로벌 규제 동향에 따르면..."과 같이 출처를 구분하여 당신의 전문 지식을 활용해 심층적인 배경을 설명하십시오.
        4. [모호성 배제] 아직 확립되지 않은 규제는 임의로 추측하지 말고 한계를 명확히 밝히십시오.
        5. [언어 및 형식] 정중한 한국어 존댓말을 사용하며, 가독성을 위해 마크다운을 적절히 활용하십시오.
        """
        
        prompt = f"{system_prompt}\n\n[검색된 가이드라인 내용]\n{context_text}\n\n[사용자 질문]\n{query}"
        
        response = client.models.generate_content(
            model=REASONING_MODEL,
            contents=prompt,
            config={"temperature": 0.1} # 사실관계 위주 답변을 위해 온도 하향 조정
        )
        return response.text, sources
    except Exception as e:
        return f"답변 생성 오류: {e}", []

def compare_multiple_documents(docs_info):
    try:
        docs_text = ""
        doc_count = len(docs_info)
        
        # 선택된 문서 수에 따라 각 문서에 할당할 최대 글자 수를 동적으로 계산 (비용 128k Threshold 초과 방지)
        # N개의 문서일 경우, 시스템 프롬프트 여유분을 빼고 N등분
        chars_per_doc = (MAX_TOTAL_CHARS - 10000) // doc_count if doc_count > 0 else MAX_TOTAL_CHARS

        for i, doc in enumerate(docs_info):
            chunk_res = supabase.table("document_chunks").select("content").eq("url", doc['url']).order("chunk_index").execute()
            full_text = " ".join([c['content'] for c in chunk_res.data])
            
            # 할당된 동적 제한치에 맞춰 텍스트 절사
            truncated_text = full_text[:chars_per_doc]
            docs_text += f"\n\n--- [문서 {i+1}: {doc.get('title', 'Unknown')} ({doc.get('agency', 'N/A')})] ---\n{truncated_text}" 

        system_prompt = """
        당신은 글로벌 규제기관(FDA, EMA, ICH 등)에서 수십 년간 근무한 최고 수준의 인허가(RA) 전문 컨설턴트입니다.
        
        [다중 문서 비교 원칙 및 엄격한 지침]
        1. [객관성 및 사실 기반] 비유적인 설명을 절대 사용하지 말고, 객관적이고 사실적인 설명을 우선시하십시오. 과장된 추임새나 아첨하는 표현을 철저히 배제하고 담백하게 사실관계 위주로만 서술하십시오.
        2. [문서 기반의 명확한 대조] 제공된 [분석 대상 문서들]의 텍스트를 바탕으로 요구 자료의 수준 차이, 용어의 정의, 실무적 기준점 차이를 객관적으로 대조하십시오.
        3. [전문적 통찰] 단순한 차이점 나열에 그치지 않고, 왜 이러한 차이가 발생했는지(예: 정책적 기조, 규제 진화 배경 등)에 대한 종합적인 통찰을 포함하십시오.
        4. [정보 출처 분리] 문서 간의 명시적 차이점과 당신의 지식 기반 배경 분석을 명확히 구분하여 서술하십시오.
        5. [언어 및 형식] 정중한 한국어 존댓말을 사용하며, 비교 요약표 등 마크다운을 적극 활용하여 체계적으로 제시하십시오.
        """

        prompt = f"{system_prompt}\n\n[분석 대상 문서들]\n{docs_text}\n\n위 문서들을 종합적이고 심층적으로 비교 분석하십시오."
        
        response = client.models.generate_content(
            model=REASONING_MODEL,
            contents=prompt,
            config={"temperature": 0.1} # 사실관계 기반 대조를 위해 온도 하향 조정
        )
        return response.text
    except Exception as e:
        return f"문서 비교 분석 오류: {e}"
