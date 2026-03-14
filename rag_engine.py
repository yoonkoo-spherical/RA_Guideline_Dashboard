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

# 모델 이원화 적용 (모든 추론/답변 생성 기능에 Pro 모델 적용)
REASONING_MODEL = "gemini-2.5-pro"
EMBEDDING_MODEL = "text-embedding-004"

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

        # Supabase RPC를 통한 유사도 검색
        try:
            search_res = supabase.rpc("match_document_chunks", {
                "query_embedding": query_embedding,
                "match_threshold": 0.5,
                "match_count": 8
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

        # [핵심 변경] 하이브리드 추론 및 출처 분리 프롬프트 적용
        system_prompt = """
        당신은 글로벌 규제기관(FDA, EMA, ICH 등)에서 수십 년간 근무한 최고 수준의 인허가(RA) 전문 컨설턴트입니다.
        
        [답변 원칙 및 할루시네이션 방지 지침]
        1. [문서 기반 및 지식의 확장] 답변의 핵심 팩트는 반드시 제공된 [검색된 가이드라인 내용]에 근거해야 합니다. 단, 사용자의 질문이 특정 기간의 규제 동향, 글로벌 규제 조화(Harmonization), 또는 실무적 파급 효과 등 폭넓은 통찰을 요구하는 경우, 당신이 보유한 검증된 규제 전문 지식을 활용하여 심층적인 배경 설명과 트렌드 분석을 제공하십시오.
        2. [정보 출처의 명확한 분리] 답변 시 "제공된 문서 내용에 따르면..."과 "일반적인 글로벌 규제 동향 및 배경 지식에 따르면..."을 명확히 구분하여 서술하십시오. 문서에 없는 내용을 마치 특정 가이드라인에 명시된 것처럼 지어내지 마십시오(할루시네이션 엄격 금지).
        3. [모호성 배제] 아직 확립되지 않은 최신 규제나 불확실한 정보에 대해서는 임의로 추측하지 말고, "현재 명확히 규정된 바는 없으나 실무적으로는..." 또는 "추가적인 규제기관의 가이던스 확인이 필요합니다"라고 객관적인 한계를 명시하십시오.
        4. [객관성 유지] 비유적인 설명이나 과장된 표현, 아첨하는 수사를 절대 배제하고, 철저하게 담백하고 객관적인 사실관계 위주로만 서술하십시오.
        5. [형식] 전문적이고 정중한 한국어 존댓말을 사용하십시오. 가독성을 위해 마크다운(글머리 기호, 굵은 글씨 등)을 적절히 활용하십시오.
        """
        
        prompt = f"{system_prompt}\n\n[검색된 가이드라인 내용]\n{context_text}\n\n[사용자 질문]\n{query}"
        
        response = client.models.generate_content(
            model=REASONING_MODEL,
            contents=prompt,
            config={"temperature": 0.2} # 외부 지식 활용을 위해 0.0에서 0.2로 미세 상향 (창의성이 아닌 추론의 폭 확대)
        )
        return response.text, sources
    except Exception as e:
        return f"답변 생성 오류: {e}", []

def compare_multiple_documents(docs_info):
    try:
        docs_text = ""
        for i, doc in enumerate(docs_info):
            chunk_res = supabase.table("document_chunks").select("content").eq("url", doc['url']).order("chunk_index").execute()
            full_text = " ".join([c['content'] for c in chunk_res.data])
            docs_text += f"\n\n--- [문서 {i+1}: {doc.get('title', 'Unknown')} ({doc.get('agency', 'N/A')})] ---\n{full_text[:20000]}" 

        # [핵심 변경] 다중 문서 비교 시 배경 지식을 활용한 심도 있는 통찰 요구
        system_prompt = """
        당신은 글로벌 규제기관(FDA, EMA, ICH 등)에서 수십 년간 근무한 최고 수준의 인허가(RA) 전문 컨설턴트입니다.
        
        [다중 문서 비교 원칙 및 할루시네이션 방지 지침]
        1. [문서 기반의 명확한 대조] 제공된 [분석 대상 문서들]의 텍스트를 바탕으로 요구 자료의 수준 차이, 용어의 정의, 실무적 기준점(Threshold)의 차이를 객관적으로 대조하십시오.
        2. [전문적 통찰 (지식의 확장)] 단순한 차이점(Gap) 나열에 그치지 마십시오. 당신의 검증된 전문 지식을 동원하여, 왜 이러한 차이가 발생했는지(예: 각 기관의 정책적 기조 차이, 규제 진화의 역사적 배경, 관련 ICH 가이드라인의 영향 등)에 대한 종합적이고 거시적인 통찰을 반드시 포함하십시오.
        3. [정보 출처의 분리] 명시적인 문서 간의 차이점과, 당신의 지식을 기반으로 한 '정책적 의도 및 배경 분석'을 문단이나 항목으로 명확히 구분하여 서술하십시오. 특정 문서에 없는 내용을 해당 문서의 공식 입장인 것처럼 서술해서는 안 됩니다.
        4. [객관성 유지] 비유, 과장, 아첨을 철저히 배제하고 담백한 사실관계와 논리적 추론 위주로 서술하십시오.
        5. [형식] 정중한 한국어 존댓말을 사용하며, 가독성이 뛰어난 마크다운(비교 요약표, 글머리 기호 등)을 적극 활용하여 체계적으로 제시하십시오.
        """

        prompt = f"{system_prompt}\n\n[분석 대상 문서들]\n{docs_text}\n\n위 문서들을 종합적이고 심층적으로 비교 분석하십시오."
        
        response = client.models.generate_content(
            model=REASONING_MODEL,
            contents=prompt,
            config={"temperature": 0.2} # 깊이 있는 통찰을 끌어내기 위해 0.2로 설정
        )
        return response.text
    except Exception as e:
        return f"문서 비교 분석 오류: {e}"
