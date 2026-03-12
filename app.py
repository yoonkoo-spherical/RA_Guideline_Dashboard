import streamlit as st
import pandas as pd
from supabase import create_client, Client
import rag_engine  # 작성된 RAG 엔진 모듈 임포트

# --- 1. 데이터베이스 연결 및 데이터 로드 ---
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase: Client = init_connection()

@st.cache_data(ttl=600)  # 10분 단위 캐시 갱신
def load_data():
    # 메인 가이드라인 데이터 로드
    docs_response = supabase.table("guidelines").select("*").execute()
    df = pd.DataFrame(docs_response.data)
    
    # 버전 비교 데이터 로드
    comp_response = supabase.table("version_comparisons").select("*").execute()
    comp_df = pd.DataFrame(comp_response.data)
    
    # 임베딩 완료된 URL 목록 로드
    chunk_response = supabase.table("document_chunks").select("url").execute()
    embedded_urls = set([item['url'] for item in chunk_response.data])
    
    return df, comp_df, embedded_urls

# --- 2. 진행률 계산 함수 ---
def calculate_progress(df, embedded_urls):
    if df.empty:
        return 0, 0, 0, 0, 0
        
    total_docs = len(df)
    
    # 요약 완료 건수 (에러 메시지 제외)
    summarized_docs = len(df[
        df['ai_summary'].notna() & 
        (df['ai_summary'] != '') & 
        (~df['ai_summary'].str.contains('텍스트 추출 불가', na=False))
    ])
    
    # 임베딩 완료 건수
    embedded_docs = len([url for url in df['url'] if url in embedded_urls])
    
    summary_pct = int((summarized_docs / total_docs) * 100) if total_docs > 0 else 0
    embed_pct = int((embedded_docs / total_docs) * 100) if total_docs > 0 else 0
    
    return total_docs, summarized_docs, summary_pct, embedded_docs, embed_pct

# --- 3. 메인 애플리케이션 ---
def main():
    st.set_page_config(page_title="RA 가이드라인 대시보드", layout="wide")
    st.title("FDA & EMA 가이드라인 통합 검색 및 분석")
    
    df, comp_df, embedded_urls = load_data()
    
    if df.empty:
        st.warning("데이터베이스에 데이터가 없습니다.")
        return

    # 사이드바: 진행률 표시 및 필터
    st.sidebar.header("📊 데이터 처리 현황")
    total, sum_cnt, sum_pct, emb_cnt, emb_pct = calculate_progress(df, embedded_urls)
    
    st.sidebar.metric("전체 수집 문서", f"{total} 건")
    st.sidebar.progress(sum_pct / 100, text=f"AI 요약 진행률: {sum_pct}% ({sum_cnt}/{total})")
    st.sidebar.progress(emb_pct / 100, text=f"AI 임베딩 진행률: {emb_pct}% ({emb_cnt}/{total})")
    
    st.sidebar.divider()
    
    st.sidebar.header("🔍 필터 옵션")
    agencies = df['agency'].dropna().unique().tolist()
    selected_agencies = st.sidebar.multiselect("규제기관 (Agency)", options=agencies, default=agencies)
    
    categories = df['category'].dropna().unique().tolist()
    selected_categories = st.sidebar.multiselect("키워드/카테고리", options=categories, default=categories)

    # 탭 구성: 문서 검색 탭 / RAG 채팅 탭
    tab1, tab2 = st.tabs(["📄 가이드라인 문서 검색", "💬 AI 가이드라인 Q&A (RAG)"])

    # --- TAB 1: 가이드라인 검색 및 비교 결과 출력 (Step 6) ---
    with tab1:
        search_query = st.text_input("가이드라인 제목 검색", "")
        
        filtered_df = df[
            (df['agency'].isin(selected_agencies)) &
            (df['category'].isin(selected_categories))
        ]
        
        if search_query:
            filtered_df = filtered_df[filtered_df['title'].str.contains(search_query, case=False, na=False)]
        
        # 정렬 로직 (요약문 있는 문서 우선)
        def has_valid_summary(text):
            if pd.isna(text) or str(text).strip() == "": return False
            if "텍스트 추출 불가" in str(text): return False
            return True
            
        filtered_df['has_summary'] = filtered_df['ai_summary'].apply(has_valid_summary)
        filtered_df = filtered_df.sort_values(by=['has_summary', 'title'], ascending=[False, True])

        st.subheader(f"검색 결과: {len(filtered_df)} 건")
        
        for index, row in filtered_df.iterrows():
            with st.expander(f"{'✅' if row['has_summary'] else '⏳'} {row['title']} ({row['agency']})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    ref_text = row.get('ref_number', 'N/A')
                    st.write(f"**발행기관:** {row['agency']} | **식별자:** {ref_text} | **상태:** {row['status']} | **분류:** {row['category']} | **발행일:** {row['published_date']}")
                    st.markdown(f"[🔗 원본 가이드라인 문서 열기]({row['url']})")
                    
                st.divider()
                
                # AI 핵심 요약 출력
                st.markdown("#### 💡 AI 핵심 요약")
                if row['has_summary']:
                    st.write(row['ai_summary'])
                else:
                    st.info("현재 AI 요약이 대기 중이거나 원문 텍스트 추출이 불가능한 문서입니다.")
                
                # Step 6: 버전 비교 리포트 출력 로직
                if not comp_df.empty:
                    # 현재 문서(new_url)에 대한 비교 데이터가 있는지 확인
                    doc_comparisons = comp_df[comp_df['new_url'] == row['url']]
                    if not doc_comparisons.empty:
                        st.divider()
                        st.markdown("#### 🔄 구버전 대비 변경점 분석 (Version History)")
                        for _, comp_row in doc_comparisons.iterrows():
                            st.info(f"**비교 대상 구버전 URL:** {comp_row['old_url']}")
                            st.write(comp_row['comparison_text'])

    # --- TAB 2: RAG 기반 Q&A 채팅 (Step 5) ---
    with tab2:
        st.markdown("#### 규제 가이드라인 AI 어시스턴트")
        st.write("임베딩이 완료된 문서를 바탕으로 질문에 답변합니다. (현재 벡터 데이터베이스 기반 검색 적용)")
        
        # 채팅 기록 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 이전 채팅 기록 출력
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 사용자 입력 처리
        if prompt := st.chat_input("규제 가이드라인에 대해 질문해보세요. (예: 바이오시밀러 임상 1상 면역원성 기준은?)"):
            # 사용자 질문 화면 출력 및 상태 저장
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # AI 답변 생성 및 화면 출력
            with st.chat_message("assistant"):
                with st.spinner("관련 가이드라인을 검색하고 답변을 생성 중입니다..."):
                    # rag_engine 모듈의 파이프라인 호출
                    response = rag_engine.ask_guideline(prompt)
                    st.markdown(response)
            
            # AI 답변 상태 저장
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
