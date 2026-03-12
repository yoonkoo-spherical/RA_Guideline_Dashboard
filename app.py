import streamlit as st
import pandas as pd
from supabase import create_client, Client
import rag_engine

@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase: Client = init_connection()

@st.cache_data(ttl=600)
def load_data():
    docs_response = supabase.table("guidelines").select("*").execute()
    df = pd.DataFrame(docs_response.data)
    
    comp_response = supabase.table("version_comparisons").select("*").execute()
    comp_df = pd.DataFrame(comp_response.data)
    
    chunk_response = supabase.table("document_chunks").select("url").execute()
    embedded_urls = set([item['url'] for item in chunk_response.data])
    
    return df, comp_df, embedded_urls

def calculate_progress(df, embedded_urls):
    if df.empty: return 0, 0, 0, 0, 0
    total_docs = len(df)
    summarized_docs = len(df[df['ai_summary'].notna() & (df['ai_summary'] != '') & (~df['ai_summary'].str.contains('텍스트 추출 불가', na=False))])
    embedded_docs = len([url for url in df['url'] if url in embedded_urls])
    summary_pct = int((summarized_docs / total_docs) * 100) if total_docs > 0 else 0
    embed_pct = int((embedded_docs / total_docs) * 100) if total_docs > 0 else 0
    return total_docs, summarized_docs, summary_pct, embedded_docs, embed_pct

def main():
    st.set_page_config(page_title="RA 가이드라인 대시보드", layout="wide")
    st.title("FDA & EMA 가이드라인 통합 검색 및 분석")
    
    df, comp_df, embedded_urls = load_data()
    
    if df.empty:
        st.warning("데이터베이스에 데이터가 없습니다.")
        return

    st.sidebar.header("📊 데이터 처리 현황")
    total, sum_cnt, sum_pct, emb_cnt, emb_pct = calculate_progress(df, embedded_urls)
    st.sidebar.metric("전체 수집 문서", f"{total} 건")
    st.sidebar.progress(sum_pct / 100, text=f"AI 요약: {sum_pct}% ({sum_cnt}/{total})")
    st.sidebar.progress(emb_pct / 100, text=f"AI 임베딩: {emb_pct}% ({emb_cnt}/{total})")
    st.sidebar.divider()
    
    st.sidebar.header("🔍 필터 옵션")
    agencies = df['agency'].dropna().unique().tolist()
    selected_agencies = st.sidebar.multiselect("규제기관 (Agency)", options=agencies, default=agencies)
    categories = df['category'].dropna().unique().tolist()
    selected_categories = st.sidebar.multiselect("키워드/카테고리", options=categories, default=categories)

    tab1, tab2, tab3 = st.tabs(["📄 가이드라인 문서 검색", "💬 AI Q&A (RAG)", "⚖️ 다중 문서 수동 비교"])

    filtered_df = df[(df['agency'].isin(selected_agencies)) & (df['category'].isin(selected_categories))]

    def check_summary(text):
        if pd.isna(text) or str(text).strip() == "": return False
        if "텍스트 추출 불가" in str(text): return False
        return True
        
    filtered_df['has_summary'] = filtered_df['ai_summary'].apply(check_summary)
    filtered_df['has_embedding'] = filtered_df['url'].isin(embedded_urls)
    
    def get_status_score(row):
        if row['has_summary'] and row['has_embedding']: return 4
        if row['has_summary'] and not row['has_embedding']: return 3
        if not row['has_summary'] and row['has_embedding']: return 2
        return 1
        
    filtered_df['status_score'] = filtered_df.apply(get_status_score, axis=1)

    # --- TAB 1: 가이드라인 검색 ---
    with tab1:
        search_query = st.text_input("가이드라인 제목 검색", "")
        tab1_df = filtered_df.copy()
        if search_query:
            tab1_df = tab1_df[tab1_df['title'].str.contains(search_query, case=False, na=False)]
        tab1_df = tab1_df.sort_values(by=['status_score', 'title'], ascending=[False, True])
        st.subheader(f"검색 결과: {len(tab1_df)} 건")
        
        for index, row in tab1_df.iterrows():
            if row['status_score'] == 4: status_icon = "🟢 [완료]"
            elif row['status_score'] == 3: status_icon = "🟡 [요약 완료]"
            elif row['status_score'] == 2: status_icon = "🟡 [임베딩 완료]"
            else: status_icon = "⏳ [대기중]"

            with st.expander(f"{status_icon} {row['title']} ({row['agency']})"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**기관:** {row['agency']} | **식별자:** {row.get('ref_number', 'N/A')} | **분류:** {row['category']}")
                    st.markdown(f"[🔗 원본 문서 열기]({row['url']})")
                
                st.divider()
                st.markdown("#### 💡 AI 핵심 요약")
                if row['has_summary']: st.write(row['ai_summary'])
                else: st.info("AI 요약 대기 중이거나 추출 불가 문서입니다.")
                
                if not row['has_embedding']:
                    st.warning("⚠️ RAG 검색용 벡터 DB에 임베딩되지 않은 문서입니다.")
                
                if not comp_df.empty:
                    doc_comparisons = comp_df[comp_df['new_url'] == row['url']]
                    if not doc_comparisons.empty:
                        st.divider()
                        st.markdown("#### 🔄 구버전 대비 변경점 분석")
                        for _, comp_row in doc_comparisons.iterrows():
                            st.write(comp_row['comparison_text'])

    # --- TAB 2: RAG 기반 Q&A 채팅 (Step 9: 출처 확인 기능 추가) ---
    with tab2:
        st.markdown("#### 규제 가이드라인 AI 어시스턴트")
        if "messages" not in st.session_state: st.session_state.messages = []
        
        # 채팅 메시지 렌더링
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): 
                st.markdown(message["content"])
                # AI 답변일 경우 출처(소스) 청크를 토글 형태로 렌더링
                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander("🔍 AI가 참고한 원문 조각(Chunks) 확인"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**[{i+1}] 출처 링크:** [{source['url']}]({source['url']})")
                            st.info(source['content'])

        if prompt := st.chat_input("규제 가이드라인에 대해 질문해보세요."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("답변을 생성 중입니다..."):
                    # 텍스트와 원문 조각 리스트를 분리해서 받음
                    response_text, sources = rag_engine.ask_guideline(prompt)
                    st.markdown(response_text)
                    
                    if sources:
                        with st.expander("🔍 AI가 참고한 원문 조각(Chunks) 확인"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**[{i+1}] 출처 링크:** [{source['url']}]({source['url']})")
                                st.info(source['content'])
            
            # session_state에 답변과 출처 데이터를 함께 저장
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "sources": sources
            })

    # --- TAB 3: 다중 문서 수동 비교 ---
    with tab3:
        st.markdown("#### ⚖️ 다중 문서 수동 비교 분석")
        embedded_only_df = filtered_df[filtered_df['has_embedding'] == True].copy()
        
        if embedded_only_df.empty:
            st.info("현재 임베딩이 완료된 문서가 없어 비교 기능을 사용할 수 없습니다.")
        else:
            df_for_selection = embedded_only_df[['title', 'agency', 'category', 'url']].copy()
            df_for_selection.insert(0, "비교 선택", False)
            
            edited_df = st.data_editor(
                df_for_selection,
                hide_index=True,
                column_config={"비교 선택": st.column_config.CheckboxColumn("비교 선택", help="비교할 문서를 선택하세요", default=False), "url": None},
                disabled=["title", "agency", "category"],
                use_container_width=True
            )
            
            selected_rows = edited_df[edited_df["비교 선택"]]
            
            if st.button("선택한 문서 비교 분석 실행", type="primary"):
                if len(selected_rows) < 2:
                    st.warning("비교를 수행하려면 문서를 2개 이상 선택해야 합니다.")
                else:
                    selected_docs_info = selected_rows.to_dict('records')
                    st.info(f"총 {len(selected_docs_info)}개의 문서를 비교 분석합니다.")
                    with st.spinner("선택된 문서들의 텍스트를 대조하고 있습니다..."):
                        comparison_result = rag_engine.compare_multiple_documents(selected_docs_info)
                    st.divider()
                    st.markdown("#### 📊 분석 결과")
                    st.markdown(comparison_result)

if __name__ == "__main__":
    main()
