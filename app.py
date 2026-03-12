import streamlit as st
import pandas as pd
from supabase import create_client, Client
import rag_engine
import json
import markdown

@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

try:
    supabase: Client = init_connection()
except Exception:
    st.error("데이터베이스 연결 설정에 문제가 발생했습니다. 관리자에게 문의하십시오.")
    st.stop()

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

def save_chat_to_db(role, content):
    try:
        supabase.table("chat_history").insert({"role": role, "content": content}).execute()
    except Exception:
        pass # 이력 저장 실패 시 앱을 중단하지 않고 무시함

def save_analysis_to_db(docs_info, result):
    try:
        supabase.table("analysis_history").insert({"docs_info": json.dumps(docs_info, ensure_ascii=False), "comparison_result": result}).execute()
    except Exception:
        pass

def delete_analysis_record(record_id):
    try:
        supabase.table("analysis_history").delete().eq("id", record_id).execute()
    except Exception:
        st.error("기록 삭제 중 오류가 발생했습니다.")

def main():
    st.set_page_config(page_title="RA 가이드라인 대시보드", layout="wide")
    st.title("FDA & EMA 가이드라인 통합 검색 및 분석")
    
    # 1. 데이터베이스 로드 예외 처리
    try:
        df, comp_df, embedded_urls = load_data()
    except Exception:
        st.error("데이터베이스에서 가이드라인 정보를 불러오는 데 실패했습니다. 네트워크 상태를 확인하거나 잠시 후 다시 시도해 주십시오.")
        return
    
    if df.empty:
        st.warning("데이터베이스에 수집된 가이드라인 데이터가 없습니다.")
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 문서 검색", "💬 RAG Q&A", "⚖️ 다중 문서 비교", "🔄 신/구버전 비교", "🗂️ 사용 이력 및 다운로드"
    ])

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
                if not row['has_embedding']: st.warning("⚠️ RAG 검색용 벡터 DB에 임베딩되지 않은 문서입니다.")

    # --- TAB 2: RAG Q&A 채팅 ---
    with tab2:
        st.markdown("#### 규제 가이드라인 AI 어시스턴트 (gemini-2.5-flash)")
        if "messages" not in st.session_state: st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): 
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander("🔍 AI가 참고한 원문 조각 확인"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**[{i+1}] 출처:** [{source['url']}]({source['url']})")

        if prompt := st.chat_input("질문을 입력하세요."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            save_chat_to_db("user", prompt)
            
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    # 2. AI 질의응답 예외 처리
                    try:
                        response_text, sources = rag_engine.ask_guideline(prompt)
                        
                        # rag_engine 내부에서 에러 텍스트를 반환한 경우에 대한 처리
                        if "답변 생성 오류" in response_text or "질문 분석 실패" in response_text:
                            st.error("AI가 답변을 생성하지 못했습니다. (API 할당량 초과 또는 네트워크 오류)")
                        else:
                            st.markdown(response_text)
                            if sources:
                                with st.expander("🔍 AI가 참고한 원문 조각 확인"):
                                    for i, source in enumerate(sources):
                                        st.markdown(f"**[{i+1}] 출처:** [{source['url']}]({source['url']})")
                            
                            st.session_state.messages.append({"role": "assistant", "content": response_text, "sources": sources})
                            save_chat_to_db("assistant", response_text)
                    except Exception:
                        st.error("예기치 않은 시스템 오류가 발생했습니다. 잠시 후 다시 시도해 주십시오.")

    # --- TAB 3: 다중 문서 수동 비교 ---
    with tab3:
        st.markdown("#### ⚖️ 다중 문서 수동 비교 분석")
        embedded_only_df = filtered_df[filtered_df['has_embedding'] == True].copy()
        
        if embedded_only_df.empty:
            st.info("임베딩이 완료된 문서가 없습니다.")
        else:
            df_for_selection = embedded_only_df[['title', 'agency', 'category', 'url']].copy()
            df_for_selection.insert(0, "비교 선택", False)
            edited_df = st.data_editor(
                df_for_selection, hide_index=True,
                column_config={"비교 선택": st.column_config.CheckboxColumn("비교 선택", default=False), "url": None},
                disabled=["title", "agency", "category"], use_container_width=True
            )
            
            selected_rows = edited_df[edited_df["비교 선택"]]
            if st.button("비교 분석 실행", type="primary"):
                if len(selected_rows) < 2: 
                    st.warning("문서를 2개 이상 선택해야 합니다.")
                else:
                    selected_docs_info = selected_rows.to_dict('records')
                    with st.spinner("문서 대조 중..."):
                        # 3. 다중 문서 비교 예외 처리
                        try:
                            comparison_result = rag_engine.compare_multiple_documents(selected_docs_info)
                            
                            if "오류" in comparison_result:
                                st.error("문서 비교 분석 중 오류가 발생했습니다. 선택한 문서의 양이 너무 많아 API 토큰 한도를 초과했을 수 있습니다.")
                            else:
                                save_analysis_to_db(selected_docs_info, comparison_result)
                                st.divider()
                                st.markdown("#### 📊 분석 결과")
                                st.markdown(comparison_result)
                        except Exception:
                            st.error("분석 서버와의 통신에 실패했습니다. 잠시 후 다시 시도해 주십시오.")

    # --- TAB 4: 신/구버전 자동 비교 이력 ---
    with tab4:
        st.markdown("#### 🔄 규제 가이드라인 신/구버전 변경점 자동 비교")
        if comp_df.empty:
            st.info("현재 문서 간의 버전 업데이트(개정) 이력이 감지되지 않았습니다.")
        else:
            for index, row in comp_df.iterrows():
                with st.expander(f"업데이트 식별자: {row['ref_number']} | 감지일: {str(row['created_at'])[:10]}"):
                    st.markdown(f"**[구버전 원문]({row['old_url']}) ➡️ [신버전 원문]({row['new_url']})**")
                    st.divider()
                    st.markdown(row['comparison_text'])

    # --- TAB 5: 사용 이력 및 다운로드 ---
    with tab5:
        st.markdown("#### 🗂️ RAG 채팅 및 분석 전체 이력")
        
        try:
            chat_data = supabase.table("chat_history").select("*").order("created_at", desc=False).execute().data
            analysis_data = supabase.table("analysis_history").select("*").order("created_at", desc=True).execute().data
        except Exception:
            st.error("이력 데이터를 불러오는 데 실패했습니다.")
            chat_data, analysis_data = [], []

        css_style = """
        <style>
            body { font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; line-height: 1.6; color: #333; max-width: 1000px; margin: 0 auto; padding: 30px; }
            h1, h2, h3 { color: #0052cc; border-bottom: 1px solid #eee; padding-bottom: 8px; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.95em; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top; }
            th { background-color: #f0f2f6; font-weight: bold; color: #31333F; }
            tr:nth-child(even) { background-color: #fafafa; }
            blockquote { border-left: 4px solid #0052cc; margin: 0; padding-left: 15px; color: #555; background-color: #f9f9f9; padding: 10px; }
            hr { border: 0; border-top: 1px solid #eee; margin: 30px 0; }
            .date-stamp { color: #888; font-size: 0.9em; margin-bottom: 20px; }
        </style>
        """

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💬 RAG 채팅 기록")
            if chat_data:
                md_chat = "# AI RAG 채팅 기록\n\n"
                for chat in chat_data:
                    role_kr = "사용자" if chat['role'] == 'user' else "AI"
                    md_chat += f"### {role_kr}\n<div class='date-stamp'>작성일시: {str(chat['created_at'])[:16]}</div>\n\n{chat['content']}\n\n---\n\n"
                
                try:
                    html_chat_content = markdown.markdown(md_chat, extensions=['tables'])
                    final_html_chat = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css_style}</head><body>{html_chat_content}</body></html>"
                    st.download_button(label="전체 채팅 기록 다운로드 (.html)", data=final_html_chat, file_name="rag_chat_history.html", mime="text/html")
                except Exception:
                    st.warning("파일 다운로드 준비 중 오류가 발생했습니다.")
                
                with st.container(height=600):
                    for chat in chat_data:
                        role_kr = "👤 사용자" if chat['role'] == 'user' else "🤖 AI"
                        st.markdown(f"**{role_kr}** ({str(chat['created_at'])[:16]})")
                        st.write(chat['content'])
                        st.divider()
            else:
                st.info("저장된 채팅 기록이 없습니다.")

        with col2:
            st.subheader("⚖️ 수동 비교 분석 기록")
            if analysis_data:
                for r in analysis_data:
                    doc_titles_list = []
                    file_title_prefix = "다중문서비교"
                    try:
                        docs = json.loads(r['docs_info'])
                        doc_titles_list = [f"- {d.get('title', 'Unknown')} ({d.get('agency', 'N/A')})" for d in docs]
                        agencies = [d.get('agency', 'NA') for d in docs]
                        file_title_prefix = "_vs_".join(agencies) + "_비교"
                    except:
                        pass
                    
                    raw_time = str(r['created_at'])[:16]
                    safe_time = raw_time.replace(":", "").replace("-", "").replace(" ", "_")
                    file_name = f"{file_title_prefix}_{safe_time}.html"

                    md_analysis = f"# 다중 문서 수동 비교 분석 리포트\n\n"
                    md_analysis += f"<div class='date-stamp'>분석 일시: {raw_time}</div>\n\n"
                    if doc_titles_list:
                        md_analysis += f"**[분석 대상 문서]**<br>" + "<br>".join(doc_titles_list) + "\n\n<hr>\n\n"
                    md_analysis += f"{r['comparison_result']}"
                    
                    try:
                        html_analysis_content = markdown.markdown(md_analysis, extensions=['tables'])
                        final_html_analysis = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css_style}</head><body>{html_analysis_content}</body></html>"
                    except Exception:
                        final_html_analysis = "HTML 변환 오류 발생"

                    with st.expander(f"분석 일시: {raw_time} | {file_title_prefix}"):
                        btn_col1, btn_col2 = st.columns([1, 1])
                        with btn_col1:
                            st.download_button(
                                label="📥 HTML 다운로드", 
                                data=final_html_analysis, 
                                file_name=file_name, 
                                mime="text/html",
                                key=f"dl_btn_{r['id']}"
                            )
                        with btn_col2:
                            if st.button("🗑️ 기록 삭제", key=f"del_btn_{r['id']}"):
                                delete_analysis_record(r['id'])
                                st.rerun()
                                
                        st.divider()
                        st.markdown(r['comparison_result'])
            else:
                st.info("저장된 비교 분석 기록이 없습니다.")

if __name__ == "__main__":
    main()
