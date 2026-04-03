import streamlit as st
import pandas as pd
from supabase import create_client, Client
import rag_engine
import json
import markdown
import time
from datetime import datetime, timedelta
from collections import defaultdict
import re
import requests

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

    embedded_urls = set()
    page_size = 1000
    for i in range(100):
        chunk_response = supabase.table("document_chunks").select("url").neq("content", "FAILED").range(i * page_size, (i + 1) * page_size - 1).execute()
        if not chunk_response.data:
            break
        embedded_urls.update(item['url'] for item in chunk_response.data)

    return df, comp_df, embedded_urls

def calculate_progress(df, embedded_urls):
    if df.empty: return 0, 0, 0, 0, 0
    total_docs = len(df)
    summarized_docs = len(df[df['ai_summary'].notna() & (df['ai_summary'] != '') & (~df['ai_summary'].str.contains('텍스트 추출 불가', na=False))])
    embedded_docs = len([url for url in df['url'] if url in embedded_urls])
    summary_pct = int((summarized_docs / total_docs) * 100) if total_docs > 0 else 0
    embed_pct = int((embedded_docs / total_docs) * 100) if total_docs > 0 else 0
    return total_docs, summarized_docs, summary_pct, embedded_docs, embed_pct

# 토큰 현황 1시간 간격 자동 갱신
@st.cache_data(ttl=3600)
def get_token_stats():
    try:
        res = supabase.table("token_usage").select("*").execute()
        df = pd.DataFrame(res.data)
        if df.empty: return 0, 0, 0
        in_t = int(df['input_tokens'].sum())
        out_t = int(df['output_tokens'].sum())
        cost = (in_t / 1000000 * 1.25) + (out_t / 1000000 * 5.00)
        return in_t, out_t, cost
    except Exception:
        return 0, 0, 0

def save_chat_to_db(role, content):
    try:
        supabase.table("chat_history").insert({"role": role, "content": content}).execute()
    except Exception:
        pass

def delete_old_chat_records():
    try:
        seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        supabase.table("chat_history").delete().lt("created_at", seven_days_ago).execute()
    except Exception:
        pass

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

def convert_to_kst(time_str):
    if pd.isna(time_str) or str(time_str).strip() == "" or str(time_str).lower() == 'nan':
        return "정보 없음"
    try:
        dt = pd.to_datetime(time_str)
        if dt.tzinfo is None:
            dt = dt.tz_localize('UTC')
        kst_dt = dt.tz_convert('Asia/Seoul')
        return kst_dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return str(time_str).replace("T", " ")[:16]

def clean_html_tags(text):
    if not text: return ""
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = text.replace('**\n**', '**\n\n**')
    return text

def get_agency_flag(agency):
    if not isinstance(agency, str):
        return "🏳️"
    clean_agency = agency.strip().upper()
    flags = {
        "FDA": "🇺🇸",
        "EMA": "🇪🇺",
        "MHRA": "🇬🇧",
        "HEALTH CANADA": "🇨🇦",
        "ICH": "🌐",
        "MFDS": "🇰🇷"
    }
    return flags.get(clean_agency, "🏳️")

def delete_document_from_db(doc_url, doc_title):
    try:
        supabase.table("deleted_docs").insert({"url": doc_url, "title": doc_title}).execute()
    except Exception:
        pass 

    try:
        supabase.table("document_chunks").delete().eq("url", doc_url).execute()
        supabase.table("guidelines").delete().eq("url", doc_url).execute()
        return True
    except Exception as e:
        st.error(f"데이터베이스 삭제 중 오류 발생: {e}")
        return False

def infer_agency_from_url(url):
    url_lower = url.lower()
    if "fda.gov" in url_lower: return "FDA"
    elif "ema.europa.eu" in url_lower: return "EMA"
    elif "gov.uk" in url_lower: return "MHRA"
    elif "canada.ca" in url_lower or "hc-sc.gc.ca" in url_lower: return "Health Canada"
    elif "ich.org" in url_lower: return "ICH"
    elif "mfds.go.kr" in url_lower: return "MFDS"
    else: return "기타"

def queue_web_discovered_urls(response_text):
    markdown_links = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', response_text)
    raw_urls = re.findall(r'(?<!\()(https?://[^\s\)\]]+)', response_text)
    
    discovered = []
    for title, url in markdown_links:
        discovered.append({"title": title, "url": url})
        
    extracted_urls = [u for t, u in markdown_links]
    for url in raw_urls:
        if url not in extracted_urls:
            discovered.append({"title": "Web Discovered Source", "url": url})
            
    for item in discovered:
        url = item['url']
        title = item['title'][:200]
        agency_inferred = infer_agency_from_url(url)
        
        try:
            existing = supabase.table("guidelines").select("url").eq("url", url).execute()
            if existing.data:
                continue
                
            deleted = supabase.table("deleted_docs").select("url").eq("url", url).execute()
            if deleted.data:
                continue
                
            supabase.table("pending_urls").insert({
                "url": url, 
                "title": title, 
                "agency": agency_inferred
            }).execute()
        except Exception:
            pass

def main():
    st.set_page_config(page_title="RA 가이드라인 대시보드", layout="wide")

    st.markdown("""
    <style>
        .stMarkdown table { width: 100%; border-collapse: collapse; font-size: 14px; table-layout: fixed; }
        .stMarkdown th, .stMarkdown td { border: 1px solid #ddd !important; padding: 12px !important; text-align: left !important; word-wrap: break-word !important; white-space: normal !important; }
        .stMarkdown th { background-color: #f4f6f8 !important; font-weight: 600 !important; color: #333 !important; }
        .stMarkdown li { margin-bottom: 8px; line-height: 1.6; }
        .stTextArea textarea { border: 1px solid #1f1f1f !important; border-radius: 4px !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("FDA & EMA 가이드라인 통합 검색 및 분석")

    try:
        df, comp_df, embedded_urls = load_data()
    except Exception:
        st.error("데이터베이스에서 가이드라인 정보를 불러오는 데 실패했습니다. 잠시 후 다시 시도해 주십시오.")
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

    agencies = df['agency'].dropna().unique().tolist()
    selected_agencies = st.sidebar.multiselect(
        "규제기관 (Agency)", options=agencies, default=agencies, format_func=lambda x: f"{get_agency_flag(x)} {x}"
    )
    categories = df['category'].dropna().unique().tolist()
    selected_categories = st.sidebar.multiselect("키워드/카테고리", options=categories, default=categories)
    st.sidebar.divider()

    now = datetime.now()
    st.sidebar.header(f"💰 {now.year}년 {now.month}월 API 토큰 현황")
    
    # 토큰 현황 데이터 및 수동 새로고침 UI
    if st.sidebar.button("🔄 토큰 현황 수동 새로고침", use_container_width=True):
        get_token_stats.clear()
        
    in_tokens, out_tokens, est_cost = get_token_stats()
    st.sidebar.write(f"- 누적 입력 토큰: **{in_tokens:,}**")
    st.sidebar.write(f"- 누적 출력 토큰: **{out_tokens:,}**")
    st.sidebar.write(f"- 예상 과금액: **${est_cost:.2f}**")
    st.sidebar.caption("※ 1시간 간격으로 자동 갱신됩니다.")

    tab_search, tab_old_new, tab_multi, tab_chat, tab_history, tab_upload = st.tabs([
        "📄 문서 검색", "🔄 신/구버전 비교", "⚖️ 다중 문서 비교", "💬 Guideline Chatbot", "🗂️ 사용 이력", "📤 PDF 수동 업로드"
    ])

    filtered_df = df[(df['agency'].isin(selected_agencies)) & (df['category'].isin(selected_categories))]

    def check_summary(text):
        if pd.isna(text) or str(text).strip() == "": return False
        if "추출 불가" in str(text): return False
        return True

    filtered_df['has_summary'] = filtered_df['ai_summary'].apply(check_summary)
    filtered_df['has_embedding'] = filtered_df['url'].isin(embedded_urls)

    def get_status_score(row):
        if row['has_summary'] and row['has_embedding']: return 4
        if row['has_summary'] and not row['has_embedding']: return 3
        if not row['has_summary'] and row['has_embedding']: return -1
        return 1

    filtered_df['status_score'] = filtered_df.apply(get_status_score, axis=1)

    error_count = len(filtered_df[filtered_df['status_score'] == -1])
    if error_count > 0:
        st.sidebar.divider()
        st.sidebar.error(f"⚠️ 데이터 불일치 문서: {error_count}건\n\n(요약은 없으나 벡터 DB에 데이터가 존재합니다. '문서 검색' 탭에서 확인하십시오.)")

    with tab_search:
        search_query = st.text_input("가이드라인 제목 검색", "")
        tab1_df = filtered_df.copy()
        if search_query:
            tab1_df = tab1_df[tab1_df['title'].str.contains(search_query, case=False, na=False)]
        tab1_df = tab1_df.sort_values(by=['status_score', 'title'], ascending=[False, True])
        st.subheader(f"검색 결과: {len(tab1_df)} 건")

        for index, row in tab1_df.iterrows():
            if row['status_score'] == 4: status_icon = "🟢 [완료]"
            elif row['status_score'] == 3: status_icon = "🟡 [요약 완료]"
            elif row['status_score'] == -1: status_icon = "🔴 [오류: 데이터 불일치]"
            else:
                if isinstance(row.get('ai_summary'), str) and "추출 불가" in row['ai_summary']: status_icon = "⚪ [추출 실패]"
                else: status_icon = "⏳ [대기중]"

            agency_flag = get_agency_flag(row['agency']) 

            with st.expander(f"{status_icon} {row['title']} ({agency_flag} {row['agency']})"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    db_added_date = convert_to_kst(row.get('created_at'))
                    st.write(f"**기관:** {agency_flag} {row['agency']} | **식별자:** {row.get('ref_number', 'N/A')} | **분류:** {row['category']} | **DB 추가일:** {db_added_date}")
                    st.markdown(f"[🔗 원본 문서 열기]({row['url']})")
                
                with col2:
                    if st.button("🗑️ 문서 수동 삭제", key=f"del_doc_{index}"):
                        with st.spinner("문서 및 임베딩 데이터를 삭제 중입니다..."):
                            if delete_document_from_db(row['url'], row['title']):
                                st.success("문서가 성공적으로 삭제되었습니다.")
                                load_data.clear()
                                time.sleep(1)
                                st.rerun()

                st.divider()
                st.markdown("#### 💡 AI 핵심 요약")
                if row['has_summary']: st.write(row['ai_summary'])
                elif isinstance(row.get('ai_summary'), str) and "추출 불가" in row['ai_summary']: st.error(row['ai_summary'])
                else: st.info("AI 요약 대기 중입니다.")

                if row['status_score'] == -1: st.warning("⚠️ 요약 텍스트가 없음에도 벡터 DB에 임베딩 데이터가 존재합니다.")
                elif not row['has_embedding']: st.warning("⚠️ RAG 검색용 벡터 DB에 임베딩되지 않은 문서입니다.")

    with tab_old_new:
        st.markdown("#### 🔄 규제 가이드라인 신/구버전 변경점 자동 비교")
        if comp_df.empty: st.info("현재 문서 간의 버전 업데이트(개정) 이력이 감지되지 않았습니다.")
        else:
            for index, row in comp_df.iterrows():
                db_added_date = convert_to_kst(row.get('created_at'))
                with st.expander(f"업데이트 식별자: {row['ref_number']} | 감지일: {db_added_date}"):
                    st.markdown(f"**[구버전 원문]({row['old_url']}) ➡️ [신버전 원문]({row['new_url']})**")
                    st.divider()
                    st.markdown(row['comparison_text'])

    with tab_multi:
        st.markdown("#### ⚖️ 다중 문서 수동 비교 분석")
        embedded_only_df = filtered_df[filtered_df['status_score'] == 4].copy() 

        multi_search_query = st.text_input("비교할 문서 검색 (쉼표(,)로 구분하여 다중 OR 검색 가능)", "")
        if multi_search_query:
            keywords = [kw.strip() for kw in multi_search_query.split(",") if kw.strip()]
            if keywords:
                pattern = '|'.join(map(re.escape, keywords))
                embedded_only_df = embedded_only_df[embedded_only_df['title'].str.contains(pattern, case=False, na=False)]

        if embedded_only_df.empty:
            st.info("임베딩 및 요약이 정상적으로 완료된 문서가 없거나 검색 조건에 맞는 문서가 없습니다.")
        else:
            embedded_only_df['상태'] = "🟢 준비 완료"
            df_for_selection = embedded_only_df[['agency', 'title', 'category', '상태', 'url']].copy()
            df_for_selection['agency'] = df_for_selection['agency'].apply(lambda x: f"{get_agency_flag(x)} {x}")
            df_for_selection.insert(0, "비교 선택", False)
            edited_df = st.data_editor(
                df_for_selection, hide_index=True,
                column_config={"비교 선택": st.column_config.CheckboxColumn("비교 선택", default=False), "url": None},
                disabled=["agency", "category", "title", "상태"], use_container_width=True
            )

            selected_rows = edited_df[edited_df["비교 선택"]]
            if st.button("비교 분석 실행", type="primary"):
                if len(selected_rows) < 2: 
                    st.warning("문서를 2개 이상 선택해야 합니다.")
                else:
                    selected_docs_info = selected_rows.to_dict('records')
                    with st.spinner("문서 대조 중..."):
                        try:
                            comparison_result = rag_engine.compare_multiple_documents(selected_docs_info)
                            if "오류" in comparison_result: st.error("문서 비교 분석 중 오류가 발생했습니다.")
                            else:
                                cleaned_result = clean_html_tags(comparison_result)
                                save_analysis_to_db(selected_docs_info, cleaned_result)
                                st.divider()
                                st.markdown("#### 📊 분석 결과")
                                st.markdown(cleaned_result, unsafe_allow_html=True)
                        except Exception:
                            st.error("분석 서버와의 통신에 실패했습니다.")

    with tab_chat:
        st.markdown("#### 💬 규제 가이드라인 AI 어시스턴트 (Guideline Chatbot)")

        st.markdown("##### 📌 필수 참조 문서 선택 (선택 사항)")
        chat_embedded_only_df = filtered_df[filtered_df['status_score'] == 4].copy()

        chat_search_query = st.text_input("참조할 문서 검색 (쉼표(,)로 구분하여 다중 검색 가능)", "", key="chat_multi_search")
        if chat_search_query:
            keywords = [kw.strip() for kw in chat_search_query.split(",") if kw.strip()]
            if keywords:
                pattern = '|'.join(map(re.escape, keywords))
                chat_embedded_only_df = chat_embedded_only_df[chat_embedded_only_df['title'].str.contains(pattern, case=False, na=False)]

        forced_docs_info = []
        if chat_embedded_only_df.empty:
            st.info("선택 가능한 문서가 없습니다.")
        else:
            chat_embedded_only_df['상태'] = "🟢 준비 완료"
            chat_df_for_selection = chat_embedded_only_df[['agency', 'title', 'category', '상태', 'url']].copy()
            chat_df_for_selection['agency'] = chat_df_for_selection['agency'].apply(lambda x: f"{get_agency_flag(x)} {x}")
            chat_df_for_selection.insert(0, "참조 선택", False)
            
            edited_chat_df = st.data_editor(
                chat_df_for_selection, hide_index=True,
                column_config={"참조 선택": st.column_config.CheckboxColumn("참조 선택", default=False), "url": None},
                disabled=["agency", "category", "title", "상태"], use_container_width=True,
                key="chat_data_editor"
            )
            
            selected_chat_rows = edited_chat_df[edited_chat_df["참조 선택"]]
            forced_docs_info = selected_chat_rows.to_dict('records')
            
        st.divider()

        if "messages" not in st.session_state: st.session_state.messages = []
        if "current_prompt" not in st.session_state: st.session_state.current_prompt = None

        # 입력창 위쪽 배치 및 다중행(Text Area) 대응 폼 설정
        with st.form("chat_input_form", clear_on_submit=True):
            user_input = st.text_area("질문을 입력하세요 (줄바꿈: Enter, 전송: Ctrl+Enter 또는 '전송' 버튼 클릭)", height=100)
            submitted = st.form_submit_button("전송", type="primary")
            if submitted and user_input.strip():
                st.session_state.current_prompt = user_input
                
        st.divider()

        # 프롬프트 제출 시 로직
        if st.session_state.current_prompt:
            prompt = st.session_state.current_prompt
            st.session_state.current_prompt = None

            st.session_state.messages.append({"role": "user", "content": prompt})
            save_chat_to_db("user", prompt)

            with st.spinner("답변 생성 중... (외부 규정 참조 시 대기열에 자동 등록됩니다)"):
                try:
                    response_text, sources = rag_engine.ask_guideline(prompt, forced_docs_info)
                    if "오류가 발생" in response_text:
                        st.error("AI가 답변을 생성하지 못했습니다.")
                        st.session_state.messages.append({"role": "assistant", "content": "오류가 발생했습니다.", "sources": []})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": response_text, "sources": sources})
                        save_chat_to_db("assistant", response_text)
                        
                        queue_web_discovered_urls(response_text)
                        
                except Exception:
                    st.error("예기치 않은 시스템 오류가 발생했습니다.")
                    st.session_state.messages.append({"role": "assistant", "content": "시스템 오류가 발생했습니다.", "sources": []})
            
            st.rerun() # 답변 렌더링 후 UI 즉시 새로고침

        # 최신 질문과 답변이 상단에 뜨도록 메시지 리스트 역순 정렬 렌더링
        pairs = []
        for i in range(0, len(st.session_state.messages), 2):
            pairs.append(st.session_state.messages[i:i+2])
            
        for pair in reversed(pairs):
            for msg in pair:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and msg.get("sources"):
                        with st.expander("🔍 AI가 참고한 관련 내부 DB 문서 확인"):
                            for idx, source in enumerate(msg["sources"]):
                                st.markdown(f"**[{idx+1}] 출처:** [{source.get('title', '원문 링크')}]({source['url']})")

    with tab_history:
        st.markdown("#### 🗂️ Chatbot 및 분석 전체 이력")
        delete_old_chat_records()

        try:
            chat_data = supabase.table("chat_history").select("*").order("created_at", desc=False).execute().data
            analysis_data = supabase.table("analysis_history").select("*").order("created_at", desc=True).execute().data
        except Exception:
            chat_data, analysis_data = [], []

        css_style = """<style>body { font-family: sans-serif; padding: 30px; } h1, h2, h3 { color: #0052cc; border-bottom: 1px solid #eee; padding-bottom: 8px; } table { border-collapse: collapse; width: 100%; margin: 20px 0; } th, td { border: 1px solid #ddd; padding: 12px; } th { background-color: #f0f2f6; } .date-stamp { color: #666; font-size: 0.9em; margin-bottom: 15px; }</style>"""

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💬 Guideline Chatbot 기록 (최근 7일)")
            if chat_data:
                daily_chats = defaultdict(list)
                for chat in chat_data:
                    chat_time_kst = convert_to_kst(chat.get('created_at'))
                    date_str = chat_time_kst.split(" ")[0]
                    daily_chats[date_str].append((chat, chat_time_kst))

                for date_str in sorted(daily_chats.keys(), reverse=True):
                    chats_for_day = daily_chats[date_str]
                    md_chat = f"# {date_str} Guideline Chatbot 기록\n\n"
                    for chat, chat_time_kst in chats_for_day:
                        role_kr = "사용자" if chat['role'] == 'user' else "AI"
                        md_chat += f"### {role_kr}\n<div class='date-stamp'>작성일시: {chat_time_kst}</div>\n\n{chat['content']}\n\n---\n\n"
                    try:
                        html_chat_content = markdown.markdown(md_chat, extensions=['tables'])
                        final_html_chat = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css_style}</head><body>{html_chat_content}</body></html>"
                        st.download_button(label=f"💾 {date_str} 채팅 기록 다운로드 (.html)", data=final_html_chat, file_name=f"guideline_chat_{date_str}.html", mime="text/html", key=f"dl_chat_{date_str}")
                    except Exception: pass

                st.divider()
                st.write("▼ 채팅 내역 확인")
                
                # 질의응답 세트 단위로 묶어서 역순으로 출력 (가장 최신 대화가 위에 오도록)
                grouped_chats = []
                temp_pair = []
                for chat in chat_data:
                    if chat['role'] == 'user':
                        if temp_pair: grouped_chats.append(temp_pair)
                        temp_pair = [chat]
                    else:
                        temp_pair.append(chat)
                        grouped_chats.append(temp_pair)
                        temp_pair = []
                if temp_pair: grouped_chats.append(temp_pair)

                with st.container(height=600):
                    for pair in reversed(grouped_chats):
                        for chat in pair:
                            role_kr = "👤 사용자" if chat['role'] == 'user' else "🤖 AI"
                            chat_time_kst = convert_to_kst(chat.get('created_at'))
                            st.markdown(f"**{role_kr}** ({chat_time_kst})")
                            st.write(chat['content'])
                        st.divider()
            else:
                st.info("최근 7일간 저장된 채팅 기록이 없습니다.")

        with col2:
            st.subheader("⚖️ 수동 비교 분석 기록")
            if analysis_data:
                for r in analysis_data:
                    file_title_prefix = "다중문서비교"
                    raw_time = convert_to_kst(r.get('created_at'))
                    safe_time = raw_time.replace(":", "").replace("-", "").replace(" ", "_")
                    file_name = f"{file_title_prefix}_{safe_time}.html"

                    md_analysis = f"# 다중 문서 수동 비교 분석 리포트\n\n<div class='date-stamp'>분석 일시: {raw_time}</div>\n\n"
                    cleaned_db_result = clean_html_tags(r.get('comparison_result', ''))
                    md_analysis += f"{cleaned_db_result}"

                    try:
                        html_analysis_content = markdown.markdown(md_analysis, extensions=['tables'])
                        final_html_analysis = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css_style}</head><body>{html_analysis_content}</body></html>"
                    except:
                        final_html_analysis = "HTML 변환 오류 발생"

                    with st.expander(f"분석 일시: {raw_time} | {file_title_prefix}"):
                        btn_col1, btn_col2 = st.columns([1, 1])
                        with btn_col1:
                            st.download_button(label="📥 HTML 다운로드", data=final_html_analysis, file_name=file_name, mime="text/html", key=f"dl_btn_{r['id']}")
                        with btn_col2:
                            if st.button("🗑️ 기록 삭제", key=f"del_btn_{r['id']}"):
                                delete_analysis_record(r['id'])
                                st.rerun()
                        st.divider()
                        st.markdown(cleaned_db_result, unsafe_allow_html=True)
            else:
                st.info("저장된 비교 분석 기록이 없습니다.")

    with tab_upload:
        st.markdown("#### 📤 로컬 PDF 가이드라인 업로드")
        col1, col2 = st.columns(2)
        with col1: agency_input = st.selectbox("발행 기관 (Agency)", ["FDA", "EMA", "MHRA", "Health Canada", "ICH", "MFDS", "기타"])
        with col2: category_input = st.text_input("카테고리/키워드 (예: CMC, 임상, 비임상)")

        uploaded_file = st.file_uploader("PDF 파일 선택", type="pdf")

        if st.button("데이터베이스 추가", type="primary"):
            if uploaded_file is not None and category_input:
                with st.spinner("파일 업로드 중입니다..."):
                    file_name = uploaded_file.name
                    file_bytes = uploaded_file.read()
                    try:
                        supabase.storage.from_("guidelines_pdf").upload(file_name, file_bytes)
                        file_url = supabase.storage.from_("guidelines_pdf").get_public_url(file_name)
                        supabase.table("guidelines").insert({"title": file_name, "agency": agency_input, "category": category_input, "url": file_url}).execute()
                        st.success("데이터베이스 추가 완료! 스케줄러에 의해 자동으로 분석 파이프라인이 실행됩니다.")
                    except Exception as e:
                        st.error(f"업로드 에러: {e}")
            else:
                st.warning("PDF 파일을 첨부하고 카테고리를 입력해 주십시오.")

if __name__ == "__main__":
    main()
