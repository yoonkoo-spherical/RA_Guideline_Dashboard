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

    # Supabase 기본 1000 한계를 우회하여 페이징 단위로 모든 청크 URL 가져오기
    embedded_urls = set()
    page_size = 1000
    for i in range(100):  # 최대 10만 개 청크까지 조회
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

# 토큰 통계: 실시간 반영을 위해 캐시를 사용하지 않음
def get_token_stats():
    try:
        # DB에서 직접 합산 데이터를 가져옴
        res = supabase.table("token_usage").select("input_tokens, output_tokens").execute()
        df = pd.DataFrame(res.data)
        if df.empty: return 0, 0, 0, "데이터 없음"
        
        in_t = int(df['input_tokens'].sum())
        out_t = int(df['output_tokens'].sum())
        
        # 비용 계산 (Gemini 1.5 Pro Pay-as-you-go 기준 추정치: $1.25 / $5.00 per 1M)
        cost = (in_t / 1000000 * 1.25) + (out_t / 1000000 * 5.00)
        update_time = datetime.now().strftime("%H:%M:%S")
        
        return in_t, out_t, cost, update_time
    except Exception:
        return 0, 0, 0, "오류 발생"

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
        "FDA": "🇺🇸", "EMA": "🇪🇺", "MHRA": "🇬🇧",
        "HEALTH CANADA": "🇨🇦", "ICH": "🌐", "MFDS": "🇰🇷"
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
            if existing.data: continue
            deleted = supabase.table("deleted_docs").select("url").eq("url", url).execute()
            if deleted.data: continue
            supabase.table("pending_urls").insert({"url": url, "title": title, "agency": agency_inferred}).execute()
        except Exception: pass

def main():
    st.set_page_config(page_title="RA 가이드라인 대시보드", layout="wide")

    st.markdown("""
    <style>
        .stMarkdown table { width: 100%; border-collapse: collapse; font-size: 14px; table-layout: fixed; }
        .stMarkdown th, .stMarkdown td { border: 1px solid #ddd !important; padding: 12px !important; text-align: left !important; word-wrap: break-word !important; white-space: normal !important; }
        .stMarkdown th { background-color: #f4f6f8 !important; font-weight: 600 !important; color: #333 !important; }
        .stMarkdown li { margin-bottom: 8px; line-height: 1.6; }
        .stTextInput div[data-baseweb="input"] { border: 1px solid #1f1f1f !important; border-radius: 4px !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("FDA & EMA 가이드라인 통합 검색 및 분석")

    try:
        df, comp_df, embedded_urls = load_data()
    except Exception:
        st.error("데이터베이스 연결 설정에 문제가 발생했습니다.")
        return

    if df.empty:
        st.warning("수집된 가이드라인 데이터가 없습니다.")
        return

    # 사이드바 1: 데이터 처리 현황
    st.sidebar.header("📊 데이터 처리 현황")
    total, sum_cnt, sum_pct, emb_cnt, emb_pct = calculate_progress(df, embedded_urls)
    st.sidebar.metric("전체 수집 문서", f"{total} 건")
    st.sidebar.progress(sum_pct / 100, text=f"AI 요약: {sum_pct}% ({sum_cnt}/{total})")
    st.sidebar.progress(emb_pct / 100, text=f"AI 임베딩: {emb_pct}% ({emb_cnt}/{total})")
    st.sidebar.divider()

    # 사이드바 2: 필터 및 토큰 모니터링 섹션
    agencies = df['agency'].dropna().unique().tolist()
    selected_agencies = st.sidebar.multiselect(
        "규제기관 (Agency)", options=agencies, default=agencies, format_func=lambda x: f"{get_agency_flag(x)} {x}"
    )
    categories = df['category'].dropna().unique().tolist()
    selected_categories = st.sidebar.multiselect("키워드/카테고리", options=categories, default=categories)
    st.sidebar.divider()

    # 실시간 토큰 사용량 디스플레이를 위한 플레이스홀더
    now = datetime.now()
    st.sidebar.header(f"💰 {now.year}년 {now.month}월 API 사용량")
    token_display_placeholder = st.sidebar.empty()
    
    if st.sidebar.button("🔃 사용량 새로고침", use_container_width=True):
        st.rerun()

    tab_search, tab_old_new, tab_multi, tab_chat, tab_history, tab_upload = st.tabs([
        "📄 문서 검색", "🔄 신/구버전 비교", "⚖️ 다중 문서 비교", "💬 Guideline Chatbot", "🗂️ 사용 이력", "📤 PDF 수동 업로드"
    ])

    filtered_df = df[(df['agency'].isin(selected_agencies)) & (df['category'].isin(selected_categories))]
    filtered_df['has_summary'] = filtered_df['ai_summary'].apply(lambda x: pd.notna(x) and str(x).strip() != "" and "추출 불가" not in str(x))
    filtered_df['has_embedding'] = filtered_df['url'].isin(embedded_urls)
    
    def get_status_score(row):
        if row['has_summary'] and row['has_embedding']: return 4
        if row['has_summary'] and not row['has_embedding']: return 3
        if not row['has_summary'] and row['has_embedding']: return -1
        return 1
    filtered_df['status_score'] = filtered_df.apply(get_status_score, axis=1)

    with tab_search:
        search_query = st.text_input("가이드라인 제목 검색", "", key="search_main")
        tab1_df = filtered_df.copy()
        if search_query:
            tab1_df = tab1_df[tab1_df['title'].str.contains(search_query, case=False, na=False)]
        tab1_df = tab1_df.sort_values(by=['status_score', 'title'], ascending=[False, True])
        st.subheader(f"검색 결과: {len(tab1_df)} 건")
        for index, row in tab1_df.iterrows():
            status_icon = "🟢 [완료]" if row['status_score'] == 4 else "🟡 [요약 완료]" if row['status_score'] == 3 else "🔴 [데이터 불일치]" if row['status_score'] == -1 else "⏳ [대기중]"
            agency_flag = get_agency_flag(row['agency']) 
            with st.expander(f"{status_icon} {row['title']} ({agency_flag} {row['agency']})"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**기관:** {agency_flag} {row['agency']} | **식별자:** {row.get('ref_number', 'N/A')} | **분류:** {row['category']} | **추가일:** {convert_to_kst(row.get('created_at'))}")
                    st.markdown(f"[🔗 원본 문서 열기]({row['url']})")
                with col2:
                    if st.button("🗑️ 삭제", key=f"del_doc_{index}"):
                        if delete_document_from_db(row['url'], row['title']):
                            st.success("삭제 완료")
                            load_data.clear()
                            st.rerun()
                st.divider()
                st.markdown("#### 💡 AI 핵심 요약")
                st.write(row['ai_summary'] if row['has_summary'] else "요약 대기 중입니다.")

    with tab_old_new:
        st.markdown("#### 🔄 신/구버전 자동 비교")
        if comp_df.empty: st.info("감지된 개정 이력이 없습니다.")
        else:
            for index, row in comp_df.iterrows():
                with st.expander(f"업데이트 식별자: {row['ref_number']} | 감지일: {convert_to_kst(row.get('created_at'))}"):
                    st.markdown(f"**[구버전]({row['old_url']}) ➡️ [신버전]({row['new_url']})**")
                    st.divider()
                    st.markdown(row['comparison_text'])

    with tab_multi:
        st.markdown("#### ⚖️ 다중 문서 수동 비교 분석")
        embedded_only_df = filtered_df[filtered_df['status_score'] == 4].copy() 
        multi_search = st.text_input("비교 문서 검색 (쉼표로 OR 검색)", "", key="multi_search")
        if multi_search:
            pattern = '|'.join([re.escape(kw.strip()) for kw in multi_search.split(",") if kw.strip()])
            embedded_only_df = embedded_only_df[embedded_only_df['title'].str.contains(pattern, case=False, na=False)]
        
        if not embedded_only_df.empty:
            embedded_only_df['상태'] = "🟢 준비 완료"
            df_sel = embedded_only_df[['agency', 'category','title', '상태', 'url']].copy()
            df_sel['agency'] = df_sel['agency'].apply(lambda x: f"{get_agency_flag(x)} {x}")
            df_sel.insert(0, "선택", False)
            edited_df = st.data_editor(df_sel, hide_index=True, column_config={"선택": st.column_config.CheckboxColumn("선택", default=False), "url": None}, disabled=["agency", "category", "title", "상태"], use_container_width=True)
            selected_rows = edited_df[edited_df["선택"]]
            if st.button("비교 분석 실행", type="primary") and len(selected_rows) >= 2:
                with st.spinner("분석 중..."):
                    res = rag_engine.compare_multiple_documents(selected_rows.to_dict('records'))
                    if "오류" not in res:
                        save_analysis_to_db(selected_rows.to_dict('records'), clean_html_tags(res))
                        st.markdown(res, unsafe_allow_html=True)
                    else: st.error(res)

    with tab_chat:
        st.markdown("#### 규제 가이드라인 AI 어시스턴트")
        if "messages" not in st.session_state: st.session_state.messages = []
        if "chat_input_val" not in st.session_state: st.session_state.chat_input_val = ""
        if "current_prompt" not in st.session_state: st.session_state.current_prompt = None

        def submit_chat():
            if st.session_state.chat_input_val.strip():
                st.session_state.current_prompt = st.session_state.chat_input_val
                st.session_state.chat_input_val = ""

        st.text_input("질문을 입력하세요 (Enter):", key="chat_input_val", on_change=submit_chat)
        st.divider()

        if st.session_state.current_prompt:
            p = st.session_state.current_prompt
            st.session_state.current_prompt = None
            st.session_state.messages.append({"role": "user", "content": p})
            save_chat_to_db("user", p)
            with st.spinner("답변 생성 중..."):
                try:
                    ans, srcs = rag_engine.ask_guideline(p)
                    st.session_state.messages.append({"role": "assistant", "content": ans, "sources": srcs})
                    save_chat_to_db("assistant", ans)
                    queue_web_discovered_urls(ans)
                except Exception as e: st.error(f"오류: {e}")

        pairs = [st.session_state.messages[i:i+2] for i in range(0, len(st.session_state.messages), 2)]
        for pair in reversed(pairs):
            for msg in pair:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and msg.get("sources"):
                        with st.expander("🔍 참고 내부 문서"):
                            for idx, s in enumerate(msg["sources"]):
                                st.markdown(f"**[{idx+1}]** [{s.get('title', 'Link')}]({s['url']})")

    with tab_history:
        st.markdown("#### 🗂️ 사용 이력")
        delete_old_chat_records()
        try:
            chat_data = supabase.table("chat_history").select("*").order("created_at", desc=False).execute().data
            analysis_data = supabase.table("analysis_history").select("*").order("created_at", desc=True).execute().data
        except Exception: chat_data, analysis_data = [], []
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💬 Chatbot 기록 (7일)")
            if chat_data:
                daily = defaultdict(list)
                for c in chat_data: daily[convert_to_kst(c['created_at']).split(" ")[0]].append(c)
                for d_str in sorted(daily.keys(), reverse=True):
                    md = f"# {d_str} Chat 기록\n\n"
                    for c in daily[d_str]: md += f"### {'사용자' if c['role']=='user' else 'AI'}\n{c['content']}\n\n"
                    st.download_button(f"💾 {d_str} 다운로드", markdown.markdown(md), f"chat_{d_str}.html", "text/html")
        with col2:
            st.subheader("⚖️ 비교 분석 기록")
            for r in analysis_data:
                with st.expander(f"분석: {convert_to_kst(r['created_at'])}"):
                    if st.button("🗑️ 삭제", key=f"del_his_{r['id']}"):
                        delete_analysis_record(r['id'])
                        st.rerun()
                    st.markdown(r['comparison_result'], unsafe_allow_html=True)

    with tab_upload:
        st.markdown("#### 📤 로컬 PDF 업로드")
        c1, c2 = st.columns(2)
        with c1: ag_in = st.selectbox("발행 기관", ["FDA", "EMA", "MHRA", "Health Canada", "ICH", "MFDS", "기타"])
        with c2: cat_in = st.text_input("카테고리 (예: CMC, 임상)")
        up_file = st.file_uploader("PDF 선택", type="pdf")
        if st.button("DB 추가", type="primary") and up_file and cat_in:
            with st.spinner("업로드 중..."):
                try:
                    supabase.storage.from_("guidelines_pdf").upload(up_file.name, up_file.read())
                    f_url = supabase.storage.from_("guidelines_pdf").get_public_url(up_file.name)
                    supabase.table("guidelines").insert({"title": up_file.name, "agency": ag_in, "category": cat_in, "url": f_url}).execute()
                    st.success("완료! 자동 분석이 시작됩니다.")
                except Exception as e: st.error(f"에러: {e}")

    # ---------------------------------------------------------
    # [최종 단계] 플레이스홀더를 최신 토큰 정보로 업데이트
    # 모든 탭 로직이 끝난 후 실행되므로 챗봇 등에서 쓴 토큰이 바로 반영됩니다.
    # ---------------------------------------------------------
    in_t, out_t, cost, u_time = get_token_stats()
    with token_display_placeholder.container():
        st.write(f"- 누적 입력 토큰: **{in_t:,}**")
        st.write(f"- 누적 출력 토큰: **{out_t:,}**")
        st.write(f"- 예상 과금액: :red[**${cost:.2f}**]")
        st.caption(f"최종 업데이트: {u_time}")
        st.caption("※ Pro 요금제 전환 시 실 과금액 추정치입니다.")

if __name__ == "__main__":
    main()
