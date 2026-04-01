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

# 토큰 통계는 실시간 반영을 위해 캐시를 사용하지 않습니다.
def get_token_stats():
    try:
        # 전체 데이터를 가져와 합산 (서버 사이드 합산 기능 부재 시)
        res = supabase.table("token_usage").select("input_tokens, output_tokens").execute()
        df = pd.DataFrame(res.data)
        if df.empty: return 0, 0, 0, "데이터 없음"
        
        in_t = int(df['input_tokens'].sum())
        out_t = int(df['output_tokens'].sum())
        
        # 비용 계산 (Gemini 1.5 Pro 기준 설정값: $1.25/$5.00 per 1M tokens)
        # 실제 Google AI Studio 과금 체계에 맞춰 수정 가능
        cost = (in_t / 1000000 * 1.25) + (out_t / 1000000 * 5.00)
        last_update = datetime.now().strftime("%H:%M:%S")
        
        return in_t, out_t, cost, last_update
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
    if not isinstance(agency, str): return "🏳️"
    clean_agency = agency.strip().upper()
    flags = {"FDA": "🇺🇸", "EMA": "🇪🇺", "MHRA": "🇬🇧", "HEALTH CANADA": "🇨🇦", "ICH": "🌐", "MFDS": "🇰🇷"}
    return flags.get(clean_agency, "🏳️")

def delete_document_from_db(doc_url, doc_title):
    try:
        supabase.table("deleted_docs").insert({"url": doc_url, "title": doc_title}).execute()
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
    discovered = [{"title": title, "url": url} for title, url in markdown_links]
    extracted_urls = [u for t, u in markdown_links]
    for url in raw_urls:
        if url not in extracted_urls: discovered.append({"title": "Web Discovered Source", "url": url})
    for item in discovered:
        url, title = item['url'], item['title'][:200]
        agency_inferred = infer_agency_from_url(url)
        try:
            if not supabase.table("guidelines").select("url").eq("url", url).execute().data and \
               not supabase.table("deleted_docs").select("url").eq("url", url).execute().data:
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
        st.error("데이터베이스 연결 실패")
        return

    if df.empty:
        st.warning("수집된 데이터가 없습니다.")
        return

    # 사이드바 상단: 데이터 현황
    st.sidebar.header("📊 데이터 처리 현황")
    total, sum_cnt, sum_pct, emb_cnt, emb_pct = calculate_progress(df, embedded_urls)
    st.sidebar.metric("전체 수집 문서", f"{total} 건")
    st.sidebar.progress(sum_pct / 100, text=f"AI 요약: {sum_pct}%")
    st.sidebar.progress(emb_pct / 100, text=f"AI 임베딩: {emb_pct}%")
    st.sidebar.divider()

    # 사이드바 중단: 필터 및 토큰 현황 플레이스홀더
    agencies = df['agency'].dropna().unique().tolist()
    st.sidebar.multiselect("규제기관 (Agency)", options=agencies, default=agencies, key="agency_filter", format_func=lambda x: f"{get_agency_flag(x)} {x}")
    
    st.sidebar.divider()
    
    # 토큰 사용량 실시간 모니터링 섹션
    now = datetime.now()
    st.sidebar.header(f"💰 {now.year}년 {now.month}월 과금 현황")
    token_display_placeholder = st.sidebar.empty()
    
    if st.sidebar.button("🔃 사용량 즉시 새로고침"):
        st.rerun()

    # 메인 탭 로직 (중략 - 기존과 동일)
    tab_search, tab_old_new, tab_multi, tab_chat, tab_history, tab_upload = st.tabs([
        "📄 문서 검색", "🔄 신/구버전 비교", "⚖️ 다중 문서 비교", "💬 Guideline Chatbot", "🗂️ 사용 이력", "📤 PDF 수동 업로드"
    ])

    # [중요] 모든 메인 비즈니스 로직이 실행된 후 최하단에서 토큰을 다시 읽어 placeholder를 업데이트합니다.
    in_tokens, out_tokens, est_cost, update_time = get_token_stats()
    
    with token_display_placeholder.container():
        st.write(f"- 누적 입력 토큰: **{in_tokens:,}**")
        st.write(f"- 누적 출력 토큰: **{out_tokens:,}**")
        st.write(f"- 예상 과금액: :red[**${est_cost:.2f}**]")
        st.caption(f"마지막 업데이트: {update_time}")
        st.caption("※ AI Studio의 'Pay-as-you-go' 요금 기준 추정치입니다.")

    # 각 탭의 상세 로직들은 기존 코드를 유지하되, 
    # rag_engine 호출 이후 별도의 새로고침 없이도 메인 루프에 의해 하단 placeholder가 갱신됩니다.

if __name__ == "__main__":
    main()
