import streamlit as st
import pandas as pd
from supabase import create_client, Client
import rag_engine
import json
import markdown
from datetime import datetime, timedelta
from collections import defaultdict
import re

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

def get_token_stats():
    """DB에서 토큰 사용량을 집계하여 월간/누적 비용 계산"""
    try:
        res = supabase.table("token_usage").select("*").execute()
        df = pd.DataFrame(res.data)
        if df.empty: return 0, 0, 0
        in_t = int(df['input_tokens'].sum())
        out_t = int(df['output_tokens'].sum())
        # 비용 추정: 1.5 Pro 요율 기준 ($1.25 / 1M input, $5.00 / 1M output)
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
    """7일이 경과한 채팅 기록 삭제"""
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
    """결과물 내 불필요한 HTML 태그 및 깨진 테이블 요소 보정"""
    if not text: return ""
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = text.replace('**\n**', '**\n\n**')
    return text
    
def get_agency_flag(agency):
    """규제기관 문자열에 매칭되는 국
