import streamlit as st
import pandas as pd
from supabase import create_client, Client

@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase: Client = init_connection()

@st.cache_data(ttl=3600)
def load_data():
    response = supabase.table("guidelines").select("*").execute()
    df = pd.DataFrame(response.data)
    return df

def render_sidebar(df):
    st.sidebar.header("필터 옵션")
    
    agencies = df['agency'].dropna().unique().tolist()
    selected_agencies = st.sidebar.multiselect("규제기관 (Agency)", options=agencies, default=agencies)
    
    categories = df['category'].dropna().unique().tolist()
    selected_categories = st.sidebar.multiselect("키워드/카테고리", options=categories, default=categories)
    
    return selected_agencies, selected_categories

def main():
    st.set_page_config(page_title="RA 가이드라인 대시보드", layout="wide")
    st.title("FDA & EMA 가이드라인 통합 검색")
    
    df = load_data()
    
    if df.empty:
        st.warning("데이터베이스에 데이터가 없습니다.")
        return

    selected_agencies, selected_categories = render_sidebar(df)
    search_query = st.text_input("가이드라인 제목 검색", "")
    
    filtered_df = df[
        (df['agency'].isin(selected_agencies)) &
        (df['category'].isin(selected_categories))
    ]
    
    if search_query:
        filtered_df = filtered_df[filtered_df['title'].str.contains(search_query, case=False, na=False)]
    
    # --- 추가된 정렬 로직 ---
    # ai_summary가 존재하고, 에러 메시지가 아닌 경우를 True로 판별
    def has_valid_summary(text):
        if pd.isna(text) or str(text).strip() == "":
            return False
        if "PDF 텍스트 추출 불가" in str(text):
            return False
        return True
        
    filtered_df['has_summary'] = filtered_df['ai_summary'].apply(has_valid_summary)
    
    # 요약문 있는 문서(True)가 먼저 오도록 내림차순 정렬, 그 다음 제목순 정렬
    filtered_df = filtered_df.sort_values(by=['has_summary', 'title'], ascending=[False, True])
    # -----------------------

    st.subheader(f"검색 결과: {len(filtered_df)} 건")
    
    for index, row in filtered_df.iterrows():
        with st.expander(f"{'✅' if row['has_summary'] else '⏳'} {row['title']} ({row['agency']})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**발행기관:** {row['agency']} | **상태:** {row['status']} | **분류:** {row['category']} | **발행/업데이트일:** {row['published_date']}")
                st.markdown(f"[🔗 원본 가이드라인 문서 열기]({row['url']})")
                
            st.divider()
            st.markdown("#### 💡 AI 핵심 요약")
            
            if row['has_summary']:
                st.write(row['ai_summary'])
            else:
                st.info("현재 AI 요약이 대기 중이거나 원문 텍스트 추출이 불가능한 문서입니다.")

if __name__ == "__main__":
    main()
