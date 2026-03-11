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
    
    st.subheader(f"검색 결과: {len(filtered_df)} 건")
    
    # 펼침/숨김 패널(Expander)을 사용한 리스트 렌더링
    for index, row in filtered_df.iterrows():
        with st.expander(f"📄 {row['title']} ({row['agency']})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**발행기관:** {row['agency']} | **상태:** {row['status']} | **분류:** {row['category']} | **발행/업데이트일:** {row['published_date']}")
                st.markdown(f"[🔗 원본 가이드라인 문서 열기]({row['url']})")
                
            st.divider()
            st.markdown("#### 💡 AI 핵심 요약")
            
            # ai_summary 데이터가 존재하는 경우 출력
            if pd.notna(row.get('ai_summary')) and str(row.get('ai_summary')).strip() != "":
                st.write(row['ai_summary'])
            else:
                st.info("현재 AI 요약이 대기 중이거나 원문 텍스트 추출이 불가능한 문서입니다.")

if __name__ == "__main__":
    main()
