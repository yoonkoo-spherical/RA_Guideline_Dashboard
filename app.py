import streamlit as st
import pandas as pd
from supabase import create_client, Client

# 1. 초기 설정 및 DB 연결
# 향후 Streamlit 배포를 고려하여 os.getenv 대신 st.secrets 활용
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase: Client = init_connection()

# 2. 데이터 호출 (캐싱 적용)
# DB 쿼리 부하를 줄이기 위해 1시간(3600초) 단위로 데이터를 캐싱
@st.cache_data(ttl=3600)
def load_data():
    response = supabase.table("guidelines").select("*").execute()
    df = pd.DataFrame(response.data)
    return df

# 3. 사이드바 필터 UI 로직
def render_sidebar(df):
    st.sidebar.header("필터 옵션")
    
    # 규제기관 필터
    agencies = df['agency'].dropna().unique().tolist()
    selected_agencies = st.sidebar.multiselect("규제기관 (Agency)", options=agencies, default=agencies)
    
    # 카테고리 필터
    categories = df['category'].dropna().unique().tolist()
    selected_categories = st.sidebar.multiselect("키워드/카테고리", options=categories, default=categories)
    
    return selected_agencies, selected_categories

# 4. 메인 화면 UI 및 필터링 적용 로직
def main():
    st.set_page_config(page_title="RA 가이드라인 대시보드", layout="wide")
    st.title("FDA & EMA 가이드라인 통합 검색")
    
    df = load_data()
    
    if df.empty:
        st.warning("데이터베이스에 데이터가 없습니다.")
        return

    # 필터 조건 수신
    selected_agencies, selected_categories = render_sidebar(df)
    
    # 텍스트 검색창
    search_query = st.text_input("가이드라인 제목 검색", "")
    
    # 데이터 필터링
    filtered_df = df[
        (df['agency'].isin(selected_agencies)) &
        (df['category'].isin(selected_categories))
    ]
    
    if search_query:
        filtered_df = filtered_df[filtered_df['title'].str.contains(search_query, case=False, na=False)]
    
    # 화면에 출력할 컬럼 순서 정리
    display_columns = ['agency', 'title', 'status', 'published_date', 'category', 'url']
    filtered_df = filtered_df[display_columns]
    
    # 결과 출력
    st.subheader(f"검색 결과: {len(filtered_df)} 건")
    
    # 데이터프레임 렌더링 (URL을 클릭 가능한 링크로 변환)
    st.dataframe(
        filtered_df,
        column_config={
            "url": st.column_config.LinkColumn("문서 링크 (URL)")
        },
        hide_index=True,
        use_container_width=True
    )

if __name__ == "__main__":
    main()
