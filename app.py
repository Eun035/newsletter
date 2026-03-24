import streamlit as st
import pandas as pd
from kiwipiepy import Kiwi
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# --- 1. 환경 설정 및 초기화 ---
st.set_page_config(page_title="뉴스 키워드 분석기", layout="wide")
st.title("📊 뉴스 키워드 분석 앱")
st.markdown("---")

# 형태소 분석기 Kiwi 초기화
kiwi = Kiwi()

# 한국어 불용어 예시 (필요에 따라 추가/수정 가능)
STOPWORDS = set([
    '하다', '있다', '되다', '것', '이', '그', '저', '수', '등', '나', '우리', '때', '않다',
    '없다', '보이다', '가다', '오다', '같다', '지다', '대해', '대해서', '말하다', '위하다'
])

# --- 2. 데이터 로드 함수 ---
@st.cache_data # 데이터 로딩 속도 향상을 위한 캐싱
def load_data(filename):
    """CSV 파일을 읽어와 주제와 본문 컬럼만 반환합니다."""
    try:
        df = pd.read_csv(filename)
        # 필수 컬럼 존재 확인
        if 'topic' not in df.columns or 'content' not in df.columns:
            st.error(f"오류: '{filename}' 파일에 'topic' 또는 'content' 컬럼이 없습니다.")
            return None
        return df[['topic', 'content']]
    except FileNotFoundError:
        st.error(f"오류: '{filename}' 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None

# --- 3. 키워드 분석 및 전처리 함수 ---
def process_text_by_topic(df, selected_topic):
    """선택된 주제의 본문을 Kiwi로 형태소 분석하여 명사 빈도수를 반환합니다."""
    
    # 1. 주제에 해당하는 데이터만 필터링
    filtered_df = df[df['topic'] == selected_topic]
    
    if filtered_df.empty:
        return {}

    # 2. 본문 텍스트 병합 및 불필요한 공백 제거
    all_content = " ".join(filtered_df['content'].astype(str).tolist())
    
    # 3. Kiwi 형태소 분석 (명사만 추출, 불용어 제거)
    # tokenize()를 사용하여 NNG(일반 명사), NNP(고유 명사) 추출
    tokens = kiwi.tokenize(all_content)
    
    nouns = [
        token.form for token in tokens 
        if token.tag.startswith('N') and len(token.form) > 1 and token.form not in STOPWORDS
    ]
    
    # 4. 빈도수 계산
    word_counts = Counter(nouns)
    return word_counts

# --- 4. 메인 앱 로직 ---

# 데이터 로드
filename = 'news_data.csv'
df = load_data(filename)

if df is not None:
    # --- 사이드바: 주제 선택 드롭다운 ---
    st.sidebar.header("뉴스 데이터 설정")
    topics = df['topic'].unique().tolist()
    selected_topic = st.sidebar.selectbox("분석할 주제 선택", topics)
    st.sidebar.markdown(f"**선택된 데이터 파일:** `{filename}`")

    # 선택된 주제에 대한 분석 실행
    with st.spinner(f"'{selected_topic}' 주제의 키워드를 분석 중입니다..."):
        word_counts = process_text_by_topic(df, selected_topic)

    # 분석 결과가 있는 경우 시각화
    if word_counts:
        st.subheader(f"✅ '{selected_topic}' 주제 분석 결과")
        
        col1, col2 = st.columns([2, 1]) # 워드클라우드를 더 넓게 배치

        # --- 4a. 기능 2: 워드클라우드 표시 ---
        with col1:
            st.markdown("#### ☁️ 워드클라우드")
            
            # 워드클라우드 생성 (한글 폰트 설정 필수)
            # 시스템에 따라 폰트 경로를 수정해야 할 수 있습니다. (예: Mac은 'AppleGothic', Windows는 'malgun.ttf')
            # 폰트 파일이 앱 파일과 같은 폴더에 있다면 파일명만 적어도 됩니다.
            wordcloud = WordCloud(
                font_path='malgun.ttf', # Windows 기본 폰트
                background_color='white',
                width=800,
                height=600,
                colormap='Dark2',
                max_words=100
            ).generate_from_frequencies(word_counts)

            # Matplotlib을 사용하여 표시
            fig, ax = plt.subplots(figsize=(10, 7.5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off') # 축 숨기기
            st.pyplot(fig) # Streamlit에 표시

        # --- 4b. 기능 3: 키워드 TOP 10 테이블 표시 ---
        with col2:
            st.markdown("#### 🏆 TOP 10 키워드")
            
            # Counter 결과를 DataFrame으로 변환 및 정렬
            top_10_df = pd.DataFrame(
                word_counts.most_common(10), 
                columns=['키워드', '빈도수']
            )
            # 1부터 시작하는 순위 컬럼 추가
            top_10_df.index = top_10_df.index + 1
            
            st.dataframe(top_10_df, use_container_width=True) # Streamlit 테이블로 표시
            
            # 다운로드 버튼 추가 (CSV 형식)
            csv = top_10_df.to_csv(index_label='순위').encode('utf-8-sig')
            st.download_button(
                label=f"'{selected_topic}' TOP 10 다운로드",
                data=csv,
                file_name=f'top10_keywords_{selected_topic}.csv',
                mime='text/csv',
            )
    else:
        st.warning(f"'{selected_topic}' 주제에 대한 본문 데이터가 없거나 전처리 결과가 비어 있습니다.")