import streamlit as st
import requests
from sqlalchemy import create_engine, inspect, text, select
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import sessionmaker
from db_manager import Base, Case, engine
import re
import logging
import json
import os
from typing import List, Tuple, Optional
import gdown

# Streamlit 설정
st.set_page_config(page_title="AI 기반 맞춤형 판례 검색 서비스", layout="wide")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 상수 정의
API_KEY = "D/spYGY15giVS64SLvtShZlNHxAbr9eDi1uU1Ca1wrqCiU+0YMwcnFy53naflVlg5wemikAYwiugNoIepbpexQ=="
API_URL = "https://api.odcloud.kr/api/15069932/v1/uddi:3799441a-4012-4caa-9955-b4d20697b555"
CACHE_FILE = "legal_terms_cache.json"
DB_FILE = os.path.join(os.path.dirname(__file__), "legal_cases.db")

# 데이터베이스 엔진 재정의
engine = create_engine(f'sqlite:///{DB_FILE}')

@st.cache_data
def get_legal_terms() -> dict:
    if os.path.exists(CACHE_FILE):
        logging.info("저장된 용어 사전 불러오기")
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            legal_terms_dict = json.load(f)
        logging.info(f"{len(legal_terms_dict)}개의 법률 용어를 캐시에서 불러왔습니다.")
    else:
        logging.info("API에서 법률 용어 데이터 가져오기 시작")
        params = {
            "serviceKey": API_KEY,
            "page": 1,
            "perPage": 1000
        }
        response = requests.get(API_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                legal_terms_dict = {item['용어명']: item['설명'] for item in data['data']}
                logging.info(f"{len(legal_terms_dict)}개의 법률 용어를 가져왔습니다.")
                
                with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(legal_terms_dict, f, ensure_ascii=False, indent=2)
                logging.info("법률 용어 데이터를 캐시 파일에 저장했습니다.")
            else:
                logging.error("API 응답에 'data' 키가 없습니다.")
                legal_terms_dict = {}
        else:
            logging.error(f"API 요청 실패: 상태 코드 {response.status_code}")
            legal_terms_dict = {}
    
    return legal_terms_dict

def download_db():
    file_id = "1rBTbbtBE5K5VgiuTvt3JgneuJ8odqCJm"
    output = DB_FILE
    gdown.download(id=file_id, output=output, quiet=False)
    logging.info(f"데이터베이스 다운로드 완료: {output}")

def check_db(session):
    inspector = inspect(engine)
    try:
        if not os.path.exists(DB_FILE):
            logging.info("데이터베이스 파일이 없습니다. 다운로드를 시작합니다.")
            download_db()
        
        for table_name in inspector.get_table_names():
            stmt = select(text('1')).select_from(text(table_name)).limit(1)
            result = session.execute(stmt)
            if result.first():
                return True
        logging.warning("데이터베이스에 테이블이 없습니다. 다운로드를 다시 시도합니다.")
        download_db()
        return False
    except Exception as e:
        logging.error(f"데이터베이스 확인 중 오류 발생: {str(e)}")
        return False
    finally:
        session.close()

@st.cache_resource
def load_cases() -> List[Case]:
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    logging.info("데이터베이스에서 판례 데이터 로딩 시작")
    try:
        check_db(session)  # 데이터베이스 확인 및 다운로드
        total_cases = session.query(Case).count()
        logging.info(f"총 {total_cases}개의 판례가 데이터베이스에 있습니다.")
        
        cases = list(session.query(Case))
        logging.info(f"총 {len(cases)}개의 판례를 로드했습니다.")
        return cases

    except Exception as e:
        logging.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return []

    finally:
        session.close()

def get_file_size(file_path: str) -> str:
    if os.path.exists(file_path):
        size_in_bytes = os.path.getsize(file_path)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_in_bytes < 1024.0:
                break
            size_in_bytes /= 1024.0
        return f"{size_in_bytes:.2f} {unit}"
    else:
        return "File not found"

@st.cache_resource
def get_vectorizer_and_matrix() -> Tuple[Optional[TfidfVectorizer], Optional[any], Optional[List[Case]]]:
    try:
        inspector = inspect(engine)
        exists = inspector.has_table('cases')
        logging.info(f"'cases' 테이블 존재 여부: {exists}")
        
        if not exists:
            logging.info("데이터베이스 다운로드 시작")
            st.write("잠시만 기다려 주세요. DB를 다운로드 하고 있습니다.")
            download_db()

        file_size = get_file_size(DB_FILE)
        logging.info(f"데이터베이스 파일 크기: {file_size}")

        exists = inspector.has_table('cases')
        if exists:
            logging.info(f"테이블이 존재합니다. 데이터 로드 시작.")
            cases = load_cases()
            if not cases:
                logging.error("케이스 데이터가 비어 있습니다.")
                return None, None, None
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([case.summary for case in cases if case.summary])
            return vectorizer, tfidf_matrix, cases
        else:
            logging.error(f"DB에 여전히 데이터가 존재하지 않습니다. 파일 크기: {file_size}")
            return None, None, None
    except Exception as e:
        logging.error(f"get_vectorizer_and_matrix 함수에서 오류 발생: {str(e)}")
        return None, None, None

def local_css():
    st.markdown("""
    <style>
    body {
        font-family: Arial, sans-serif;
        line-height: 1.6;
        color: #333;
    }
    .legal-term {
        color: #007bff;
        cursor: help;
    }
    .tooltip-inner {
        max-width: 300px;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

def highlight_legal_terms(text: str) -> str:
    terms = get_legal_terms()
    for term, explanation in terms.items():
        pattern = r'\b' + re.escape(term) + r'\b'
        replacement = f'<span class="legal-term" title="{explanation}">{term}</span>'
        text = re.sub(pattern, replacement, text)
    return text

def show_main_page():
    st.title("AI 기반 맞춤형 판례 검색 서비스")
    st.write("당신의 상황에 가장 적합한 판례를 찾아드립니다")

    st.image("static/photo.png", width=200)

    if st.button("바로 시작"):
        st.session_state.page = "search"
    else:
        st.write("시작하려면 '바로 시작' 버튼을 클릭하세요.")

def show_search_page():
    st.title("법률 판례 검색")

    st.sidebar.title("법률 분야 선택")
    legal_fields = ['민사', '가사', '형사A(생활형)', '형사B(일반형)', '행정', '기업', '근로자', '특허/저작권', '금융조세', '개인정보/ict', '잘모르겠습니다']
    selected_fields = st.sidebar.multiselect("법률 분야를 선택하세요:", legal_fields)

    st.header("상황 설명")
    st.write("아래 가이드라인을 참고하여 귀하의 법률 상황을 자세히 설명해주세요.")

    st.subheader("작성 가이드라인")
    st.markdown("""
    1. 사건의 발생 시기와 장소를 명시해주세요.
    2. 관련된 사람들의 관계를 설명해주세요. (예: 고용주-직원, 판매자-구매자)
    3. 사건의 경과를 시간 순서대로 설명해주세요.
    4. 문제가 되는 행위나 상황을 구체적으로 설명해주세요.
    5. 현재 상황과 귀하가 알고 싶은 법률적 문제를 명확히 해주세요.
    6. 분야를 제한하면 더욱 빠르게 검색할 수 있고, 더 정확한 정보가 나옵니다.
    """)

    st.subheader("예시")
    st.write("""
    2023년 3월 1일, 서울시 강남구의 한 아파트를 2년 계약으로 월세 100만원에 임대했습니다. 
    계약 당시 집주인과 구두로 2년 후 재계약 시 월세를 5% 이상 올리지 않기로 약속했습니다. 
    그러나 계약 만료 3개월 전인 2024년 12월, 집주인이 갑자기 월세를 150만원으로 50% 인상하겠다고 통보했습니다. 
    이를 거부하면 퇴거해야 한다고 합니다. 구두 약속은 법적 효력이 있는지, 
    그리고 이런 과도한 월세 인상이 법적으로 가능한지 알고 싶습니다.
    """)

    user_input = st.text_area("상황 설명:", height=200)

    if st.button("검색"):
        if user_input and len(user_input) > 3:
            st.session_state.user_input = user_input
            st.session_state.selected_fields = selected_fields
            st.session_state.page = "result"
        else:
            st.error("검색어가 없거나 너무 짧습니다")

def show_result_page():
    st.title("판례 검색 결과")

    user_input = st.session_state.user_input
    selected_fields = st.session_state.selected_fields

    with st.spinner('판례를 검색 중입니다...'):
        result = get_vectorizer_and_matrix()
        if result is None or len(result) != 3:
            st.error("데이터를 불러오는 데 실패했습니다. 관리자에게 문의해주세요.")
            return
        
        vectorizer, tfidf_matrix, cases = result

        if not selected_fields or '잘모르겠습니다' in selected_fields:
            filtered_cases = cases
            filtered_tfidf_matrix = tfidf_matrix
        else:
            filtered_cases = [case for case in cases if case.class_name in selected_fields]
            filtered_tfidf_matrix = vectorizer.transform([case.summary for case in filtered_cases if case.summary])
        
        if not filtered_cases:
            st.warning("선택한 법률 분야에 해당하는 판례가 없습니다. 다른 분야를 선택해주세요.")
            return

        user_vector = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, filtered_tfidf_matrix)
        most_similar_idx = similarities.argmax()
        case = filtered_cases[most_similar_idx]

    st.subheader("요약")
    st.markdown(highlight_legal_terms(case.summary), unsafe_allow_html=True)
    
    if case.jdgmnQuestion:
        st.subheader("핵심 질문")
        st.markdown(highlight_legal_terms(case.jdgmnQuestion), unsafe_allow_html=True)
    
    if case.jdgmnAnswer:
        st.subheader("답변")
        st.markdown(highlight_legal_terms(case.jdgmnAnswer), unsafe_allow_html=True)

    if st.button("다시 검색하기"):
        st.session_state.page = "search"

def main():
    local_css()

    if 'page' not in st.session_state:
        st.session_state.page = "main"

    if st.session_state.page == "main":
        show_main_page()
    elif st.session_state.page == "search":
        show_search_page()
    elif st.session_state.page == "result":
        show_result_page()

if __name__ == '__main__':
    main()