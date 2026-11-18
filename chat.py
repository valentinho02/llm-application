
import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response

load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="소득세 AI 챗봇",
    page_icon="💰",
    layout="centered"
)

# CSS 스타일
st.markdown("""
<style>
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    h1 {
        text-align: center;
        font-weight: 750;
    }

    .subtitle {
        text-align: center;
        color: #888;
        font-size: 15px;
        margin-bottom: 2rem;
    }

    [data-testid="stChatMessage"] {
        border-radius: 15px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }

    [data-testid="stChatInput"] {
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)


# 제목
st.title("💰 소득세 AI 챗봇")
st.markdown(
    '<p class="subtitle">소득세법을 기반으로 세금 관련 질문에 답변해드립니다.</p>',
    unsafe_allow_html=True
)

st.divider()


# Streamlit 세션 상태 초기화
if "message_list" not in st.session_state:
    st.session_state.message_list = []


# 사이드바
with st.sidebar:
    st.header("💰 소득세 챗봇")
    st.caption("소득세법 기반 RAG 시스템")

    st.divider()

    st.subheader("대화 관리")

    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.message_list = []
        st.rerun()

    st.divider()
    st.caption("Powered by LangChain · Pinecone · OpenAI")


# 기존 대화 기록 표시
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 첫 화면 안내
if not st.session_state.message_list:
    st.info("👋 안녕하세요! 소득세에 관해 궁금한 내용을 질문해 주세요.")

    st.markdown("""
    **예시 질문**
    - 연봉 5천만 원 직장인의 소득세는 얼마인가요?
    - 근로소득공제란 무엇인가요?
    - 종합소득세 세율은 어떻게 되나요?
    """)


# 사용자 입력
if user_question := st.chat_input(
    "소득세에 관련된 궁금한 내용을 물어보세요!"
):

    # 사용자 메시지 화면 출력
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_question)

    # 사용자 메시지 기록 저장
    st.session_state.message_list.append({
        "role": "user",
        "content": user_question
    })

    # AI 답변 생성
    with st.chat_message("assistant", avatar="🤖"):
        try:
            # get_ai_response()는 스트리밍 응답을 반환
            ai_response = get_ai_response(user_question)

            # 실시간 스트리밍 출력
            ai_message = st.write_stream(ai_response)

            # AI 답변 기록 저장
            st.session_state.message_list.append({
                "role": "assistant",
                "content": ai_message
            })

        except Exception as e:
            st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
