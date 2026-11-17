
import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic import hub
from langchain_classic.chains import RetrievalQA
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_ai_message(user_message):

    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'tax-vector-index'
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
        namespace="tax-rag-v2"
    )

    llm = ChatOpenAI(model='gpt-4o')
    client = Client()

    prompt = client.pull_prompt(
        "rlm/rag-prompt",
        dangerously_pull_public_prompt=True
    )

    retriever = database.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    dictionary = {
        "직장인": ["근로소득", "총급여액", "근로소득공제"],
        "회사원": ["근로소득", "급여", "근로소득공제"],
        "연봉": ["총급여액", "근로소득금액", "근로소득공제"],
        "월급": ["근로소득", "원천징수", "간이세액표"],
        "소득세": ["종합소득과세표준", "세율", "산출세액", "제55조"],
        "프리랜서": ["사업소득", "원천징수", "종합소득"],
    }

    prompt = ChatPromptTemplate.from_template("""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그럴 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        질문 : {question}
    """).partial(dictionary=str(dictionary))

    dictionary_chain = prompt | llm | StrOutputParser()

    tax_chain = {"query": dictionary_chain} | qa_chain

    ai_message = tax_chain.invoke({"question": user_message})

    return ai_message["result"]


st.set_page_config(page_title="소득세 챗봇")
st.title("소득세 챗봇")
st.caption("소득세에 관련된 모든 것을 답변해드립니다.")

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(
    placeholder="소득세에 관련된 궁금한 내용들을 물어보세요!"
):
    with st.chat_message("user"):
        st.write(user_question)

    st.session_state.message_list.append({
        "role": "user",
        "content": user_question
    })

    with st.spinner("답변 생성중"):
        ai_message = get_ai_message(user_question)

        with st.chat_message("ai"):
            st.write(ai_message)

        st.session_state.message_list.append({
            "role": "ai",
            "content": ai_message
        })
