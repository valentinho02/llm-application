
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

from llm import get_ai_message

load_dotenv()



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
