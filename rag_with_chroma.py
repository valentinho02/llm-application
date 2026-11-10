"""
1. 문서 내용을 읽고
2. 쪼개고(토큰 수 초과로 답변을 생성하지 못할 수 있고, 문서가 길면 답변 생성이 오래 걸리기 때문)
3. 임베딩해서 벡터 데이터베이스에 저장
4. 질문이 있을 때, 벡터 데이터베이스에 유사도 검색
5. 유사도 검색으로 가져온 문서를 llm에 질문과 함께 전달
"""

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic import hub
from langchain_classic.chains import RetrievalQA

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 200, # 겹치는 양
)

loader = Docx2txtLoader('./tax.docx')
document_list = loader.load_and_split(text_splitter=text_splitter)
print(len(document_list))

"""
recursively splitter : 리스트로 구분자를 주어 여러가지로 구분할 수 있다
"""
load_dotenv()
embedding = OpenAIEmbeddings(model='text-embedding-3-large')

#database = Chroma.from_documents(documents = document_list, embedding=embedding, collection_name='chroma-tax', persist_directory="./chroma")
database = Chroma(collection_name='chroma-tax', persist_directory="./chroma", embedding_function=embedding) # 생성된 파일 가져오기
query = '연봉 5천만원인 직장인의 소득세는 얼마인가요?' 
retreived_docs = database.similarity_search(query, k=2) #default 는 4

print(retreived_docs)

llm = ChatOpenAI(model = 'gpt-4o')

prompt = f"""[Identity]
-당신은 최고의 한국 소득세 전문가입니다
-[Context]를 참고해서 사용자의 질문에 답변해주세요

[Context]
{retreived_docs}

Question: {query}
"""

ai_message = llm.invoke(prompt)
print(ai_message.content)

from langsmith import Client

client = Client()
prompt = client.pull_prompt(
    "rlm/rag-prompt",
    dangerously_pull_public_prompt=True
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever = database.as_retriever(),
    chain_type_kwargs = {"prompt" : prompt}
)
aimessage = qa_chain({"query" : query})
print(ai_message)