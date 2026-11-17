import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
)

loader = Docx2txtLoader("./tax_with_markdown.docx")
document_list = loader.load_and_split(text_splitter=text_splitter)

print(f"문서 분할 완료: {len(document_list)}개 청크")

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o", temperature=0)


def initialize_database():
    index_name = "tax-vector-index"
    namespace = "tax-rag-v2"

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    database = PineconeVectorStore(
        index_name=index_name,
        embedding=embedding,
        namespace=namespace,
    )

    if database.similarity_search("소득세", k=1) == []:
        print("새로운 Pinecone DB 생성 중...")
        database.add_documents(document_list)

    return database


database = initialize_database()

query = "연봉 5천만원인 직장인의 소득세는 얼마인가요?"

KEYWORD_DICT = {
    "직장인": ["근로소득", "총급여액", "근로소득공제"],
    "회사원": ["근로소득", "급여", "근로소득공제"],
    "연봉": ["총급여액", "근로소득금액", "근로소득공제"],
    "월급": ["근로소득", "원천징수", "간이세액표"],
    "소득세": ["종합소득과세표준", "세율", "산출세액", "제55조"],
    "프리랜서": ["사업소득", "원천징수", "종합소득"],
}


def expand_query(query: str) -> str:
    expanded_terms = []

    for keyword, terms in KEYWORD_DICT.items():
        if keyword in query:
            expanded_terms.extend(terms)

    expanded_terms = list(dict.fromkeys(expanded_terms))

    if not expanded_terms:
        return query

    return (
        f"{query}\n"
        f"관련 법률 검색어: {', '.join(expanded_terms)}"
    )


expanded_query = expand_query(query)
print(f"expanded_query: {expanded_query}")

rewrite_prompt = ChatPromptTemplate.from_template("""
당신은 한국 소득세법 문서 검색을 돕는 검색어 생성기입니다.

사용자 질문의 의미를 유지하면서 법률 문서 검색에
적합한 질의로 재작성하세요.

[법률 용어 사전]
{dictionary}

[규칙]
1. 질문의 의미와 금액을 유지하세요.
2. 관련 법률 용어를 활용하세요.
3. 직장인을 무조건 거주자로 단정하지 마세요.
4. 사용자가 제공하지 않은 조건을 만들지 마세요.
5. 답변하지 말고 검색 질의만 반환하세요.

[질문]
{question}
""")

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

rewritten_query = rewrite_chain.invoke({
    "dictionary": str(KEYWORD_DICT),
    "question": expanded_query,
})

print(f"rewritten_query: {rewritten_query}")

retrieved_docs = database.similarity_search(
    rewritten_query,
    k=10,
)

for article in ["제47조(근로소득공제)", "제55조(세율)"]:
    found = any(
        article in doc.page_content
        for doc in retrieved_docs
    )
    print(f"{article}: {found}")

context = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)

answer_prompt = ChatPromptTemplate.from_template("""
당신은 한국 소득세법 문서 기반 질의응답 AI입니다.

[규칙]
1. 반드시 제공된 Context를 근거로 답변하세요.
2. Context에 없는 세율이나 공제율을 만들어내지 마세요.
3. 근로소득금액과 과세표준을 구분하세요.
4. 계산에 필요한 정보가 부족하면 부족하다고 설명하세요.
5. 계산이 가능하다면 계산 과정과 근거 조항을 제시하세요.
6. 정확한 금액을 계산할 수 없다면 임의로 계산하지 마세요.

[Context]
{context}

[사용자 질문]
{question}
""")

answer_chain = answer_prompt | llm | StrOutputParser()

answer = answer_chain.invoke({
    "context": context,
    "question": query,
})

print(f"최종답변: {answer}")