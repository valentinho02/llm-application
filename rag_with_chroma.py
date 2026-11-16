
"""
1. 문서 내용을 읽고
2. 쪼개고
3. 임베딩해서 벡터 데이터베이스에 저장
4. 질문이 있을 때, 벡터 데이터베이스에 유사도 검색
5. 유사도 검색으로 가져온 문서를 LLM에 질문과 함께 전달


6. 키워드 사전을 이용한 Query Expansion
7. LLM을 이용한 Query Rewriting
8. 검색 결과 확인 및 근거 기반 답변 생성
"""

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. 문서 로딩 및 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
)

loader = Docx2txtLoader('./tax_with_markdown.docx')
document_list = loader.load_and_split(
    text_splitter=text_splitter
)

print(f"문서 분할 완료: {len(document_list)}개 청크")


# 2. Embedding 및 Chroma DB
load_dotenv()

embedding = OpenAIEmbeddings(
    model='text-embedding-3-large'
)


def initialize_database():
    persist_directory = "./chroma"
    collection_name = "chroma-tax"

    # 기존 Chroma DB 불러오기
    # DB가 없으면 새 컬렉션이 생성됨
    database = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    # 저장된 문서가 없으면 임베딩 후 Chroma에 저장
    if database._collection.count() == 0:
        print("새로운 Chroma DB 생성 중...")

        database.add_documents(document_list)

    print(f"저장된 청크 수: {database._collection.count()}")

    return database


# document_list와 embedding을 생성한 후 함수 실행
database = initialize_database()


# 3. 사용자 질문 및 LLM
query = '연봉 5천만원인 직장인의 소득세는 얼마인가요?'

llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0
)


# 4. Query Expansion - 키워드 사전
KEYWORD_DICT = {
    "직장인": ["근로소득", "총급여액", "근로소득공제"],
    "회사원": ["근로소득", "급여", "근로소득공제"],
    "연봉": ["총급여액", "근로소득금액", "근로소득공제"],
    "월급": ["근로소득", "원천징수", "간이세액표"],
    "소득세": ["종합소득과세표준", "세율", "산출세액", "제55조"],
    "프리랜서": ["사업소득", "원천징수", "종합소득"],
}


def expand_query(query: str) -> str:
    """사용자 질문에 관련 법률 검색어 추가"""

    expanded_terms = []

    for keyword, terms in KEYWORD_DICT.items():
        if keyword in query:
            expanded_terms.extend(terms)

    # 중복 키워드 제거
    expanded_terms = list(dict.fromkeys(expanded_terms))

    if not expanded_terms:
        return query

    return (
        f"{query}\n"
        f"관련 법률 검색어: {', '.join(expanded_terms)}"
    )


expanded_query = expand_query(query)

print(f"expanded_query: {expanded_query}")


# 5. Query Rewriting - LLM으로 검색 질의 재작성
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

# LCEL: Prompt -> LLM -> Output Parser
rewrite_chain = rewrite_prompt | llm | StrOutputParser()

rewritten_query = rewrite_chain.invoke({
    "dictionary": str(KEYWORD_DICT),
    "question": expanded_query
})

print(f"rewritten_query : {rewritten_query}")


# 6. Retrieval - 재작성된 질문으로 Chroma 검색
retrieved_docs = database.similarity_search(
    rewritten_query,
    k=10,
)



# 7. 검색 품질 확인
for article in ["제47조(근로소득공제)", "제55조(세율)"]:
    found = any(
        article in doc.page_content
        for doc in retrieved_docs
    )

    print(f"{article}: {found}")


# 8. 검색 결과를 Context로 변환
context = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)


# 9. Answer Generation - 검색 문서를 기반으로 답변 생성
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
    "question": rewritten_query,
})

print(f"최종답변 {answer}")
"""

문서 분할 완료: 225개 청크
새로운 Chroma DB 생성 중...
저장된 청크 수: 225
expanded_query: 연봉 5천만원인 직장인의 소득세는 얼마인가요?
관련 법률 검색어: 근로소득, 총급여액, 근로소득공제, 근로소득금액, 종합소득과세표준, 세율, 산출세액, 제55조
rewritten_query : 총급여액 5천만원 근로소득자의 종합소득과세표준 및 산출세액 계산 방법
제47조(근로소득공제): True
제55조(세율): True
최종답변 총급여액 5천만원인 근로소득자의 종합소득과세표준 및 산출세액을 계산하는 방법은 다음과 같습니다.

1. **근로소득금액 계산**:
   - 총급여액에서 근로소득공제를 차감하여 근로소득금액을 계산합니다.
   - 근로소득공제는 제47조에 따라 계산됩니다. 총급여액이 5천만원인 경우, 근로소득공제는 다음과 같이 계산됩니다:
     - 총급여액이 5천만원이므로, 근로소득공제는 2천만원을 초과하지 않으므로, 공제액은 2천만원입니다.
   - 따라서, 근로소득금액 = 총급여액 - 근로소득공제 = 5천만원 - 2천만원 = 3천만원

2. **종합소득과세표준 계산**:
   - 종합소득과세표준은 종합소득금액에서 종합소득공제를 차감한 금액입니다.
   - 종합소득공제에 대한 정보가 제공되지 않았으므로, 종합소득과세표준을 정확히 계산할 수 없습니다. 종합소득공제에 대한 정보가 필요합니다.

3. **종합소득산출세액 계산**:
   - 종합소득과세표준에 제55조에 따른 세율을 적용하여 종합소득산출세액을 계산합니다.
   - 종합소득과세표준이 3천만원이라고 가정하면, 세율은 1,400만원 초과 5,000만원 이하 구간에 해당하므로, 84만원 + (1,400만원을 초과하는 금액의 15퍼센트)입니다.
   - 따라서, 종합소득산출세액 = 84만원 + (3천만원 - 1,400만원) × 15% = 84만원 + 1,600만원 × 15% = 84만원 + 240만원 = 324만원

정확한 종합소득과세표준을 계산하기 위해서는 종합소득공제에 대한 추가 정보가 필요합니다.
inho@jeong-inhoui-MacBookPro llm-application % 
"""