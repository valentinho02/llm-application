
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()


def get_ai_message(user_message):

    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    database = PineconeVectorStore.from_existing_index(
        index_name="tax-vector-index",
        embedding=embedding,
        namespace="tax-rag-v2"
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    dictionary = {
        "직장인": ["근로소득", "총급여액", "근로소득공제"],
        "회사원": ["근로소득", "급여", "근로소득공제"],
        "연봉": ["총급여액", "근로소득금액", "근로소득공제"],
        "월급": ["근로소득", "원천징수", "간이세액표"],
        "소득세": ["종합소득과세표준", "세율", "산출세액", "제55조"],
        "프리랜서": ["사업소득", "원천징수", "종합소득"],
    }

    # 1. Query Rewriting
    rewrite_prompt = ChatPromptTemplate.from_template("""
당신은 한국 소득세법 RAG 시스템의 검색 질의 재작성 전문가입니다.

사용자의 질문을 법률 문서 검색에 적합하도록 재작성하세요.

[법률 용어 사전]
{dictionary}

[규칙]
1. 사용자의 원래 질문 의도를 유지하세요.
2. 질문에 포함된 금액을 반드시 유지하세요.
3. 소득세 계산 질문을 과세표준 계산 질문으로 바꾸지 마세요.
4. 직장인 또는 근로소득자의 소득세 계산 질문에는
   다음 용어를 포함하세요.
   - 총급여액
   - 근로소득공제
   - 제47조
   - 종합소득과세표준
   - 소득세율
   - 제55조
5. 사용자가 제공하지 않은 조건을 추가하지 마세요.
6. 답변하지 말고 검색 질의만 반환하세요.

[질문]
{question}
""").partial(dictionary=str(dictionary))

    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    rewritten_query = rewrite_chain.invoke({
        "question": user_message
    })

    print(f"\n원본 질문: {user_message}")
    print(f"재작성 질문: {rewritten_query}")

    # 2. Retrieval
    retriever = database.as_retriever(
        search_kwargs={"k": 10}
    )

    retrieved_docs = retriever.invoke(rewritten_query)

    print(f"\n검색된 문서 수: {len(retrieved_docs)}")

    for article in ["제47조(근로소득공제)", "제55조(세율)"]:
        found = any(
            article in doc.page_content
            for doc in retrieved_docs
        )
        print(f"{article}: {found}")

    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n===== 검색 문서 {i} =====")
        print(doc.page_content[:500])

    # 3. Answer Generation
    answer_prompt = ChatPromptTemplate.from_template("""
당신은 한국 소득세법 문서를 기반으로 답변하는 AI입니다.

반드시 제공된 Context를 검토하여 답변하세요.

[규칙]
1. 검색된 문서에서 질문과 관련된 법률 조항을 찾아 활용하세요.
2. 근로소득공제 계산에는 제47조를 참고하세요.
3. 종합소득세율 계산에는 제55조를 참고하세요.
4. 총급여액과 종합소득과세표준을 구분하세요.
5. Context에 근거가 없는 금액이나 세율은 만들어내지 마세요.
6. 계산 가능한 부분은 계산 과정과 근거 조항을 제시하세요.
7. 계산에 필요한 정보가 부족하다면 부족한 항목을 설명하세요.

[Context]
{context}

[Question]
{question}

[Answer]
""")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": answer_prompt},
        return_source_documents=True
    )

    # 검색에는 재작성 질의를 사용하고,
    # 답변 생성에는 원본 질문을 전달
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    answer_chain = (
        answer_prompt
        | llm
        | StrOutputParser()
    )

    answer = answer_chain.invoke({
        "context": context,
        "question": user_message
    })

    return answer
