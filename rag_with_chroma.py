"""
1. 문서 내용을 읽고
2. 쪼개고(토큰 수 초과로 답변을 생성하지 못할 수 있고, 문서가 길면 답변 생성이 오래 걸리기 때문)
3. 임베딩해서 벡터 데이터베이스에 저장
4. 질문이 있을 때, 벡터 데이터베이스에 유사도 검색
5. 유사도 검색으로 가져온 문서를 llm에 질문과 함께 전달
"""

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 200, # 겹치는 양
)

loader = Docx2txtLoader('./tax.docx')
document = loader.load_and_split(text_splitter=text_splitter)
print(len(document))

"""
recursively splitter : 리스트로 구분자를 주어 여러가지로 구분할 수 있다
"""
