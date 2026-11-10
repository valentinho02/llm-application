import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "API 키를 찾을 수 없습니다."

llm = ChatOpenAI(model="gpt-4o-mini")

response = llm.invoke("안녕! 너는 누구야?")

print(response.content)

"""
result : 

inho@jeong-inhoui-MacBookPro llm-application % python3 llm_test.py 
안녕하세요! 저는 인공지능 언어 모델인 ChatGPT입니다. 질문에 답하거나, 정보를 제공하거나, 다양한 주제에 대해 대화할 수 있어요. 도움이 필요하시면 말씀해 주세요!
"""