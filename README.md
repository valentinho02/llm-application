# 소득세법 AI 챗봇

한국 소득세법 문서에서 질문과 관련된 조항을 검색하고, 검색한 내용을 바탕으로 답변을 생성하는 RAG 학습 프로젝트입니다. Chroma로 문서 검색을 실험한 뒤 Pinecone을 연결하고, 앞선 대화를 활용해 후속 질문을 처리하는 챗봇을 구현했습니다.

## 문제 해결

### 1. Word 문서의 세율표를 검색에 포함하기

원본 Word 문서를 `Docx2txtLoader`로 읽었을 때 세율표가 추출되지 않아, 검색 결과에 과세표준 구간과 해당 세율의 관계가 빠졌습니다. 표를 Markdown 텍스트로 변환해 `tax_with_markdown.docx`에 담았습니다. 전처리한 문서를 `Docx2txtLoader`로 읽고 `RecursiveCharacterTextSplitter`로 분할한 뒤 임베딩했습니다. 두 검색 실험 코드는 청크 크기 1,500자와 중첩 200자를 사용합니다.

### 2. 앞선 대화를 반영해 후속 질문 검색하기

‘연봉 5천만 원인 직장인의 소득세는 얼마인가요?’ 다음에 ‘그럼 1억 원은?’이라고 물으면, 후속 질문만으로는 금액이 무엇을 뜻하는지 파악하기 어려웠습니다. `llm.py`에서 `ChatMessageHistory`와 `RunnableWithMessageHistory`를 연결해 대화 기록을 다음 요청에 전달했습니다. `create_history_aware_retriever`는 그 기록을 참고해 생략된 맥락을 채운 검색 질문을 만들고, Pinecone에서 관련 조항을 찾습니다.

개발 중 확인한 데모에서 개선 전에는 ‘그럼 1억 원은?’의 맥락을 놓쳤고, 개선 후에는 ‘그럼 3천만 원은?’을 직전 연봉 질문의 후속 질문으로 처리했습니다. 대화 기록은 현재 프로세스의 메모리에 저장되며 데모 호출은 고정 세션 ID를 사용합니다.

## 처리 흐름

1. **문서 색인**은 Word 표의 Markdown 변환부터 문서 분할, OpenAI `text-embedding-3-large` 임베딩, 벡터 저장 순서로 진행합니다.
2. **챗봇 검색**은 법률 용어 사전으로 질문을 다듬고, 대화 기록을 참고해 독립적인 검색 질문을 만든 뒤 Pinecone에서 관련 청크 4개를 가져옵니다.
3. **답변 생성**은 검색된 조항과 대화 기록을 GPT-4o에 전달하고, 답변을 스트리밍합니다.

| 파일 | 역할 |
| --- | --- |
| [rag_with_chroma.py](rag_with_chroma.py) | Chroma 기반 단일 질문 검색 실험 |
| [rag_with_pinecone.py](rag_with_pinecone.py) | Pinecone 색인 준비와 단일 질문 검색 실험 |
| [llm.py](llm.py) | 대화 기록을 활용하는 Pinecone 검색과 답변 스트리밍 |
| [config.py](config.py) | 답변 형식을 보여 주는 프롬프트 예시 |

## 검색 실험 실행

저장소 루트에서 Python 3.13 가상환경을 만든 뒤 의존성을 설치합니다.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install langchain-classic langchain-pinecone
```

프로젝트 루트의 `.env`에 `OPENAI_API_KEY`를 설정합니다. Pinecone을 사용한다면 `PINECONE_API_KEY`도 설정합니다.

```bash
python rag_with_chroma.py
# Pinecone 실험 및 챗봇용 색인 준비
python rag_with_pinecone.py
```

Pinecone 실험과 챗봇은 `tax-vector-index`의 `tax-rag-v2` 네임스페이스를 공유합니다. 기존 인덱스를 사용할 때는 임베딩 모델에 맞춰 벡터 차원이 3,072인지 확인합니다.

## 데모의 한계

후속 질문 처리 데모는 대화 맥락이 이어지는지 확인하는 데 사용했습니다. 화면의 예시 답변에는 연봉을 과세표준으로 그대로 적용한 계산이 포함되어 있습니다. 세액 계산의 정확도는 별도 검증이 필요합니다.
