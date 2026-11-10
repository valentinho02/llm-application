# rag_study

RAG(Retrieval-Augmented Generation)의 개념을 학습하고 실습 결과를 기록하는 저장소입니다.

## 학습 주제

- 문서 수집과 전처리
- 텍스트 분할(Chunking)
- 임베딩과 벡터 검색
- 검색 결과를 활용한 답변 생성
- 검색 품질과 답변 품질 평가

## 폴더 구성

- `notes/`: 개념 정리와 학습 기록
- `experiments/`: 실습 코드와 실험 기록

## 진행 방식

학습한 내용을 notes에 정리하고, 실습 코드와 결과를 experiments에 기록합니다.

## Python 환경

Python 3.13 기반 프로젝트 전용 가상환경 `.venv`를 사용합니다.

```bash
source .venv/bin/activate
python embedding.py
```

VS Code에서는 `Python: Select Interpreter`에서 `.venv/bin/python`을 선택합니다.
노트북도 이 가상환경을 커널로 선택하면 됩니다.

### 설치된 학습 도구

- API·임베딩: OpenAI SDK, python-dotenv, tiktoken, sentence-transformers
- RAG 프레임워크: LangChain, LlamaIndex 및 OpenAI 연동 패키지
- 검색: Chroma, FAISS, Qdrant 클라이언트, BM25
- 문서 처리: pypdf, PyMuPDF, python-docx, Beautiful Soup, lxml, openpyxl
- 데이터 분석: NumPy, pandas, SciPy, scikit-learn, matplotlib, seaborn
- 실습·평가: JupyterLab, ipykernel, datasets, Ragas, Streamlit, pytest

JupyterLab은 `jupyter lab`으로 실행합니다. 로컬 임베딩 모델의 가중치는
모델을 선택한 뒤 처음 사용할 때 별도로 다운로드됩니다.
OpenAI API를 사용하는 코드는 `.env`의 `OPENAI_API_KEY` 설정이 필요합니다.

### 환경 재설치

`requirements.txt`에는 학습용 패키지 목록을, `requirements-lock.txt`에는
설치 확인된 전체 패키지 버전을 기록합니다. 잠금 목록은 macOS Apple Silicon과
Python 3.13 환경을 기준으로 합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

다른 운영체제나 Python 버전에서는 `requirements.txt`를 사용해 의존성을 다시 해결합니다.
