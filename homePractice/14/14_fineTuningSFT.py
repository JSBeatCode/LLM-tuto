"""
LangGraph를 이용한 Corrective RAG (CRAG)
- ChromaDB는 스크립트와 동일한 위치에 저장 (중복 생성 방지)
- PDF 파일(교재.pdf)도 동일한 위치에 있어야 함
- .env 파일에 OPENAI_API_KEY, TAVILY_API_KEY 설정 필요
"""

import os
import warnings
import sys


from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from typing import List
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from pprint import pprint

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv(override=True)

# ── 경로 기준: 이 .py 파일이 있는 디렉터리 ──────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_crag_db")
PDF_PATH   = os.path.join(BASE_DIR, "교재.pdf")

# ── API 키 확인 ───────────────────────────────────────────────────────────────
def check_env():
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if missing:
        print(f"[ERROR] .env에 다음 키가 없습니다: {', '.join(missing)}")
        sys.exit(1)
    os.environ.setdefault("USER_AGENT", "MyCustomAgent")
    print("API 키 확인 완료.")

# ── LLM / Tavily 초기화 ───────────────────────────────────────────────────────
def build_llm_and_tools():

    llm   = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024, temperature=0.1)
    tavily = TavilySearchResults(max_results=5)
    return llm, tavily

# ── PDF 로드 & 청킹 ───────────────────────────────────────────────────────────
# 교재.pdf
#     │
#     ▼
# PDF 존재 확인
#     │
#     ▼
# PDF 읽기
#     │
#     ▼
# 페이지별(Document) 생성
#     │
#     ▼
# 모든 페이지를 하나의 큰 문서로 합침
#     │
#     ▼
# 1000글자씩 자름
# (100글자는 겹치게)
#     │
#     ▼
# Chunk 리스트 반환
def load_pdf():

    if not os.path.exists(PDF_PATH):
        print(f"[ERROR] PDF 파일을 찾을 수 없습니다: {PDF_PATH}")
        sys.exit(1)

    print("PDF 로드 중...")
    try:
        # PyPDFLoader: LangChain이 제공하는 PDF 읽기 도구
        loader = PyPDFLoader(PDF_PATH, password="")
        # 이 코드가 실제로 PDF를 읽습니다.
        # pages = [
        # Document(1page),
        # Document(2page),
        # Document(3page)
        # ]
        pages  = loader.load()
    except Exception:
        loader = PyPDFLoader(PDF_PATH)
        pages  = loader.load()

    # Document는 LangChain에서 사용하는 문서 객체
    # page_content="" -> 내용이 없는 문서를 하나 만든 것입니다.
    corpus = Document(page_content="")
    for page in pages:
        # 페이지 합치기 (1page += 2page)
        # +"\n---\n" -> 페이지 구분선입니다.
        corpus.page_content += page.page_content + "\n---\n"
    print(f"  전체 텍스트 길이: {len(corpus.page_content)}")

    # RAG의 핵심
    # chunk_size=1000 
    #   -> 1000글자 정도마다 잘라라
    #     5000글자
    #     ↓
    #     1000
    #     1000
    #     1000
    #     1000
    #     1000
    # chunk_overlap=100 
    #   -> 만약 ABCDEFGHIJ 를 5글자씩 자르면 ABCDE FGHIJ 
    #   -> 중간 의미가 끊길 수 있습니다. 그래서 100글자를 겹칩니다.
    #   -> 예를 들어 
    #     Chunk1: 1234567890, Chunk2: 67890ABCDE 
    #     처럼 뒤의 67890 를 다음 Chunk에도 넣습니다. 
    #     이렇게 하면 문맥(Context)이 자연스럽게 이어집니다.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # 이 코드가 실제로 문서를 잘라 줍니다.
    # [corpus] -> 대괄호를 사용하는 이유는 split_documents()가 Document 객체 하나가 아니라 Document들의 리스트를 입력으로 받기 때문입니다.
    chunks   = splitter.split_documents([corpus])
    print(f"  청크 수: {len(chunks)}")

    # chunks = [
    # Document,
    # Document,
    # Document,
    # ...
    # ]
    return chunks

# ── ChromaDB 구축 (중복 방지) ─────────────────────────────────────────────────
# Chunk → Embedding → VectorDB → Retriever
def build_vectorstore(chunks=None):
    """
    PDF에서 만든 Chunk들을 ChromaDB(Vector Database)에 저장하고,
    나중에 질문이 들어오면 관련 문서를 검색할 수 있는 Retriever를 생성한다.

    이미 ChromaDB가 존재하면 새로 만들지 않고 그대로 재사용한다.
    """
    # Embedding 모델 생성
    # 강아지
    # ↓
    # [0.21,
    # -0.44,
    # 0.11,
    # ...
    # 1536개]
    # Embedding 후 이런 숫자 배열이 됩니다.
    # 이 숫자를 벡터(Vector) 라고 합니다.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"ChromaDB 로드 (기존): {CHROMA_DIR}")
        # 기존 ChromaDB를 메모리로 불러옵니다.
        vector_store = Chroma(
            persist_directory=CHROMA_DIR,
            # 기존 DB를 불러오는데 Embedding 모델이 또 필요한 이유: 
            # 질문도 벡터로 변환해야 하기 때문입니다.
            embedding_function=embeddings,
        )
    else:
        if chunks is None:
            chunks = load_pdf()
        print(f"ChromaDB 신규 생성: {CHROMA_DIR}")
        
        # 문장들을 AI가 검색(Semantic Search)할 수 있는 데이터베이스로 변환하여 저장하는 코드
        # 
        # Chroma.from_documents -> 만약 이 함수가 없다면 개발자가 직접
        # - Document를 하나씩 꺼내고
        # - Embedding API를 호출하고
        # - Vector를 생성하고
        # - DB에 저장하고
        # - 메타데이터를 관리하고
        # - 인덱스를 생성하고
        # - 파일로 저장하는
        # 모든 과정을 직접 구현해야 합니다.
        vector_store = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=CHROMA_DIR,
        )
        print("  ChromaDB 저장 완료.")
    # VectorDB
    # ↓
    # 검색 가능한 객체로 변환
    # search_kwargs={"k": 5} -> 가장 관련 있는 문서 5개만 가져와라
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever

# ── 1. Retrieval Grader ㄴ───────────────────────────────────────────────────────
# "Retriever가 가져온 문서가 정말 질문과 관련 있는지 GPT에게 다시 검사시키는 AI 심사관" 을 만드는 함수입니다.
# 질문
# 
# ↓
# 
# Retriever
# 
# ↓
# 
# 문서1
# 문서2
# 문서3
# 문서4
# 문서5
# 
# ↓
# 
# Retrieval Grader
# 
# ↓
# 
# 관련
# 관련
# 관련 없음
# 관련
# 관련 없음
# 
# ↓
# 
# GPT
def build_retrieval_grader(llm):

   # response -> 
   # {
   #     "grade":
   #     "이 문서는...
   #      Result : Not Relevant"
   # }
    def parse_relevance_score(response):
        return "No" if "not relevant" in response["grade"].lower() else "Yes"

    # 면접관에게 주는 채점 기준
    # 예시를 많이 넣은 이유 -> Few-shot Prompt
    system = (
        "당신은 주어진 [질문]과 검색된 [문서] 사이의 연관성을 판단해야 합니다.\n"
        "[질문]을 답변하기 위한 정보를 [문서]가 포함하고 있는지를 설명하세요.\n"
        "그 이후, [결과]에 'Relevant' 또는 'Not Relevant'를 출력하세요.\n"
        "다음은 예시 답변입니다.\n\n"
        "예시)\n"
        "[답변]: 이 질문은 공룡의 멸종 원인을 묻는 질문으로, 문서는 공룡과 관련이 없는 내용입니다. [결과]: Not Relevant\n\n"
        "[답변]: 이 질문은 프롬프트 엔지니어링의 장점을 묻는 질문인데, 문서는 프롬프트 엔지니어링에 대해 다루고 있습니다. [결과]: Relevant\n\n"
        "[답변]: 이 질문은 지구 온난화의 주요 원인을 묻는 질문으로, 문서는 중세 시대의 역사적 사건들에 대해 다루고 있습니다. [결과]: Not Relevant\n\n"
        "[답변]: 이 질문은 로마 제국의 멸망 원인을 묻는 질문으로, 문서는 로마 제국의 경제 구조와 사회적 변화에 대해 다루고 있습니다. [결과]: Relevant\n\n"
        "[답변]: 이 질문은 현대 인공지능의 윤리적 문제를 묻는 질문으로, 문서는 최신 영화 리뷰와 평점에 대해 다루고 있습니다. [결과]: Not Relevant\n\n"
        "시작\n\n"
        "[답변]:"
    )
    # GPT에게 보낼 Prompt를 만드는 것입니다.
    grade_prompt = ChatPromptTemplate([
        ("system", system),
        ("human", "[문서]: \n\n {document} \n\n [질문]: {question} [답변]: 이 질문은 \n"),
    ])
    # 나중에 retrieval_grader.invoke(...)를 호출할 때 입력이 들어옵니다.
    # 예를 들어, 이 함수는 단지 평가기(객체)만 만들어서 반환합니다.
    # 아래와 같이 입력 데이터가 옴
    # {
    #     "question": "RAG란?",
    #     "document": "RAG는 Retrieval-Augmented Generation이다."
    # }
    retrieval_grader = (
        # 입력에서 필요한 값만 꺼냅니다. 거의 아래와 같음
        # question = x["question"]
        # document = x["document"]
        # x는 나중에 invoke로 호출시 invoke안에 넣은 값
        # RunnableParallel 은 나중에 invoke로 실행되면 langchain을 구현하기 위해 작업지시를 내린것이지, 지금 question과 document를 꺼내란 얘기는 아님
        RunnableParallel(
            question=lambda x: x["question"],
            document=lambda x: x["document"],
        )
        # grade는 grade_prompt | llm |   StrOutputParser() 의 실행 결과입니다.
        #
        # 아래의 결과에 
        # {
        #    "question": "...",
        #    "document": "..."
        # }
        # grade 를 추가
        # {
        #     "question": "...",
        #     "document": "...",
        #     "grade": "이 문서는 질문과 관련이 있습니다. [결과]: Relevant"
        # }
        .assign(grade=grade_prompt | llm | StrOutputParser())
        # 위에서 grade 까지 추가된 데이터{...}를 parse_relevance_score에 파라미터로 던져줌
        #
        # 아래와 같은 결과를 냄:
        # {
        #  question,
        #  document,
        #  grade,
        #  score
        # }
        .assign(score=parse_relevance_score)
    )
    return retrieval_grader

# ── 2. RAG Chain ──────────────────────────────────────────────────────────────
# 질문과 검색된 문서를 GPT에게 보내서 최종 답변을 생성하는 RAG 파이프라인을 만드는 함수
def build_rag_chain(llm):

    prompt = ChatPromptTemplate([
        ("user", """당신은 QA(Question-Answering)을 수행하는 Assistant입니다. 다음의 Context를 이용하여 Question에 답변하세요.
최소 3문장에서 최대 5문장으로 답변하고, 정확한 답변을 제공하세요.
만약 모든 Context를 다 확인해도 정보가 없다면, \"정보가 부족하여 답변할 수 없습니다.\"를 출력하세요.
답변은 한국어로 작성하세요.
---
Context: {context}
---
Question: {question}"""),
    ])

    # prompt -> 지시
    # llm -> 실제 답변 생성
    # StrOutputParser -> 답변 문자열만 꺼냄
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain

# ── 3. Question Rewriter ──────────────────────────────────────────────────────
# 사용자의 질문을 GPT를 이용해서 더 명확한 질문으로 바꿔주는(Rewriter) 객체를 만드는 함수
# 예를 들어 입력이:
# {
#     "question": "RAG 장점이 뭐야?"
# }
# 이면, 최종 결과:
# RAG(Retrieval-Augmented Generation)의 주요 장점은 무엇인가요?
def build_question_rewriter(llm):

    system = """당신은 주어진 질문을 재작성해야 합니다.
질문의 의미가 더 명확하게 드러나도록 Rewrite 하세요.
영어를 한국어로 번역하지 마세요."""

    re_write_prompt = ChatPromptTemplate([
        ("system", system),
        ("user", "원본 질문: {question} \n 새로운 질문:"),
    ])

    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter

# ── LangGraph State 정의 ──────────────────────────────────────────────────────
# 질문이 들어오면 어떤 순서로 처리할지(흐름도)를 만드는 함수
# 질문
#   │
#   ▼
# retrieve          (문서 검색)
#   │
#   ▼
# grade_documents   (문서 관련성 검사)
#   │
#   ├── 관련 문서 있음 ─────────► generate (최종 답변 생성)
#   │
#   └── 관련 문서 없음
#            │
#            ▼
# transform_query   (질문 재작성)
#            │
#            ▼
# web_search        (웹 검색)
#            │
#            ▼
# generate          (최종 답변 생성)
def build_graph(retriever, retrieval_grader, rag_chain, question_rewriter, tavily):

    # LangGraph에서 공유되는 데이터(State)를 정의하는 코드
    class CRAGState(TypedDict):
        question:          str
        generation:        str
        needed_web_search: bool
        documents:         List[str]

    # ── 노드 함수들 ─────────────────────────────────────────────────────────

    # 사용자의 질문으로 VectorDB를 검색해서 관련 문서를 가져오는 노드입니다
    def retrieve(state):
        print("---검색---")
        question  = state["question"]        
        # documents = [
        #     Document(...),
        #     Document(...),
        #     ...
        # ]
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    # 검색된 문서를 참고해서 GPT가 최종 답변을 생성하는 노드
    def generate(state):
        print("---생성---")
        question  = state["question"]
        documents = state["documents"]

        print("**************DOCUMENTS****************")
        # 검색된 문서 내용을 화면에 출력합니다. 디버깅(확인용) 코드
        print("\n---\n".join([doc.page_content for doc in documents]))
        print("***************************************")

        # GPT가 최종 답변을 생성
        # {
        #     "context": documents,
        #     "question": "RAG란?"
        # }
        # ↓
        # GPT: RAG는 Retrieval-Augmented Generation의 약자로...
        # ↓
        # return {
        #     "documents": documents,
        #     "question": question,
        #     "generation": generation
        # }
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    # Retriever가 찾은 문서들 중에서 정말 관련 있는 문서만 남기는 함수
    def grade_documents(state):
        print("---쿼리 연관성 검색---")
        question      = state["question"]
        documents     = state["documents"]
        # 관련 있는 문서만 저장할 리스트
        filtered_docs = []
        # 기본값은 웹 검색이 필요(True)
        needed_web_search = True

        for doc in documents:
            # → GPT에게 "이 문서가 질문과 관련 있나요?" 를 물어보고,
            # 결과인 "Yes" 또는 "No"를 가져옵니다.
            score = retrieval_grader.invoke(
                {"question": question, "document": doc.page_content}
            )["score"]

            if score == "No":
                print("---GRADE: 관련성 없음---")
            else:
                print("---GRADE: 관련성 확인---")
                # → 관련 있는 문서는 저장하고,
                # 웹 검색은 필요 없다(False)고 표시합니다.
                filtered_docs.append(doc)
                needed_web_search = False

        return {
            "documents":         filtered_docs,
            "question":          question,
            "needed_web_search": needed_web_search,
        }

    # 질문을 더 검색하기 좋은 형태로 바꾸는 함수
    def transform_query(state):
        print("---쿼리 재조합---")
        question  = state["question"]
        documents = state["documents"]

        # → GPT에게 질문을 보내서 더 명확한 질문으로 다시 작성합니다.
        better_question = question_rewriter.invoke({"question": question})
        print(f"BEFORE: {question}")
        print(f"AFTER:  {better_question}")
        return {"documents": documents, "question": better_question}

    def web_search(state):
        print("---WEB SEARCH---")
        question  = state["question"]
        documents = state["documents"]

        # 한국어 질문 → 영문 검색 쿼리 변환
        # 한국어 질문을 영문 검색어로 변환합니다.
        # "RAG란?"
        # ↓
        # "What is Retrieval-Augmented Generation"
        # Tavily는 영문 검색이 더 잘 되기 때문에 변환합니다.
        _llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=256, temperature=0.1)
        get_query_prompt = ChatPromptTemplate([
            ("system", """주어진 [질문]에 대한 정보를 얻기 위해, 웹 검색을 하고 싶습니다. [질문]을 적절한 영문 쿼리로 변환하세요.
쿼리는 주로 단어들의 집합으로 나타내집니다."""),
            ("user", "[질문]: {question}"),
        ])
        query_chain = get_query_prompt | _llm | StrOutputParser()
        new_query   = query_chain.invoke({"question": question})

        print(f"---Query Generation---\nBefore: {question}\nAfter: {new_query}\n")

        docs = tavily.invoke(new_query)
        try:
            # → 검색 결과의 본문(content)만 하나의 문자열로 합칩니다.
            web_results = "\n--------\n".join([doc["content"] for doc in docs])
        except Exception as e:
            print(new_query, e)
            web_results = ""
        # 웹 검색 결과를 Document 객체로 만듭니다.
        web_doc = Document(page_content=web_results)
        print(web_doc)
        # → 기존 문서 목록에 웹 검색 결과를 추가합니다.
        # 기존 문서
        # ├─ PDF 문서1
        # ├─ PDF 문서2
        # ↓
        # 추가 후
        # ↓
        # ├─ PDF 문서1
        # ├─ PDF 문서2
        # └─ 웹 검색 결과
        documents.append(web_doc)

        # 웹 검색 결과가 추가된 문서를 다음 노드(generate)로 전달
        return {"documents": documents, "question": question}

    # ── 조건부 엣지 함수 ─────────────────────────────────────────────────────
    # 다음에 어떤 노드로 이동할지 결정하는 함수
    def decide_to_generate(state):
        print("---평가 결과를 확인합니다---")
        # → 이전 노드(grade_documents)에서 저장한, 웹 서치 필요여부 
        # True 또는 False 를 가져옵니다.
        needed_web_search = state["needed_web_search"]

        if needed_web_search:
            print("---DECISION: 지금 Context로 답변 불가---")
            # transform_query
            # (질문 재작성 → 웹 검색)
            return "transform_query"
        else:
            print("---DECISION: GENERATE---")
            # generate
            # (바로 답변 생성)
            return "generate"

    # ── 그래프 조립 ──────────────────────────────────────────────────────────
    # LangGraph의 실행 흐름(그래프)을 만드는 코드
    
    # CRAGState를 사용하는 새로운 그래프를 생성
    workflow = StateGraph(CRAGState)

    # → 그래프에 노드(작업)를 등록합니다.
    # 예를 들어, retrieve 노드가 실행되면
    # retrieve() 함수가 실행됩니다.
    workflow.add_node("retrieve",        retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate",        generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search)

    # 그래프의 시작 노드를 retrieve로 지정합니다
    # 질문 입력
    #     │
    #     ▼
    # retrieve
    # 부터 시작합니다.
    workflow.set_entry_point("retrieve")

    # → retrieve가 끝나면
    # 자동으로 grade_documents 로 이동
    workflow.add_edge("retrieve", "grade_documents")

    # grade_documents가 끝난 후에는 조건에 따라 다음 노드를 결정
    # decide_to_generate()가 return "generate" 를 반환하면
    # generate 로 이동. 반대상황은 transform_query.
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate":        "generate",
        },
    )

    # → 질문 재작성 후에는 무조건 web_search 를 실행
    workflow.add_edge("transform_query", "web_search_node")

    # → 웹 검색이 끝나면 최종 답변을 생성합니다.
    workflow.add_edge("web_search_node", "generate")

    # → 답변 생성이 끝나면 그래프를 종료합니다.
    workflow.add_edge("generate",        END)

    # → 지금까지 만든 그래프를 실행 가능한 객체(app)로 변환합니다.
    app = workflow.compile()
    return app

# ── 에이전트 실행/' ─────────────────────────────────────────────────────────────
# LangGraph(CRAG) 실제 실행 함수
# 질문을 입력받아 LangGraph를 실행하고, 최종 답변을 반환하는 함수
def run_crag(app, question: str) -> str:

    print("\n" + "=" * 60)
    print(f"질문: {question}")
    print("=" * 60)

    inputs = {"question": question}
    final_value = None

    # build_graph()에서 만든 LangGraph(app)를 실행합니다.
    # stream()은 노드가 하나 끝날 때마다 결과를 반환합니다.
    for output in app.stream(inputs):
        # → 현재 실행이 끝난 노드의 결과를 가져옵니다.
        # output.items 가져오는 결과 형태:
        # {
        #     "retrieve": {
        #         "question": "...",
        #         "documents": [...]
        #     }
        # }
        # ->
        # key → "retrieve"
        # value → {...}
        for key, value in output.items():
            pprint(f"Node '{key}':")
            # → 가장 마지막으로 실행된 노드의 결과를 저장합니다.
            # 마지막에는 보통 generate 노드의 결과가 저장됨.
            final_value = value
        pprint("---")

    # → 마지막 결과에서 GPT가 생성한 답변(generation)만 꺼냅니다.
    # 최종 답변(generation)은 마지막 노드에서만 만들어지기 때문
    # value = {
    #     "question": "...",
    #     "documents": [...],
    #     "generation": "RAG는 ..."
    # }
    answer = final_value.get("generation", "(답변 없음)")
    print("\n[최종 답변]")
    print(answer)
    return answer

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    check_env()

    llm, tavily = build_llm_and_tools()

    # ChromaDB가 없으면 PDF를 로드하여 생성, 있으면 바로 로드
    if not (os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR)):
        chunks = load_pdf()
        retriever = build_vectorstore(chunks)
    else:
        retriever = build_vectorstore()

    retrieval_grader   = build_retrieval_grader(llm)
    rag_chain          = build_rag_chain(llm)
    question_rewriter  = build_question_rewriter(llm)

    app = build_graph(retriever, retrieval_grader, rag_chain, question_rewriter, tavily)

    # ── 테스트 케이스 1: 로컬 문서에서 답변 가능한 질문 ──────────────────────
    run_crag(app, "강아지와 고양이 중 누가 더 뇌세포가 많을까요?")

    # ── 테스트 케이스 2: 로컬 문서에서 답변 가능한 질문 ────────────
    run_crag(app, "RAG의 HyDE 방법이 뭔가요?")

    # ── 테스트 케이스 3: 로컬 문서에 없는 질문 → 웹 검색 fallback ────────────
    run_crag(app, "who is Sam Altman?")

if __name__ == "__main__":
    main()