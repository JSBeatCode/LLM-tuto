"""
[Topic 8] LangChain을 이용한 RAG 만들기
- 네이버 뉴스 API로 기사 수집
- WebBaseLoader로 본문 크롤링
- ChromaDB 벡터 DB 구축
- LangChain LCEL RAG 체인 실행
"""

import os
import re
import requests
import nest_asyncio
import jsonlines

from dotenv import load_dotenv

# 모듈 최상단에서 .env 로드
# → 이후 os.getenv() 호출이 모두 .env 값을 정상 반영
load_dotenv(override=True)

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# ── 상수 ──────────────────────────────────────────────────────────────────────
MODEL           = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
# CHROMA_DIR      = "./chroma_Web"
CHROMA_DIR      = "./homePractice/08_langChainRAG/chroma_Web"
# DOCS_JSONL      = "docs.jsonl"
DOCS_JSONL = "./homePractice/08_langChainRAG/docs.jsonl"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200

# load_dotenv() 이후에 호출되므로 .env 값이 정상 반영됨
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
NAVER_CLIENT_ID     = os.getenv("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "")

SEARCH_TOPICS = ["LLM", "생성 인공지능", "GPT", "가전제품"]


# ── STEP 0. 환경 설정 ─────────────────────────────────────────────────────────
def setup():
    """환경 변수 유효성 검사 및 기본 설정"""
    # 사실 아래 코드에서 `USER_AGENT`는 `.env`에서 불러오는 게 아니에요!
    # ```python
    # os.environ.setdefault("USER_AGENT", "MyCustomAgent")
    # ```
    # `setdefault`는 해당 환경변수가 **없을 때만** 기본값을 세팅하는 함수입니다. 즉, `"MyCustomAgent"`라는 값이 코드에 하드코딩되어 있어요.
    # **USER_AGENT가 필요한 이유**는 `WebBaseLoader`가 웹 크롤링 시 HTTP 요청 헤더에 `User-Agent` 값을 포함시키는데, 이게 없으면 LangChain이 경고를 출력하기 때문입니다. 네이버 뉴스 서버 입장에서는 "어떤 클라이언트가 요청을 보내는지" 식별하는 정보예요.
    os.environ.setdefault("USER_AGENT", "MyCustomAgent")

    # 아럐는 Python의 **비동기(async) 이벤트 루프 중첩 문제**를 해결하는 코드예요.
    # ```
    # RuntimeError: This event loop is already running.
    # ```
    # **Jupyter 노트북은** 자체적으로 이벤트 루프를 항상 실행 중인 환경이라서, `WebBaseLoader`의 `aload()` (비동기 로딩)를 호출하면 위 에러가 발생합니다.
    # `nest_asyncio.apply()`는 이 제한을 풀어줘서 **이미 실행 중인 루프 안에서도 비동기 함수를 실행할 수 있게** 해줍니다.
    # ---
    # 즉, 이 코드는 원래 Jupyter 노트북용 코드였기 때문에 들어간 것이고, **`.py` 파일로 실행하는 지금은 사실 없어도 됩니다.** 있어도 오류는 나지 않으니 그냥 둬도 무방해요.
    nest_asyncio.apply()  # Jupyter 비동기 처리 호환

    if not OPENAI_API_KEY:
        raise EnvironmentError(".env 파일에 OPENAI_API_KEY가 없습니다.")
    print("[STEP 0] OPENAI_API_KEY 정상 로드 완료")

    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        raise EnvironmentError(".env 파일에 NAVER_CLIENT_ID 또는 NAVER_CLIENT_SECRET가 없습니다.")
    print("[STEP 0] NAVER API KEY 정상 로드 완료")


# ── STEP 1. 네이버 뉴스 링크 수집 ────────────────────────────────────────────
def get_naver_news_links(query: str, num_links: int = 100) -> list[str]:
    """네이버 뉴스 검색 API로 기사 URL 수집 (네이버 뉴스 형식만 필터링)"""
    url = (
        f"https://openapi.naver.com/v1/search/news.json"
        f"?query={query}&display={num_links}&sort=sim"
    )
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    response = requests.get(url, headers=headers)
    
    # 👉 result 구조 예시:
    # result = {
    #     "items": [
    #         {"link": "https://n.news.naver.com/..."},
    #         {"link": "https://blog.naver.com/..."},
    #     ]
    # }
    result = response.json()

    # 파이썬에서 많이 쓰는 문법(리스트 컴프리헨션)
    filtered = [
        item["link"]
        
        # result 딕셔너리에서 "items" 키 가져오기. 없으면 [] (빈 리스트)
        # 가져와서 for loop로 하나씩 item 으로 꺼냄
        for item in result.get("items", [])
        
        # 링크 안에 "네이버 뉴스 주소"가 포함되어 있으면만 사용
        if "n.news.naver.com/mnews/article/" in item["link"]
    ]

    # filtered[0] if filtered else 'N/A' => filtered 리스트가 있으면 → 첫 번째 값. 없으면 → 'N/A'
    # 이 코드는 아래랑 완전히 동일하다 👇
    # if filtered:
    #     result = filtered[0]
    # else:
    #     result = "N/A"
    print(f"  [{query}] {len(filtered)}개 링크 수집 / 예시: {filtered[0] if filtered else 'N/A'}")
    return filtered


def collect_links() -> list[str]:
    """여러 키워드로 뉴스 링크 수집 후 중복 제거"""
    print("\n[STEP 1] 네이버 뉴스 링크 수집")
    all_links = []
    for topic in SEARCH_TOPICS:
        all_links += get_naver_news_links(topic, 100)

    unique_links = list(set(all_links))
    print(f"  총 {len(all_links)}개 수집 → 중복 제거 후 {len(unique_links)}개")
    return unique_links


# ── STEP 2. 웹 페이지 본문 로딩 ───────────────────────────────────────────────
# “뉴스 URL 리스트를 받아서 → 실제 기사 본문을 긁어와서 → Document 형태로 반환” 하는 코드야
# 🧠 WebBaseLoader가 뭐냐?
# 👉 쉽게 말하면:
# 웹페이지(URL)를 읽어서
# 본문 텍스트만 뽑아주는 도구
def load_documents(links: list[str]) -> list[Document]:
    """WebBaseLoader로 뉴스 본문 비동기 크롤링"""
    print("\n[STEP 2] WebBaseLoader로 본문 수집 (비동기)")
    loader = WebBaseLoader(
        web_paths=links,
        # HTML 구조 중에서 “본문만 골라라” 
        bs_kwargs=dict(
            # 👉 bs4.SoupStrainer는 **“HTML에서 필요한 부분만 골라서 가져오는 필터”**다
            # ❌ SoupStrainer 없으면 👉 전체 HTML 다 가져옴
            parse_only=bs4.SoupStrainer(
                # "이 class만 가져와라"
                class_=("newsct", "newsct-body")  # 네이버 뉴스 본문 HTML 요소
            )
        ),
        requests_per_second=10,
        show_progress=True,
    )

    # 💡 aload() vs load()
    # | 함수      | 의미          |
    # | ------- | ----------- |
    # | load()  | 일반 실행       |
    # | aload() | 비동기 실행 (빠름) |
    docs = loader.aload()  # 비동기 로딩
    print(f"  {len(docs)}개 문서 로드 완료")
    return docs


# ── STEP 3. 전처리 ────────────────────────────────────────────────────────────
NOISE_TEXTS = [
    "구독중 구독자 0 응원수 0 더보기",
    "쏠쏠정보 0 흥미진진 0 공감백배 0 분석탁월 0 후속강추 0",
    "댓글 본문 요약봇 본문 요약봇",
    "도움말 자동 추출 기술로 요약된 내용입니다. 요약 기술의 특성상 본문의 주요 내용이 제외될 수 있어, 전체 맥락을 이해하기 위해서는 기사 본문 전체보기를 권장합니다. 닫기",
    "텍스트 음성 변환 서비스 사용하기 성별 남성 여성 말하기 속도 느림 보통 빠름",
    "이동 통신망을 이용하여 음성을 재생하면 별도의 데이터 통화료가 부과될 수 있습니다. 본문듣기 시작",
    "닫기 글자 크기 변경하기 가1단계 작게 가2단계 보통 가3단계 크게 가4단계 아주크게 가5단계 최대크게 SNS 보내기 인쇄하기",
    "PICK 안내 언론사가 주요기사로선정한 기사입니다. 언론사별 바로가기 닫기",
    "응원 닫기",
    "구독 구독중 구독자 0 응원수 0 ",
]

SPLIT_MARKERS = [
    ("구독 해지되었습니다.", "after"),   # 이 마커 뒤 내용 사용
    ("구독 메인에서 바로 보는 언론사 편집 뉴스 지금 바로 구독해보세요!", "before"),  # 이 마커 앞 내용 사용
]


# ❌ 완전히 일반적인 함수는 아님 ⭕ "네이버 뉴스 맞춤형" 함수
def clean_text(doc: Document) -> Document:
    """단일 문서 텍스트 정제"""
    text = doc.page_content
    text = text.replace("\t", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    for marker, direction in SPLIT_MARKERS:
        parts = text.split(marker)
        if len(parts) > 1:
            text = parts[1] if direction == "after" else parts[0]

    for noise in NOISE_TEXTS:
        text = text.replace(noise, "")

    text = re.sub(r"\s+", " ", text).strip()
    doc.page_content = text
    return doc


def preprocess(docs: list[Document]) -> list[Document]:
    """전체 문서 전처리 (노이즈 제거, 공백 정규화)"""
    print("\n[STEP 3] 전처리 (노이즈 제거)")
    preprocessed = [clean_text(doc) for doc in docs]
    print(f"  {len(preprocessed)}개 문서 전처리 완료")
    return preprocessed


# ── STEP 4. 문서 저장 & 로드 (선택) ───────────────────────────────────────────
# “크롤링한 문서를 파일로 저장”
def save_docs_to_jsonl(documents: list[Document], file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 폴더 없으면 자동 생성
    with jsonlines.open(file_path, mode="w") as writer:
        for doc in documents:
            writer.write(doc.model_dump())  # Pydantic v2 호환

# “크롤링한 문서를 파일로 저장한 것을 다시 불러오기 위한 코드”
def load_docs_from_jsonl(file_path: str) -> list[Document]:
    documents = []
    with jsonlines.open(file_path, mode="r") as reader:
        for doc in reader:
            documents.append(Document(**doc))
    return documents


# ── STEP 5. 청킹 & 벡터 DB 구축 ───────────────────────────────────────────────
def build_vector_db(docs: list[Document]) -> Chroma:
    """문서를 청킹하고 ChromaDB에 임베딩 저장"""
    print("\n[STEP 5] 청킹 & ChromaDB 구축")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"  청크 수: {len(chunks)}")

    # 기존 컬렉션 초기화
    Chroma().delete_collection()

    # 텍스트 문서를 벡터로 변환해서 검색 가능한 DB(Chroma)에 저장한다
    db = Chroma.from_documents(
            # 쪼개진 텍스트 리스트 (Document 형태)
            # 👉 의미:
            # 문서를 그대로 넣는 게 아니라
            # 잘게 쪼개서 넣는다
            # 👉 왜?
            # 긴 문장은 검색 정확도 떨어짐
            # → 짧게 쪼개야 잘 찾는다
            documents=chunks, 

            # 👉 이걸로 뭐하냐?
            # 비슷한 의미끼리 가까워짐
            # 🔥 핵심 개념
            # 문장 의미를 숫자로 표현
            # → 의미 기반 검색 가능
            # 텍스트를 벡터(숫자)로 변환하는 모델
            embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),  


            # DB를 파일로 저장할 경로 (재사용 가능)
            # 👉 의미:
            # DB를 디스크에 저장
            # 👉 없으면:
            # 프로그램 종료 → 데이터 날아감 ❌
            # 👉 있으면:
            # 다음 실행 때 재사용 가능 ⭕
            persist_directory=CHROMA_DIR,  

            # 벡터 검색 방식 설정 (거리 계산 방식)
            # 텍스트 데이터 → AI가 이해하는 숫자 공간으로 변환 → 검색 가능한 DB 생성
            collection_metadata={"hnsw:space": "l2"},  
    )
    print(f"  ChromaDB 저장 완료 → {CHROMA_DIR}")
    return db


# ── STEP 6. RAG 체인 구성 ────────────────────────────────────────────────────
def build_rag_chain(db: Chroma):
    """RAG 체인 생성 (retriever → prompt → LLM → 파싱)"""

    # 벡터 DB를 "검색 엔진"으로 변환
    # 👉 db는 그냥 데이터 저장소고
    # 👉 retriever는 검색 기능이 있는 객체
    retriever = db.as_retriever()

    # 검색된 문서들을 LLM이 읽기 좋게 정리
    # 📦 입력
    # docs = [
    #     Document(page_content="기사 내용1", metadata={"source": "URL1"}),
    #     Document(page_content="기사 내용2", metadata={"source": "URL2"}),
    # ]
    # 💡 결과 예
    # 기사 내용1
    # URL: https://news1
    # 기사 내용2
    # URL: https://news2
    def format_docs(docs):
        """검색된 문서 리스트를 ---로 구분된 텍스트 + URL로 변환"""
        return "\n---\n".join(
            [doc.page_content + "\nURL: " + doc.metadata["source"] for doc in docs]
        )

    prompt = ChatPromptTemplate([
        ("user", """당신은 QA(Question-Answering)을 수행하는 Assistant입니다.
다음의 Context를 이용하여 Question에 답변하세요.
최소 3문장에서 최대 5문장으로 답변하고, 정확한 답변을 제공하세요.
만약 모든 Context를 다 확인해도 정보가 없다면,
"정보가 부족하여 답변할 수 없습니다."를 출력하세요.
---
Context: {context}
---
Question: {question}""")
    ])

    llm = ChatOpenAI(model_name=MODEL, temperature=0.1)

    # 기본 RAG 체인
    # 🧠 전체 흐름 먼저
    # 질문
    # → context 생성 (문서 검색)
    # → prompt 구성
    # → LLM 실행
    # → 문자열 출력
    rag_chain = (

        # [🧩 전체적인 의미]
        # "question" → 그대로 사용
        # "context" → 질문 기반으로 문서 검색해서 생성
        #   🔍 retriever | format_docs
        #     👉 의미:
        #       질문
        #       → retriever로 관련 문서 찾기
        #       → format_docs로 문자열 변환
        #   🔍 RunnablePassthrough()
        #     👉 의미:
        #       입력값 그대로 통과
        #     👉 예:
        #       rag_chain.invoke("AI란 무엇인가?")
        #     👉 결과:
        #       question = "AI란 무엇인가?"
        # [🔥 여기까지 결과]
        #   {
        #       "context": "관련 기사 내용들...",
        #       "question": "AI란 무엇인가?"
        #   }
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm        
        # LLM 결과 → 문자열로 변환
        | StrOutputParser()
    )

    # context + answer 동시 반환 체인 (RunnableParallel + assign)    
    # 이미 준비된 context와 question을 받아서 답변만 만드는 체인
    rag_chain_from_docs = prompt | llm | StrOutputParser()

    # 🔍 rag_chain_with_source = RunnableParallel(         {"context": retriever | format_docs, "question": RunnablePassthrough()}     ) 
    # 👉 이 부분은 질문 하나를 받아서 동시에 2개 값을 만들어.
    #   입력 질문
    #   ├─ context: retriever로 관련 문서 검색 후 format_docs로 정리
    #   └─ question: 입력 질문 그대로 통과
    #   
    #   예를 들어 사용자가:
    #   rag_chain_with_source.invoke("인공지능의 최근 발전 방식은?")
    #   라고 하면 중간 결과는 이렇게 돼.
    #   {
    #       "context": "검색된 뉴스 기사 내용...\nURL: ...",
    #       "question": "인공지능의 최근 발전 방식은?"
    #   }
    # 🔍 .assign(answer=rag_chain_from_docs) 
    # 👉 기존 결과에 answer라는 새 항목을 추가하는 거야.
    #   즉:
    #   {
    #       "context": "...",
    #       "question": "..."
    #   }
    #   여기에 답변을 하나 더 붙여서:
    #   {
    #       "context": "...",
    #       "question": "...",
    #       "answer": "LLM이 만든 최종 답변"
    #   }
    #   이렇게 만든다.
    # 
    # 왜 이걸 쓰냐? 이유는 아주 실무적이야.
    #   1. LLM이 어떤 문서를 보고 답했는지 확인
    #   2. 답변이 근거 기반인지 검증
    #   3. 출처 URL을 같이 보여주기
    #   4. 디버깅하기
    rag_chain_with_source = RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain, rag_chain_with_source


# ── STEP 7. 질문 테스트 ───────────────────────────────────────────────────────
def run_queries(rag_chain, rag_chain_with_source) -> None:
    """RAG 체인으로 테스트 질문 실행"""
    print("\n[STEP 7] RAG 체인 테스트")

    test_questions = [
        "도메인 특화 언어 모델이란 무엇입니까? 어떤 예시가 있나요?",
        "인공지능의 최근 발전 방식은? 관련 링크도 보여주세요",
        "알리바바의 언어 모델 이름은?",
    ]

    for q in test_questions:
        print(f"\n  질문: {q}")
        answer = rag_chain.invoke(q)
        print(f"  답변: {answer}")
        print("  " + "-" * 60)

    # context + answer 함께 반환 테스트
    print("\n[STEP 7-2] RunnableParallel (context + answer 동시 반환)")
    result = rag_chain_with_source.invoke("인공지능의 최근 발전 방식은? 관련 링크도 보여주세요")
    print(f"  question : {result['question']}")
    print(f"  answer   : {result['answer']}")
    print(f"  context  : {result['context'][:200]}...")  # 너무 길어서 일부만 출력


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    # 0. 환경 설정
    setup()

    # 1. 뉴스 링크 수집
    links = collect_links()

    # 2. 본문 로딩
    docs = load_documents(links)

    # 3. 전처리
    preprocessed_docs = preprocess(docs)

    # 4. 문서 저장 (선택적 재사용)
    save_docs_to_jsonl(preprocessed_docs, DOCS_JSONL)
    print(f"\n[STEP 4] 문서 저장 완료 → {DOCS_JSONL}")

    # 5. 벡터 DB 구축
    db = build_vector_db(preprocessed_docs)

    # 6. RAG 체인 구성
    print("\n[STEP 6] RAG 체인 구성")
    rag_chain, rag_chain_with_source = build_rag_chain(db)

    # 7. 테스트 질문 실행
    run_queries(rag_chain, rag_chain_with_source)


if __name__ == "__main__":
    main()