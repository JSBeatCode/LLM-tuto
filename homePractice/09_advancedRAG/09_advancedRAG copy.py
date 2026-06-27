"""
[실습] Advanced RAG
RAG의 기본 베이스 체인에서 시작하여, 다양한 고급 기능을 단계적으로 추가합니다.

실행 순서:
  STEP 1 : 데이터 로딩 & 인덱싱 (PDF → Chunk → ChromaDB)
  STEP 2 : 기본 RAG 체인 실행 (번역 체인 포함)
  STEP 3 : Multi-Query Retriever
  STEP 4 : Ensemble Retriever (BM25 + Semantic)
  STEP 5 : Contextual Retrieval (Ensemble 포함)

실행 방법:
  1. .env 파일에 OPENAI_API_KEY=sk-... 형식으로 저장
  2. papers.zip 파일을 이 .py 파일과 동일한 폴더에 위치
  3. python advanced_rag.py 실행
  4. 메뉴에서 원하는 STEP 번호 입력
"""

import os
import glob
import zipfile
import logging

from dotenv import load_dotenv
from tqdm import tqdm

# ── LangChain ──────────────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# from langchain.schema import Document
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser
# from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

import tiktoken
import openai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ── .env 로드 ─────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── 경로 설정 : 모든 파일을 이 .py와 동일한 폴더에 저장 ─────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PAPERS_ZIP = os.path.join(BASE_DIR, "papers.zip")
PAPERS_DIR = os.path.join(BASE_DIR, "papers")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ── 전역 상태 (STEP 간 공유) ──────────────────────────────────────────────────
all_papers   = []
token_chunks = []
db           = None
retriever    = None
llm          = None
embeddings   = None

QUESTIONS = [
    'Exaone 언어 모델이 다른 모델과 다른 점은 무엇인가요?',
    'Phi-3 언어 모델은 어떤 데이터로 학습했나요?',
    'Qwen 2의 다국어 성능은 어떻게 나타났나요?',
    'Gemma의 스몰 모델은 어떻게 학습했나요?',
]


# ═══════════════════════════════════════════════════════════════════════════════
# 공통 유틸
# ═══════════════════════════════════════════════════════════════════════════════

def validate_api_key():
    """OpenAI API 키 유효성 검증"""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        client.models.list()
        print("✅ OPENAI_API_KEY가 정상적으로 설정되어 있습니다.")
        return True
    except Exception:
        print("❌ API 키가 유효하지 않습니다! .env 파일을 확인하세요.")
        return False


def format_docs(docs):
    """Document 리스트를 하나의 문자열로 포맷"""
    return "\n---\n".join(
        [doc.page_content + '\nURL: ' + doc.metadata['source'] for doc in docs]
    )


def retriever_with_score(query):
    """유사도 점수를 메타데이터에 포함하여 Document 반환"""
    docs, scores = zip(*db.similarity_search_with_score(query))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score
    return docs


def print_results(questions, result):
    """질문-답변 쌍 출력"""
    for i, ans in enumerate(result):
        print(f"\n📌 Question: {questions[i]}")
        print(f"💬 Answer  : {ans}")
        print("─" * 60)


def build_base_chains():
    """번역 체인, 프롬프트, LLM 등 공통 컴포넌트 반환"""
    translate_prompt = ChatPromptTemplate(
        [
            ('system', '주어진 질문을 영어로 변환하세요.'),
            ('user', 'Question: {question}')
        ]
    )
    translate_chain = translate_prompt | llm | StrOutputParser()

    prompt = ChatPromptTemplate([
        ("user", '''당신은 QA(Question-Answering)을 수행하는 Assistant입니다.
다음의 Context를 이용하여 Question에 한국어로 답변하세요.
정확한 답변을 제공하세요.
만약 모든 Context를 다 확인해도 정보가 없다면, "정보가 부족하여 답변할 수 없습니다."를 출력하세요.
---
Context: {context}
---
Question: {question}''')
    ])

    return translate_chain, prompt


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 : 데이터 로딩 & 인덱싱
# ═══════════════════════════════════════════════════════════════════════════════

def step1_indexing():
    """
    STEP 1 : 데이터 불러오기 & 인덱싱
    - papers.zip 압축 해제
    - PyMuPDF로 PDF 로드 → 논문 단위 Document 생성
    - tiktoken 기반 토큰 청킹
    - ChromaDB에 임베딩 저장
    """
    global all_papers, token_chunks, db, retriever, llm, embeddings

    print("\n" + "═" * 60)
    print("  STEP 1 : 데이터 로딩 & 인덱싱")
    print("═" * 60)

    # ── LLM / 임베딩 초기화 ─────────────────────────────────────────────────
    llm        = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

    # ── papers.zip 압축 해제 ─────────────────────────────────────────────────
    if not os.path.exists(PAPERS_DIR):
        print(f"📦 {PAPERS_ZIP} 압축 해제 중...")
        with zipfile.ZipFile(PAPERS_ZIP, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
        print(f"✅ {PAPERS_DIR} 폴더에 압축 해제 완료")
    else:
        print(f"✅ papers 폴더가 이미 존재합니다: {PAPERS_DIR}")

    # ── PDF 로드 ─────────────────────────────────────────────────────────────
    pdf_files = glob.glob(os.path.join(PAPERS_DIR, "*.pdf"))
    print(f"\n📄 발견된 PDF 파일 수: {len(pdf_files)}")

    all_papers = []
    for i, path_paper in enumerate(pdf_files):
        loader = PyMuPDFLoader(path_paper)
        pages  = loader.load()
        doc    = Document(
            page_content='',
            metadata={'index': i, 'source': pages[0].metadata['source']}
        )
        for page in pages:
            doc.page_content += page.page_content
        all_papers.append(doc)

    print(f"✅ 총 {len(all_papers)}개 논문 로드 완료")

    # ── 토큰 수 확인 ─────────────────────────────────────────────────────────
    encoder = tiktoken.encoding_for_model('gpt-4o-mini')
    print("\n📊 논문별 토큰 수:")
    for paper in all_papers:
        token_count = len(encoder.encode(paper.page_content))
        source_name = os.path.basename(paper.metadata['source'])
        print(f"  {token_count:>8,} tokens  |  {source_name}")

    # ── 청킹 ─────────────────────────────────────────────────────────────────
    print("\n✂️  토큰 단위 청킹 중 (chunk_size=2000, overlap=200)...")
    token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o-mini",
        chunk_size=2000,
        chunk_overlap=200,
    )
    token_chunks = token_splitter.split_documents(all_papers)
    print(f"✅ 총 청크 수: {len(token_chunks)}")

    # ── ChromaDB 구축 (디스크 저장) ──────────────────────────────────────────
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"\n🗄️  기존 ChromaDB를 불러옵니다: {CHROMA_DIR}")
        db = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_metadata={'hnsw:space': 'l2'}
        )
        print("✅ 기존 ChromaDB 로드 완료 (임베딩 API 호출 없음)")
    else:
        print(f"\n🗄️  ChromaDB 신규 구축 중 → 저장 위치: {CHROMA_DIR}")
        db = Chroma.from_documents(
            documents=token_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
            collection_metadata={'hnsw:space': 'l2'}
        )
        print(f"✅ ChromaDB 구축 & 디스크 저장 완료 → {CHROMA_DIR}")
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # ── 유사도 점수 예시 ─────────────────────────────────────────────────────
    print("\n🔍 유사도 점수 예시 (query: 'How does Exaone achieve good evaluation results?')")
    sample = RunnableLambda(retriever_with_score).invoke(
        "How does Exaone achieve good evaluation results?"
    )
    for doc in sample:
        print(f"  score={doc.metadata.get('score', 'N/A'):.4f}  |  {os.path.basename(doc.metadata['source'])}")

    print("\n✅ STEP 1 완료 — 이제 STEP 2~5를 실행할 수 있습니다.")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 : 기본 RAG 체인
# ═══════════════════════════════════════════════════════════════════════════════

def step2_basic_rag():
    """
    STEP 2 : 기본 RAG 체인
    - 한국어 질문 → 영어 번역 체인
    - Semantic Retriever (ChromaDB)
    - GPT-4o-mini로 한국어 답변 생성
    """
    global db, retriever, llm

    print("\n" + "═" * 60)
    print("  STEP 2 : 기본 RAG 체인 (번역 체인 포함)")
    print("═" * 60)

    translate_chain, prompt = build_base_chains()
    retriever = db.as_retriever(search_kwargs={"k": 5})

    rag_chain = (
        {"context": translate_chain | retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n🚀 RAG 체인 실행 중...")
    result = rag_chain.batch(QUESTIONS)
    print_results(QUESTIONS, result)
    print("\n✅ STEP 2 완료")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 : Multi-Query Retriever
# ═══════════════════════════════════════════════════════════════════════════════

def step3_multi_query():
    """
    STEP 3 : Multi-Query Retriever
    - 하나의 질문을 LLM이 3가지 영문 버전으로 확장
    - 다양한 관점에서 검색하여 거리 기반 유사도 한계 보완
    """
    global db, llm

    print("\n" + "═" * 60)
    print("  STEP 3 : Multi-Query Retriever")
    print("═" * 60)

    # Multi-Query 로깅 활성화
    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

    rewrite_prompt = PromptTemplate(template="""
당신은 AI 언어 모델 어시스턴트입니다.
주어진 사용자 질문을 벡터 데이터베이스에서 관련 문서를 검색하기 위해
3가지 다른 영문 버전으로 생성하는 것이 당신의 작업입니다.
사용자 질문에 대한 여러 관점을 생성함으로써,
당신은 거리 기반 유사성 검색의 한계를 극복할 수 있도록
사용자에게 도움을 주는 것이 목표입니다. 이러한 대체 질문들을
새로운 줄로 구분하여 제공하세요.
---
원본 질문: {question}

""")

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=llm,
        prompt=rewrite_prompt,
    )

    # 예시 테스트
    print("\n🔍 Multi-Query 예시 (query: 'Phi-3는 어느 회사 모델?')")
    sample_docs = multi_query_retriever.invoke("Phi-3는 어느 회사 모델?")
    print(f"  검색된 문서 수: {len(sample_docs)}")

    _, prompt = build_base_chains()

    rag_chain = (
        {"context": multi_query_retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n🚀 Multi-Query RAG 체인 실행 중...")
    result = rag_chain.batch(QUESTIONS)
    print_results(QUESTIONS, result)
    print("\n✅ STEP 3 완료")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 : Ensemble Retriever (BM25 + Semantic)
# ═══════════════════════════════════════════════════════════════════════════════

def step4_ensemble():
    """
    STEP 4 : Ensemble Retriever
    - BM25 (Lexical / 키워드 기반) + ChromaDB (Semantic / 의미 기반) 하이브리드
    - 가중치 BM25:Semantic = 3:7
    """
    global db, token_chunks, llm

    print("\n" + "═" * 60)
    print("  STEP 4 : Ensemble Retriever (BM25 + Semantic)")
    print("═" * 60)

    bm25_retriever    = BM25Retriever.from_documents(token_chunks)
    bm25_retriever.k  = 5
    semantic_retriever = db.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.3, 0.7]
    )

    translate_chain, prompt = build_base_chains()

    rag_chain = (
        {"context": translate_chain | ensemble_retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n🚀 Ensemble RAG 체인 실행 중...")
    result = rag_chain.batch(QUESTIONS)
    print_results(QUESTIONS, result)
    print("\n✅ STEP 4 완료")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 : Contextual Retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def step5_contextual_retrieval():
    """
    STEP 5 : Contextual Retrieval (Anthropic 제안 기법)
    - 전체 문서 맥락을 참조해 청크별 영문 헤더(1~2문장)를 LLM으로 생성
    - 헤더를 청크 앞에 prepend 후 ChromaDB & BM25 재구성
    - Ensemble Retriever (BM25:Semantic = 5:5) 와 결합
    """
    global db, token_chunks, all_papers, embeddings, llm

    print("\n" + "═" * 60)
    print("  STEP 5 : Contextual Retrieval")
    print("═" * 60)
    print("⚠️  주의: 모든 청크에 LLM 호출이 발생합니다 (API 비용 / 시간 소요)")

    # ── Contextual Header 생성 체인 ──────────────────────────────────────────
    context_prompt = ChatPromptTemplate(
        [
            ('user', '''
당신은 문서 분석을 전문으로 하는 AI 어시스턴트입니다.
주어진 Document의 일부인 Chunk에 대해
간결하고 관련성 있는 짧은 설명을 생성하세요.

# Input Format

- [Document]: `<document> {document} </document>`
- [Chunk]: `<chunk> {chunk} </chunk>`

아래의 가이드라인을 참고하여,
이 부분에 대해 간결한 영문 Context을 작성하세요 (1-2문장).

1. 텍스트 부분에서 논의된 주요 주제나 개념을 포함하세요.
2. 문서 전체의 문맥에서 관련 정보나 비교를 언급하세요.
3. 가능한 경우, 이 정보가 문서의 전체적인 주제나 목적과 어떻게 연관되는지를 설명하세요.
4. 중요한 정보를 제공하는 주요 항목과 수치를 포함하세요.

텍스트 부분의 검색 정확성 개선을 위해,
문서의 전체 맥락에 해당하는 Context만을 출력하세요.
답변은 간결하게 작성하세요.

Context:
            ''')
        ]
    )
    context_chain = context_prompt | llm | StrOutputParser()

    # ── 예시 확인 ────────────────────────────────────────────────────────────
    print("\n🔍 Context 생성 예시 (chunk index=40):")
    sample_chunk = token_chunks[40]
    sample_doc   = all_papers[sample_chunk.metadata['index']].page_content
    sample_ctx   = context_chain.invoke({
        'document': sample_doc,
        'chunk': sample_chunk.page_content
    })
    print(f"  [Generated Context]\n  {sample_ctx}")
    print("  " + "─" * 50)
    print(f"  [Original Chunk (첫 200자)]\n  {sample_chunk.page_content[:200]}...")

    # ── 전체 청크에 Context 헤더 추가 (Rate Limit 재시도 포함) ───────────────
    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        wait=wait_exponential(multiplier=1, min=5, max=60),  # 5초~60초 사이 지수 증가
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def invoke_with_retry(inputs):
        return context_chain.invoke(inputs)

    print(f"\n⏳ 전체 {len(token_chunks)}개 청크에 Context 헤더 추가 중...")
    print("  ※ Rate Limit 발생 시 자동으로 재시도합니다.")
    for i, chunk in enumerate(tqdm(token_chunks)):
        doc     = all_papers[chunk.metadata['index']].page_content
        context = invoke_with_retry({'document': doc, 'chunk': chunk.page_content})
        token_chunks[i].page_content = context + '\n\n' + token_chunks[i].page_content

    # ── ChromaDB 재구성 (Contextual 청크 반영, 기존 DB 덮어쓰기) ──────────────
    import shutil
    import gc
    import time

    print(f"\n🗄️  기존 ChromaDB 삭제 후 Contextual 버전으로 재구성 중...")

    # Windows에서 SQLite 파일 잠금 해제를 위해 DB 연결 명시적으로 종료
    if db is not None:
        try:
            db._client.close()   # ChromaDB 클라이언트 연결 종료
        except Exception:
            pass
        globals()['db'] = None   # 전역 참조 제거
        gc.collect()             # 가비지 컬렉션으로 파일 핸들 해제
        time.sleep(1)            # Windows 파일 잠금 해제 대기

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"  🗑️  기존 {CHROMA_DIR} 삭제 완료")

    db = Chroma.from_documents(
        documents=token_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_metadata={'hnsw:space': 'l2'}
    )
    print(f"✅ Contextual ChromaDB 재구성 & 디스크 저장 완료 → {CHROMA_DIR}")

    # ── Ensemble Retriever (5:5) ──────────────────────────────────────────────
    bm25_retriever    = BM25Retriever.from_documents(token_chunks)
    bm25_retriever.k  = 5
    semantic_retriever = db.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.5, 0.5]
    )

    translate_chain, prompt = build_base_chains()

    rag_chain = (
        {"context": translate_chain | ensemble_retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n🚀 Contextual RAG 체인 실행 중...")
    import time
    result = []
    for i, question in enumerate(QUESTIONS):
        print(f"  ⏳ [{i+1}/{len(QUESTIONS)}] 질문 처리 중...")
        ans = rag_chain.invoke(question)
        result.append(ans)
        if i < len(QUESTIONS) - 1:
            time.sleep(5)   # TPM Rate Limit 방지용 대기
    print_results(QUESTIONS, result)
    print("\n✅ STEP 5 완료")


# ═══════════════════════════════════════════════════════════════════════════════
# 메인 메뉴
# ═══════════════════════════════════════════════════════════════════════════════

STEP_MAP = {
    "1": ("STEP 1 : 데이터 로딩 & 인덱싱",              step1_indexing),
    "2": ("STEP 2 : 기본 RAG 체인",                     step2_basic_rag),
    "3": ("STEP 3 : Multi-Query Retriever",              step3_multi_query),
    "4": ("STEP 4 : Ensemble Retriever (BM25 + Semantic)", step4_ensemble),
    "5": ("STEP 5 : Contextual Retrieval",               step5_contextual_retrieval),
}


def print_menu():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║          Advanced RAG — 실행 메뉴                      ║")
    print("╠" + "═" * 58 + "╣")
    for key, (name, _) in STEP_MAP.items():
        print(f"║  [{key}]  {name:<50}║")
    print("╠" + "═" * 58 + "╣")
    print("║  [all] STEP 1~5 순서대로 전체 실행                     ║")
    print("║  [q]   종료                                             ║")
    print("╚" + "═" * 58 + "╝")


def check_step1_done(step_num: str) -> bool:
    """STEP 1 완료 여부 확인 (STEP 2 이상 실행 전 체크)"""
    if step_num == "1" or step_num == "all":
        return True
    if db is None or not token_chunks or not all_papers:
        print("\n⚠️  먼저 STEP 1을 실행하여 데이터를 로드하세요.")
        return False
    return True


def main():
    print("\n" + "═" * 60)
    print("  Advanced RAG 실습")
    print("═" * 60)

    if not validate_api_key():
        return

    while True:
        print_menu()
        choice = input("\n실행할 STEP을 선택하세요 > ").strip().lower()

        if choice == "q":
            print("종료합니다.")
            break
        elif choice == "all":
            if not check_step1_done("all"):
                continue
            for key in STEP_MAP:
                _, fn = STEP_MAP[key]
                fn()
        elif choice in STEP_MAP:
            if not check_step1_done(choice):
                continue
            _, fn = STEP_MAP[choice]
            fn()
        else:
            print("⚠️  올바른 번호를 입력하세요 (1~5, all, q)")


if __name__ == "__main__":
    main()