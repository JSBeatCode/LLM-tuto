"""
[실습] 오픈 모델을 이용한 RAG 만들기
- LLM     : Alibaba Cloud - Qwen/Qwen2.5-7B-Instruct (4-bit 양자화)
- 임베딩   : Microsoft - intfloat/multilingual-e5-small
- 벡터 DB  : ChromaDB
- 데이터   : Wikipedia (한국어)
"""

import os
import torch

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
# ─────────────────────────────────────────
# 0. 경로 설정 & 환경변수 로드
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMB_MODEL_DIR = os.path.join(BASE_DIR, "e5_small")  # 임베딩 모델: .py 옆 폴더에 저장
CHROMA_DIR    = os.path.join(BASE_DIR, "RAG_db")    # ChromaDB:     .py 옆 폴더에 저장
# LLM 모델(Qwen)은 HuggingFace 이 경로 캐시에 자동 저장
LLM_CPU_MODEL_DIR = os.path.join(BASE_DIR, "qwen2.5-1.5b")        # LLM (CPU용 1.5B):  .py 옆 폴더에 저장
LLM_GPU_MODEL_DIR = os.path.join(BASE_DIR, "qwen2.5-7b")          # LLM (GPU용 7B):    .py 옆 폴더에 저장

load_dotenv(os.path.join(BASE_DIR, ".env"))
HF_TOKEN = os.getenv("HF_TOKEN")  # Llama 등 gated 모델 사용 시 필요


# ─────────────────────────────────────────
# 1. HuggingFace 로그인 (Qwen은 불필요, Llama 등 gated 모델 사용 시 활성화)
# ─────────────────────────────────────────
def login_huggingface():
    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        print("[INFO] HuggingFace 로그인 완료")
    else:
        print("[INFO] HF_TOKEN 없음 → HuggingFace 로그인 생략 (Qwen 사용 시 불필요)")


# ─────────────────────────────────────────
# 2. LLM 로드 (Qwen2.5-7B, 4-bit 양자화)
# ─────────────────────────────────────────
def load_llm():

    # HuggingFace 모델을 불러오고 실행하기 위한 도구들
    from transformers import (
        AutoModelForCausalLM,  # Qwen 같은 생성형 LLM 모델 본체를 불러오는 클래스
        AutoTokenizer,         # 사람의 문장을 모델이 이해하는 숫자 토큰으로 바꾸는 도구
        pipeline,              # 모델 + 토크나이저 + 생성 설정을 묶어 쉽게 실행하게 해주는 도구
    )

    # HuggingFace 모델을 LangChain에서 사용할 수 있게 연결해주는 도구들
    from langchain_huggingface import (
        HuggingFacePipeline,   # HuggingFace pipeline을 LangChain용 LLM으로 감싸는 어댑터
        ChatHuggingFace,       # LLM을 채팅 모델 형태로 사용할 수 있게 해주는 래퍼
    )
    # ── 모델 선택 ──────────────────────────────────────────────────────────
    # GPU(CUDA) 있을 경우 : Qwen2.5-7B-Instruct (4-bit 양자화, 고품질)
    # GPU 없을 경우 (CPU) : Qwen2.5-1.5B-Instruct (경량, RAG 구조 학습용)
    use_cuda = torch.cuda.is_available()
    model_id = "Qwen/Qwen2.5-7B-Instruct" if use_cuda else "Qwen/Qwen2.5-1.5B-Instruct"

    print(f"[INFO] LLM 로드 중: {model_id}")
    print(f"[INFO] GPU(CUDA) 사용 가능 여부: {use_cuda}")
    if use_cuda:
        print(f"[INFO] GPU 이름: {torch.cuda.get_device_name(0)}")

    if use_cuda:
        # ── GPU 환경: 4-bit NF4 양자화 적용 (VRAM 절약) ──────────────────
        # from_pretrained()가 HuggingFace 기본 캐시를 자동으로 사용
        #   최초 실행 시 다운로드 / 이후 실행 시 캐시에서 바로 로드

        # .py 옆 폴더에 없으면 최초 1회 다운로드, 이후엔 로컬에서 바로 로드
        if not os.path.exists(LLM_GPU_MODEL_DIR):
            print(f"[INFO] LLM 최초 다운로드 → {LLM_GPU_MODEL_DIR}")
            snapshot_download(repo_id=model_id, local_dir=LLM_GPU_MODEL_DIR)
        else:
            print(f"[INFO] 로컬 LLM 발견, 다운로드 생략 → {LLM_GPU_MODEL_DIR}")

        from transformers import BitsAndBytesConfig

        # 양자화 코딩: 모델을 더 작은 용량으로 압축해서 GPU 메모리를 덜 쓰게 만드는 기술
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,              # 모델을 4-bit로 압축해서 로드 → GPU 메모리 절약
            bnb_4bit_use_double_quant=True, # 압축 정보를 한 번 더 효율화 → 추가 메모리 절약
            bnb_4bit_quant_type="nf4",      # LLM에 자주 쓰이는 4-bit 압축 방식
            bnb_4bit_compute_dtype=torch.bfloat16, # 저장은 4-bit, 계산은 bfloat16으로 수행
        )

        # 토크나이저: "인공지능이 뭐야?" -> [1234, 5678, 90, ...]
        tokenizer = AutoTokenizer.from_pretrained(LLM_GPU_MODEL_DIR)

        # Qwen LLM 모델 본체를 불러오는 코드
        model = AutoModelForCausalLM.from_pretrained(
            LLM_GPU_MODEL_DIR,              # 로컬 폴더에서 로드 (.py 옆)
            torch_dtype="auto",             # 모델에 맞는 데이터 타입을 자동 선택
            quantization_config=bnb_config, # 위에서 만든 4-bit 양자화 설정 적용
            device_map={"": 0},             # 모델 전체를 0번 GPU에 올림
            # attn_implementation="flash_attention_2"  # 고성능 GPU에서 속도 개선용 옵션. T4 미지원, A100 이상에서 활성화
        )
    else:
        # ── CPU 환경: 1.5B 경량 모델, 양자화 없이 로드 ────────────────────
        # .py 옆 폴더에 없으면 최초 1회 다운로드, 이후엔 로컬에서 바로 로드
        # RAM 요구량 약 3~4GB / 답변 품질은 7B보다 낮지만 RAG 구조 학습에 충분
        if not os.path.exists(LLM_CPU_MODEL_DIR):
            print(f"[INFO] LLM 최초 다운로드 → {LLM_CPU_MODEL_DIR}")
            snapshot_download(repo_id=model_id, local_dir=LLM_CPU_MODEL_DIR)
        else:
            print(f"[INFO] 로컬 LLM 발견, 다운로드 생략 → {LLM_CPU_MODEL_DIR}")

        print("[INFO] CPU 모드 → Qwen2.5-1.5B-Instruct 로드 (RAM 약 3~4GB 필요)")

        # 토크나이저: "인공지능이 뭐야?" -> [1234, 5678, 90, ...]
        tokenizer = AutoTokenizer.from_pretrained(LLM_CPU_MODEL_DIR)

        # HuggingFace에서 지정한 LLM 모델을 다운로드/로드해서, CPU에서 실행하도록 설정하는 코드
        model = AutoModelForCausalLM.from_pretrained(
            LLM_CPU_MODEL_DIR,        # 로컬 폴더에서 로드 (.py 옆)
            torch_dtype=torch.float32, # CPU에서 안정적으로 계산하기 위한 숫자 형식
            device_map="cpu",         # 모델을 GPU가 아닌 CPU에 올려서 실행
        )

    print("[INFO] LLM 로드 완료")

    # Generation 파라미터 (Qwen2.5 공식 generation_config 참고)
    # Qwen 모델이 답변을 생성하는 방식을 설정
    gen_config = dict(
        do_sample=True,
        max_new_tokens=512,
        repetition_penalty=1.05,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )

    # 모델 + 토크나이저 + 생성 옵션을 하나로 묶어서 “텍스트 생성기”처럼 사용할 수 있게 만드는 것
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        **gen_config,
    )

    # LangChain 연결: Pipeline → ChatHuggingFace (채팅 템플릿 지원)
    # HuggingFace 모델을 LangChain에서 사용할 수 있게 바꾸는 단계
    # HuggingFacePipeline: HuggingFace 실행기를 LangChain이 이해할 수 있는 형태로 감싼다.
    llm = HuggingFacePipeline(pipeline=pipe)

    # LangChain용 LLM을 채팅 모델 형태로 한 번 더 감싸는 코드.
    #     system 메시지
    #     user 메시지
    #     assistant 답변
    # 구조로 대화해야 되서.
    chat_model = ChatHuggingFace(llm=llm, tokenizer=tokenizer)

    print("[INFO] LangChain ChatHuggingFace 연결 완료")
    return chat_model


# ─────────────────────────────────────────
# 3. 임베딩 모델 로드 (multilingual-e5-small, CPU)
# ─────────────────────────────────────────
# 문서를 벡터로 바꿔주는 임베딩 모델을 준비하는 코드
# multilingual-e5-small 임베딩 모델을 로컬에 다운로드/로드해서,
# 문서와 질문을 숫자 벡터로 바꿀 준비를 하는 함수
# embeddings: "트랜스포머는 인공지능 모델 구조입니다." -> [0.12, -0.44, 0.87, ...]
def load_embeddings():
    # HuggingFace 임베딩 모델을 LangChain에서 사용할 수 있게 해주는 클래스
    from langchain_huggingface import HuggingFaceEmbeddings

    # HuggingFace Hub에서 모델 파일 전체를 로컬 폴더로 다운로드하는 함수
    from huggingface_hub import snapshot_download

    # 사용할 다국어 임베딩 모델 이름
    # 한국어 문서도 벡터로 변환할 수 있음
    model_name = "intfloat/multilingual-e5-small"

    # 임베딩 모델이 로컬 폴더에 없으면 최초 1회 다운로드
    if not os.path.exists(EMB_MODEL_DIR):
        print(f"[INFO] 임베딩 모델 다운로드 중: {model_name}")

        # HuggingFace 모델을 EMB_MODEL_DIR 폴더에 저장
        snapshot_download(repo_id=model_name, local_dir=EMB_MODEL_DIR)

        print(f"[INFO] 임베딩 모델 저장 완료: {EMB_MODEL_DIR}")
    else:
        # 이미 다운로드된 모델이 있으면 재다운로드하지 않음
        print(f"[INFO] 로컬 임베딩 모델 발견, 다운로드 생략: {EMB_MODEL_DIR}")

    # 로컬에 저장된 임베딩 모델을 LangChain용 embeddings 객체로 로드
    embeddings = HuggingFaceEmbeddings(
        model_name=EMB_MODEL_DIR,           # 로컬 임베딩 모델 폴더 경로
        model_kwargs={"device": "cpu"},     # 임베딩 계산은 CPU에서 수행
    )

    print("[INFO] 임베딩 모델 로드 완료")

    # ChromaDB에서 문서/질문을 벡터화할 때 사용할 객체 반환
    return embeddings


# ─────────────────────────────────────────
# 4. Wikipedia 데이터 수집 & 청킹
# ─────────────────────────────────────────
def load_and_split_documents():
    import json
    import time
    import random
    from langchain_community.document_loaders import WikipediaLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    DOCS_CACHE = os.path.join(BASE_DIR, "wiki_docs_cache.json")

    # ── 로컬 캐시가 있으면 Wikipedia 재수집 없이 바로 로드 ────────────────
    if os.path.exists(DOCS_CACHE):
        print(f"[INFO] 로컬 캐시 발견 → Wikipedia 재수집 생략: {DOCS_CACHE}")
        with open(DOCS_CACHE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in raw]
        print(f"[INFO] 캐시에서 {len(docs)}개 문서 로드 완료")
    else:
        # ── 최초 실행 시 Wikipedia에서 수집 후 로컬에 저장 ──────────────────
        topics = ["챗GPT", "인공지능", "트랜스포머_(기계_학습)", "GPT-4", "GPT-4o"]
        docs = []
        MAX_RETRY = 3  # 네트워크 오류 시 재시도 횟수

        # DELAY_BETWEEN = 5   # 토픽 간 요청 간격 (초) - Wikipedia Rate Limit 대응
        # DELAY_RETRY   = 10  # 실패 후 재시도 대기 (초)
        DELAY_BETWEEN = (3, 12)   # 토픽 간 요청 간격: 3~12초 랜덤
        DELAY_RETRY   = (3, 12)   # 실패 후 재시도 대기: 3~12초 랜덤

        print("[INFO] Wikipedia 문서 수집 중... (최초 1회)")
        
        # topics에 있는 검색어를 하나씩 Wikipedia에서 수집하고,
        # 오류가 나면 최대 MAX_RETRY번까지 재시도한 뒤,
        # 그래도 실패하면 해당 검색어는 건너뛰는 코드
        for i, query in enumerate(topics):
            for attempt in range(1, MAX_RETRY + 1):
                try:
                    # 한국어 Wikipedia에서 현재 query로 문서 검색/로드 준비
                    loader = WikipediaLoader(
                        query=query,                  # 현재 검색어
                        lang="ko",                    # 한국어 Wikipedia 사용
                        load_max_docs=3,              # 검색어당 최대 3개 문서 수집
                        doc_content_chars_max=10000,  # 문서 하나당 최대 10,000자까지 가져옴
                    )
                    docs += loader.load()
                    print(f"  [OK] '{query}' 수집 완료")
                    # 마지막 토픽이 아니면 다음 요청 전에 대기
                    if i < len(topics) - 1:
                        delay_between_random = random.uniform(DELAY_BETWEEN)
                        print(f"  [대기] 다음 요청까지 {delay_between_random:.1f}초 대기 중...")
                        # time.sleep(DELAY_BETWEEN)
                        time.sleep(delay_between_random)
                    break
                except Exception as e:
                    print(f"  [재시도 {attempt}/{MAX_RETRY}] '{query}' 오류: {e}")
                    if attempt < MAX_RETRY:
                        delay_retry_random = random.uniform(DELAY_RETRY)
                        print(f"  [대기] {delay_retry_random:1f}초 후 재시도...")
                        # time.sleep(DELAY_RETRY)
                        time.sleep(delay_retry_random)
                    else:
                        print(f"  [SKIP] '{query}' 최대 재시도 초과, 건너뜁니다.")

        print(f"[INFO] 수집된 문서 수: {len(docs)}")

        # 로컬 캐시로 저장
        with open(DOCS_CACHE, "w", encoding="utf-8") as f:
            json.dump(
                [{"content": d.page_content, "metadata": d.metadata} for d in docs],
                f, ensure_ascii=False, indent=2
            )
        print(f"[INFO] Wikipedia 문서 캐시 저장 완료: {DOCS_CACHE}")

    # 긴 Wikipedia 문서들을 800자 정도의 작은 조각으로 나누고,
    # 조각끼리 80자 정도 겹치게 만들어서 검색하기 좋은 형태로 바꾸는 코드
    # 긴 문서를 RAG 검색에 적합한 작은 조각(chunk)으로 나누는 도구 생성
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,    # 청크 하나의 최대 크기: 약 800자
        chunk_overlap=80,  # 청크끼리 80자씩 겹치게 해서 문맥 끊김 방지
    )

    # | 구분       | 의미                    |
    # | -------- | --------------------- |
    # | `docs`   | Wikipedia에서 가져온 원본 문서 |
    # | `chunks` | 원본 문서를 작게 나눈 조각들      |
    # Wikipedia에서 가져온 원본 문서들을 작은 청크 목록으로 분할
    chunks = text_splitter.split_documents(docs)

    print(f"[INFO] 청킹 완료 → 총 {len(chunks)}개 청크")
    return chunks


# ─────────────────────────────────────────
# 5. ChromaDB 벡터 스토어 구축
# ─────────────────────────────────────────
# 문서 청크들을 ChromaDB라는 벡터DB에 저장하고, 질문이 들어왔을 때 관련 문서를 찾아오는 검색기 retriever를 만드는 함수
def build_vector_store(chunks, embeddings):
    from langchain_chroma import Chroma

    # DB가 이미 존재하면 로드, 없으면 새로 구축
    if os.path.exists(CHROMA_DIR):
        print(f"[INFO] 기존 ChromaDB 발견 → 로드합니다: {CHROMA_DIR}")
        db = Chroma(
            # ChromaDB가 저장된 폴더를 지정
            persist_directory=CHROMA_DIR,

            # 기존 DB를 사용할 때도 임베딩 함수가 필요
            # 질문: GPT-4o 출시일은?
            # ↓
            # 질문 벡터: [0.21, -0.15, 0.72, ...]
            # ↓
            # DB 안의 문서 벡터들과 비교
            embedding_function=embeddings,

            # 벡터끼리 얼마나 가까운지 비교할 때 쓰는 거리 계산 방식
            # 유클리드 거리: 두 점 사이의 직선거리로 유사도를 비교하겠다
            collection_metadata={"hnsw:space": "l2"},
        )
    else:
        print(f"[INFO] ChromaDB 구축 중 (최초 1회, CPU 작업으로 시간이 걸릴 수 있습니다): {CHROMA_DIR}")
        db = Chroma.from_documents(

            # 저장할 문서 조각들이야. 앞에서 RecursiveCharacterTextSplitter로 나눈 결과물
            documents=chunks,

            embedding=embeddings,
            persist_directory=CHROMA_DIR,
            collection_metadata={"hnsw:space": "l2"},
        )
        print("[INFO] ChromaDB 구축 및 저장 완료")

    # 질문과 가장 관련 있는 문서 청크를 상위 4개만 가져오겠다
    retriever = db.as_retriever(search_kwargs={"k": 4})
    print("[INFO] ChromaDB 로드 완료")
    return retriever


# ─────────────────────────────────────────
# 6. RAG Chain 구성
# ─────────────────────────────────────────
# "검색된 자료를 보고, 그 안에 있는 내용만 근거로 한국어 답변해"
# retriever가 찾은 문서들을 context로 정리하고,
# 사용자 질문 question과 함께 LLM에 넣어서,
# 최종적으로 context + question + answer를 함께 반환하는 RAG 체인을 만드는 코드
def build_rag_chain(chat_model, retriever):
    # LLM에게 보낼 대화 양식을 미리 만들어두는 도구
    # system: AI에게 주는 기본 규칙
    # user: 사용자의 질문
    # assistant: AI 답변    
    from langchain_core.prompts import ChatPromptTemplate

    # LLM의 출력 결과를 문자열로 정리해주는 도구
    # 프롬프트 → LLM 답변 → 문자열로 변환
    from langchain_core.output_parsers import StrOutputParser

    # LangChain에서 여러 작업을 병렬처럼 묶어서 실행할 때 쓰는 도구
    # RunnableParallel: 하나의 입력 질문에서 context와 question을 동시에 만들어내기 위한 도구
    # RunnablePassthrough: 입력값을 그대로 통과시키는 도구
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough

    # 성능 향상을 위한 영문 시스템 프롬프트 + 한국어 답변 요청
    # 여기서는 RAG용 프롬프트 템플릿을 만들고 있어.
    RAG_prompt = ChatPromptTemplate(
        [
            (
                "system",
                """Answer the user's Question from the Context.
Keep your answer ground in the facts of the Context.
If the Context doesn't contain the facts to answer,
just output '답변할 수 없습니다.'
Please answer in Korean.""",
            ),
            
# user 메시지는 실제로 LLM에게 들어갈 사용자 입력 형식이야.
# ---
# | 변수           | 의미                       |
# | ------------ | ------------------------ |
# | `{context}`  | retriever가 검색해온 관련 문서 내용 |
# | `{question}` | 사용자가 입력한 질문              |
# ---
# Context: 
# 주제: 인공지능
# 인공지능은 인간의 학습 능력, 추론 능력...
# ---
# Question: 인공지능은 어떤 분야인가요?
            (
                "user",
                """Context: {context}
---
Question: {question}""",
            ),
        ]
    )

    def format_docs(docs):
        # retriever가 찾아온 문서 청크들을 LLM에게 넣기 좋은 문자열로 변환
        # 각 청크 앞에 Wikipedia 문서 제목을 붙이고, 청크끼리는 --- 로 구분
        return "\n---\n".join(
            "주제: " + doc.metadata["title"] + "\n" + doc.page_content
            for doc in docs
        )

    # LLM 답변 생성 체인
    # context와 question을 RAG_prompt에 넣고,
    # chat_model이 답변을 생성한 뒤,
    # 결과를 문자열로 변환
    rag_chain_from_docs = RAG_prompt | chat_model | StrOutputParser()

    # 최종 RAG 체인 구성
    # 입력 질문 하나로부터 context, question을 만들고,
    # 그 둘을 기반으로 answer까지 생성
    # ---
    # 사용자가 질문을 하나 입력해.
    # "트랜스포머가 뭐예요?"
    # 그러면 이 질문 하나를 가지고 두 가지 작업을 해.
    # ---
    # 1. "context": retriever | format_docs
    #   이건 사용자의 질문을 이용해서 관련 문서를 검색하고,
    #   그 문서들을 LLM용 context 문자열로 정리
    # 2. "question": RunnablePassthrough()
    #   이건 사용자 질문을 그대로 question 값으로 넘기는 부분이야.
    #   예를 들어 입력이:
    #   "트랜스포머가 뭐예요?"
    #   이면 그대로:
    #   question = "트랜스포머가 뭐예요?" 가 돼.
    rag_chain_with_source = RunnableParallel(
        {
            "context": retriever | format_docs,      # 질문으로 관련 문서 검색 후 context 문자열 생성
            "question": RunnablePassthrough(),       # 사용자 질문을 그대로 question에 전달
        }

    # 이 부분은 앞에서 만든 context와 question을 이용해서 추가로 answer 값을 만들어 붙이는 코드야.
    # {
    #     "context": "검색된 문서 내용...",
    #     "question": "트랜스포머가 뭐예요?"
    # }
    # 이런 코드가 아래 처럼 됨. 어떤 context를 보고 답변했는지도 같이 확인할 수 있는 구조야.
    # {
    #     "context": "검색된 문서 내용...",
    #     "question": "트랜스포머가 뭐예요?",
    #     "answer": "트랜스포머는 어텐션 메커니즘을 기반으로 한..."
    # }
    ).assign(
        answer=rag_chain_from_docs                   # context + question을 이용해 LLM 답변 생성
    )

    print("[INFO] RAG Chain 구성 완료")

    # context, question, answer를 함께 반환하는 RAG 체인 반환
    return rag_chain_with_source


# ─────────────────────────────────────────
# 7. 질의응답 실행
# ─────────────────────────────────────────
def run_queries(rag_chain):
    questions = [
        "인공지능은 어떤 분야인가요?",
        "트랜스포머가 뭐예요?",
        "GPT5는 언제 나와요?",          # 컨텍스트에 없으므로 '답변할 수 없습니다' 예상
        "알리바바의 거대 언어 모델 이름은?",
        "인공지능의 위험은 없나요?",
        "GPT-4o의 출시일은 언제인가요?",
    ]

    print("\n" + "=" * 60)
    print("RAG 질의응답 시작")
    print("=" * 60)

    for question in questions:
        print(f"\n[질문] {question}")
        result = rag_chain.invoke(question)
        print(f"[답변] {result['answer']}")
        print("-" * 60)


# ─────────────────────────────────────────
# main
# ─────────────────────────────────────────
def main():
    # Step 1. HuggingFace 로그인 (gated 모델 사용 시)
    login_huggingface()

    # Step 2. LLM 로드 (Qwen2.5-7B, 4-bit 양자화)
    chat_model = load_llm()

    # Step 3. 임베딩 모델 로드 (multilingual-e5-small, CPU)
    embeddings = load_embeddings()

    # Step 4. Wikipedia 데이터 수집 & 청킹
    chunks = load_and_split_documents()

    # Step 5. ChromaDB 벡터 스토어 구축
    retriever = build_vector_store(chunks, embeddings)

    # Step 6. RAG Chain 구성
    rag_chain = build_rag_chain(chat_model, retriever)

    # Step 7. 질의응답 실행
    run_queries(rag_chain)


if __name__ == "__main__":
    main()