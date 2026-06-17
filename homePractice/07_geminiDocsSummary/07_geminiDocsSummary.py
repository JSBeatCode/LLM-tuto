"""
Gemini를 이용한 문서 요약 애플리케이션
PDF 문서를 불러와 3가지 방식(Stuff / Map-Reduce / Refine)으로 요약합니다.

필요 패키지 설치:
    pip install openai langchain langchain-google-genai langchain-community tiktoken pymupdf python-dotenv tqdm

.env 파일 형식:
    GOOGLE_API_KEY=your_google_api_key_here
"""
import os  
# 운영체제(OS) 관련 기능을 사용하기 위한 모듈
# → 환경변수 읽기, 파일 경로 처리 등에 사용


from dotenv import load_dotenv  
# .env 파일에 저장된 환경변수를 Python으로 불러오는 함수
# → API 키를 코드에 직접 쓰지 않고 안전하게 관리할 수 있음


from tqdm import tqdm  
# 반복문 진행 상태를 보여주는 라이브러리 (프로그레스바)
# → Map-Reduce, Refine에서 "몇 개 처리 중인지" 시각적으로 보여줌


from langchain_google_genai import ChatGoogleGenerativeAI  
# Google Gemini 모델을 LangChain에서 사용하기 위한 클래스
# → LLM(대형 언어 모델) 역할 (요약, 생성 등)


from langchain_community.document_loaders import PyMuPDFLoader  
# PDF 파일을 읽어서 텍스트로 변환해주는 로더
# → 내부적으로 PyMuPDF 라이브러리를 사용해서 PDF 파싱


from langchain_core.documents import Document  
# LangChain에서 사용하는 문서 객체
# → 텍스트(page_content) + 메타데이터를 함께 저장하는 구조


from langchain_core.prompts import ChatPromptTemplate  
# LLM에게 전달할 프롬프트(지시문)를 구조화해서 만드는 클래스
# → system / user 역할을 나눠서 프롬프트 구성 가능


from langchain_core.output_parsers import StrOutputParser  
# LLM의 응답을 문자열(str) 형태로 변환하는 파서
# → 체인 결과를 깔끔한 텍스트로 받기 위해 사용


from langchain_text_splitters import RecursiveCharacterTextSplitter  
# 긴 텍스트를 여러 개의 작은 덩어리(chunk)로 나누는 도구
# → LLM 입력 제한(토큰 제한)을 피하기 위해 필수

# ── 환경 설정 ──────────────────────────────────────────────────────────────────

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(".env 파일에 GOOGLE_API_KEY가 설정되지 않았습니다.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 모델 설정
# gemini-2.0-flash-exp : 최신 모델 (요청 초과 시 gemini-1.5-flash 로 변경)
# MODEL = "gemini-2.0-flash-exp"
# MODEL = "gemini-2.0-flash"
MODEL = "gemini-3-flash-preview"

# PDF_PATH = "./example_paper.pdf"   # 요약할 PDF 파일 경로
PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_paper.pdf")

# 청크 설정
CHUNK_SIZE    = 10000
CHUNK_OVERLAP = 500


# ── Step 1. API 연결 확인 ──────────────────────────────────────────────────────

# def check_api():
#     import google.generativeai as genai
#     try:
#         models = list(genai.list_models())
#         print(f"✅ Google GenAI API Key 정상 설정 완료 (사용 가능한 모델 수: {len(models)})")
#     except Exception:
#         print("❌ API 키가 유효하지 않습니다!")
#         raise

# 수정 — 신버전 google.genai 패키지 사용
def check_api():
    from google import genai
    try:
        client = genai.Client()
        models = list(client.models.list())
        print(f"✅ Google GenAI API Key 정상 설정 완료 (사용 가능한 모델 수: {len(models)})")
    except Exception:
        print("❌ API 키가 유효하지 않습니다!")
        raise


# ── Step 2. PDF 불러오기 ───────────────────────────────────────────────────────

def load_pdf(path: str) -> Document:
    """PDF를 페이지별로 로드한 뒤 하나의 Document로 합칩니다."""
    loader = PyMuPDFLoader(path)
    pages  = loader.load()

    corpus = Document(page_content="")
    for page in pages:
        corpus.page_content += page.page_content + "\n"

    corpus.page_content = corpus.page_content.replace("\n\n", "\n")

    print(f"✅ PDF 로드 완료 | 총 {len(pages)}페이지 | 전체 문자 수: {len(corpus.page_content)}")
    return corpus


# ── Step 3-1. 요약 방법 1 : Stuff ─────────────────────────────────────────────

def summarize_stuff(llm: ChatGoogleGenerativeAI, corpus: Document) -> str:
    """전체 문서를 한 번에 프롬프트에 넣어 요약합니다.
    주의: 문서가 너무 길면 컨텍스트 길이 초과 에러가 발생할 수 있습니다."""

    print("\n[방법 1] Stuff 요약 시작...")

    prompt = ChatPromptTemplate([
        ("system", """주어진 논문의 내용을 읽고 한국어로 요약하세요.
요약은 1-3개의 문단과 문단별 5개의 문장으로 작성하세요.
"""),
        ("user", "{text}"),
    ])

    chain  = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": corpus.page_content})

    print("✅ Stuff 요약 완료")
    return result


# ── Step 3-2. 요약 방법 2 : Map-Reduce ────────────────────────────────────────

def summarize_map_reduce(llm: ChatGoogleGenerativeAI, chunks: list) -> str:
    """청크별로 요약(Map) 후 최종 요약(Reduce)을 수행합니다."""

    print("\n[방법 2] Map-Reduce 요약 시작...")

    # Map 단계 : 각 청크를 영어로 요약 (비용 절감)
    map_prompt = ChatPromptTemplate([
        ("system", """주어진 논문의 내용을 읽고 영어로 요약하세요.
요약은 1-3개의 문단과 문단별 5개의 문장으로 작성하세요.
"""),
        ("user", "{text}"),
    ])

    map_chain    = map_prompt | llm | StrOutputParser()
    raw_summaries = []

    print(f"  → Map 단계: 총 {len(chunks)}개 청크 처리 중...")
    for i in tqdm(range(len(chunks))):
        response = map_chain.invoke(chunks[i].page_content)
        raw_summaries.append(response)

    # Reduce 단계 : 요약본들을 한국어로 최종 요약
    print("  → Reduce 단계: 최종 요약 생성 중...")

    reduce_prompt = ChatPromptTemplate([
        ("system", """논문 요약문의 리스트가 주어집니다.
이를 읽고, 전체 주제를 포함하는 최종 요약을 한국어로 작성하세요.
요약은 5개의 문단과 문단별 4-8개의 문장으로 작성하세요.
답변은 한국어로 작성하세요.
"""),
        ("user", "{text}\n---\nSummary:\n"),
    ])

    reduce_chain = reduce_prompt | llm | StrOutputParser()
    result = reduce_chain.invoke("\n---\n".join(raw_summaries))

    print("✅ Map-Reduce 요약 완료")
    return result


# ── Step 3-3. 요약 방법 3 : Refine ────────────────────────────────────────────

def summarize_refine(llm: ChatGoogleGenerativeAI, chunks: list) -> str:
    """첫 청크로 초기 요약을 만들고, 이후 청크마다 요약을 점진적으로 보완합니다."""

    print("\n[방법 3] Refine 요약 시작...")

    # 첫 번째 청크로 초기 요약 생성
    first_prompt = ChatPromptTemplate([
        ("system", """주어진 논문의 내용을 읽고 한국어로 요약하세요.
요약은 1-3개의 문단과 문단별 5개의 문장으로 작성하세요.
"""),
        ("user", "{text}"),
    ])

    first_messages = first_prompt.format_messages(text=chunks[0].page_content)
    summary = llm.invoke(first_messages).content

    # 이후 청크마다 요약 보완
    refine_prompt = ChatPromptTemplate([
        ("system", """논문의 현재 시점까지의 요약이 주어집니다.
이를 읽고, 새롭게 주어지는 내용과 비교하여 논문 요약을 보완하세요.
요약은 5개의 문단과 문단별 4-8개의 문장으로 작성하세요.
답변은 한국어로 작성하세요.
"""),
        ("user", "현재 시점까지의 요약: {previous_summary}\n---\n새로운 내용: {new_text}"),
    ])

    refine_chain = refine_prompt | llm | StrOutputParser()

    print(f"  → 총 {len(chunks)}개 청크 순차 처리 중...")
    for i in tqdm(range(1, len(chunks))):
        summary = refine_chain.invoke({
            "previous_summary": summary,
            "new_text": chunks[i].page_content,
        })

    print("✅ Refine 요약 완료")
    return summary


# ── Main ───────────────────────────────────────────────────────────────────────

def main():

    # Step 1. API 확인
    check_api()

    # Step 2. LLM 초기화
    # 일반적으로 긴 출력·최종 요약에는 고성능 모델,
    # 짧은 출력(Map 단계 등)에는 경량 모델을 사용하는 것이 효율적입니다.
    llm = ChatGoogleGenerativeAI(model=MODEL)
    print(f"✅ 모델 초기화 완료: {MODEL}")

    # Step 3. PDF 로드
    corpus = load_pdf(PDF_PATH)

    # Step 4. 텍스트 청크 분리 (Map-Reduce / Refine 용)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents([corpus])
    print(f"✅ 텍스트 분할 완료 | 총 {len(chunks)}개 청크")

    # ── 요약 방법 선택 ──────────────────────────────────────────
    print("\n요약 방법을 선택하세요:")
    print("  1. Stuff       (전체 문서를 한 번에 요약)")
    print("  2. Map-Reduce  (청크별 요약 → 최종 통합 요약)")
    print("  3. Refine      (청크를 순서대로 보며 요약 누적)")
    print("  4. 모두 실행")

    choice = input("\n선택 (1/2/3/4): ").strip()

    results = {}

    if choice in ("1", "4"):
        results["Stuff"] = summarize_stuff(llm, corpus)

    if choice in ("2", "4"):
        results["Map-Reduce"] = summarize_map_reduce(llm, chunks)

    if choice in ("3", "4"):
        results["Refine"] = summarize_refine(llm, chunks)

    # ── 결과 출력 ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    for method, summary in results.items():
        print(f"\n📝 [{method}] 요약 결과\n")
        print(summary)
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()