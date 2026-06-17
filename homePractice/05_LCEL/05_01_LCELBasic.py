"""
[Topic 1] LCEL 기본 구조 & 체인 구성
- | 파이프 연산자로 Prompt + LLM 연결
- invoke() 실행, 매개변수 1개 / 2개 체인 실습
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ── 환경 설정 ──────────────────────────────────────────────
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-4o-mini"


# ── Step 1. LLM 모델 초기화 ────────────────────────────────
def init_llm():
    llm = ChatOpenAI(
        temperature=0.5,
        model=MODEL,
        max_tokens=1000
    )
    return llm


# ── Step 2. 매개변수 1개짜리 체인 ──────────────────────────
def run_single_param_chain(llm):
    """
    topic 하나를 입력받아 영어 농담 + 한국어 설명을 출력하는 체인
    """
    print("\n" + "="*60)
    print("▶ Step 2. 매개변수 1개짜리 체인 (topic → 영어 농담)")
    print("="*60)

    fun_chat_template = ChatPromptTemplate([
        ('user', """Tell me an English joke about {topic}.
Then, explain in Korean why it is funny to English speakers.
Provide a Korean translation of the joke as well.""")
    ])

    # | 파이프 연산자로 Prompt + LLM 연결
    joke_chain = fun_chat_template | llm

    print(f"\n[체인 구조]\n{joke_chain}\n")

    # invoke() 로 실행 — dict 형식으로 전달
    response = joke_chain.invoke({'topic': 'eggs'})

    print("[결과]")
    print(response.content)


# ── Step 3. 매개변수 2개짜리 체인 ──────────────────────────
def run_double_param_chain(llm):
    """
    A, B 두 개의 매개변수를 받아 두 캐릭터의 대화를 생성하는 체인
    """
    print("\n" + "="*60)
    print("▶ Step 3. 매개변수 2개짜리 체인 (A, B → 두 캐릭터 대화)")
    print("="*60)

    prompt = ChatPromptTemplate([
        ('system', '당신은 재미있고 교훈적인 이야기를 씁니다.'),
        ('user', '{A}와 {B}가 만났을 때의 대화를 써 주세요.')
    ])

    chain = prompt | llm

    # invoke() — 매개변수 2개를 dict로 전달
    response = chain.invoke({'A': '햄릿', 'B': '슈퍼마리오'})

    print("[결과]")
    print(response.content)


# ── main ───────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  [Topic 1] LCEL 기본 구조 & 체인 구성")
    print("=" * 60)

    llm = init_llm()

    # Step 2: 매개변수 1개 체인
    run_single_param_chain(llm)

    # Step 3: 매개변수 2개 체인
    run_double_param_chain(llm)

    print("\n" + "="*60)
    print("  실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()