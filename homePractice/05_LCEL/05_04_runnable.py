"""
[Topic 4] Runnables — 데이터 흐름 제어
- RunnablePassthrough : 이전 체인 출력을 그대로 다음 단계로 전달
- RunnableParallel    : 여러 체인을 병렬 실행 후 결과를 dict로 합침
- .assign()           : 기존 결과에 새 체인 결과 추가
- Lambda              : dict 입력에서 특정 값만 추출하여 다음 체인에 전달
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# ── 환경 설정 ──────────────────────────────────────────────
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-4o-mini"


# ── LLM 초기화 ─────────────────────────────────────────────
def init_llm():
    return ChatOpenAI(temperature=0.5, model=MODEL, max_tokens=1000)


# ── Step 1. RunnablePassthrough ────────────────────────────
def run_passthrough(llm):
    """
    RunnablePassthrough: 직전 체인의 출력을 그대로 다음 단계로 전달합니다.
    여기서는 StrOutputParser()의 결과(문자열)를
    {'answer': ...} 형태의 dict로 감싸서 전달합니다.
    """
    print("\n" + "="*60)
    print("▶ Step 1. RunnablePassthrough — 이전 출력 그대로 전달")
    print("="*60)

    prompt1 = ChatPromptTemplate(["{director}의 대표 작품은 무엇입니까?"])

    chain1 = (
        prompt1
        | llm
        | StrOutputParser()
        | {'answer': RunnablePassthrough()}  # 이전 출력 → dict의 'answer' 키로 전달
    )

    response = chain1.invoke("스티븐 스필버그")

    print("\n[결과 타입]:", type(response))
    print("\n[결과]")
    print(f"  answer: {response['answer']}")


# ── Step 2. RunnableParallel — 병렬 체인 실행 ──────────────
def run_parallel_basic(llm):
    """
    RunnableParallel: 서로 다른 체인을 동시에 병렬 실행하고
    결과를 하나의 dict로 합칩니다.
    """
    print("\n" + "="*60)
    print("▶ Step 2. RunnableParallel — 병렬 체인 실행")
    print("="*60)

    prompt1 = ChatPromptTemplate(["색깔을 하나 알려주세요, 색깔만 출력하세요."])
    prompt2 = ChatPromptTemplate(["음식을 하나 알려주세요, 음식만 출력하세요."])

    chain1 = prompt1 | llm | StrOutputParser()
    chain2 = prompt2 | llm | StrOutputParser()

    # chain1, chain2를 병렬로 실행 → 결과를 {'color': ..., 'food': ...} 로 합침
    chain3 = RunnableParallel(color=chain1, food=chain2)

    response = chain3.invoke({})

    print("\n[결과 타입]:", type(response))
    print("\n[결과]")
    print(f"  color : {response['color']}")
    print(f"  food  : {response['food']}")


# ── Step 3. 체인 간 연결 — chain1 결과 → chain2 입력 ────────
def run_chain_connection(llm):
    """
    chain1의 출력을 chain2의 입력으로 자연스럽게 연결합니다.
    dict 감싸기로 변수명을 매핑합니다.
    """
    print("\n" + "="*60)
    print("▶ Step 3. 체인 간 연결 — chain1 출력 → chain2 입력")
    print("="*60)

    prompt1 = ChatPromptTemplate(
        ["잭슨빌은 어느 나라의 도시입니까? 나라 이름만 출력하세요."]
    )
    prompt2 = ChatPromptTemplate(
        ["{country}의 대표적인 인물 3명을 나열하세요. 인물의 이름만 출력하세요."]
    )

    chain1 = prompt1 | llm | StrOutputParser()

    # chain1 결과(나라 이름)를 'country' 키에 담아 prompt2로 연결
    chain2 = (
        {"country": chain1}
        | prompt2
        | llm
        | StrOutputParser()
    )

    response = chain2.invoke({})

    print("\n[결과]")
    print(response)


# ── Step 4. .assign() — 기존 결과에 새 결과 추가 ───────────
def run_assign(llm):
    """
    .assign(): RunnableParallel 결과에 새 체인 결과를 추가합니다.
    chain1의 결과(country)를 보존하면서, chain2 결과(people)를 추가합니다.
    """
    print("\n" + "="*60)
    print("▶ Step 4. .assign() — 기존 결과에 새 체인 결과 추가")
    print("="*60)

    prompt1 = ChatPromptTemplate(
        ["잭슨빌은 어느 나라의 도시입니까? 나라 이름만 출력하세요."]
    )
    prompt2 = ChatPromptTemplate(
        ["{country}의 대표적인 인물 3명을 나열하세요. 인물의 이름만 출력하세요."]
    )

    chain1 = prompt1 | llm | StrOutputParser()
    chain2 = prompt2 | llm | StrOutputParser()

    # RunnableParallel로 country 먼저 구하고,
    # .assign()으로 people 결과를 추가
    chain3 = RunnableParallel(country=chain1).assign(people=chain2)

    response = chain3.invoke({})

    print("\n[결과 타입]:", type(response))
    print("\n[결과]")
    print(f"  country : {response['country']}")
    print(f"  people  : {response['people']}")


# ── Step 5. Lambda — dict 값 추출 ─────────────────────────
def run_lambda(llm):
    """
    Lambda 함수: invoke()에서 전달된 dict에서 특정 값을 꺼내
    다음 체인에 전달합니다.
    """
    print("\n" + "="*60)
    print("▶ Step 5. Lambda — dict 값 추출로 다음 체인에 전달")
    print("="*60)

    prompt1 = ChatPromptTemplate(
        ["{city}는 어느 나라의 도시인가요? 나라 이름만 출력하세요."]
    )
    prompt2 = ChatPromptTemplate(
        ["{country}의 유명한 인물은 누가 있나요? {num}명의 이름을 나열하세요. "
         "사람 이름만 ,로 구분하여 나열하세요."]
    )

    chain1 = prompt1 | llm | StrOutputParser()

    chain2 = (
        RunnableParallel(
            country=chain1,
            num=lambda x: x['num']   # invoke dict에서 'num' 값 추출
        )
        | prompt2
        | llm
        | StrOutputParser()
    )

    response = chain2.invoke({"city": "잭슨빌", "num": "3"})

    print("\n[결과]")
    print(response)


# ── Step 6. .assign() + Lambda — 중간 과정 모두 출력 ────────
def run_assign_with_lambda(llm):
    """
    체인을 분리하고 RunnableParallel + .assign()을 사용하면
    중간 결과(country, num)와 최종 결과(res)를 모두 출력할 수 있습니다.
    """
    print("\n" + "="*60)
    print("▶ Step 6. .assign() + Lambda — 중간 과정 포함 전체 출력")
    print("="*60)

    prompt1 = ChatPromptTemplate(
        ["{city}는 어느 나라의 도시인가요? 나라 이름만 출력하세요."]
    )
    prompt2 = ChatPromptTemplate(
        ["{country}의 유명한 인물은 누가 있나요? {num}명의 이름을 나열하세요. "
         "사람 이름만 ,로 구분하여 나열하세요."]
    )

    chain1 = prompt1 | llm | StrOutputParser()
    chain4 = prompt2 | llm | StrOutputParser()

    # country와 num을 병렬로 준비 → .assign()으로 최종 결과(res) 추가
    chain3 = (
        RunnableParallel(country=chain1, num=lambda x: x['num'])
        .assign(res=chain4)
    )

    response = chain3.invoke({"city": "부산", "num": "3"})

    print("\n[결과 타입]:", type(response))
    print("\n[결과]")
    print(f"  country : {response['country']}")
    print(f"  num     : {response['num']}")
    print(f"  res     : {response['res']}")


# ── Step 7. JsonOutputParser + 체인 연결 응용 ──────────────
def run_json_chain_connection(llm):
    """
    JsonOutputParser를 사용해 chain1의 구조화된 결과를
    chain2의 입력으로 자동 연결합니다.
    """
    print("\n" + "="*60)
    print("▶ Step 7. JsonOutputParser + 체인 간 연결 응용")
    print("="*60)

    prompt1 = ChatPromptTemplate(
        ["영화 배우와 대표작을 하나 나열하세요. "
         "json 형식으로 출력하고, 각 항목은 actor, movie로 표시하세요."]
    )
    prompt2 = ChatPromptTemplate(
        ["{actor}는 {movie}에서 어떤 역할을 했습니까?"]
    )

    # chain1: JSON으로 {'actor': ..., 'movie': ...} 반환
    chain1 = prompt1 | llm | JsonOutputParser()

    # chain2: chain1 결과를 그대로 prompt2 변수에 매핑
    chain2 = chain1 | prompt2 | llm | StrOutputParser()

    response = chain2.invoke({})

    print("\n[결과]")
    print(response)


# ── main ───────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  [Topic 4] Runnables — 데이터 흐름 제어")
    print("=" * 60)

    llm = init_llm()

    # Step 1: RunnablePassthrough
    run_passthrough(llm)

    # Step 2: RunnableParallel 기본
    run_parallel_basic(llm)

    # Step 3: 체인 간 연결
    run_chain_connection(llm)

    # Step 4: .assign()
    run_assign(llm)

    # # Step 5: Lambda
    # run_lambda(llm)

    # # Step 6: .assign() + Lambda (중간 과정 포함)
    # run_assign_with_lambda(llm)

    # # Step 7: JsonOutputParser + 체인 연결 응용
    # run_json_chain_connection(llm)

    print("\n" + "="*60)
    print("  실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()