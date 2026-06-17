"""
[Topic 5] 멀티 체인 연결 패턴
- chain1 결과 → chain2 입력으로 연결
- RunnableParallel + .assign() 으로 중간 과정 보존
- Lambda 함수로 외부 매개변수 추가 전달
- JsonOutputParser + 체인 간 데이터 전달 응용
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


# ── Step 1. 기본 멀티 체인 연결 ────────────────────────────
def run_basic_chain_connection(llm):
    """
    chain1 출력(나라 이름)을 {"country": chain1} 형태로 감싸서
    chain2의 {country} 변수에 자동 연결합니다.

    패턴: {"변수명": chain1} | prompt2 | llm | parser
    """
    print("\n" + "="*60)
    print("▶ Step 1. 기본 멀티 체인 연결")
    print("           chain1 출력 → chain2 입력으로 자동 연결")
    print("="*60)

    prompt1 = ChatPromptTemplate(
        ["잭슨빌은 어느 나라의 도시입니까? 나라 이름만 출력하세요."]
    )
    prompt2 = ChatPromptTemplate(
        ["{country}의 대표적인 인물 3명을 나열하세요. 인물의 이름만 출력하세요."]
    )

    chain1 = prompt1 | llm | StrOutputParser()

    # chain1의 출력값을 'country' 키에 담아 prompt2로 전달
    chain2 = (
        {"country": chain1}
        | prompt2
        | llm
        | StrOutputParser()
    )

    print("\n[체인 구조]")
    print("  chain1 : prompt1 | llm | StrOutputParser")
    print("  chain2 : {country: chain1} | prompt2 | llm | StrOutputParser")

    response = chain2.invoke({})

    print("\n[결과]")
    print(response)


# ── Step 2. RunnableParallel + .assign() ───────────────────
def run_parallel_with_assign(llm):
    """
    RunnableParallel로 chain1 결과(country)를 실행하고,
    .assign()으로 chain2 결과(people)를 추가합니다.
    → 중간값 country와 최종값 people을 모두 dict로 반환합니다.

    패턴: RunnableParallel(key=chain1).assign(key2=chain2)
    """
    print("\n" + "="*60)
    print("▶ Step 2. RunnableParallel + .assign()")
    print("           중간 결과(country) + 최종 결과(people) 함께 출력")
    print("="*60)

    prompt1 = ChatPromptTemplate(
        ["잭슨빌은 어느 나라의 도시입니까? 나라 이름만 출력하세요."]
    )
    prompt2 = ChatPromptTemplate(
        ["{country}의 대표적인 인물 3명을 나열하세요. 인물의 이름만 출력하세요."]
    )

    chain1 = prompt1 | llm | StrOutputParser()
    chain2 = prompt2 | llm | StrOutputParser()

    # country 결과를 유지하면서, 그 값을 chain2에 자동 전달해 people 추가
    chain3 = RunnableParallel(country=chain1).assign(people=chain2)

    print("\n[체인 구조]")
    print("  chain3 : RunnableParallel(country=chain1).assign(people=chain2)")

    response = chain3.invoke({})

    print("\n[결과 타입]:", type(response))
    print("\n[결과]")
    print(f"  country : {response['country']}")
    print(f"  people  : {response['people']}")


# ── Step 3. Lambda로 외부 매개변수 추가 전달 ───────────────
def run_lambda_extra_param(llm):
    """
    invoke()로 넘기는 dict에서 Lambda 함수로 값을 꺼내
    체인 중간에 추가 매개변수로 전달합니다.

    패턴: RunnableParallel(a=chain1, b=lambda x: x['key'])
    """
    print("\n" + "="*60)
    print("▶ Step 3. Lambda — invoke dict에서 외부 매개변수 추출")
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
            num=lambda x: x['num']    # invoke dict의 'num' 값을 그대로 추출
        )
        | prompt2
        | llm
        | StrOutputParser()
    )

    print("\n[체인 구조]")
    print("  RunnableParallel(country=chain1, num=lambda x: x['num'])")
    print("  → prompt2 | llm | StrOutputParser")

    response = chain2.invoke({"city": "잭슨빌", "num": "3"})

    print("\n[결과]")
    print(response)


# ── Step 4. Lambda + .assign() — 중간 과정 모두 보존 ────────
def run_lambda_with_assign(llm):
    """
    Lambda로 외부 매개변수(num)를 추출하고,
    .assign()으로 최종 결과(res)를 추가해
    country / num / res 세 가지를 모두 반환합니다.
    """
    print("\n" + "="*60)
    print("▶ Step 4. Lambda + .assign() — 중간 과정 전체 보존")
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

    chain3 = (
        RunnableParallel(country=chain1, num=lambda x: x['num'])
        .assign(res=chain4)
    )

    print("\n[체인 구조]")
    print("  RunnableParallel(country=chain1, num=lambda x: x['num'])")
    print("  .assign(res=chain4)")

    response = chain3.invoke({"city": "부산", "num": "3"})

    print("\n[결과 타입]:", type(response))
    print("\n[결과]")
    print(f"  country : {response['country']}")
    print(f"  num     : {response['num']}")
    print(f"  res     : {response['res']}")


# ── Step 5. JsonOutputParser + 체인 간 데이터 전달 ──────────
def run_json_chain(llm):
    """
    chain1이 JsonOutputParser로 dict를 반환하면,
    그 dict의 key가 chain2 프롬프트의 변수에 자동 매핑됩니다.
    별도의 변수 감싸기 없이 자동 연결됩니다.

    패턴: chain1(JsonOutputParser 반환) | prompt2 | llm | parser
    """
    print("\n" + "="*60)
    print("▶ Step 5. JsonOutputParser + 체인 간 데이터 전달 응용")
    print("           JSON key → 다음 프롬프트 변수에 자동 매핑")
    print("="*60)

    prompt1 = ChatPromptTemplate(
        ["영화 배우와 대표작을 하나 나열하세요. "
         "json 형식으로 출력하고, 각 항목은 actor, movie로 표시하세요."]
    )
    prompt2 = ChatPromptTemplate(
        ["{actor}는 {movie}에서 어떤 역할을 했습니까?"]
    )

    # chain1: {'actor': '...', 'movie': '...'} 형태의 dict 반환
    chain1 = prompt1 | llm | JsonOutputParser()

    # chain1의 dict key(actor, movie)가 prompt2의 변수에 자동 매핑
    chain2 = chain1 | prompt2 | llm | StrOutputParser()

    print("\n[체인 구조]")
    print("  chain1 : prompt1 | llm | JsonOutputParser  → {'actor':..., 'movie':...}")
    print("  chain2 : chain1 | prompt2 | llm | StrOutputParser")
    print("           ↑ JSON key가 {actor}, {movie} 변수에 자동 매핑")

    response = chain2.invoke({})

    print("\n[결과]")
    print(response)


# ── main ───────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  [Topic 5] 멀티 체인 연결 패턴")
    print("=" * 60)

    llm = init_llm()

    # # Step 1: 기본 멀티 체인 연결
    # run_basic_chain_connection(llm)

    # # Step 2: RunnableParallel + .assign()
    # run_parallel_with_assign(llm)

    # Step 3: Lambda로 외부 매개변수 추가 전달
    run_lambda_extra_param(llm)

    # Step 4: Lambda + .assign() 중간 과정 전체 보존
    run_lambda_with_assign(llm)

    # Step 5: JsonOutputParser + 체인 간 데이터 전달 응용
    run_json_chain(llm)

    print("\n" + "="*60)
    print("  실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()