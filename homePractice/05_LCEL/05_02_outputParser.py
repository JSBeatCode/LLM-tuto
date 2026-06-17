"""
[Topic 2] Output Parser — 출력 형식 제어
- StrOutputParser  : 출력 결과를 문자열로 변환
- JsonOutputParser : 출력 결과를 JSON(dict)으로 변환
- Pydantic BaseModel로 JSON 스키마 고정
- with_structured_output() (Structured Output)
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

# ── 환경 설정 ──────────────────────────────────────────────
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL       = "gpt-4o-mini"
MODEL_4O    = "gpt-4o"          # with_structured_output()은 tool calling 지원 모델 필요


# ── Pydantic 스키마 정의 ────────────────────────────────────
class Recipe(BaseModel):
    name        : str       = Field(description="음식 이름")
    difficulty  : str       = Field(description="만들기의 난이도")
    origin      : str       = Field(description="원산지")
    ingredients : list[str] = Field(description="재료")
    instructions: list[str] = Field(description="조리법")
    tip         : str       = Field(description="조리 과정 팁")


# ── LLM 초기화 ─────────────────────────────────────────────
def init_llm(model: str = MODEL):
    return ChatOpenAI(temperature=0.5, model=model, max_tokens=1000)


# ── Step 1. StrOutputParser ────────────────────────────────
def run_str_output_parser(llm):
    """
    StrOutputParser: LLM 응답(AIMessage)을 순수 문자열로 변환
    """
    print("\n" + "="*60)
    print("▶ Step 1. StrOutputParser — 문자열 변환")
    print("="*60)

    recipe_template = ChatPromptTemplate([
        ('system', '당신은 전세계의 조리법을 아는 쉐프입니다.'),
        ('user',   '저는 {ingredient}를 이용한 환상적인 파인 다이닝을 만들고 싶습니다. 추천해주세요!')
    ])

    # Prompt | LLM | StrOutputParser
    recipe_chain = recipe_template | llm | StrOutputParser()

    response = recipe_chain.invoke({'ingredient': '연두부, 에너지바, 바나나'})

    print("\n[결과 타입]:", type(response))
    print("\n[결과]\n", response)


# ── Step 2. JsonOutputParser (스키마 없음) ─────────────────
def run_json_output_parser(llm):
    """
    JsonOutputParser: 출력을 JSON dict로 변환
    단, 스키마 미지정 시 실행마다 key 구조가 달라질 수 있음
    """
    print("\n" + "="*60)
    print("▶ Step 2. JsonOutputParser — JSON 변환 (스키마 없음)")
    print("="*60)

    json_parser = JsonOutputParser()

    print("\n[JsonOutputParser 포맷 지시문]")
    print(json_parser.get_format_instructions())

    recipe_template = ChatPromptTemplate([
        ('system', '당신은 전세계의 이색적인 퓨전 조리법의 전문가입니다.'),
        ('user',   '''저는 {ingredient}를 이용한 환상적인 퓨전 다이닝을 만들고 싶습니다. 추천해주세요!
    레시피에 대한 정보를 JSON 형식으로 출력해주세요.''')
    ])

    recipe_chain = recipe_template | llm | json_parser

    response = recipe_chain.invoke({'ingredient': '콜라'})

    print("\n[결과 타입]:", type(response))
    print("\n[결과]")
    for k, v in response.items():
        print(f"  {k}: {v}")


# ── Step 3. JsonOutputParser + Pydantic 스키마 ─────────────
def run_pydantic_json_parser(llm):
    """
    Pydantic BaseModel로 JSON 스키마를 고정 → 매번 동일한 key 구조 보장
    """
    print("\n" + "="*60)
    print("▶ Step 3. JsonOutputParser + Pydantic — 스키마 고정")
    print("="*60)

    parser = JsonOutputParser(pydantic_object=Recipe)

    print("\n[Pydantic 적용 포맷 지시문]")
    print(parser.get_format_instructions())

    recipe_template = ChatPromptTemplate([
        ('system', '당신은 전세계의 이색적인 퓨전 조리법의 전문가입니다.'),
        ('user',   '''저는 {ingredient}를 이용한 실험적인 음식을 만들고 싶습니다. 추천해주세요!
    레시피에 대한 정보를 JSON 형식으로 출력해주세요. 결과는 한국어로 작성하세요.
    {instruction}''')
    ])

    recipe_chain = recipe_template | llm | parser

    response = recipe_chain.invoke({
        'ingredient' : '생강',
        'instruction': parser.get_format_instructions()
    })

    print("\n[결과 타입]:", type(response))
    print("\n[결과]")
    for k, v in response.items():
        print(f"  {k}: {v}")


# ── Step 4. with_structured_output() ──────────────────────
def run_structured_output():
    """
    with_structured_output(): LangChain의 Structured Output 기능
    Tool Calling을 지원하는 gpt-4o / gpt-4o-mini 필요
    결과를 Pydantic 객체로 직접 반환 (파서 없이)
    """
    print("\n" + "="*60)
    print("▶ Step 4. with_structured_output() — Structured Output")
    print("="*60)

    # Structured Output은 tool calling 지원 모델 필요
    llm_4o = init_llm(MODEL_4O)

    structured_llm = llm_4o.with_structured_output(Recipe)
    response = structured_llm.invoke("생강으로 만들 수 있는 요리 레시피 알려주세요.")

    print("\n[결과 타입]:", type(response))
    print("\n[결과 — Pydantic 객체]")
    print(f"  음식 이름  : {response.name}")
    print(f"  난이도     : {response.difficulty}")
    print(f"  원산지     : {response.origin}")
    print(f"  재료       : {response.ingredients}")
    print(f"  조리법     : {response.instructions}")
    print(f"  팁         : {response.tip}")


# ── main ───────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  [Topic 2] Output Parser — 출력 형식 제어")
    print("=" * 60)

    llm = init_llm()

    # Step 1: StrOutputParser
    run_str_output_parser(llm)

    # Step 2: JsonOutputParser (스키마 없음)
    run_json_output_parser(llm)

    # Step 3: JsonOutputParser + Pydantic 스키마 고정
    run_pydantic_json_parser(llm)

    # Step 4: with_structured_output()
    run_structured_output()

    print("\n" + "="*60)
    print("  실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()