"""
LangChain Tool과 Agent 실습 (LCEL 방식 - 최신 권장)
------------------------------------------------------
- 4개의 Tool 정의 및 테스트 (Wikipedia, Tavily 웹검색, Python REPL, 커스텀 곱셈)
- LLM에 Tool Binding (llm.bind_tools)
- LCEL 기반 수동 Agent 루프 (AgentExecutor 대체)
"""

import os
import warnings

# DeprecationWarning 억제 (langchain-community, langchain-experimental sunset 경고)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

# ── .env 로드 ──────────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 없습니다.")
if not TAVILY_API_KEY:
    raise ValueError(".env 파일에 TAVILY_API_KEY가 없습니다.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["USER_AGENT"]     = "MyCustomAgent"

# ── imports ────────────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, Tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Wikipedia (standalone 패키지 미출시 → community 유지)
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Tavily (standalone 패키지 사용)
from langchain_tavily import TavilySearch

# Python REPL
from langchain_experimental.tools.python.tool import PythonREPLTool


# ══════════════════════════════════════════════════════════════════════════════
# 1. LLM 초기화
# ══════════════════════════════════════════════════════════════════════════════
def init_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", max_tokens=1024, temperature=0.1)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Tool 정의
# ══════════════════════════════════════════════════════════════════════════════
def build_tools() -> tuple:
    # ── (1) Wikipedia 검색 툴 ──────────────────────────────────────────────────
    api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=5000)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    # ── (2) Tavily 웹 검색 툴 (langchain-tavily 독립 패키지) ───────────────────
    web_search = TavilySearch(max_results=3)

    # ── (3) Python REPL 툴 ────────────────────────────────────────────────────
    repl_tool = PythonREPLTool()

    # ── (4) 커스텀 곱셈 툴 (@tool 데코레이터) ──────────────────────────────────
    @tool
    def multiply(x: int, y: int) -> int:
        "x와 y를 입력받아, x와 y를 곱한 결과를 반환합니다."
        return x * y

    return wiki, web_search, repl_tool, multiply


# ══════════════════════════════════════════════════════════════════════════════
# 3. Tool 개별 테스트
# ══════════════════════════════════════════════════════════════════════════════
def test_tools(wiki, web_search, repl_tool, multiply) -> None:
    print("\n" + "=" * 60)
    print("[Tool 개별 테스트]")
    print("=" * 60)

    print("\n▶ Wikipedia 검색 - 'A.I'")
    result = wiki.invoke({"query": "A.I"})
    print(result[:500], "...(생략)")

    print("\n▶ Tavily 웹 검색 - 'Claude 3.5 Sonnet'")
    result = web_search.invoke({"query": "Claude 3.5 Sonnet"})
    # TavilySearch 결과는 dict → results 키 안에 리스트
    results_list = result.get("results", result) if isinstance(result, dict) else result
    for r in results_list[:3]:
        url = r.get("url", r) if isinstance(r, dict) else r
        print(f"  - {url}")

    print("\n▶ Python REPL 실행 - 'for i in range(10): print(i)'")
    example_code = "for i in range(10): print(i)"
    result = repl_tool.invoke({"input": example_code})  # PythonREPLTool은 input 키 사용
    print(result)

    print("\n▶ multiply 툴 - 10 × 99")
    result = multiply.invoke({"x": 10, "y": 99})
    print(f"  결과: {result}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. LLM Tool Binding 테스트
# ══════════════════════════════════════════════════════════════════════════════
def test_tool_binding(llm: ChatOpenAI, tools: list) -> None:
    print("\n" + "=" * 60)
    print("[LLM Tool Binding 테스트]")
    print("=" * 60)

    llm_with_tools = llm.bind_tools(tools)

    print("\n▶ '29392 * 23919는 뭐야?' → tool_calls 확인")
    response = llm_with_tools.invoke("29392 * 23919는 뭐야?")
    print(f"  tool_calls: {response.tool_calls}")

    print("\n▶ '양자 컴퓨터의 정의가 뭐야?' → tool_calls 확인")
    response = llm_with_tools.invoke("양자 컴퓨터의 정의가 뭐야?")
    print(f"  tool_calls: {response.tool_calls}")

    # tool_calls 결과를 실제 툴에 직접 전달
    wiki, _, _, multiply = tools[1], tools[2], tools[3], tools[0]

    print("\n▶ LLM이 선택한 인자로 multiply 직접 실행")
    args = llm_with_tools.invoke("29392 * 23919는 뭐야?").tool_calls[0]["args"]
    result = multiply.invoke(args)
    print(f"  결과: {result}")

    print("\n▶ LLM이 선택한 인자로 wikipedia 직접 실행")
    args = llm_with_tools.invoke("Quantum Computing의 정의가 뭐야?").tool_calls[0]["args"]
    result = wiki.invoke(args)
    print(result[:500], "...(생략)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. LCEL 기반 Agent 루프 (AgentExecutor 대체)
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """당신은 주어진 툴을 활용해 사용자의 질문에 정확하게 답변하는 AI 어시스턴트입니다.
툴이 필요하면 반드시 툴을 사용하고, 툴 결과를 바탕으로 최종 답변을 생성하세요."""

def build_tool_map(tools: list) -> dict:
    """툴 이름 → 툴 객체 매핑 딕셔너리 생성"""
    return {t.name: t for t in tools}


def run_agent(llm_with_tools: ChatOpenAI, tool_map: dict,
              user_input: str, max_iterations: int = 10) -> str:
    """
    LCEL 수동 Agent 루프
    - LLM 호출 → tool_calls 있으면 툴 실행 → 결과를 ToolMessage로 추가 → 반복
    - tool_calls 없으면 최종 답변으로 종료
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]

    for i in range(max_iterations):
        print(f"\n  [루프 {i + 1}] LLM 호출 중...")
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        # tool_calls가 없으면 최종 답변
        if not response.tool_calls:
            print("  → 최종 답변 생성")
            return response.content

        # tool_calls가 있으면 각 툴 실행
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_call_id = tc["id"]

            print(f"  → 툴 호출: {tool_name}({tool_args})")

            if tool_name not in tool_map:
                tool_result = f"오류: '{tool_name}' 툴을 찾을 수 없습니다."
            else:
                try:
                    tool_result = str(tool_map[tool_name].invoke(tool_args))
                except Exception as e:
                    tool_result = f"툴 실행 오류: {e}"

            print(f"  → 툴 결과: {str(tool_result)[:200]}")

            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call_id,
                )
            )

    return "최대 반복 횟수에 도달했습니다."


def run_agent_examples(llm: ChatOpenAI, tools: list) -> None:
    print("\n" + "=" * 60)
    print("[Agent 실행 예시 - LCEL 방식]")
    print("=" * 60)

    llm_with_tools = llm.bind_tools(tools)
    tool_map = build_tool_map(tools)

    examples = [
        "Open Source Multimodal LLM Model 추천해줘",
        "레오나르도 디카프리오의 출생년도를 찾은 뒤, 각 숫자를 순서대로 곱해줘.",
        "원주율을 30자리까지 출력해줘.",
    ]

    for idx, query in enumerate(examples, 1):
        print(f"\n{'─' * 60}")
        print(f"▶ 예시 {idx}: {query}")
        answer = run_agent(llm_with_tools, tool_map, query)
        print(f"\n  최종 답변: {answer}")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # 1. LLM
    llm = init_llm()

    # 2. Tool 생성
    wiki, web_search, repl_tool, multiply = build_tools()
    tools = [multiply, wiki, web_search, repl_tool]

    # 3. 각 Tool 개별 테스트
    test_tools(wiki, web_search, repl_tool, multiply)

    # 4. LLM Tool Binding 테스트
    test_tool_binding(llm, tools)

    # 5. LCEL Agent 실행
    run_agent_examples(llm, tools)


if __name__ == "__main__":
    main()