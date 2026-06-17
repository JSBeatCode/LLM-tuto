"""
LangChain Tool과 Agent 실습
- 4개의 Tool 정의 및 테스트 (Wikipedia, Tavily 웹검색, Python REPL, 커스텀 곱셈)
- LLM에 Tool Binding
- Structured Chat Agent 구성 및 실행
"""

import os
from dotenv import load_dotenv

# ── .env 로드 ──────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 없습니다.")
if not TAVILY_API_KEY:
    raise ValueError(".env 파일에 TAVILY_API_KEY가 없습니다.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["USER_AGENT"]     = "MyCustomAgent"

# ── 공통 imports ───────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool, tool
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate


# ══════════════════════════════════════════════════════════
# 1. LLM 초기화
# ══════════════════════════════════════════════════════════
def init_llm():
    return ChatOpenAI(model="gpt-4o-mini", max_tokens=1024, temperature=0.1)


# ══════════════════════════════════════════════════════════
# 2. Tool 정의
# ══════════════════════════════════════════════════════════

def build_tools():
    # ── (1) Wikipedia 검색 툴 ─────────────────────────────
    api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=5000)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    # ── (2) Tavily 웹 검색 툴 ─────────────────────────────
    web_search = TavilySearchResults(max_results=3)

    # ── (3) Python REPL 툴 ───────────────────────────────
    python_repl = PythonREPL()
    repl_tool = Tool(
        name="python_repl",
        description=(
            "A Python shell. Use this to execute python commands. "
            "Input should be a valid python command. "
            "If you want to see the output of a value, you should print it out with `print(...)`."
        ),
        func=python_repl.run,
    )

    # ── (4) 커스텀 곱셈 툴 (@tool 데코레이터) ──────────────
    @tool
    def multiply(x: int, y: int) -> int:
        "x와 y를 입력받아, x와 y를 곱한 결과를 반환합니다."
        return x * y

    return wiki, web_search, repl_tool, multiply


# ══════════════════════════════════════════════════════════
# 3. Tool 개별 테스트
# ══════════════════════════════════════════════════════════

def test_tools(wiki, web_search, repl_tool, multiply):
    print("\n" + "=" * 60)
    print("[Tool 테스트]")
    print("=" * 60)

    # Wikipedia
    print("\n▶ Wikipedia 검색 - 'A.I'")
    result = wiki.invoke({"query": "A.I"})
    print(result[:500], "...(생략)")

    # Tavily 웹 검색
    print("\n▶ Tavily 웹 검색 - 'Claude 3.5 Sonnet'")
    result = web_search.invoke("Claude 3.5 Sonnet")
    for r in result:
        print(f"  - {r['url']}")

    # Python REPL
    print("\n▶ Python REPL 실행 - 'for i in range(10): print(i)'")
    example_code = "for i in range(10): print(i)"
    result = repl_tool.invoke({"input": example_code})
    print(result)

    # 커스텀 곱셈
    print("\n▶ multiply 툴 - 10 × 99")
    result = multiply.invoke({"x": 10, "y": 99})
    print(f"  결과: {result}")


# ══════════════════════════════════════════════════════════
# 4. LLM에 Tool Binding 테스트
# ══════════════════════════════════════════════════════════

def test_tool_binding(llm, tools):
    print("\n" + "=" * 60)
    print("[LLM Tool Binding 테스트]")
    print("=" * 60)

    llm_with_tools = llm.bind_tools(tools)

    # 곱셈 → multiply 선택 확인
    print("\n▶ '29392 * 23919는 뭐야?' → tool_calls 확인")
    response = llm_with_tools.invoke("29392 * 23919는 뭐야?")
    print(f"  tool_calls: {response.tool_calls}")

    # 백과사전 질문 → wikipedia 선택 확인
    print("\n▶ '양자 컴퓨터의 정의가 뭐야?' → tool_calls 확인")
    response = llm_with_tools.invoke("양자 컴퓨터의 정의가 뭐야?")
    print(f"  tool_calls: {response.tool_calls}")

    # tool_calls 결과를 실제 툴에 전달
    wiki, web_search, repl_tool, multiply = tools[1], tools[2], tools[3], tools[0]

    print("\n▶ LLM이 선택한 인자로 multiply 직접 실행")
    args = llm_with_tools.invoke("29392 * 23919는 뭐야?").tool_calls[0]["args"]
    result = multiply.invoke(args)
    print(f"  결과: {result}")

    print("\n▶ LLM이 선택한 인자로 wikipedia 직접 실행")
    args = llm_with_tools.invoke("Quantum Computing의 정의가 뭐야?").tool_calls[0]["args"]
    result = wiki.invoke(args)
    print(result[:500], "...(생략)")

    return llm_with_tools


# ══════════════════════════════════════════════════════════
# 5. Structured Chat Agent 구성 및 실행
# ══════════════════════════════════════════════════════════

def build_agent(llm, tools):
    agent_prompt = ChatPromptTemplate(messages=[
        ("system", """
최대한 정확히 질문에 답변하세요. 당신은 다음의 툴을 사용할 수 있습니다:
{tools}

action 키 (tool name)와 action_input 키를 포함하는 json 형태로 출력하세요.

action의 값은 "Final Answer" 또는 {tool_names} 중 하나여야 합니다.
반드시 하나의 json 형태만 출력하세요. 다음은 예시입니다.
```
{{

  "action": $TOOL_NAME,

  "action_input": $INPUT

}}
```

아래의 포맷으로 답변하세요.:

Question: 최종적으로 답변해야 하는 질문
Thought: 무엇을 해야 하는지를 항상 떠올리세요.
Action:
```
$JSON_BLOB
```
Observation: 액션의 실행 결과
... (이 Thought/Action/Observation 은 10번 이내로 반복될 수 있습니다.)

Thought: 이제 답을 알겠다!
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}
```
주어진 포맷을 잘 지켜야 하며, 한 번에 Thought와 Action을 동시에 쓰지 마세요.
Question, Thought, Action, Observation의 출력 뒤에는 항상 줄을 바꾸세요.

이제 입력이 주어집니다. json blob 형태로 출력해야 함을 명심하고,
Thought의 경우에는 항상 Thought: 로 시작해야 합니다.
포맷은 Action:```$JSON_BLOB``` 이후 Observation 입니다.
"""),
        ("user", """Question: {input}
Thought: {agent_scratchpad}"""),
    ])

    agent = create_structured_chat_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


def run_agent_examples(agent_executor):
    print("\n" + "=" * 60)
    print("[Agent 실행 예시]")
    print("=" * 60)

    # 예시 1: 웹 검색 활용
    print("\n▶ 예시 1: Open Source Multimodal LLM 추천")
    result = agent_executor.invoke({"input": "Open Source Multimodal LLM Model 추천해줘"})
    print(f"\n최종 답변: {result['output']}")

    # 예시 2: 웹 검색 + 곱셈 툴 연계
    print("\n▶ 예시 2: 레오나르도 디카프리오 출생년도 각 숫자 곱하기")
    result = agent_executor.invoke({"input": "레오나르도 디카프리오의 출생년도를 찾은 뒤, 각 숫자를 순서대로 곱해줘."})
    print(f"\n최종 답변: {result['output']}")

    # 예시 3: Python REPL 활용
    print("\n▶ 예시 3: 원주율 30자리 출력")
    result = agent_executor.invoke({"input": "원주율을 30자리까지 출력해줘."})
    print(f"\n최종 답변: {result['output']}")


# ══════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════

def main():
    # 1. LLM
    llm = init_llm()

    # 2. Tool 생성
    wiki, web_search, repl_tool, multiply = build_tools()
    tools = [multiply, wiki, web_search, repl_tool]

    # 3. 각 Tool 개별 테스트
    test_tools(wiki, web_search, repl_tool, multiply)

    # 4. LLM + Tool Binding 테스트
    test_tool_binding(llm, tools)

    # 5. Agent 구성 및 실행
    agent_executor = build_agent(llm, tools)
    run_agent_examples(agent_executor)


if __name__ == "__main__":
    main()