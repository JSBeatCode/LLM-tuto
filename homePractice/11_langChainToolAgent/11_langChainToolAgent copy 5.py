"""
LangChain Tool과 Agent 실습 (LCEL 방식 - 최신 권장)
------------------------------------------------------
- 4개의 Tool 정의 및 테스트 (Wikipedia, Tavily 웹검색, Python REPL, 커스텀 곱셈)
- LLM에 Tool Binding (llm.bind_tools)
- LCEL 기반 수동 Agent 루프 (AgentExecutor 대체)
"""

import os
import warnings
import requests

# DeprecationWarning 억제 (langchain-community, langchain-experimental sunset 경고)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

# ── .env 로드 ──────────────────────────────────────────────────────────────────
# load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
load_dotenv(override=True)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# print('debug1-1', os.path.join(os.path.dirname(os.path.abspath(__file__))))
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print('debug1-2', BASE_DIR)
# load_dotenv(os.path.join(BASE_DIR, ".env"))
# load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# 임시 진단 코드
print("ENV PATH:", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# print('debug1-3', OPENAI_API_KEY)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 없습니다.")
if not TAVILY_API_KEY:
    raise ValueError(".env 파일에 TAVILY_API_KEY가 없습니다.")
# print('debug1-3', os.environ["OPENAI_API_KEY"])
# print('debug1-4', OPENAI_API_KEY)
# print('debug1-5', os.getenv("TAVILY_API_KEY"))
# print('debug1-6', TAVILY_API_KEY)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["USER_AGENT"]     = "MyCustomAgent"

# ── imports ────────────────────────────────────────────────────────────────────
# OpenAI의 Chat 모델을 LangChain에서 사용할 수 있게 해주는 클래스
# GPT-4o, GPT-4.1, GPT-5 등의 채팅형 모델을 호출할 때 사용
from langchain_openai import ChatOpenAI

# Tool 생성 관련 클래스 및 데코레이터
# @tool : 일반 Python 함수를 Tool로 변환
# Tool  : Tool 객체를 직접 생성할 때 사용
from langchain_core.tools import tool, Tool

# LangChain에서 사용하는 메시지 객체들
# HumanMessage : 사용자의 질문
# AIMessage    : LLM의 응답
# ToolMessage  : Tool 실행 결과를 LLM에게 전달하는 메시지
# SystemMessage: LLM의 역할 및 행동 규칙을 정의하는 시스템 프롬프트
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)

# Tavily 웹 검색 Tool
# 최신 인터넷 검색 결과를 얻기 위한 Tool
# Wikipedia와 달리 실시간 정보를 조회할 수 있음
# (예: 최신 뉴스, 최근 출시된 모델, 현재 시점의 정보 등)
from langchain_tavily import TavilySearch

# Python 코드를 실행할 수 있는 Tool
# Agent가 계산, 데이터 처리, 반복문 수행 등을 위해 Python을 직접 실행할 수 있음
# 예:
#   "원주율 30자리 출력"
#   "10000개의 난수를 생성해서 평균 계산"
#   "CSV 파일 분석"
#
# 주의:
# 실제 Python 코드가 실행되므로 보안 측면에서 신중히 사용해야 함
from langchain_experimental.tools.python.tool import PythonREPLTool


# ══════════════════════════════════════════════════════════════════════════════
# 1. LLM 초기화
# ══════════════════════════════════════════════════════════════════════════════
def init_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", max_tokens=1024, temperature=0.1)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Tool 정의
# ══════════════════════════════════════════════════════════════════════════════
# LLM은 내부적으로
# 사용 가능한 Tool 목록
# 1. wikipedia
# 2. tavily
# 3. python_repl
# 4. multiply
# 를 전달받음.
# 예를 들어
# 질문:
# 29392 * 23919는?
# LLM 판단:
# multiply Tool 설명 발견
# ↓
# 곱셈 Tool 사용
def build_tools() -> tuple:
    # ── (1) Wikipedia 검색 툴 (requests 직접 호출 - User-Agent 차단 우회) ────────
    # query 입력
    #     ↓
    # 1차 호출
    # ↓
    # 검색어로 위키 문서 제목 찾기
    # ↓
    # 2차 호출
    # ↓
    # 찾은 문서 제목으로 본문 요약 가져오기
    # ↓
    # 결과 반환
    def _search_wikipedia(query: str) -> str:
        """
        Wikipedia에서 검색 후 상위 3개 문서의 요약을 반환한다.

        Args:
            query (str):
                검색할 키워드

        Returns:
            str:
                검색 결과 요약 문자열
        """

        # Wikipedia API가 비정상적인 요청으로 판단하지 않도록
        # 일반 브라우저의 User-Agent를 설정
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        # 1단계: 검색으로 페이지 제목 목록 가져오기
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",     # 조회
            "list": "search",      # 검색 기능 사용
            "srsearch": query,     # 검색어
            "srlimit": 3,          # 최대 3건
            "format": "json",      # JSON 응답
        }

        # 검색 API 호출
        resp = requests.get(search_url, params=search_params, headers=headers, timeout=10)
        
        # HTTP 오류 발생 시 예외 발생
        resp.raise_for_status()

        # 검색 결과 추출
        # .get("query", {}): query 있으면 줘. 없으면 None 줘
        #   -> 예를 들어 API가 실패하면 query 가 null이면 에러가 나기 때문에 이 코드를 씀
        # 응답 JSON 예시
        #
        # {
        #   "query": {
        #       "search": [
        #           {...},
        #           {...}
        #       ]
        #   }
        # }
        #
        # 여기서 search 리스트 추출
        results = resp.json().get("query", {}).get("search", [])

        if not results:
            return f"'{query}'에 대한 Wikipedia 검색 결과가 없습니다."

        # 2단계: 상위 결과의 본문 요약 가져오기
        summaries = []
        # 상위 3개 결과 반복
        for item in results[:3]:
            # 문서 제목
            title = item["title"]

            # 해당 문서 본문 추출용 파라미터
            extract_params = {

                # query API 사용
                "action": "query",

                # extracts 기능 사용
                # 문서 내용 가져오기
                "prop": "extracts",

                # 문서 전체가 아니라
                # 서론 부분만 가져오기
                "exintro": True,

                # HTML 제거
                #
                # False면
                # <p>...</p>
                # 같은 태그 포함
                "explaintext": True,

                # 가져올 문서 제목
                "titles": title,

                # JSON 응답
                "format": "json",
            }

            # 문서 내용 조회 API 호출
            r = requests.get(search_url, params=extract_params, headers=headers, timeout=10)
            # HTTP 오류 발생 시 Exception 발생
            r.raise_for_status()
            # 응답 구조 예시
            #
            # {
            #   "query": {
            #      "pages": {
            #          "12345": {
            #              ...
            #          }
            #      }
            #   }
            # }
            #
            pages = r.json().get("query", {}).get("pages", {})
            # pages는 dict 구조
            #
            # {
            #   "12345": {...},
            #   "67890": {...}
            # }
            #
            # value만 순회 
            for page in pages.values():
                # page.get("extract","") -> extract 키가 있으면 그 값을 가져와라. 없으면 빈 문자열("")을 사용해라.
                # strip() -> 문자열 앞뒤의 공백, 스페이스, 탭, 개행(\n) 을 제거해준다.
                extract = page.get("extract", "").strip()
                # 본문이 존재하면 저장
                if extract:
                    # 최대 1500글자만 저장
                    #
                    # 너무 길면 토큰 낭비
                    summaries.append(f"[{title}]\n{extract[:1500]}")
        # ---------------------------
        # 최종 결과 반환
        # ---------------------------

        # summaries가 존재하면
        #
        # [문서1]
        # 내용...
        #
        # [문서2]
        # 내용...
        #
        # 형식으로 합쳐서 반환.
        # 문서는 찾았는데 extract가 없는 경우 else 처리.
        return "\n\n".join(summaries) if summaries else f"'{query}' 내용을 가져올 수 없습니다."

    wiki = Tool(
        name="wikipedia",
        # Tool 설명
        #
        # LLM이 Tool 선택 여부를 판단할 때 읽는 설명
        #
        # 사람이 읽으라고 쓰는 것이 아니라
        # GPT가 읽고 이해하는 프롬프트 역할
        description=(
            "위키피디아에서 정보를 검색합니다. "
            "인물, 개념, 역사적 사건 등 백과사전적 정보를 찾을 때 사용하세요. "
            "입력은 검색할 키워드 또는 문장입니다."
        ),
        func=_search_wikipedia,
    )

    # ── (2) Tavily 웹 검색 툴 (langchain-tavily 독립 패키지) ───────────────────
    # @tool로 만든 Tool은 아니지만, LangChain이 인식할 수 있는 Tool 객체(또는 Tool 인터페이스 구현체)야.
    # Tavily 개발자가 미리 만들어 놓은 Tool
    web_search = TavilySearch(max_results=3)

    # ── (3) Python REPL 툴 ────────────────────────────────────────────────────
    # @tool로 만든 Tool은 아니지만, LangChain이 인식할 수 있는 Tool 객체(또는 Tool 인터페이스 구현체)야.
    # 이미 Tool로 만들어진 클래스
    repl_tool = PythonREPLTool()

    # ── (4) 커스텀 곱셈 툴 (@tool 데코레이터) ──────────────────────────────────
    # 왜 multiply는 @tool로 만들까?
    # 둘 다 결국 Tool이야.
    # 내부적으로는 거의 이런 객체로 변환돼.
    # StructuredTool(
    #     name="multiply",
    #     description="x와 y를 입력받아..."
    # )
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
    result = wiki.invoke("A.I")
    print(result[:500], "...(생략)")

    print("\n▶ Tavily 웹 검색 - 'Claude 3.5 Sonnet'")
    # result 이런 형식의 출력결과: 
    # {
    #     "results": [
    #         {
    #             "title": "Claude 3.5 Sonnet",
    #             "url": "https://www.anthropic.com",
    #             "content": "..."
    #         },
    #         {
    #             "title": "...",
    #             "url": "...",
    #             "content": "..."
    #         }
    #     ]
    # }
    # TavilySearch 결과는 dict → results 키 안에 리스트
    result = web_search.invoke({"query": "Claude 3.5 Sonnet"})
    # isinstance(result, dict) -> 객체가 dict 타입인지 확인
    # result.get("results", result) 
    #     -> 딕셔너리에 "results" 키가 있으면 가져오고
    #     -> 없으면 원본 result 반환
    results_list = result.get("results", result) if isinstance(result, dict) else result
    # results_list 는 항상
    # [
    #     {...},
    #     {...},
    #     {...}
    # ]
    # 형태의 리스트가 되도록 정규화(normalize)
    for r in results_list[:3]:
        # 결과 항목이 dict라면 url 필드 추출
        #
        # 예:
        # {
        #     "title": "...",
        #     "url": "https://..."
        # }
        #
        # 결과 항목이 문자열이면 그대로 사용
        url = r.get("url", r) if isinstance(r, dict) else r
        print(f"  - {url}")

    print("\n▶ Python REPL 실행 - 'for i in range(10): print(i)'")
    example_code = "for i in range(10): print(i)"
    result = repl_tool.invoke(example_code)
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

    # ---이 부분에서 부터 직접 호출한다. 
    #  예를 들면
    # 직원: 계산기 써주세요.
    # 라고 말한 상태지
    # 실제로 계산기 버튼을 누른 건 아님.
    # 그래서 
    # 작성자가 직접 계산기 버튼을 눌러봄.
    print("\n▶ LLM이 선택한 인자로 multiply 직접 실행")
    args = llm_with_tools.invoke("29392 * 23919는 뭐야?").tool_calls[0]["args"]
    result = multiply.invoke(args)
    print(f"  결과: {result}")

    print("\n▶ LLM이 선택한 인자로 wikipedia 직접 실행")
    args = llm_with_tools.invoke("Quantum Computing의 정의가 뭐야?").tool_calls[0]["args"]
    # Tool 객체는 문자열 입력 → args에서 query/input 값 추출
    # args = {
    #     "query": "Quantum Computing"
    # }
    # 이렇게 올 수도 있고,
    # 어떤 Tool은
    # args = {
    #     "__arg1": "Quantum Computing"
    # }
    # 이렇게 올 수도 있어.
    query_str = args.get("query", args.get("__arg1", str(args)))
    result = wiki.invoke(query_str)
    print(result[:500], "...(생략)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. LCEL 기반 Agent 루프 (AgentExecutor 대체)
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """당신은 주어진 툴을 활용해 사용자의 질문에 정확하게 답변하는 AI 어시스턴트입니다.
툴이 필요하면 반드시 툴을 사용하고, 툴 결과를 바탕으로 최종 답변을 생성하세요."""

def build_tool_map(tools: list) -> dict:
    """
    Tool 객체 리스트를 받아

    {
        "tool이름": tool객체
    }

    형태의 딕셔너리로 변환한다.

    예:

    tools = [multiply, wiki]

    ↓

    {
        "multiply": multiply,
        "wikipedia": wiki
    }
    """
    """툴 이름 → 툴 객체 매핑 딕셔너리 생성"""
    return {t.name: t for t in tools}

# 실제로
# 1. LLM 호출
# 2. Tool 선택
# 3. Tool 실행
# 4. Tool 결과 전달
# 5. 최종 답변 생성
# 을 반복하는 진짜 Agent 역할은 전부 run_agent()가 하고 있기 때문이야.
def run_agent(llm_with_tools: ChatOpenAI, tool_map: dict,
              user_input: str, max_iterations: int = 10) -> str:
# 사용자 질문
#       ↓
# LLM 호출
#       ↓
# Wikipedia 검색 필요 판단
#       ↓
# Wikipedia Tool 호출
#       ↓
# 출생년도 발견
#       ↓
# LLM 재호출
#       ↓
# 곱셈 필요 판단
#       ↓
# multiply Tool 호출
#       ↓
# 곱셈 결과 획득
#       ↓
# LLM 재호출
#       ↓
# 최종 답변 생성
    """
    LCEL 수동 Agent 루프
    - LLM 호출 → tool_calls 있으면 툴 실행 → 결과를 ToolMessage로 추가 → 반복
    - tool_calls 없으면 최종 답변으로 종료
    """
# [
#     SystemMessage(
#         "Tool을 활용해서 답변해라"
#     ),
#     HumanMessage(
#         "원주율을 30자리까지 출력해줘"
#     )
# ]    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]

    for i in range(max_iterations):
        print(f"\n  [루프 {i + 1}] LLM 호출 중...")
        # 사용자: 29392 * 23919는 뭐야?
        # LLM:
        # 잠깐만요.
        # 계산기를 써야겠네요.
        # 계산기에
        # 29392
        # 23919
        # 넣고 계산해주세요.
        # 이게 바로
        # tool_calls
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        # tool_calls가 없으면 최종 답변
        # tool_call = []
        # tool_call 가 있으면:
        #     AIMessage(
        #         content="",
        #         tool_calls=[
        #             {
        #                 "name":"wikipedia",
        #                 "args":{
        #                     "query":"Leonardo DiCaprio"
        #                 }
        #             }
        #         ]
        #     )        
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
                    # 실제 Tool 실행s
                    # 예
                    # tool_map["wikipedia"]
                    # ↓
                    # wiki
                    # ↓
                    # str(tool_map[tool_name].invoke 이것은 결국 == wiki.invoke(
                    # {"query":"Leonardo DiCaprio"}
                    # )
                    # 실행.                    
                    t = tool_map[tool_name]
                    # Tool 객체(단일 문자열 입력)는 딕셔너리에서 값을 추출해 전달
                    # StructuredTool(@tool 데코레이터)은 딕셔너리 그대로 전달
                    if isinstance(t, Tool):
                        if isinstance(tool_args, dict):
                            str_input = next(iter(tool_args.values()), "")
                        else:
                            str_input = str(tool_args)
                        tool_result = str(t.invoke(str_input))
                    else:
                        tool_result = str(tool_map[tool_name].invoke(tool_args))
                    # tool_result의 값: '1964 (레오의 출생년)'
                except Exception as e:
                    tool_result = f"툴 실행 오류: {e}"

            # 앞 200자만 출력.
            print(f"  → 툴 결과: {str(tool_result)[:200]}")

            # Tool 결과를 LLM에게 다음 loop에 전달하기 위한 메시지 생성
            # ---
            # Agent:
            # wiki.invoke(...)
            # 실행.
            # 결과:
            # Leonardo DiCaprio was born in 1974.
            # 획득.
            # 그런데 문제.
            # LLM은 이 결과를 모름.
            # 왜?
            # Tool 실행은 Agent가 했고
            # LLM은 못 봤으니까  
            # -> "Tool 실행 결과를 LLM에게 알려주는 코드"
            # ---
            # ToolMessage가 하는 일
            # Agent가 LLM에게 말하는 것.
            # LLM아
            # 네가 요청한 Wikipedia 결과가 왔어.
            # 결과는:
            # Leonardo DiCaprio was born in 1974.            
            messages.append(
                # ToolMessage(
                #     content="""
                #     Leonardo DiCaprio
                #     born 1974
                #     """,
                #     tool_call_id="call_abc123"
                # )              
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