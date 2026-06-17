# =============================================================
# Topic 7 : Sequential Tool Call (parallel_tool_calls=False)
# - parallel_tool_calls=False 옵션으로 순차 실행 강제
# - Tool 호출 → 메시지 전달 → Tool 호출 → ... 반복 구조 이해
# - while tool_calls: 루프로 종료 조건 제어
# - msgs 리스트에 대화 히스토리를 누적하는 패턴
# - available_functions 딕셔너리로 tool-함수 매핑
# - Topic 6 (Parallel) 과의 구조 비교
# =============================================================

import openai
import requests
import json
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# =============================================================
# 0. 환경 설정
# =============================================================

load_dotenv(override=True)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"

# API 키 유효성 검증
try:
    client.models.list()
    print("OPENAI_API_KEY가 정상적으로 설정되어 있습니다.\n")
except Exception:
    print("API 키가 유효하지 않습니다! .env 파일을 확인해 주세요.")
    exit()


# =============================================================
# 1. 네이버 뉴스 API - get_news() 함수
#    (Sequential Tool Call 에서 한 번씩 순차 실행될 외부 함수)
# =============================================================

def get_news(query):
    """네이버 뉴스 API로 query 검색 후 포맷된 문자열 반환"""

    url = f"https://openapi.naver.com/v1/search/news.json?query={query}&display=30&sort=sim"
    headers = {
        'X-Naver-Client-Id':     os.getenv("NAVER_CLIENT_ID"),
        'X-Naver-Client-Secret': os.getenv("NAVER_CLIENT_SECRET")
    }

    response = requests.get(url, headers=headers)
    news_items = response.json().get('items', [])

    result = ""
    for item in news_items:
        result += f"제목: {item['title']}\n"
        result += f"URL: {item['link']}\n"
        result += f"내용: {item['description']}\n"
        result += "---\n"

    return result


# =============================================================
# 2. Tool 스키마 정의 (Topic 5, 6 과 동일)
# =============================================================

class Web_Search(BaseModel):
# 1️⃣ query: str
# 👉
# 변수 이름: query
# 타입: 문자열(str)
# 즉:
# "query는 문자열이다"
# 2️⃣ Field(description="...")
# 👉
# 이 변수에 대한 설명
# LLM한테 보여주는 가이드
# 즉:
# "query는 이렇게 써라 (키워드 규칙 등)"
    """query를 이용하여 뉴스 검색"""
    query: str = Field(description="""검색 키워드
규칙:
1. 최대 2개 단어로 구성
2. 불필요한 조사나 형용사 제외
3. 핵심 명사만 포함

예시:
- (좋음) "개봉영화", "영화"
- (나쁨) "새로 개봉한 영화", "요즘 인기있는 영화"
        """)

# 설명서를 LLM에게 전달
tools = [openai.pydantic_function_tool(Web_Search)]


# =============================================================
# 3. Sequential Tool Call 봇 함수 - news_bot_v3()
#
#    Topic 6 의 news_bot_v2() 와의 핵심 차이:
#    ┌──────────────────────────────────────────────────────────┐
#    │ news_bot_v2  │ parallel_tool_calls 미설정 (기본값 True)  │
#    │              │ → tool_calls 복수 반환 가능               │
#    │              │ → stream 파라미터 지원                    │
#    ├──────────────────────────────────────────────────────────┤
#    │ news_bot_v3  │ parallel_tool_calls = False               │
#    │              │ → tool_calls 항상 1개씩만 반환            │
#    │              │ → stream 파라미터 없음 (while 루프 때문)  │
#    └──────────────────────────────────────────────────────────┘
# =============================================================

def news_bot_v3(messages, model=MODEL):

    response = client.chat.completions.create(
        model=model,
        messages=messages,

        tools=tools,
        tool_choice='auto',

        temperature=0,
        max_tokens=1024,

        parallel_tool_calls=False,
        # False : 한 번의 응답에 tool_calls 를 1개씩만 반환
        #         → Tool 호출 → 결과 전달 → Tool 호출 → 결과 전달 → ... 순차 반복
        # True  : 한 번의 응답에 tool_calls 를 복수로 반환 (Topic 6 방식)
    )

    return response.choices[0].message


# =============================================================
# 4. parallel_tool_calls=False 동작 확인
#    - 복수 플랫폼 질문이어도 tool_calls 가 1개씩만 반환됨
# =============================================================

print("=" * 60)
print("[ 예제 1 ] parallel_tool_calls=False - tool_calls 1개씩 반환 확인")
print("=" * 60)

tool_call_result = news_bot_v3([
    {
        "role": "user",
        "content": "넷플릭스 왓챠 디즈니플러스 웨이브 신작 추천해줘."
    }
])

print(f"반환된 tool_calls 개수: {len(tool_call_result.tool_calls)}")  # 항상 1개
print(f"tool name      : {tool_call_result.tool_calls[0].function.name}")
print(f"tool arguments : {tool_call_result.tool_calls[0].function.arguments}")
print()


# =============================================================
# 5. Sequential Tool Call 핵심 처리 패턴 - news_qa_v3()
#
#    Topic 6 (Parallel, for 루프) 와의 구조 비교:
#    ┌──────────────────────────────────────────────────────────┐
#    │ Topic 6  │ if tool_calls: → for 루프 → 일괄 전달        │
#    │          │ tool_calls N개를 한 번에 처리                 │
#    ├──────────────────────────────────────────────────────────┤
#    │ Topic 7  │ while tool_calls: → 1개 처리 → 재호출 반복   │
#    │          │ tool_calls 1개씩 처리 후 msgs 에 누적         │
#    └──────────────────────────────────────────────────────────┘
#
#    while 루프 흐름:
#    ① msgs 에 현재 tool_call_result (assistant) 추가
#    ② tool_calls[0] 의 name, arguments 로 함수 실행
#    ③ tool 결과를 msgs 에 추가
#    ④ 업데이트된 msgs 전체로 다시 news_bot_v3() 호출
#    ⑤ 새 응답에 tool_calls 가 없으면 → while 탈출 → 최종 답변 출력
# =============================================================

def news_qa_v3(prompt, model=MODEL):
    """
    Sequential Tool Call 처리 함수
    - parallel_tool_calls=False 로 tool_calls 를 1개씩만 받음
    - while 루프로 tool_calls 가 없어질 때까지 순차 반복
    - msgs 리스트에 전체 대화 히스토리를 누적
    """

    print('Prompt:', prompt)

    # Web_Search라는 이름이 오면 get_news 함수를 실행해라
    available_functions = {'Web_Search': get_news}

    # msgs : 대화 히스토리를 누적하는 리스트
    # while 루프를 거칠 때마다 assistant + tool 메시지가 쌓임
    msgs = [
        {
            "role": "user",
            "content": prompt
        },
    ]

    # 첫 번째 호출
    tool_call_result = news_bot_v3(msgs)

    print('---')
    print('News_Bot: Call ', end='')

    # while 루프 : tool_calls 가 존재하는 동안 반복
    # parallel_tool_calls=False 이므로 매 루프마다 tool_calls 는 1개
    while tool_call_result.tool_calls:

        # ① 현재 assistant 응답(tool_calls 포함)을 히스토리에 추가
        msgs.append(tool_call_result)

        # LLM이 “이 함수 써라 + 이 값으로 써라”라고 준 데이터 꺼내는 코드
        # LLM 응답:
        # {
        # "tool_calls": [
        #     {
        #     "function": {
        #         "name": "Web_Search",
        #         "arguments": "{\"query\":\"넷플릭스\"}"
        #     }
        #     }
        # ]
        # }
        name      = tool_call_result.tool_calls[0].function.name
        arguments = tool_call_result.tool_calls[0].function.arguments
        print(name, arguments)

        # ② tool 이름과 arguments 로 실제 함수 실행
        search_result = available_functions[name](**json.loads(arguments))

        print('---')
        print('News_Bot:', end='')

        # ③ tool 결과를 히스토리에 추가
        msgs.append(
            {
                "role": "tool",
                "content": search_result,
                "tool_call_id": tool_call_result.tool_calls[0].id
            }
        )

        # ④ 누적된 msgs 전체로 다시 호출
        #    → GPT 가 다음 검색어를 판단하거나, 충분하면 최종 답변 반환
        tool_call_result = news_bot_v3(msgs, model=model)

    # ⑤ while 탈출 = tool_calls 없음 = 최종 content 답변 완성
    print(tool_call_result.content)
    print('\n')


# =============================================================
# 6. 실행 예제
# =============================================================

prompt = "넷플릭스, 왓챠, 디즈니플러스, 웨이브 신작 추천해줘."

print("=" * 60)
print("[ 예제 2 ] news_qa_v3() - gpt-4o-mini (Sequential Tool Call)")
print("=" * 60)
news_qa_v3(prompt)

print("=" * 60)
print("[ 예제 3 ] news_qa_v3() - gpt-4o (Sequential Tool Call)")
print("=" * 60)
news_qa_v3(prompt, model='gpt-4o')


# =============================================================
# 학습 포인트 요약
# =============================================================
# 1. [parallel_tool_calls=False]
#    - 한 번의 응답에 tool_calls 를 반드시 1개씩만 반환
#    - Tool → 결과 전달 → Tool → 결과 전달 → ... 순차 구조 강제
#    - 검색 결과를 한 번에 몰아받지 않아 컨텍스트 부담 감소
#    → Topic 6 의 할루시네이션 문제를 해결하는 방법
#
# 2. [while tool_calls: 루프]
#    - if (Topic 6) 와 달리 while 을 사용하는 이유:
#      응답이 몇 번의 Tool Call 을 요구할지 사전에 알 수 없기 때문
#    - tool_calls 가 없는 응답이 오면 자동으로 루프 탈출
#    - 루프 탈출 후 tool_call_result.content 가 최종 답변
#
# 3. [msgs 누적 패턴]
#    루프를 거칠 때마다 msgs 에 아래 두 가지가 쌓임:
#    - assistant 메시지 (tool_calls 포함)
#    - tool 메시지     (검색 결과 + tool_call_id)
#    → GPT 는 매 호출마다 전체 히스토리를 보고 다음 행동을 판단
#
# 4. [Topic 6 vs Topic 7 최종 비교]
#    ┌──────────────────┬──────────────────┬──────────────────┐
#    │ 항목             │ Topic 6 Parallel │ Topic 7 Sequential│
#    ├──────────────────┼──────────────────┼──────────────────┤
#    │ tool_calls 반환  │ N개 동시          │ 1개씩            │
#    │ 루프 구조        │ if + for         │ while            │
#    │ 결과 전달 방식   │ 한꺼번에          │ 하나씩 누적       │
#    │ 할루시네이션     │ 발생 가능         │ 상대적으로 낮음   │
#    │ API 호출 횟수    │ 적음             │ 많음             │
#    └──────────────────┴──────────────────┴──────────────────┘