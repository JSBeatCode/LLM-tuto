# =============================================================
# Topic 6 : Parallel Tool Call
# - 하나의 질문에 여러 Tool Call이 동시에 필요한 상황 이해
# - GPT가 tool_calls 를 복수로 반환하는 구조 파악
# - for 루프로 다수 tool 결과를 일괄 수집 → tool_messages 리스트 구성
# - 복수 tool 결과를 한꺼번에 전달하는 메시지 구조
# - 한계점 : 컨텍스트 과부하로 인한 할루시네이션 가능성
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
#    (Parallel Tool Call 에서 여러 번 실행될 외부 함수)
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
# 2. Tool 스키마 정의 (Topic 5 와 동일)
# =============================================================

class Web_Search(BaseModel):
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

tools = [openai.pydantic_function_tool(Web_Search)]


# =============================================================
# 3. Tool 이 포함된 뉴스 봇 함수 - news_bot_v2()
#    (Topic 5 와 동일 - Parallel Tool Call 의 기반 함수)
#    parallel_tool_calls 옵션을 별도 지정하지 않으면
#    기본값 True → GPT 가 필요 시 여러 tool_calls 를 한 번에 반환
# =============================================================

def news_bot_v2(messages, stream=False, model=MODEL):

    response = client.chat.completions.create(
        model=model,
        messages=messages,

        tools=tools,
        tool_choice='auto',

        temperature=0.1,
        max_tokens=1024,

        stream=stream
        # parallel_tool_calls 를 명시하지 않으면 기본값 True
        # → GPT 가 여러 검색어를 판단하면 tool_calls 를 복수로 반환
    )

    if stream:
        return response

    return response.choices[0].message


# =============================================================
# 4. Parallel Tool Call 동작 확인
#    - 여러 OTT 플랫폼을 한 번에 묻는 질문
#    - GPT 가 플랫폼별 검색어를 판단해 tool_calls 를 복수로 반환
# =============================================================

print("=" * 60)
print("[ 예제 1 ] Parallel Tool Call - tool_calls 복수 반환 확인")
print("=" * 60)

prompt = "넷플릭스 왓챠 디즈니플러스 웨이브 신작 추천해줘."

tool_call_result = news_bot_v2([
    {
        "role": "user",
        "content": prompt
    }
])

print(f"tool_calls 개수: {len(tool_call_result.tool_calls)}")
print("반환된 tool_calls 목록:")
for i, tc in enumerate(tool_call_result.tool_calls):
    print(f"  [{i}] id={tc.id} | name={tc.function.name} | arguments={tc.function.arguments}")
print()


# =============================================================
# 5. Parallel Tool Call 핵심 처리 패턴
#
#    Topic 5 (단일 Tool Call) 와의 차이점:
#    ┌─────────────────────────────────────────────────────┐
#    │ Topic 5 (단일)   │ tool_calls[0] 만 처리            │
#    │ Topic 6 (복수)   │ for tool_call in tool_calls: 순회 │
#    └─────────────────────────────────────────────────────┘
#
#    처리 흐름:
#    1) tool_calls 전체를 for 루프로 순회
#    2) 각 tool_call 마다 get_news() 실행
#    3) 결과를 tool_messages 리스트에 누적
#    4) [user 메시지] + [tool_call_result] + tool_messages 를
#       한꺼번에 전달해 최종 답변 요청
# =============================================================

def news_qa_v2(prompt, model=MODEL):
    """
    Parallel Tool Call 처리 함수
    - GPT 가 반환한 복수의 tool_calls 를 for 루프로 일괄 처리
    - 모든 검색 결과를 tool_messages 리스트에 모아 한꺼번에 전달
    """

    print('Prompt:', prompt)

    available_functions = {'Web_Search': get_news}

    # 1) 프롬프트 전달 → GPT 가 복수 tool_calls 반환
    tool_call_result = news_bot_v2([
        {
            "role": "user",
            "content": prompt
        },
    ])

    print('---')
    print('News_Bot: Call ', end='')

    if tool_call_result.tool_calls:

        # 2) tool_messages 리스트 : 복수 tool 결과를 누적할 공간
        tool_messages = []

        # 3) tool_calls 전체를 for 루프로 순회
        #    단일 Tool Call(Topic 5) 은 tool_calls[0] 하나만 처리했지만
        #    Parallel 은 tool_calls 전체를 순회
        for tool_call in tool_call_result.tool_calls:

            name      = tool_call.function.name
            arguments = tool_call.function.arguments
            print(name, arguments)

            # 4) 각 tool_call 에 대해 실제 함수 실행
            search_result = available_functions[name](**json.loads(arguments))

            print('---')
            print('News_Bot:', end='')

            # 5) 각 결과를 tool_call_id 와 함께 리스트에 추가
            #    tool_call_id 로 어떤 tool_call 의 결과인지 매핑
            tool_messages.append(
                {
                    "role": "tool",
                    "content": search_result,
                    "tool_call_id": tool_call.id     # 각 tool_call 의 고유 id
                }
            )

        # 6) 모든 tool 결과를 한꺼번에 포함해 최종 답변 요청
        #    메시지 구조: [user] + [assistant(tool_calls)] + [tool × N개]
        response = news_bot_v2(
            [
                {"role": "user", "content": prompt}, # "넷플릭스 왓챠 디즈니플러스 웨이브 신작 추천해줘."
                tool_call_result                    # assistant 의 tool_calls 메시지 # tool_call_result = {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "Web_Search", "arguments": "{\"query\": \"넷플릭스\"}"}}, {"id": "call_2", "function": {"name": "Web_Search", "arguments": "{\"query\": \"왓챠\"}"}}, {"id": "call_3", "function": {"name": "Web_Search", "arguments": "{\"query\": \"디즈니플러스\"}"}}, {"id": "call_4", "function": {"name": "Web_Search", "arguments": "{\"query\": \"웨이브\"}"}}]}
            ] + tool_messages,                      # tool 결과 N개를 리스트로 연결 # tool_messages = [{"role": "tool", "content": "제목: 넷플릭스 신작 공개 URL: https://... 내용: 이번 달 넷플릭스 신작은...", "tool_call_id": "call_1"}, {"role": "tool", "content": "제목: 넷플릭스 인기 영화 URL: https://... 내용: ...", "tool_call_id": "call_2"}]
            stream=True,
            model=model
        )
        # 이 호출의 원리:
        # 1. 사용자가 이런 질문을 했고
        # 2. 내가 아까 이런 검색들을 하라고 했고
        # 3. 실제 결과가 이렇게 들어왔네
        # → 이제 이걸 종합해서 답변 만들자

        for chunk in response:
            print(chunk.choices[0].delta.content, end='', flush=True)

    else:
        print('Nothing')
        print(tool_call_result.content)

    print('\n')


# =============================================================
# 6. 실행 예제
# =============================================================

print("=" * 60)
print("[ 예제 2 ] news_qa_v2() - gpt-4o-mini (Parallel Tool Call)")
print("=" * 60)
news_qa_v2(prompt)

print("=" * 60)
print("[ 예제 3 ] news_qa_v2() - gpt-4o (Parallel Tool Call)")
print("할루시네이션 비교: 한꺼번에 긴 컨텍스트를 받을 때 모델별 차이 확인")
print("=" * 60)
news_qa_v2(prompt, model='gpt-4o')


# =============================================================
# 학습 포인트 요약
# =============================================================
# 1. [Parallel Tool Call 이란?]
#    - 하나의 사용자 메시지에 대해 GPT 가 여러 tool_calls 를 동시에 반환
#    - parallel_tool_calls 를 명시하지 않으면 기본값 True (활성화 상태)
#    - 예) "넷플릭스, 왓챠, 디즈니플러스, 웨이브" → 4개 tool_calls 동시 반환
#
# 2. [단일 vs Parallel Tool Call 핵심 차이]
#    - 단일 (Topic 5) : tool_calls[0] 하나만 처리
#    - Parallel       : for tool_call in tool_calls 로 전체 순회
#    - tool_messages 리스트에 결과를 누적 후 한꺼번에 전달
#
# 3. [메시지 구조]
#    [user(질문)] + [assistant(tool_calls 복수)] + [tool×N개]
#    - tool 메시지마다 tool_call_id 로 어떤 요청의 결과인지 반드시 매핑
#
# 4. [한계점 - 할루시네이션]
#    - 검색 결과 N개를 한꺼번에 전달하면 컨텍스트가 매우 길어짐
#    - 입력이 너무 길면 LLM 이 내용을 혼동하거나 잘못된 정보를 생성할 수 있음
#    - 이 문제를 해결하는 것이 Topic 7 의 Sequential Tool Call
#      (parallel_tool_calls=False + while 루프)