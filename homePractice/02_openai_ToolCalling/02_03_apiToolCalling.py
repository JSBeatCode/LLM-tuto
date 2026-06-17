# =============================================================
# Topic 5 : Tool Calling 기초
# - Tool의 개념과 동작 원리
# - Pydantic BaseModel + Field 로 Tool 스키마 정의
# - openai.pydantic_function_tool() 로 OpenAI 포맷 변환
# - tool_choice 옵션 (auto / none / required) 차이
# - tool_calls 응답 구조 파악 (id / name / arguments)
# - available_functions 딕셔너리로 tool-함수 매핑 및 실행
# - 단일 Tool Call 을 포함한 완성된 news_qa() 함수 구현
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
#    (Tool Calling 에서 실제로 실행될 외부 함수)
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
# 2. Tool 스키마 정의
#    - Pydantic BaseModel 로 Tool의 파라미터 구조를 정의
#    - class 이름       → Tool 이름 (Web_Search)
#    - class docstring  → Tool 설명 (LLM이 어떤 역할인지 판단에 사용)
#    - Field description → 파라미터 설명 (LLM이 값을 어떻게 채울지 판단에 사용)
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

# Pydantic 모델을 OpenAI가 인식하는 tools 포맷으로 변환
tools = [openai.pydantic_function_tool(Web_Search)]

print("[ Tool 스키마 확인 ]")
print(tools)
print()


# =============================================================
# 3. Tool 이 포함된 뉴스 봇 함수 - news_bot_v2()
#    - tools 파라미터로 사용 가능한 Tool 목록 전달
#    - tool_choice 로 LLM의 Tool 사용 방식 제어
#    - stream 여부에 따라 반환값 형태가 달라짐
# =============================================================

def news_bot_v2(messages, stream=False, model=MODEL):

    response = client.chat.completions.create(
        model=model,
        messages=messages,

        # 사용할 툴 목록 전달
        # LLM이 description, name, parameter 값을 보고 스스로 판단
        tools=tools,
        tool_choice='auto',
        # 'auto'     : LLM이 자율적으로 Tool 사용 여부 판단  ← 기본값
        # 'none'     : Tool 사용하지 않음 (일반 대화만)
        # 'required' : 무조건 Tool 사용 강제

        temperature=0.1,
        max_tokens=1024,

        stream=stream
        # stream=True  → response 객체(이터레이터) 반환
        # stream=False → response.choices[0].message 반환
    )

    # 스트리밍 여부에 따라 반환값 다르게 처리
    if stream:
        return response

    return response.choices[0].message


# =============================================================
# 4. tool_choice='auto' 동작 확인
#    - 일상 대화  → tool_calls 없음 (content 로 응답)
#    - 뉴스 질문  → tool_calls 있음 (LLM이 검색 필요 판단)
# =============================================================

print("=" * 60)
print("[ 예제 1 ] tool_choice='auto' - Tool 불필요한 경우 (일상 대화)")
print("=" * 60)

result_no_tool = news_bot_v2([
    {
        "role": "user",
        "content": "안녕하세요! 오늘 날씨가 좋네요."
    }
])

print("tool_calls 존재 여부:", result_no_tool.tool_calls)   # None
print("content 응답:", result_no_tool.content)
print()


print("=" * 60)
print("[ 예제 2 ] tool_choice='auto' - Tool 필요한 경우 (뉴스 검색 질문)")
print("=" * 60)

tool_call_result = news_bot_v2([
    {
        "role": "user",
        "content": "요즘 새로 개봉한 영화는 무엇이 있나요?"
    }
])

print("tool_calls 존재 여부:", tool_call_result.tool_calls)  # tool_calls 객체
print()


# =============================================================
# 5. tool_calls 응답 구조 파악
#    tool_calls[0] 의 구성 요소:
#    - id        : 이 tool call 의 고유 ID (tool 결과 전달 시 매핑에 사용)
#    - name      : 실행할 Tool 이름 (예: "Web_Search")
#    - arguments : Tool 에 전달할 인수 (JSON 문자열 형태)
# =============================================================

print("=" * 60)
print("[ tool_calls 응답 구조 분석 ]")
print("=" * 60)

tool_id        = tool_call_result.tool_calls[0].id
tool_name      = tool_call_result.tool_calls[0].function.name
tool_arguments = tool_call_result.tool_calls[0].function.arguments

print(f"tool id        : {tool_id}")
print(f"tool name      : {tool_name}")
print(f"tool arguments : {tool_arguments}")   # JSON 문자열
print()


# =============================================================
# 6. tool_calls 결과를 이용한 함수 실행 3단계 패턴
#    LLM은 함수를 직접 실행하지 않음 → 우리가 실행 후 결과 전달
#
#    단계 1) json.loads(arguments)  : JSON 문자열 → dict 변환
#    단계 2) available_functions    : tool 이름(문자열) → 실제 함수 매핑
#    단계 3) available_functions[name](**dict) : dict 언패킹으로 함수 실행
# =============================================================

print("=" * 60)
print("[ tool_calls → 함수 실행 3단계 패턴 ]")
print("=" * 60)

# 단계 1) JSON 문자열 → dict
example_json = '{"query":"영화"}'
example_dict = json.loads(example_json)
print(f"단계 1) json.loads 결과: {example_dict}, 타입: {type(example_dict)}")

# 단계 2) tool 이름 → 실제 함수 매핑
available_functions = {'Web_Search': get_news}
print(f"단계 2) available_functions['Web_Search']: {available_functions['Web_Search']}")

# 단계 3) dict 언패킹으로 함수 실행
# available_functions['Web_Search'](**{'query':'영화'})
# == get_news(query='영화') 와 동일
print("단계 3) 3단계 조합 실행: available_functions[name](**json.loads(arguments))")
search_result = available_functions[tool_name](**json.loads(tool_arguments))
print(search_result[:300], "...\n")  # 너무 길므로 앞부분만 출력


# =============================================================
# 7. Tool 결과를 메시지에 포함해 LLM 에 전달
#    tool 역할 메시지: role="tool", tool_call_id 로 id 매핑 필수
#
#    메시지 순서:
#    user(질문) → assistant(tool_calls) → tool(검색결과) → assistant(최종 답변)
# =============================================================

print("=" * 60)
print("[ 예제 3 ] tool 결과를 메시지 히스토리에 포함해 최종 답변 받기")
print("=" * 60)

result = news_bot_v2([
    {
        "role": "user",
        "content": "요즘 새로 개봉한 영화는 무엇이 있나요?"
    },
    tool_call_result,           # assistant 의 tool_calls 메시지
    {
        "role": "tool",
        "content": search_result,
        "tool_call_id": tool_call_result.tool_calls[0].id   # id 매핑 필수
    }
])

print(result.content)
print()


# =============================================================
# 8. 완성된 단일 Tool Call QA 함수 - news_qa()
#    전체 흐름을 하나의 함수로 통합:
#    프롬프트 입력 → tool 필요 여부 판단 → 검색 실행 → 스트리밍 최종 답변
# =============================================================

def news_qa(prompt):
    """
    사용자 프롬프트를 받아 Tool Calling 여부를 자동 판단하고 답변하는 함수
    - 뉴스 관련 질문 : Web_Search Tool 호출 → get_news() 실행 → 스트리밍 답변
    - 일반 대화      : Tool 호출 없이 바로 content 응답
    """

    available_functions = {'Web_Search': get_news}

    print('Prompt:', prompt)

    # 1) 프롬프트로 tool 필요 여부 판단
    tool_call_result = news_bot_v2([
        {
            "role": "user",
            "content": prompt
        }
    ])

    print('---')
    print('News_Bot: Call ', end='')

    if tool_call_result.tool_calls:     # tool_calls 가 존재하면 Tool 실행

        name      = tool_call_result.tool_calls[0].function.name
        arguments = tool_call_result.tool_calls[0].function.arguments

        print(name, arguments)

        # 2) tool 이름과 arguments 로 실제 함수 실행
        search_result = available_functions[name](**json.loads(arguments))

        print('---')
        print('News_Bot:', end='')

        # 3) 검색 결과를 포함해 최종 답변 요청 (스트리밍)
        response = news_bot_v2(
            [
                {"role": "user",  "content": prompt},
                tool_call_result,
                {"role": "tool",  "content": search_result,
                 "tool_call_id": tool_call_result.tool_calls[0].id}
            ],
            stream=True     # 최종 출력은 스트리밍으로 UX 향상
        )

        for chunk in response:
            print(chunk.choices[0].delta.content, end='', flush=True)

    else:                               # tool_calls 없으면 일반 대화 응답
        print('Nothing')
        print(tool_call_result.content)

    print('\n')


# =============================================================
# 9. news_qa() 실행 예제
# =============================================================

print("=" * 60)
print("[ 예제 4 ] news_qa() - 뉴스 관련 질문 (Tool 사용)")
print("=" * 60)
news_qa("넷플릭스 신작 추천해줘.")

print("=" * 60)
print("[ 예제 5 ] news_qa() - 일반 대화 (Tool 미사용)")
print("=" * 60)
news_qa("회사 가기 싫어.")


# =============================================================
# 학습 포인트 요약
# =============================================================
# 1. [Tool 스키마 정의]
#    - Pydantic BaseModel 상속 → class 이름이 Tool 이름이 됨
#    - docstring → LLM이 Tool의 역할을 판단하는 설명
#    - Field(description=...) → LLM이 파라미터 값을 어떻게 채울지 안내
#    - openai.pydantic_function_tool() 로 OpenAI 포맷 자동 변환
#
# 2. [tool_choice 옵션]
#    - 'auto'     : LLM이 필요 여부를 스스로 판단 (일반적으로 이 값 사용)
#    - 'none'     : Tool을 사용하지 않고 일반 대화만 수행
#    - 'required' : 반드시 Tool을 호출하도록 강제
#
# 3. [tool_calls 응답 구조]
#    - tool_calls 가 None  → 일반 content 응답 (Tool 불필요)
#    - tool_calls 가 존재 → LLM이 실행할 함수 이름 + 인수를 반환
#    - id / name / arguments 세 가지 정보가 핵심
#
# 4. [함수 실행 3단계 패턴]
#    available_functions[name](**json.loads(arguments))
#    → Tool 이름(문자열)으로 함수 조회 + JSON 인수를 dict로 변환 후 언패킹 실행
#
# 5. [tool 역할 메시지 전달]
#    - role="tool" 메시지로 검색 결과를 히스토리에 추가
#    - tool_call_id 를 반드시 매핑해야 LLM이 어떤 요청의 결과인지 인식