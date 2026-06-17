# =============================================================
# Topic 4 : 뉴스 요약 봇 기초 + 스트리밍
# - 검색 결과를 system 메시지로 컨텍스트에 주입하는 패턴
# - multi-turn 메시지 구조 (system → user → user)
# - 다양한 출력 스타일 지정 (요약 / 대화체 / 뉴스 리포팅)
# - stream=True 스트리밍 응답 처리 및 UX 활용법
# =============================================================

import openai
import requests
import tiktoken
import os
from dotenv import load_dotenv

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
#    (Topic 3의 결과물 - Topic 4의 입력으로 사용)
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
# 2. 뉴스 요약 봇 기초 - news_bot()
#    검색 결과를 컨텍스트로 주입하여 LLM이 요약하도록 구성
# =============================================================

def news_bot(messages):
    """일반 응답 봇 (스트리밍 없음, content 문자열 반환)"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=2048,
    )

    # 간편한 출력을 위해 content만 추출
    return response.choices[0].message.content


# =============================================================
# 3. 실행 예제 1 : system 메시지로 뉴스 컨텍스트 주입 후 요약
#    패턴: system(검색결과 + 지시) + user(뉴스 원문)
# =============================================================

print("=" * 60)
print("[ 예제 1 ] system 메시지 컨텍스트 주입 - 기본 요약")
print("=" * 60)

query = '삼성 라이온즈'
news_result = get_news(query)

# 토큰 수 체크 (입력 컨텍스트 크기 파악)
tokenizer = tiktoken.encoding_for_model('gpt-4o-mini')
print(f"검색어: {query}")
print(f"문자 수: {len(news_result)}")
print(f"토큰 수: {len(tokenizer.encode(news_result))}\n")

response = news_bot([
    {
        # 검색 결과를 system 메시지로 주입하는 핵심 패턴
        "role": "system",
        "content": f"""
뉴스 검색 결과가 주어집니다.
{query}에 대한 뉴스를 요약하세요.
---
"""
    },
    {
        "role": "user",
        "content": news_result
    }
])

print(response)


# =============================================================
# 4. 실행 예제 2 : multi-turn 메시지로 출력 스타일 변경
#    패턴: user(뉴스 원문) → user(스타일 지시)
#    → 같은 데이터로 전혀 다른 형식의 출력 생성 가능
# =============================================================

print("\n" + "=" * 60)
print("[ 예제 2 ] multi-turn 메시지 - 대화체 스타일 출력")
print("=" * 60)

query = '삼성 라이온즈'
news_result = get_news(query)

response = news_bot([
    {
        "role": "user",
        "content": news_result      # 1번째 user: 뉴스 원문 전달
    },
    {
        "role": "user",             # 2번째 user: 출력 형식 지시
        "content": f"""
---
위 전체 내용을 종합하여 {query}의 동향과 미래에 대해 논쟁하는 두 사람의 대화 내용을 만들어줘.
한 명은 전문적인 말투, 한 명은 어리숙한 말투를 사용하고, 형식은 아래와 같아.
---
A:(A의 대화)
B:(B의 대화)
        """
    }
])

print(response)


# =============================================================
# 5. 스트리밍 봇 - news_bot_stream()
#    stream=True 로 토큰 단위 실시간 출력 (ChatGPT UX와 동일)
# =============================================================

def news_bot_stream(messages):
    """스트리밍 응답 봇 (response 객체 전체를 반환, 호출부에서 chunk 처리)"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=2048,
        stream=True     # 스트리밍 활성화
    )

    # content가 아닌 response 전체를 return (호출부에서 for chunk in response 처리)
    return response


# =============================================================
# 6. 실행 예제 3 : 스트리밍으로 뉴스 리포팅 출력
#    LLM 어플리케이션의 최종 출력 단계에서 사용하면 UX 향상
# =============================================================

print("\n" + "=" * 60)
print("[ 예제 3 ] stream=True - 실시간 스트리밍 뉴스 리포팅")
print("=" * 60)

query = 'LLM'
news_result = get_news(query)

stream_response = news_bot_stream([
    {
        "role": "user",
        "content": news_result
    },
    {
        "role": "user",
        "content": """
위 전체 내용을 종합하여 뉴스 리포팅을 해줘.
---
'제목':
'본문':
        """
    }
])

# 스트리밍 chunk 순회 처리
# chunk.choices[0].delta.content : 이번 chunk에 담긴 텍스트 조각
for chunk in stream_response:
    print(chunk.choices[0].delta.content, end='', flush=True)

print()  # 마지막 줄바꿈


# =============================================================
# 학습 포인트 요약
# =============================================================
# 1. [컨텍스트 주입 패턴]
#    - 외부 데이터(뉴스)를 system 또는 user 메시지에 담아 LLM에 전달
#    - system 메시지: 역할 + 검색결과 + 요약 지시를 한 번에 설정
#    - 뉴스 원문을 먼저 user로 전달 후 두 번째 user로 스타일 지시 가능
#
# 2. [tiktoken으로 토큰 수 확인]
#    - 입력 데이터가 max_tokens 한도를 초과하는지 사전 체크에 유용
#    - tiktoken.encoding_for_model('gpt-4o-mini') 로 모델별 토크나이저 사용
#
# 3. [stream=True 스트리밍]
#    - stream=True 시 response가 이터레이터로 반환됨
#    - for chunk in response 로 순회하며 delta.content 출력
#    - end='' + flush=True 조합으로 실시간 타이핑 효과 구현
#    - 최종 출력 단계에서만 사용하는 것이 일반적 (중간 처리에는 부적합)
#
# 4. [news_bot vs news_bot_stream 차이]
#    - news_bot      : content 문자열 직접 반환 → 후처리·재사용 편리
#    - news_bot_stream : response 객체 반환 → 호출부에서 chunk 직접 처리