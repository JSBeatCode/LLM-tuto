# =============================================================================
# Topic 1. LangChain 환경 설정 & LLM 기본 호출
# =============================================================================
# 학습 목표:
#   1. 패키지 설치 및 API Key 설정 (.env 활용)
#   2. ChatOpenAI로 모델 인스턴스 생성 (gpt-4o-mini, o1-mini)
#   3. .invoke()로 LLM 호출 및 AIMessage 응답 처리
#   4. .content로 텍스트 추출
# =============================================================================

# [사전 준비] 아래 패키지를 먼저 설치하세요.
# pip install langchain langchain_openai openai python-dotenv

# [사전 준비] .env 파일을 이 파일과 같은 폴더에 생성하고 아래 내용을 입력하세요.
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

import os
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# =============================================================================
# 0. 환경 설정: .env 파일에서 API Key 로드
# =============================================================================

load_dotenv(override=True)  # .env 파일을 읽어 환경변수로 등록

client = openai.OpenAI()  # API Key는 환경변수에서 자동으로 읽어옴

# API Key 유효성 검증
try:
    client.models.list()
    print("✅ OPENAI_API_KEY가 정상적으로 설정되어 있습니다.\n")
except Exception:
    print("❌ API 키가 유효하지 않습니다! .env 파일을 확인해주세요.")
    exit()


# =============================================================================
# 1. ChatOpenAI로 모델 인스턴스 생성
# =============================================================================

# gpt-4o-mini : 빠르고 가벼운 모델 (일반적인 실습에 적합)
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.1, max_tokens=1024)

print("=" * 60)
print("▶ [모델: gpt-4o-mini] 인스턴스 생성 완료")
print("=" * 60)


# =============================================================================
# 2. .invoke()로 LLM 호출 — AIMessage 응답 객체 확인
# =============================================================================

question = '''전 세계적으로 흥행한 영화에 나오는 유명한 명대사를 하나 알려주세요.
대사가 나온 배경과 의미도 설명해 주세요.'''

print(f"\n[질문]\n{question}\n")

# invoke() 호출 → 반환값은 AIMessage 객체
response = llm.invoke(question)

print("[AIMessage 객체 전체 출력]")
print(response)  # AIMessage(content='...', ...)
print()


# =============================================================================
# 3. .content로 텍스트만 추출
# =============================================================================

print("[.content로 텍스트만 추출]")
print(response.content)
print()


# =============================================================================
# 4. 다른 질문으로 추가 호출 연습
# =============================================================================

print("=" * 60)
print("▶ [추가 호출 연습]")
print("=" * 60)

question2 = '''울림을 주는 영화 명대사를 하나 알려주세요.
대사가 나온 배경과 의미도 설명해 주세요.'''

print(f"\n[질문]\n{question2}\n")

# invoke() 실행 후 .content 바로 추출
answer2 = llm.invoke(question2).content

print("[응답]")
print(answer2)
print()


# =============================================================================
# 5. (심화) o1-mini 모델 사용법
# =============================================================================
# o1 계열 모델은 temperature=1.0 고정, max_completion_tokens 사용에 주의
# API Key로 o1-mini 접근 권한이 없는 경우 에러가 발생할 수 있습니다.

print("=" * 60)
print("▶ [심화] o1-mini 모델 호출")
print("=" * 60)

try:
    o1 = ChatOpenAI(
        model='o1-mini',
        temperature=1.0,
        max_completion_tokens=25000
    )

    question3 = "BERT와 같은 인코더 기반 모델이 이후 어떻게 임베딩 모델로 진화했는지 알려주세요."
    print(f"\n[질문]\n{question3}\n")

    response3 = o1.invoke(question3)
    print("[응답]")
    print(response3.content)

except Exception as e:
    print(f"⚠️  o1-mini 호출 실패 (API 접근 권한 또는 설정 문제): {e}")
    print("   → gpt-4o-mini 로 대체하여 실행합니다.\n")

    fallback_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.1, max_tokens=1024)
    question3 = "BERT와 같은 인코더 기반 모델이 이후 어떻게 임베딩 모델로 진화했는지 알려주세요."
    print(f"[질문]\n{question3}\n")
    print("[응답]")
    print(fallback_llm.invoke(question3).content)


# =============================================================================
# [참고] Deprecated된 호출 방식 (사용 금지)
# =============================================================================
# llm.predict(question)   ← Deprecated
# llm.run(question)       ← Deprecated
# llm(question)           ← Deprecated
# → 반드시 llm.invoke(question) 을 사용하세요.