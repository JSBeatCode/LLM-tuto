# =============================================================================
# Topic 2. Prompt Template
# =============================================================================
# 학습 목표:
#   1. PromptTemplate — 단일/다중 변수 템플릿 작성 및 .format()
#   2. ChatPromptTemplate — system/user 역할 구분, .format_messages()
#   3. 두 템플릿의 차이점 및 올바른 사용법 비교
# =============================================================================

# [사전 준비] 패키지 설치
# pip install langchain langchain_openai openai python-dotenv

# [사전 준비] 이 파일과 같은 폴더에 .env 파일 생성 후 아래 내용 입력
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

import os
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


# =============================================================================
# 0. 환경 설정
# =============================================================================

load_dotenv(override=True)  # .env 파일에서 API Key 로드

client = openai.OpenAI()

try:
    client.models.list()
    print("✅ OPENAI_API_KEY가 정상적으로 설정되어 있습니다.\n")
except Exception:
    print("❌ API 키가 유효하지 않습니다! .env 파일을 확인해주세요.")
    exit()

# 기본 LLM 설정
MODEL = 'gpt-4o-mini'
llm = ChatOpenAI(model=MODEL, temperature=0.1, max_tokens=1024)


# =============================================================================
# 1. PromptTemplate — 단일 변수
# =============================================================================
# 하나의 변수({term})를 가진 단순 텍스트 프롬프트 템플릿

print("=" * 60)
print("▶ [1] PromptTemplate — 단일 변수")
print("=" * 60)

explain_template = """당신은 어려운 용어를 초등학생 수준의 레벨로 쉽게 설명하는 챗봇입니다.
{term}에 대해 설명해 주세요."""

print("[템플릿 원문]")
print(explain_template)

# PromptTemplate 객체 생성
explain_prompt = PromptTemplate(template=explain_template)

# .format()으로 변수 채우기 → 완성된 문자열 반환
formatted = explain_prompt.format(term="트랜스포머 네트워크")
print("\n[.format() 결과 — 완성된 프롬프트 문자열]")
print(formatted)

# LLM 호출
print("\n[LLM 응답]")
response = llm.invoke(formatted).content
print(response)


# =============================================================================
# 2. PromptTemplate — 다중 변수
# =============================================================================
# 두 개의 변수({topic}, {language})를 가진 템플릿

print("\n" + "=" * 60)
print("▶ [2] PromptTemplate — 다중 변수")
print("=" * 60)

translate_template = "{topic}에 대해 {language}로 설명하세요."
translate_prompt = PromptTemplate(template=translate_template)

# 두 변수를 동시에 채워서 완성
formatted2 = translate_prompt.format(topic='torschlusspanik', language='한국어')
print("[.format() 결과 — 완성된 프롬프트 문자열]")
print(formatted2)

print("\n[LLM 응답]")
response2 = llm.invoke(formatted2).content
print(response2)


# =============================================================================
# 3. ChatPromptTemplate — system/user 역할 구분
# =============================================================================
# 채팅 메시지 형식: system(역할 지시) + user(실제 질문)
# .format()이 아닌 .format_messages() 를 사용해야 함!

print("\n" + "=" * 60)
print("▶ [3] ChatPromptTemplate — system/user 역할 구분")
print("=" * 60)

chat_prompt = ChatPromptTemplate([
    ("system", "당신은 항상 부정적인 말만 하는 챗봇입니다. 첫 문장은 항상 사용자의 의견을 반박하세요."),
    ("user",   "{A}를 배우면 어떤 유용한 점이 있나요?")
    # system, user(=human), ai(=assistant)
])

# .format_messages()로 변수 채우기 → 메시지 리스트 반환
messages = chat_prompt.format_messages(A='LangChain')
print("[.format_messages() 결과 — 메시지 리스트]")
for msg in messages:
    print(f"  [{msg.type.upper()}] {msg.content}")

print("\n[LLM 응답]")
response3 = llm.invoke(messages).content
print(response3)


# =============================================================================
# 4. ❌ 잘못된 사용법 비교 — ChatPromptTemplate에 .format() 사용 시
# =============================================================================

print("\n" + "=" * 60)
print("▶ [4] ❌ 잘못된 사용법 — ChatPromptTemplate에 .format() 사용 시")
print("=" * 60)

print("ChatPromptTemplate에 .format()을 사용하면 에러가 발생합니다.")
print("→ ChatPromptTemplate은 반드시 .format_messages()를 사용해야 합니다.\n")

try:
    wrong_result = chat_prompt.format(A='LangChain')  # ← 잘못된 사용
    print("[.format() 결과]")
    print(wrong_result)
    print("\n⚠️  결과가 출력됐더라도 이 형식은 LLM에 직접 전달하기 어렵습니다.")
    print("   채팅 메시지 구조가 사라지고 단순 문자열로 변환되어 역할 구분이 무의미해집니다.")
except Exception as e:
    print(f"❌ 에러 발생: {e}")


# =============================================================================
# 5. PromptTemplate vs ChatPromptTemplate 한눈에 비교
# =============================================================================

print("\n" + "=" * 60)
print("▶ [5] PromptTemplate vs ChatPromptTemplate 비교 요약")
print("=" * 60)

summary = """
┌─────────────────────────┬──────────────────────────────┬──────────────────────────────────┐
│         항목            │      PromptTemplate           │     ChatPromptTemplate           │
├─────────────────────────┼──────────────────────────────┼──────────────────────────────────┤
│ 반환 타입               │ 완성된 문자열 (str)           │ 메시지 리스트 (list[Message])    │
│ 변수 채우는 메서드      │ .format(변수=값)              │ .format_messages(변수=값)        │
│ 역할 구분 (system/user) │ ❌ 없음                       │ ✅ 있음                          │
│ 주요 사용 목적          │ 단순 텍스트 프롬프트           │ 채팅 기반 프롬프트               │
│ LLM에 전달 방식         │ llm.invoke(str)               │ llm.invoke(messages)             │
└─────────────────────────┴──────────────────────────────┴──────────────────────────────────┘

💡 핵심 원칙:
   - PromptTemplate   → .format()        → 문자열 → llm.invoke(문자열)
   - ChatPromptTemplate → .format_messages() → 메시지 리스트 → llm.invoke(메시지 리스트)
"""
print(summary)