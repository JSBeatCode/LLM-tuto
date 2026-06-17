# =============================================================================
# Topic 4. Few-Shot Prompting
# =============================================================================
# 학습 목표:
#   1. FewShotPromptTemplate으로 예시 기반 프롬프트 구성
#   2. prefix + examples + example_prompt + suffix 구조 이해
#   3. Step-by-step 추론 유도 및 복잡한 질문 처리
# =============================================================================

# [사전 준비] 패키지 설치
# pip install langchain langchain_openai openai python-dotenv

# [사전 준비] 이 파일과 같은 폴더에 .env 파일 생성 후 아래 내용 입력
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate


# =============================================================================
# 0. 환경 설정
# =============================================================================

load_dotenv(override=True)

client = openai.OpenAI()

try:
    client.models.list()
    print("✅ OPENAI_API_KEY가 정상적으로 설정되어 있습니다.\n")
except Exception:
    print("❌ API 키가 유효하지 않습니다! .env 파일을 확인해주세요.")
    exit()

MODEL = 'gpt-4o-mini'   # gpt-4o 접근 권한이 있다면 'gpt-4o' 로 변경 가능
llm = ChatOpenAI(model=MODEL, temperature=0, max_tokens=1024)


# =============================================================================
# 1. Few-Shot examples 정의
# =============================================================================
# 각 example은 {"question": ..., "answer": ...} 딕셔너리로 구성
# answer 안에 Step-by-step 추론 과정(Follow up → Intermediate answer)을 담아
# LLM이 같은 방식으로 추론하도록 유도합니다.

print("=" * 60)
print("▶ [1] Few-Shot examples 정의")
print("=" * 60)

examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali
""",
    },
    {
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate Answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate Answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate Answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate Answer: New Zealand.
So the final answer is: No
""",
    },
]

print(f"총 {len(examples)}개의 예시가 정의되었습니다.")
for i, ex in enumerate(examples):
    print(f"\n[Example {i+1}]")
    print(f"  Question: {ex['question']}")


# =============================================================================
# 2. example_prompt — 예시 하나를 어떤 형식으로 표현할지 정의
# =============================================================================
# examples 리스트의 각 딕셔너리를 받아 문자열로 변환하는 역할
# {question}과 {answer} 두 변수를 사용

print("\n" + "=" * 60)
print("▶ [2] example_prompt — 예시 단건 포맷 템플릿")
print("=" * 60)

example_prompt = PromptTemplate(template="Question: {question}\n{answer}")

print("[example_prompt.format() 결과 — 예시 1개가 어떻게 표현되는지 확인]")
print(example_prompt.format(**examples[0]))


# =============================================================================
# 3. FewShotPromptTemplate 조립
# =============================================================================
# ┌─────────────────────────────────────────────────────┐
# │ prefix       : LLM에게 주는 전체 지시사항            │
# │ examples     : 예시 딕셔너리 리스트                  │
# │ example_prompt: 예시 하나를 어떤 형식으로 표현할지   │
# │ suffix       : 실제 질문이 들어가는 자리 ({input})   │
# └─────────────────────────────────────────────────────┘

print("\n" + "=" * 60)
print("▶ [3] FewShotPromptTemplate 조립")
print("=" * 60)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="질문-답변 형식의 예시가 주어집니다. 같은 방식으로 답변하세요.",
    suffix="Question: {input}",
    # prefix, suffix는 선택사항(Optional)이지만 품질 향상에 중요
)

# .format()으로 완성된 프롬프트 문자열 확인
test_question = "What is the age of the director of a movie which got a best international film in Oscar in 2010?"
formatted_prompt = few_shot_prompt.format(input=test_question)

print("[완성된 전체 프롬프트 구조 확인]")
print("-" * 60)
print(formatted_prompt)
print("-" * 60)


# =============================================================================
# 4. 구조 설명 — prefix / examples / suffix 역할 정리
# =============================================================================

print("\n" + "=" * 60)
print("▶ [4] FewShotPromptTemplate 구조 한눈에 보기")
print("=" * 60)

structure = """
┌─────────────────────────────────────────────────────────────┐
│  FewShotPromptTemplate 구조                                 │
│                                                             │
│  [prefix]                                                   │
│   └─ "질문-답변 형식의 예시가 주어집니다. 같은 방식으로…"    │
│       → LLM에게 전체 맥락과 규칙을 알려줌                   │
│                                                             │
│  [examples] × N개  (example_prompt 형식으로 삽입됨)         │
│   ├─ Example 1: Muhammad Ali vs Alan Turing                 │
│   └─ Example 2: Jaws vs Casino Royale 감독                  │
│       → "이런 식으로 추론해!" 라는 패턴을 보여줌            │
│                                                             │
│  [suffix]                                                   │
│   └─ "Question: {input}"                                    │
│       → 실제로 풀어야 할 질문이 여기 들어감                  │
└─────────────────────────────────────────────────────────────┘
"""
print(structure)


# =============================================================================
# 5. LLM 호출 — Step-by-step 추론 유도
# =============================================================================

print("=" * 60)
print("▶ [5] LLM 호출 — Step-by-step 추론 유도")
print("=" * 60)

question1 = "What is the age of the director of a movie which got a best international film in Oscar in 2010?"
X1 = few_shot_prompt.format(input=question1)

print(f"[질문]\n{question1}\n")
print("[LLM 응답 — Few-Shot으로 유도된 Step-by-step 추론]")
print(llm.invoke(X1).content)


# =============================================================================
# 6. 날짜 정보 추가 질문 — 현재 나이 계산
# =============================================================================

print("\n" + "=" * 60)
print("▶ [6] 날짜 정보를 추가한 질문 — 현재 나이 계산")
print("=" * 60)

# suffix의 {input}에 날짜 정보를 함께 넘겨 더 정확한 답변 유도
question2 = "This is 2024 Dec. What is the age of the director of a movie which got a best international film in Oscar in 2010?\n"
X2 = few_shot_prompt.format(input=question2)

print(f"[질문]\n{question2}")
print("[LLM 응답]")
print(llm.invoke(X2).content)


# =============================================================================
# 7. 한국어 질문 — Few-Shot 패턴의 언어 범용성 확인
# =============================================================================

print("\n" + "=" * 60)
print("▶ [7] 한국어 질문으로도 동일한 추론 패턴 적용")
print("=" * 60)

question3 = "스티븐 스필버그의 영화 중 가장 많은 상을 받은 영화의 주연 배우는?"
X3 = few_shot_prompt.format(input=question3)

print(f"[질문]\n{question3}\n")
print("[LLM 응답]")
print(llm.invoke(X3).content)


# =============================================================================
# 8. Zero-Shot vs Few-Shot 비교 — 같은 질문에 예시 없이 호출
# =============================================================================

print("\n" + "=" * 60)
print("▶ [8] Zero-Shot vs Few-Shot 비교")
print("=" * 60)

# Zero-Shot: 예시 없이 질문만 바로 던지기
zero_shot_q = "Who lived longer, Cleopatra or Julius Caesar?"

print(f"[공통 질문]\n{zero_shot_q}\n")

print("[Zero-Shot 응답 — 예시 없이 바로 질문]")
zero_shot_response = llm.invoke(zero_shot_q).content
print(zero_shot_response)

print("\n[Few-Shot 응답 — 예시 2개 제공 후 같은 질문]")
few_shot_response = llm.invoke(few_shot_prompt.format(input=zero_shot_q)).content
print(few_shot_response)

print("""
💡 핵심 인사이트:
   - Zero-Shot : LLM이 자유롭게 답변 → 형식이 제각각
   - Few-Shot  : 예시를 보고 패턴을 학습 → Follow up / Intermediate answer
                 형식을 그대로 따라 Step-by-step으로 추론
   → 복잡한 추론이 필요한 질문일수록 Few-Shot의 효과가 커집니다.
""")