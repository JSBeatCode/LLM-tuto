"""
주제 2 — 텍스트 생성 & 프롬프트 엔지니어링
==========================================
학습 내용:
  1. system / user / assistant 역할(role) 이해
  2. 기본 Chat Completions 호출 및 응답 파싱
  3. 파라미터 실습 : temperature, max_tokens, n
  4. seed 파라미터 : 재현 가능한 출력 만들기
  5. 멀티턴 대화 : 대화 이력을 누적하여 문맥 유지
"""

import os
from dotenv import load_dotenv
import openai


# ── 환경변수 로드 ─────────────────────────────────────────
load_dotenv(override=True)  # 같은 폴더의 .env 파일을 자동으로 읽어옵니다

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL = "gpt-4o-mini"


# ══════════════════════════════════════════════════════════
def step1_roles() -> None:
    """
    [STEP 1] system / user / assistant 역할 이해
    --------------------------------------------
    - system    : AI의 행동 방식·페르소나를 지정
    - user      : 사용자의 입력 메시지
    - assistant : AI의 응답 (대화 이력 구성 시 사용)

    예시: system 프롬프트로 '속담·격언을 인용하는 AI' 설정
    """
    print("=" * 55)
    print("[STEP 1] system / user / assistant 역할 이해")
    print("=" * 55)

    system_prompt = "당신은 모든 대화에 속담이나 격언을 인용합니다."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": "GPT가 세상을 지배할까?"},
    ]

    print(f"  system  : {system_prompt}")
    print(f"  user    : {messages[1]['content']}")
    print()

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    answer = response.choices[0].message.content
    print(f"  assistant : {answer}")
    print()


# ══════════════════════════════════════════════════════════
def step2_temperature() -> None:
    """
    [STEP 2] temperature 파라미터
    ------------------------------
    temperature (0 ~ 2):
      - 0에 가까울수록 → 일관되고 정해진 답변
      - 2에 가까울수록 → 창의적이고 다양한 답변

    같은 질문을 temperature 0.0 / 1.0 / 1.8 로 각각 호출하여 차이를 비교합니다.
    """
    print("=" * 55)
    print("[STEP 2] temperature 파라미터 비교")
    print("=" * 55)

    messages = [
        {"role": "system", "content": "당신은 시인입니다. 짧은 두 줄 시로 답하세요."},
        {"role": "user",   "content": "봄비에 대해 표현해 주세요."},
    ]

    for temp in [0.0, 1.0, 1.8]:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temp,
            max_tokens=100,
        )
        print(f"  [temperature = {temp}]")
        print(f"  {response.choices[0].message.content.strip()}")
        print()


# ══════════════════════════════════════════════════════════
def step3_max_tokens_and_n() -> None:
    """
    [STEP 3] max_tokens · n 파라미터
    ---------------------------------
    - max_tokens (= max_completion_tokens) : 출력 토큰 수 상한
    - n : 한 번의 호출로 여러 개의 응답 생성

    예시: 톨킨 스타일 자기소개를 n=3 으로 3개 동시 생성
    """
    print("=" * 55)
    print("[STEP 3] max_tokens · n 파라미터")
    print("=" * 55)

    messages = [
        {"role": "system", "content": "J.R.R. 톨킨의 반지의 제왕 스타일로 답변하세요."},
        {"role": "user",   "content": "당신의 자기소개를 해 주세요."},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.9,
        max_tokens=150,   # 응답 하나당 최대 150 토큰
        n=3,              # 응답 3개 동시 생성
    )

    print(f"  생성된 응답 수 : {len(response.choices)}개\n")
    for i, choice in enumerate(response.choices):
        print(f"  [응답 #{i}]")
        print(f"  {choice.message.content.strip()}")
        print()


# ══════════════════════════════════════════════════════════
def step4_seed() -> None:
    """
    [STEP 4] seed 파라미터 — 재현 가능한 출력
    ------------------------------------------
    seed 를 고정하면 같은 입력에 대해 최대한 동일한 출력을 반환합니다.
    (완전히 동일하지 않을 수 있지만, 편차가 크게 줄어듭니다.)

    같은 seed 값으로 두 번 호출하여 결과를 비교합니다.
    """
    print("=" * 55)
    print("[STEP 4] seed 파라미터 — 재현성 확인")
    print("=" * 55)

    messages = [
        {"role": "system", "content": "당신은 건강한 식단과 식이의 전문가입니다."},
        {"role": "user",   "content": "건강한 아침 식사의 조합 예시를 3개만 추천해 주세요."},
    ]

    SEED = 2943

    for trial in range(1, 3):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            max_tokens=300,
            seed=SEED,
        )
        print(f"  [시도 #{trial}  |  seed={SEED}]")
        print(f"  {response.choices[0].message.content.strip()}")
        print()

    print("  → 두 응답이 거의 동일하면 seed 가 정상 작동 중입니다.")
    print()


# ══════════════════════════════════════════════════════════
def step5_multiturn() -> None:
    """
    [STEP 5] 멀티턴 대화 — 대화 이력을 누적하여 문맥 유지
    -------------------------------------------------------
    OpenAI API 는 stateless(무상태) 이므로,
    이전 대화를 기억시키려면 messages 배열에 이력을 직접 누적해야 합니다.

    구조:
      messages = [
          {"role": "system",    "content": "..."},  # 초기 1회
          {"role": "user",      "content": "1번째 질문"},
          {"role": "assistant", "content": "1번째 답변"},  ← 누적
          {"role": "user",      "content": "2번째 질문"},
          ...
      ]

    예시: 메이저리그 전문가와의 3턴 대화
    """
    print("=" * 55)
    print("[STEP 5] 멀티턴 대화 — 문맥 유지")
    print("=" * 55)

    messages = [
        {"role": "system", "content": "당신은 메이저리그 야구 전문가입니다."},
    ]

    # 대화 시나리오 정의
    conversations = [
        "2024년 월드 시리즈는 LA 다저스가 우승했대! 몇 년 만이지?",
        "4년 전에도 우승했구나, 그 때 활약한 선수는 누구였어?",
        "그 선수는 지금도 다저스 소속이야?",
    ]

    for turn, user_msg in enumerate(conversations, start=1):
        # 사용자 메시지 추가
        messages.append({"role": "user", "content": user_msg})

        response = client.chat.completions.create(
            model=MODEL,   # chatgpt-4o-latest 는 특정 계정만 접근 가능 → MODEL 상수 사용
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )

        assistant_msg = response.choices[0].message.content

        # AI 응답을 이력에 추가 (다음 턴에서 문맥으로 활용)
        messages.append({"role": "assistant", "content": assistant_msg})

        print(f"  [턴 {turn}]")
        print(f"  User      : {user_msg}")
        print(f"  Assistant : {assistant_msg.strip()}")
        print(f"  (누적 메시지 수: {len(messages)}개)")
        print()

    print("  → messages 배열이 쌓일수록 이전 대화 내용을 '기억'합니다.")
    print()


# ══════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "★" * 55)
    print("   주제 2 — 텍스트 생성 & 프롬프트 엔지니어링")
    print("★" * 55 + "\n")

    step1_roles()
    step2_temperature()
    step3_max_tokens_and_n()
    step4_seed()
    step5_multiturn()

    print("=" * 55)
    print("주제 2 실습 완료 ✅")
    print("=" * 55)


if __name__ == "__main__":
    main()