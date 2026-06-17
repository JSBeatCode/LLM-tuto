"""
주제 3 — 토큰 & 비용 관리
==========================
학습 내용:
  1. tiktoken 으로 모델별 토크나이저 확인
  2. 텍스트 → 토큰 ID 변환 및 토큰 수 계산
  3. 한국어 vs 영어 토큰 효율 비교
  4. response.usage 분석 (prompt / completion / total tokens)
  5. cached_tokens · reasoning_tokens · audio_tokens 개념 이해
  6. 모델별 가격표 기반 호출 비용 자동 계산
  7. 비용 최적화 팁 실습 (max_tokens 제한, 짧은 system 프롬프트)
"""

import os
from dotenv import load_dotenv
import openai
import tiktoken


# ── 환경변수 로드 ─────────────────────────────────────────
load_dotenv(override=True)  # 같은 폴더의 .env 파일을 자동으로 읽어옵니다

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL = "gpt-4o-mini"

# ── 모델별 1K 토큰당 가격 (USD, 2024년 기준) ──────────────
# 최신 가격은 https://openai.com/pricing 에서 확인하세요.
PRICING = {
    "gpt-4o": {
        "input":  0.0025,   # $2.50  / 1M tokens
        "output": 0.0100,   # $10.00 / 1M tokens
    },
    "gpt-4o-mini": {
        "input":  0.000150, # $0.150 / 1M tokens
        "output": 0.000600, # $0.600 / 1M tokens
    },
}


def calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """prompt/completion 토큰 수와 모델명으로 예상 비용(USD)을 계산합니다."""
    if model not in PRICING:
        return -1.0
    price = PRICING[model]
    cost = (prompt_tokens / 1_000_000) * price["input"] \
         + (completion_tokens / 1_000_000) * price["output"]
    return cost


# ══════════════════════════════════════════════════════════
def step1_tokenizer() -> None:
    """
    [STEP 1] tiktoken — 모델별 토크나이저 확인
    -------------------------------------------
    OpenAI 모델마다 사용하는 토크나이저(Encoding)가 다릅니다.
    - gpt-4o 계열  : o200k_base
    - gpt-3.5/4    : cl100k_base
    tiktoken.encoding_for_model() 로 해당 모델의 토크나이저를 가져옵니다.
    """
    print("=" * 55)
    print("[STEP 1] tiktoken — 모델별 토크나이저 확인")
    print("=" * 55)

    models_to_check = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"]

    for m in models_to_check:
        enc = tiktoken.encoding_for_model(m)
        print(f"  {m:<20} → Encoding: {enc.name}")

    print()


# ══════════════════════════════════════════════════════════
def step2_count_tokens() -> None:
    """
    [STEP 2] 텍스트 → 토큰 ID 변환 및 토큰 수 계산
    -------------------------------------------------
    encode() : 문자열을 토큰 ID 리스트로 변환
    decode() : 토큰 ID 리스트를 다시 문자열로 복원

    글자 수 != 토큰 수 임을 직접 확인합니다.
    """
    print("=" * 55)
    print("[STEP 2] 텍스트 → 토큰 변환 및 토큰 수 계산")
    print("=" * 55)

    enc = tiktoken.encoding_for_model("gpt-4o")

    prompt = "GPT 모델별 토크나이저를 확인하고, 프롬프트 토큰의 개수를 구할 수 있습니다."
    tokens = enc.encode(prompt)

    print(f"  원본 텍스트  : {prompt}")
    print(f"  토큰 ID 목록 : {tokens}")
    print(f"  글자 수      : {len(prompt)}")
    print(f"  토큰 수      : {len(tokens)}")
    print()

    # 토큰 ID → 텍스트 복원
    print("  [토큰 ID별 대응 텍스트]")
    for tid in tokens:
        piece = enc.decode([tid])
        print(f"    {tid:>8}  →  {repr(piece)}")

    print()


# ══════════════════════════════════════════════════════════
def step3_kor_vs_eng() -> None:
    """
    [STEP 3] 한국어 vs 영어 토큰 효율 비교
    ----------------------------------------
    GPT 토크나이저는 영어 기준으로 최적화되어 있습니다.
    같은 의미의 문장이라도 한국어가 더 많은 토큰을 소비할 수 있습니다.
    → 비용 측면에서 언어 선택이 영향을 미칠 수 있습니다.
    """
    print("=" * 55)
    print("[STEP 3] 한국어 vs 영어 토큰 효율 비교")
    print("=" * 55)

    enc = tiktoken.encoding_for_model("gpt-4o")

    pairs = [
        ("안녕하세요, 저는 인공지능 어시스턴트입니다.",
         "Hello, I am an artificial intelligence assistant."),
        ("오늘 날씨가 매우 맑고 따뜻합니다.",
         "The weather is very clear and warm today."),
        ("대한민국의 수도는 서울입니다.",
         "The capital of South Korea is Seoul."),
    ]

    for kor, eng in pairs:
        kor_tokens = len(enc.encode(kor))
        eng_tokens = len(enc.encode(eng))
        ratio = kor_tokens / eng_tokens
        print(f"  KOR ({kor_tokens:>2} tokens) : {kor}")
        print(f"  ENG ({eng_tokens:>2} tokens) : {eng}")
        print(f"  → 한국어가 영어 대비 {ratio:.1f}배 토큰 사용\n")

    print()


# ══════════════════════════════════════════════════════════
def step4_response_usage() -> None:
    """
    [STEP 4] response.usage 분석
    ------------------------------
    API 호출 후 응답 객체의 usage 필드에서 토큰 사용량을 확인합니다.

    주요 필드:
      - prompt_tokens       : 입력(system+user 메시지) 토큰 수
      - completion_tokens   : 출력(assistant 응답) 토큰 수
      - total_tokens        : prompt + completion 합계

    세부 필드 (completion_tokens_details):
      - reasoning_tokens    : o1 계열 추론 토큰 (내부 Chain-of-Thought)
      - audio_tokens        : 음성 입출력 토큰
      - accepted_prediction_tokens : 예측 자동완성 수락 토큰

    세부 필드 (prompt_tokens_details):
      - cached_tokens       : 동일 입력 재사용 시 캐시된 토큰 (50% 할인)
      - audio_tokens        : 음성 입력 토큰
    """
    print("=" * 55)
    print("[STEP 4] response.usage 분석")
    print("=" * 55)

    messages = [
        {"role": "system", "content": "당신은 파이썬 전문가입니다."},
        {"role": "user",   "content": "파이썬 리스트 컴프리헨션의 장점을 3가지 알려주세요."},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=300,
    )

    usage = response.usage
    print(f"  모델              : {response.model}")
    print()
    print(f"  [기본 토큰 사용량]")
    print(f"  prompt_tokens     : {usage.prompt_tokens}")
    print(f"  completion_tokens : {usage.completion_tokens}")
    print(f"  total_tokens      : {usage.total_tokens}")
    print()

    # 세부 토큰 정보
    c_detail = usage.completion_tokens_details
    p_detail = usage.prompt_tokens_details
    print(f"  [completion_tokens_details]")
    print(f"  reasoning_tokens            : {c_detail.reasoning_tokens}")
    print(f"  audio_tokens                : {c_detail.audio_tokens}")
    print(f"  accepted_prediction_tokens  : {c_detail.accepted_prediction_tokens}")
    print()
    print(f"  [prompt_tokens_details]")
    print(f"  cached_tokens  : {p_detail.cached_tokens}  ← 0이면 캐시 미사용 (첫 호출)")
    print(f"  audio_tokens   : {p_detail.audio_tokens}")
    print()

    # 비용 계산
    cost = calc_cost(MODEL, usage.prompt_tokens, usage.completion_tokens)
    print(f"  예상 비용 : ${cost:.8f} USD")
    print()
    print(f"  [AI 응답]")
    print(f"  {response.choices[0].message.content.strip()}")
    print()


# ══════════════════════════════════════════════════════════
def step5_cached_tokens() -> None:
    """
    [STEP 5] cached_tokens — 동일 입력 캐시 할인
    -----------------------------------------------
    완전히 동일한 system + user 메시지를 짧은 시간 안에 두 번 호출하면
    두 번째 호출부터 prompt_tokens_details.cached_tokens 값이 올라갑니다.
    캐시된 토큰은 입력 비용의 50% 할인이 적용됩니다.

    ※ 캐시는 정확히 동일한 prefix 에만 적용되며,
      내용이 조금이라도 바뀌면 캐시 미사용 상태로 돌아갑니다.
    """
    print("=" * 55)
    print("[STEP 5] cached_tokens — 동일 입력 재호출 실험")
    print("=" * 55)

    # 긴 system 프롬프트를 사용해야 캐시 효과가 나타납니다 (1024 토큰 이상 권장)
    long_system = (
        "당신은 데이터 분석 전문가입니다. "
        "사용자의 질문에 정확하고 친절하게 답변하세요. " * 60  # 반복으로 길이 확보
    )
    messages = [
        {"role": "system", "content": long_system},
        {"role": "user",   "content": "데이터 분석이란 무엇인가요? 한 문장으로 설명해주세요."},
    ]

    for trial in range(1, 3):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=80,
        )
        usage = response.usage
        p_detail = usage.prompt_tokens_details
        print(f"  [호출 #{trial}]")
        print(f"  prompt_tokens   : {usage.prompt_tokens}")
        print(f"  cached_tokens   : {p_detail.cached_tokens}"
              + ("  ← 캐시 적용! (50% 할인)" if p_detail.cached_tokens > 0 else "  ← 캐시 미적용 (첫 호출)"))
        print()

    print("  ※ cached_tokens 가 0 이라면 호출 간격이 너무 길거나")
    print("    캐시 임계값(1024 토큰)에 미달된 것일 수 있습니다.")
    print()


# ══════════════════════════════════════════════════════════
def step6_cost_comparison() -> None:
    """
    [STEP 6] 모델 & max_tokens 조합별 비용 비교
    ----------------------------------------------
    같은 질문을 gpt-4o / gpt-4o-mini 로 각각 호출하여
    실제 토큰 사용량과 비용 차이를 비교합니다.

    비용 최적화 전략:
      ① 작업 난이도에 맞는 모델 선택 (무조건 큰 모델 X)
      ② max_tokens 로 불필요한 장황한 답변 억제
      ③ system 프롬프트를 간결하게 유지
      ④ 반복 prefix 는 캐시 활용
    """
    print("=" * 55)
    print("[STEP 6] 모델 & max_tokens 조합별 비용 비교")
    print("=" * 55)

    question = "파이썬에서 딕셔너리와 리스트의 차이를 간단히 설명해주세요."
    messages = [
        {"role": "system", "content": "당신은 파이썬 강사입니다. 간결하게 답변하세요."},
        {"role": "user",   "content": question},
    ]

    scenarios = [
        ("gpt-4o",      500),
        ("gpt-4o",      100),
        ("gpt-4o-mini", 500),
        ("gpt-4o-mini", 100),
    ]

    print(f"  {'모델':<15} {'max_tokens':>10} {'prompt':>8} {'completion':>12} {'total':>7}  {'비용(USD)':>14}")
    print("  " + "-" * 73)

    for model, max_tok in scenarios:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tok,
        )
        u = resp.usage
        cost = calc_cost(model, u.prompt_tokens, u.completion_tokens)
        print(f"  {model:<15} {max_tok:>10} {u.prompt_tokens:>8} "
              f"{u.completion_tokens:>12} {u.total_tokens:>7}  ${cost:.8f}")

    print()
    print("  → max_tokens 를 줄이거나 gpt-4o-mini 를 사용하면 비용이 크게 절감됩니다.")
    print()


# ══════════════════════════════════════════════════════════
def step7_token_budget_tip() -> None:
    """
    [STEP 7] 실무 비용 최적화 팁 — 사전 토큰 수 확인
    --------------------------------------------------
    API 를 호출하기 전에 tiktoken 으로 토큰 수를 미리 계산하면
    예산 초과를 방지하고 max_tokens 값을 적절히 설정할 수 있습니다.

    메시지 리스트 전체의 토큰 수를 계산하는 헬퍼 함수를 작성합니다.
    """
    print("=" * 55)
    print("[STEP 7] 실무 팁 — 호출 전 토큰 수 사전 계산")
    print("=" * 55)

    def count_messages_tokens(messages: list[dict], model: str = "gpt-4o-mini") -> int:
        """
        messages 배열 전체의 토큰 수를 계산합니다.
        실제 API 청구 토큰과 약간 차이가 있을 수 있습니다.
        (메시지 구조 오버헤드 미포함)
        """
        enc = tiktoken.encoding_for_model(model)
        total = 0
        for msg in messages:
            total += len(enc.encode(msg["role"]))
            total += len(enc.encode(msg["content"]))
        return total

    test_messages = [
        {"role": "system",  "content": "당신은 여행 전문가입니다."},
        {"role": "user",    "content": "서울에서 부산까지 기차로 여행할 때 추천 코스를 알려주세요."},
        {"role": "assistant","content": "KTX를 이용하시면 약 2시간 30분이 소요됩니다."},
        {"role": "user",    "content": "부산에서 꼭 먹어야 할 음식은 무엇인가요?"},
    ]

    token_count = count_messages_tokens(test_messages, MODEL)
    cost_per_call = calc_cost(MODEL, token_count, 0)

    print(f"  메시지 수        : {len(test_messages)}개")
    print(f"  사전 계산 토큰   : {token_count} tokens")
    print(f"  입력만의 비용    : ${cost_per_call:.8f} USD")
    print()

    # gpt-4o-mini 컨텍스트 윈도우 한도 안내
    context_limit = 128_000
    remaining = context_limit - token_count
    print(f"  모델 컨텍스트 한도 : {context_limit:,} tokens")
    print(f"  남은 여유 토큰     : {remaining:,} tokens ({remaining/context_limit*100:.1f}%)")
    print()
    print("  [실무 활용 체크리스트]")
    print("  ✅ 호출 전 토큰 수 확인으로 컨텍스트 초과 방지")
    print("  ✅ max_tokens 를 응답 예상 길이에 맞게 설정")
    print("  ✅ 단순 작업은 gpt-4o-mini, 복잡한 추론은 gpt-4o 사용")
    print("  ✅ 반복 호출 시 동일 prefix 유지로 cached_tokens 활용")
    print("  ✅ 불필요하게 긴 system 프롬프트 지양")
    print()


# ══════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "★" * 55)
    print("   주제 3 — 토큰 & 비용 관리")
    print("★" * 55 + "\n")

    step1_tokenizer()
    step2_count_tokens()
    step3_kor_vs_eng()
    step4_response_usage()
    step5_cached_tokens()
    step6_cost_comparison()
    step7_token_budget_tip()

    print("=" * 55)
    print("주제 3 실습 완료 ✅")
    print("=" * 55)


if __name__ == "__main__":
    main()