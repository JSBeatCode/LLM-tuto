"""
주제 1 — API 기초 & 환경 설정
============================
- openai 패키지 버전 확인
- API 키 설정 및 클라이언트 초기화
- API 키 유효성 검증
- 사용 가능한 모델 목록 조회
- 간단한 첫 번째 API 호출
"""

import os
import openai


# ── 상수 ──────────────────────────────────────────────────
API_KEY = ""   # ← 본인의 API 키로 교체하세요
MODEL   = "gpt-4o-mini"


# ══════════════════════════════════════════════════════════
def step1_check_version() -> None:
    """설치된 openai 패키지 버전을 출력합니다."""
    print("=" * 50)
    print("[STEP 1] openai 패키지 버전 확인")
    print("=" * 50)
    print(f"  openai 버전 : {openai.__version__}")
    print()


# ══════════════════════════════════════════════════════════
def step2_init_client() -> openai.OpenAI:
    # 윈도우 CMD 에서: set OPENAI_API_KEY=sk-xxxx (내 OPEN API KEY)
    existing_key = os.environ.get("OPENAI_API_KEY", "")
    
    # 환경변수가 없거나 플레이스홀더 상태면 상수로 덮어쓰기
    if not existing_key or existing_key == "your_api_key_here":
        os.environ["OPENAI_API_KEY"] = API_KEY
        print("  환경변수 OPENAI_API_KEY 를 코드 내 상수로 설정했습니다.")
    else:
        print("  환경변수 OPENAI_API_KEY 가 이미 설정되어 있습니다.")

    client = openai.OpenAI()
    print("  OpenAI 클라이언트 생성 완료")
    print()
    return client

# ══════════════════════════════════════════════════════════
def step3_validate_key(client: openai.OpenAI) -> bool:
    """
    모델 목록 조회를 통해 API 키가 유효한지 검증합니다.

    Returns:
        True  → 키 정상
        False → 키 유효하지 않음
    """
    print("=" * 50)
    print("[STEP 3] API 키 유효성 검증")
    print("=" * 50)

    try:
        client.models.list()
        print("  ✅ API 키가 정상적으로 설정되어 있습니다.")
        print()
        return True
    except openai.AuthenticationError:
        print("  ❌ API 키가 유효하지 않습니다. API_KEY 상수를 확인하세요.")
        print()
        return False
    except openai.APIConnectionError:
        print("  ❌ 네트워크 연결에 실패했습니다. 인터넷 환경을 확인하세요.")
        print()
        return False


# ══════════════════════════════════════════════════════════
def step4_list_models(client: openai.OpenAI) -> None:
    """
    계정에서 사용 가능한 모델 목록을 id 기준 정렬하여 출력합니다.
    GPT 및 o1/o3 계열 모델만 필터링합니다.
    """
    print("=" * 50)
    print("[STEP 4] 사용 가능한 GPT 모델 목록")
    print("=" * 50)

    models = client.models.list()

    # gpt 또는 o1/o3 계열 모델만 필터링 후 id 기준 정렬
    gpt_models = sorted(
        [m for m in models.data if "gpt" in m.id or m.id.startswith("o")],
        key=lambda m: m.id
    )

    for model in gpt_models:
        print(f"  - {model.id}")

    print(f"\n  총 {len(gpt_models)}개의 GPT/추론 모델을 사용할 수 있습니다.")
    print()


# ══════════════════════════════════════════════════════════
def step5_first_api_call(client: openai.OpenAI) -> None:
    
    # 가장 간단한 형태의 Chat Completions 호출 예시입니다.
    # - system : AI의 역할 지정
    # - user   : 사용자 질문
    # 응답 객체의 구조와 메시지 추출 방법을 확인합니다.
    
    print("=" * 50)
    print("[STEP 5] 첫 번째 API 호출")
    print("=" * 50)

    messages = [
        {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다. 답변은 2문장 이내로 간결하게 합니다."},
        {"role": "user",   "content": "OpenAI API 를 배우면 어떤 점이 좋을까요?"},
    ]

    print(f"  모델  : {MODEL}")
    print(f"  질문  : {messages[1]['content']}")
    print()

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    # 응답 객체 구조 살펴보기
    print("  [응답 객체 주요 필드]")
    print(f"  - id                        : {response.id}")
    print(f"  - model                     : {response.model}")
    print(f"  - finish_reason             : {response.choices[0].finish_reason}")
    print(f"  - usage.prompt_tokens       : {response.usage.prompt_tokens}")
    print(f"  - usage.completion_tokens   : {response.usage.completion_tokens}")
    print(f"  - usage.total_tokens        : {response.usage.total_tokens}")
    print()
    print("  [AI 응답 메시지]")
    print(f"  {response.choices[0].message.content}")
    print()


# ══════════════════════════════════════════════════════════
def main() -> None:
    step1_check_version()

    client = step2_init_client()

    is_valid = step3_validate_key(client)
    if not is_valid:
        print("API 키 문제로 이후 단계를 진행할 수 없습니다. 종료합니다.")
        return

    step4_list_models(client)
    step5_first_api_call(client)

    print("=" * 50)
    print("주제 1 실습 완료 ✅")
    print("=" * 50)


if __name__ == "__main__":
    main()