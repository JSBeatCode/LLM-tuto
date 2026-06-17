# ============================================================
# Topic 1. 환경 설정 & Assistants API 핵심 구조 이해
# - openai 패키지 설치 및 API 키 설정
# - Assistant / Thread / Run / RunStep 4가지 핵심 객체 개념
# - 공통 유틸리티 함수 정의
# ============================================================

import openai
import os
from dotenv import load_dotenv


# ─────────────────────────────────────────
# 1. 환경 변수 로드 및 클라이언트 초기화
# ─────────────────────────────────────────

load_dotenv(override=True)  # .env 파일에서 환경 변수 로드
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API 키 유효성 검증
try:
    client.models.list()
    print("✅ OPENAI_API_KEY가 정상적으로 설정되어 있습니다.\n")
except Exception:
    print("❌ API 키가 유효하지 않습니다! .env 파일을 확인해 주세요.")
    exit()


# ─────────────────────────────────────────
# 2. Assistants API 핵심 구조 설명 출력
# ─────────────────────────────────────────
#
# Assistants API는 4가지 핵심 객체로 구성됩니다:
#
# ┌─────────────┬──────────────────────────────────────────────────┐
# │   객체       │  역할                                            │
# ├─────────────┼──────────────────────────────────────────────────┤
# │ Assistant   │ LLM + Tool이 결합된 AI 에이전트 객체             │
# │ Thread      │ 대화 메시지가 순서대로 저장되는 공간             │
# │ Run         │ Assistant와 Thread를 연결해 실행하는 객체        │
# │ RunStep     │ Run 실행 시 메시지/툴 사용을 단계별로 기록       │
# └─────────────┴──────────────────────────────────────────────────┘
#
# 동작 흐름:
#   1) Assistant 생성 (이름, 지침, 툴, 모델 설정)
#   2) Thread 생성 (사용자 메시지 포함)
#   3) Run 생성 (Assistant + Thread 연결 → 실행)
#   4) Run 상태 확인 (queued → in_progress → completed)
#   5) Thread 메시지에서 최종 답변 조회


# ─────────────────────────────────────────
# 3. 공통 유틸리티 함수 정의
# ─────────────────────────────────────────

def create_thread(message: str):
    """
    사용자 메시지를 담은 Thread를 생성합니다.

    Args:
        message (str): 스레드에 담을 첫 번째 사용자 메시지

    Returns:
        Thread: 생성된 스레드 객체
    """
    thread = client.beta.threads.create(
        messages=[{"role": "user", "content": message}]
    )
    return thread


def create_run(thread, assistant):
    """
    Thread와 Assistant를 연결하여 Run을 생성(실행)합니다.

    Args:
        thread: 실행할 스레드 객체
        assistant: 연결할 어시스턴트 객체

    Returns:
        Run: 생성된 런 객체
    """
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    return run


def get_run_status(thread, run):
    """
    실행 중인 Run의 현재 상태를 조회합니다.

    Args:
        thread: 조회할 스레드 객체
        run: 조회할 런 객체
    

    Returns:
        Run: 최신 상태가 반영된 런 객체
        (status 값: queued / in_progress / requires_action / completed / failed)
    """
    run_status = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    return run_status


def list_threads_messages(thread):
    """
    Thread에 저장된 전체 대화 메시지를 출력합니다.
    최신 메시지가 먼저 조회되므로 역순으로 출력합니다.

    Args:
        thread: 메시지를 조회할 스레드 객체

    Returns:
        list: 메시지 객체 리스트
    """
    messages = client.beta.threads.messages.list(thread_id=thread.id)

    # 역순으로 출력 (오래된 메시지 → 최신 메시지 순)
    for message in reversed(messages.data):
        print(message.role)
        print(message.content[0].text.value)
        print("---")

    return messages.data


def add_thread_message(thread, message: str):
    """
    기존 Thread에 새 사용자 메시지를 추가합니다. (Multi-Turn 대화용)

    Args:
        thread: 메시지를 추가할 스레드 객체
        message (str): 추가할 사용자 메시지

    Returns:
        Thread: 원본 스레드 객체 (메시지 추가 후)
    """
    thread_message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message,
    )
    print(f"📨 메시지 추가됨: {thread_message.id}")
    return thread


# ─────────────────────────────────────────
# 4. 동작 확인 — 샘플 어시스턴트로 전체 흐름 테스트
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 50)
    print("  Assistants API 핵심 구조 동작 확인")
    print("=" * 50)

    # Step 1: Assistant 생성
    print("\n[Step 1] Assistant 생성 중...")
    sample_assistant = client.beta.assistants.create(
        name="샘플 어시스턴트",
        instructions="사용자의 질문에 간결하고 친절하게 답변하세요.",
        model="gpt-4o-mini",
    )
    print(f"  ✅ Assistant 생성 완료 | ID: {sample_assistant.id}")

    # Step 2: Thread 생성
    print("\n[Step 2] Thread 생성 중...")
    sample_thread = create_thread("안녕하세요! 간단히 자기소개 해줄 수 있나요?")
    print(f"  ✅ Thread 생성 완료 | ID: {sample_thread.id}")

    # Step 3: Run 생성 (실행)
    print("\n[Step 3] Run 생성 중...")
    sample_run = create_run(sample_thread, sample_assistant)
    print(f"  ✅ Run 생성 완료 | ID: {sample_run.id} | 초기 상태: {sample_run.status}")

    # Step 4: Run 상태 확인 (완료까지 대기)
    print("\n[Step 4] Run 상태 확인 중...")
    import time
    while True:
        status = get_run_status(sample_thread, sample_run)
        print(f"  🔄 현재 상태: {status.status}")
        if status.status == "completed":
            print("  ✅ Run 완료!")
            break
        elif status.status in ("failed", "cancelled", "expired"):
            print(f"  ❌ Run 실패: {status.status}")
            break
        time.sleep(1)

    # Step 5: 최종 대화 내용 출력
    print("\n[Step 5] 대화 내용 출력")
    print("-" * 30)
    list_threads_messages(sample_thread)

    # 리소스 정리 (생성한 어시스턴트 삭제)
    client.beta.assistants.delete(sample_assistant.id)
    print("\n🗑️  샘플 어시스턴트 삭제 완료")
    print("\n✅ Topic 1 실행 완료!")