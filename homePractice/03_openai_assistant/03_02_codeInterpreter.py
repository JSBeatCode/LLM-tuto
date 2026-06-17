# ============================================================
# Topic 2. Code Interpreter — 코드 실행 도구 활용
# - code_interpreter 툴을 가진 어시스턴트 생성
# - Thread 생성 → Run 실행 → 상태 확인 전체 흐름 실습
# - RunStep으로 AI가 생성한 중간 코드 확인 (list_run_steps)
# ============================================================

import openai
import os
import time
from dotenv import load_dotenv


# ─────────────────────────────────────────
# 1. 환경 변수 로드 및 클라이언트 초기화
# ─────────────────────────────────────────

load_dotenv(override=True)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    client.models.list()
    print("✅ OPENAI_API_KEY가 정상적으로 설정되어 있습니다.\n")
except Exception:
    print("❌ API 키가 유효하지 않습니다! .env 파일을 확인해 주세요.")
    exit()


# ─────────────────────────────────────────
# 2. 공통 유틸리티 함수 (Topic 1과 동일)
# ─────────────────────────────────────────

def create_thread(message: str):
    """사용자 메시지를 담은 Thread를 생성합니다."""
    thread = client.beta.threads.create(
        messages=[{"role": "user", "content": message}]
    )
    return thread


def create_run(thread, assistant):
    """Thread와 Assistant를 연결하여 Run을 생성(실행)합니다."""
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    return run


def get_run_status(thread, run):
    """실행 중인 Run의 현재 상태를 조회합니다."""
    run_status = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    return run_status


def list_threads_messages(thread):
    """Thread에 저장된 전체 대화 메시지를 시간순으로 출력합니다."""
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    for message in reversed(messages.data):
        print(message.role)
        print(message.content[0].text.value)
        print("---")
    return messages.data


def wait_for_run(thread, run):
    """
    Run이 완료(completed) 또는 오류 상태가 될 때까지 대기합니다.

    Args:
        thread: 실행 중인 스레드 객체
        run: 대기할 런 객체

    Returns:
        Run: 최종 상태의 런 객체
    """
    while True:
        status = get_run_status(thread, run)
        print(f"  🔄 현재 상태: {status.status}")
        if status.status == "completed":
            print("  ✅ Run 완료!\n")
            return status
        elif status.status in ("failed", "cancelled", "expired"):
            print(f"  ❌ Run 종료: {status.status}\n")
            return status
        time.sleep(1)


# ─────────────────────────────────────────
# 3. Code Interpreter 전용 함수
# ─────────────────────────────────────────

def list_run_steps(thread, run, tool_only: bool = True):
    """
    Run의 단계별 실행 내역(RunStep)을 조회하고 출력합니다.

    RunStep에는 두 가지 타입이 있습니다:
      - message_creation : 어시스턴트가 메시지를 생성한 단계
      - tool_calls       : 어시스턴트가 도구(Code Interpreter 등)를 실행한 단계

    Args:
        thread   : 조회할 스레드 객체
        run      : 조회할 런 객체
        tool_only: True이면 tool_calls 단계의 코드만 출력 (기본값: True)

    Returns:
        SyncCursorPage[RunStep]: 런스텝 목록
    """
    run_steps = client.beta.threads.runs.steps.list(
        thread_id=thread.id,
        run_id=run.id,
    )

    print("[ RunStep 실행 내역 ]")
    # 오래된 단계부터 순서대로 출력 (역순 순회)
    for i in range(len(run_steps.data), 0, -1):
        step = run_steps.data[i - 1]
        detail = step.step_details

        if detail.type == "tool_calls":
            print(f"\n▶ Step {len(run_steps.data) - i + 1} | 타입: tool_calls")
            print("  AI가 생성한 코드:")
            print("  " + "-" * 40)
            # Code Interpreter가 생성한 코드 출력
            code_input = detail.tool_calls[0].code_interpreter.input
            for line in code_input.split("\n"):
                print(f"  {line}")
            print("  " + "-" * 40)

        elif not tool_only:
            print(f"\n▶ Step {len(run_steps.data) - i + 1} | 타입: message_creation")

        print()

    return run_steps


# ─────────────────────────────────────────
# 4. 메인 실행
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("  Topic 2. Code Interpreter — 코드 실행 도구 활용")
    print("=" * 55)

    # ── Step 1: Code Interpreter 툴을 가진 어시스턴트 생성 ──
    print("\n[Step 1] 수학 선생님 어시스턴트 생성 중...")

    math_assistant = client.beta.assistants.create(
        name="수학 선생님",
        instructions="파이썬 코드를 사용해 주어진 문제를 해결하고, 풀이과정을 자세히 설명하세요.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4o-mini",   # 접근 가능한 모델로 설정 (원본: gpt-4o)
        temperature=0.2
    )
    print(f"  ✅ 어시스턴트 생성 완료")
    print(f"  ID       : {math_assistant.id}")
    print(f"  모델     : {math_assistant.model}")
    print(f"  툴       : {[t.type for t in math_assistant.tools]}")

    # ── Step 2: Thread 생성 ──
    print("\n[Step 2] Thread 생성 중...")

    question = "413보다 큰 소수 중 네번째로 작은 소수의 세제곱수는 무엇입니까?"
    math_thread = create_thread(question)

    print(f"  ✅ Thread 생성 완료 | ID: {math_thread.id}")
    print(f"  질문: {question}")

    # ── Step 3: Run 생성 및 실행 ──
    print("\n[Step 3] Run 생성 및 실행 중...")

    math_run = create_run(math_thread, math_assistant)
    print(f"  ✅ Run 생성 완료 | ID: {math_run.id}")
    print(f"  초기 상태: {math_run.status}")

    # ── Step 4: Run 완료까지 상태 확인 대기 ──
    print("\n[Step 4] Run 상태 확인 중...")
    math_run = wait_for_run(math_thread, math_run)

    # ── Step 5: RunStep으로 AI가 생성한 중간 코드 확인 ──
    print("\n[Step 5] RunStep — AI가 생성한 코드 확인")
    print("-" * 55)
    list_run_steps(math_thread, math_run, tool_only=True)

    # ── Step 6: 최종 대화 내용 출력 ──
    print("\n[Step 6] 최종 대화 내용 출력")
    print("-" * 55)
    list_threads_messages(math_thread)

    # ── 리소스 정리 ──
    client.beta.assistants.delete(math_assistant.id)
    print("\n🗑️  어시스턴트 삭제 완료")
    print("✅ Topic 2 실행 완료!")