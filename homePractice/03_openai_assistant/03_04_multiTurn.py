# ============================================================
# Topic 4. Multi-Turn 대화 구현
# - 기존 스레드에 메시지 추가 (add_thread_message)
# - Run을 재실행해 이전 대화 맥락을 유지하며 연속 질문
# - 누적된 대화 흐름 전체 확인
#
# ※ Multi-Turn은 어떤 어시스턴트에도 적용 가능한 개념입니다.
#    이 파일에서는 Topic 3의 햄릿 롤플레잉 봇을 재사용해 시연합니다.
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
# 2. 공통 유틸리티 함수 (Topic 1~3과 동일)
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
    """
    Thread에 저장된 전체 대화 메시지를 시간순으로 출력합니다.
    Multi-Turn에서는 대화가 누적되므로 전체 흐름을 한눈에 확인할 수 있습니다.
    """
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    for message in reversed(messages.data):
        print(message.role)
        print(message.content[0].text.value)
        print("---")
    return messages.data


def wait_for_run(thread, run):
    """Run이 완료 또는 오류 상태가 될 때까지 1초 간격으로 대기합니다."""
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
# 3. Multi-Turn 전용 함수
# ─────────────────────────────────────────

def add_thread_message(thread, message: str):
    """
    기존 Thread에 새 사용자 메시지를 추가합니다.

    Chat Completion API와의 차이점:
      - Chat API : 매 요청마다 전체 대화 히스토리를 messages 배열에 직접 담아 전달
      - Assistants API : Thread가 대화를 서버에서 관리하므로
                         새 메시지만 추가하면 이전 맥락이 자동으로 유지됨

    Args:
        thread     : 메시지를 추가할 기존 스레드 객체
        message    : 추가할 사용자 메시지 문자열

    Returns:
        Thread: 원본 스레드 객체 (메시지 추가 후)
    """
    thread_message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message,
    )
    print(f"  📨 메시지 추가됨 | ID: {thread_message.id}")
    print(f"  내용: {message}")
    return thread


def run_and_wait(thread, assistant):
    """
    Thread에 Run을 생성하고 완료까지 대기하는 헬퍼 함수입니다.
    Multi-Turn에서 매 턴마다 반복 사용합니다.

    Args:
        thread   : 실행할 스레드 객체
        assistant: 연결할 어시스턴트 객체

    Returns:
        Run: 완료된 런 객체
    """
    run = create_run(thread, assistant)
    print(f"  Run ID: {run.id} | 초기 상태: {run.status}")
    return wait_for_run(thread, run)


def get_latest_answer(thread):
    """
    Thread에서 가장 최근 어시스턴트 답변 하나만 가져와 출력합니다.
    Multi-Turn에서 각 턴의 응답만 간결하게 확인할 때 사용합니다.

    Args:
        thread: 조회할 스레드 객체

    Returns:
        str: 가장 최근 어시스턴트 답변 텍스트
    """
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    # 가장 최근 메시지(index 0)가 어시스턴트 답변
    latest = messages.data[0]
    answer = latest.content[0].text.value
    print(f"\n  🤖 어시스턴트 답변:")
    print(f"  {answer}")
    return answer


# ─────────────────────────────────────────
# 4. 어시스턴트 + Vector Store 준비 함수
#    (Topic 3의 햄릿 롤플레잉 봇 재사용)
# ─────────────────────────────────────────

def setup_roleplay_assistant(pdf_path: str):
    """
    햄릿 롤플레잉 어시스턴트와 Vector Store를 생성·연결합니다.
    Multi-Turn 실습의 무대 설정용 함수입니다.

    Args:
        pdf_path (str): 업로드할 PDF 파일 경로

    Returns:
        tuple: (assistant, vector_store)
    """
    # Vector Store 생성 및 파일 업로드
    print("  [준비] Vector Store 생성 중...")
    vector_store = client.vector_stores.create(name="햄릿_멀티턴")

    print(f"  [준비] PDF 업로드 중: {pdf_path}")
    file_batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=[open(pdf_path, "rb")]
    )
    print(f"  [준비] 업로드 완료 | 상태: {file_batch.status}")

    # 어시스턴트 생성
    assistant = client.beta.assistants.create(
        name="햄릿 롤플레잉 봇",
        instructions="""당신은 문학 속의 인물이 되어, 사용자와 롤 플레이를 해야 합니다.
주어진 파일의 문체와 스타일을 참고하여, 실감나고 사실적으로 답변하세요.
인물의 극중 말투와 최대한 유사하게 답변하세요.""",
        model="gpt-4o-mini",
        tools=[{"type": "file_search"}],
        temperature=0
    )

    # 벡터스토어 연결
    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
    )
    print(f"  [준비] 어시스턴트 생성 및 Vector Store 연결 완료 | ID: {assistant.id}")

    return assistant, vector_store


# ─────────────────────────────────────────
# 5. 메인 실행
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("  Topic 4. Multi-Turn 대화 구현")
    print("=" * 55)

    # ── 사전 준비: 어시스턴트 및 Vector Store 세팅 ──
    print("\n[사전 준비] 햄릿 롤플레잉 어시스턴트 세팅 중...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PDF_PATH = os.path.join(BASE_DIR, "Hamlet_KOR.pdf")   # ← 파일명 필요 시 수정

    roleplay_assistant, hamlet_store = setup_roleplay_assistant(PDF_PATH)

    # ══════════════════════════════════════════════
    # Turn 1 : 첫 번째 질문 — Thread 최초 생성
    # ══════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  [Turn 1] 첫 번째 질문")
    print("=" * 55)

    first_question = "햄릿, 당신은 어떻게 죽었나요?"
    print(f"\n  💬 질문: {first_question}")

    # Thread 생성 (첫 질문 포함)
    roleplay_thread = create_thread(first_question)
    print(f"  Thread ID: {roleplay_thread.id}")

    # Run 실행 및 대기
    print("\n  Run 실행 중...")
    run_and_wait(roleplay_thread, roleplay_assistant)

    # 최신 답변만 확인
    get_latest_answer(roleplay_thread)

    # ══════════════════════════════════════════════
    # Turn 2 : 두 번째 질문 — 기존 Thread에 메시지 추가
    # ══════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  [Turn 2] 두 번째 질문 — 기존 Thread에 메시지 추가")
    print("=" * 55)
    #
    # 핵심 포인트:
    #   새 Thread를 만들지 않고, 기존 roleplay_thread에 메시지를 추가합니다.
    #   Thread가 이전 대화를 서버에서 유지하고 있으므로,
    #   어시스턴트는 Turn 1의 맥락을 그대로 이어받아 답변합니다.
    #

    second_question = "후회하거나 되돌리고 싶은 것이 있나요?"
    print(f"\n  💬 질문: {second_question}")

    # 기존 스레드에 메시지 추가
    print("\n  기존 Thread에 메시지 추가 중...")
    add_thread_message(roleplay_thread, second_question)

    # Run 재실행 및 대기
    print("\n  Run 재실행 중...")
    run_and_wait(roleplay_thread, roleplay_assistant)

    # 최신 답변만 확인
    get_latest_answer(roleplay_thread)

    # ══════════════════════════════════════════════
    # Turn 3 : 세 번째 질문 — 맥락이 계속 이어짐
    # ══════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  [Turn 3] 세 번째 질문 — 맥락 연속성 확인")
    print("=" * 55)

    third_question = "오필리아에 대한 감정은 어땠나요?"
    print(f"\n  💬 질문: {third_question}")

    print("\n  기존 Thread에 메시지 추가 중...")
    add_thread_message(roleplay_thread, third_question)

    print("\n  Run 재실행 중...")
    run_and_wait(roleplay_thread, roleplay_assistant)

    get_latest_answer(roleplay_thread)

    # ══════════════════════════════════════════════
    # 전체 대화 흐름 확인
    # ══════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  [전체 대화 흐름] 누적된 Thread 메시지 전체 출력")
    print("=" * 55)
    print()
    list_threads_messages(roleplay_thread)

    # ── 리소스 정리 ──
    client.beta.assistants.delete(roleplay_assistant.id)
    client.vector_stores.delete(hamlet_store.id)
    print("\n🗑️  어시스턴트 및 Vector Store 삭제 완료")
    print("✅ Topic 4 실행 완료!")