# ============================================================
# Topic 5. Function Call — 외부 함수 연동
# - function 타입 툴을 가진 어시스턴트 생성 및 함수 스키마 정의
# - Run 실행 후 requires_action 상태 처리 흐름 이해
# - submit_tool_outputs으로 함수 실행 결과 전달 및 응답 완성
# ============================================================

import openai
import os
import time
import random
import warnings
from dotenv import load_dotenv

# openai SDK v2에서 Assistants API deprecated 경고 억제
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
# 2. 공통 유틸리티 함수
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
        role_label = "🧑 user" if message.role == "user" else "🤖 assistant"
        print(f"[{role_label}]")
        print(message.content[0].text.value)
        print("---")
    return messages.data


def get_latest_answer(thread) -> str:
    """
    Thread에서 가장 최근 어시스턴트 답변 하나만 가져와 출력합니다.
    각 턴이 끝난 직후 해당 턴의 응답만 확인할 때 사용합니다.
    """
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    latest = messages.data[0]
    answer = latest.content[0].text.value
    print(f"  🤖 어시스턴트 답변:")
    print(f"  {answer}")
    return answer


# ─────────────────────────────────────────
# 3. 실제 Python 함수 정의
#
#    어시스턴트가 "이 함수를 실행해 달라"고 요청하면
#    개발자가 직접 실행하고 결과를 submit_tool_outputs으로 전달합니다.
#    (어시스턴트가 직접 실행하는 것이 아님!)
# ─────────────────────────────────────────

def examine_server() -> str:
    """
    데이터 서버 상태를 점검하는 가상의 함수입니다. (원본 노트북 코드 기반)
    실제 환경에서는 서버 API 호출, DB 쿼리 등으로 구현합니다.

    Returns:
        str: "1" (정상) 또는 "-1, <비정상 이유>" (비정상)
    """
    results = [
        "1",
        "-1, 서버에 버블티를 쏟음",
        "-1, 디스크 용량 초과",
        "-1, 네트워크 연결 불안정",
    ]
    return random.choice(results)


def get_current_time() -> str:
    """현재 날짜와 시각을 반환하는 함수입니다."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ─────────────────────────────────────────
# 4. Function Call 핵심 설정 및 함수
# ─────────────────────────────────────────

# 어시스턴트에 등록할 함수 스키마 목록
FUNCTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "examine_server",
            "description": "데이터 서버가 정상적으로 작동중인지 검사합니다. 정상이면 1, 비정상이면 -1과 이유를 반환합니다.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "현재 날짜와 시각을 반환합니다.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# 함수 이름 → 실제 Python 함수 매핑 테이블
FUNCTION_MAP = {
    "examine_server": examine_server,
    "get_current_time": get_current_time,
}


def create_function_assistant(name: str, instructions: str):
    """Function Call 툴을 탑재한 어시스턴트를 생성합니다."""
    assistant = client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model="gpt-4o-mini",
        tools=FUNCTION_TOOLS,
    )
    print(f"  ✅ 어시스턴트 생성 완료 | ID: {assistant.id}")
    print(f"  등록된 함수: {[t['function']['name'] for t in FUNCTION_TOOLS]}")
    return assistant


def execute_tool_calls(tool_calls: list) -> list:
    """
    어시스턴트가 요청한 tool_calls를 실제로 실행하고
    submit_tool_outputs에 전달할 결과 목록을 반환합니다.
    """
    tool_outputs = []
    for tool_call in tool_calls:
        func_name = tool_call.function.name
        call_id   = tool_call.id
        print(f"\n  🔧 함수 실행 요청: {func_name}() | call_id: {call_id}")
        if func_name in FUNCTION_MAP:
            result = FUNCTION_MAP[func_name]()
            print(f"  📤 실행 결과: {result}")
        else:
            result = f"오류: '{func_name}' 함수를 찾을 수 없습니다."
            print(f"  ❌ {result}")
        tool_outputs.append({"tool_call_id": call_id, "output": str(result)})
    return tool_outputs


def run_with_function_call(thread, assistant):
    """
    Function Call을 포함한 Run 실행 및 완료 처리를 수행합니다.

    상태 흐름:
      [함수 필요한 질문] queued → in_progress → requires_action
                         → (함수 실행 + submit_tool_outputs)
                         → in_progress → completed

      [함수 불필요 질문] queued → in_progress → completed
                         (requires_action 없이 바로 완료)

    tool_choice 전략:
      - "auto" (기본값, 권장): 모델이 맥락에 따라 함수 호출 여부를 스스로 판단
        → instructions에 명확한 조건을 작성해 모델을 유도하는 것이 핵심
      - "required": 모든 질문에 무조건 함수 호출 강제
        → 함수가 불필요한 질문에도 호출되는 부작용 있음 (비권장)
    """
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        # tool_choice 기본값 "auto" 사용
        # → 모델이 질문 맥락에 따라 함수 호출 여부를 스스로 판단
    )
    print(f"  Run ID: {run.id} | 초기 상태: {run.status}")

    while True:
        status = get_run_status(thread, run)
        print(f"  🔄 현재 상태: {status.status}")

        if status.status == "completed":
            print("  ✅ Run 완료!")
            return status

        elif status.status in ("failed", "cancelled", "expired"):
            print(f"  ❌ Run 종료: {status.status}")
            return status

        elif status.status == "requires_action":
            # 어시스턴트가 함수 실행을 요청한 상태
            # submit_tool_outputs을 보내지 않으면 Run이 영원히 대기함
            print("  ⚡ requires_action — 함수 실행 후 결과를 제출합니다.")
            tool_calls = status.required_action.submit_tool_outputs.tool_calls
            print(f"  요청된 함수 수: {len(tool_calls)}개")

            tool_outputs = execute_tool_calls(tool_calls)

            print(f"\n  📨 submit_tool_outputs 전송 중...")
            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs,
            )
            print(f"  제출 완료 | Run 상태: {run.status}")

        time.sleep(1)


# ─────────────────────────────────────────
# 5. 메인 실행
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("  Topic 5. Function Call — 외부 함수 연동")
    print("=" * 55)

    # ── Step 1: Function 타입 툴을 가진 어시스턴트 생성 ──
    print("\n[Step 1] Function Call 어시스턴트 생성 중...")
    server_assistant = create_function_assistant(
        name="서버 관리 챗봇",
        instructions="""당신은 노토랩의 서버 정보를 제공하는 챗봇입니다.
다음 규칙을 반드시 따르세요:

1. 서버 상태, 서버 정상 여부에 관한 질문 → examine_server() 함수를 반드시 먼저 호출하고, 그 결과를 바탕으로 답변하세요.
2. 현재 시각, 날짜, 몇 시인지에 관한 질문 → get_current_time() 함수를 반드시 먼저 호출하고, 그 결과를 바탕으로 답변하세요.
3. 위 두 경우 외의 일반적인 질문 → 함수를 호출하지 말고 직접 친절하게 답변하세요.

함수를 호출할 때는 사용자에게 먼저 공지한 후 실행하고, 단순히 결과만 나열하지 말고 친절하게 설명하세요."""
    )

    # ── Step 2: Thread 생성 ──
    print("\n[Step 2] Thread 생성 중...")
    q1 = "지금 데이터 서버가 잘 작동중인가요?"
    server_thread = create_thread(q1)
    print(f"  ✅ Thread 생성 완료 | ID: {server_thread.id}")
    print(f"  💬 질문: {q1}")

    # ── Step 3: 첫 번째 Run — examine_server() 호출 ──
    print("\n[Step 3] Run 실행 — examine_server() 함수 호출")
    print("-" * 55)
    #
    # 흐름:
    #   queued → in_progress
    #   → requires_action (어시스턴트가 examine_server() 요청)
    #   → 직접 실행 → submit_tool_outputs
    #   → in_progress → completed
    #
    run_with_function_call(server_thread, server_assistant)
    print()
    get_latest_answer(server_thread)

    # ── Step 4: 두 번째 질문 — get_current_time() 함수 호출 ──
    print("\n[Step 4] 두 번째 질문 — get_current_time() 함수 호출")
    print("-" * 55)
    q2 = "지금 몇 시야?"
    print(f"  💬 질문: {q2}")
    client.beta.threads.messages.create(
        thread_id=server_thread.id,
        role="user",
        content=q2,
    )
    print()
    run_with_function_call(server_thread, server_assistant)
    print()
    get_latest_answer(server_thread)

    # ── Step 5: 세 번째 질문 — 함수 불필요 (auto 모드 검증) ──
    print("\n[Step 5] 세 번째 질문 — 함수 호출 없이 바로 답변 (auto 모드 검증)")
    print("-" * 55)
    #
    # tool_choice="auto"(기본값)이므로,
    # 모델이 이 질문엔 함수가 필요 없다고 판단하면
    # requires_action 없이 바로 completed가 됩니다.
    #
    q3 = "안녕? 반가워"
    print(f"  💬 질문: {q3}")
    print(f"  ※ requires_action 없이 바로 completed가 되어야 정상")
    client.beta.threads.messages.create(
        thread_id=server_thread.id,
        role="user",
        content=q3,
    )
    print()
    run_with_function_call(server_thread, server_assistant)
    print()
    get_latest_answer(server_thread)

    # ── Step 6: 전체 누적 대화 흐름 출력 ──
    print("\n[Step 6] 전체 누적 대화 흐름 출력")
    print("=" * 55)
    list_threads_messages(server_thread)

    # ── 리소스 정리 ──
    client.beta.assistants.delete(server_assistant.id)
    print("\n🗑️  어시스턴트 삭제 완료")
    print("✅ Topic 5 실행 완료!")