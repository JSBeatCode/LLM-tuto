# ============================================================
# Topic 3. File Search & Vector Store (RAG)
# - Vector Store 생성 및 PDF 파일 업로드
# - file_search 툴을 가진 어시스턴트 생성
# - 벡터스토어를 어시스턴트에 연결 (assistants.update)
# - 파일 기반 질의응답 및 출처 인용(【source】) 확인
# ============================================================

import openai
import os
import time
from importlib.metadata import version
from packaging.version import Version
from dotenv import load_dotenv

# openai SDK 버전 확인 출력 (v2.x → client.vector_stores, v1.x → client.beta.vector_stores)
_openai_version = version("openai")
print(f"ℹ️  openai SDK 버전: {_openai_version}")


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
# 2. 공통 유틸리티 함수 (Topic 1, 2와 동일)
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
# 3. File Search & Vector Store 전용 함수
# ─────────────────────────────────────────

def create_vector_store(name: str):
    """
    Vector Store를 생성합니다.
    Vector Store는 업로드한 파일을 청크 단위로 분할·저장하는 공간입니다.
    File Search 툴이 이 저장소를 검색해 관련 내용을 찾아냅니다. (RAG 구조)

    ※ openai SDK v2부터 client.beta.vector_stores → client.vector_stores 로 변경됨

    Args:
        name (str): 벡터스토어 이름

    Returns:
        VectorStore: 생성된 벡터스토어 객체
    """
    vector_store = client.vector_stores.create(name=name)
    print(f"  ✅ Vector Store 생성 완료")
    print(f"  ID  : {vector_store.id}")
    print(f"  이름: {vector_store.name}")
    return vector_store


def upload_file_to_vector_store(vector_store, file_path: str):
    """
    PDF 파일을 Vector Store에 업로드하고 처리 완료까지 대기합니다.
    upload_and_poll을 사용해 업로드 + 처리 완료를 한 번에 처리합니다.

    ※ openai SDK v2부터 client.beta.vector_stores → client.vector_stores 로 변경됨

    Args:
        vector_store : 업로드 대상 벡터스토어 객체
        file_path (str): 업로드할 파일 경로 (PDF 권장)

    Returns:
        VectorStoreFileBatch: 파일 배치 객체 (status, file_counts 포함)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    print(f"  📂 파일 업로드 중: {file_path}")
    file_streams = [open(file_path, "rb")]

    file_batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=file_streams
    )

    print(f"  ✅ 업로드 완료 | 상태: {file_batch.status}")
    print(f"  파일 수: {file_batch.file_counts}")
    return file_batch


def create_file_search_assistant(name: str, instructions: str):
    """
    File Search 툴을 탑재한 어시스턴트를 생성합니다.
    생성 시점에는 벡터스토어가 연결되지 않으며,
    이후 attach_vector_store()로 연결합니다.

    Args:
        name (str)        : 어시스턴트 이름
        instructions (str): 어시스턴트 행동 지침

    Returns:
        Assistant: 생성된 어시스턴트 객체
    """
    assistant = client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model="gpt-4o-mini",
        tools=[{"type": "file_search"}],
        temperature=0
    )
    print(f"  ✅ 어시스턴트 생성 완료")
    print(f"  ID  : {assistant.id}")
    print(f"  툴  : {[t.type for t in assistant.tools]}")
    return assistant


def attach_vector_store(assistant, vector_store):
    """
    생성된 Vector Store를 어시스턴트에 연결합니다.
    연결 후 어시스턴트는 File Search 시 해당 벡터스토어를 검색합니다.

    Args:
        assistant   : 업데이트할 어시스턴트 객체
        vector_store: 연결할 벡터스토어 객체

    Returns:
        Assistant: 벡터스토어가 연결된 업데이트된 어시스턴트 객체
    """
    updated_assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store.id]
            }
        },
    )
    print(f"  ✅ Vector Store 연결 완료")
    print(f"  연결된 Vector Store ID: {vector_store.id}")
    return updated_assistant


# ─────────────────────────────────────────
# 4. 메인 실행
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("  Topic 3. File Search & Vector Store (RAG)")
    print("=" * 55)

    # ── Step 1: Vector Store 생성 ──
    print("\n[Step 1] Vector Store 생성 중...")
    hamlet_store = create_vector_store(name="햄릿")

    # ── Step 2: PDF 파일 업로드 ──
    # 📌 파일명만 수정하면 됩니다. 이 .py 파일과 같은 폴더에 PDF를 두세요.
    #    지원 파일 형식: https://platform.openai.com/docs/assistants/tools/file-search/supported-files
    print("\n[Step 2] PDF 파일 Vector Store에 업로드 중...")
    # __file__ 기준으로 경로를 잡으면 어느 디렉토리에서 실행해도 파일을 찾을 수 있음
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PDF_FILE_PATH = os.path.join(BASE_DIR, "Hamlet_KOR.pdf")   # ← 파일명만 수정
    upload_file_to_vector_store(hamlet_store, PDF_FILE_PATH)

    # ── Step 3: File Search 어시스턴트 생성 ──
    print("\n[Step 3] File Search 어시스턴트 생성 중...")
    roleplay_assistant = create_file_search_assistant(
        name="롤 플레잉 봇",
        instructions="""당신은 문학 속의 인물이 되어, 사용자와 롤 플레이를 해야 합니다.
주어진 파일의 문체와 스타일을 참고하여, 실감나고 사실적으로 답변하세요.
인물의 극중 말투와 최대한 유사하게 답변하세요."""
    )

    # ── Step 4: 어시스턴트에 Vector Store 연결 ──
    print("\n[Step 4] 어시스턴트에 Vector Store 연결 중...")
    roleplay_assistant = attach_vector_store(roleplay_assistant, hamlet_store)

    # ── Step 5: 첫 번째 질문 — Thread 생성 및 Run 실행 ──
    print("\n[Step 5] 첫 번째 질문 — Thread 생성 및 Run 실행")
    question_1 = "햄릿, 당신은 어떻게 죽었나요?"
    print(f"  질문: {question_1}")

    roleplay_thread = create_thread(question_1)
    roleplay_run = create_run(roleplay_thread, roleplay_assistant)

    print("\n  Run 상태 확인 중...")
    wait_for_run(roleplay_thread, roleplay_run)

    # ── Step 6: 출처 인용(【source】) 포함 답변 확인 ──
    # File Search는 RAG 방식으로 동작합니다:
    #   1) 질문을 벡터로 변환
    #   2) Vector Store에서 유사한 청크 K개 검색
    #   3) 검색된 내용을 컨텍스트에 포함해 답변 생성
    #   4) 출처는 【N:M†source】 형식으로 인용 표기됨
    print("\n[Step 6] 답변 및 출처 인용 확인")
    print("-" * 55)
    list_threads_messages(roleplay_thread)

    # ── 리소스 정리 ──
    client.beta.assistants.delete(roleplay_assistant.id)
    client.vector_stores.delete(hamlet_store.id)
    print("\n🗑️  어시스턴트 및 Vector Store 삭제 완료")
    print("✅ Topic 3 실행 완료!")