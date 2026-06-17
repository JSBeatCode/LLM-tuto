"""
[Topic 4] 멀티모달 - LangChain과 멀티모달 모델을 활용한 스마트 냉장고 앱
============================================================
작동 순서:
  1. 냉장고 사진 업로드
  2. gpt-4o Vision → 재료 목록 추출
  3. gpt-4o → 만들 수 있는 음식 추천
  4. DALL-E-3 → 추천 음식 이미지 생성
  5. Gradio UI에서 결과를 단계별로 표시 (비동기 스트리밍)

실행 방법:
  1. 같은 폴더에 .env 파일 생성 후 아래 내용 입력:
       OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
  2. 필요 패키지 설치:
       pip install langchain langchain-openai langchain-community gradio python-dotenv pillow requests
  3. 스크립트 실행:
       python topic4_multimodal_smart_fridge.py
"""

# ── 0. 환경 설정 ─────────────────────────────────────────────
import os  # 운영체제 관련 기능을 사용하기 위한 모듈. 여기서는 환경변수에서 API 키를 읽을 때 사용
import base64  # 이미지 파일을 base64 문자열로 변환할 때 사용하는 모듈
import requests  # URL로부터 이미지나 데이터를 다운로드할 때 사용하는 라이브러리

from io import BytesIO  # 메모리 안에서 파일처럼 데이터를 다루기 위한 도구
from dotenv import load_dotenv  # .env 파일에 저장된 환경변수를 Python으로 불러오기 위한 함수
load_dotenv(override=True)
import openai  # OpenAI API를 직접 호출하기 위한 공식 라이브러리
from PIL import Image  # 이미지를 열고 저장하고 변환하기 위한 Pillow 라이브러리

from langchain_openai import ChatOpenAI  # LangChain에서 OpenAI 채팅 모델을 사용하기 위한 클래스
from langchain_core.prompts import ChatPromptTemplate  # 프롬프트 템플릿을 만들기 위한 클래스
from langchain_core.output_parsers import StrOutputParser  # AI 응답을 문자열 형태로 변환하기 위한 파서

import gradio as gr  # 웹 화면 UI를 쉽게 만들기 위한 Gradio 라이브러리

# ── 모델 상수 ────────────────────────────────────────────────
VISION_MODEL   = "gpt-4o"       # Vision + 텍스트 생성
IMAGE_MODEL    = "dall-e-3"     # 이미지 생성
MAX_TOKENS     = 1024


# ── 1. OpenAI 클라이언트 & LangChain LLM 초기화 ──────────────
def init_clients():
    """API 키를 검증하고 클라이언트/LLM 객체를 반환합니다."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ .env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")

    client = openai.OpenAI(api_key=api_key)

    try:
        client.models.list()
        print("✅ OPENAI_API_KEY가 정상적으로 설정되어 있습니다.")
    except Exception as e:
        raise ValueError(f"❌ API 키가 유효하지 않습니다: {e}")

    llm = ChatOpenAI(model=VISION_MODEL, max_tokens=MAX_TOKENS)
    return client, llm


# ── 2. LangChain 체인 구성 ────────────────────────────────────
def build_chains(llm):
    """재료 추출 체인과 메뉴 추천 체인을 생성하여 반환합니다."""

    # 체인 1: 냉장고 이미지 → 재료 목록
    listing_prompt = ChatPromptTemplate([
        ("system", """음식 재료에 대한 이미지가 주어집니다.
해당 이미지에서 확인할 수 있는 모든 음식 재료의 목록을
리스트로 출력하세요. 답변은 영어로 작성하세요."""),
        ("user", [{
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,{image}"},
        }])
    ])
    list_chain = listing_prompt | llm | StrOutputParser()

    # 체인 2: 재료 목록 → 음식 추천
    recommendation_prompt = ChatPromptTemplate([
        ("system", """음식 재료 리스트가 아래에 주어집니다:
해당 재료를 이용해 만들 수 있는 일반적인 음식을 구상하세요.
음식을 간단한 설명으로 묘사하세요.
음식과 묘사 이외에 다른 설명은 추가하지 마세요."""),
        ("user", "{food}")
    ])
    recommendation_chain = recommendation_prompt | llm | StrOutputParser()

    return list_chain, recommendation_chain


# ── 3. 이미지 처리 유틸리티 ───────────────────────────────────
# (image: Image.Image) => 변수명: 타입. image → PIL Image 타입이다
# str: => 이 함수는 문자열(str)을 반환한다
def encode_image_pil(image: Image.Image) -> str:
    """PIL Image를 base64 문자열로 변환하는 함수"""

    buffered = BytesIO()  # 메모리 상에 임시 파일 공간 생성 (파일처럼 사용)

    image.save(buffered, format="JPEG")  
    # PIL 이미지를 JPEG 형식으로 메모리에 저장

    return base64.b64encode(buffered.getvalue()).decode("utf-8")  
    # 메모리에 저장된 이미지 데이터를 꺼내서
    # base64로 인코딩한 후 문자열로 변환해서 반환


def draw_image(client: openai.OpenAI, prompt: str) -> Image.Image:
    """DALL-E-3로 음식 이미지를 생성하고 PIL Image를 반환합니다."""
    response = client.images.generate(
        model=IMAGE_MODEL,
        prompt=f"A nice dinner with {prompt}",
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    img_response = requests.get(image_url)
    return Image.open(BytesIO(img_response.content))


# ── 4. Gradio 비동기 처리 함수 (단계별 yield) ─────────────────
def make_process_fn(client, list_chain, recommendation_chain):
    """
    Gradio 이벤트 핸들러를 반환합니다.
    yield를 사용해 각 단계 완료 시 UI를 점진적으로 업데이트합니다.
    """
    def process(image: Image.Image):
        """이미지를 받아서 재료 추출 → 메뉴 추천 → 이미지 생성까지 수행"""
        # 🔹 이미지가 없을 경우 처리
        if image is None:
            yield "이미지를 먼저 업로드해 주세요.", None, None
            return  # 함수 종료

        # 🔹 Step 1: 재료 분석 시작 메시지 출력
        yield "⏳ 재료를 분석 중입니다...", None, None

        # 🔹 이미지 → base64 변환
        image_encoded = encode_image_pil(image)

        # 🔹 AI에게 이미지 전달 → 재료 목록 추출
        ingredients = list_chain.invoke({"image": image_encoded})

        # 🔹 Step 2: 재료 결과 + 메뉴 추천 진행 중 표시
        yield ingredients, "⏳ 메뉴를 추천 중입니다...", None

        # 🔹 AI에게 재료 전달 → 메뉴 추천
        menu = recommendation_chain.invoke({"food": ingredients})

        # 🔹 Step 3: 재료 + 메뉴 출력
        yield ingredients, menu, None

        # 🔹 Step 4: 이미지 생성 중 표시
        yield ingredients, menu + "\n\n⏳ 음식 이미지를 생성 중입니다...", None

        # 🔹 메뉴 → 이미지 생성
        menu_image = draw_image(client, menu)

        # 🔹 Step 5: 최종 결과 출력 (이미지 포함)
        yield ingredients, menu, menu_image

    return process


# ── 5. Gradio UI 구성 ─────────────────────────────────────────
# -> gr.Blocks => "이 함수는 gr.Blocks 타입을 반환한다"
def build_ui(process_fn) -> gr.Blocks:
    """Gradio를 이용해 웹 UI를 구성하는 함수"""

    # Gradio 화면 전체 컨테이너 생성
    # as demo => 이 화면을 demo 라는 변수에 담는다.
    # with => 이 코드 아래에 코딩하는 요소들을 전부 demo에 넣어라
    with gr.Blocks(title="🧊 스마트 냉장고") as demo:

        # 🔹 앱 제목 및 설명
        gr.Markdown("""
# 🧊 스마트 냉장고
냉장고 사진을 업로드하면 AI가 재료를 인식하고,
음식을 추천하며, 이미지를 생성합니다.
        """)

        # 🔹 입력 영역 (이미지 업로드 + 버튼)
        with gr.Row():  # 가로 정렬
            # scale=1 => "이 영역이 차지하는 크기 비율"
            # 예시: 
            # with gr.Row():
            #     with gr.Column(scale=1):
            #         A
            #     with gr.Column(scale=2):
            #         B
            #   결과: [A] [   B   ]
            with gr.Column(scale=1):  # 세로 정렬
                # gr.Image => 사용자가 이미지를 업로드할 수 있는 입력창을 만든다
                image_input = gr.Image(
                    type="pil",  # PIL Image 형태로 받기
                    label="📷 냉장고 이미지 업로드"
                )

                submit_btn = gr.Button(
                    "menu recommendation", # 버튼 텍스트
                    variant="primary" # 버튼 스타일
                )

        # 구분선
        gr.Markdown("---")

        # 🔹 출력 영역 (3개 결과)
        with gr.Row():
            # 재료 목록 출력창
            ingredients_output = gr.Textbox(
                label="🥕 인식된 재료 목록",
                lines=6,
                placeholder="재료 목록이 여기에 표시됩니다."
            )

            # 추천 메뉴 출력창
            menu_output = gr.Textbox(
                label="📋 추천 메뉴",
                lines=6,
                placeholder="추천 음식이 여기에 표시됩니다."
            )

            # 생성된 이미지 출력창
            image_output = gr.Image(
                type="pil",
                label="🖼️ 생성된 음식 이미지"
            )

        # 🔹 버튼 클릭 이벤트 연결 (핵심)
        submit_btn.click(
            fn=process_fn, # 버튼 → process_fn 실행
            inputs=image_input, # 이 값을 함수에 넣어라
            # 함수 결과를 어디에 보여줄지
            # 1번째 결과 → ingredients_output
            # 2번째 결과 → menu_output
            # 3번째 결과 → image_output
            outputs=[ingredients_output, menu_output, image_output] 
        )

    return demo  # 완성된 UI 반환


# ── 6. 메인 진입점 ────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  🧊 스마트 냉장고 앱 시작")
    print("=" * 50)

    # 클라이언트 초기화
    client, llm = init_clients()

    # LangChain 체인 구성
    list_chain, recommendation_chain = build_chains(llm)
    print("✅ LangChain 체인 구성 완료")

    # Gradio 프로세스 함수 생성
    process_fn = make_process_fn(client, list_chain, recommendation_chain)

    # UI 빌드 및 실행
    demo = build_ui(process_fn)
    print("✅ Gradio UI 구성 완료")
    print("🚀 앱을 실행합니다. 브라우저에서 http://localhost:7860 을 열어주세요.\n")
    demo.launch()


if __name__ == "__main__":
    main()