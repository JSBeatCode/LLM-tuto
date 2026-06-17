"""
주제 4 — 멀티모달 (이미지 · 음성)
===================================
학습 내용:
  1. DALL-E 3 — 텍스트 프롬프트로 이미지 생성
  2. DALL-E 3 — revised_prompt 확인 (모델이 수정한 프롬프트)
  3. Vision   — URL 이미지를 GPT-4o 에 전달하여 분석
  4. Vision   — 로컬 이미지를 base64 로 인코딩하여 전달
  5. TTS      — 텍스트를 음성(MP3)으로 변환 저장
  6. Whisper  — 음성 파일을 텍스트로 전사 (STT)
  7. Whisper  — prompt 파라미터로 고유명사 인식률 보정
  8. gpt-4o-audio-preview — 텍스트 입력 → 텍스트+음성 동시 출력
  9. gpt-4o-audio-preview — 음성 입력 → 텍스트 응답

참고: 일부 step 은 생성된 파일(이미지·음성)을 다음 step 에서 재사용합니다.
      순서대로 실행하는 것을 권장합니다.
"""

import os
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv
import openai


# ── 환경변수 로드 ─────────────────────────────────────────
load_dotenv(override=True)

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── 출력 파일 경로 상수 ───────────────────────────────────
IMAGE_PATH   = Path("generated_image.png")   # DALL-E 로 생성한 이미지
TTS_PATH     = Path("tts_output.mp3")        # TTS 로 생성한 음성
AUDIO_PATH   = Path("audio_response.mp3")    # gpt-4o-audio-preview 음성 응답


# ══════════════════════════════════════════════════════════
def step1_dalle_generate() -> str:
    """
    [STEP 1] DALL-E 3 — 이미지 생성
    ---------------------------------
    텍스트 프롬프트를 입력하면 이미지 URL 을 반환합니다.
    - model   : "dall-e-3" (고화질) / "dall-e-2" (저비용)
    - size    : 1024x1024 / 1792x1024 / 1024x1792
    - quality : "standard" / "hd" (2배 비용)
    - n       : dall-e-3 은 1개만 지원

    이미지 URL 은 약 1시간 후 만료되므로 파일로 저장합니다.
    """
    print("=" * 55)
    print("[STEP 1] DALL-E 3 — 이미지 생성")
    print("=" * 55)

    prompt = (
        "A small house cat with a tabby pattern, sitting on a wooden table "
        "and yawning while looking into a mirror. "
        "In the mirror's reflection, a strong and regal lion is shown roaring dramatically. "
        "The setting is a minimalist modern room with neutral tones and soft natural lighting."
    )

    print(f"  프롬프트 : {prompt[:80]}...")
    print()

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    print(f"  생성된 이미지 URL : {image_url[:80]}...")
    print()

    # URL → 로컬 파일 저장
    img_data = requests.get(image_url).content
    IMAGE_PATH.write_bytes(img_data)
    print(f"  이미지 저장 완료 : {IMAGE_PATH}")
    print()

    return image_url


# ══════════════════════════════════════════════════════════
def step2_dalle_revised_prompt(image_url: str) -> None:
    """
    [STEP 2] DALL-E 3 — revised_prompt 확인
    -----------------------------------------
    DALL-E 3 는 안전 정책 및 품질 향상을 위해
    입력 프롬프트를 내부적으로 수정(revised)할 수 있습니다.
    response.data[0].revised_prompt 로 실제 사용된 프롬프트를 확인합니다.

    또한 반대 방향의 프롬프트(사자 → 거울에 고양이)를 생성하여
    창의적·비일상적 프롬프트에서의 환각 현상도 관찰합니다.
    """
    print("=" * 55)
    print("[STEP 2] DALL-E 3 — revised_prompt 확인")
    print("=" * 55)

    hard_prompt = (
        "A fierce lion standing tall in front of a modern, full-length mirror "
        "in the middle of a lush jungle. "
        "In the mirror's reflection, a tiny domestic kitten is seen sitting innocently, "
        "its big eyes and relaxed posture creating a humorous and surreal contrast."
    )

    response = client.images.generate(
        model="dall-e-3",
        prompt=hard_prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    print(f"  [원본 프롬프트]")
    print(f"  {hard_prompt[:120]}...")
    print()
    print(f"  [revised_prompt (DALL-E 가 실제 사용한 프롬프트)]")
    print(f"  {response.data[0].revised_prompt}")
    print()
    print("  ※ 흔하지 않은 장면일수록 revised_prompt 수정폭이 커집니다.")
    print()


# ══════════════════════════════════════════════════════════
def step3_vision_url(image_url: str) -> None:
    """
    [STEP 3] Vision — URL 이미지 전달 및 분석
    ------------------------------------------
    content 필드에 {"type": "image_url", "image_url": {"url": ...}} 를
    추가하면 GPT-4o 가 이미지를 직접 인식합니다.

    이미지 URL 이 공개 접근 가능한 경우 이 방식을 사용합니다.
    """
    print("=" * 55)
    print("[STEP 3] Vision — URL 이미지 분석")
    print("=" * 55)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "이 그림을 묘사하고, 일반적인 그림과 비교해서 특이한 점을 언급하세요.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=400,
    )

    print(f"  [GPT-4o 이미지 분석 결과]")
    print(f"  {response.choices[0].message.content.strip()}")
    print()


# ══════════════════════════════════════════════════════════
def step4_vision_base64() -> None:
    """
    [STEP 4] Vision — 로컬 이미지 base64 인코딩 전달
    --------------------------------------------------
    로컬에 저장된 이미지 파일을 base64 로 인코딩하여
    data URI 형식으로 전달합니다.
    URL 이 만료되었거나 비공개 이미지일 때 이 방식을 사용합니다.

    data URI 형식:
      "data:{media_type};base64,{base64_string}"
    """
    print("=" * 55)
    print("[STEP 4] Vision — base64 이미지 분석")
    print("=" * 55)

    if not IMAGE_PATH.exists():
        print(f"  ⚠️  {IMAGE_PATH} 파일이 없습니다. STEP 1 을 먼저 실행하세요.")
        print()
        return

    # 이미지 → base64 인코딩
    base64_image = base64.b64encode(IMAGE_PATH.read_bytes()).decode("utf-8")
    print(f"  이미지 파일  : {IMAGE_PATH} ({IMAGE_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"  base64 길이  : {len(base64_image)} 문자")
    print()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "이 그림을 전시회에서 소개한다고 생각하고, 즐겁고 유쾌하게, 유머를 섞어 홍보하세요.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=512,
    )

    print(f"  [GPT-4o 유쾌한 전시 홍보 문구]")
    print(f"  {response.choices[0].message.content.strip()}")
    print()


# ══════════════════════════════════════════════════════════
def step5_tts() -> None:
    """
    [STEP 5] TTS — 텍스트를 음성(MP3) 파일로 변환
    -----------------------------------------------
    - model  : "tts-1" (빠름) / "tts-1-hd" (고품질, 2배 비용)
    - voice  : alloy / echo / fable / onyx / nova / shimmer
    - input  : 변환할 텍스트 (최대 4096 문자)

    생성된 MP3 파일을 로컬에 저장합니다.
    """
    print("=" * 55)
    print("[STEP 5] TTS — 텍스트 음성 변환")
    print("=" * 55)

    voices = ["alloy", "nova", "shimmer"]
    text = (
        "LLM은 Large Language Model의 약자입니다. "
        "대용량의 코퍼스를 학습시킨 머신러닝 모델로, "
        "Qwen QWQ, Llama 3.3 70B 모델이 최근 출시되었습니다."
    )

    print(f"  변환 텍스트 : {text}")
    print()

    for voice in voices:
        out_path = Path(f"tts_{voice}.mp3")
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
        )
        # 음성 파일 저장
        with out_path.open("wb") as f:
            f.write(response.content)
        print(f"  [{voice}] 저장 완료 → {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # 이후 step 에서 재사용할 파일을 nova 로 고정
    Path("tts_nova.mp3").rename(TTS_PATH) if Path("tts_nova.mp3").exists() else None
    print()
    print(f"  ※ {TTS_PATH} 파일을 이후 STEP(Whisper)에서 사용합니다.")
    print()


# ══════════════════════════════════════════════════════════
def step6_whisper_basic() -> None:
    """
    [STEP 6] Whisper — 음성 파일 텍스트 전사 (기본)
    -------------------------------------------------
    오디오 파일을 업로드하면 텍스트로 변환합니다.
    - model   : "whisper-1" (현재 유일한 Whisper API 모델)
    - file    : mp3 / mp4 / wav / m4a 등 지원
    - language: 명시하면 인식 정확도 향상 ("ko", "en" 등)
    """
    print("=" * 55)
    print("[STEP 6] Whisper — 기본 음성 전사 (STT)")
    print("=" * 55)

    if not TTS_PATH.exists():
        print(f"  ⚠️  {TTS_PATH} 파일이 없습니다. STEP 5 를 먼저 실행하세요.")
        print()
        return

    with TTS_PATH.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ko",   # 한국어 명시로 인식 정확도 향상
        )

    print(f"  오디오 파일  : {TTS_PATH}")
    print(f"  전사 결과    : {transcript.text}")
    print()


# ══════════════════════════════════════════════════════════
def step7_whisper_with_prompt() -> None:
    """
    [STEP 7] Whisper — prompt 파라미터로 고유명사 보정
    ---------------------------------------------------
    Whisper 는 발음이 유사한 고유명사(모델명, 브랜드 등)를
    잘못 인식하는 경우가 있습니다.

    prompt 파라미터에 관련 단어를 힌트로 제공하면
    해당 단어들이 올바르게 인식될 확률이 높아집니다.

    예시: "GWEN-QWQ LAMA" → "Qwen QWQ, Llama"
    """
    print("=" * 55)
    print("[STEP 7] Whisper — prompt 파라미터로 고유명사 보정")
    print("=" * 55)

    if not TTS_PATH.exists():
        print(f"  ⚠️  {TTS_PATH} 파일이 없습니다. STEP 5 를 먼저 실행하세요.")
        print()
        return

    # prompt 없이 호출
    with TTS_PATH.open("rb") as f:
        result_plain = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="ko",
        )

    # prompt 에 고유명사 힌트 제공
    with TTS_PATH.open("rb") as f:
        result_prompted = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="ko",
            prompt="Qwen QWQ, Llama 3.3 70B",   # 인식 보정 힌트
        )

    print(f"  [prompt 없음]  : {result_plain.text}")
    print(f"  [prompt 있음]  : {result_prompted.text}")
    print()
    print("  → 고유명사 및 브랜드명이 포함된 음성은 prompt 힌트가 효과적입니다.")
    print()


# ══════════════════════════════════════════════════════════
def step8_audio_preview_output() -> None:
    """
    [STEP 8] gpt-4o-audio-preview — 텍스트 입력 → 텍스트 + 음성 동시 출력
    -----------------------------------------------------------------------
    modalities 에 "audio" 를 포함하면 텍스트 응답과 함께
    음성(base64 MP3)도 함께 반환됩니다.

    audio 파라미터:
      - voice  : alloy / ash / ballad / coral / echo / fable / onyx / nova / shimmer / verse
      - format : mp3 / opus / aac / flac

    ※ gpt-4o-audio-preview 모델 접근 권한이 필요합니다.
      권한이 없을 경우 이 step 은 건너뜁니다.
    """
    print("=" * 55)
    print("[STEP 8] gpt-4o-audio-preview — 음성 출력")
    print("=" * 55)

    messages = [
        {
            "role": "system",
            "content": "당신은 전혀 공감하지 않으며, 상대를 불쾌하게 합니다. 답변은 짜증스러운 반말로 하세요.",
        },
        {"role": "user", "content": "오늘 회사 가기 싫어."},
    ]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "ash", "format": "mp3"},
            messages=messages,
            temperature=0.3,
            max_tokens=200,
        )

        # 텍스트 응답 출력
        text_response = completion.choices[0].message.content
        print(f"  [텍스트 응답] : {text_response}")
        print()

        # 음성 데이터 저장
        audio_data = base64.b64decode(completion.choices[0].message.audio.data)
        AUDIO_PATH.write_bytes(audio_data)
        print(f"  [음성 파일 저장] : {AUDIO_PATH} ({AUDIO_PATH.stat().st_size / 1024:.1f} KB)")
        print()

    except openai.NotFoundError:
        print("  ⚠️  gpt-4o-audio-preview 모델 접근 권한이 없습니다. STEP 8 을 건너뜁니다.")
        print()


# ══════════════════════════════════════════════════════════
def step9_audio_preview_input() -> None:
    """
    [STEP 9] gpt-4o-audio-preview — 음성 입력 → 텍스트 응답
    --------------------------------------------------------
    content 에 {"type": "input_audio", "input_audio": {"data": ..., "format": ...}} 를
    추가하면 음성 파일을 직접 모델에 입력할 수 있습니다.

    Whisper 와의 차이:
      - Whisper     : 음성 → 텍스트 전사만 수행
      - audio-preview : 음성 이해 + 대화 응답까지 한 번에 처리

    ※ STEP 8 에서 생성된 음성 파일을 입력으로 사용합니다.
    ※ gpt-4o-audio-preview 모델 접근 권한이 필요합니다.
    """
    print("=" * 55)
    print("[STEP 9] gpt-4o-audio-preview — 음성 입력")
    print("=" * 55)

    if not AUDIO_PATH.exists():
        print(f"  ⚠️  {AUDIO_PATH} 파일이 없습니다. STEP 8 을 먼저 실행하세요.")
        print()
        return

    encoded_audio = base64.b64encode(AUDIO_PATH.read_bytes()).decode("utf-8")

    messages = [
        {"role": "system",    "content": "당신은 회사에 가기 싫은 직장인입니다."},
        {"role": "assistant", "content": "오늘 회사 가기 싫어."},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_audio,
                        "format": "mp3",
                    },
                }
            ],
        },
    ]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text"],
            messages=messages,
            max_tokens=200,
        )

        print(f"  [음성 입력에 대한 텍스트 응답]")
        print(f"  {completion.choices[0].message.content.strip()}")
        print()

    except openai.NotFoundError:
        print("  ⚠️  gpt-4o-audio-preview 모델 접근 권한이 없습니다. STEP 9 을 건너뜁니다.")
        print()


# ══════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "★" * 55)
    print("   주제 4 — 멀티모달 (이미지 · 음성)")
    print("★" * 55 + "\n")

    # ── 이미지 관련 ──────────────────────────────
    image_url = step1_dalle_generate()
    step2_dalle_revised_prompt(image_url)
    step3_vision_url(image_url)
    step4_vision_base64()

    # ── 음성 관련 ────────────────────────────────
    step5_tts()
    step6_whisper_basic()
    step7_whisper_with_prompt()

    # ── 오디오 미리보기 (권한 필요) ──────────────
    step8_audio_preview_output()
    step9_audio_preview_input()

    print("=" * 55)
    print("주제 4 실습 완료 ✅")
    print()
    print("  생성된 파일 목록:")
    for f in [IMAGE_PATH, TTS_PATH, AUDIO_PATH,
              Path("tts_alloy.mp3"), Path("tts_shimmer.mp3")]:
        if f.exists():
            print(f"  - {f} ({f.stat().st_size / 1024:.1f} KB)")
    print("=" * 55)


if __name__ == "__main__":
    main()