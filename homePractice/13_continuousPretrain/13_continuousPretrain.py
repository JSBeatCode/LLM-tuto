"""
[실습 13] Continuous Pretraining을 이용한 도메인 지식 주입하기

Continuous Pretraining은 이미 Pretrain된 모델(sLLM)에 새로운 도메인 코퍼스를
추가 학습시켜 도메인 지식을 주입하는 기법입니다.
이 스크립트는 의료 도메인 코퍼스(medical_corpus.json)를 이용해
gemma-3-1b-pt 모델에 의료 지식을 주입합니다.

실행 환경: CUDA GPU 필요 (Python 3.11)
GPU 호환성:
    - GTX 1650 등 Turing 이하: fp16 모드로 자동 전환, optim=adamw_torch 사용
    - RTX 30xx (Ampere) 이상: bf16 + paged_adamw_8bit 사용 가능
    - bitsandbytes Windows 미지원으로 adamw_torch 옵티마이저를 기본으로 사용

실행 환경 (RTX 4000 Ada, 20GB VRAM, Linux):
    - Ada Lovelace (sm_89): bf16 지원
    - Linux 환경: bitsandbytes 지원 → paged_adamw_8bit 사용
    - VRAM 20GB: max_seq_length=2500, batch_size=2 복원

사용 전 준비물:
    1. .env 파일 생성 (.env.example 참고) 후 HF_TOKEN 입력 (필요한 경우)
    2. medical_corpus.json 파일을 이 스크립트와 동일한 폴더에 위치
"""

# -----------------------------------------------------------------------
# 표준 라이브러리
# -----------------------------------------------------------------------
import os
import sys

# -----------------------------------------------------------------------
# 서드파티 라이브러리
# -----------------------------------------------------------------------
# | 종류                 | 라이브러리                         | 역할                  |
# | ------------------ | ----------------------------- | ------------------- |
# | 📊 데이터 분석          | numpy, matplotlib, seaborn    | 데이터 분석 및 그래프        |
# | 🔥 AI(GPU)         | torch                         | 딥러닝 및 GPU 사용        |
# | 📚 데이터셋            | datasets                      | 학습 데이터 읽기           |
# | 🔑 환경설정            | dotenv                        | .env 파일 읽기          |
# | 🤗 HuggingFace/LLM | transformers, huggingface_hub | LLM 로드 및 학습         |
# | 🔗 LangChain       | langchain_huggingface         | LLM을 LangChain에서 사용 |

# 그래프를 그리는 라이브러리
import matplotlib
# 숫자 계산을 아주 빠르게 해주는 라이브러리
import numpy as np
# 예쁜 그래프를 그리는 라이브러리
import seaborn as sns
# LLM 자체를 움직이는 핵심 엔진
import torch
# Hugging Face의 데이터셋 라이브러리입니다.
# JSON, CSV, Parquet, 등을 쉽게 읽습니다.
# medical_corpus.json를 학습 가능한 형태로 바꿔줍니다.
from datasets import load_dataset
from dotenv import load_dotenv
# Hugging Face 로그인용입니다
from huggingface_hub import login
# LangChain이 Hugging Face 모델을 사용할 수 있도록 연결해주는 어댑터(Adapter)
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    # Gemma, Llama, Qwen, Mistral, 모두 이걸로 불러올 수 있습니다.
    AutoModelForCausalLM,
    # 안녕하세요. -> [1542, 9134, 523] 이렇게 변환합니다. (LLM은 숫자만 이해)
    AutoTokenizer,
    # 학습 데이터를 모델이 먹을 수 있는 형태로
    DataCollatorForLanguageModeling,
    # 학습을 진행하는 클래스입니다. 직접 학습 코드를 작성하면 코드를 수백 줄 작성하기 때문에 사용한다.
    Trainer,
    # 학습 옵션 설정
    TrainingArguments,
    # LLM은 문자열 → Tokenizer  → Token(ID)  → Tensor  → Model  → Token(ID)  → 문자열
    # 이 과정을 모두 거쳐야 하지만 pipeline을 쓰면
    # → pipe("안녕하세요.") 이렇게 가능
    pipeline,
)

matplotlib.use("Agg")  # GUI 없는 환경에서도 저장 가능하도록 설정 (pyplot import 전에 설정 필요)
import matplotlib.pyplot as plt  # noqa: E402  (matplotlib.use 이후에 import 해야 하므로 순서 유지)

# -----------------------------------------------------------------------
# 0. 경로 및 환경 설정
# -----------------------------------------------------------------------
# 이 .py 파일이 위치한 디렉토리를 기준 경로로 사용 (모든 산출물은 이 위치에 생성)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# .env 파일 로드 (BASE_DIR 기준)
load_dotenv(os.path.join(BASE_DIR, ".env"))

# 데이터 / 출력 경로 (모두 BASE_DIR 하위에 위치하도록 고정)
CORPUS_PATH = os.path.join(BASE_DIR, "medical_corpus.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")  # 학습 체크포인트 / 텐서보드 로그
MODEL_ID = "unsloth/gemma-3-1b-pt"              # 베이스 모델 주소
MODEL_NAME = MODEL_ID.split("/")[1]
MODEL_SAVE_DIR = os.path.join(BASE_DIR, f"{MODEL_NAME}-MED")  # 학습 완료 모델 저장 경로

# VRAM 메모리 단편화 방지 (OOM 에러 완화)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # -------------------------------------------------------------------
    # 1. 사전 점검: 학습 데이터 존재 확인
    # -------------------------------------------------------------------
    if not os.path.exists(CORPUS_PATH):
        print(f"[오류] 학습 데이터 파일이 없습니다: {CORPUS_PATH}")
        print("medical_corpus.json 파일을 이 스크립트와 동일한 폴더에 위치시킨 후 다시 실행해 주세요.")
        sys.exit(1)

    # -------------------------------------------------------------------
    # 2. GPU 사용 가능 여부 확인 및 GPU 스펙에 맞는 학습 설정 결정
    #    - bf16: Ampere(RTX 30xx, sm_80) 이상에서만 지원
    #    - paged_adamw_8bit: bitsandbytes 필요 → Windows 미지원이므로 adamw_torch 사용
    # -------------------------------------------------------------------
    if not torch.cuda.is_available():
        print("[경고] CUDA GPU가 감지되지 않았습니다.")
        print("Continuous Pretraining은 CUDA GPU 환경이 필요합니다.")
        print("GPU가 있는 환경(Colab, 클라우드 GPU 서버 등)에서 실행해 주세요.")
        sys.exit(1)

    # @@ 내 컴퓨터의 GPU를 조사해서, 그 GPU에 맞는 최적의 학습 옵션을 자동으로 선택하는 코드
    
    # GPU 이름을 알려줘:
    #     만약 GPU가 여러 개라면
    #     GPU 0: GTX 1650
    #     GPU 1: GTX 3050
    #     GPU 2: GTX 4050
    #     이렇게 번호가 붙어서 '(0)' 첫번째 것을 가져옴
    gpu_name = torch.cuda.get_device_name(0)
    
    # GPU의 연산 능력(Compute Capability)을 알려줌.
    # | Compute Capability | GPU         |
    # | ------------------ | ----------- |
    # | 7.5                | GTX1650     |
    # | 8.0                | RTX30xx     |
    # | 8.6                | RTX3090     |
    # | 8.9                | RTX4090     |
    # | 9.x                | Blackwell 등 |
    # 번호가 높을수록 새로운 기능을 지원합니다.
    compute_capability = torch.cuda.get_device_capability(0)  # (major, minor) 튜플
    
    # 만약 GTX1650이면
    # (7,5)
    # 이므로
    # 7 >= 8
    # ↓
    # False
    # bf16은 Brain Floating Point 16이라는 숫자 표현 방식입니다.
    # 하지만 RTX30xx 이상에서만 제대로 지원됩니다.
    use_bf16 = compute_capability[0] >= 8  # Ampere(sm_80) 이상이면 bf16 지원
    
    # 둘 중 하나만 사용하겠다는 의미
    use_fp16 = not use_bf16                # Turing(sm_75) 이하면 fp16 사용

    # Ada Lovelace + Linux → bitsandbytes 지원 → paged_adamw_8bit 사용 가능
    # optimaizer:내 컴퓨터의 GPU를 조사해서, 그 GPU에 맞는 최적의 학습 옵션을 자동으로 선택하는 코드
    # | 항목         | adamw_torch | paged_adamw_8bit |
    # | ---------- | ----------- | ---------------- |
    # | 구현         | PyTorch 기본  | bitsandbytes     |
    # | 메모리 사용     | 많음          | 매우 적음            |
    # | 속도         | 보통          | 빠른 경우가 많음        |
    # | VRAM 절약    | ❌           | ✅                |
    # | 대형 LLM 학습  | △           | ✅                |
    # | Windows 지원 | ✅           | 제한적(환경에 따라 다름)   |
    optimizer = "paged_adamw_8bit" if use_bf16 else "adamw_torch"

    print(f"[INFO] GPU: {gpu_name} (Compute Capability: {compute_capability[0]}.{compute_capability[1]})")
    if use_bf16:
        print("[INFO] 정밀도: bf16 (Ampere 이상 지원)")
    else:
        print("[INFO] 정밀도: fp16 (Turing 이하, bf16 미지원 → fp16으로 대체)")
    print(f"[INFO] 옵티마이저: {optimizer}")

    # -------------------------------------------------------------------
    # 3. (선택) HuggingFace 로그인
    #    - HF_TOKEN이 .env에 설정되어 있고 placeholder가 아닌 경우에만 로그인 시도
    # -------------------------------------------------------------------
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and hf_token != "your_huggingface_token_here":
        login(token=hf_token)
        print("[INFO] HuggingFace 로그인 완료")
    else:
        print("[INFO] HF_TOKEN 미설정 - 로그인 생략 (Gemma는 보통 로그인 불필요)")

    # -------------------------------------------------------------------
    # 4. 모델 및 토크나이저 로드
    #    - cache_dir을 .py 위치의 hf_cache 폴더로 지정
    #    - 재실행 시 hf_cache 폴더가 존재하면 다운로드 생략 (중복 다운로드 방지)
    #    - 전역 캐시(~/.cache/huggingface)는 사용하지 않음
                                                # -------------------------------------------------------------------
    print(f"## MODEL: {MODEL_NAME}")

    # @@ GPU가 행렬(Matrix) 계산을 더 빠르고 효율적으로 하도록 설정하는 코드
    # "highest" : 가장 정확하지만 느릴 수 있음
    # "high" : 정확도는 거의 유지하면서 더 빠름 ✅ (가장 많이 사용)
    # "medium" : 더 빠르지만 정확도가 조금 더 낮을 수 있음
    # 왜 사용하는가? -> 학습 속도를 높이기 위해서입니다.
    torch.set_float32_matmul_precision("high")

    # @@ 모델/토크나이저 캐시를 .py 파일과 같은 위치의 hf_cache 폴더에 저장
    HF_CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    # "안녕하세요"
    # ↓
    # Tokenizer
    # ↓
    # [912, 4312, 77]
    # 이처럼 문자를 숫자(Token ID)로 변환해야 모델이 이해해기 때문에 tokenizer를 쓰고,
    # 이 코드는,
    # Gemma  → Gemma Tokenizer
    # Llama  → Llama Tokenizer
    # Qwen   → Qwen Tokenizer
    # 처럼 자동으로 맞는 Tokenizer를 가져옵니다.
    # from_pretrained() -> 이미 학습되어 공개된 모델의 Tokenizer를 불러오는 함수
    print("[INFO] 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_DIR)

    print("[INFO] 모델 로드 중... (최초 1회만 다운로드 → hf_cache 폴더에 저장, 이후 재사용)")
    # @@ 자동으로 Gemma 모델을 다운로드(또는 캐시에서 읽어서) GPU에 로드합니다.
    # 사전학습된 Gemma 모델을 불러오고, GPU/CPU와 데이터 타입을 자동 설정한다.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",          # GPU에 맞는 데이터 타입 자동 선택
        device_map="auto",           # GPU/CPU 자동 선택
        attn_implementation="eager", # 기본 Attention 방식 사용
        cache_dir=HF_CACHE_DIR,      # 모델 캐시 저장 위치
    )

    # -------------------------------------------------------------------
    # 5. 학습 전 베이스 모델 성능 확인
    # -------------------------------------------------------------------
    # LLM이 답변을 어떤 방식으로 생성할지 설정합니다.
    
    # dict() -> 즉 아래 코드와 같습니다.
    # gen_config = {
    #     "do_sample": True,
    #     "max_new_tokens": 512,
    #     ...
    # }
    gen_config = dict(
        do_sample=True,           # 랜덤성을 적용하여 자연스러운 답변 생성
        max_new_tokens=512,       # 최대 512토큰까지 생성
        temperature=0.7,          # 답변의 창의성 조절
        top_p=0.95,               # 확률이 높은 후보만 선택
        top_k=64,                 # 상위 64개 후보 중에서 선택
        repetition_penalty=1.05,  # 같은 내용 반복 억제
    )

    # @@ LLM을 실제로 사용할 수 있는 형태로 만드는 과정
    # 앞에서 만든
    #   model
    #   tokenizer
    # 를 합쳐서
    #   pipe("안녕하세요")
    # 처럼 쉽게 사용할 수 있게 됩니다.
    pipe = pipeline(
        # 텍스트 생성용이라는 뜻
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # 모델의 출력에 입력한 질문도 함께 포함해서 반환합니다.
        return_full_text=True,
        # 앞에서 만든 gen_config를 그대로 전달. 즉, 아래와 같음
        # pipeline(
            # ...
            # do_sample=True,
            # max_new_tokens=512,
            # temperature=0.7,
            # top_p=0.95,
            # top_k=64,
            # repetition_penalty=1.05,
        # )
        **gen_config,
    )
    
    # @@ Hugging Face Pipeline을 LangChain에서 사용할 수 있는 형태로 변환합니다.
    # 이 과정을 거쳐야, base_llm.stream(...) 처럼 사용할 수 있고 LangChain 기능을 사용할 수 있습니다.
    base_llm = HuggingFacePipeline(pipeline=pipe)

    # 학습하기 전에 "현재 모델의 실력"을 테스트
    # 테스트 할 질문들
    eval_inputs = [
        "질문: 고혈압의 약물 치료는 어떻게 해야 하나요? \n답변:",
        "질문: 축농증의 수술이 필요한 경우는 언제인가요? \n답변:",
        "질문: 불안장애 환자의 식이의 특징은 무엇입니까? \n답변:",
        "질문: 갑상선 기능 항진증 치료에 사용하는 대표적인 약물들은? \n답변:",
    ]

    print("\n========== [학습 전] 베이스 모델 출력 ==========")
    for q in eval_inputs:
        # end=""는 출력 후 줄바꿈을 하지 말라는 뜻
        print(f"{q}", end="")
        # 모델가 답변을 조금씩 생성할 때마다 하나씩 받아옵니다.
        # 예를 들어 모델이
        # 고혈압 치료는 생활습관 개선과...
        # 을 생성하면 내부적으로는
        # "고혈압"
        # ↓
        # " 치료는"
        # ↓
        # " 생활습관"
        # ↓
        # " 개선과..."
        # 처럼 조각(chunk) 단위로 전달됩니다.
        # 여기서 s는 그 조각 하나입니다.
        # 답변이 실시간으로 타이핑되는 것처럼 보입니다.
        for s in base_llm.stream(q):
            print(s, end="")
        print("\n-------")

    # -------------------------------------------------------------------
    # 6. 학습 데이터 로드
    #    - Continuous Pretraining은 새로운 지식 습득이 목적이므로 train/test split 불필요
    # -------------------------------------------------------------------
    print("\n[INFO] 학습 데이터 로드 중...")
    # JSON 파일을 Hugging Face Dataset 객체로 변환합니다.
    # "json" -> 파일 형식이 JSON 이다,
    # data_files={"train": [CORPUS_PATH]} -> CORPUS_PATH에 있는 파일을 train 데이터셋으로 사용해.
    data = load_dataset("json", data_files={"train": [CORPUS_PATH]})
    # 데이터 순서를 랜덤하게 섞습니다.
    data = data.shuffle()
    print(data)
    # 첫 번째 데이터를 200글자만 출력합니다.
    print("샘플 예시:", data["train"][0]["generated_text"][:200], "...")

    # -------------------------------------------------------------------
    # 7. 학습 데이터 토큰 길이 분포 분석
    # ------------------------------------------------------------------
    # @@ 학습 데이터의 토큰 길이를 분석해서, 적절한 학습 설정(max_seq_length 등)을 결정하기 위한 함수
    # 각 문장이:
        # 평균 몇 토큰인지
        # 가장 긴 문장은 몇 토큰인지
        # 가장 짧은 문장은 몇 토큰인지
        # 토큰 길이가 어떻게 분포되어 있는지
    # 왜 필요할까?
        # LLM은 한 번에 처리할 수 있는 토큰 수가 제한되어 있습니다.
        # max_seq_length를 얼마로 설정해야 할지 판단하기 어렵습니다.
        # 그래서 먼저 데이터의 길이를 분석합니다.  
    # parameter
        # | Parameter     | 의미               | 기본값                |
        # | ------------- | ---------------- | ------------------ |
        # | `dataset`     | 분석할 데이터셋         | 없음(필수)             |
        # | `text_column` | 분석할 텍스트 컬럼 이름    | `"generated_text"` |
        # | `bins`        | 히스토그램 막대 개수      | `30`               |
        # | `tokenized`   | 이미 토큰화된 데이터인지 여부 | `False`            |
        # | `save_path`   | 그래프 저장 경로        | `None`             |        
    # bins
        # bins=10
        # █ █ █ █ █ █ █ █ █ █
        # bins=30
        # ▇▆▅▄▃▂▁...
        # 막대가 더 촘촘하게 그려집니다.
    # tokenized
        # False → 아직 문자열(Text) 상태
        # "고혈압 치료..."
        # True → 이미 토큰(ID) 상태
        # [1543, 231, 88, 912]
    def analyze_token_distribution(dataset, text_column="generated_text", bins=30, tokenized=False, save_path=None):
    
        # @@ 각 데이터가 몇 개의 토큰으로 이루어져 있는지 계산해서 token_counts 리스트에 저장
        # token_counts = [120, 98, 340, 156, 89]
        token_counts = []
        if not tokenized:
            for text in dataset[text_column]:
                # 토큰으로 변환합니다.
                    # 예를 들어
                    # "고혈압 치료"
                    # ↓
                    # [523, 91, 1402, 88]
                tokens = tokenizer.encode(text)
                token_counts.append(len(tokens))
        else:
            for tokens in dataset["input_ids"]:
                token_counts.append(len(tokens))

        # np -> numpy 라이브러리
        stats = {
            "평균 토큰 수": np.mean(token_counts),
            "중앙값": np.median(token_counts),
            "최소 토큰 수": min(token_counts),
            "최대 토큰 수": max(token_counts),
            # 토큰 길이가 얼마나 들쭉날쭉한지 나타냅니다.
            "표준편차": np.std(token_counts),
            "90퍼센타일": np.percentile(token_counts, 90),
            # 전체 데이터 중 95%가 이 값 이하가 되는 경계값을 구해줘
            # 토큰 개수가 아래처럼 10개의 문장에 있다고 가정해 봅시다.
                # 100
                # 150
                # 200
                # 250
                # 300
                # 350
                # 400
                # 500
                # 700
                # 1800
            # 95% 지점이 거의 마지막 값 근처이므로 결과가 약 1800 정도가 됩니다.
                # 100   ✅
                # 150   ✅
                # 200   ✅
                # 250   ✅
                # 300   ✅
                # 350   ✅
                # 400   ✅
                # 500   ✅
                # 700   ✅
                # 1800  ← 95% 지점 근처
            # 왜 이걸 구할까?
                # 예를 들어 max_seq_length를 정해야 합니다.
                # max_seq_length = LLM이 한 번에 읽을 수 있는 최대 토큰 수
                # 95% 정도의 데이터만 모두 포함하도록 max_seq_length=2048 같은 값을 선택.
                # -> 1800이 아니라 2048?
                    # 1024
                    # 2048
                    # 4096
                    # 8192
                    # 처럼 2의 거듭제곱 값을 많이 사용    
            "95퍼센타일": np.percentile(token_counts, 95),
            "99퍼센타일": np.percentile(token_counts, 99),
            "총 샘플 수": len(token_counts),
        }

        # @@ 그래프 생성        
        # 그래프를 그릴 도화지를 만듭니다. 가로, 세로 = 12, 6
        plt.figure(figsize=(12, 6))
        
        # 히스토그램 실제 그래프를 그립니다.
        # token_counts = 토큰 길이 분포
        # bins = 막대 몇 개로 나눌지
        # kde = 막대그래프 위에 부드러운 곡선도 함께 그립니다.
        sns.histplot(data=token_counts, bins=bins, kde=True)
        plt.title(f"Token Length Distribution for {text_column}")
        
        # X축
        plt.xlabel("Token Count")
        
        # Y축
        plt.ylabel("Frequency")
        
        # 여백 자동 조정
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"[INFO] 토큰 분포 그래프 저장: {save_path}")
        plt.close()

        # savefig() → 그래프를 파일로 저장
        # close() → 그래프를 메모리에서 제거
        # return stats → 계산 결과를 함수 밖으로 반환
        print("\n=== 토큰 수 통계 ===")
        for key, value in stats.items():
            print(f"{key}: {value:.1f}")

        return stats

    print("\n[INFO] 원본 데이터 토큰 분포 분석 중...")
    # @@ 원본 학습 데이터(train)의 토큰 길이를 분석하고, 그래프를 PNG 파일로 저장하는 함수 호출
    # 파라미터가 더 없는 이유: 필요 없는 값은 생략한 것입니다.
    analyze_token_distribution(
        # 앞에서 불러온 데이터셋 중 학습용(train) 데이터를 전달
        # 함수의 dataset 매개변수에 들어갑니다.
        data["train"],
        save_path=os.path.join(BASE_DIR, "token_distribution_original.png"),
    )

    # -------------------------------------------------------------------
    # 8. 오버랩 청킹 (긴 텍스트를 max_seq_length 단위로, 문맥 보존을 위해 오버랩 적용)
    # -------------------------------------------------------------------
    # RTX 4000 Ada (VRAM 20GB) 환경에 맞게 복원된 값
    # max_seq_length: 512 → 2500 (원본값 복원, VRAM 여유로 긴 문맥 학습 가능)
    # overlap_size:   128 → 800  (원본값 복원)
    max_seq_length = 2500
    text_field_name = "generated_text"
    overlap_size = 800

    # @@ 긴 문장을 일정한 길이로 잘라(Chunking), 앞뒤 문맥이 이어지도록 일부를 겹쳐서(Overlap) 나누는 함수
    # 예를 들어
    #     1 2 3 4 5 6 7 8 9 10
    #     max_seq_length = 5
    #     overlap = 2
    # 이면
    #     Chunk1
    #     1 2 3 4 5
    #     Chunk2
    #     4 5 6 7 8
    #     Chunk3
    #     7 8 9 10
    # 문맥(Context)이 끊기지 않습니다.
    # | Parameter        | 의미                      |
    # | ---------------- | ----------------------- |
    # | `examples`       | 잘라야 할 원본 데이터            |
    # | `tokenizer`      | 문장을 토큰으로 변환하는 Tokenizer |
    # | `max_seq_length` | 한 Chunk의 최대 토큰 길이       |
    # | `text_field`     | 텍스트가 들어있는 컬럼 이름         |
    # | `overlap`        | 다음 Chunk와 겹칠 토큰 개수      |
        # examples = {
        #     "generated_text": [
        #         "고혈압은...",
        #         "당뇨병은..."
        #     ]
        # }
        # examples["generated_text"]
    def chunk_text_with_overlap(examples, tokenizer, max_seq_length, text_field, overlap):
        """
        데이터셋의 각 텍스트를 토큰화하고 오버랩을 적용하여 max_seq_length 길이의 청크로 나눕니다.
        """
        # Overlap은 Chunk 크기보다 작아야 한다. 안 그러면 청킹을 할 수 없다.
        if overlap >= max_seq_length:
            raise ValueError("Overlap size must be smaller than max_seq_length")

        # 다음 Chunk를 시작할 위치(이동 거리)를 계산합니다.
        # 예를 들어
        #     stride = 7
        #     이면,
        #     Chunk1
        #     1 2 3 4 5 6 7 8 9 10
        #     Chunk2
        #             8 9 10 11 12 13 14 15 16 17
        stride = max_seq_length - overlap
        # texts = examples["generated_text"]
        texts = examples[text_field]

        # 토큰화된 결과를 저장
        all_token_ids = []
        # 문장 하나씩 토큰화
        for text in texts:
            # 토큰화
            # add_special_tokens=False
            # → [BOS], [EOS] 같은 특수 토큰을 붙이지 않음
            # truncation=False
            # → 길어도 자르지 않음
            # padding=False
            # → 짧아도 빈칸을 채우지 않음
            tokenized_output = tokenizer(text, add_special_tokens=False, truncation=False, padding=False)
            # tokenizer()의 결과는 여러 정보를 담고 있습니다.
                # tokenized_output = {
                #     "input_ids": [12, 35, 88],
                #     "attention_mask": [1, 1, 1]
                # }
            all_token_ids.append(tokenized_output["input_ids"])

        # 최종적으로는 아래 처럼 저장됩니다.
        #     chunked_input_ids = [
        #         [1,2,3,4,5],
        #         [4,5,6,7,8],
        #         [7,8,9,10]
        #     ]
        chunked_input_ids = []
        chunked_attention_mask = []

        for token_ids in all_token_ids:
            for i in range(0, len(token_ids), stride):
                # Chunk 자르기
                # 예를 들어:
                #     token_ids
                #     1 2 3 4 5 6 7 8 9 10
                #     max_seq_length = 5
                # 이면,
                #     i=0
                #     token_ids[0:5]
                #     ↓
                #     1 2 3 4 5
                #     i=3
                #     token_ids[3:8]
                #     ↓
                #     4 5 6 7 8
                #     i=6
                #     token_ids[6:11]
                #     ↓
                #     7 8 9 10
                # 이렇게 Overlap이 적용
                chunk = token_ids[i : i + max_seq_length]
                
                # 빈 Chunk 제외
                if len(chunk) > 0:
                    # Chunk 저장
                    chunked_input_ids.append(chunk)
                    # Attention Mask는 "이 토큰은 실제 데이터다."를 표시하는 값
                        # 예를 들어
                            # chunk = [1,2,3,4,5]
                        # 이면
                            # [1] * 5
                            # ↓
                            # [1,1,1,1,1]
                    chunked_attention_mask.append([1] * len(chunk))

        # Chunk들을 Dataset 형식으로 반환
            # {
                # "input_ids": [
                    # [1,2,3,4,5],
                    # [4,5,6,7,8]
                # ],
                # "attention_mask": [
                    # [1,1,1,1,1],
                    # [1,1,1,1,1]
                # ]
            # }
        return {
            "input_ids": chunked_input_ids,
            "attention_mask": chunked_attention_mask,
        }

    # 학습 데이터 전체를 Overlap Chunk로 변환합니다.
    print("\n[INFO] 오버랩 청킹 진행 중...")
    
    # @@ train 데이터의 모든 문장에 chunk_text_with_overlap() 함수를 적용합니다.
    # map() -> "데이터 하나하나에 같은 함수를 실행해라"
    chunked_dataset_overlap = data["train"].map(
    
        # 데이터에 적용할 함수
        chunk_text_with_overlap,
        
        # 여러 개의 데이터를 한 번에 처리
        batched=True,
        
        # 기존 컬럼(generated_text) 제거
        # 원래는 generated_text 컬럼이 있었는데, Chunking 후에는 input_ids, attention_mask만 필요합니다.
        remove_columns=data["train"].column_names,
        
        # 함수에 전달할 추가 인자, 즉:
            # chunk_text_with_overlap(
                # examples,
                # tokenizer=tokenizer,
                # max_seq_length=max_seq_length,
                # text_field=text_field_name,
                # overlap=overlap_size
            # )                
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": max_seq_length,
            "text_field": text_field_name,
            "overlap": overlap_size,
        },
    )



    # @@ 청킹이 잘 되었는지 확인하고, 청킹된 데이터를 다시 분석하는 코드    
    print(f"Original dataset size: {len(data['train'])}")
    print(f"Chunked dataset size (with overlap): {len(chunked_dataset_overlap)}")

    print("\n[INFO] 청킹된 데이터 토큰 분포 분석 중...")
    # 청킹된 데이터를 분석:
        # tokenized=True 
            # input_ids = [15, 82, 391, ...] 처럼 토큰화가 끝난 상태입니다.
            # 다시 토큰화하지 않고, 바로 토큰 개수만 계산합니다.            
        # 이번에는 그래프를
            # token_distribution_original.png
            # → 원본 데이터 분석
            # token_distribution_chunked.png
            # → 청킹 후 데이터 분석
    analyze_token_distribution(
        chunked_dataset_overlap,
        tokenized=True,
        save_path=os.path.join(BASE_DIR, "token_distribution_chunked.png"),
    )

    # @@학습을 시작하기 전에 모델을 학습용으로 설정하는 코드
    # -------------------------------------------------------------------
    # 9. 학습 파라미터 설정 및 Trainer 구성
    #    - SFTTrainer 대신 HuggingFace 기본 Trainer 사용
    #      (trl 버전에 따라 SFTTrainer의 _chunked_ce_forward가 CausalLMOutputWithPast와
    #       호환되지 않는 문제를 회피)
    #    - chunked_dataset_overlap은 이미 input_ids/attention_mask로 전처리된 상태이므로
    #      DataCollatorForLanguageModeling으로 labels를 자동 생성
    # -------------------------------------------------------------------
    # GPU 메모리 절약을 위한 Gradient Checkpointing
    model.gradient_checkpointing_enable()
    # gradient_checkpointing과 use_cache는 함께 사용할 수 없어서 False로 설정합니다.
    #     추론(Inference) → use_cache=True
    #     학습(Training) → use_cache=False
    model.config.use_cache = False  # checkpointing 사용 시 use_cache는 False

    # Gemma 같은 모델은 기본적으로 pad_token이 없는 경우가 있습니다.
    tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰을 eos 토큰으로 설정

    # 학습 직전에 데이터를 모델이 먹을 수 있는 형태로 자동 변환해주는 객체를 만듭니다.
    # DataCollator: input_ids를 그대로 labels로 복사 (CLM 방식, mlm=False)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM (GPT 방식) → 다음 토큰 예측. GPT, Gemma처럼 "다음 토큰 예측(Causal LM)" 방식으로 학습한다.
    )

    # @@ TrainingArguments = "Trainer에게 학습을 어떻게 할지 알려주는 설정집"
    training_args = TrainingArguments(
        report_to="tensorboard",           # 텐서보드로 학습 모니터링
        num_train_epochs=3,                # 전체 데이터 3회 반복 학습
        # GPU가 한 번에 2개의 데이터를 처리합니다.
        per_device_train_batch_size=2,     # 1 → 2 (VRAM 20GB 여유로 복원). 
        #  GPU는 2개씩 처리하지만, 8번 모아서 한 번 Weight를 업데이트합니다.
        gradient_accumulation_steps=8,    # 16 → 8 (배치 2로 늘린 만큼 조정, 실질 Batch 16 유지).
        # Cosine 곡선 형태로 천천히 줄여가는 방식
        lr_scheduler_type="cosine",   
        # Weight를 얼마나 크게 수정할지 결정합니다.
        #     크면 → 빨리 배우지만 불안정
        #     작으면 → 천천히 안정적으로 학습
        learning_rate=5e-5,               # 기본 Pretrain보다 낮은 학습률. Weight를 얼마나 크게 수정할지 결정합니다.
        # 처음 3% 구간은 천천히 학습률을 올립니다.
        # 학습 초반을 안정적으로 만들기 위한 기법입니다.
        warmup_ratio=0.03,
        # GPU에 맞는 정밀도(Float 타입) 를 사용합니다.
        # 앞에서 GPU를 검사해서 결정한 값입니다.
        bf16=use_bf16,                    # Ada Lovelace → bf16=True
        fp16=use_fp16,                    # Ada Lovelace → fp16=False
        optim=optimizer,                  # Linux → paged_adamw_8bit
        # 학습 결과를 저장할 폴더입니다.
        output_dir=OUTPUT_DIR,
        # 25 Step마다 로그를 출력합니다.
        logging_steps=25,
        # Checkpoint를 1개만 유지합니다.
        # 오래된 것은 자동 삭제합니다.
        save_total_limit=1,               # 체크포인트를 최대 1개만 유지
        # 기존 결과가 있으면 덮어쓰기 합니다.
        overwrite_output_dir=True,        # output_dir 재실행 시 덮어쓰기
        # CPU → GPU 데이터 전송을 빠르게 해서 학습 속도를 조금 향상시킵니다.
        dataloader_pin_memory=True,       # 1 → 복원 (VRAM 여유, 데이터 로딩 속도 향상)
    )

    print("\n[INFO] 학습 시작...")

    # 실제 학습을 시작하는 핵심 코드
    # | 파라미터            | 의미                         |
    # | --------------- | -------------------------- |
    # | `model`         | 학습할 Gemma 모델               |
    # | `args`          | 학습 설정(`TrainingArguments`) |
    # | `train_dataset` | 학습 데이터                     |
    # | `data_collator` | 데이터를 모델 입력 형태로 변환          |
    # | `tokenizer`     | 토크나이저                      |
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=chunked_dataset_overlap,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 데이터 읽기
    # ↓
    # Batch 생성
    # ↓
    # 모델 실행
    # ↓
    # Loss 계산
    # ↓
    # Optimizer 실행
    # ↓
    # Weight 업데이트
    # ↓
    # 반복(Epoch 종료까지)
    trainer.train()

    print("[INFO] 학습 완료.")
    print(f"[INFO] TensorBoard 확인: tensorboard --logdir {os.path.join(OUTPUT_DIR, 'runs')} --port 2025 --bind_all")

    # -------------------------------------------------------------------
    # 10. 학습 후 평가
    # -------------------------------------------------------------------
    # @@ 학습 전과 같은 질문을 다시 해서, 모델가 얼마나 좋아졌는지 확인합니다.
    
    # 모델을 "학습 모드"에서 "평가(추론) 모드"로 변경합니다.
    #     model.train() → 학습 모드
    #     model.eval() → 평가(추론) 모드
    model.eval()

    # 사용하지 않는 GPU 메모리를 정리합니다.
    torch.cuda.empty_cache()

    print("\n========== [학습 후] 파인튜닝 모델 출력 ==========")
    # 학습 전에 사용했던 동일한 질문 4개를 다시 사용합니다.
    for q in eval_inputs:
        print(f"{q}", end="")
        # 모델이 답변을 생성
        # 실무에서는 학습 후 Pipeline을 다시 생성하는 것이 더 명확하고 안전한 방법입니다.
        for s in base_llm.stream(q):
            print(s, end="")
        print("\n-------")

    # -------------------------------------------------------------------
    # 11. 모델 저장 (로컬) - 이미 저장된 경우 재저장 방지
    # -------------------------------------------------------------------
    # @@ 파인튜닝된 모델과 Tokenizer를 로컬 디스크에 저장합니다.
    # 모델과 Tokenizer는 항상 함께 저장해야 나중에 다시 사용할 수 있습니다.
    if os.path.exists(MODEL_SAVE_DIR) and os.listdir(MODEL_SAVE_DIR):
        print(f"\n[INFO] 모델이 이미 저장되어 있습니다: {MODEL_SAVE_DIR} (재저장 생략)")
    else:
        print(f"\n[INFO] 모델 저장 중: {MODEL_SAVE_DIR}")
        # 파인튜닝된 모델의 가중치(Weights)와 설정을 저장합니다.
        #     safe_serialization=False
        #         저장 형식을 지정합니다.
        #         False → pytorch_model.bin
        #         True → model.safetensors
        model.save_pretrained(MODEL_SAVE_DIR, safe_serialization=False)
        
        # 모델만 저장하면 안 되고, Tokenizer도 반드시 함께 저장해야 합니다.
        tokenizer.save_pretrained(MODEL_SAVE_DIR)
        print("[INFO] 모델 저장 완료.")

    # -------------------------------------------------------------------
    # 12. (선택) HuggingFace Hub 업로드
    #     - HF_WRITE_TOKEN과 HF_USERNAME이 .env에 설정된 경우에만 실행
    # -------------------------------------------------------------------
    # @@  내 컴퓨터에 저장한 모델을 Hugging Face 서버에 업로드합니다.
    
    # .env 파일에서
    #     HF_WRITE_TOKEN=xxxxxxxx
    #     HF_USERNAME=myname
    hf_write_token = os.getenv("HF_WRITE_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    # 둘 다 존재하면 업로드를 진행합니다.
    if hf_write_token and hf_username:
        print(f"\n[INFO] HuggingFace Hub 업로드 중: {hf_username}/{os.path.basename(MODEL_SAVE_DIR)}")
        # Hugging Face에 로그인합니다.
        login(token=hf_write_token)
        # 저장소 이름 생성: EX) hong/gemma-3-1b-pt-MED
        repo_id = f"{hf_username}/{os.path.basename(MODEL_SAVE_DIR)}"
        # 모델 업로드 ⭐
        model.push_to_hub(repo_id)
        # Tokenizer도 같이 업로드합니다.
        # 모델만 올리면 다른 사람이 사용할 수 없습니다.
        # 모델과 Tokenizer는 한 세트입니다.
        #     예를 들어
        #         Gemma Tokenizer
        #             안녕하세요
        #             ↓
        #             [315, 912, 45]
        #         Qwen Tokenizer
        #            안녕하세요
        #             ↓
        #            [8821, 41]
        #     번호가 완전히 다릅니다.
        #     만약 Tokenizer를 바꿔버리면?
        #         Gemma 모델은
        #           315
        #             ↓
        #           "안"
        #     라고 학습했는데, 다른 Tokenizer에서는
        #           315
        #           ↓
        #           "고양이"
        #     일 수도 있습니다.
        tokenizer.push_to_hub(repo_id)
        print("[INFO] 업로드 완료.")
    else:
        print("\n[INFO] HF_WRITE_TOKEN / HF_USERNAME 미설정 - Hub 업로드 생략")

    print("\n[완료] Continuous Pretraining 파이프라인 종료.")


if __name__ == "__main__":
    main()