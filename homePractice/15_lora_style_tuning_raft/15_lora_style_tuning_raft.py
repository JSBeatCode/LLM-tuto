"""
[실습] LoRA를 이용한 Style Tuning: RAFT
=========================================

이미 학습된 QA 모델을 QLoRA(4bit 양자화 + LoRA) 기반 SFT로 파인튜닝하여,
'건방진 QA 봇'으로 스타일을 변환하는 실습 스크립트.

실행 방법
---------
    python 15_lora_style_tuning_raft.py --mode all      # 학습 + 평가 + 저장 (기본값)
    python 15_lora_style_tuning_raft.py --mode train     # 학습만 수행
    python 15_lora_style_tuning_raft.py --mode eval       # 저장된 LoRA 어댑터로 평가만 수행

사전 준비
---------
1. 이 스크립트와 같은 폴더에 '.env' 파일을 만들고 아래 값을 채워주세요.
       HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx   # gated 모델 접근 시 필요, 없으면 비워둬도 됨
       HF_PUSH_TO_HUB=false                        # true 로 바꾸면 학습 후 허깅페이스 허브에 업로드
       HF_HUB_USERNAME=your_hf_username            # HF_PUSH_TO_HUB=true 인 경우에만 필요
2. 이 스크립트와 같은 폴더에 'RAG_Data_full.csv', 'RAG_Data_full_neg.csv'
   (cp949 인코딩, context/question/cot 컬럼 포함)를 위치시켜 주세요.
"""

# 타입 힌트를 편하게 쓰기 위한 옵션입니다.
# 예전에는
#     def test() -> MyClass:
# 처럼 아직 만들어지지 않은 클래스를 타입으로 쓰면 에러가 날 수도 있었는데,
# 이걸 사용하면 문자열처럼 나중에 해석합니다.
#     def test() -> MyClass
from __future__ import annotations

# ── 표준 라이브러리 ──────────────────────────────────────────────

# 명령어 옵션을 읽습니다.
# python train.py --mode train 여기서 '--mode train' 를 읽어오는 라이브러리
import argparse
# 운영체제(OS) 기능 사용
# 예를 들면
#     폴더 생성
#     파일 존재 확인
#     환경변수
#     경로 생성
# 거의 모든 Python 프로젝트가 사용합니다.
import os
# 프로그램 종료용
import sys

# ── 로컬 캐시 경로를 다른 라이브러리 임포트보다 먼저 설정 ─────────
# huggingface_hub / transformers / datasets / matplotlib 가 캐시 경로를 읽기
# 전에 환경변수를 지정해야 스크립트와 동일한 폴더에 캐시가 쌓인다.
# 시스템에 이미 같은 환경변수가 설정돼 있어도 이 스크립트 실행 중에는
# 반드시 로컬 경로를 쓰도록 setdefault 대신 직접 대입으로 강제 override 한다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HF_CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
MPL_CACHE_DIR = os.path.join(BASE_DIR, "mpl_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
# NOTE: TRANSFORMERS_CACHE 는 최신 transformers 에서 deprecated 되어
# HF_HOME 으로 대체되었으므로 더 이상 별도로 지정하지 않는다.
os.environ["MPLCONFIGDIR"] = MPL_CACHE_DIR

# ── 서드파티 라이브러리 (알파벳 순) ───────────────────────────────
# 그래프 그리는 라이브러리입니다.
import matplotlib
# GUI 없이 png 파일만 저장하도록 설정합니다.
matplotlib.use("Agg")  # 헤드리스 환경에서 그래프를 파일로만 저장
# 실제로 그래프를 그리는 함수입니다.
import matplotlib.pyplot as plt
# 수치 계산 라이브러리
import numpy as np
# 데이터프레임 라이브러리
# CSV를 datasets가 읽기 때문에 없어도 됩니다.
import pandas as pd
# 예쁜 그래프. matplotlib보다 보기 좋게 그립니다.
import seaborn as sns
# LLM은 모두 Tensor 연산으로 돌아갑니다.
# 예)
#     모델
#     GPU
#     학습
#     Gradient
#     Loss
# 전부 torch가 처리합니다.
import torch
# GPU를 쉽게 사용하게 해줍니다.
#     CUDA
#     DDP
#     Multi GPU
# 를 직접 구현해야 했는데, Accelerate가 대신 처리합니다.
from accelerate import Accelerator
# HuggingFace 데이터셋 라이브러리.
# CSV를 읽어서 Dataset 객체로 만들어 줍니다.
from datasets import load_dataset
# .env 파일을 읽습니다.
from dotenv import load_dotenv
# 허깅페이스 로그인
# 모델 다운로드, 업로드
from huggingface_hub import login
# Transformers Pipeline을 LangChain LLM 형태로 바꿔줍니다.
from langchain_huggingface import HuggingFacePipeline

# PEFT는 
#   "거대한 LLM 전체를 학습하지 않고, 아주 작은 부분만 추가해서 학습하는 라이브러리"입니다.
# 예를 들어,
#     원본 Gemma 모델: 120억 개의 파라미터
#     LoRA가 추가하는 파라미터: 수백만 개 수준
# PEFT는 이 추가된 작은 파라미터(LoRA Adapter)만 학습하도록 만들어 줍니다.
# 그래서 얻는 장점은
#     🚀 학습 속도가 빠르고
#     💾 GPU 메모리를 적게 사용하며
#     📦 저장 용량도 매우 작습니다.
# import 설명:
#     LoRAConfig -> LoRA 설정
#         r
#         alpha
#         dropout
#     등을 지정합니다.
#     get_peft_model() -> 기존 모델을
#         Gemma
#         ↓
#         Gemma + LoRA
#     로 바꿔줍니다.
#     prepare_model_for_kbit_training() -> QLoRA 학습을 위한 준비 작업입니다.
#         4bit 모델을 학습 가능하게 내부 설정을 변경합니다.
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# Transformers는 LLM을 쉽게 불러오고 사용하는 라이브러리입니다.
#     🤖 Gemma 모델 불러오기 (AutoModelForCausalLM)
#     🔤 토크나이저 불러오기 (AutoTokenizer)
#     ⚡ 4bit 양자화 설정 (BitsAndBytesConfig)
#     💬 추론(질문→답변) 실행 (pipeline)
# 을 모두 transformers가 담당합니다.
from transformers import (
    AutoModelForCausalLM, # LLM 모델 로드
    AutoTokenizer, # 토크나이저 로드
    BitsAndBytesConfig, # 4bit 양자화 설정
    pipeline, # 추론(Inference)을 쉽게 하는 API입니다.
)
# SFTConfig -> 학습 설정
# SFTTrainer -> 
#     Forward
#     Loss
#     Backward
#     Optimizer
# 모두 수행하는 Trainer입니다.
# 예전에는 직접 학습 루프를 작성해야 했지만,
# TRL이 대부분 자동으로 처리해 줍니다.
from trl import SFTConfig, SFTTrainer

# ── 환경변수 로드 ────────────────────────────────────────────────
load_dotenv(override=True)

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "").strip()
HF_PUSH_TO_HUB = os.getenv("HF_PUSH_TO_HUB", "false").strip().lower() == "true"
HF_HUB_USERNAME = os.getenv("HF_HUB_USERNAME", "").strip()

# ── 경로 및 상수 설정 (모두 스크립트와 동일한 폴더 기준) ───────────
MODEL_ID = "unsloth/gemma-3-12b-it"
MODEL_NAME = MODEL_ID.split("/")[1]

DATA_FILES = [
    os.path.join(BASE_DIR, "RAG_Data_full.csv"),
    os.path.join(BASE_DIR, "RAG_Data_full_neg.csv"),
]
DATA_ENCODING = "cp949"

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ADAPTER_DIR = os.path.join(BASE_DIR, f"{MODEL_NAME}-Rude-LORA")
TOKEN_DIST_PLOT_PATH = os.path.join(OUTPUT_DIR, "token_length_distribution.png")

# LLM이 답변을 생성할 때 사용하는 생성 옵션(Generation Config)
GEN_CONFIG = dict(
# LLM이 확률적으로 단어를 선택합니다.
# 예를 들어
#     질문
#         대한민국의 수도는?
#     확률이
#         서울   99%
#         부산   0.5%
#         인천   0.5%
# 이렇게 샘플링을 하면 확률을 보고 선택합니다.
    do_sample=True,
    # 답변 길이 제한입니다.
    max_new_tokens=1024,
    # 누적 확률 95% 안에 있는 후보만 사용합니다.
    #     서울      50%
    #     부산      70%
    #     인천      85%
    #     대전      93%
    #     제주      97%
    # 95%까지만 사용하므로
    #     서울
    #     부산
    #     인천
    #     대전
    # 까지만 후보가 됩니다.
    top_p=0.95,
    # 확률이 높은 64개 단어만 후보로 사용합니다.
    # 그 안에서 top_p가 또 적용됩니다.
    top_k=64,
)


# ══════════════════════════════════════════════════════════════
# 1. 유틸리티: Hugging Face 로그인
# ══════════════════════════════════════════════════════════════
def hf_login_if_needed() -> None:
    """토큰이 .env 에 있는 경우에만 로그인한다 (불필요한 API 호출 방지)."""
    if HUGGINGFACE_TOKEN:
        login(token=HUGGINGFACE_TOKEN)
        print("[INFO] Hugging Face 로그인 완료")
    else:
        print("[INFO] HUGGINGFACE_TOKEN 이 설정되지 않아 로그인을 건너뜁니다.")


# ══════════════════════════════════════════════════════════════
# 2. 모델 / 토크나이저 로드 (4bit 양자화)
# ══════════════════════════════════════════════════════════════
# Gemma 모델과 Tokenizer를 4bit(QLoRA용)로 메모리에 로드하는 함수
def load_model_and_tokenizer():
    """
    이미 HF_HUB_CACHE 에 모델이 캐시되어 있다면 huggingface_hub 가
    자동으로 재사용하므로 별도의 중복 다운로드 방지 로직이 필요 없다.
    (from_pretrained 는 캐시 존재 여부를 자체적으로 확인한다.)
    """
    # GPU 행렬곱(Matrix Multiplication)의 연산 방식을 조금 더 빠르게 설정
    # GPU 연산 최적화 옵션입니다.
    torch.set_float32_matmul_precision("high")

        # 양자화(Quantization):
        # Gemma 12B 모델은 원래
        #     Gemma 12B
        #     ↓
        #     약 24GB (FP16)
        # 정도의 GPU 메모리가 필요합니다.
        # 그런데 대부분의 GPU는 12~16GB. 메모리가 부족해서 실행조차 안 됩니다.
        # -> 그래서 원래 모델을 큰 압축파일로 만드는 것입니다.
        # 이 모델을 어떤 방식으로 압축해서 메모리에 올릴지"를 설정하는 객체: 모델을 로드하는 방법을 정의하는 설정
    bnb_config = BitsAndBytesConfig(
        # 16 -> 4bit 압축해서 메모리에 올립니다.
        load_in_4bit=True,
        # 4bit로 한 번 압축하고 그 압축 정보까지 한 번 더 압축합니다.
        bnb_4bit_use_double_quant=True,
        # NF4: 4bit 압축 방식입니다. 
        bnb_4bit_quant_type="nf4",
        # 모델은 4bit 로 저장되어 있지만 계산은 bfloat16 으로 합니다.
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"[INFO] 모델 로드 중: {MODEL_ID}")
    # Tokenizer를 불러옵니다.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # 실제로 Gemma를 불러오는 부분입니다.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        # 자동으로 bfloat16 또는 float16를 선택합니다.
        # GPU에 맞게 결정됩니다.
        torch_dtype="auto",
        # 아까 만든 양자화 4bit 설정을 적용합니다.
        quantization_config=bnb_config,
        # GPU 없으면 CPU 자동으로 올립니다.
        device_map="auto",
        # Attention 구현 방식을 지정합니다.
        attn_implementation="eager",
    )
    # GPU는 이런 데이터를 한 번에 계산(batch)해야 하기 때문에 길이를 맞춰야 함.
    #     안녕하세요. -> [10, 52]
    #     안녕하세요. 저는 AI입니다. -> [10, 52, 301, 77, 99]
    # 길이를 맞춤
    #     문장1 ->[10, 52, PAD, PAD, PAD]
    #     문장2 -> [10, 52, 301, 77, 99]
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ══════════════════════════════════════════════════════════════
# 3. 데이터 로드
# ══════════════════════════════════════════════════════════════
def load_training_data():
    for path in DATA_FILES:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"학습 데이터 파일을 찾을 수 없습니다: {path}\n"
                "스크립트와 동일한 폴더에 CSV 파일을 위치시켜 주세요."
            )

    data = load_dataset(
        "csv",
        encoding=DATA_ENCODING,
        data_files={"train": DATA_FILES},
    )
    data = data.shuffle()
    return data


# ══════════════════════════════════════════════════════════════
# 4. 프롬프트 포맷 함수
# ══════════════════════════════════════════════════════════════
def convert_format(tokenizer, context, question, answer=None, add_generation_prompt=False):
    """프롬프트 엔지니어링만으로 페르소나를 부여한 버전 (베이스라인 비교용)."""
    chat = [
        {
            "role": "system",
            "content": (
                "너는 무척 거만한 AI야. 사용자가 물어본 [Question]에 대해 주어진 "
                "[Context]를 참고해서 반말로 대답해.\n"
                "정답을 알고 있다면, 대답은 무조건 '그것도 몰라?'로 시작해야 해.\n"
                "그 뒤에 [Context]에서 관련 있는 부분을 '여기' 로 인용하면서 설명해.\n"
                "이후 답변을 요약하며 거만하고 무례하게 답변해.\n"
                "모르는 경우에는 '내가 그딴 걸 어떻게 알아?'라고만 대답해."
            ),
        },
        {
            "role": "user",
            "content": f"Context: {context}\n---\nQuestion: {question}",
        },
    ]
    if answer:
        chat.append({"role": "assistant", "content": f"{answer}"})
    return {
        "text": tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    }


def convert_format_without_pe(tokenizer, context, question, answer=None, add_generation_prompt=False):
    """실제 학습에 사용하는, 짧은 시스템 프롬프트 버전."""
    chat = [
        {
            "role": "system",
            "content": "주어진 [Context]를 참고하여, [Question]에 거만하게 대답하세요.",
        },
        {
            "role": "user",
            "content": f"Context: {context}\n---\nQuestion: {question}",
        },
    ]
    if answer:
        chat.append({"role": "assistant", "content": f"{answer}"})
    return {
        "text": tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    }


def build_prompt_completion(tokenizer, context, question, answer):
    """
    학습용 (prompt, completion) 쌍을 생성한다.

    구버전 trl 은 `DataCollatorForCompletionOnlyLM` + response_template 매칭으로
    답변 부분만 loss 계산을 했지만, 최신 trl(0.16+)에서는 이 클래스가 완전히
    제거되고 prompt/completion 컬럼을 분리해 넘기면 `completion_only_loss=True`
    설정만으로 동일한 효과(답변 부분만 학습)를 낸다.
    """
    prompt_text = convert_format_without_pe(
        tokenizer, context, question, add_generation_prompt=True
    )["text"]
    completion_text = f"{answer}{tokenizer.eos_token}"
    return {"prompt": prompt_text, "completion": completion_text}


# ══════════════════════════════════════════════════════════════
# 5. 베이스라인(파인튜닝 전) 응답 확인
# ══════════════════════════════════════════════════════════════
def run_baseline_check(model, tokenizer, data):
    example = data["train"][0]
    context, question = example["context"], example["question"]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        **GEN_CONFIG,
    )
    base_llm = HuggingFacePipeline(pipeline=pipe, pipeline_kwargs=GEN_CONFIG)

    test_prompt = convert_format(
        tokenizer, context, question, add_generation_prompt=True
    )["text"]

    print("\n[베이스라인 프롬프트]")
    print(test_prompt)
    print("\n[베이스라인 응답 (프롬프트 엔지니어링만 적용)]")
    for chunk in base_llm.stream(test_prompt):
        print(chunk, end="")
    print()


# ══════════════════════════════════════════════════════════════
# 6. 토큰 길이 분포 분석
# ══════════════════════════════════════════════════════════════
# 내 학습 데이터가 모델이 처리할 수 있는 길이인지 미리 확인하는 것
# 분석하는 이유 -> 주로 아래 3가지를 확인합니다.
#     너무 긴 데이터가 있는가?
#         모델 최대 길이(예: 2048, 4096)를 넘으면 잘려서 학습됩니다.
#     대부분 데이터가 어느 정도 길이인가?
#         max_seq_length를 적절히 정할 수 있습니다.
#     이상치(Outlier)가 있는가?
#         예를 들어 대부분 100토큰인데 하나만 5000토큰이면 데이터 오류를 의심할 수 있습니다.
def analyze_token_distribution(tokenizer, dataset, bins=30):
    """prompt + completion 을 합친 전체 시퀀스 길이 기준으로 분포를 계산한다."""
    token_counts = [
        len(tokenizer.encode(prompt + completion))
        for prompt, completion in zip(dataset["prompt"], dataset["completion"])
    ]

    stats = {
        "평균 토큰 수": np.mean(token_counts),
        "중앙값": np.median(token_counts),
        "최소 토큰 수": min(token_counts),
        "최대 토큰 수": max(token_counts),
        "표준편차": np.std(token_counts),
        "90퍼센타일": np.percentile(token_counts, 90),
        "95퍼센타일": np.percentile(token_counts, 95),
        "99퍼센타일": np.percentile(token_counts, 99),
        "총 샘플 수": len(token_counts),
    }

    plt.figure(figsize=(12, 6))
    sns.histplot(data=token_counts, bins=bins, kde=True)
    plt.title("Token Length Distribution (prompt + completion)")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(TOKEN_DIST_PLOT_PATH)
    plt.close()
    print(f"[INFO] 토큰 길이 분포 그래프 저장: {TOKEN_DIST_PLOT_PATH}")

    print("\n=== 토큰 수 통계 ===")
    for key, value in stats.items():
        print(f"{key}: {value:.1f}")

    return stats


# ══════════════════════════════════════════════════════════════
# 7. LoRA 설정
# ══════════════════════════════════════════════════════════════
# LoRA 
#     -> Low-Rank Adaptation
#     -> 작은 행렬(Low-Rank Matrix)만 학습해서 기존 모델을 새로운 작업에 적응시키는 방법

# Gemma 모델에 LoRA를 붙여서, 일부만 학습 가능하게 만드는 함수입니다.
# LoRA를 적용하기 전
#     Gemma 모델
#         [Layer]
#         [Layer]
#         [Layer]
#         [Layer]
#     모든 파라미터를 학습해야 합니다.
#     → 메모리 많이 사용
#     → 학습 느림
# LoRA를 적용한 후
#     Gemma 모델 (동결)
#         [Layer] + LoRA
#         [Layer] + LoRA
#         [Layer] + LoRA
#         [Layer] + LoRA
#     기존 Gemma는 그대로 유지하고, LoRA라는 작은 학습용 레이어만 추가합니다.
# 학습은 LoRA만 합니다.
def apply_lora(model):
    # 옵션: GPU 메모리를 절약합니다. 대신 학습 속도는 약간 느려집니다.
    model.gradient_checkpointing_enable()
    # 학습을 위해 Cache 기능 비활성화
    model.config.use_cache = False
    # 4bit(QLoRA) 모델을 학습 가능하도록 준비
    model = prepare_model_for_kbit_training(model)

    # LoRA의 학습 설정을 만듭니다.
    lora_config = LoraConfig(
        # LoRA의 크기(Rank)를 의미합니다.
        #     작을수록
        #         메모리 ↓
        #         학습량 ↓
        r=4,
        # LoRA의 학습 강도를 조절합니다. 값이 클수록 LoRA의 영향력이 커집니다.
        lora_alpha=8,
        # LoRA를 어느 Layer에 붙일지 지정
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        # LoRA Dropout 
        #     -> 학습할 때 일부 뉴런을 랜덤하게 꺼버리는 기법입니다
        #     -> 모델이 특정 데이터만 너무 외우는 것(과적합, Overfitting) 을 막기 위함
        # 0 은 이 기능을 사용하지 않습니다.
        lora_dropout=0,
        # Bias는 학습하지 않습니다. LoRA만 학습합니다.
        bias="none",
        # 현재 작업이 텍스트 생성(LLM) 임을 알려줍니다.
        task_type="CAUSAL_LM",
    )

    # 기존 Gemma에 LoRA를 붙입니다.
    model = get_peft_model(model, lora_config)
    # 학습되는 파라미터 수를 출력합니다.
    model.print_trainable_parameters()
    return model


# ══════════════════════════════════════════════════════════════
# 8. SFT 학습
# ══════════════════════════════════════════════════════════════
# LoRA가 적용된 Gemma 모델을 학습 데이터로 파인튜닝(SFT)하는 함수입니다

# SFT는 Supervised Fine-Tuning의 약자입니다.
#     Supervised = 정답이 있는
#     Fine-Tuning = 기존 모델을 추가 학습
# 즉, 정답이 있는 데이터를 이용해 기존 LLM을 추가 학습시키는 방법입니다.
# SFT 학습의 하이퍼파라미터 설정
def train_model(model, tokenizer, data):
    sft_config = SFTConfig(
        # 학습 로그를 외부 서비스(WandB, TensorBoard 등)에 보내지 않습니다.
        report_to="none",
        # 전체 학습 데이터를 2번 반복 학습합니다.
        num_train_epochs=2,
        # GPU가 한 번에 1개의 데이터만 학습합니다.
        # 메모리가 부족할 때 많이 사용합니다.
        per_device_train_batch_size=1,
        # 4번 계산 후 한 번 업데이트
        #     1개 학습
        #     1개 학습
        #     1개 학습
        #     1개 학습
        #         ↓
        #     가중치 업데이트
        # 실제로는 Batch Size = 4처럼 동작합니다.
        gradient_accumulation_steps=4,
        # 입력 토큰의 최대 길이를 1100개로 제한합니다. 넘으면 잘라냅니다.
        max_length=1100,
        # 답변(Completion) 부분만 학습. 질문은 학습 안합니다.
        completion_only_loss=True,  # prompt/completion 데이터셋에서 답변 부분만 loss 계산
        # 학습률을 점차 감소
        lr_scheduler_type="cosine",
        # 학습 속도를 결정하는 값입니다.
        #     1e-4 = 0.0001
        # 값이 크면 빨리 배우지만 불안정할 수 있고,
        # 작으면 안정적이지만 느립니다.
        learning_rate=1e-4,
        # 초반 3%는 천천히 학습 시작
        # 갑자기 큰 학습률로 시작하는 것을 방지합니다.
        warmup_ratio=0.03,
        # 연산을 bfloat16으로 수행합니다.
        # 메모리를 절약하면서도 성능을 유지할 수 있습니다.
        bf16=True,
        # 메모리 절약형 AdamW 사용
        # 메모리를 적게 사용하도록 최적화된 방식입니다.
        optim="paged_adamw_8bit",
        # 학습 결과 저장 위치
        output_dir=OUTPUT_DIR,
        # 25번 학습할 때마다 현재 Loss 등의 정보를 출력합니다.
        logging_steps=25,
    )

    # SFTTrainer(학습기)를 생성하는 코드
    # 이 모델을, 이 데이터로, 이 설정대로 학습해.
    trainer = SFTTrainer(
        model=model,
        # 학습에 사용할 데이터셋을 지정합니다.
        train_dataset=data["train"],
        # 앞에서 만든 SFT 학습 설정을 적용합니다.
        args=sft_config,
        # 텍스트를 토큰(Token) 으로 변환할 Tokenizer를 지정합니다.
        processing_class=tokenizer,
    )
    # Accelerate를 초기화합니다
    # 학습 환경(GPU 등) 관리자 생성.
    accelerator = Accelerator()
    # 멀티 GPU 환경에서는 메인 프로세스가 먼저 작업하도록 합니다.
    # 예를 들어 데이터 다운로드나 전처리를 여러 GPU가 동시에 하지 않도록 순서를 보장합니다.
    # GPU가 1개라면 사실상 큰 차이는 없습니다.
    with accelerator.main_process_first():
        # 실제 SFT 학습을 시작
        trainer.train()

    return model


# ══════════════════════════════════════════════════════════════
# 9. 학습 결과 평가
# ══════════════════════════════════════════════════════════════
# 학습이 완료된 LoRA 모델에게 실제 질문을 해 보고, 원하는 답변을 잘 생성하는지 평가하는 함수입니다.
def evaluate_model(model, tokenizer):
    # 학습 종료, 추론 모드로 전환
    model.eval()
    # 현재 모델은 내부적으로 여러 데이터 타입이 섞여 있습니다.
    #     원본 Gemma : 4bit (양자화)
    #     LoRA        : bfloat16
    #     입력 Tensor : bfloat16
    # 데이터 타입을 bfloat16으로 맞춰 dtype 충돌 방지
    model = model.to(torch.bfloat16)  # 양자화 + LoRA 조합에서 dtype 불일치 방지

    # "모델에게 질문하면 답변을 생성하는 엔진"을 만듭니다.
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # 결과에 입력 프롬프트까지 함께 반환합니다.
        return_full_text=True,
        # 앞에서 만든 생성 옵션을 적용합니다.
        **GEN_CONFIG,
    )
    # 그 엔진을 LangChain에서 사용할 수 있도록 포장(Wrapper)하는 역할을 합니다.
    tuned_llm = HuggingFacePipeline(pipeline=pipe, pipeline_kwargs=GEN_CONFIG)

    test_cases = [
        {
            "context": (
                "참고래의 등 부분은 밤회색이며, 배 쪽은 하얗다.\n"
                "튀어나온 두 쌍의 숨구멍이 있으며, 납작하고 넓은 주둥이를 가지고 있다.\n"
                "두 개의 밝은 색 문양이 숨구멍 뒤에서 시작해 몸의 측면으로 따라가 꼬리로 이어진다.\n"
                "오른쪽 턱에 하얀색 무늬가 있으며, 왼쪽은 회색 또는 검은색이다.\n"
                "참고래는 턱에서 몸 밑의 중앙부까지 이어지는 56에서 100개의 주름을 지니고 있는데,\n"
                "먹이를 잡을 때 목을 팽창시키기 쉽게 하기 위한 것이다.\n"
                "이들의 등지느러미의 길이는 60센티미터 정도이다.\n"
                "가슴지느러미는 아주 작으며, 꼬리는 넓고 V자 모양이며 끝은 뾰족한 편이다."
            ),
            "question": "참고래의 주름은 어떤 용도인가요? 등지느러미는 몇 CM인가요?",
        },
        {
            "context": (
                "LG 트윈스가 베테랑 사이드암 심창민을 영입, 2025 시즌을 대비한 불펜 뎁스 강화에 성공했다.\n\n"
                "LG 구단은 18일 심창민 영입을 공식 발표했다. 심창민은 지난 2011년 경남고를 졸업하고 "
                "신인 드래프트 1라운드, 전체 4순위로 삼성 라이온즈에 입단하며 프로 커리어를 시작했다. "
                "KBO리그 통산 11시즌동안 485경기 491이닝 31승 29패 80홀드 51세이브 4.22의 "
                "평균자책점을 기록했다.\n\n"
                "2016년에는 삼성의 마무리 투수 자리를 꿰찼다. 62경기 72⅔이닝 2승 6패 25세이브 4홀드 "
                "평균자책점 2.97로 리그 정상급 클로저의 면모를 보여줬다. 이후 2018년까지 삼성 필승조의 "
                "핵심으로 제 몫을 해줬다."
            ),
            "question": "심창민은 어느 팀에서 데뷔했나요? 그의 전성기는 언제인가요?",
        },
    ]

    for i, case in enumerate(test_cases, start=1):
        test_prompt = convert_format_without_pe(
            tokenizer, case["context"], case["question"], add_generation_prompt=True
        )["text"]
        print(f"\n[평가 예시 {i}] 질문: {case['question']}")
        print("[튜닝된 모델 응답]")
        for chunk in tuned_llm.stream(test_prompt):
            print(chunk, end="")
        print()

    return model


# ══════════════════════════════════════════════════════════════
# 10. 모델 저장 / (선택) 허브 업로드
# ══════════════════════════════════════════════════════════════
# "이미 학습이 끝난 모델이 저장되어 있으면 다시 학습하지 말자." 라는 용도입니다.
def adapter_already_saved() -> bool:
    """LoRA 어댑터가 이미 저장되어 있는지 sentinel 파일로 확인 (재학습 방지)."""
    sentinel = os.path.join(ADAPTER_DIR, "adapter_config.json")
    return os.path.exists(sentinel)


# 학습이 끝난 LoRA 모델을 저장하고, 필요하면 Hugging Face Hub에도 업로드하는 함수입니다.
def save_model(model, tokenizer):
    # LoRA 모델 저장
    model.save_pretrained(ADAPTER_DIR)
    # Tokenizer 저장
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"[INFO] LoRA 어댑터 저장 완료: {ADAPTER_DIR}")

    if HF_PUSH_TO_HUB:
        if not HF_HUB_USERNAME:
            print("[WARN] HF_HUB_USERNAME 이 비어 있어 허브 업로드를 건너뜁니다.")
            return
        # 업로드할 저장소 이름을 만듬:
        #     HF_HUB_USERNAME = hong
        #     ADAPTER_DIR = ./gemma-style-lora
        repo_id = f"{HF_HUB_USERNAME}/{os.path.basename(ADAPTER_DIR)}"
        # LoRA 모델을 Hugging Face Hub에 업로드합니다.
        model.push_to_hub(repo_id)
        # Tokenizer도 함께 업로드합니다.
        tokenizer.push_to_hub(repo_id)
        print(f"[INFO] 허브 업로드 완료: {repo_id}")
    else:
        print("[INFO] HF_PUSH_TO_HUB=false 이므로 허브 업로드를 건너뜁니다.")

# 저장된 LoRA 어댑터를 원본 Gemma 모델에 다시 연결하여, 
# 재학습 없이 바로 평가(추론)할 수 있는 모델을 만드는 함수입니다.
def load_adapter_for_eval(tokenizer):
    """이미 저장된 LoRA 어댑터를 베이스 모델에 로드한다 (재학습 없이 평가만 수행)."""
    # PeftModel은 저장된 LoRA 어댑터를 원본 모델에 연결하여, 
    # 학습된 모델을 복원하는 클래스
    from peft import PeftModel

    # 양자화 설정 시작
    bnb_config = BitsAndBytesConfig(
        # 모델을 4bit로 압축하여 메모리에 로드합니다.
        # 메모리를 크게 절약할 수 있습니다.
        load_in_4bit=True,
        # 4bit 데이터를 한 번 더 양자화(Double Quantization) 하여 
        # 메모리를 추가로 절약합니다.
        bnb_4bit_use_double_quant=True,
        # 4bit 양자화 방식을 NF4(Normal Float 4) 로 사용합니다.
        # QLoRA에서 가장 많이 사용하는 방식입니다.
        bnb_4bit_quant_type="nf4",
        # 모델은 4bit로 저장하지만, 실제 연산은 bfloat16으로 수행합니다.
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # 원본 Gemma 모델을 메모리에 로드하는 코드
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        # 모델이 권장하는 데이터 타입을 자동으로 사용합니다.
        torch_dtype="auto",
        # 앞에서 만든 4bit 양자화 설정을 적용하여 모델을 불러옵니다.
        quantization_config=bnb_config,
        # GPU, CPU 등 사용 가능한 장치에 모델을 자동으로 배치합니다.
        device_map="auto",
        # Attention 연산 방식을 eager 모드로 사용합니다.
        attn_implementation="eager",
    )
    # 저장된 LoRA 어댑터를 원본 Gemma 모델에 연결하여, 학습이 반영된 모델을 복원하는 코드
    #     base_model 
    #     -> 원본 Gemma 모델입니다.
    #     ADAPTER_DIR 
    #     -> gemma-3-12b-it-Rude-LORA/
    #         adapter_config.json
    #         adapter_model.safetensors
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    return model


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════
# main의 실행경로:
# | 실행 모드          | 모델 로드 | 데이터 로드 | Prompt 생성 | 베이스라인 평가 | 토큰 분석 | LoRA 학습 | LoRA 저장 | 저장된 LoRA 로드 | 최종 평가 |
# | -------------- | ----- | ------ | --------- | -------- | ----- | ------- | ------- | ----------- | ----- |
# | `--mode eval`  | ✅     | ❌      | ❌         | ❌        | ❌     | ❌       | ❌       | ✅           | ✅     |
# | `--mode train` | ✅     | ✅      | ✅         | ❌        | ❌     | ✅*      | ✅*      | ✅**         | ❌     |
# | `--mode all`   | ✅     | ✅      | ✅         | ✅        | ✅     | ✅*      | ✅*      | ✅**         | ✅     |
def main():
    # 명령행 옵션을 처리할 Parser 객체를 생성합니다.
    # description은 --help를 실행했을 때 표시되는 설명입니다.
    parser = argparse.ArgumentParser(description="LoRA Style Tuning: RAFT 실습 스크립트")
    # 새로운 실행 옵션을 등록합니다.
    parser.add_argument(
        # --mode라는 옵션을 만듭니다.
        "--mode",
        # 사용 가능한 값은 3개만 허용합니다.
        # 다른 값을 입력하면 오류가 발생합니다.
        choices=["all", "train", "eval"],
        # --mode를 생략하면 기본값으로 "all"을 사용합니다.
        default="all",
        # --help를 입력했을 때 사용자에게 보여주는 설명입니다.
        help="all: 학습+평가+저장 / train: 학습만 / eval: 저장된 어댑터로 평가만",
    )
    # 사용자가 입력한 명령행 옵션을 읽어서 args에 저장합니다.
    args = parser.parse_args()

    # Hugging Face 로그인
    hf_login_if_needed()

    # 실행 모드가 eval인지 확인합니다.
    if args.mode == "eval":
        # 저장된 LoRA 어댑터가 있는지 확인합니다.
        if not adapter_already_saved():
            # LoRA가 없으면 오류 메시지를 출력하고 프로그램을 종료합니다.
            sys.exit(
                f"[ERROR] 저장된 LoRA 어댑터를 찾을 수 없습니다: {ADAPTER_DIR}\n"
                "먼저 --mode train 또는 --mode all 로 학습을 진행해 주세요."
            )
        # 원본 모델과 토크나이저를 로드합니다.
        # 여기서 _는 모델은 사용하지 않겠다는 의미입니다.
        #     -> 왜냐하면 곧바로 아래에서 'model = ...' 이런식으로
        #         다시 model 변수에 저장하므로 기존 모델 변수는 필요 없습니다.
        _, tokenizer = load_model_and_tokenizer()
        # 원본 Gemma 모델에 저장된 LoRA를 연결하여 학습된 모델을 복원합니다.
        model = load_adapter_for_eval(tokenizer)
        # 복원된 모델로 테스트 질문을 생성해 성능을 평가합니다.
        evaluate_model(model, tokenizer)
        return

    # train 또는 all 모드
    # 아래 코드들은 원본 모델과 학습 데이터를 불러온 뒤, 
    # 모든 데이터를 prompt와 completion 형태로 변환하여 
    # SFT 학습이 가능한 데이터셋을 만드는 과정입니다.

    # 원본 Gemma 모델과 토크나이저를 불러옵니다.
    #     model : 학습할 모델
    #     tokenizer : 문장을 토큰으로 변환
    model, tokenizer = load_model_and_tokenizer()
    # CSV 파일을 읽어 학습 데이터를 메모리에 로드합니다.
    data = load_training_data()
    # 데이터의 모든 행(row) 에 동일한 함수를 적용합니다.
    #     lambda x:
    #         -> 현재 처리 중인 한 개의 데이터(row) 를 의미합니다.
    #         -> 한개의 데이터: x = { "context": "...", "question": "...", "cot": "..." }
    #     build_prompt_completion    
    #         -> 각 데이터를 학습에 사용할 Prompt와 Completion 형태로 변환합니다.
    data = data.map(
        lambda x: build_prompt_completion(tokenizer, x["context"], x["question"], x["cot"])
    )

    if args.mode == "all":
        # 튜닝 전 원본 모델의 답변을 확인합니다.
        # 나중에 튜닝 후 결과와 비교하기 위한 기준(Baseline)입니다.
        run_baseline_check(model, tokenizer, data)
        # 학습 데이터의 토큰 길이 분포를 분석합니다.
        #     평균 길이
        #     최대 길이
        #     95% 구간
        #     히스토그램
        # 등을 확인합니다.
        analyze_token_distribution(tokenizer, data["train"])

    # 이미 학습된 LoRA 어댑터가 있는지 확인합니다.
    if adapter_already_saved():
        print(
            f"[INFO] 이미 학습된 어댑터가 존재합니다: {ADAPTER_DIR}\n"
            "재학습을 원하면 해당 폴더를 삭제한 뒤 다시 실행해 주세요. 학습을 건너뜁니다."
        )
        # 기존에 저장된 LoRA를 불러와 모델을 복원합니다.
        model = load_adapter_for_eval(tokenizer)
    else:
        # 처음 학습하는 경우입니다.
        # 원본 Gemma 모델에 LoRA를 적용합니다.
        model = apply_lora(model)
        # SFT 학습을 수행하여 LoRA 가중치를 학습합니다.
        model = train_model(model, tokenizer, data)
        # 학습된 LoRA 어댑터를 저장합니다.
        save_model(model, tokenizer)

    if args.mode == "all":
        # 튜닝 후 성능 평가
        evaluate_model(model, tokenizer)


if __name__ == "__main__":
    main()