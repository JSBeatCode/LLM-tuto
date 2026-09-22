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

from __future__ import annotations

# ── 표준 라이브러리 ──────────────────────────────────────────────
import argparse
import os
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
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["MPLCONFIGDIR"] = MPL_CACHE_DIR

# ── 서드파티 라이브러리 (알파벳 순) ───────────────────────────────
import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경에서 그래프를 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from accelerate import Accelerator
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

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

GEN_CONFIG = dict(
    do_sample=True,
    max_new_tokens=1024,
    top_p=0.95,
    top_k=64,
)

RESPONSE_TEMPLATE = "<start_of_turn>model"


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
def load_model_and_tokenizer():
    """
    이미 HF_HUB_CACHE 에 모델이 캐시되어 있다면 huggingface_hub 가
    자동으로 재사용하므로 별도의 중복 다운로드 방지 로직이 필요 없다.
    (from_pretrained 는 캐시 존재 여부를 자체적으로 확인한다.)
    """
    torch.set_float32_matmul_precision("high")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"[INFO] 모델 로드 중: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )
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
def analyze_token_distribution(tokenizer, dataset, text_column="text", bins=30):
    token_counts = [len(tokenizer.encode(text)) for text in dataset[text_column]]

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
    plt.title(f"Token Length Distribution for {text_column}")
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
def apply_lora(model):
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ══════════════════════════════════════════════════════════════
# 8. SFT 학습
# ══════════════════════════════════════════════════════════════
def train_model(model, tokenizer, data):
    sft_config = SFTConfig(
        report_to="none",
        num_train_epochs=2,
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_seq_length=1100,
        lr_scheduler_type="cosine",
        learning_rate=1e-4,
        warmup_ratio=0.03,
        bf16=True,
        optim="paged_adamw_8bit",
        output_dir=OUTPUT_DIR,
        logging_steps=25,
    )

    collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        args=sft_config,
        data_collator=collator,
    )

    accelerator = Accelerator()
    with accelerator.main_process_first():
        trainer.train()

    return model


# ══════════════════════════════════════════════════════════════
# 9. 학습 결과 평가
# ══════════════════════════════════════════════════════════════
def evaluate_model(model, tokenizer):
    model.eval()
    model = model.to(torch.bfloat16)  # 양자화 + LoRA 조합에서 dtype 불일치 방지

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        **GEN_CONFIG,
    )
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
def adapter_already_saved() -> bool:
    """LoRA 어댑터가 이미 저장되어 있는지 sentinel 파일로 확인 (재학습 방지)."""
    sentinel = os.path.join(ADAPTER_DIR, "adapter_config.json")
    return os.path.exists(sentinel)


def save_model(model, tokenizer):
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"[INFO] LoRA 어댑터 저장 완료: {ADAPTER_DIR}")

    if HF_PUSH_TO_HUB:
        if not HF_HUB_USERNAME:
            print("[WARN] HF_HUB_USERNAME 이 비어 있어 허브 업로드를 건너뜁니다.")
            return
        repo_id = f"{HF_HUB_USERNAME}/{os.path.basename(ADAPTER_DIR)}"
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print(f"[INFO] 허브 업로드 완료: {repo_id}")
    else:
        print("[INFO] HF_PUSH_TO_HUB=false 이므로 허브 업로드를 건너뜁니다.")


def load_adapter_for_eval(tokenizer):
    """이미 저장된 LoRA 어댑터를 베이스 모델에 로드한다 (재학습 없이 평가만 수행)."""
    from peft import PeftModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    return model


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="LoRA Style Tuning: RAFT 실습 스크립트")
    parser.add_argument(
        "--mode",
        choices=["all", "train", "eval"],
        default="all",
        help="all: 학습+평가+저장 / train: 학습만 / eval: 저장된 어댑터로 평가만",
    )
    args = parser.parse_args()

    hf_login_if_needed()

    if args.mode == "eval":
        if not adapter_already_saved():
            sys.exit(
                f"[ERROR] 저장된 LoRA 어댑터를 찾을 수 없습니다: {ADAPTER_DIR}\n"
                "먼저 --mode train 또는 --mode all 로 학습을 진행해 주세요."
            )
        _, tokenizer = load_model_and_tokenizer()
        model = load_adapter_for_eval(tokenizer)
        evaluate_model(model, tokenizer)
        return

    # train 또는 all
    model, tokenizer = load_model_and_tokenizer()
    data = load_training_data()
    data = data.map(
        lambda x: convert_format_without_pe(tokenizer, x["context"], x["question"], x["cot"])
    )

    if args.mode == "all":
        run_baseline_check(model, tokenizer, data)
        analyze_token_distribution(tokenizer, data["train"])

    if adapter_already_saved():
        print(
            f"[INFO] 이미 학습된 어댑터가 존재합니다: {ADAPTER_DIR}\n"
            "재학습을 원하면 해당 폴더를 삭제한 뒤 다시 실행해 주세요. 학습을 건너뜁니다."
        )
        model = load_adapter_for_eval(tokenizer)
    else:
        model = apply_lora(model)
        model = train_model(model, tokenizer, data)
        save_model(model, tokenizer)

    if args.mode == "all":
        evaluate_model(model, tokenizer)


if __name__ == "__main__":
    main()
