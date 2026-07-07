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

        # 내 컴퓨터의 GPU를 조사해서, 그 GPU에 맞는 최적의 학습 옵션을 자동으로 선택하는 코드
    
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

    # GPU가 행렬(Matrix) 계산을 더 빠르고 효율적으로 하도록 설정하는 코드
    # "highest" : 가장 정확하지만 느릴 수 있음
    # "high" : 정확도는 거의 유지하면서 더 빠름 ✅ (가장 많이 사용)
    # "medium" : 더 빠르지만 정확도가 조금 더 낮을 수 있음
    # 왜 사용하는가? -> 학습 속도를 높이기 위해서입니다.
    torch.set_float32_matmul_precision("high")

    # 모델/토크나이저 캐시를 .py 파일과 같은 위치의 hf_cache 폴더에 저장
    HF_CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    # "안녕하세요"
    # ↓
    # Tokenizer
    # ↓
    # [912, 4312, 77]
    # 이처럼 문자를 숫자(Token ID)로 변환해야 모델이 이해
    # Gemma  → Gemma Tokenizer
    # Llama  → Llama Tokenizer
    # Qwen   → Qwen Tokenizer
    # 처럼 자동으로 맞는 Tokenizer를 가져옵니다.
    # from_pretrained() -> 이미 학습되어 공개된 모델의 Tokenizer를 불러오는 함수
    print("[INFO] 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_DIR)

    print("[INFO] 모델 로드 중... (최초 1회만 다운로드 → hf_cache 폴더에 저장, 이후 재사용)")
    # 자동으로 Gemma 모델을 다운로드(또는 캐시에서 읽어서) GPU에 로드합니다.
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
    gen_config = dict(
        do_sample=True,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        repetition_penalty=1.05,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        **gen_config,
    )
    base_llm = HuggingFacePipeline(pipeline=pipe)

    eval_inputs = [
        "질문: 고혈압의 약물 치료는 어떻게 해야 하나요? \n답변:",
        "질문: 축농증의 수술이 필요한 경우는 언제인가요? \n답변:",
        "질문: 불안장애 환자의 식이의 특징은 무엇입니까? \n답변:",
        "질문: 갑상선 기능 항진증 치료에 사용하는 대표적인 약물들은? \n답변:",
    ]

    print("\n========== [학습 전] 베이스 모델 출력 ==========")
    for q in eval_inputs:
        print(f"{q}", end="")
        for s in base_llm.stream(q):
            print(s, end="")
        print("\n-------")

    # -------------------------------------------------------------------
    # 6. 학습 데이터 로드
    #    - Continuous Pretraining은 새로운 지식 습득이 목적이므로 train/test split 불필요
    # -------------------------------------------------------------------
    print("\n[INFO] 학습 데이터 로드 중...")
    data = load_dataset("json", data_files={"train": [CORPUS_PATH]})
    data = data.shuffle()
    print(data)
    print("샘플 예시:", data["train"][0]["generated_text"][:200], "...")

    # -------------------------------------------------------------------
    # 7. 학습 데이터 토큰 길이 분포 분석
    # -------------------------------------------------------------------
    def analyze_token_distribution(dataset, text_column="generated_text", bins=30, tokenized=False, save_path=None):
        token_counts = []
        if not tokenized:
            for text in dataset[text_column]:
                tokens = tokenizer.encode(text)
                token_counts.append(len(tokens))
        else:
            for tokens in dataset["input_ids"]:
                token_counts.append(len(tokens))

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

        if save_path:
            plt.savefig(save_path)
            print(f"[INFO] 토큰 분포 그래프 저장: {save_path}")
        plt.close()

        print("\n=== 토큰 수 통계 ===")
        for key, value in stats.items():
            print(f"{key}: {value:.1f}")

        return stats

    print("\n[INFO] 원본 데이터 토큰 분포 분석 중...")
    analyze_token_distribution(
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

    def chunk_text_with_overlap(examples, tokenizer, max_seq_length, text_field, overlap):
        """
        데이터셋의 각 텍스트를 토큰화하고 오버랩을 적용하여 max_seq_length 길이의 청크로 나눕니다.
        """
        if overlap >= max_seq_length:
            raise ValueError("Overlap size must be smaller than max_seq_length")

        stride = max_seq_length - overlap
        texts = examples[text_field]

        all_token_ids = []
        for text in texts:
            tokenized_output = tokenizer(text, add_special_tokens=False, truncation=False, padding=False)
            all_token_ids.append(tokenized_output["input_ids"])

        chunked_input_ids = []
        chunked_attention_mask = []

        for token_ids in all_token_ids:
            for i in range(0, len(token_ids), stride):
                chunk = token_ids[i : i + max_seq_length]
                if len(chunk) > 0:
                    chunked_input_ids.append(chunk)
                    chunked_attention_mask.append([1] * len(chunk))

        return {
            "input_ids": chunked_input_ids,
            "attention_mask": chunked_attention_mask,
        }

    print("\n[INFO] 오버랩 청킹 진행 중...")
    chunked_dataset_overlap = data["train"].map(
        chunk_text_with_overlap,
        batched=True,
        remove_columns=data["train"].column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": max_seq_length,
            "text_field": text_field_name,
            "overlap": overlap_size,
        },
    )

    print(f"Original dataset size: {len(data['train'])}")
    print(f"Chunked dataset size (with overlap): {len(chunked_dataset_overlap)}")

    print("\n[INFO] 청킹된 데이터 토큰 분포 분석 중...")
    analyze_token_distribution(
        chunked_dataset_overlap,
        tokenized=True,
        save_path=os.path.join(BASE_DIR, "token_distribution_chunked.png"),
    )

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
    model.config.use_cache = False  # checkpointing 사용 시 use_cache는 False

    tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰을 eos 토큰으로 설정

    # DataCollator: input_ids를 그대로 labels로 복사 (CLM 방식, mlm=False)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM (GPT 방식) → 다음 토큰 예측
    )

    training_args = TrainingArguments(
        report_to="tensorboard",           # 텐서보드로 학습 모니터링
        num_train_epochs=3,                # 전체 데이터 3회 반복 학습
        per_device_train_batch_size=2,     # 1 → 2 (VRAM 20GB 여유로 복원)
        gradient_accumulation_steps=8,    # 16 → 8 (배치 2로 늘린 만큼 조정, 실질 Batch 16 유지)
        lr_scheduler_type="cosine",
        learning_rate=5e-5,               # 기본 Pretrain보다 낮은 학습률
        warmup_ratio=0.03,
        bf16=use_bf16,                    # Ada Lovelace → bf16=True
        fp16=use_fp16,                    # Ada Lovelace → fp16=False
        optim=optimizer,                  # Linux → paged_adamw_8bit
        output_dir=OUTPUT_DIR,
        logging_steps=25,
        save_total_limit=1,               # 체크포인트를 최대 1개만 유지
        overwrite_output_dir=True,        # output_dir 재실행 시 덮어쓰기
        dataloader_pin_memory=True,       # 1 → 복원 (VRAM 여유, 데이터 로딩 속도 향상)
    )

    print("\n[INFO] 학습 시작...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=chunked_dataset_overlap,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    print("[INFO] 학습 완료.")
    print(f"[INFO] TensorBoard 확인: tensorboard --logdir {os.path.join(OUTPUT_DIR, 'runs')} --port 2025 --bind_all")

    # -------------------------------------------------------------------
    # 10. 학습 후 평가
    # -------------------------------------------------------------------
    model.eval()
    torch.cuda.empty_cache()

    print("\n========== [학습 후] 파인튜닝 모델 출력 ==========")
    for q in eval_inputs:
        print(f"{q}", end="")
        for s in base_llm.stream(q):
            print(s, end="")
        print("\n-------")

    # -------------------------------------------------------------------
    # 11. 모델 저장 (로컬) - 이미 저장된 경우 재저장 방지
    # -------------------------------------------------------------------
    if os.path.exists(MODEL_SAVE_DIR) and os.listdir(MODEL_SAVE_DIR):
        print(f"\n[INFO] 모델이 이미 저장되어 있습니다: {MODEL_SAVE_DIR} (재저장 생략)")
    else:
        print(f"\n[INFO] 모델 저장 중: {MODEL_SAVE_DIR}")
        model.save_pretrained(MODEL_SAVE_DIR, safe_serialization=False)
        tokenizer.save_pretrained(MODEL_SAVE_DIR)
        print("[INFO] 모델 저장 완료.")

    # -------------------------------------------------------------------
    # 12. (선택) HuggingFace Hub 업로드
    #     - HF_WRITE_TOKEN과 HF_USERNAME이 .env에 설정된 경우에만 실행
    # -------------------------------------------------------------------
    hf_write_token = os.getenv("HF_WRITE_TOKEN")
    hf_username = os.getenv("HF_USERNAME")

    if hf_write_token and hf_username:
        print(f"\n[INFO] HuggingFace Hub 업로드 중: {hf_username}/{os.path.basename(MODEL_SAVE_DIR)}")
        login(token=hf_write_token)
        repo_id = f"{hf_username}/{os.path.basename(MODEL_SAVE_DIR)}"
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print("[INFO] 업로드 완료.")
    else:
        print("\n[INFO] HF_WRITE_TOKEN / HF_USERNAME 미설정 - Hub 업로드 생략")

    print("\n[완료] Continuous Pretraining 파이프라인 종료.")


if __name__ == "__main__":
    main()