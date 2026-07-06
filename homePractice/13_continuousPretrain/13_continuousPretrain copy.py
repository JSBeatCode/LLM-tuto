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
import matplotlib
import numpy as np
import seaborn as sns
import torch
from accelerate import Accelerator
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import SFTConfig, SFTTrainer

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

    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)  # (major, minor) 튜플
    use_bf16 = compute_capability[0] >= 8  # Ampere(sm_80) 이상이면 bf16 지원
    use_fp16 = not use_bf16                # Turing(sm_75) 이하면 fp16 사용
    # bitsandbytes는 Windows에서 미지원 → adamw_torch로 대체
    optimizer = "adamw_torch"

    print(f"[INFO] GPU: {gpu_name} (Compute Capability: {compute_capability[0]}.{compute_capability[1]})")
    if use_bf16:
        print("[INFO] 정밀도: bf16 (Ampere 이상 지원)")
    else:
        print("[INFO] 정밀도: fp16 (Turing 이하, bf16 미지원 → fp16으로 대체)")
    print(f"[INFO] 옵티마이저: {optimizer} (bitsandbytes Windows 미지원 → adamw_torch 사용)")

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
    #    - 이미 로컬 캐시(~/.cache/huggingface)에 있으면 자동으로 재사용됨
    #      (중복 다운로드 방지는 huggingface_hub 라이브러리가 캐시로 처리)
    # -------------------------------------------------------------------
    print(f"## MODEL: {MODEL_NAME}")

    torch.set_float32_matmul_precision("high")

    print("[INFO] 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("[INFO] 모델 로드 중... (최초 1회는 다운로드, 이후 캐시에서 로드)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager",
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
    # 9. 학습 파라미터 설정 및 SFTTrainer 구성
    # -------------------------------------------------------------------
    # GPU 메모리 절약을 위한 Gradient Checkpointing
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # checkpointing 사용 시 use_cache는 False

    tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰을 eos 토큰으로 설정

    sft_config = SFTConfig(
        report_to="tensorboard",           # 텐서보드로 학습 모니터링
        num_train_epochs=3,                # 전체 데이터 3회 반복 학습
        dataset_text_field="generated_text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,    # 실질적으로 Batch 16 효과
        max_seq_length=max_seq_length,    # 2500
        lr_scheduler_type="cosine",
        learning_rate=5e-5,               # 기본 Pretrain보다 낮은 학습률
        warmup_ratio=0.03,
        bf16=use_bf16,                    # Ampere 이상이면 bf16, 아니면 False
        fp16=use_fp16,                    # Turing 이하이면 fp16, 아니면 False
        optim=optimizer,                  # Windows: adamw_torch / Linux+GPU: paged_adamw_8bit 가능
        output_dir=OUTPUT_DIR,
        logging_steps=25,
    )

    print("\n[INFO] 학습 시작...")
    accelerator = Accelerator()
    trainer = SFTTrainer(
        model=model,
        train_dataset=chunked_dataset_overlap,
        args=sft_config,
    )

    with accelerator.main_process_first():
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