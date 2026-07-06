"""
[실습 13] Continuous Pretraining - 도메인 지식 주입

목적: 사전학습된 sLLM에 의료 코퍼스를 추가 학습시켜 의료 도메인 지식을 주입
환경: GTX 1650 (VRAM 4GB), Windows, Python 3.11, CUDA 12.1

실행 전 준비:
  1. .env 파일 생성 (HF_TOKEN 입력)
  2. medical_corpus.json 을 이 파일과 같은 폴더에 위치
  3. pip install -r requirements.txt
"""

# -----------------------------------------------------------------------
# 표준 라이브러리
# -----------------------------------------------------------------------
import json
import os
import sys

# -----------------------------------------------------------------------
# 서드파티 라이브러리
# -----------------------------------------------------------------------
import matplotlib
import numpy as np
import seaborn as sns
import torch
from datasets import Dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# -----------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(override=True)

# VRAM 단편화 방지
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CORPUS_PATH    = os.path.join(BASE_DIR, "medical_corpus.json")
HF_CACHE_DIR   = os.path.join(BASE_DIR, "hf_cache")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")
MODEL_ID       = "unsloth/gemma-3-1b-pt"
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "gemma-3-1b-pt-MED")

# GTX 1650 (4GB VRAM) 맞춤 학습 파라미터
MAX_SEQ_LENGTH             = 512
OVERLAP_SIZE               = 128
BATCH_SIZE                 = 1
GRADIENT_ACCUMULATION      = 16   # 실질 배치 16
NUM_EPOCHS                 = 3
LEARNING_RATE              = 5e-5

# 추론 파라미터
MAX_NEW_TOKENS = 300

EVAL_QUESTIONS = [
    "질문: 고혈압의 약물 치료는 어떻게 해야 하나요?\n답변:",
    "질문: 축농증의 수술이 필요한 경우는 언제인가요?\n답변:",
    "질문: 불안장애 환자의 식이의 특징은 무엇입니까?\n답변:",
    "질문: 갑상선 기능 항진증 치료에 사용하는 대표적인 약물들은?\n답변:",
]

for d in [HF_CACHE_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)


# -----------------------------------------------------------------------
# 헬퍼 함수
# -----------------------------------------------------------------------
def check_env():
    """실행 전 환경 점검"""
    if not os.path.exists(CORPUS_PATH):
        print(f"[오류] 학습 데이터 없음: {CORPUS_PATH}")
        print("medical_corpus.json 을 스크립트와 같은 폴더에 넣어주세요.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[오류] CUDA GPU를 찾을 수 없습니다.")
        sys.exit(1)

    name = torch.cuda.get_device_name(0)
    cc   = torch.cuda.get_device_capability(0)
    print(f"[INFO] GPU: {name}  (Compute Capability {cc[0]}.{cc[1]})")
    return cc


def hf_login():
    """HuggingFace 로그인 (토큰이 있을 때만)"""
    token = os.getenv("HF_TOKEN", "")
    if token and token != "your_token_here":
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("[INFO] HuggingFace 로그인 완료")


def load_model_and_tokenizer(dtype):
    """모델·토크나이저 로드 (hf_cache 재사용으로 중복 다운로드 방지)"""
    print(f"[INFO] 모델 로드: {MODEL_ID}  dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE_DIR
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map={"": 0},   # GPU 0 에 명시적으로 단일 배치
        attn_implementation="eager",
        cache_dir=HF_CACHE_DIR,
    )
    print(f"[INFO] 모델 로드 완료 — 파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    return model, tokenizer


def generate_answer(model, tokenizer, prompt: str, dtype) -> str:
    """단일 프롬프트에 대해 모델 답변 생성"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=64,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )
    # 프롬프트 부분 제거 후 답변만 반환
    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def run_evaluation(model, tokenizer, dtype, label: str):
    """학습 전/후 동일한 질문으로 모델 출력 비교"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)
    model.eval()
    torch.cuda.empty_cache()
    for q in EVAL_QUESTIONS:
        print(f"\n{q}")
        answer = generate_answer(model, tokenizer, q, dtype)
        print(answer)
        print("-" * 40)


def plot_token_distribution(lengths: list, title: str, save_path: str):
    """토큰 길이 분포 히스토그램 저장"""
    stats = {
        "평균":      np.mean(lengths),
        "중앙값":    np.median(lengths),
        "최소":      min(lengths),
        "최대":      max(lengths),
        "표준편차":  np.std(lengths),
        "90퍼센타일": np.percentile(lengths, 90),
        "95퍼센타일": np.percentile(lengths, 95),
        "99퍼센타일": np.percentile(lengths, 99),
        "샘플 수":   len(lengths),
    }
    print(f"\n=== 토큰 분포 통계 [{title}] ===")
    for k, v in stats.items():
        print(f"  {k}: {v:.1f}")

    if os.path.exists(save_path):
        print(f"[INFO] 그래프 이미 존재, 생략: {save_path}")
    else:
        plt.figure(figsize=(10, 5))
        sns.histplot(lengths, bins=30, kde=True)
        plt.title(f"Token Length Distribution — {title}")
        plt.xlabel("Token Count")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] 그래프 저장: {save_path}")


CHUNK_CACHE_PATH = os.path.join(BASE_DIR, "chunked_dataset_cache.json")


def prepare_dataset(tokenizer) -> Dataset:
    """
    medical_corpus.json 로드 → 토큰화 → 오버랩 청킹
    - 청킹 결과는 chunked_dataset_cache.json 에 저장
    - 재실행 시 캐시 파일이 있으면 청킹 과정 생략 (시간 절약)
    """
    # 캐시가 있으면 바로 로드
    if os.path.exists(CHUNK_CACHE_PATH):
        print(f"\n[INFO] 청킹 캐시 발견 → 재사용: {CHUNK_CACHE_PATH}")
        with open(CHUNK_CACHE_PATH, encoding="utf-8") as f:
            all_input_ids = json.load(f)
        print(f"[INFO] 캐시에서 청크 {len(all_input_ids)}건 로드 완료")
        return Dataset.from_dict({"input_ids": all_input_ids})

    # 캐시 없으면 새로 처리
    print("\n[INFO] 학습 데이터 로드 중...")
    with open(CORPUS_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    # generated_text 필드 추출 (리스트 또는 딕셔너리 모두 대응)
    if isinstance(raw, list):
        texts = [item["generated_text"] for item in raw if "generated_text" in item]
    else:
        texts = [raw["generated_text"]]

    print(f"[INFO] 원본 문서 수: {len(texts)}")

    # 원본 토큰 길이 분포
    orig_lengths = [len(tokenizer.encode(t)) for t in texts]
    plot_token_distribution(
        orig_lengths, "원본 데이터",
        os.path.join(BASE_DIR, "token_dist_original.png")
    )

    # 오버랩 청킹
    print(f"\n[INFO] 오버랩 청킹 (max={MAX_SEQ_LENGTH}, overlap={OVERLAP_SIZE})...")
    stride = MAX_SEQ_LENGTH - OVERLAP_SIZE
    all_input_ids = []

    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        for i in range(0, len(ids), stride):
            chunk = ids[i: i + MAX_SEQ_LENGTH]
            if len(chunk) > 10:   # 너무 짧은 조각 제외
                all_input_ids.append(chunk)

    chunk_lengths = [len(ids) for ids in all_input_ids]
    plot_token_distribution(
        chunk_lengths, "청킹 후 데이터",
        os.path.join(BASE_DIR, "token_dist_chunked.png")
    )
    print(f"[INFO] 원본 {len(texts)}건 → 청크 {len(all_input_ids)}건")

    # 청킹 결과 캐시 저장 (다음 실행 시 재사용)
    with open(CHUNK_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_input_ids, f)
    print(f"[INFO] 청킹 캐시 저장: {CHUNK_CACHE_PATH}")

    dataset = Dataset.from_dict({"input_ids": all_input_ids})
    return dataset


# -----------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------
def main():
    # 1. 환경 점검
    cc = check_env()
    use_fp16  = cc[0] < 8   # Ampere(sm_80) 미만이면 fp16
    dtype     = torch.float16 if use_fp16 else torch.bfloat16
    print(f"[INFO] 학습 dtype: {'fp16' if use_fp16 else 'bf16'}")

    # 2. HuggingFace 로그인
    hf_login()

    # 3. 모델 로드
    model, tokenizer = load_model_and_tokenizer(dtype)

    # 4. 학습 전 평가
    run_evaluation(model, tokenizer, dtype, "[학습 전] 베이스 모델 출력")

    # 5. 데이터셋 준비
    dataset = prepare_dataset(tokenizer)

    # 6. 학습
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,   # Causal LM (다음 토큰 예측)
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=use_fp16,
        bf16=not use_fp16,
        optim="adamw_torch",
        logging_steps=50,
        save_strategy="no",           # 체크포인트 저장 안 함 (VRAM/디스크 절약)
        report_to="none",             # TensorBoard 등 외부 리포트 비활성화
        dataloader_pin_memory=False,  # VRAM 절약
    )

    print("\n[INFO] 학습 시작...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,   # FutureWarning 방지 (tokenizer= 대신 사용)
    )
    trainer.train()
    print("[INFO] 학습 완료.")

    # 7. 학습 후 평가
    run_evaluation(model, tokenizer, dtype, "[학습 후] 파인튜닝 모델 출력")

    # 8. 모델 저장 (이미 있으면 생략)
    if os.path.exists(MODEL_SAVE_DIR) and os.listdir(MODEL_SAVE_DIR):
        print(f"\n[INFO] 저장된 모델이 이미 있습니다: {MODEL_SAVE_DIR} (생략)")
    else:
        print(f"\n[INFO] 모델 저장 중: {MODEL_SAVE_DIR}")
        model.save_pretrained(MODEL_SAVE_DIR, safe_serialization=False)
        tokenizer.save_pretrained(MODEL_SAVE_DIR)
        print("[INFO] 저장 완료.")

    print("\n[완료] Continuous Pretraining 파이프라인 종료.")


if __name__ == "__main__":
    main()