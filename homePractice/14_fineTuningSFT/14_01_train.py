"""
[실습] Instruction Tuning - train.py
- 베이스 모델 : NotoriousH2/gemma-3-1b-pt-MED (의료 CPT 완료 모델)
- 채팅 템플릿 : unsloth/gemma-3-1b-it 토크나이저
- 학습 데이터 : med_qa_data.json (의료 QA) + beomi/KoAlpaca-RealQA (한국어 일반 QA)
- 학습 방법  : SFTTrainer + DataCollatorForCompletionOnlyLM (답변 부분만 학습)
- 실행 결과  : outputs/ (체크포인트), gemma-3-1b-pt-MED-Instruct/ (최종 모델)
"""


# | 라이브러리             | 역할              | 이 코드에서 하는 일                |
# | ----------------- | --------------- | -------------------------- |
# | `os`              | 운영체제 기능         | 파일 경로 관리                   |
# | `dotenv`          | 환경변수            | HF_TOKEN 읽기                |
# | `matplotlib`      | 그래프             | 토큰 분포 저장                   |
# | `numpy`           | 수학 계산           | 평균, 중앙값, 백분위 계산            |
# | `seaborn`         | 시각화             | 토큰 히스토그램                   |
# | `torch`           | 딥러닝 엔진          | 모델 학습과 GPU 연산              |
# | `accelerate`      | 학습 환경 관리        | GPU/멀티 GPU 지원              |
# | `datasets`        | 데이터셋 관리         | JSON 로드, 데이터 병합            |
# | `huggingface_hub` | Hugging Face 연동 | 로그인, 모델 다운로드               |
# | `transformers`    | LLM 핵심 라이브러리    | 모델과 토크나이저 로드               |
# | `trl`             | LLM SFT/RLHF    | Instruction Tuning(SFT) 실행 |

# ── 표준 라이브러리 ──────────────────────────────────────────────────────────
import os

# ── 서드파티: 환경변수 ────────────────────────────────────────────────────────
from dotenv import load_dotenv

# ── 서드파티: 시각화 ──────────────────────────────────────────────────────────
# 토큰 분포 저장 그래프
import matplotlib.pyplot as plt
# 평균, 중앙값, 백분위 계산
import numpy as np
# 시각화 토큰 히스토그램
import seaborn as sns

# ── 서드파티: HuggingFace ─────────────────────────────────────────────────────
# 모델 학습과 GPU 연산
import torch
# GPU/멀티 GPU 지원
from accelerate import Accelerator

# datasets는 Hugging Face에서 만든 라이브러리이다.
# concatenate_datasets:
#     두 데이터셋을 붙인다.
#         의료 QA
#         +
#         KoAlpaca
#         ↓
#         최종 train
#     이렇게 만든다.
# DatasetDict
#     ├── train
#     └── test
#     이런 구조를 가진다.
from datasets import DatasetDict, concatenate_datasets, load_dataset
# snapshot_download -> 모델을 로컬에 저장한다.
from huggingface_hub import login, snapshot_download
# AutoTokenizer
#     안녕하세요
#     ↓
#     [534, 91, 778]
    
# AutoModelForCausalLM
#     예를 들어
#         나는 오늘 밥을
#         ↓
#         먹었다
#     를 예측한다.
from transformers import AutoModelForCausalLM, AutoTokenizer

# TRL은 Transformer Reinforcement Learning의 약자
# SFTConfig: 학습 설정
# SFTTrainer: 아래를 전부 알아서 처리한다. 한 줄로 학습 시작할 수 있음. 
    # Dataset
    # ↓
    # Tokenizer
    # ↓
    # Loss 계산
    # ↓
    # Optimizer
    # ↓
    # Checkpoint 저장
from trl import SFTConfig, SFTTrainer

# trl 버전에 따라 DataCollatorForCompletionOnlyLM 위치가 다름
# - 구버전 (0.8.x 이하) : trl
# - 신버전 (0.9.x 이상) : trl.trainer
# DataCollator: 데이터를 모아서 하나의 Batch로 만드는 역할
# DataCollatorForCompletionOnlyLM이 바로 그 채점 선생님 역할을 하는 클래스다.
#     시험지를 채점한다고 생각하면 된다.
#         질문 = 문제
#         답변 = 학생이 쓴 답
#     채점 선생님은 문제는 채점하지 않고, 학생이 쓴 답만 채점한다.
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    from trl.trainer import DataCollatorForCompletionOnlyLM


# ─────────────────────────────────────────
# 0. 경로 설정 & 환경변수 로드
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 모델 저장 경로 (.py 옆에 저장)
BASE_MODEL_DIR      = os.path.join(BASE_DIR, "gemma-3-1b-pt-MED")        # 학습할 베이스 모델
INSTRUCT_TOK_DIR    = os.path.join(BASE_DIR, "gemma-3-1b-it-tokenizer")   # Chat Template용 토크나이저
FINAL_MODEL_DIR     = os.path.join(BASE_DIR, "gemma-3-1b-pt-MED-Instruct") # 최종 저장 모델

# 데이터 & 출력 경로 (.py 옆에 저장)
DATASET_CACHE_DIR   = os.path.join(BASE_DIR, "dataset_cache")  # HuggingFace 데이터셋 캐시
MED_QA_PATH         = os.path.join(BASE_DIR, "med_qa_data.json") # 의료 QA 데이터 (직접 준비)
OUTPUT_DIR          = os.path.join(BASE_DIR, "outputs")           # 체크포인트 저장 위치

load_dotenv(override=True)
HF_TOKEN = os.getenv("HF_TOKEN")  # beomi/KoAlpaca-RealQA 접근에 필요 (인증 필요 데이터셋)


# ─────────────────────────────────────────
# 1. HuggingFace 로그인
# ─────────────────────────────────────────
def login_huggingface():
    if HF_TOKEN:
        login(token=HF_TOKEN)
        print("[INFO] HuggingFace 로그인 완료")
    else:
        print("[WARNING] HF_TOKEN 없음 → KoAlpaca-RealQA 다운로드 시 오류 발생할 수 있습니다.")


# ─────────────────────────────────────────
# 2. 모델 & 토크나이저 로드
# ─────────────────────────────────────────
def load_model_and_tokenizer():
    base_model_id   = "NotoriousH2/gemma-3-1b-pt-MED"  # 의료 CPT 완료 베이스 모델
    instruct_tok_id = "unsloth/gemma-3-1b-it"           # Chat Template 확보용 (토크나이저만 사용)

    # ── 베이스 모델 다운로드 (최초 1회) ─────────────────────────────────────
    # 단순 폴더 존재 여부가 아닌 model.safetensors 로 완전 다운로드 여부 확인
    # (폴더만 생성되고 다운로드가 중단된 경우를 방지)
    base_model_complete = os.path.exists(os.path.join(BASE_MODEL_DIR, "model.safetensors"))
    if not base_model_complete:
        print(f"[INFO] 베이스 모델 다운로드 중 → {BASE_MODEL_DIR}")
        print(f"[INFO] 약 2GB 크기입니다. 네트워크 상태에 따라 시간이 걸릴 수 있습니다.")
        snapshot_download(repo_id=base_model_id, local_dir=BASE_MODEL_DIR)
    else:
        print(f"[INFO] 로컬 베이스 모델 발견, 다운로드 생략 → {BASE_MODEL_DIR}")

    # ── Chat Template용 토크나이저 다운로드 (최초 1회) ───────────────────────
    # 베이스 모델(gemma-3-1b-pt-MED)은 Chat Template이 없어서
    # Instruct 모델의 토크나이저에서 Chat Template만 가져옴
    instruct_tok_complete = os.path.exists(os.path.join(INSTRUCT_TOK_DIR, "tokenizer.json"))
    if not instruct_tok_complete:
        print(f"[INFO] Chat Template 토크나이저 다운로드 중 → {INSTRUCT_TOK_DIR}")
        snapshot_download(repo_id=instruct_tok_id, local_dir=INSTRUCT_TOK_DIR)
    else:
        print(f"[INFO] 로컬 Chat Template 토크나이저 발견, 다운로드 생략 → {INSTRUCT_TOK_DIR}")

    # ── 베이스 모델 로드 ─────────────────────────────────────────────────────
    # @@ 학습할 LLM 모델을 메모리(GPU/CPU)로 불러오는 코드 
    # LLM을 사용할 수 있는 상태로 메모리에 올리는 과정 전체
    print("[INFO] 베이스 모델 로드 중...")
    # AutoModelForCausalLM: LLM을 읽어오는 클래스 
    #     -> 모델을 직접 만들지 말고, 이미 만들어진 모델을 자동으로 찾아서 불러와라
    #     LLM은 크게 두 가지가 있다.
    #         ① 모델 구조(설계도)
    #         ② 학습된 가중치(Weight)
    #     Hugging Face가 이러한 것을 사용자가 몰라도 모델이 로드되도록 만든 것이 AutoModel
    # from_pretrained: 이미 학습되어 있는 모델을 불러오는 함수
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        # 숫자의 자료형(정밀도)를 자동으로 선택
        # | 자료형             |   메모리 | 속도 |                  정확도 |
        # | --------------- | ----: | -: | -------------------: |
        # | float32 (FP32)  | 4Byte | 느림 |                ★★★★★ |
        # | float16 (FP16)  | 2Byte | 빠름 |                ★★★★☆ |
        # | bfloat16 (BF16) | 2Byte | 빠름 | ★★★★★ (학습 안정성이 더 좋음) |
        # 비유하면 📷 사진 화질이라고 생각하면 쉽다.
        #     FP32 → 4K 사진 📸 (용량 큼, 아주 선명)
        #     FP16 → Full HD 📷 (용량 절반, 대부분 충분히 선명)
        #     BF16 → Full HD인데 AI 작업에 더 최적화된 형식 🤖
        torch_dtype="auto",
        # 모델을 어디에 올릴지 자동으로 결정해라.
        #     -> GPU가 있으면 GPU에 모델을 올리고 없으면 CPU, GPU가 여러대면 자동으로 나눠 올림
        device_map="auto",
        # attention:
        #     LLM은 문장을 읽을 때
        #         나는 오늘 학교에서 친구를 만났다.
        #     를 한 글자씩 읽는 것이 아니라
        #     단어들끼리
        #         누가 누구와 관련 있는가?
        #     를 계속 계산
        #     -> eager: 가장 기본적인 Attention 방식
        # | 방식                    | 특징                      |
        # | --------------------- | ----------------------- |
        # | **eager**             | 가장 안정적, 속도는 보통          |
        # | **sdpa**              | PyTorch 최적화, eager보다 빠름 |
        # | **flash_attention_2** | 가장 빠름, GPU 제약이 있음       |
        attn_implementation="eager",
        # attn_implementation="flash_attention_2"  # A100 이상 고성능 GPU에서 활성화
    )

    
    # ── Chat Template이 포함된 토크나이저 로드 ──────────────────────────────
    # @@ 모델은 의료 모델을 사용하지만, 토크나이저는 Instruct 모델의 토크나이저를 사용한다
    # 베이스 모델의 토크나이저를 먼저 확인하고, Chat Template이 없으면 Instruct 토크나이저 사용

    # tokenizer 객체 생성
    #     gemma-3-1b-it-tokenizer
    #         ↓
    #         tokenizer.json
    #         tokenizer_config.json
    #         special_tokens_map.json
    #         chat_template
    #     등을 메모리로 가져온다.
    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_TOK_DIR)
    print(f"[INFO] Chat Template 확인 (앞 50자): {str(tokenizer.chat_template)[:50]}")

    # 모델명만 출력
    #     "NotoriousH2/gemma-3-1b-pt-MED"
    #     ↓
    #     [
    #     "NotoriousH2",
    #     "gemma-3-1b-pt-MED"
    #     ]
    #     가 된다.
    model_name = base_model_id.split("/")[1]
    print(f"[INFO] 모델 로드 완료: {model_name}")
    return model, tokenizer, model_name


# ─────────────────────────────────────────
# 3. 학습 데이터 준비
# ─────────────────────────────────────────
def prepare_datasets():
    # ── 의료 QA 데이터 로드 ──────────────────────────────────────────────────
    # med_qa_data.json은 .py와 동일한 폴더에 직접 준비해야 합니다.
    if not os.path.exists(MED_QA_PATH):
        raise FileNotFoundError(
            f"[ERROR] 의료 QA 데이터 파일을 찾을 수 없습니다: {MED_QA_PATH}\n"
            f"        med_qa_data.json 파일을 .py와 동일한 폴더에 준비해 주세요."
        )

    print("[INFO] 의료 QA 데이터 로드 중...")
    # med_qa_data.json 의 데이터를 꺼내와서 medQA_data에 아래와 같이 담는다.
    #     DatasetDict
    #     │
    #     └── train
    #           ├── question
    #           └── answer
    medQA_data = load_dataset(
        "json",
        # MED_QA_PATH에 있는 JSON 파일을 학습(train) 데이터셋으로 읽어와
        # 결과:
        #     DatasetDict
        #     │
        #     └── train
        data_files={"train": [MED_QA_PATH]},
        # 다운로드한 데이터셋을 이 폴더에 저장(캐시)해 둬 -> 매번 인터넷에서 다시 다운로드하지 않는다.
        cache_dir=DATASET_CACHE_DIR,
    )
    
    # train 95% / test 5% 분할
    # 분할 후에는 medQA_data가 이렇게 바뀐다.
    #     DatasetDict
    #     │
    #     ├── train (95%)
    #     └── test  (5%)
    medQA_data = medQA_data["train"].train_test_split(test_size=0.05, seed=42)
    print(f"[INFO] 의료 QA - train: {len(medQA_data['train'])}개, test: {len(medQA_data['test'])}개")

    # ── 한국어 일반 QA 데이터 로드 (KoAlpaca-RealQA) ────────────────────────
    # 인증 필요 데이터셋: HF_TOKEN 로그인 필수
    # 일반 QA를 함께 학습하면 의료 특화로 인한 Catastrophic Forgetting 방지 가능
    print("[INFO] KoAlpaca-RealQA 데이터 로드 중...")
    
    # @@ 이 코드는 Hugging Face에서 beomi/KoAlpaca-RealQA 데이터셋을 다운로드한다.
    # @@ KoAlpaca-RealQA를 다운로드한 뒤 95:5로 분할하고, 의료 데이터와의 비율을 맞추기 위해 학습(train) 데이터는 1/3만 사용
    
    # 다운로드 후 realQA_data는
    #     DatasetDict
    #     │
    #     └── train (전체 데이터)
    realQA_data = load_dataset(
        "beomi/KoAlpaca-RealQA",
        cache_dir=DATASET_CACHE_DIR,
    )
    # Train / Test 분리
    #     95%
    #     ↓
    #     train
    #     5%
    #     ↓
    #     test
    # 로 나눈다.
    realQA_data = realQA_data["train"].train_test_split(test_size=0.05, seed=42)

    # 전체 중 1/3만 선택 (의료 데이터와 비율 균형 조절)
    # train 데이터를 3등분해서 첫 번째 조각만 사용한다.
    # 예를 들어 
    #     train 데이터
    #         1
    #         2
    #         3
    #         4
    #         5
    #         6
    #         7
    #         8
    #         9
    # 라면
    #     Shard 0
    #         1
    #         4
    #         7
    #     Shard 1
    #         2
    #         5
    #         8
    #     Shard 2
    #         3
    #         6
    #         9
    # 처럼 골고루 나눈다.
    # index=0 이므로 첫 번째 Shard만 사용한다.
    realQA_data["train"] = realQA_data["train"].shard(num_shards=3, index=0)
    print(f"[INFO] KoAlpaca-RealQA - train: {len(realQA_data['train'])}개 (1/3 선택), test: {len(realQA_data['test'])}개")

    # ── 두 데이터셋 합치기 ────────────────────────────────────────────────────
    merged_train = concatenate_datasets([realQA_data["train"], medQA_data["train"]])
    merged_test  = concatenate_datasets([realQA_data["test"],  medQA_data["test"]])

    # data 모양은 아래처럼 되게 된다.
    #     data
    #     │
    #     ├── train (8185개)
    #     └── test  (1050개)
    # shuffle -> 섞는다.
    #     만약 안 섞으면
    #         일반 QA
    #         일반 QA
    #         일반 QA
    #         ...
    #         (5866개)
    #         ----------------
    #         의료 QA
    #         의료 QA
    #         의료 QA
    #         ...
    #         (2319개)
    #     이런 순서로 학습한다.
    #     그러면 모델은
    #         "처음에는 일반 QA만 계속 학습하고, 나중에는 의료 QA만 계속 학습"
    # seed=42 
    #     -> 항상 같은 방식으로 섞어라. 안그러면 실험을 재현하기 어려움.
    data = DatasetDict({
        "train": merged_train.shuffle(seed=42),
        "test":  merged_test.shuffle(seed=42),
    })
    print(f"[INFO] 최종 데이터 - train: {len(data['train'])}개, test: {len(data['test'])}개")
    return data


# ─────────────────────────────────────────
# 4. Chat Template 변환
# ─────────────────────────────────────────
# @@ QA 데이터를 Gemma가 학습할 수 있는 대화(Chat) 형태로 바꿔주는 역할
#     원래 데이터는 이렇게 생겼다.
#         question
#             감기의 원인은?
#         answer
#             바이러스입니다.
#     하지만 Gemma는 이런 형식을 학습하지 않는다.
#     Gemma는
#         <start_of_turn>user
#             감기의 원인은?
#         <end_of_turn>
#         <start_of_turn>model
#             바이러스입니다.
#         <end_of_turn>
#     이런 채팅 형식(Chat Template) 을 학습한다.
def convert_to_chat_format(data, tokenizer):
    # Gemma Chat 형식으로 바꾼다.
    def convert_format(question, answer=None):
        # question, answer 세트(학습 시) 또는 question만 입력(추론 시)
        chat = [{"role": "user", "content": f"{question}"}]

        if answer:
            # 아래처럼 코딩하면
            #     [
            #         {
            #             "role":"user",
            #             "content":"감기의 원인은?"
            #         }
            #     ]
            # 에서 
            #     [
            #         {
            #             "role":"user",
            #             "content":"감기의 원인은?"
            #         },
            #         {
            #             "role":"assistant",
            #             "content":"바이러스입니다."
            #         }
            #     ]
            chat.append({"role": "assistant", "content": f"{answer}"})
            # 학습 데이터: Generation Prompt 없음 (answer까지 포함)
            # 방금 만든 Python 객체를 Gemma 형식 문자열로 바꾼다.
            # tokenize=False 
            #     -> 문자열만 만들라는 뜻이다.
            #     결과는
            #         "<bos><start_of_turn>..."
            #     이다.
            #     아직
            #         [123,456,789]
            #     같은 토큰으로 변환하지 않는다.
            #     토큰화는 나중에 SFTTrainer가 자동으로 수행한다.
            return {"text": tokenizer.apply_chat_template(chat, tokenize=False)}

        # 추론 데이터: add_generation_prompt=True (모델이 이어서 생성하도록)
        # if answer: -> 질문만 있으면, 
        # add_generation_prompt=True
        #     -> "이제 네 차례야. 답을 이어서 생성해." 
        return {"text": tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)}

    print("[INFO] Chat Template 변환 중...")
    # 데이터가
    # 1
    #     질문
    #     답변
    # 2
    #     질문
    #     답변
    # 3
    #     질문
    #     답변
    # 이라면, map()은 모든 데이터를 하나씩 꺼내서 convert_format(...)을 실행 함
    # 변환 전
    #     train
    #     │
    #     ├── question
    #     └── answer
    # ↓
    # 변환 후
    #     train
    #     │
    #     ├── question
    #     ├── answer
    #     └── text
    # text 컬럼이 추가
    #     text:
    #         <bos>
    #         <start_of_turn>user
    #             감기의 원인은?
    #         <end_of_turn>
    #         <start_of_turn>model
    #             바이러스입니다.
    data = data.map(lambda x: convert_format(x["question"], x["answer"]))
    print("[INFO] Chat Template 변환 완료")
    print(f"[예시] {data['train'][0]['text'][:200]}...")
    return data


# ─────────────────────────────────────────
# 5. 토큰 분포 분석
# ─────────────────────────────────────────
# 내 데이터가 평균 몇 토큰인지, 가장 긴 문장은 몇 토큰인지 분석하는 함수
# | 파라미터                 | 타입                  | 의미               | 이번 코드에서 전달되는 값  |
# | -------------------- | ------------------- | ---------------- | --------------- |
# | `dataset`            | HuggingFace Dataset | 분석할 데이터셋         | `data["train"]` |
# | `tokenizer`          | AutoTokenizer       | 문장을 토큰으로 변환하는 객체 | `tokenizer`     |
# | `text_column="text"` | 문자열                 | 토큰 길이를 계산할 컬럼 이름 | `"text"`        |
# | `bins=30`            | 정수                  | 히스토그램 막대 개수      | `30`            |
# --------------------------------------------------------------------------------------------------
# | 파라미터          | 쉽게 말하면                |
# | ------------- | --------------------- |
# | `dataset`     | **분석할 데이터**           |
# | `tokenizer`   | **문장을 토큰으로 세는 도구**    |
# | `text_column` | **어느 컬럼을 분석할 것인지**    |
# | `bins`        | **그래프를 몇 칸으로 나눌 것인지** |
def analyze_token_distribution(dataset, tokenizer, text_column="text", bins=30):
    # 토큰 수 계산
    # 데이터셋의 모든 문장을 토큰으로 변환한 뒤, 각 문장의 토큰 개수만 리스트에 저장한다
    # dataset[text_column] = dataset["text"]
    #     1.
    #         <bos><start_of_turn>user
    #         감기란?
    #         ...
    #     2.
    #         <bos><start_of_turn>user
    #         폐렴이란?
    #         ...
    #     3.
    #         ...
    #     처럼 text 컬럼의 모든 문장을 가져온다.
    # tokenizer.encode(text)
    #     -> 문자열을 토큰으로 바꿈
    #         감기의 원인은 바이러스입니다.
    #         ↓
    #         [2, 451, 781, 93, 1021, 77]
    token_counts = [len(tokenizer.encode(text)) for text in dataset[text_column]]

    # 기본 통계 계산
    stats = {
        "평균 토큰 수":  np.mean(token_counts),
        "중앙값":        np.median(token_counts),
        "최소 토큰 수":  min(token_counts),
        "최대 토큰 수":  max(token_counts),
        "표준편차":      np.std(token_counts),
        "90퍼센타일":   np.percentile(token_counts, 90),
        "95퍼센타일":   np.percentile(token_counts, 95),
        "99퍼센타일":   np.percentile(token_counts, 99),
        "총 샘플 수":    len(token_counts),
    }

    # 분포 시각화
    plt.figure(figsize=(12, 6))
    sns.histplot(data=token_counts, bins=bins, kde=True)
    plt.title(f"Token Length Distribution ({text_column})")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.tight_layout()

    # .py 옆에 이미지 저장
    chart_path = os.path.join(BASE_DIR, "token_distribution.png")
    plt.savefig(chart_path)
    plt.show()
    print(f"[INFO] 토큰 분포 차트 저장 완료: {chart_path}")

    print("\n=== 토큰 수 통계 ===")
    for key, value in stats.items():
        print(f"{key}: {value:.1f}")

    return stats


# ─────────────────────────────────────────
# 6. SFT 학습
# ─────────────────────────────────────────
# 준비된 모델과 데이터로 SFT(Instruction Tuning)를 수행하는 함수
# | 파라미터        | 타입                   | 의미                      | 어디서 전달되는가                                         |
# | ----------- | -------------------- | ----------------------- | ------------------------------------------------- |
# | `model`     | AutoModelForCausalLM | 학습할 Gemma 모델            | `load_model_and_tokenizer()`                      |
# | `tokenizer` | AutoTokenizer        | 토크나이저(Chat Template 포함) | `load_model_and_tokenizer()`                      |
# | `data`      | DatasetDict          | 학습/검증 데이터               | `prepare_datasets()` + `convert_to_chat_format()` |
def train(model, tokenizer, data):
    # GPU 메모리 절약: Gradient Checkpointing 활성화
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # Checkpointing 사용 시 반드시 False

    tokenizer.pad_token = tokenizer.eos_token

    # SFTTrainer에게 학습을 어떻게 진행할지 알려주는 설정 파일
    # | 설정                            | 의미                  | 이번 코드의 값         |
    # | ----------------------------- | ------------------- | ---------------- |
    # | `report_to`                   | 학습 로그 저장 위치         | TensorBoard      |
    # | `eval_strategy`               | 언제 검증할지             | Step마다           |
    # | `eval_steps`                  | 검증 주기               | 100 Step         |
    # | `save_total_limit`            | 체크포인트 최대 개수         | 3개               |
    # | `load_best_model_at_end`      | 가장 좋은 모델 자동 선택      | True             |
    # | `metric_for_best_model`       | 어떤 기준으로 좋은 모델인지     | `eval_loss`      |
    # | `num_train_epochs`            | 전체 학습 횟수            | 3 Epoch          |
    # | `dataset_text_field`          | 학습할 컬럼              | `"text"`         |
    # | `per_device_train_batch_size` | GPU 한 장당 Batch Size | 2                |
    # | `gradient_accumulation_steps` | Gradient 누적 횟수      | 8                |
    # | `max_seq_length`              | 최대 토큰 길이            | 800              |
    # | `lr_scheduler_type`           | Learning Rate 감소 방식 | Cosine           |
    # | `learning_rate`               | 초기 Learning Rate    | `2e-5`           |
    # | `warmup_ratio`                | Warmup 비율           | 3%               |
    # | `bf16`                        | bfloat16 사용 여부      | True             |
    # | `optim`                       | Optimizer           | paged_adamw_8bit |
    # | `output_dir`                  | 체크포인트 저장 폴더         | outputs          |
    # | `logging_steps`               | 로그 출력 주기            | 100 Step         |
    # | `save_steps`                  | 모델 저장 주기            | 100 Step         |
    sft_config = SFTConfig(
        report_to="tensorboard",        # TensorBoard로 학습 곡선 기록

        # ── 검증 관련 설정 ───────────────────────────────────────────────────
        eval_strategy="steps",          # step 단위로 검증
        eval_steps=100,                 # 100 step마다 validation loss 계산
        save_total_limit=3,             # 체크포인트 최대 3개만 유지
        load_best_model_at_end=True,    # 학습 종료 후 val_loss 최저 모델 자동 로드
        metric_for_best_model="eval_loss",  # 최고 모델 기준: Evaluation Loss

        # ── 학습 기본 설정 ───────────────────────────────────────────────────
        num_train_epochs=3,
        dataset_text_field="text",      # 데이터셋의 'text' 컬럼 사용

        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # 실질 배치 크기 = 2 × 8 = 16

        max_seq_length=800,             # 토큰 분포 분석 결과 참고하여 설정
        lr_scheduler_type="cosine",

        learning_rate=2e-5,             # Instruction Tuning은 CPT보다 낮은 LR 사용
        warmup_ratio=0.03,

        bf16=True,                      # bfloat16 학습 (A100, H100 권장; T4는 fp16 사용)
        optim="paged_adamw_8bit",       # 8-bit AdamW로 GPU 메모리 절약

        output_dir=OUTPUT_DIR,          # 체크포인트 저장 위치 (.py 옆 outputs/)
        logging_steps=100,
        save_steps=100,                 # 100 step마다 체크포인트 저장
    )
    
    # @@ Loss는 "얼마나 틀렸는가?"
    # Loss가 작을수록 잘 맞춘다.

    # @@ SFT는 Supervised Fine-Tuning
    #   -> 정답지를 보면서 공부하는 것

    # @@ Epoch는 교과서를 한 바퀴 다 읽은 횟수이다.
    # -> 문제가 100개 있다.
    #     1번
    #     2번
    #     3번
    #     ...
    #     100번
    # 여기까지 다 풀었다.
    # 그러면
    #     Epoch = 1
    # 이다.
    
    # @@ Epoch 많이 공부하면 더 좋지 않을까?
    # -> 시험 문제가
    #     대한민국 수도?
    # 뿐이라고 하자.
    # 학생이
    #     서울
    #     서울
    #     서울
    #     서울
    #     서울
    # 만 외운다.
    # 그러면 다른 문제는 못 푼다.
    # 이걸 과적합(Overfitting) 이라고 한다.
    # 그래서 보통 2~5 Epoch만 학습한다.

    # @@ Validation Loss
    # -> 작아짐: 처음 보는 문제도 잘 푼다는 뜻

    # ── DataCollator: 답변 부분만 학습 ──────────────────────────────────────
    # response_template 이후 토큰만 Next Token Prediction 학습 대상으로 지정
    # 질문 부분은 loss 계산에서 제외 → 더 효율적인 학습
    response_template = "<start_of_turn>model"
    # 답변 부분만 Loss를 계산하도록 만드는 것
    #     질문
    #     ↓
    #     Loss 없음
    #     ----------------
    #     답변
    #     ↓
    #     Loss 계산
    # 이렇게 만든다.
    # 질문까지 학습하지 않으므로 Instruction Tuning에서는 훨씬 효율적이다.
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # GPU 사용을 관리하는 관리자
    accelerator = Accelerator()

    # LLM을 학습시키는 엔진
    trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"],      # 검증 데이터: 과적합 감지용
        args=sft_config,
        data_collator=collator,
    )

    print("[INFO] SFT 학습 시작...")
    print(f"[INFO] 체크포인트 저장 위치: {OUTPUT_DIR}")

    # with accelerator.main_process_first()
    #     -> 쉽게 말하면
    #         GPU0
    #         ↓
    #         먼저 실행
    #         ↓
    #         GPU1
    #         ↓
    #         GPU2
    #     순서를 맞춰주는 것이다.
    with accelerator.main_process_first():
        # 실제로 AI를 공부시키는 코드
        trainer.train()

    # Loss 해석 가이드:
    # Cross Entropy Loss → 평균 예측 확률 = e^(-Loss)
    # 예) Loss 0.2 → e^(-0.2) ≈ 81% 확률로 다음 토큰 예측

    print("[INFO] 학습 완료")
    return model, tokenizer


# ─────────────────────────────────────────
# 7. 최종 모델 저장
# ─────────────────────────────────────────
# 학습이 끝난 모델과 토크나이저를 디스크에 저장하는 함수
def save_model(model, tokenizer):
    if os.path.exists(FINAL_MODEL_DIR):
        print(f"[INFO] 최종 모델이 이미 존재합니다: {FINAL_MODEL_DIR}")
        print("[INFO] 덮어쓰려면 해당 폴더를 삭제 후 재실행하세요.")
        return

    print(f"[INFO] 최종 모델 저장 중 → {FINAL_MODEL_DIR}")
    # safe_serialization=False
    #     -> 모델을 어떤 파일 형식으로 저장할지 결정한다.
    #     기존 PyTorch 저장 방식:
    #         False
    #         pytorch_model.bin
    #     Safetensors 방식:
    #         True
    #         model.safetensors
    model.save_pretrained(FINAL_MODEL_DIR, safe_serialization=False)
    # 모델만 저장하면 안 된다. 토크나이저도 같이 저장해야 한다.
    # 문장을 토큰으로 바꾸는 규칙도 모델의 일부이기 때문이다.
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"[INFO] 모델 저장 완료: {FINAL_MODEL_DIR}")

    # ── HuggingFace Hub 업로드 (선택 사항) ──────────────────────────────────
    # Write 권한 토큰 필요. 업로드하려면 아래 주석을 해제하고 username을 설정하세요.
    # username = os.getenv("HF_USERNAME")
    # model.push_to_hub(f"{username}/{os.path.basename(FINAL_MODEL_DIR)}")
    # tokenizer.push_to_hub(f"{username}/{os.path.basename(FINAL_MODEL_DIR)}")


# ─────────────────────────────────────────
# main
# ─────────────────────────────────────────
def main():
    # Step 1. HuggingFace 로그인
    login_huggingface()

    # Step 2. 모델 & 토크나이저 로드
    model, tokenizer, model_name = load_model_and_tokenizer()

    # Step 3. 학습 데이터 준비
    data = prepare_datasets()

    # Step 4. Chat Template 변환
    data = convert_to_chat_format(data, tokenizer)

    # Step 5. 토큰 분포 분석 (max_seq_length 설정 참고용)
    analyze_token_distribution(data["train"], tokenizer)

    # Step 6. SFT 학습
    model, tokenizer = train(model, tokenizer, data)

    # Step 7. 최종 모델 저장
    save_model(model, tokenizer)


if __name__ == "__main__":
    main()