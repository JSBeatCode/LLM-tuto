"""
[실습] Instruction Tuning - eval.py
- 역할    : train.py로 학습된 모델을 로드해서 추론/평가 수행
- 평가 ①  : 최종 모델(gemma-3-1b-pt-MED-Instruct)로 의료 질문 테스트
- 평가 ②  : 동일 질문 10회 반복 → 답변 일관성 확인
- 평가 ③  : 일반 상식 질문 → Catastrophic Forgetting 검증
- 평가 ④  : 특정 체크포인트 로드 후 최종 모델과 비교
- 실행 전제: train.py 실행 완료 + gemma-3-1b-pt-MED-Instruct/ 폴더 존재
"""

# ── 표준 라이브러리 ──────────────────────────────────────────────────────────
import os

# ── 서드파티: 환경변수 ────────────────────────────────────────────────────────
from dotenv import load_dotenv

# ── 서드파티: HuggingFace ─────────────────────────────────────────────────────
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ── 서드파티: LangChain ───────────────────────────────────────────────────────
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


# ─────────────────────────────────────────
# 0. 경로 설정 & 환경변수 로드
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# train.py 에서 저장한 경로와 동일하게 맞춤
FINAL_MODEL_DIR  = os.path.join(BASE_DIR, "gemma-3-1b-pt-MED-Instruct")  # 최종 학습 모델
INSTRUCT_TOK_DIR = os.path.join(BASE_DIR, "gemma-3-1b-it-tokenizer")      # Chat Template 토크나이저
OUTPUT_DIR       = os.path.join(BASE_DIR, "outputs")                       # 체크포인트 폴더

load_dotenv(override=True)


# ─────────────────────────────────────────
# 1. 모델 & 토크나이저 로드
# ─────────────────────────────────────────
def load_model(model_dir: str):
    """
    주어진 경로에서 모델과 토크나이저를 로드.
    최종 모델과 체크포인트 비교 시 재사용.
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"[ERROR] 모델 폴더를 찾을 수 없습니다: {model_dir}\n"
            f"        train.py를 먼저 실행해서 모델을 학습/저장해 주세요."
        )

    # 토크나이저: Chat Template이 포함된 instruct 토크나이저 사용
    if not os.path.exists(INSTRUCT_TOK_DIR):
        raise FileNotFoundError(
            f"[ERROR] 토크나이저 폴더를 찾을 수 없습니다: {INSTRUCT_TOK_DIR}\n"
            f"        train.py를 먼저 실행해 주세요."
        )

    print(f"[INFO] 모델 로드 중: {model_dir}")
    # model_dir 폴더에 저장된 모델을 불러옵니다.
    # 저장된 AI 모델을 메모리(GPU)에 올린다.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto",
        # Attention은 현재 단어가 다른 단어들을 얼마나 참고할지 계산하는 알고리즘
        #     철수는   영희에게   사과를   줬다.   그는   기뻐했다.
        #     -> "나는 지금 어떤 단어를 가장 중요하게 봐야 하지?" 
        #     -> "그는" 이 누구일까요?
        attn_implementation="eager",
        # attn_implementation="flash_attention_2"  # A100 이상에서 활성화
    )
    # 모델을 학습 모드 → 추론 모드로 변경합니다.
    model.eval()
    # GPU에서 사용하지 않는 메모리를 비웁니다.
    torch.cuda.empty_cache()

    # 토크나이저를 불러옵니다.
    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_TOK_DIR)
    print(f"[INFO] 모델 로드 완료: {os.path.basename(model_dir)}")
    return model, tokenizer


# ─────────────────────────────────────────
# 2. LangChain 체인 구성
# ─────────────────────────────────────────
def build_chain(model, tokenizer):
    # 파인튜닝 모델이므로 공식 파라미터를 따를 필요는 없음
    # 답변을 생성하는 방식을 설정
    gen_config = dict(
        do_sample=True,
        max_new_tokens=1024,
        repetition_penalty=1.1,
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        stop_sequence="<end_of_turn>",  # 프롬프트의 끝 문자 표시
    )

    # Transformers 모델을 쉽게 사용할 수 있는 추론 엔진으로 만듭니다.
    #     질문
    #      ↓
    #     토큰화
    #      ↓
    #     모델 실행
    #      ↓
    #     답변 생성
    # 이 과정을 한 번에 처리합니다. pipe가 없으면 이것을 전부 직접 작성해야 합니다.
    # *Transformer = GPT나 Gemma를 만드는 엔진(설계도)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        **gen_config,
    )

    # HuggingFace 모델을 LangChain에 연결한다.
    llm = HuggingFacePipeline(pipeline=pipe, pipeline_kwargs=gen_config)
    # 일반 LLM을 ChatGPT처럼 채팅 가능한 모델로 만든다.
    chat_model = ChatHuggingFace(llm=llm, tokenizer=tokenizer)

    # 단순 질문 → 답변 형태의 체인
    # 사용자의 질문을 모델이 이해하는 형식으로 만든다.
    prompt = ChatPromptTemplate([("user", "{question}\n")])
    # 질문부터 최종 답변까지 하나의 흐름으로 연결한다.
    chain = prompt | chat_model | StrOutputParser()

    print("[INFO] LangChain 체인 구성 완료")
    return chain


# ─────────────────────────────────────────
# 3. 평가 ① 의료 질문 테스트
# ─────────────────────────────────────────
# ➡️ 의료 지식을 잘 배웠는지 확인
def eval_medical_questions(chain):
    print("\n" + "=" * 60)
    print("평가 ① 의료 질문 테스트")
    print("=" * 60)

    inputs = [
        "제 1형 당뇨병은 어떻게 치료하나요?",
        "고혈압인데 먹으면 안되는 거 있나요?",
        "충치가 너무 아파요.",
        "불안장애 환자의 식이는 어떻게 해야 되나요?",
    ]

    for question in inputs:
        print(f"\n[질문] {question}")
        # 질문을 AI에게 보내고 답변 문자열(String)을 받아오는 함수
        # 실행하면 내부적으로 다음 순서로 동작:
        #     "충치가 너무 아파요."
        #             ↓
        #     Prompt 생성
        #             ↓
        #     Tokenizer가 토큰으로 변환
        #             ↓
        #     Gemma 모델 추론
        #             ↓
        #     토큰 생성
        #             ↓
        #     문장으로 변환
        #             ↓
        #     "진통제를 복용하시고..."
        print(f"[답변] {chain.invoke(question)}")
        print("-" * 60)


# ─────────────────────────────────────────
# 4. 평가 ② 동일 질문 반복 → 일관성 확인
# ─────────────────────────────────────────
# ➡️ 같은 질문에 답변이 일관적인지 확인
def eval_consistency(chain, repeat: int = 10):
    print("\n" + "=" * 60)
    print(f"평가 ② 동일 질문 {repeat}회 반복 → 답변 일관성 확인")
    print("=" * 60)

    question = "충치가 너무 아파요."
    print(f"[질문] {question} (총 {repeat}회 반복)\n")

    # 똑같은 질무을 10번 함.
    # 모델 설정에 temperature = 0.7 이기 때문에 실행할 때마다 조금씩 다른 답변이 나올 수 있습니다.
    # 예를 들어
    #     1번째
    #         진통제를 드시고 치과에 가세요.
    #     2번째
    #         냉찜질을 해보시고 치과를 방문하세요.
    #     3번째
    #         소금물 가글도 도움이 될 수 있습니다.
    # 처럼 표현은 달라지지만 핵심 내용은 같아야 좋은 모델입니다.
    for i in range(1, repeat + 1):
        print(f"--- 반복 {i}/{repeat} ---")
        print(f"[답변] {chain.invoke(question)}")


# ─────────────────────────────────────────
# 5. 평가 ③ Catastrophic Forgetting 검증
# ─────────────────────────────────────────
# ➡️ 의료와 상관없는 일반 상식도 잘 답하는지 확인
def eval_catastrophic_forgetting(chain):
    """
    의료 데이터가 아닌 일반 질문으로 테스트.
    KoAlpaca-RealQA를 혼합 학습했기 때문에
    일반 지식도 유지되어야 정상 (Catastrophic Forgetting 미발생).
    """
    print("\n" + "=" * 60)
    print("평가 ③ Catastrophic Forgetting 검증 (일반 상식 질문)")
    print("=" * 60)
    print("* 의료 특화 학습 후에도 일반 질문에 답할 수 있으면 정상\n")

    inputs = [
        "사과는 왜 빨간가요?",
        "저는 사과 12개가 있었는데, 친구에게 절반을 나눠줬어요. 남은 사과는 몇 개인가요?",
        "곰과 사자 중 누가 더 큰가요?",
    ]

    for question in inputs:
        print(f"\n[질문] {question}")
        print(f"[답변] {chain.invoke(question)}")
        print("-" * 60)


# ─────────────────────────────────────────
# 6. 평가 ④ 특정 체크포인트와 최종 모델 비교
# ─────────────────────────────────────────
# 체크포인트란? 
#     -> 학습 중간에 저장한 모델입니다.
#     예를 들어 모델을 10시간 동안 학습한다고 해보겠습니다. 아래와 같이 저장합니다.
#     학습 시작
#         │
#         ├── 10%  → checkpoint-100
#         │
#         ├── 30%  → checkpoint-500
#         │
#         ├── 60%  → checkpoint-1000
#         │
#         ├── 90%  → checkpoint-1536
#         │
#         └── 100% → 최종 모델(gemma-3-1b-pt-MED-Instruct)
#     왜 체크포인트를 저장할까?
#         컴퓨터가 꺼지거나
#         오류가 나거나
#         중간 성능을 비교하고 싶다면
def eval_checkpoint_compare(tokenizer):
    """
    train.py의 outputs/ 폴더에서 가장 최근 체크포인트를 찾아
    최종 모델과 동일 질문으로 비교 평가.
    """
    print("\n" + "=" * 60)
    print("평가 ④ 체크포인트 vs 최종 모델 비교")
    print("=" * 60)

    # outputs/ 폴더에서 체크포인트 목록 자동 탐색
    if not os.path.exists(OUTPUT_DIR):
        print(f"[SKIP] 체크포인트 폴더 없음: {OUTPUT_DIR}")
        return

    # checkpoint로 시작하는 폴더만 찾아옵니다.
    # sorted() -> 찾은 체크포인트를 이름순으로 정렬하는 것입니다.
    #     checkpoints = [
    #         "checkpoint-500",
    #         "checkpoint-1000",
    #         "checkpoint-1536"
    #     ]
    checkpoints = sorted([
        d for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ])

    # 체크포인트가 하나도 없으면 평가를 하지 않는다.
    if not checkpoints:
        print(f"[SKIP] 체크포인트 없음: {OUTPUT_DIR}")
        return

    # 가장 마지막 체크포인트 선택 (노트북 기준 checkpoint-1533)
    ckpt_name = checkpoints[-1]
    ckpt_dir  = os.path.join(OUTPUT_DIR, ckpt_name)
    print(f"[INFO] 비교 대상 체크포인트: {ckpt_name}\n")

    compare_questions = [
        "제 1형 당뇨병은 어떻게 치료하나요?",
        "고혈압인데 먹으면 안되는 거 있나요?",
        "충치가 너무 아파요.",
        "불안장애 환자의 식이는 어떻게 해야 되나요?",
    ]

    # ── 체크포인트 모델 로드 & 평가 ─────────────────────────────────────────
    # "학습 중간에 저장된 모델은 어떻게 답하는지 확인해보자."

    # 한 줄 요약: 체크포인트 모델을 메모리에 로드한다.
    ckpt_model, _ = load_model(ckpt_dir)

    # 방금 불러온 체크포인트 모델을 질문 → 답변 할 수 있는 형태로 만듭니다.
    ckpt_chain    = build_chain(ckpt_model, tokenizer)

    print(f"[{ckpt_name}] 답변")
    # 한 줄 요약: 체크포인트 모델에게 질문하고 답변을 출력한다.
    # 왜 이 과정을 하는 걸까?
    #     바로 뒤에서 최종 모델에게도 똑같은 질문을 합니다.
    #     학습 중간 모델과 최종 모델의 답변을 비교하여,
    #     "학습이 진행되면서 답변이 얼마나 좋아졌는지" 확인
    for question in compare_questions:
        print(f"\n[질문] {question}")
        print(f"[답변] {ckpt_chain.invoke(question)}")
        print("-" * 60)

    # 메모리 해제 후 최종 모델 로드
    del ckpt_model
    torch.cuda.empty_cache()

    # ── 최종 모델 로드 & 평가 ────────────────────────────────────────────────
    final_model, _ = load_model(FINAL_MODEL_DIR)
    final_chain    = build_chain(final_model, tokenizer)

    print(f"\n[최종 모델] 답변")
    for question in compare_questions:
        print(f"\n[질문] {question}")
        print(f"[답변] {final_chain.invoke(question)}")
        print("-" * 60)


# ─────────────────────────────────────────
# main
# ─────────────────────────────────────────
def main():
    # Step 1. 최종 모델 로드
    model, tokenizer = load_model(FINAL_MODEL_DIR)

    # Step 2. LangChain 체인 구성
    chain = build_chain(model, tokenizer)

    # Step 3. 평가 ① 의료 질문 테스트
    eval_medical_questions(chain)

    # Step 4. 평가 ② 동일 질문 10회 반복 → 일관성 확인
    eval_consistency(chain, repeat=10)

    # Step 5. 평가 ③ Catastrophic Forgetting 검증
    eval_catastrophic_forgetting(chain)

    # Step 6. 평가 ④ 특정 체크포인트 vs 최종 모델 비교
    # 메모리 해제 후 새 모델을 로드하므로 마지막에 실행
    del model
    torch.cuda.empty_cache()
    eval_checkpoint_compare(tokenizer)


if __name__ == "__main__":
    main()