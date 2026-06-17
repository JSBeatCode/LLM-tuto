"""
[Topic 3] 실전 실습 — 리뷰 답변 생성기
- CSV에서 리뷰 로드 후 체인 적용
- batch()로 다수 입력 일괄 처리
"""

import os
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── 환경 설정 ──────────────────────────────────────────────
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-4o-mini"

# reviews.csv 경로 (이 .py 파일과 같은 폴더에 위치)
CSV_PATH = "./reviews.csv"

# CSV 없을 때 사용할 샘플 리뷰
SAMPLE_REVIEWS = [
    "음식이 너무 맛있어요! 특히 김치찌개가 일품이었습니다. 꼭 다시 오고 싶어요.",
    "서비스가 너무 불친절했어요. 직원이 주문도 제대로 안 받고 무시하는 느낌이었어요.",
    "가격 대비 양이 너무 적어요. 맛은 괜찮은데 배가 안 불러서 아쉬웠습니다.",
    "분위기가 아늑하고 좋았어요. 데이트 코스로 딱이에요!",
    "음식이 나오는 데 1시간이나 걸렸어요. 너무 오래 기다렸습니다.",
    "단골이 된 지 3년째인데 항상 맛이 일정해서 좋아요. 믿고 먹을 수 있는 집!",
]


# ── Step 1. 리뷰 데이터 로드 ───────────────────────────────
def load_reviews() -> list[str]:
    """
    reviews.csv에서 리뷰 목록을 불러옵니다.
    CSV 파일이 없으면 샘플 리뷰를 사용합니다.
    """
    print("\n" + "="*60)
    print("▶ Step 1. 리뷰 데이터 로드")
    print("="*60)

    if os.path.exists(CSV_PATH):
        reviews = pd.read_csv(CSV_PATH)
        print(f"\n[CSV 로드 성공] shape: {reviews.shape}")
        print(reviews.head())
        review_list = reviews['Review'].to_list()
        print(f"\n총 리뷰 수: {len(review_list)}개")
    else:
        print(f"\n[주의] '{CSV_PATH}' 파일을 찾을 수 없습니다.")
        print("→ 샘플 리뷰 데이터로 대체합니다.\n")
        review_list = SAMPLE_REVIEWS
        for i, r in enumerate(review_list):
            print(f"  [{i}] {r}")

    return review_list


# ── Step 2. 리뷰 답변 체인 구성 ───────────────────────────
def build_reply_chain(llm):
    """
    퉁명스러운 사장님 페르소나로 리뷰에 답변하는 체인을 구성합니다.
    Prompt | LLM | StrOutputParser
    """
    print("\n" + "="*60)
    print("▶ Step 2. 리뷰 답변 체인 구성")
    print("="*60)

    reply_template = ChatPromptTemplate([
        ('system', '''
당신은 반말과 퉁명스러운 태도가 독특한 인기를 끄는 유명한 맛집의 사장님입니다.
손님의 리뷰에 대해, 5문장 정도의 댓글을 작성하세요.
부정적인 리뷰에는 반박하고, 긍정적인 리뷰에는 심드렁하게 답변하세요.
이모지를 아주 많이 쓰세요.'''),
        ('user', '{review}')
    ])

    reply_chain = reply_template | llm | StrOutputParser()

    print("\n[체인 구조]")
    print("  ChatPromptTemplate | ChatOpenAI | StrOutputParser")

    return reply_chain


# ── Step 3. 단건 invoke() 실행 ────────────────────────────
def run_single_invoke(reply_chain, review_list: list[str]):
    """
    invoke()로 리뷰 1개에 대한 답변을 생성합니다.
    """
    print("\n" + "="*60)
    print("▶ Step 3. 단건 invoke() — 리뷰 1개 처리")
    print("="*60)

    target_review = review_list[0]
    print(f"\n[입력 리뷰]\n  {target_review}")

    response = reply_chain.invoke({'review': target_review})

    print(f"\n[사장님 답변]\n{response}")


# ── Step 4. batch() 일괄 처리 ─────────────────────────────
def run_batch(reply_chain, review_list: list[str]):
    """
    batch()로 여러 리뷰를 한꺼번에 처리합니다.
    입력은 dict 리스트 형식으로 전달합니다.
    """
    print("\n" + "="*60)
    print("▶ Step 4. batch() — 다수 리뷰 일괄 처리")
    print("="*60)

    # 상위 6개 리뷰 선택
    batch_reviews = review_list[:6]
    batch_input   = [{'review': r} for r in batch_reviews]

    print(f"\n[일괄 처리 대상: {len(batch_reviews)}개 리뷰]")
    for i, r in enumerate(batch_reviews):
        print(f"  [{i}] {r}")

    print("\n[batch() 실행 중...]\n")
    responses = reply_chain.batch(batch_input)

    print("[사장님 답변 목록]")
    for i, (review, reply) in enumerate(zip(batch_reviews, responses)):
        print(f"\n{'─'*50}")
        print(f"[리뷰 {i}] {review}")
        print(f"[답변 {i}]\n{reply}")


# ── main ───────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  [Topic 3] 실전 실습 — 리뷰 답변 생성기")
    print("=" * 60)

    # LLM 초기화
    llm = ChatOpenAI(temperature=0.5, model=MODEL, max_tokens=1000)

    # Step 1: 리뷰 데이터 로드
    review_list = load_reviews()

    # Step 2: 체인 구성
    reply_chain = build_reply_chain(llm)

    # Step 3: 단건 invoke
    run_single_invoke(reply_chain, review_list)

    # Step 4: batch() 일괄 처리
    run_batch(reply_chain, review_list)

    print("\n" + "="*60)
    print("  실습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()