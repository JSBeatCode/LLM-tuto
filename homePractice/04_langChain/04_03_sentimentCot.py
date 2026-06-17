# =============================================================================
# Topic 3. 실전 프로젝트 — 감성 분류기 & CoT 프롬프트 엔지니어링
# =============================================================================
# 학습 목표:
#   1. CSV 리뷰 데이터 로드 및 LLM 기반 긍정/부정 자동 분류
#   2. 정확도 측정 루프 및 evaluate() 함수 모듈화
#   3. 기본 프롬프트 → CoT 요약 방식 → 항목별 구조화 방식으로 성능 개선 실험
# =============================================================================

# [사전 준비] 패키지 설치
# pip install langchain langchain_openai openai python-dotenv pandas

# [사전 준비] 이 파일과 같은 폴더에 .env 파일 생성 후 아래 내용 입력
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# [사전 준비] reviews.csv 파일을 이 파일과 같은 폴더에 준비하세요.
# 컬럼 구성: Num(인덱스) / Review(리뷰 텍스트) / Label(1:긍정, -1:부정)

import openai
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# =============================================================================
# 0. 환경 설정
# =============================================================================

load_dotenv(override=True)

client = openai.OpenAI()

try:
    client.models.list()
    print("✅ OPENAI_API_KEY가 정상적으로 설정되어 있습니다.\n")
except Exception:
    print("❌ API 키가 유효하지 않습니다! .env 파일을 확인해주세요.")
    exit()

MODEL = 'gpt-4o-mini'
llm = ChatOpenAI(model=MODEL, temperature=0, max_tokens=1024)


# =============================================================================
# 1. CSV 리뷰 데이터 로드
# =============================================================================
# 컬럼 설명:
#   - Num   : 리뷰 인덱스 번호
#   - Review: 리뷰 텍스트
#   - Label : 긍정/부정 레이블 (1: 긍정, -1: 부정)

print("=" * 60)
print("▶ [1] 리뷰 데이터 로드")
print("=" * 60)

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_FILE_PATH = os.path.join(BASE_DIR, "reviews.csv")
    reviews = pd.read_csv(CSV_FILE_PATH)
    print(f"데이터 shape: {reviews.shape}")
    print(reviews.head())
    print()
except FileNotFoundError:
    print("❌ reviews.csv 파일이 없습니다! 이 파일과 같은 폴더에 reviews.csv를 넣어주세요.")
    exit()


# =============================================================================
# 2. 기본 프롬프트로 전체 데이터 분류 + 정확도 측정
# =============================================================================

print("=" * 60)
print("▶ [2] 기본 프롬프트 — 전체 리뷰 분류 및 정확도 측정")
print("=" * 60)

# 가장 단순한 형태의 시스템 프롬프트
basic_system_prompt = '''
주어지는 입력이 긍정/부정 중 어떤 내용을 담고 있는지 분류하세요.

분류 결과: 부정, 혹은 분류 결과: 긍정 을 출력하면 됩니다.
위 형식을 지키고, 분류 결과 뒤에는 별도의 내용을 출력하지 마세요.
'''

basic_prompt = ChatPromptTemplate([
    ("system", basic_system_prompt),
    ("user", '{review}')
])

correct = 0
incorrect = 0

for idx, review, label in zip(reviews['Num'], reviews['Review'], reviews['Label']):
    print(f'#{idx} : (정답 {correct} / 오답 {incorrect})')
    print(review)
    response = llm.invoke(basic_prompt.format_messages(review=review)).content
    print(f"{response}  /  실제 레이블: {label}")

    if label == -1:
        if '분류 결과: 부정' in response:
            correct += 1
        else:
            incorrect += 1
            print('⚠️  오분류!')
    else:
        if '분류 결과: 긍정' in response:
            correct += 1
        else:
            incorrect += 1
            print('⚠️  오분류!')
    print()

print(f"✅ 정답: {correct}")
print(f"❌ 오답: {incorrect}")
print(f"📊 기본 프롬프트 정확도: {correct / (correct + incorrect):.2%}\n")


# =============================================================================
# 3. 어려운 리뷰(hard_reviews) 분리
# =============================================================================
# 기본 프롬프트에서 자주 틀리는 9개의 어려운 리뷰를 따로 분리하여 집중 테스트

print("=" * 60)
print("▶ [3] 어려운 리뷰(hard_reviews) 분리")
print("=" * 60)

# reviews.iloc: 데이터에서 특정 행(번호)만 골라서 새로운 데이터로 만든 것
hard_reviews = reviews.iloc[[1, 5, 6, 13, 16, 34, 38, 45, 46]]
print(f"hard_reviews shape: {hard_reviews.shape}")
print(hard_reviews)
print()


# =============================================================================
# 4. evaluate() 함수 — 프롬프트 성능 측정 모듈
# =============================================================================
# system_prompt와 리뷰 데이터프레임을 받아 분류 정확도를 출력하는 범용 함수

def evaluate(system_prompt: str, reviews: pd.DataFrame) -> float:
    """
    주어진 system_prompt로 리뷰를 분류하고 정확도를 반환하는 함수.

    Args:
        system_prompt : LLM에게 줄 시스템 역할 지시문
        reviews       : Num / Review / Label 컬럼을 가진 데이터프레임

    Returns:
        accuracy (float): 분류 정확도
    """
    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("user", '{review}')
    ])

    correct = 0
    incorrect = 0

    for idx, review, label in zip(reviews['Num'], reviews['Review'], reviews['Label']):
        print(f'#{idx} : (정답 {correct} / 오답 {incorrect})')
        print(review)
        response = llm.invoke(prompt.format_messages(review=review)).content
        print(f"{response}  /  실제 레이블: {label}")

        if label == -1:
            if '분류 결과: 부정' in response:
                correct += 1
            else:
                incorrect += 1
                print('⚠️  오분류!')
        else:
            if '분류 결과: 긍정' in response:
                correct += 1
            else:
                incorrect += 1
                print('⚠️  오분류!')
        print()

    accuracy = correct / (correct + incorrect)
    print(f"✅ 정답: {correct}")
    print(f"❌ 오답: {incorrect}")
    print(f"📊 정확도: {accuracy:.2%}\n")
    return accuracy


# =============================================================================
# 5. CoT 방식 1 — 요약 후 분류 (Chain-of-Thought 유도)
# =============================================================================
# 리뷰를 먼저 30자 이내로 요약하게 한 뒤 분류
# → 추론 과정이 생기면서 정확도가 향상됨

print("=" * 60)
print("▶ [5] CoT 방식 1 — 요약 후 분류 프롬프트")
print("=" * 60)

cot_prompt_v1 = '''
음식점에 대한 리뷰가 주어지면, 사용자의 의도를 파악하여
30자 이내로 요약하고, 음식점에 대한 긍정/부정 여부를 분류하여 출력하세요.

답변의 마지막에 분류 결과: 부정, 혹은 분류 결과: 긍정 을 출력하면 됩니다.
위 형식을 지키고, 분류 결과 뒤에는 별도의 내용을 출력하지 마세요.
---

'''

accuracy_v1 = evaluate(cot_prompt_v1, hard_reviews)


# =============================================================================
# 6. CoT 방식 2 — 항목별 구조화 분석 후 분류
# =============================================================================
# 음식 / 서비스 / 가격 / 비교 / 분위기 등 6가지 항목을 먼저 분석 후 분류
# → 구조화된 CoT로 복잡한 리뷰에서 강점 발휘

print("=" * 60)
print("▶ [6] CoT 방식 2 — 항목별 구조화 분석 프롬프트")
print("=" * 60)

cot_prompt_v2 = '''
음식점에 대한 리뷰가 주어지면, 아래의 요소에 대한 견해를 각각 30자 이내로 출력하고,
이를 바탕으로 '긍정/부정'중 하나로 최종 분류 결과를 출력하세요.
각 요소에 대한 언급이 없는 경우 생략하세요.
일반적으로, 부정에서 긍정으로 끝나는 경우 긍정 리뷰입니다.
긍정에서 부정으로 끝나는 경우 부정 리뷰입니다.
답변의 마지막에 분류 결과: 부정, 혹은 분류 결과: 긍정 을 출력하면 됩니다.
위 형식을 지키고, 분류 결과 뒤에는 별도의 내용을 출력하지 마세요.
---
1. 음식
2. 서비스
3. 가격 대비 만족도
4. 다른 식당과의 비교
5. 분위기와 음악 등
6. 분류 결과
'''

accuracy_v2 = evaluate(cot_prompt_v2, hard_reviews)


# =============================================================================
# 7. 프롬프트 엔지니어링 결과 비교 요약
# =============================================================================

print("=" * 60)
print("▶ [7] 프롬프트 버전별 성능 비교 (hard_reviews 기준)")
print("=" * 60)
print(f"  CoT 방식 1 (요약 후 분류)       : {accuracy_v1:.2%}")
print(f"  CoT 방식 2 (항목별 구조화 분류) : {accuracy_v2:.2%}")
print()
print("""
💡 핵심 인사이트:
   - 단순 분류 지시 → CoT 요약 유도만으로도 정확도가 향상됩니다.
   - 항목별 구조화 분석은 복잡한 감성이 섞인 리뷰에서 특히 강점을 보입니다.
   - 프롬프트 엔지니어링은 모델 교체 없이 성능을 높이는 가장 비용 효율적인 방법입니다.
""")