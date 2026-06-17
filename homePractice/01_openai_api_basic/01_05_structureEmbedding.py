"""
주제 5 — 고급 기능 (Embedding & Structured Output)
===================================================
학습 내용:
  1. Embedding 기초 — 텍스트를 벡터로 변환
  2. 임베딩 모델 3종 비교 (차원 수, 용도)
  3. 코사인 거리 & 유클리드 거리로 유사도 계산
  4. 뉴스 헤드라인 유사도 검색 실습 (노트북 예제 그대로)
  5. 유사도 기반 간단한 텍스트 검색 엔진 구현
  6. Structured Output — Pydantic 모델로 JSON 출력 보장
  7. 중첩 Pydantic 모델 실습 (여행지 정보 파싱)
  8. Structured Output 활용 — 상품 리뷰 자동 분류기
"""

import os
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
import openai


# ── 환경변수 로드 ─────────────────────────────────────────
load_dotenv(override=True)

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMB_MODEL_LARGE = "text-embedding-3-large"   # 3072차원
EMB_MODEL_SMALL = "text-embedding-3-small"   # 1536차원
CHAT_MODEL      = "gpt-4o-mini"


# ══════════════════════════════════════════════════════════
# ■ 거리 계산 유틸 함수
# ══════════════════════════════════════════════════════════

def cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> np.ndarray:
    """
    코사인 거리 = 1 - 코사인 유사도
    값이 0 에 가까울수록 → 의미가 유사한 문장
    값이 1 에 가까울수록 → 의미가 다른 문장
    """
    dot_product = np.dot(embedding1, embedding2.T)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2, axis=1)
    similarity = dot_product / (norm1 * norm2)
    return 1 - similarity


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> np.ndarray:
    """
    유클리드 거리 = 두 벡터 사이의 직선 거리
    값이 작을수록 → 의미가 유사한 문장
    """
    return np.linalg.norm(embedding2 - embedding1, axis=1)


def get_embedding(text: str, model: str = EMB_MODEL_LARGE) -> np.ndarray:
    """텍스트 한 문장의 임베딩 벡터를 반환합니다."""
    response = client.embeddings.create(model=model, input=text)
    return np.array(response.data[0].embedding, dtype="float32")


def get_embeddings(texts: list[str], model: str = EMB_MODEL_LARGE) -> np.ndarray:
    """텍스트 목록 전체의 임베딩 벡터를 한 번의 호출로 반환합니다."""
    response = client.embeddings.create(model=model, input=texts)
    return np.array([r.embedding for r in response.data], dtype="float32")


# ══════════════════════════════════════════════════════════
def step1_embedding_basics() -> None:
    """
    [STEP 1] Embedding 기초 — 텍스트를 벡터로 변환
    ------------------------------------------------
    임베딩(Embedding)이란 텍스트를 고차원 숫자 벡터로 변환하는 것입니다.
    의미가 비슷한 문장일수록 벡터 공간에서 가까운 위치에 놓입니다.

    활용 예시:
      - 검색 (RAG): 질문과 문서 간 유사도를 계산해 관련 문서 검색
      - 추천: 텍스트 연관성 기반 콘텐츠 추천
      - 분류: 레이블 없이 의미 기반으로 군집화
    """
    print("=" * 55)
    print("[STEP 1] Embedding 기초")
    print("=" * 55)

    text = '삼성SDS, 생성형 AI 사업 본격화…"패브릭스"와 "브리티 코파일럿" 기반 공공 디지털 혁신 지원'

    response = client.embeddings.create(
        model=EMB_MODEL_LARGE,
        input=text,
    )

    emb = response.data[0].embedding

    print(f"  입력 텍스트  : {text}")
    print(f"  임베딩 차원  : {len(emb)}")
    print(f"  벡터 앞 5개  : {emb[:5]}")
    print(f"  벡터 타입    : {type(emb[0]).__name__}")
    print()
    print("  → 텍스트가 3072개의 숫자로 표현됩니다.")
    print("  → 이 벡터 값 자체는 해석하기 어렵지만,")
    print("    다른 텍스트의 벡터와 거리를 계산하면 의미적 유사도를 알 수 있습니다.")
    print()


# ══════════════════════════════════════════════════════════
def step2_model_comparison() -> None:
    """
    [STEP 2] 임베딩 모델 3종 비교
    --------------------------------
    OpenAI 임베딩 모델:
      - text-embedding-3-large  : 3072차원, 고성능, 높은 비용
      - text-embedding-3-small  : 1536차원, 빠르고 저렴
      - text-embedding-ada-002  : 1536차원, 구버전 (레거시)

    같은 문장에 대해 차원 수와 응답 시간을 비교합니다.
    """
    print("=" * 55)
    print("[STEP 2] 임베딩 모델 3종 비교")
    print("=" * 55)

    text = "인공지능이 미래를 바꿀 것입니다."
    models = [
        ("text-embedding-3-large", "최신 · 고성능"),
        ("text-embedding-3-small", "최신 · 경량"),
        ("text-embedding-ada-002", "구버전 · 레거시"),
    ]

    print(f"  입력 텍스트: {text}\n")
    print(f"  {'모델':<30} {'차원':>6}  {'비고'}")
    print("  " + "-" * 55)

    for model_name, note in models:
        resp = client.embeddings.create(model=model_name, input=text)
        dim = len(resp.data[0].embedding)
        print(f"  {model_name:<30} {dim:>6}  {note}")

    print()
    print("  → 일반적인 유사도 검색에는 text-embedding-3-small 로도 충분합니다.")
    print("  → 정밀도가 중요한 경우 text-embedding-3-large 를 사용하세요.")
    print()


# ══════════════════════════════════════════════════════════
def step3_distance_metrics() -> None:
    """
    [STEP 3] 코사인 거리 vs 유클리드 거리
    ----------------------------------------
    두 가지 유사도 지표를 직접 계산하고 차이를 이해합니다.

    코사인 거리:
      - 벡터의 방향만 비교 (크기 무관)
      - 텍스트 길이 차이에 강건함
      - 일반적으로 임베딩 유사도에 더 많이 사용

    유클리드 거리:
      - 벡터 공간에서의 직선 거리
      - 벡터 크기(norm)도 반영됨
    """
    print("=" * 55)
    print("[STEP 3] 코사인 거리 vs 유클리드 거리")
    print("=" * 55)

    sentences = [
        "나는 강아지를 좋아한다.",         # 기준 문장
        "나는 개를 매우 좋아합니다.",       # 유사한 의미
        "강아지는 귀여운 동물이다.",        # 관련 있는 문장
        "오늘 주식 시장이 급락했다.",       # 전혀 다른 주제
    ]

    print(f"  기준 문장: \"{sentences[0]}\"\n")

    # 기준 문장 임베딩
    query_emb = get_embedding(sentences[0])
    # 비교 문장 임베딩
    target_embs = get_embeddings(sentences[1:])

    cosine_dists   = cosine_distance(query_emb, target_embs)
    euclidean_dists = euclidean_distance(query_emb, target_embs)

    print(f"  {'비교 문장':<28} {'코사인 거리':>12}  {'유클리드 거리':>14}")
    print("  " + "-" * 60)

    for i, sentence in enumerate(sentences[1:]):
        print(f"  {sentence:<28} {cosine_dists[i]:>12.4f}  {euclidean_dists[i]:>14.4f}")

    print()
    print("  → 거리 값이 작을수록 기준 문장과 의미가 유사합니다.")
    print()


# ══════════════════════════════════════════════════════════
def step4_news_similarity() -> None:
    """
    [STEP 4] 뉴스 헤드라인 유사도 검색 (노트북 예제)
    --------------------------------------------------
    노트북 원본 예제를 그대로 실습합니다.
    쿼리 뉴스와 4개의 후보 뉴스 간 유사도를 계산하여
    가장 관련 있는 기사를 찾습니다.
    """
    print("=" * 55)
    print("[STEP 4] 뉴스 헤드라인 유사도 검색")
    print("=" * 55)

    query = '삼성SDS, 생성형 AI 사업 본격화…"패브릭스"와 "브리티 코파일럿" 기반 공공 디지털 혁신 지원'

    target_texts = [
        "신한투자-'삼성SDS, 생성형AI 솔루션으로 성장모멘텀…목표가↑'",
        "AI가 年 310조 경제효과 창출…정부, AI 일상화에 7102억 투입",
        "1114회 로또 1등 각 15억원씩…1곳서 수동 5명(종합)",
        "교보증권, 디지털 전환 가속화…생성형 AI 활용 사내교육 실시",
    ]

    # 쿼리 임베딩
    query_emb = get_embedding(query)
    query_emb_2d = query_emb.reshape(1, -1)   # 거리 함수 형태 맞추기

    # 후보 텍스트 임베딩
    target_embeds = get_embeddings(target_texts)

    # 거리 계산
    cosine_dists    = cosine_distance(query_emb, target_embeds)
    euclidean_dists = euclidean_distance(query_emb, target_embeds)

    print(f"  Query: {query}\n  {'─'*50}")

    for i, text in enumerate(target_texts):
        print(f"  {text}")
        print(f"  코사인 거리: {cosine_dists[i]:.4f}  |  유클리드 거리: {euclidean_dists[i]:.4f}")
        print(f"  {'─'*50}")

    # 가장 유사한 문장
    best_idx = int(np.argmin(cosine_dists))
    print(f"\n  ✅ 가장 유사한 텍스트 (코사인 거리 기준):")
    print(f"  → [{best_idx}] {target_texts[best_idx]}")
    print()


# ══════════════════════════════════════════════════════════
def step5_simple_search_engine() -> None:
    """
    [STEP 5] 유사도 기반 간단한 텍스트 검색 엔진
    -----------------------------------------------
    임베딩을 활용한 시맨틱 검색(Semantic Search)을 구현합니다.
    키워드 일치가 아닌 '의미' 기반으로 가장 관련 있는 문서를 찾습니다.
    이것이 RAG(Retrieval-Augmented Generation)의 핵심 원리입니다.
    """
    print("=" * 55)
    print("[STEP 5] 유사도 기반 간단한 텍스트 검색 엔진")
    print("=" * 55)

    # 가상 문서 DB (실제 RAG 에서는 수백~수천 개의 문서가 들어갑니다)
    documents = [
        "파이썬은 간결한 문법으로 초보자도 배우기 쉬운 프로그래밍 언어입니다.",
        "딥러닝은 인공 신경망을 여러 층으로 쌓아 복잡한 패턴을 학습합니다.",
        "RAG는 LLM에 외부 문서 검색 기능을 결합해 최신 정보를 활용합니다.",
        "도커(Docker)는 애플리케이션을 컨테이너로 패키징하여 배포를 단순화합니다.",
        "트랜스포머 아키텍처는 어텐션 메커니즘을 기반으로 한 딥러닝 모델입니다.",
        "SQL은 관계형 데이터베이스에서 데이터를 조회·수정하는 언어입니다.",
    ]

    queries = [
        "LLM 을 더 똑똑하게 만드는 방법이 있나요?",
        "데이터를 표 형태로 저장하고 검색하려면?",
        "뉴럴넷을 여러 층으로 쌓는 기술은?",
    ]

    # 문서 임베딩 (실제 RAG 에서는 이 단계를 미리 계산하여 DB에 저장합니다)
    print("  문서 DB 임베딩 중...\n")
    doc_embs = get_embeddings(documents)

    for query in queries:
        query_emb = get_embedding(query)
        distances = cosine_distance(query_emb, doc_embs)
        ranked = np.argsort(distances)  # 거리 오름차순 정렬

        print(f"  🔍 질문: {query}")
        print(f"  📄 가장 관련 있는 문서 Top-2:")
        for rank, idx in enumerate(ranked[:2], start=1):
            print(f"    {rank}위 (거리: {distances[idx]:.4f}) → {documents[idx]}")
        print()

    print("  → 이것이 RAG 의 'Retrieval' 단계입니다.")
    print("  → 검색된 문서를 GPT 에 전달하면 최신 정보 기반 응답이 가능합니다.")
    print()


# ══════════════════════════════════════════════════════════
def step6_structured_output_basic() -> None:
    """
    [STEP 6] Structured Output — Pydantic 으로 JSON 출력 보장
    -----------------------------------------------------------
    일반 Chat Completions 는 자유 텍스트를 반환하므로
    JSON 파싱 실패 위험이 있습니다.

    client.beta.chat.completions.parse() 를 사용하면:
      - response_format 에 Pydantic 모델을 지정
      - API 가 스키마를 보장하여 항상 파싱 가능한 JSON 반환
      - .parsed 속성으로 바로 Pydantic 객체 접근 가능
    """
    print("=" * 55)
    print("[STEP 6] Structured Output — 기본 사용법")
    print("=" * 55)

    # ── Pydantic 스키마 정의 ──
    class MovieInfo(BaseModel):
        제목: str
        감독: str
        장르: list[str]
        줄거리_요약: str
        개봉연도: int

    user_text = """
    인셉션(Inception)은 크리스토퍼 놀란 감독의 2010년 작품입니다.
    꿈 속에 침투해 생각을 심는 '인셉션' 기술을 다루는 SF 스릴러로,
    레오나르도 디카프리오가 주연을 맡았습니다.
    """

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "다음 텍스트를 분석하세요. 새로운 내용은 추가하지 마세요."},
            {"role": "user",   "content": user_text},
        ],
        response_format=MovieInfo,
    )

    movie: MovieInfo = completion.choices[0].message.parsed

    print("  [입력 텍스트]")
    print(f"  {user_text.strip()}\n")
    print("  [파싱 결과 (Pydantic 객체)]")
    print(f"  제목        : {movie.제목}")
    print(f"  감독        : {movie.감독}")
    print(f"  장르        : {movie.장르}")
    print(f"  줄거리 요약 : {movie.줄거리_요약}")
    print(f"  개봉연도    : {movie.개봉연도}")
    print()
    print("  [JSON 직렬화]")
    print(f"  {movie.model_dump_json(indent=2)}")
    print()


# ══════════════════════════════════════════════════════════
def step7_structured_output_nested() -> None:
    """
    [STEP 7] 중첩 Pydantic 모델 — 여행지 정보 파싱 (노트북 예제)
    --------------------------------------------------------------
    노트북 원본 예제를 그대로 실습합니다.
    TravelDesc(장소 단위) → TravelInfo(장소 목록) 중첩 구조.
    여러 문서에서 구조화된 데이터를 추출하는 전형적인 패턴입니다.
    """
    print("=" * 55)
    print("[STEP 7] 중첩 Pydantic 모델 — 여행지 정보 파싱")
    print("=" * 55)

    # ── Pydantic 스키마 정의 (노트북 원본) ──
    class TravelDesc(BaseModel):
        장소명: str
        특징: list[str]
        여행팁: list[str]

    class TravelInfo(BaseModel):
        여행지: list[TravelDesc]

    venice_text = """베니스는 이탈리아의 대표적인 관광도시이다.
117개의 작은 섬으로 이루어져 있으며 운하와 다리로 연결되어 있다.
산마르코 광장, 두칼레 궁전 등 중세 시대의 건축물이 잘 보존되어 있다.
곤돌라를 타고 운하를 둘러보는 것이 대표적인 관광 활동이며
세계적으로 유명한 베니스 카니발이 매년 2월에 열린다."""

    santorini_text = """산토리니는 그리스의 대표적인 휴양지이다.
에게해의 칼데라 절벽 위에 하얀 집들이 늘어서 있는 풍경으로 유명하다.
화산 폭발로 형성된 독특한 지형과 아름다운 석양이 매력적이다.
피라와 이아 마을의 절벽 위 산책로는 관광객들에게 인기가 높다.
성수기인 7-8월은 매우 혼잡하고 숙박비가 비싸므로 봄이나 가을에 방문하는 것이 좋다."""

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "다음 데이터를 분석하세요. 새로운 내용은 추가하지 마세요."},
            {"role": "user",   "content": venice_text},
            {"role": "user",   "content": santorini_text},
        ],
        response_format=TravelInfo,
    )

    travelinfo: TravelInfo = completion.choices[0].message.parsed

    print(f"  파싱된 여행지 수: {len(travelinfo.여행지)}개\n")
    for place in travelinfo.여행지:
        print(f"  ✈️  {place.장소명}")
        print(f"  특징:")
        for f in place.특징:
            print(f"    - {f}")
        print(f"  여행 팁:")
        for t in place.여행팁:
            print(f"    - {t}")
        print()


# ══════════════════════════════════════════════════════════
def step8_structured_review_classifier() -> None:
    """
    [STEP 8] Structured Output 활용 — 상품 리뷰 자동 분류기
    ---------------------------------------------------------
    Structured Output 의 실무 활용 예시입니다.
    리뷰 텍스트를 입력하면 감성(긍정/부정/중립), 별점, 핵심 키워드,
    개선 요청 여부를 자동으로 추출합니다.
    """
    print("=" * 55)
    print("[STEP 8] Structured Output 활용 — 리뷰 분류기")
    print("=" * 55)

    # ── Pydantic 스키마 정의 ──
    class ReviewAnalysis(BaseModel):
        감성: str           # "긍정" | "부정" | "중립"
        별점_예측: int      # 1 ~ 5
        핵심_키워드: list[str]
        개선_요청_여부: bool
        한줄_요약: str

    reviews = [
        "배송이 엄청 빨랐고 포장도 꼼꼼했어요! 제품 품질도 기대 이상입니다. 다음에도 꼭 구매할게요.",
        "사진과 실제 색상이 너무 달라요. 반품하고 싶은데 고객센터 연결도 안 되고 정말 실망입니다.",
        "평범한 제품이에요. 가격 대비 나쁘지 않은데 특별히 좋은 점도 없어요.",
    ]

    for i, review in enumerate(reviews, start=1):
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "다음 상품 리뷰를 분석하세요."},
                {"role": "user",   "content": review},
            ],
            response_format=ReviewAnalysis,
        )
        result: ReviewAnalysis = completion.choices[0].message.parsed

        print(f"  [리뷰 {i}] {review[:35]}{'...' if len(review) > 35 else ''}")
        print(f"  감성        : {result.감성}")
        print(f"  별점 예측   : {'★' * result.별점_예측}{'☆' * (5 - result.별점_예측)} ({result.별점_예측}/5)")
        print(f"  핵심 키워드 : {', '.join(result.핵심_키워드)}")
        print(f"  개선 요청   : {'있음' if result.개선_요청_여부 else '없음'}")
        print(f"  한줄 요약   : {result.한줄_요약}")
        print()

    print("  → 이 패턴은 CS 자동화, 감성 분석, 데이터 파이프라인에 바로 활용 가능합니다.")
    print()


# ══════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "★" * 55)
    print("   주제 5 — 고급 기능 (Embedding & Structured Output)")
    print("★" * 55 + "\n")

    # ── Embedding 파트 ──
    step1_embedding_basics()
    step2_model_comparison()
    step3_distance_metrics()
    step4_news_similarity()
    step5_simple_search_engine()

    # ── Structured Output 파트 ──
    step6_structured_output_basic()
    step7_structured_output_nested()
    step8_structured_review_classifier()

    print("=" * 55)
    print("주제 5 실습 완료 ✅")
    print("=" * 55)


if __name__ == "__main__":
    main()