📚 학습 주제 분류 (5개)
Topic 1. LCEL 기본 구조 & 체인 구성

| 파이프 연산자로 Prompt + LLM 연결
invoke() 실행, 매개변수 1개 / 2개 체인 실습

Topic 2. Output Parser — 출력 형식 제어

StrOutputParser (문자열 변환)
JsonOutputParser (JSON 변환)
Pydantic BaseModel로 JSON 스키마 고정
with_structured_output() (Structured Output)

Topic 3. 실전 실습 — 리뷰 답변 생성기

CSV에서 리뷰 로드 후 체인 적용
batch()로 다수 입력 일괄 처리

Topic 4. Runnables — 데이터 흐름 제어

RunnablePassthrough (이전 출력 그대로 전달)
RunnableParallel (병렬 체인 실행)
.assign() (기존 결과에 새 체인 결과 추가)
Lambda로 dict 값 추출

Topic 5. 멀티 체인 연결 패턴

chain1 결과 → chain2 입력으로 연결
JsonOutputParser + 체인 간 데이터 전달 응용



너는 llm 및 ai api 전문가야. 
내가 llm 및 ai api 관련 코드가 담긴 '.ipynb' 파일을 주면 너가 보고 분석한 후에 주제별로 정리해 줄 수 있어?
내가 코드파일을 줄테니 이 코드가 전체적으로 어떤 것을 하는지 주제별로 주요 기능들 요약 정리 해줘.

//

//

너는 llm 및 ai api 전문가야. 
내가 llm 및 ai api 관련 코드가 담긴 '.ipynb' 파일을 주면 너가 보고 분석할 수 있어?
내가 코드파일을 줄테니 이 코드가 전체적으로 어떤 것을 하는지 설명해 주고 주요 기능들 요약 정리 해줘.

//

그러면 이 코드에서 배울 수 있는 것을 주제로 나누고 싶어. 배울 수 있는 주제를 무엇 무엇으로 나눌 수 있어? 코드에 나온 순차적인 순서로 주제를 나눠주되, 묶을 수 있는 것은 좀 묶어줄 수 있어?

//

[요청사항]
아까 너가 제시한 주제들 중 
'
Topic 4. Runnables — 데이터 흐름 제어

RunnablePassthrough (이전 출력 그대로 전달)
RunnableParallel (병렬 체인 실행)
.assign() (기존 결과에 새 체인 결과 추가)
Lambda로 dict 값 추출
' 
을 코딩을 하고 싶어. 

---
[코딩 참고사항]
1.이 주제로 실행 가능하도록 하나의 '.py' 파일을 만들고 싶어.
2.내가 준 .ipynb 파일의 코드를 최대한 활용해줘.
3.API_KEY 를 불러오는 방식은 '.env' 파일을 생성하고 'from dotenv import load_dotenv'라는 코드를 활용해줘.
4.참고로 파이썬 3.11 버전을 쓰고 있어.
