📚 학습 주제 분류 (총 4개)

Topic 1. LangChain 환경 설정 & LLM 기본 호출

패키지 설치 및 API Key 설정
ChatOpenAI로 모델 인스턴스 생성 (gpt-4o-mini, o1-mini)
.invoke()로 LLM 호출 및 AIMessage 응답 처리
.content로 텍스트 추출


💡 "LangChain으로 LLM을 처음 연결하고 호출하는 방법"


Topic 2. Prompt Template

PromptTemplate — 단일 변수/다중 변수 템플릿 작성 및 .format()
ChatPromptTemplate — system/user 역할 구분, .format_messages()
두 템플릿의 차이점 및 올바른 사용법 비교


💡 "프롬프트를 재사용 가능한 구조로 설계하는 방법"


Topic 3. 실전 프로젝트 — 감성 분류기 & CoT 프롬프트 엔지니어링

CSV 리뷰 데이터 로드 및 LLM 기반 긍정/부정 자동 분류
정확도 측정 루프 및 evaluate() 함수 모듈화
기본 프롬프트 → CoT 요약 방식 → 항목별 구조화 방식으로 성능 개선 실험


💡 "실제 데이터에 LangChain을 적용하고, 프롬프트 엔지니어링으로 성능을 높이는 방법"


Topic 4. Few-Shot Prompting

FewShotPromptTemplate으로 예시 기반 프롬프트 구성
prefix + examples + suffix 구조 이해
Step-by-step 추론 유도 및 복잡한 질문 처리


💡 "예시를 제공해서 LLM의 추론 방식을 원하는 형태로 유도하는 방법"


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
Topic 4. Few-Shot Prompting

FewShotPromptTemplate으로 예시 기반 프롬프트 구성
prefix + examples + suffix 구조 이해
Step-by-step 추론 유도 및 복잡한 질문 처리


💡 "예시를 제공해서 LLM의 추론 방식을 원하는 형태로 유도하는 방법"
' 
을 코딩을 하고 싶어. 

---
[코딩 참고사항]
1.이 주제로 실행 가능하도록 하나의 '.py' 파일을 만들고 싶어.
2.내가 준 .ipynb 파일의 코드를 최대한 활용해줘.
3.API_KEY 를 불러오는 방식은 '.env' 파일을 생성하고 'from dotenv import load_dotenv'라는 코드를 활용해줘.
4.참고로 파이썬 3.11 버전을 쓰고 있어.
