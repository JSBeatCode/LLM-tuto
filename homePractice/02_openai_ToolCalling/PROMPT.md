📚 최종 학습 주제 분류 (업데이트)
Topic 1 — OpenAI API 기본 설정

라이브러리 설치, API 키 설정 및 검증, 클라이언트 초기화

Topic 2 — LLM으로 코드 생성하기

코드 생성 프롬프트 설계, temperature=0 활용, 생성 코드 재사용 패턴

Topic 3 — 외부 API 연동 및 데이터 가공

네이버 뉴스 API 호출, JSON 파싱, get_news(query) 함수 구현

Topic 4 — 뉴스 요약 봇 기초 + 스트리밍

검색 결과를 컨텍스트로 주입하는 패턴, multi-turn 메시지 구조, stream=True 처리

Topic 5 — Tool Calling 기초

Tool 개념과 동작 원리, Pydantic으로 Tool 스키마 정의, tool_choice 옵션

Topic 6 — Parallel Tool Call ← ✅ 추가

GPT가 여러 tool_calls를 한 번에 반환하는 구조 이해
for 루프로 다수 tool 결과를 일괄 수집 후 한꺼번에 전달
한계점: 입력 컨텍스트가 길어져 할루시네이션 발생 가능

Topic 7 — Sequential Tool Call (parallel_tool_calls=False) ← ✅ 추가

parallel_tool_calls=False 옵션으로 순차 실행 강제
while tool_calls: 루프로 Tool → 메시지 → Tool 반복 처리
available_functions 딕셔너리로 tool-함수 매핑

//

너는 llm 및 ai api 전문가야. 
내가 llm 및 ai api 관련 코드가 담긴 '.ipynb' 파일을 주면 너가 보고 분석할 수 있어?
내가 코드파일을 줄테니 이 코드가 전체적으로 어떤 것을 하는지 설명해 주고 주요 기능들 요약 정리 해줘.

//

그러면 이 코드에서 배울 수 있는 것을 주제로 나누고 싶어. 배울 수 있는 주제를 무엇 무엇으로 나눌 수 있어? 코드에 나온 순차적인 순서로 주제를 나눠주되, 묶을 수 있는 것은 좀 묶어줄 수 있어?

//

그러면 너가 제시한 주제로 분류하자.
그러면 이제 각 주제별로 코딩을 하고 싶어. 
각 주제별로 실행 가능하도록 하나씩 '.py' 파일을 만들고 싶어.
내가 준 .ipynb 파일의 코드를 최대한 활용해줘.
API_KEY 를 불러오는 방식은 '.env' 파일을 생성하고 'from dotenv import load_dotenv'라는 코드를 활용해줘.
참고로 파이썬 3.11 버전을 쓰고 있어.

---

먼저 주제1 부터 만들어 줘.

//

[요청사항]
아까 너가 제시한 주제들 중 'Topic 7 — Sequential Tool Call (parallel_tool_calls=False) ← ✅ 추가' 코딩을 하고 싶어. 

---
[코딩 참고사항]
1.이 주제로 실행 가능하도록 하나의 '.py' 파일을 만들고 싶어.
2.내가 준 .ipynb 파일의 코드를 최대한 활용해줘.
3.API_KEY 를 불러오는 방식은 '.env' 파일을 생성하고 'from dotenv import load_dotenv'라는 코드를 활용해줘.
4.참고로 파이썬 3.11 버전을 쓰고 있어.
