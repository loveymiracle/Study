# 01. OpenAI Agent (Agents Overview)

# OpenAI 플랫폼을 사용하여 사용자를 대신해 행동(예: 컴퓨터 제어 등)을 수행할 수 있는 **에이전트(Agent)** 를 구축할 수 있습니다.  
# Python용 **Agent SDK**를 사용하면 백엔드에서 이러한 에이전트의 **오케스트레이션(조율) 로직**을 만들 수 있습니다.

**OpenAI Agents SDK**는 **에이전트 기반 AI 애플리케이션**을 개발할 수 있게 해주는 도구입니다.  

### 주요 기능 요약:

# - **에이전트 루프 (Agent Loop)**:  
#  도구 실행 → 결과 전달 → LLM 호출 → 반복 실행 → 완료까지 자동 처리

# - **핸드오프 (Handoffs)**:  
#  여러 에이전트 간의 **협업과 위임**을 유연하게 처리 가능

# - **가드레일 (Guardrails)**:  
#  에이전트 입력을 **사전 검사/검증**하여, 조건을 만족하지 않으면 **조기 종료 가능**

# - **함수 기반 도구 (Function Tools)**:  
#  Python 함수 하나를 **자동으로 에이전트 도구로 변환**,  
#  **Pydantic 기반 스키마 자동 생성** 및 검증 포함

# - **추적(Tracing)**:  
#  워크플로우를 **시각화/디버깅/모니터링** 가능하며,
#  OpenAI의 평가/파인튜닝/디스틸레이션 툴과 통합 가능

from dotenv import load_dotenv
load_dotenv() 

import openai

Model = "gpt-5-mini"

### Hello World 예제

# | 메서드 | 호출 방식 | 특징 | 언제 쓰나 | 핵심 차이 |
# |---|---|---|---|---|
# | `Runner.run(...)` | `await Runner.run(...)` | 비동기적, 에이전트 루프 자동 실행, 도구 & 핸드오프 지원 | FastAPI, Jupyter, 서버 환경 | `await` 필요, 비동기 환경용 |
# | `Runner.run_sync(...)` | `Runner.run_sync(...)` | 동기 실행으로 첫 번째 메서드 래핑, 스크립트/테스트 환경 적합 | 일반 Python 스크립트, 테스트 | `await` 불필요, 동기 환경용 |
# | `Runner.run_streamed(...)` | `await Runner.run_streamed(...)` | 중간 응답을 이벤트로 실시간 전송 가능 | 챗봇 UI, 실시간 응답이 필요한 서비스 | 답변을 조각조각 실시간 수신 |
# - Jupyter notebook은 기본적으로 이벤트 루프가 이미 실행 중이므로  `await Runner.run(...)` 사용  |


# Agent(에이전트 정의)와 Runner(실행 관리자) 불러오기
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="당신은 도움되는 도우미입니다.",
    model=Model
)

# 비동기적으로 에이전트를 실행하여 사용자 요청에 대한 응답을 받음
# 요청: "재귀적 프로그래밍에 대한 짧은 시를 3줄 이내로 써주세요."
result = await Runner.run(starting_agent=agent, 
                          input="재귀적 프로그래밍에 대한 짧은 시를 3줄 이내로 써주세요.")

# 최종 응답 결과를 출력
print(result.final_output)


### Simple Handoff Example

# 언어에 따라 적절한 에이전트에 작업을 위임(handoff)합니다.

# Handoffs는 LLM에게 **도구(tool)** 로 표현됩니다.  
# 예) `Korean agent`에 대한 핸드오프 → LLM 도구 이름: `transfer_to_korean_agent`
# ```
# 도구 이름 자동 생성 규칙 (에이전트 이름 → 도구 이름 자동 변환)
# "Korean agent"   →  transfer_to_korean_agent
# "Billing agent"  →  transfer_to_billing_agent
# "English agent"  →  transfer_to_english_agent
# ```

# **핸드오프 지정 방법 2가지:**
# 1. **Agent 인스턴스 직접 전달** : `handoffs=[korean_agent, english_agent]`
# 2. **`handoff()` 함수 사용** : `handoffs=[handoff(agent, on_handoff=콜백, ...)]`  
#  → 콜백(`on_handoff`), 도구 이름/설명 재정의, 입력 데이터 타입, 입력 필터 등 **고급 옵션** 제공  
#  → 심화 내용은 `04_Handoffs.py` 참고

1. Agent 인스턴스 직접 전달 예시
from agents import Agent, Runner

# 한국어 에이전트 생성: 한국어만 사용 가능
korean_agent = Agent(
    name="Korean agent",
    instructions="당신은 한국어만 할 수 있습니다.",
    model=Model
)

# 영어 에이전트 생성: 영어만 사용 가능
english_agent = Agent(
    name="English agent",
    instructions="당신은 영어만 할 수 있습니다.",
    model=Model
)

# 분류 역할의 핸드오프 에이전트 생성
# 입력된 문장의 언어를 판별하여 적절한 에이전트(한국어 or 영어)에게 전달
handoff_agent = Agent(
    name="Classify agent",
    instructions="요청에 사용된 언어에 따라 적절한 에이전트에게 넘겨주세요.",
    model=Model,
    handoffs=[korean_agent, english_agent],  # 연결할 하위 에이전트 목록
)

# Agent orchenstration 실행
result = await Runner.run(handoff_agent, input="당신은 행복합니까?")
print(result.final_output)  # 한국어 에이전트가 응답
print()
result = await Runner.run(handoff_agent, input="Are you happy?")
print(result.final_output)  # 영어 에이전트가 응답
