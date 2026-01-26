# CrewAI Agent 개발 템플릿 가이드

이 문서는 `src/` 폴더의 영수증 추출 에이전트 구조를 기반으로,
**다른 파트의 에이전트들도 동일한 패턴으로 개발**할 수 있도록 정리한 가이드입니다.

---

## 📁 표준 폴더 구조

```
your_agent_project/
├── .env                          # 환경 변수 (API 키 등)
├── requirements.txt              # 의존성 패키지
├── README.md                     # 프로젝트 설명
│
└── src/
    ├── main.py                   # 진입점 (CLI)
    ├── crew.py                   # Crew 정의 및 실행 로직
    │
    ├── config/
    │   ├── agents.yaml           # Agent 정의
    │   └── tasks.yaml            # Task 정의
    │
    └── tools/
        ├── __init__.py           # Tool 패키지 초기화
        ├── your_tool_1.py        # 커스텀 Tool 1
        └── your_tool_2.py        # 커스텀 Tool 2
```

---

## 🛠️ 1단계: Tool 파일 작성

### 파일 위치: `src/tools/your_tool.py`

```python
"""
YourCustomTool - 도구 설명
"""
from crewai.tools import BaseTool


class YourCustomTool(BaseTool):
    """도구의 상세 설명"""
    
    name: str = "Your Tool Name"
    description: str = (
        "이 도구가 무엇을 하는지 에이전트가 이해할 수 있도록 명확하게 작성. "
        "어떤 입력을 받고 어떤 출력을 반환하는지 설명."
    )
    
    def _run(self, param1: str, param2: str = "") -> str:
        """
        도구의 핵심 로직
        
        Args:
            param1: 필수 파라미터 설명
            param2: 선택 파라미터 설명
            
        Returns:
            처리 결과 문자열 (JSON 권장)
        """
        if not param1:
            return "Error: param1이 제공되지 않았습니다."
        
        try:
            # 여기에 실제 로직 구현
            result = f"처리 결과: {param1}"
            return result
            
        except Exception as e:
            return f"Error: {str(e)}"
```

### 파일 위치: `src/tools/__init__.py`

```python
"""
Tools package initialization
"""
from .your_tool import YourCustomTool

__all__ = ['YourCustomTool']
```

---

## 👤 2단계: Agent 정의 (YAML)

### 파일 위치: `src/config/agents.yaml`

```yaml
# Agent 이름 (crew.py에서 참조할 키)
your_agent_name:
  role: >
    에이전트의 역할 (예: Data Analyst, Code Reviewer 등)
  
  goal: >
    에이전트가 달성해야 할 목표를 구체적으로 작성.
    무엇을 어떻게 처리해야 하는지 명확하게.
  
  backstory: >
    에이전트의 배경 스토리. 전문성과 접근 방식을 설명.
    
    [핵심 원칙]
    1) 첫 번째 원칙
    2) 두 번째 원칙
    3) 세 번째 원칙
    
    [주의사항]
    - 하지 말아야 할 것들
    - 우선순위 규칙
  
  verbose: true
  allow_delegation: false
  max_iter: 10
```

### 💡 Agent 정의 팁

| 항목 | 설명 |
|------|------|
| `role` | 에이전트의 직책/역할 (1줄) |
| `goal` | 달성할 목표 (2~3줄) |
| `backstory` | 상세 지침, 원칙, 주의사항 (길게 작성 가능) |
| `verbose` | true면 상세 로그 출력 |
| `allow_delegation` | 다른 에이전트에게 위임 허용 여부 |
| `max_iter` | 최대 반복 횟수 |

---

## 📋 3단계: Task 정의 (YAML)

### 파일 위치: `src/config/tasks.yaml`

```yaml
# Task 이름 (crew.py에서 참조할 키)
your_task_name:
  description: >
    태스크 상세 설명.
    
    입력:
    - input_param: {input_param}
    - retry_notes: {retry_notes}
    
    ===========================
    [처리 절차]
    ===========================
    
    STEP 1) 첫 번째 단계
    - 세부 지침 1
    - 세부 지침 2
    
    STEP 2) 두 번째 단계
    - 세부 지침 1
    - 세부 지침 2
    
    ===========================
    [검증 규칙]
    ===========================
    1) 정규식 검증: XXX 형식
    2) 산술 검증: A + B = C
    3) 필수 필드 확인
    
    ===========================
    [출력 형식]
    ===========================
    반드시 아래 형식으로 출력:
    {
      "field1": "value1",
      "field2": "value2"
    }
  
  expected_output: >
    기대하는 출력 형식 설명 (예: 유효한 JSON 객체)
  
  agent: your_agent_name  # agents.yaml의 키와 일치해야 함
```

### 💡 Task 정의 팁

- `{변수명}` 형태로 런타임 변수 주입 가능
- `description`에 충분히 상세한 지침 작성
- `expected_output`은 간결하게 요약

---

## ⚙️ 4단계: Crew 정의

### 파일 위치: `src/crew.py`

```python
"""
Your Agent Crew - 설명
"""
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from crewai import Agent, Task, Crew

# Tool import
from tools.your_tool import YourCustomTool


def _load_yaml(file_path: str) -> dict:
    """YAML 파일 로드"""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_output(data: Dict[str, Any]) -> List[str]:
    """
    출력 검증 로직 (가드레일 역할)
    실패 사유를 문자열 리스트로 반환
    """
    errors: List[str] = []
    
    # 필수 키 확인
    required_keys = ["field1", "field2"]
    for k in required_keys:
        if k not in data:
            errors.append(f"Missing key: {k}")
    
    # 추가 검증 로직...
    
    return errors


class YourAgentCrew:
    """에이전트 Crew 클래스"""
    
    def __init__(self):
        base_dir = Path(__file__).parent
        self.agents_config = _load_yaml(str(base_dir / "config" / "agents.yaml"))
        self.tasks_config = _load_yaml(str(base_dir / "config" / "tasks.yaml"))
        
        # Tool 인스턴스 생성
        self.your_tool = YourCustomTool()
    
    def _create_agent(self, agent_name: str, agent_config: dict) -> Agent:
        """Agent 생성"""
        return Agent(
            role=agent_config.get("role", ""),
            goal=agent_config.get("goal", ""),
            backstory=agent_config.get("backstory", ""),
            verbose=agent_config.get("verbose", False),
            allow_delegation=agent_config.get("allow_delegation", False),
            max_iter=agent_config.get("max_iter", 10),
        )
    
    def _create_task(self, task_name: str, task_config: dict, agent: Agent) -> Task:
        """Task 생성"""
        return Task(
            description=task_config.get("description", ""),
            expected_output=task_config.get("expected_output", ""),
            agent=agent,
            tools=[self.your_tool],  # 사용할 Tool 목록
        )
    
    def create_crew(self) -> Crew:
        """Crew 생성"""
        agents = {}
        for agent_name, agent_config in self.agents_config.items():
            agents[agent_name] = self._create_agent(agent_name, agent_config)
        
        tasks: List[Task] = []
        for task_name, task_config in self.tasks_config.items():
            agent_name = task_config.get("agent")
            if agent_name not in agents:
                raise ValueError(f"Task '{task_name}' references unknown agent '{agent_name}'")
            tasks.append(self._create_task(task_name, task_config, agents[agent_name]))
        
        return Crew(agents=list(agents.values()), tasks=tasks, verbose=True)
    
    def run(self, input_param: str, max_attempts: int = 3) -> str:
        """
        실행 메서드 (검증 + 재시도 포함)
        """
        crew = self.create_crew()
        
        retry_notes = ""
        last_raw = ""
        
        for attempt in range(1, max_attempts + 1):
            inputs = {
                "input_param": input_param,
                "retry_notes": retry_notes,
            }
            raw = crew.kickoff(inputs=inputs)
            last_raw = str(raw)
            
            # JSON 파싱 시도
            try:
                parsed = json.loads(last_raw)
            except:
                retry_notes = f"[Attempt {attempt}] JSON 파싱 실패. 유효한 JSON만 출력하세요."
                continue
            
            # 검증
            errors = _validate_output(parsed)
            if not errors:
                return json.dumps(parsed, ensure_ascii=False)
            
            # 실패 시 재시도 노트 갱신
            retry_notes = (
                f"[Attempt {attempt}] 아래 오류를 수정하세요:\n"
                + "\n".join(f"- {e}" for e in errors)
            )
        
        return last_raw  # 최종 실패 시 원본 반환


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python crew.py <input>")
        sys.exit(1)
    
    crew = YourAgentCrew()
    result = crew.run(sys.argv[1])
    print(result)
```

---

## 🚀 5단계: 진입점 (main.py)

### 파일 위치: `src/main.py`

```python
"""
Your Agent System - Main Entry Point
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv(Path(__file__).parent.parent / ".env")

# src 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from crew import YourAgentCrew


def main():
    if len(sys.argv) < 2:
        print("=" * 60)
        print("Your Agent System")
        print("=" * 60)
        print("\n사용법:")
        print("  python main.py <input>")
        sys.exit(1)
    
    input_param = sys.argv[1]
    
    # API Key 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY 환경 변수를 설정해주세요.")
        sys.exit(1)
    
    print("🚀 에이전트 시작")
    print("-" * 60)
    
    try:
        crew = YourAgentCrew()
        result = crew.run(input_param)
        
        print("-" * 60)
        print("✅ 완료")
        print(result)
        
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## 📦 6단계: 의존성 파일

### 파일 위치: `requirements.txt`

```
crewai>=0.11.0
python-dotenv
pyyaml
openai
```

### 파일 위치: `.env`

```
OPENAI_API_KEY=your-api-key-here
```

---

## ✅ 체크리스트

새 에이전트 프로젝트 생성 시 확인할 사항:

- [ ] `src/tools/` 에 커스텀 Tool 클래스 작성
- [ ] `src/tools/__init__.py` 에서 Tool export
- [ ] `src/config/agents.yaml` 에 Agent 정의
- [ ] `src/config/tasks.yaml` 에 Task 정의
- [ ] `src/crew.py` 에서 Crew 클래스 구현
- [ ] `src/main.py` 진입점 작성
- [ ] `.env` 파일에 API 키 설정
- [ ] `requirements.txt` 의존성 정의

---

## 🔄 여러 에이전트 연동 (Multi-Agent)

여러 에이전트가 순차적으로 협업하는 경우:

### `agents.yaml` (여러 에이전트 정의)

```yaml
agent_1:
  role: Data Collector
  goal: 데이터 수집
  backstory: ...

agent_2:
  role: Data Analyzer
  goal: 수집된 데이터 분석
  backstory: ...

agent_3:
  role: Report Generator
  goal: 분석 결과로 보고서 생성
  backstory: ...
```

### `tasks.yaml` (순차 Task 정의)

```yaml
task_1:
  description: 데이터 수집
  agent: agent_1

task_2:
  description: 데이터 분석 (이전 Task 결과 사용)
  agent: agent_2

task_3:
  description: 보고서 생성
  agent: agent_3
```

### `crew.py`

```python
# Crew 생성 시 순서대로 tasks 배열에 추가
# CrewAI가 자동으로 순차 실행
crew = Crew(
    agents=[agent_1, agent_2, agent_3],
    tasks=[task_1, task_2, task_3],  # 순서대로 실행
    verbose=True
)
```

---

## 📚 참고 자료

- [CrewAI 공식 문서](https://docs.crewai.com/)
- [CrewAI Tools 문서](https://docs.crewai.com/tools/)
- 현재 프로젝트 예시: `src/` 폴더 참조
