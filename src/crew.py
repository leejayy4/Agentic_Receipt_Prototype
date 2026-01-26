"""
Receipt Recognition & Extraction Crew - Orchestrator
"""
import os
import sys
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from crewai import Agent, Task, Crew, LLM
from pydantic import BaseModel, Field

# Tool imports
from tools import (
    VisionClassificationTool,
    ImageQualityOptimizationTool,
    ReceiptSAMTool,
    VisionExtractionTool
)
from tools.compliance_tools import (
    ValidateBusinessNumberNTSTool,
    ValidateVATCalculationTool
)


class ReceiptData(BaseModel):
    """영수증 추출 데이터 모델"""
    business_number: str = Field(..., description="사업자등록번호 (XXX-XX-XXXXX)")
    store_name: str = Field(..., description="상호명")
    total_amount: int = Field(..., description="총 결제 금액 (정수)")
    transaction_datetime: str = Field(..., description="거래일시 (YYYY-MM-DD HH:MM:SS)")
    vat_amount: Optional[int] = Field(None, description="부가세 (정수)")


class ComplianceResult(BaseModel):
    """컴플라이언스 검증 결과 모델 (추출 데이터 포함)"""
    extracted_data: ReceiptData = Field(..., description="추출된 5가지 핵심 영수증 데이터")
    needs_review: bool = Field(..., description="추가 검토 필요 여부 (사업자 폐업이나 산술 오류 시 True)")
    errors: List[str] = Field(default_factory=list, description="발견된 오류 목록")
    nts_status: str = Field(..., description="국세청 사업자 상태")
    vat_validation: str = Field(..., description="부가세 산술 검증 결과")


def _load_yaml(file_path: str) -> dict:
    """YAML 파일 로드"""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """모델 출력에서 JSON 객체를 추출합니다."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(0).strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _to_int_amount(v: Any) -> Optional[int]:
    """금액 데이터를 정수로 변환합니다."""
    if v is None: return None
    if isinstance(v, (int, float)): return int(round(v))
    if isinstance(v, str):
        s = v.strip().replace(",", "").replace(" ", "")
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
    return None


def _normalize_business_number(v: Any) -> Optional[str]:
    """사업자등록번호 형식을 보정합니다 (XXX-XX-XXXXX)."""
    if v is None: return None
    s = str(v).strip().replace(" ", "")
    if re.fullmatch(r"\d{3}-\d{2}-\d{5}", s):
        return s
    digits = re.sub(r"\D", "", s)
    if len(digits) == 10:
        return f"{digits[0:3]}-{digits[3:5]}-{digits[5:10]}"
    return s


def _validate_fields(data: Dict[str, Any]) -> List[str]:
    """영수증 데이터의 유효성을 검증합니다 (5개 필드)."""
    errors: List[str] = []
    
    # 5개 필드만 검증
    required_keys = ["business_number", "store_name", "total_amount", "transaction_datetime", "vat_amount"]
    
    # 사업자번호 검증 및 보정
    bn = _normalize_business_number(data.get("business_number"))
    if bn and not re.fullmatch(r"\d{3}-\d{2}-\d{5}", bn):
        errors.append(f"사업자번호 형식이 올바르지 않습니다: {bn}")
    data["business_number"] = bn

    # 거래일시 형식 검증
    dt = data.get("transaction_datetime")
    if dt and not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", str(dt)):
        errors.append(f"거래일시 형식이 올바르지 않습니다: {dt} (YYYY-MM-DD HH:MM:SS 필요)")

    # 금액 정수 변환
    total = _to_int_amount(data.get("total_amount"))
    vat = _to_int_amount(data.get("vat_amount"))
    
    if total is not None:
        data["total_amount"] = total
    if vat is not None:
        data["vat_amount"] = vat
    
    return errors


def save_result_callback(result):
    """최종 결과를 파일로 저장하는 콜백 함수"""
    try:
        output_file = Path("output_result.json")
        content = str(result)
        
        # JSON 형식인 경우 예쁘게 저장 시도
        try:
            # Markdown 코드 블록 제거 로직 재사용
            m = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(content)
        except Exception:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
                
        print(f"\n[Hook] 최종 결과가 '{output_file}'에 저장되었습니다.")
    except Exception as e:
        print(f"\n[Hook Error] 결과 저장 중 오류 발생: {e}")
    return result


class ReceiptProcessingCrew:
    """Receipt Classification, Optimization and Extraction Crew Class"""
    
    def __init__(self):
        base_dir = Path(__file__).parent
        self.agents_config = _load_yaml(str(base_dir / "config" / "agents.yaml"))
        self.tasks_config = _load_yaml(str(base_dir / "config" / "tasks.yaml"))
        
        # Tool Instances
        self.vision_tool = VisionClassificationTool()
        self.optimization_tool = ImageQualityOptimizationTool()
        self.sam_tool = ReceiptSAMTool()
        self.extraction_tool = VisionExtractionTool()
        self.nts_tool = ValidateBusinessNumberNTSTool()
        self.vat_tool = ValidateVATCalculationTool()
    
    def _create_agent(self, agent_name: str, agent_config: dict) -> Agent:
        """Create Agent with appropriate tools"""
        tools = []
        if agent_name == "receipt_classifier":
            tools = [self.vision_tool]
        elif agent_name == "image_quality_optimizer":
            tools = [self.sam_tool, self.optimization_tool]
        elif agent_name == "receipt_extraction_specialist":
            tools = [self.extraction_tool]
        elif agent_name == "compliance_auditor":
            tools = [self.nts_tool, self.vat_tool]
            
        return Agent(
            role=agent_config.get("role", ""),
            goal=agent_config.get("goal", ""),
            backstory=agent_config.get("backstory", ""),
            max_iter=agent_config.get("max_iter", 25),
            max_rpm=agent_config.get("max_rpm"),
            tools=tools,
            llm=LLM(model=agent_config.get("llm", "gpt-4o-mini"))
        )
    
    def _create_task(self, task_name: str, task_config: dict, agent: Agent, context: List[Task] = None, guardrails: List[Any] = None, output_pydantic: Any = None) -> Task:
        """Create Task with context, guardrail and structured output support"""
        return Task(
            description=task_config.get("description", ""),
            expected_output=task_config.get("expected_output", ""),
            agent=agent,
            context=context,
            guardrails=guardrails,
            output_pydantic=output_pydantic
        )
    
    def run(self, image_path: str) -> str:
        """Execute the full processing pipeline."""
        agents = {}
        for agent_name, agent_config in self.agents_config.items():
            agents[agent_name] = self._create_agent(agent_name, agent_config)

        # 1. Classification & Optimization Tasks
        classify_task = self._create_task("classify_receipt_task", self.tasks_config["classify_receipt_task"], agents["receipt_classifier"])
        optimize_task = self._create_task("optimize_receipt_image_task", self.tasks_config["optimize_receipt_image_task"], agents["image_quality_optimizer"], context=[classify_task])
        
        # 2. Extraction Task (with Guardrail)
        def validation_guardrail(task_output):
            """Validate extracted JSON against critical fields."""
            text = task_output.raw
            parsed = _extract_json_object(text)
            
            if parsed is None:
                 return (False, "JSON Parsing Failed: Return ONLY a valid JSON object.")
                 
            errors = _validate_fields(parsed)
            if errors:
                return (False, "Validation Errors:\n" + "\n".join(f"- {e}" for e in errors))
                
            return (True, task_output)

        extract_task = self._create_task(
            "extract_receipt_data_task", 
            self.tasks_config["extract_receipt_data_task"], 
            agents["receipt_extraction_specialist"], 
            context=[classify_task, optimize_task],
            guardrails=[validation_guardrail],
            output_pydantic=ReceiptData  # 구조화된 출력 적용
        )

        # 3. Compliance Task
        compliance_task = self._create_task(
            "compliance_audit_task",
            self.tasks_config["compliance_audit_task"],
            agents["compliance_auditor"],
            context=[extract_task],
            output_pydantic=ComplianceResult  # 구조화된 출력 적용
        )

        # Execute full crew
        # Note: We can now run them all in one Crew since guardrails handle the validation retry logic internally!
        full_crew = Crew(
            agents=list(agents.values()), 
            tasks=[classify_task, optimize_task, extract_task, compliance_task], 
            verbose=True,
            memory=True,
            after_kickoff_callbacks=[save_result_callback]
        )
        
        result = full_crew.kickoff(inputs={"image_path": image_path})
        return str(result)
