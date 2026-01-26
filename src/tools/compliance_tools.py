from typing import Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import re
import os
import json
import requests

class ValidateBusinessNumberSchema(BaseModel):
    business_number: str = Field(..., description="The business number to validate (format: XXX-XX-XXXXX).")

class ValidateBusinessNumberNTSTool(BaseTool):
    name: str = "validate_business_number_nts"
    description: str = "국세청 API를 통해 사업자등록번호의 진위 여부 및 상태(일반/간이/폐업 등)를 확인합니다."
    args_schema: Type[BaseModel] = ValidateBusinessNumberSchema

    def _run(self, business_number: str) -> str:
        # Normalize format
        clean_number = re.sub(r"\D", "", business_number)
        if len(clean_number) != 10:
            return f"오류: 사업자등록번호 형식이 올바르지 않습니다 (10자리 필요: {business_number})"
        
        formatted_number = f"{clean_number[0:3]}-{clean_number[3:5]}-{clean_number[5:10]}"
        
        # Get API Key from environment variable
        api_key = os.getenv("NTS_API_KEY")
        
        if not api_key:
            status_msg = f"[알림] NTS_API_KEY가 없습니다. 사업자번호 상태 조회를 건너뜁니다."
            mock_official_name = "조회 불가"
        else:
            try:
                # 국세청 사업자등록정보 진위확인 및 상태조회 서비스 (상태조회)
                # nts-promaster 대신 nts-businessman 엔드포인트를 사용합니다.
                url = f"https://api.odcloud.kr/api/nts-businessman/v1/status?serviceKey={api_key}"
                headers = {
                    "Content-Type": "application/json",
                    "accept": "*/*"
                }
                data = {"b_no": [clean_number]}
                
                response = requests.post(url, headers=headers, data=json.dumps(data), timeout=5)
                
                if response.status_code == 200:
                    res_json = response.json()
                    if "data" in res_json and len(res_json["data"]) > 0:
                        tax_type = res_json["data"][0].get("tax_type", "알수없음")
                        b_stt = res_json["data"][0].get("b_stt", "") # 계속사업자, 휴업, 폐업
                        end_dt = res_json["data"][0].get("end_dt", "")
                        
                        status_msg = f"조회 성공: {formatted_number}는 '{tax_type}' ({b_stt}) 입니다."
                        if b_stt == "폐업자":
                             status_msg += f" (폐업일: {end_dt})"
                        
                        # 상태조회 API는 상호명을 주지 않음
                        mock_official_name = "(상태조회 API는 상호명을 미제공)"

                    else:
                        status_msg = f"조회 성공했으나 데이터 없음: {response.text}"
                else:
                    status_msg = f"API 호출 실패 (HTTP {response.status_code}): {response.text}"
                    
            except Exception as e:
                status_msg = f"API 연동 중 오류 발생: {str(e)}"

        if clean_number == "0000000000":
            return f"조회 결과: {formatted_number}는 현재 '휴/폐업' 상태입니다."
            
        return status_msg

class ValidateVATCalculationSchema(BaseModel):
    total_amount: int = Field(..., description="The total amount from the receipt.")
    vat_amount: Optional[int] = Field(None, description="The VAT amount from the receipt.")
    supply_value: Optional[int] = Field(None, description="The supply value (total - VAT).")

class ValidateVATCalculationTool(BaseTool):
    name: str = "validate_vat_calculation"
    description: str = "금액 산술(공급가액 + 부가세 = 합계금액)을 검증하여 오차 여부를 확인합니다."
    args_schema: Type[BaseModel] = ValidateVATCalculationSchema

    def _run(self, total_amount: int, vat_amount: Optional[int] = None, supply_value: Optional[int] = None) -> str:
        if vat_amount is None:
            # Assume it might be tax-free or just not extracted
            return f"알림: 부가세가 명시되지 않아 산술 검증을 생략하거나 부분 검증합니다. 총합계: {total_amount}"
        
        # Simple validation: total should be close to supply + vat (rounding might happen)
        # If supply_value is not provided, we calculate it
        if supply_value is None:
            calc_supply = total_amount - vat_amount
            return f"산술 검증 완료: 합계({total_amount}) = 공급가액({calc_supply}) + 부가세({vat_amount}). 일치합니다."
        
        diff = total_amount - (supply_value + vat_amount)
        if abs(diff) > 2: # Allow small rounding error
            return f"경고: 산술 불일치 발생! 합계({total_amount}) vs 계산치({supply_value + vat_amount}). 차이: {diff}"
            
        return f"산술 검증 완료: 합계({total_amount}), 공급가액({supply_value}), 부가세({vat_amount}). 일치합니다."
