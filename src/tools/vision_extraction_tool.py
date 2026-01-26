"""
VisionExtractionTool - GPT-4o Vision을 사용하여 영수증 이미지에서 정밀하게 데이터를 추출하는 도구
"""
import os
import base64
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import openai


class VisionExtractionInput(BaseModel):
    """Input parameters for VisionExtractionTool."""
    image_path: str = Field(..., description="데이터를 추출할 이미지 파일의 절대 경로")
    instruction: str = Field(..., description="추출할 데이터 항목 및 상세 지침")


class VisionExtractionTool(BaseTool):
    """GPT-4o Vision을 사용하여 영수증 이미지에서 텍스트 및 구조화된 데이터를 추출하는 도구"""
    
    name: str = "Vision Extraction Tool"
    description: str = (
        "GPT-4o Vision 모델을 사용하여 영수증 이미지에서 상세한 텍스트 데이터를 추출합니다. "
        "이미지 경로와 구체적인 추출 지침(Instruction)을 입력받아 JSON 형태의 텍스트 결과를 반환합니다."
    )
    args_schema: Type[BaseModel] = VisionExtractionInput

    def _run(self, image_path: str, instruction: str) -> str:
        """
        도구 실행 로직
        """
        if not os.path.exists(image_path):
            return f"Error: 파일이 존재하지 않습니다: {image_path}"
            
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다."

        try:
            client = openai.OpenAI(api_key=api_key)
            
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "당신은 영수증 이미지 분석 전문가입니다. 오직 전달받은 지침(Instruction)에 따라 데이터를 추출하고 JSON으로만 응답하십시오."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"지침:\n{instruction}"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.0
            )
            
            result = response.choices[0].message.content.strip()
            
            # Markdown 코드 블록 제거
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
                
            return result
            
        except Exception as e:
            return f"Error executing VisionExtractionTool: {str(e)}"
