"""
VisionClassificationTool - GPT-4o Vision을 사용하여 이미지를 분석하고 분류하는 도구
"""
import base64
import os
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from openai import OpenAI


def encode_image(image_path):
    """이미지 파일을 base64 문자열로 인코딩합니다."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class VisionClassificationToolInput(BaseModel):
    """Input parameters for VisionClassificationTool."""
    image_path: str = Field(..., description="분석할 이미지 파일의 절대 경로 (예: D:/ocr_convert/receipt.jpg)")
    query: str = Field(..., description="이미지에 대해 질문할 내용 (예: '이 영수증이 스캔된 것인지, 휴대폰으로 찍은 것인지 판별해줘.')")


class VisionClassificationTool(BaseTool):
    """GPT-4o Vision을 사용하여 이미지를 분석하고 분류하는 도구"""
    
    name: str = "Vision Classification Tool"
    description: str = (
        "GPT-4o Vision 모델을 사용하여 주어진 이미지를 분석하고 질문에 답변합니다. "
        "이미지 경로와 질문(query)을 입력받아 텍스트로 결과를 반환합니다."
    )
    args_schema: Type[BaseModel] = VisionClassificationToolInput

    def _run(self, image_path: str, query: str) -> str:
        """
        도구 실행 로직
        
        Args:
            image_path: 분석할 이미지 경로
            query: 이미지에 대한 질문
            
        Returns:
            모델의 분석 결과 텍스트
        """
        if not os.path.exists(image_path):
            return f"Error: 파일이 존재하지 않습니다: {image_path}"
            
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다."

        try:
            client = OpenAI(api_key=api_key)
            
            base64_image = encode_image(image_path)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can analyze images."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error executing VisionClassificationTool: {str(e)}"
