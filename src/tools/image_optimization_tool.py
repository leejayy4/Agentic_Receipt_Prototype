"""
ImageQualityOptimizationTool - OpenCV를 사용하여 이미지의 품질을 개선하고 OCR 가독성을 높이는 도구
"""
import cv2
import numpy as np
import os
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class ImageOptimizationInput(BaseModel):
    """Input parameters for ImageQualityOptimizationTool."""
    image_path: str = Field(..., description="처리할 이미지 파일의 절대 경로")
    output_path: str = Field(None, description="결과물을 저장할 경로 (기본값은 원본 파일명에 _optimized 추가)")

class ImageQualityOptimizationTool(BaseTool):
    """OpenCV를 사용하여 이미지의 기울기 보정, 노이즈 제거, 선명도 개선을 수행하는 도구"""
    
    name: str = "Image Quality Optimization Tool"
    description: str = (
        "이미지의 기울기를 계산하여 회전 보정을 수행하고, "
        "글자를 또렷하게 하며 배경 잡음을 제거하여 OCR 가독성을 향상시킵니다."
    )
    args_schema: Type[BaseModel] = ImageOptimizationInput

    def _run(self, image_path: str, output_path: str = None) -> str:
        if not os.path.exists(image_path):
            return f"Error: 파일이 존재하지 않습니다: {image_path}"

        try:
            # 1. 이미지 로드
            img = cv2.imread(image_path)
            if img is None:
                return f"Error: 이미지를 불러올 수 없습니다: {image_path}"

            # 2. 기울기 보정 (Deskew)
            # 회전 보정 전에 그레이스케일로 작업
            deskewed = self._deskew(img)

            # 3. 가독성 개선 (Sharpening & Thresholding)
            # 선명하게 만들기 (Sharpening kernel)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(deskewed, -1, kernel)
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
            
            # 노이즈 제거 (OCR 방해 요소 제거)
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # 적응형 임계값 처리 (배경 노이즈는 날리고 글자만 강조)
            # 이진화(Binary) 이미지로 만들어 OCR 엔진이 글자만 인식하기 좋게 함
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 21, 10
            )

            # 4. 결과 저장
            if output_path is None:
                base, ext = os.path.splitext(image_path)
                output_path = f"{base}_optimized{ext}"
            
            cv2.imwrite(output_path, thresh)
            
            return f"이미지 최적화 완료: {output_path}"

        except Exception as e:
            return f"Error processing image: {str(e)}"

    def _deskew(self, image):
        """이미지의 기울기를 계산하고 회전 보정합니다."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        
        # 텍스트 영역의 각도를 찾기 위한 임계값 처리
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # 텍스트 영역의 좌표 추출
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        # cv2.minAreaRect는 -90 ~ 0 사이의 각도를 반환함
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
