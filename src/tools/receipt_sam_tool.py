"""
ReceiptSAMTool - SAM(Segment Anything Model)을 사용하여 영수증 영역을 추출하고 크롭하는 도구
"""
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from segment_anything import sam_model_registry, SamPredictor

class ReceiptSAMInput(BaseModel):
    """Input parameters for ReceiptSAMTool."""
    image_path: str = Field(..., description="처리할 영수증 이미지 파일의 절대 경로")
    output_path: str = Field(None, description="결과물을 저장할 경로")

class ReceiptSAMTool(BaseTool):
    """SAM을 사용하여 사진 속 영수증을 정확하게 찾아내고 안전하게 크롭하는 도구"""
    
    name: str = "Receipt SAM Tool"
    description: str = (
        "사진(photo) 기반 영수증 이미지에서 영수증 중심 영역을 안정적으로 찾아 "
        "OCR에 적합하도록 정밀하게 크롭(crop)하여 반환합니다."
    )
    args_schema: Type[BaseModel] = ReceiptSAMInput

    def _run(self, image_path: str, output_path: str = None) -> str:
        if not os.path.exists(image_path):
            return f"Error: 파일이 존재하지 않습니다: {image_path}"

        # 모델 체크포인트 경로
        base_dir = Path(__file__).parent.parent.parent
        checkpoint_path = base_dir / "models" / "sam_vit_b.pth"
        
        if not checkpoint_path.exists():
            return f"Error: SAM 모델 가중치 파일이 없습니다. {checkpoint_path} 위치에 파일이 필요합니다."

        try:
            # 1. 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                return f"Error: 이미지를 로드할 수 없습니다: {image_path}"
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2. SAM 모델 설정
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            sam.to(device=device)
            predictor = SamPredictor(sam)
            predictor.set_image(image_rgb)
            
            # 3. 영수증 영역 예측을 위한 멀티 포인트 프롬프트 도입
            # 접힌 선이 있거나 영역이 분절되어 보이는 경우를 대비해 십자형(5점) 프롬프트를 사용
            # 이를 통해 영수증의 상하좌우가 하나의 객체임을 모델에게 명확히 전달함
            h, w = image.shape[:2]
            input_points = np.array([
                [w // 2, h // 2],    # 중앙
                [w // 2, h // 4],    # 상단
                [w // 2, 3 * h // 4], # 하단
                [w // 4, h // 2],    # 좌측
                [3 * w // 4, h // 2]  # 우측
            ])
            input_labels = np.array([1, 1, 1, 1, 1]) # 모든 점을 객체 영역으로 지정
            
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            
            # 면적이 가장 큰 마스크를 선택하여 '부분 영역'만 잡히는 현상 방지
            # 단, SAM의 신뢰도(score)가 어느 정도 보장된 경우에만 면적 우선순위 적용
            best_mask_idx = np.argmax(scores)
            max_area = 0
            for i, (m, s) in enumerate(zip(masks, scores)):
                area = np.sum(m)
                # 점수가 0.85 이상이면서 가장 넓은 마스크를 찾음
                if s > 0.85 and area > max_area:
                    max_area = area
                    best_mask_idx = i
            
            mask = masks[best_mask_idx]
            
            # 4. 바운딩 박스 계산 및 크롭
            y, x = np.where(mask)
            if len(x) == 0 or len(y) == 0:
                return "Error: 영수증 영역을 찾지 못했습니다."
                
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            
            # 약간의 여백 추가 (OCR 안전성)
            pad = 20
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w, x_max + pad)
            y_max = min(h, y_max + pad)
            
            cropped_image = image[y_min:y_max, x_min:x_max]
            
            # 5. 결과 저장
            if output_path is None:
                base, ext = os.path.splitext(image_path)
                output_path = f"{base}_sam_cropped{ext}"
            
            cv2.imwrite(output_path, cropped_image)
            
            return f"SAM 크롭 완료: {output_path}"

        except Exception as e:
            return f"Error in ReceiptSAMTool: {str(e)}"
