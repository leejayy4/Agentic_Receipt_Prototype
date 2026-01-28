## 설치 및 시작하기 (Installation & Getting Started)

### 1. 사전 요구 사항 (Prerequisites)
*   OpenAI API Key
*   국세청 사업자조회 API Key
(두 개의 Key 모두 현재 .env 파일에 있습니다)

### 2. 프로젝트 설치
```bash
# 레포지토리 클론 (또는 다운로드)
git clone <repository-url>
cd Agentic_Receipt_Prototype

# 가상환경 생성 및 활성화
python -m venv venv_312
# Windows
.\venv_312\Scripts\activate
# Mac/Linux
source venv_312/bin/activate

# 의존성 패키지 설치
pip install -r requirements.txt
```
> **참고**: `torch`와 `segment-anything` 설치 시 시스템 환경에 따라 추가 설정이 필요할 수 있습니다.

### 3. 환경 변수 설정
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 API 키를 입력하세요.
```ini
OPENAI_API_KEY=sk-your-openai-api-key-here
```

## 💻 사용 방법 (Usage)

터미널에서 `src/main.py`를 실행하며 처리할 이미지의 경로를 인자로 전달합니다.

```bash
python src/main.py "D:\Path\To\Your\receipt_image.jpg"
```

### 실행 예시
```bash
(venv_312) D:\Agentic_Receipt_Prototype> python src/main.py "D:\Agentic_Receipt_Prototype\photo_sample\my_receipt.jpg"
```
