# Stock Screener Web App

AI 기반 주식 스크리닝 및 예측 시스템 - 모바일 웹 버전

## 🌟 기능

- **📊 스크리닝**: 기술적 지표 기반 종목 스크리닝
- **🤖 AI 예측**: 머신러닝 기반 주가 예측
- **📱 모바일 최적화**: 아이폰/안드로이드 반응형 UI
- **🌍 다국가 지원**: 한국, 미국, 스웨덴 시장

## 🚀 빠른 시작

### 로컬 개발

#### 백엔드
```bash
cd backend
pip install -r requirements.txt
python main.py
```

#### 프론트엔드
```bash
cd frontend
npm install
npm run dev
```

### Docker로 실행

```bash
docker-compose up -d
```

- 프론트엔드: http://localhost
- 백엔드 API: http://localhost:8000
- API 문서: http://localhost:8000/docs

## 📦 라즈베리파이 배포

자세한 내용은 [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md)를 참조하세요.

## 🏗️ 프로젝트 구조

```
web_app/
├── backend/
│   ├── api/              # API 엔드포인트
│   ├── core/             # 핵심 비즈니스 로직
│   ├── main.py           # FastAPI 앱
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/   # React 컴포넌트
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
└── docker-compose.yml
```

## 🔧 기술 스택

### 백엔드
- FastAPI
- Python 3.12
- scikit-learn, XGBoost, LightGBM
- pandas, numpy

### 프론트엔드
- React 18
- TypeScript
- Vite
- Axios
- Recharts

## 📱 모바일 접속

1. 라즈베리파이 IP 주소 확인
2. 모바일 브라우저에서 `http://[IP주소]` 접속
3. 홈 화면에 추가하여 앱처럼 사용

## 🔐 보안

프로덕션 환경에서는:
- CORS 설정 수정 (특정 도메인만 허용)
- HTTPS 설정
- API 인증 추가

## 📄 라이선스

MIT License

## 🤝 기여

Pull Request 환영합니다!
