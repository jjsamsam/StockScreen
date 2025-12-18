# Software Specification: Stock Screening & Prediction System (v1.0.2)

## 1. Project Overview
AI 기반 주식 스크리닝 및 예측 시스템은 한국(KOSPI/KOSDAQ), 미국, 스웨덴 시장의 방대한 데이터를 분석하여 사용자에게 기술적 매수/매도 신호와 AI 예측 데이터를 제공하는 웹 애플리케이션입니다.

- **Objective**: 기술적 지표 기반 스크리닝과 머신러닝 기반 가격 예측을 통합하여 투자 의사결정을 지원함.
- **Key Philosophy**: 모듈화된 백엔드 로직과 직관적이고 반응이 빠른 모던 웹 인터페이스의 결합.

---

## 2. System Architecture
시스템은 **Core Logic**, **API Layer (Backend)**, **User Interface (Frontend)**의 3계층 구조로 이루어져 있습니다.

### 2.1 Core Logic Layer (`multiple/`)
시스템의 심장부로, 데이터 수집, 가공 및 예측 엔진이 포함됩니다.
- **`enhanced_screener.py`**: ARIMA 및 앙상블 ML 모델(XGBoost, LightGBM 등)을 사용한 주가 예측 엔진.
- **`enhanced_screening_conditions.py`**: 강화된 기술적 분석 알고리즘 (Bollinger Bands, RSI, MACD 조합).
- **`cache_manager.py` / `csv_manager.py`**: 데이터 효율성을 위한 로컬 캐싱 및 종목 마스터 데이터 관리.
- **`utils.py`**: 기술적 지표 계산 로직 라이브러리.

### 2.2 Backend Layer (`web_app/backend/`)
Core Logic을 RESTful API로 래핑하여 프론트엔드와 통신합니다.
- **Framework**: FastAPI (Python 3.x 기반)
- **Services**: `ScreeningService`, `PredictionService`, `DataService`를 통한 도메인 로직 분리.
- **API Endpoints**: 
    - `/api/data/*`: 시장 및 종목 정보 조회.
    - `/api/screening`: 대량 종목 스크리닝 실행.
    - `/api/predict`: 특정 종목 AI 심층 분석 및 예측.
    - `/api/chart`: 시계열 차트 데이터(OHLCV + Indicators) 제공.

### 2.3 Frontend Layer (`web_app/frontend/`)
사용자 경험(UX)을 극대화한 SPA (Single Page Application).
- **Tech Stack**: React, TypeScript, Vite.
- **Visuals**: `lightweight-charts` (TradingView 기반)를 활용한 인터랙티브 기술 차트.
- **Localization**: `translations.ts`를 통한 실시간 KO/EN 다국어 지원.
- **Styling**: Glassmorphism 및 다크 모드 기반의 프리미엄 UI 디자인.

---

## 3. Detailed Features

### 3.1 Advanced Screening
- **Match Modes**: AND(모든 조건 일치) 또는 OR(하나라도 일치) 로직 지원.
- **Scope**: 상위 100개부터 전체 시장(최대 10,000개)까지 확장 가능한 스크리닝 범위.
- **Export**: 분석 결과를 CSV 파일로 추출 가능 (Localized).

### 3.2 AI Price Prediction
- **Multi-Model Ensemble**: ARIMA, XGBoost, Random Forest 등의 결과를 가중 평균하여 예측.
- **Confidence Scoring**: 예측의 신뢰도를 계산하여 "높은 신뢰도", "보수적 접근" 등의 어드바이스 제공.
- **Forecast Horizon**: 1일에서 최대 30일까지의 단기/중기 예측 기간 설정 가능.

### 3.3 Dynamic Charting
- **Technical Indicators**: MA (5, 20, 60, 120, 240), 볼린저 밴드, 거래량 비율, RSI.
- **Responsiveness**: 전체 화면 모드 지원 및 다양한 타임프레임(1개월~5년) 스케일링.

---

## 4. Development & Operation
- **Path Management**: 백엔드 core 디렉토리를 `sys.path`에 추가하여 로직의 재사용성과 독립성을 동시에 확보.
- **Logger**: `logger_config`를 통한 전 계층 통합 로그 시스템 구축.
- **Consistency**: 프론트엔드와 백엔드 간의 데이터 타입 일관성을 위해 TypeScript 인터페이스 정의.

---

## 5. Technical Specification (Dev Ops)
- **Frontend Port**: `5173` (Vite Default)
- **Backend Port**: `8000` (FastAPI / Uvicorn)
- **Data Source**: Yahoo Finance API (`yfinance`)를 통한 실시간성 데이터 수집.
- **Build System**: NPM (Frontend), Python Virtual Env (Backend).
