"""
FastAPI 백엔드 메인 애플리케이션
"""

import sys
import os

# ✅ 먼저 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # web_app
grandparent_dir = os.path.dirname(parent_dir)  # multiple

# Python 경로에 추가
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

# ✅ 작업 디렉토리를 프로젝트 루트로 변경 (CSV 파일 찾기 위해)
os.chdir(grandparent_dir)
print(f"✅ Working directory: {os.getcwd()}")
print(f"✅ stock_data exists: {os.path.exists('stock_data')}")

# 이제 import
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

from api import prediction, screening, data, health

app = FastAPI(
    title="Stock Screener API",
    description="AI 기반 주식 스크리닝 및 예측 API",
    version="1.0.0"
)

# CORS 설정 (모바일 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(prediction.router, prefix="/api", tags=["prediction"])
app.include_router(screening.router, prefix="/api", tags=["screening"])
app.include_router(data.router, prefix="/api", tags=["data"])

# 정적 파일 서빙 (프론트엔드)
# app.mount("/", StaticFiles(directory="../frontend/build", html=True), name="static")

@app.get("/")
async def root():
    return {
        "message": "Stock Screener API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
