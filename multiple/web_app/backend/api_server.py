"""
FastAPI 백엔드 메인 애플리케이션
"""

import sys
import os

# ✅ 먼저 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # web_app
grandparent_dir = os.path.dirname(parent_dir)  # multiple

# Docker 환경 감지 (PYTHONPATH에 /app/parent가 있는지 등으로 확인 가능하지만, 간단히 /app 확인)
IN_DOCKER = os.path.exists('/app/parent')

if IN_DOCKER:
    # Docker 환경에서는 PYTHONPATH가 이미 설정되어 있으므로 추가 조작 최소화
    # /app/parent 가 multiple 디렉토리 역할을 함
    print(f"✅ Running in Docker environment")
else:
    # 로컬 개발 환경
    if grandparent_dir not in sys.path:
        sys.path.insert(0, grandparent_dir)
    
    # 작업 디렉토리 변경 (로컬에서만)
    os.chdir(grandparent_dir)

print(f"✅ Working directory: {os.getcwd()}")

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
    # 현재 디렉토리(backend)를 Python 경로에 추가하여 reload 시 api_server 모듈을 찾을 수 있게 합니다.
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 개발 편의를 위해 reload=True 설정 (코드가 수정되면 서버가 자동 재시작됩니다)
    # 단, reload=True일 때는 '파일명:앱객체' 문자열 형태로 전달해야 합니다.
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
