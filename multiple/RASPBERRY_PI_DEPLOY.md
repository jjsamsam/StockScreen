# 🍓 라즈베리 파이 배포 가이드

이 문서는 Stock Screener 웹앱을 라즈베리 파이에 배포하는 방법을 설명합니다.

## 📋 사전 요구사항

- 라즈베리 파이 (Raspberry Pi 3B+ 이상 권장)
- Docker가 설치되어 있어야 함
- SSH 접속 가능
- GitHub에서 코드 동기화 가능

---

## 🚀 배포 단계

### 1단계: SSH로 라즈베리 파이 접속

```bash
ssh jjsam3@192.168.50.10
```
> IP 주소는 본인의 라즈베리 파이 IP로 변경하세요.

---

### 2단계: 프로젝트 디렉토리로 이동

```bash
cd ~/StockScreen/multiple/web_app
```

---

### 3단계: 최신 코드 가져오기

```bash
git pull origin main
```

충돌이 발생하면 다음 명령어로 강제 동기화:
```bash
git fetch origin
git reset --hard origin/main
```

---

### 4단계: Docker 컨테이너 재빌드 및 시작

```bash
docker compose down
docker compose up -d --build
```

> ⏱️ 빌드에 약 2-5분 정도 소요됩니다.

---

### 5단계: 배포 확인

```bash
docker ps
```

다음과 같이 2개의 컨테이너가 실행 중이어야 합니다:
- `stock-screener-frontend` (포트 80)
- `stock-screener-backend` (포트 8000)

---

## 🌐 웹앱 접속

브라우저에서 다음 주소로 접속:
- **http://192.168.50.10** (라즈베리 파이 IP)

---

## 🔧 유용한 명령어 모음

### 컨테이너 상태 확인
```bash
docker ps
```

### 컨테이너 로그 확인
```bash
# 백엔드 로그
docker logs stock-screener-backend

# 프론트엔드 로그
docker logs stock-screener-frontend

# 실시간 로그 (따라가기)
docker logs -f stock-screener-backend
```

### 컨테이너 재시작 (빌드 없이)
```bash
docker compose restart
```

### 컨테이너 중지
```bash
docker compose down
```

### 컨테이너 시작 (이미 빌드된 경우)
```bash
docker compose up -d
```

### 특정 컨테이너만 재빌드
```bash
# 프론트엔드만 재빌드
docker compose up -d --build frontend

# 백엔드만 재빌드
docker compose up -d --build backend
```

### Docker 이미지 정리 (디스크 공간 확보)
```bash
docker system prune -a
```
> ⚠️ 이 명령어는 모든 미사용 이미지를 삭제합니다. 다음 빌드 시 시간이 더 걸릴 수 있습니다.

---

## ❗ 문제 해결

### 문제: `docker compose` 명령어를 찾을 수 없음
```bash
# docker-compose (하이픈) 대신 docker compose (공백) 사용
docker compose --version
```

### 문제: 포트가 이미 사용 중
```bash
# 사용 중인 프로세스 확인
sudo lsof -i :80
sudo lsof -i :8000

# 해당 프로세스 종료
sudo kill -9 <PID>
```

### 문제: 컨테이너가 시작되지 않음
```bash
# 로그 확인
docker logs stock-screener-backend
docker logs stock-screener-frontend
```

### 문제: 디스크 공간 부족
```bash
# 디스크 사용량 확인
df -h

# Docker 정리
docker system prune -a
```

### 문제: 변경사항이 반영되지 않음
```bash
# 강제 재빌드
docker compose down
docker compose build --no-cache
docker compose up -d
```

---

## 📱 모바일 접속

같은 네트워크에 연결된 모바일 기기에서:
1. WiFi가 라즈베리 파이와 같은 네트워크인지 확인
2. 브라우저에서 `http://192.168.50.10` 접속

---

## 🔄 빠른 배포 요약

```bash
# SSH 접속
ssh jjsam3@192.168.50.10

# 디렉토리 이동
cd ~/StockScreen/multiple/web_app

# 최신 코드 가져오기
git pull origin main

# Docker 재빌드
docker compose down
docker compose up -d --build

# 확인
docker ps
```

---

## 📅 최종 업데이트
- 2026-01-30: 초기 문서 작성
