# Stock Screener 라즈베리파이 3B+ 설치 가이드

## 📋 사전 요구사항

- 라즈베리파이 3B+ (1GB RAM)
- MicroSD 카드 (최소 16GB, 권장 32GB)
- 안정적인 인터넷 연결
- 전원 어댑터 (5V 2.5A)

## 🚀 설치 단계

### 1. Raspberry Pi OS 설치

1. **Raspberry Pi Imager 다운로드**
   - https://www.raspberrypi.com/software/

2. **OS 설치**
   - OS: Raspberry Pi OS Lite (64-bit) 권장
   - 설정에서 SSH 활성화
   - WiFi 설정 (선택사항)

3. **초기 설정**
   ```bash
   # SSH로 접속
   ssh pi@raspberrypi.local
   # 기본 비밀번호: raspberry
   
   # 시스템 업데이트
   sudo apt update && sudo apt upgrade -y
   ```

### 2. Docker 설치

```bash
# Docker 설치 스크립트 실행
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 현재 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER

# Docker Compose 설치
sudo apt install -y docker-compose

# 재부팅
sudo reboot
```

### 3. 애플리케이션 배포

```bash
# 프로젝트 클론 (GitHub에서)
cd ~
git clone https://github.com/YOUR_USERNAME/StockScreen.git
cd StockScreen/multiple/web_app

# Docker 이미지 빌드 및 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

### 4. 방화벽 및 포트 설정

```bash
# 포트 80 (프론트엔드) 및 8000 (백엔드) 열기
sudo ufw allow 80/tcp
sudo ufw allow 8000/tcp
sudo ufw enable
```

### 5. 자동 시작 설정

```bash
# systemd 서비스 파일 생성
sudo nano /etc/systemd/system/stock-screener.service
```

다음 내용 입력:
```ini
[Unit]
Description=Stock Screener Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/pi/StockScreen/multiple/web_app
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
User=pi

[Install]
WantedBy=multi-user.target
```

서비스 활성화:
```bash
sudo systemctl enable stock-screener.service
sudo systemctl start stock-screener.service
```

## 📱 모바일 접속

### 로컬 네트워크에서

1. 라즈베리파이 IP 주소 확인:
   ```bash
   hostname -I
   ```

2. 모바일 브라우저에서 접속:
   ```
   http://[라즈베리파이_IP주소]
   예: http://192.168.1.100
   ```

### 외부에서 접속 (선택사항)

#### 방법 1: 포트 포워딩
1. 공유기 관리 페이지 접속
2. 포트 포워딩 설정:
   - 외부 포트: 80 → 내부 IP: [라즈베리파이 IP], 포트: 80

#### 방법 2: Tailscale (권장)
```bash
# Tailscale 설치
curl -fsSL https://tailscale.com/install.sh | sh

# Tailscale 시작
sudo tailscale up

# 제공된 URL로 인증
```

이제 어디서든 Tailscale IP로 접속 가능!

## ⚙️ 성능 최적화 (라즈베리파이 3B+ 전용)

### 1. 메모리 최적화

```bash
# swap 파일 크기 증가 (1GB → 2GB)
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=2048로 변경
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### 2. CPU 오버클럭 (선택사항, 주의 필요)

```bash
sudo nano /boot/config.txt
```

다음 추가:
```ini
# 안전한 오버클럭
over_voltage=2
arm_freq=1350
```

### 3. 불필요한 서비스 비활성화

```bash
# Bluetooth 비활성화 (사용하지 않는 경우)
sudo systemctl disable bluetooth
sudo systemctl disable hciuart
```

## 🔧 문제 해결

### 메모리 부족 오류
```bash
# Docker 메모리 제한 설정
# docker-compose.yml에 추가:
services:
  backend:
    mem_limit: 512m
```

### 느린 예측 속도
- 예측 기간을 짧게 설정 (7일 이하)
- 동시 예측 수 제한

### 컨테이너 재시작
```bash
cd ~/StockScreen/multiple/web_app
docker-compose restart
```

### 로그 확인
```bash
# 전체 로그
docker-compose logs

# 백엔드만
docker-compose logs backend

# 실시간 로그
docker-compose logs -f
```

## 📊 모니터링

### 시스템 리소스 확인
```bash
# CPU, 메모리 사용률
htop

# Docker 컨테이너 상태
docker stats
```

### 온도 확인
```bash
vcgencmd measure_temp
```

60°C 이상이면 냉각 필요!

## 🔄 업데이트

```bash
cd ~/StockScreen/multiple/web_app
git pull
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 💡 팁

1. **정기 재부팅**: 주 1회 재부팅 권장
   ```bash
   sudo crontab -e
   # 매주 일요일 새벽 3시 재부팅
   0 3 * * 0 /sbin/shutdown -r now
   ```

2. **자동 업데이트**: 시스템 자동 업데이트 설정
   ```bash
   sudo apt install unattended-upgrades
   sudo dpkg-reconfigure -plow unattended-upgrades
   ```

3. **백업**: 중요 데이터 정기 백업
   ```bash
   # 캐시 및 모델 백업
   tar -czf backup-$(date +%Y%m%d).tar.gz cache/ models/ stock_data/
   ```

## 🎉 완료!

이제 아이폰이나 안드로이드에서 브라우저를 열고 라즈베리파이 IP로 접속하면 됩니다!

**홈 화면에 추가하기**:
- iOS: Safari에서 공유 버튼 → "홈 화면에 추가"
- Android: Chrome에서 메뉴 → "홈 화면에 추가"

## 📞 지원

문제가 발생하면 GitHub Issues에 문의하세요!
