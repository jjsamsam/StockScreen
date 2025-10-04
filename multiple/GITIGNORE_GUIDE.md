# 📝 .gitignore 가이드

## 생성된 파일
[.gitignore](.gitignore)

---

## 🎯 제외되는 파일/폴더

### Python 관련
```
__pycache__/          # Python 컴파일 파일
*.pyc, *.pyo         # 바이트코드
*.so                 # 공유 라이브러리
```

### 캐시 디렉토리
```
.cache/              # 일반 캐시
.claude/             # Claude Code 캐시
cache/               # 애플리케이션 캐시
```

### IDE 설정
```
.vscode/             # VS Code 설정
.idea/               # PyCharm 설정
*.swp, *.swo        # Vim 임시 파일
```

### 로그 파일
```
*.log                # 로그 파일
logs/                # 로그 디렉토리
```

### 환경 변수
```
.env                 # 환경 변수 파일 (비밀 정보)
.env.local
```

### 빌드 결과물
```
build/               # 빌드 디렉토리
dist/                # 배포 디렉토리
*.egg-info/          # 패키지 정보
```

---

## 📊 데이터 파일 처리

### 현재 설정 (주석 처리됨)
```gitignore
# stock_data/
# *.csv
```

### 옵션 1: CSV 파일 포함 (현재)
```bash
# CSV 파일을 git에 커밋 (기본값)
git add stock_data/*.csv
git commit -m "Update stock data"
```

**장점:**
- 팀원들이 동일한 데이터로 테스트 가능
- 데이터 버전 관리

**단점:**
- 저장소 크기 증가
- 데이터 업데이트 시마다 커밋 필요

### 옵션 2: CSV 파일 제외
```gitignore
# .gitignore에서 주석 제거
stock_data/
*.csv
```

**장점:**
- 저장소 크기 작게 유지
- 각자 최신 데이터 다운로드

**단점:**
- 팀원마다 다른 데이터 사용 가능
- 초기 설정 필요

**권장:**
- 샘플 데이터만 포함: `stock_data/sample_*.csv`
- 전체 데이터는 제외

---

## 🚀 Git 명령어 가이드

### 1. .gitignore 적용
```bash
# .gitignore 파일 추가
git add .gitignore

# 이미 추적 중인 불필요한 파일 제거
git rm --cached -r __pycache__
git rm --cached -r .cache
git rm --cached -r cache

# 커밋
git commit -m "Add .gitignore for Python project"
```

### 2. 현재 상태 확인
```bash
# 추적되지 않는 파일 확인
git status

# 무시되는 파일 확인
git status --ignored
```

### 3. 특정 파일 강제 추가
```bash
# .gitignore에 있어도 강제로 추가
git add -f cache/important_file.pkl
```

---

## 📋 적용된 변경 사항

### 제거된 파일
```
✅ __pycache__/chart_window.cpython-312.pyc
✅ __pycache__/dialogs.cpython-312.pyc
✅ __pycache__/screener.cpython-312.pyc
✅ __pycache__/utils.cpython-312.pyc
```

### 향후 무시될 파일
```
🚫 cache/*.pkl (캐시 파일)
🚫 .claude/* (Claude Code 캐시)
🚫 *.pyc (Python 컴파일 파일)
🚫 *.log (로그 파일)
🚫 .env (환경 변수)
```

---

## 🔍 .gitignore 확인 방법

### 파일이 무시되는지 확인
```bash
# 특정 파일이 무시되는지 확인
git check-ignore -v __pycache__/test.pyc

# 출력 예시:
# .gitignore:2:__pycache__/    __pycache__/test.pyc
```

### 무시되는 모든 파일 보기
```bash
git status --ignored
```

---

## 🎓 .gitignore 패턴 문법

### 기본 패턴
```gitignore
# 특정 파일
secret.txt

# 특정 확장자
*.log

# 특정 디렉토리
cache/

# 디렉토리 내 특정 패턴
logs/*.log

# 모든 하위 디렉토리
**/*.pyc
```

### 예외 패턴
```gitignore
# 모든 .log 무시
*.log

# 하지만 important.log는 포함
!important.log
```

### 주석
```gitignore
# 이것은 주석입니다
*.tmp  # 줄 끝 주석도 가능
```

---

## 🛠️ 프로젝트별 커스터마이징

### AI 모델 파일 제외
```gitignore
# 큰 모델 파일 제외
*.pkl
*.h5
*.pt
*.pth
models/
```

### 데이터 파일 관리
```gitignore
# 큰 데이터 파일 제외
stock_data/*
!stock_data/sample_*.csv  # 샘플만 포함
```

### 문서 빌드 제외
```gitignore
# Sphinx, MkDocs 등
docs/_build/
site/
```

---

## 📝 Git 커밋 가이드

### .gitignore 추가 커밋
```bash
git add .gitignore
git commit -m "Add .gitignore

- Python 캐시 파일 제외 (__pycache__, *.pyc)
- 애플리케이션 캐시 제외 (.cache, .claude, cache/)
- IDE 설정 파일 제외 (.vscode, .idea)
- 로그 파일 제외 (*.log, logs/)
- 환경 변수 파일 제외 (.env)
- 빌드 결과물 제외 (build/, dist/)
"
```

### 캐시 파일 제거 커밋
```bash
git add -A
git commit -m "Remove cached files from git tracking

- __pycache__/ 제거
- .gitignore 적용으로 향후 자동 제외
"
```

---

## ⚠️ 주의사항

### 1. 이미 커밋된 파일
.gitignore는 **아직 추적되지 않은 파일**만 무시합니다.
이미 커밋된 파일은 명시적으로 제거해야 합니다:
```bash
git rm --cached filename
```

### 2. 민감한 정보
비밀번호, API 키 등이 포함된 파일은 반드시 .gitignore에 추가:
```gitignore
.env
secrets.json
config/credentials.py
```

**이미 커밋된 경우:**
```bash
# 히스토리에서 완전히 제거 (주의!)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch secrets.json" \
  --prune-empty --tag-name-filter cat -- --all
```

### 3. 팀 협업
.gitignore는 모든 팀원이 공유해야 합니다.
개인 설정은 `.git/info/exclude`에 추가:
```bash
# .git/info/exclude 편집
echo "my_local_file.txt" >> .git/info/exclude
```

---

## 🔗 참고 자료

### 공식 템플릿
- GitHub: https://github.com/github/gitignore
- Python: https://github.com/github/gitignore/blob/main/Python.gitignore

### 생성 도구
- gitignore.io: https://www.toptal.com/developers/gitignore
  - 사용: Python, PyQt, Windows, macOS, VS Code 선택

---

## 📊 현재 프로젝트 상태

### 추적 중인 파일 (예시)
```
✅ *.py                    # Python 소스 코드
✅ *.md                    # 문서
✅ stock_data/*.csv        # 데이터 파일 (선택사항)
✅ requirements.txt        # 의존성
```

### 무시되는 파일
```
🚫 __pycache__/           # Python 캐시
🚫 .cache/                # 일반 캐시
🚫 .claude/               # Claude Code
🚫 cache/                 # 앱 캐시
🚫 *.log                  # 로그
🚫 .env                   # 환경 변수
```

---

## ✅ 체크리스트

설정 완료 확인:
- [x] .gitignore 파일 생성
- [x] __pycache__ 제거
- [x] cache/ 무시 설정
- [x] .claude/ 무시 설정
- [ ] .env 파일 확인 (있으면 .gitignore에 추가)
- [ ] stock_data/ 처리 방침 결정
- [ ] git commit 및 push

---

**작성:** Claude Code Assistant
**일자:** 2025-10-04
**파일:** .gitignore, GITIGNORE_GUIDE.md
**상태:** ✅ 설정 완료
