# 🚨 보안 경고: 데이터베이스 자격 증명 노출

## 현재 상황
GitHub에 PostgreSQL 데이터베이스 자격 증명이 **공개적으로 노출**되었습니다.

## 즉시 조치 사항

### 1️⃣ **Railway 데이터베이스 비밀번호 변경 (최우선)**

Railway 대시보드에서 즉시 데이터베이스 비밀번호를 변경하세요:

1. Railway 대시보드 로그인: https://railway.app
2. 프로젝트 선택
3. PostgreSQL 서비스 클릭
4. "Variables" 탭 클릭
5. 새 비밀번호 생성
6. 새 연결 URL 복사

### 2️⃣ **환경 변수 설정**

프로젝트 루트에 `.env` 파일을 생성하세요 (이 파일은 절대 커밋하지 마세요):

```bash
# .env 파일 생성
cp .env.example .env
```

`.env` 파일을 열어 실제 값을 입력하세요:

```env
# Local PostgreSQL Database
LOCAL_DATABASE_URL=postgresql://two_tower_user:twotower2024@localhost:5432/two_tower_db

# Railway PostgreSQL Database (새 비밀번호 사용!)
RAILWAY_DATABASE_URL=postgresql://postgres:NEW_PASSWORD@crossover.proxy.rlwy.net:47399/railway
```

### 3️⃣ **Git 히스토리에서 민감한 정보 제거**

노출된 자격 증명은 이미 Git 히스토리에 저장되어 있습니다. 다음 중 하나를 선택하세요:

**옵션 A: BFG Repo-Cleaner 사용 (권장)**

```bash
# BFG 다운로드
# https://rtyley.github.io/bfg-repo-cleaner/

# 비밀번호 문자열을 포함한 모든 파일에서 제거
java -jar bfg.jar --replace-text passwords.txt

# 변경사항 적용
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 강제 푸시 (주의!)
git push --force
```

**옵션 B: git filter-branch 사용**

```bash
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch scripts/migrate_*.py" \
  --prune-empty --tag-name-filter cat -- --all

git push --force
```

**⚠️ 주의**: 이 작업은 되돌릴 수 없으며, 모든 협업자가 저장소를 다시 클론해야 합니다.

### 4️⃣ **Python 의존성 설치**

환경 변수를 사용하기 위해 `python-dotenv` 패키지가 필요합니다:

```bash
cd backend_web
pip install python-dotenv
```

### 5️⃣ **코드 수정 완료 확인**

다음 파일들이 수정되었는지 확인하세요:
- ✅ `scripts/migrate_reviews_filtered.py`
- ✅ `scripts/migrate_reviews_only.py`
- ✅ `scripts/migrate_simple.py`
- ✅ `scripts/migrate_all_with_ids.py`
- ✅ `scripts/migrate_local_to_railway.py`

모든 파일이 이제 환경 변수를 사용합니다.

### 6️⃣ **데이터베이스 접근 로그 확인**

Railway 대시보드에서 데이터베이스 접근 로그를 확인하여 **의심스러운 활동**이 있는지 점검하세요.

### 7️⃣ **데이터 무결성 검증**

```bash
# Railway DB 접속하여 데이터 확인
# 의심스러운 변경사항이 있는지 확인
```

## 향후 예방 조치

### ✅ 완료된 조치
- [x] 모든 스크립트가 환경 변수 사용하도록 수정
- [x] `.env.example` 파일 생성
- [x] `.gitignore`에 `.env` 파일 포함 확인

### 🔒 추가 보안 권장사항

1. **IP 화이트리스트 설정**: Railway에서 허용된 IP만 접근하도록 제한
2. **읽기 전용 계정 생성**: 필요한 경우 읽기 전용 DB 계정 사용
3. **정기적인 비밀번호 변경**: 3개월마다 비밀번호 변경
4. **2FA 활성화**: Railway 계정에 2단계 인증 설정
5. **모니터링 설정**: 비정상적인 접근 시도 알림 설정

## 참고 자료

- [GitHub: 민감한 데이터 제거하기](https://docs.github.com/ko/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)
- [Railway 보안 가이드](https://docs.railway.app/reference/security)

## 문의사항

보안 관련 질문이 있으면 즉시 팀에 문의하세요.

---
**생성일**: 2024-11-14
**상태**: 🔴 긴급 조치 필요















