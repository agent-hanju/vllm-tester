# vLLM Tester

vLLM 서버에 대한 Chat Completions 요청을 브라우저에서 직접 테스트하는 단일 페이지 도구.  
Python 표준 라이브러리만 사용

## 실행

```bash
python server.py
# 기본값: --port 8080, --target http://localhost:8000

python server.py --port 9000 --target http://10.0.0.5:8000

python server.py --port 8080 --target http://localhost:8000 --api-key sk-xxx
```

브라우저에서 `http://localhost:8080` 접속.

## 기능

- Tool 정의 및 Message 목록 편집
- 드래그로 항목 순서 변경
- Hyperparameter 설정 (temperature, top_p, max_tokens 등)
- 스트리밍 응답 실시간 출력 / 중간 중단
- 요청 JSON 미리보기 및 복사
- 대화 내보내기 / 불러오기 (JSON)
- 다크모드 토글

## 구조

| 파일 | 역할 |
|------|------|
| `server.py` | HTTP 서버 + vLLM 프록시 (CORS 우회) |
| `vllm-tester.html` | 테스터 UI |
| `vllm-tester.css` | 테마 스타일 (CSS 변수, 다크모드) |
| `vllm-tester-base.css` | 기본 스타일 (CSS 변수 없음, lite 버전용) |

## 테스트

```bash
npm install
npx playwright test
```

| 파일 | 내용 |
|------|------|
| `tests/tools.spec.js` | Tool CRUD + 유효성 검사 |
| `tests/messages.spec.js` | Message CRUD + 중첩 tool call |
| `tests/import-export.spec.js` | JSON 내보내기 / 불러오기 |
| `tests/drag-sort.spec.js` | 드래그 정렬 |

## 브랜치

| 브랜치 | 설명 |
|--------|------|
| `main` / `dev` | 기본 개발 라인 |
| `lite` | ES5 + XHR 호환 버전 (`v0.1.0-lite`) |
