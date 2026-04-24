# vLLM Tester

vLLM 서버에 대한 Chat Completions 요청을 브라우저에서 직접 테스트하는 단일 페이지 도구.  
Python 표준 라이브러리만 사용

## 실행

```bash
python vllm_tester.py
# 기본값: --port 8080, --target http://localhost:8000

python vllm_tester.py --port 9000 --target http://10.0.0.5:8000

python vllm_tester.py --port 8080 --target http://localhost:8000 --api-key sk-xxx
```

브라우저에서 `http://localhost:8080` 접속.

## 구조

| 파일 | 역할 |
|------|------|
| `vllm_tester.py` | HTTP 서버 + vLLM 프록시 (CORS 우회) |
| `vllm-tester.html` | 테스터 UI (단일 파일, 외부 의존 없음) |
