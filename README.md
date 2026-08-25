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
- Message별 `content: string | ContentPart[]` wire format 선택
- Text / Image / Audio / Video Content Part 편집 및 드래그 정렬
- 로컬 미디어 파일을 브라우저에서 Base64 Data URL로 변환
- 드래그로 항목 순서 변경
- Hyperparameter 설정 (temperature, top_p, max_tokens 등)
- 스트리밍 응답 실시간 출력 / 중간 중단
- 요청 JSON 미리보기 및 복사
- 대화 내보내기 / 불러오기 (JSON)
- 다크모드 토글

## 멀티모달 Message Content

각 Message의 **Content Format**에서 `String` 또는 `Content Parts`를 선택할 수 있다. 두 형식은 서로 normalize하지 않으므로 아래처럼 wire format 자체가 다른 요청을 각각 테스트할 수 있다.

```json
{"role":"user","content":"Hello"}
```

```json
{"role":"user","content":[{"type":"text","text":"Hello"}]}
```

Content Parts는 다음 canonical 형태만 지원한다.

| UI | Wire type | 값 |
|---|---|---|
| Text | `text` | `text` |
| Image | `image_url` | `image_url.url` |
| Audio | `audio_url` | `audio_url.url` |
| Video | `video_url` | `video_url.url` |

미디어는 로컬 파일 선택 후 브라우저의 `FileReader.readAsDataURL()`로 변환된다. 생성된 `data:image/...;base64,...` 등의 전체 Data URL이 request JSON에 직접 들어가며 별도 파일 업로드나 서버 임시 저장은 발생하지 않는다. HTTP/HTTPS URL, `file://`, PDF/DOCX/ZIP 같은 범용 파일 첨부는 지원하지 않는다.

role별 새 part 추가 범위는 다음과 같다.

- `system`, `tool`: Text
- `user`, `assistant`: Text, Image, Audio, Video

role을 바꿨을 때 이미 존재하는 비권장 part는 삭제하지 않는다. warning을 표시하지만 전송은 허용하므로 서버의 실제 호환성 응답을 확인할 수 있다.

Request Body의 **Base64 축약 표시**는 화면 표시만 줄인다. SEND, 전체 Request JSON 복사, Export에는 항상 전체 Base64가 사용된다.

### Base64 크기와 메모리

Base64 payload는 원본 `N` bytes에 대해 대략 `4 * ceil(N / 3)` bytes이며 Data URL prefix와 JSON 구조가 추가된다. 브라우저에서는 FileReader 결과, DOM 값, JSON 직렬화/fetch body가 동시에 존재할 수 있다.

`server.py`는 body를 해석하거나 저장하지 않는 opaque proxy지만, 현재 구현은 `Content-Length` 크기의 request body 전체를 메모리에 읽은 뒤 upstream으로 전달한다. 서버가 thread 기반이므로 동시 대용량 요청의 메모리 사용량은 요청 수에 따라 증가한다. 특히 큰 영상은 작은 fixture부터 단계적으로 크기를 늘리면서 브라우저, proxy, vLLM 서버의 메모리 여유를 함께 확인하는 것을 권장한다.

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
npx playwright install chromium
npx playwright test
python -m unittest discover -s tests -p "test_*.py"
```

| 파일 | 내용 |
|------|------|
| `tests/tools.spec.js` | Tool CRUD + 유효성 검사 |
| `tests/messages.spec.js` | Message CRUD + 중첩 tool call |
| `tests/import-export.spec.js` | JSON 내보내기 / 불러오기 |
| `tests/drag-sort.spec.js` | 드래그 정렬 |
| `tests/multimodal-request.spec.js` | Base64 파일 입력, SEND/COPY/EXPORT/Preview |
| `tests/test_proxy.py` | Python stdlib proxy의 opaque body 전달 |

## 브랜치

| 브랜치 | 설명 |
|--------|------|
| `main` / `dev` | 기본 개발 라인 |
| `lite` | ES5 + XHR 호환 버전 (`v0.1.0-lite`) |
