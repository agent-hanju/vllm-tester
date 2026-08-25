# PLAN — Multimodal `Message.content` 개선

## 1. 문서 목적

`vllm-tester`의 Message 편집기를 OpenAI-compatible Chat Completions의 wire format을 직접 시험할 수 있는 형태로 확장한다.

핵심 원칙은 `Message.content`를 내부에서 한 형식으로 정규화하지 않는 것이다. 사용자가 각 Message마다 아래 두 표현 중 하나를 직접 선택하고, 선택한 표현이 SEND·COPY·EXPORT까지 그대로 유지되어야 한다.

```text
Message.content = string | ContentPart[]
```

따라서 아래 두 요청은 의미가 비슷하더라도 서로 다른 테스트 케이스로 취급한다.

```json
{"role":"user","content":"Hello"}
```

```json
{"role":"user","content":[{"type":"text","text":"Hello"}]}
```

## 2. 목표

- Message별로 `content` 형식을 `String` 또는 `Content Parts`로 선택한다.
- `ContentPart[]` 편집 UI에서 `text`, `image`, `audio`, `video`만 지원한다.
- 이미지·음성·영상은 로컬 파일을 브라우저에서 Base64 Data URL로 변환해 request JSON에 직접 포함한다.
- `user`와 `assistant`는 네 종류의 part를 모두 새로 추가할 수 있고, `system`과 `tool`은 text part만 새로 추가할 수 있다.
- role 변경으로 비권장 part가 생겨도 자동 삭제하거나 SEND를 막지 않고 warning만 표시한다.
- String/Content Parts 변환의 손실 여부를 사용자에게 명확히 알리고, 손실 가능 변환은 명시적 확인 뒤에만 수행한다.
- Import/Export가 `content`의 원래 union branch와 배열 순서, 지원 part의 값을 보존한다.
- Request Preview에서만 Base64를 축약할 수 있고 실제 SEND·COPY·EXPORT에는 전체 값을 사용한다.
- 기존 DOM-as-state, `outer > .items > .item`, drag-sort 구조를 그대로 확장한다.
- Python 표준 라이브러리 프록시는 opaque body 전달 구조를 유지한다. 대용량 Base64의 메모리 특성은 문서화하고 검증한다.

## 3. Non-goals

- 임의의/generic content-part type이나 raw JSON part 편집기
- PDF, DOCX, ZIP 등 범용 파일 첨부
- HTTP/HTTPS URL 입력
- `file://` 또는 vLLM 서버 로컬 경로 입력
- 브라우저에서 선택한 파일의 서버 업로드, 임시 저장, 공유 스토리지, 정리 API
- 이미지 리사이즈·압축, 음성 변환, 영상 프레임 추출 또는 transcoding
- 모델별 modality 자동 탐지 또는 특정 모델의 실제 지원 여부 보장
- role capability 위반을 클라이언트에서 강제로 차단
- Base64 request의 임의 hard size limit 도입
- `lite` 브랜치의 ES5/XHR 동시 구현. 이 계획은 현재 `main` 구조를 대상으로 하며 `lite` 이식은 별도 작업으로 둔다.
- 지원 범위 밖의 content-part 및 임의 추가 필드에 대한 round-trip 보장

## 4. 현재 구조와 변경 경계

현재 구현은 다음 특성을 가진다.

- `vllm-tester.html`이 HTML template, DOM 조작, request 생성, response 처리, Import/Export를 모두 포함한다.
- `createMessageElement()`가 `.f-content textarea` 하나를 만들고 `buildRequest()`가 모든 role의 content를 string으로 직렬화한다.
- `parseImportedRequest()`는 배열 content를 `JSON.stringify()`한 문자열로 바꾼다. 이 동작은 wire 표현을 잃으므로 제거 대상이다.
- Message 안의 tool calls도 `outer > .items > .item` 구조를 사용하며 `enableDragSort()`로 중첩 정렬한다.
- `renderRequestView()`는 현재 전체 body를 그대로 pretty-print한다.
- `copyResponseToMessage()`와 non-streaming response renderer는 content가 string이라고 가정한다.
- `server.py::_proxy()`는 `Content-Length`만큼 request body를 메모리에 읽은 후 내용을 해석하지 않고 upstream에 전달한다.
- Playwright는 `python -m http.server 8080`으로 정적 페이지를 띄우며 Chromium E2E를 수행한다.

변경은 주로 프런트엔드의 Message content 편집/직렬화 경계에서 수행한다. `server.py`의 proxy 동작은 기능 구현을 위해 변경하지 않는다.

## 5. Wire 데이터 모델

### 5.1 지원 union

```text
MessageContent
  = string
  | ContentPart[]

ContentPart
  = TextPart
  | ImagePart
  | AudioPart
  | VideoPart
```

UI의 논리 타입과 wire object의 대응은 다음과 같이 고정한다.

| UI 타입 | Wire `type` | 값 경로 | 허용 값 |
|---|---|---|---|
| `text` | `text` | `text` | string |
| `image` | `image_url` | `image_url.url` | `data:image/*;base64,...` |
| `audio` | `audio_url` | `audio_url.url` | `data:audio/*;base64,...` |
| `video` | `video_url` | `video_url.url` | `data:video/*;base64,...` |

예시:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "이 자료를 설명해줘"},
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,iVBORw0KGgo..."
      }
    }
  ]
}
```

### 5.2 보존 계약

Import에 성공한 지원 데이터는 다음을 보존한다.

- `content`가 string이었는지 array였는지
- array part의 순서와 개수
- text 값(빈 문자열 포함)
- media Data URL 전체 문자열
- Message role 및 기존 `reasoning_content`, `tool_calls`, `tool_call_id`

여기서 “원본 표현 보존”은 지원 스키마 안에서의 구조적 round-trip을 뜻한다. JSON 공백, object key 순서, 지원하지 않는 part type이나 임의 추가 필드까지 byte-for-byte 보존하는 generic JSON 저장소를 만들지는 않는다.

## 6. DOM-as-state 설계

별도 전역 application state를 추가하지 않는다. 각 Message DOM이 현재 content 형식과 값을 소유한다.

`tpl-message`의 `.f-content`를 다음 구조로 확장한다.

```html
<div class="f-content">
  <label>Content Format</label>
  <select class="content-format">
    <option value="string">String</option>
    <option value="parts">Content Parts</option>
  </select>

  <div class="content-string-editor">
    <label>Content</label>
    <textarea class="content-string"></textarea>
  </div>

  <div class="content-parts-editor" hidden>
    <div class="content-parts-container">
      <div class="clear-row"><button class="clear-content-parts">× 전체 삭제</button></div>
      <p class="empty">— Content Part 없음 —</p>
      <div class="items"></div>
    </div>
    <div class="add-row content-part-actions">
      <button data-part="text">+ Text</button>
      <button data-part="image">+ Image</button>
      <button data-part="audio">+ Audio</button>
      <button data-part="video">+ Video</button>
    </div>
    <div class="content-role-warning validation v-warn" hidden></div>
  </div>
</div>
```

새 `tpl-content-part`는 모든 part에 공통인 `.item` 구조와 up/down/delete 버튼을 가진다. 정적으로 하나씩 존재하는 text/media 편집 영역은 template 안에 모두 선언하고 `hidden`으로 제어한다.

```text
div.content-part.item
  ├─ item-head + type badge + up/down/delete
  ├─ text editor                         [text만 표시]
  └─ media editor                        [image/audio/video만 표시]
       ├─ input[type=file]
       ├─ 파일명/MIME/원본 크기 표시
       ├─ 파일 선택/교체 버튼
       ├─ hidden data-url value
       └─ validation/status
```

Base64 Data URL의 실질 상태는 file input의 경로가 아니라 DOM 안의 전용 hidden form control 값에 둔다. file input은 파일 선택 이벤트를 받는 용도일 뿐이며, 보안상 브라우저가 실제 로컬 경로를 제공하지 않는다는 전제에 의존한다. Import된 Data URL에는 원본 파일명이 없으므로 `Imported Data URL`과 MIME/계산 크기만 표시한다.

추가/삭제/정렬 구현은 기존 계약을 따른다.

- `createContentPartElement(init, container)`는 DOM 요소를 반환만 한다.
- 호출부가 `.content-parts-container > .items`에 append하고 `checkEmpty(outer)`를 호출한다.
- 삭제 listener에 넘기는 container는 반드시 `.content-parts-container` outer다.
- `createMessageElement()` 안에서 `enableDragSort(contentPartsContainer)`를 한 번 호출한다.
- part와 parent Message가 모두 draggable이므로 기존 `mousedown` 잠금 및 `dragstart.stopPropagation()` 규칙을 그대로 사용한다.
- content part의 디자인에는 `.content-part` 같은 별도 클래스를 쓰고 구조용 `.item`/`.items`에 타입별 스타일을 넣지 않는다.

## 7. 파일 선택과 Base64 생성

media part별 file input의 `accept`는 각각 `image/*`, `audio/*`, `video/*`로 제한한다. 사용자가 파일을 선택하면 다음 순서로 처리한다.

1. 기존 part 값을 즉시 지우지 않고 선택 파일의 MIME을 확인한다.
2. `FileReader.readAsDataURL(file)`을 실행한다.
3. `load` 성공 시 결과가 예상 modality의 `data:*/*;base64,` 형식인지 검증한다.
4. 성공하면 hidden Data URL, 파일명, MIME, 원본 byte 크기를 한 번에 교체한다.
5. `error` 또는 형식 불일치 시 기존 값을 유지하고 part에 오류를 표시한다.

FileReader가 생성한 전체 문자열만 wire 값으로 사용한다. Base64 payload를 다시 decode/re-encode하지 않는다. 이로써 불필요한 복사와 데이터 변형을 피한다.

UI에는 원본 byte 크기와 Base64 확장 추정치를 표시한다. Base64 payload 길이는 대략 `4 * ceil(rawBytes / 3)`이며 Data URL prefix와 JSON 구조가 추가된다. 크기는 정보/warning 용도이고 SEND를 차단하지 않는다.

## 8. Role capability 정책

Role capability는 “UI에서 새 part를 추가할 수 있는가”에만 적용한다.

| role | String | + Text | + Image | + Audio | + Video |
|---|---:|---:|---:|---:|---:|
| `system` | O | O | X | X | X |
| `user` | O | O | O | O | O |
| `assistant` | O | O | O | O | O |
| `tool` | O | O | X | X | X |

`applyRole()`은 기존 reasoning/tool-call 영역 표시와 함께 add button visibility 및 warning을 갱신한다.

role 변경 시 정책:

- 현재 content와 part DOM을 삭제하거나 변환하지 않는다.
- 새 role에서 권장되지 않는 기존 media part에는 warning badge를 표시한다.
- Message 단위 warning에 비권장 part 종류와 개수를 요약한다.
- SEND, COPY, EXPORT는 막지 않고 그대로 직렬화한다.
- Import한 `system`/`tool` media part도 같은 warning만 표시한다.

예를 들어 `user`의 image part를 유지한 채 role을 `tool`로 바꾸면 `+ Image` 버튼은 숨겨지지만 image part와 full Data URL은 남고 전송도 가능하다. 서버의 4xx 응답 자체가 tester의 관찰 결과가 될 수 있기 때문이다.

## 9. String ↔ Content Parts 변환 규칙

`content-format` 선택 변경은 단순 hide/show가 아니라 명시적 변환 작업이다. 비활성 editor의 오래된 값을 실수로 되살리지 않도록, 전환이 확정될 때 대상 editor를 항상 현재 source로 교체한다.

### 9.1 String → Content Parts

항상 무손실로 처리한다.

```text
"Hello"
  →
[{"type":"text","text":"Hello"}]
```

- 빈 string도 빈 text part 하나로 바꾼다.
- 기존 parts DOM은 `replaceChildren()`로 비운 뒤 text part 하나만 생성한다.
- 변환 후 `checkEmpty(contentPartsContainer)`를 호출한다.

### 9.2 Content Parts → String

다음 조건을 모두 만족하면 무손실로 즉시 변환한다.

- part가 정확히 하나다.
- 그 part의 type이 text다.

그 외의 경우에는 전환 전에 dialog를 띄운다.

```text
String으로 전환하면 Content Part 일부가 제거됩니다.
- text part: N개
- media part: N개

[취소] [Text만 합쳐 전환]
```

사용자가 `Text만 합쳐 전환`을 선택하면 part 순서대로 text 값만 추출해 `\n` 하나로 연결하고, media part는 버린다. text part가 없으면 빈 string이 된다. 취소하면 format selector와 DOM 모두 `Content Parts` 상태로 되돌린다.

브라우저 기본 `confirm()`만으로 손실 내용과 선택지를 충분히 표현하기 어려우므로 template에 정적 dialog를 두는 방식을 우선한다. 테스트에서는 취소와 확정 경로를 모두 검증한다.

## 10. 직렬화와 Validation

`buildRequest()`는 Message별 `.content-format`을 읽어 다음처럼 분기한다.

- `string`: `.content-string.value`를 그대로 `message.content`에 사용
- `parts`: `.content-parts-container > .items > .item`을 DOM 순서대로 읽어 canonical wire part 생성

직렬화 helper를 분리한다.

```text
readMessageContent(messageDiv)
readContentPart(partDiv)
createContentPartElement(init, container)
validateContentPart(partDiv)
updateRoleCapability(messageDiv)
```

Validation은 blocking error와 advisory warning을 구분한다.

| 조건 | 수준 | 동작 |
|---|---|---|
| 지원하지 않는 imported part type | Import error | 전체 Import 취소, 기존 DOM 유지 |
| array가 아닌 structured content | Import error | 전체 Import 취소 |
| text part의 `text`가 string이 아님 | Import error | 전체 Import 취소 |
| media URL이 string이 아님 | Import error | 전체 Import 취소 |
| media URL이 Base64 Data URL이 아님 | Import error | URL/`file://`/raw Base64를 받지 않음 |
| file 선택 결과가 선택 modality와 다름 | 편집 error | 기존 part 값 유지 |
| media part에 Data URL이 없음 | blocking error | SEND/COPY/EXPORT 전에 해당 Message로 이동해 안내 |
| role capability와 기존 part 불일치 | warning | 데이터 유지, SEND/COPY/EXPORT 허용 |
| 대용량 Base64 | information/warning | 크기 표시, 전송 허용 |

role warning 이외의 blocking error는 UI가 지원한다고 주장한 canonical part를 잘못 직렬화하지 않기 위한 것이다. tester의 비표준 role 조합 실험은 허용하지만, URL/file 입력이나 불완전한 media part처럼 명시적으로 범위에서 제외한 상태는 전송하지 않는다.

## 11. Import / Export

### 11.1 Import

`parseImportedRequest()`의 `contentToString()` 정규화를 제거하고 content union을 그대로 분기한다.

- string이면 `{ contentFormat: 'string', content: 원본 string }`
- array이면 각 part를 지원 logical type으로 parse하고 `{ contentFormat: 'parts', contentParts: [...] }`
- `null`, object, number 등은 명시적 Import error
- 빈 array는 유효한 `ContentPart[]`로 유지
- role capability 위반은 parse 성공 후 warning만 표시
- 어느 Message에서든 구조 오류가 나면 현재와 같이 Import 전체를 원자적으로 취소

`createMessageElement(init)`는 union 형태를 보고 정확한 format selector와 DOM을 생성한다. Import 시 string을 single text part로, array를 JSON 문자열 textarea로 바꾸지 않는다.

### 11.2 Export

`exportRequestJson()`은 `buildRequest()`가 만든 full body를 사용한다. Preview 축약 설정을 참조하지 않는다. accepted Import → 즉시 Export 시 지원 content에 대해 다음이 같아야 한다.

- `typeof content`/array 여부
- part 순서와 개수
- 각 part의 canonical 값
- Data URL 전체 문자열

### 11.3 Response → Messages 복사

`copyResponseToMessage()`가 `responseMessage.content`를 string으로 강제하지 않도록 수정한다.

- string response content는 String editor로 복사
- 지원 part array는 Content Parts editor로 복사
- non-streaming renderer는 array를 `[object Object]`로 표시하지 않고 part 요약 또는 pretty JSON으로 표시
- unsupported response part는 원본 응답 표시에는 남기되, Message 복사 시 지원 범위 오류를 명확히 보여주고 자동 손실 복사를 하지 않는다.

streaming은 기존 text delta 경로를 유지한다. structured array delta가 들어오면 part 배열로 누적하고, string delta와 structured delta가 혼합되면 이미 누적된 string을 text part로 승격해 순서를 보존한다. 결과 복사 시 최종 누적 형태에 따라 String 또는 Content Parts를 선택한다.

## 12. Request Preview, COPY, SEND

Request Preview에 `Base64 축약 표시` checkbox와 `전체 Request JSON 복사` 버튼을 추가한다. 축약 표시는 기본값을 켠다.

Preview 전용 JSON stringify replacer는 Data URL을 다음과 같은 설명 문자열로 바꾼다.

```text
data:image/png;base64,<omitted: 24576 chars>
```

중요한 분리 규칙:

- `buildRequest()`는 언제나 full body만 만든다.
- `renderRequestView(body, { compactBase64 })`만 표시용 문자열을 축약한다.
- SEND의 `JSON.stringify(body)`는 full body를 사용한다.
- COPY는 preview DOM의 `textContent`를 복사하지 않고 full body를 별도로 stringify한다.
- EXPORT도 preview DOM이나 compact serializer를 사용하지 않는다.
- 축약 구현은 full body를 deep-clone하지 않고 preview stringify 단계에서만 문자열을 대체한다.

Preview가 “마지막으로 실제 전송한 body”를 나타내는 현재 의미를 유지하기 위해 `lastRequestBody` 참조를 보관한다. checkbox 토글과 COPY는 이 body를 사용한다. DOM 편집만 하고 아직 SEND하지 않은 상태의 Export는 현재 DOM에서 새 body를 생성한다.

clipboard API를 사용할 수 없는 환경에서는 기존 외부 의존성 없이 임시 textarea 기반 copy fallback을 둔다. 성공 상태에는 “전체 Base64 포함”임을 표시해 축약 preview를 복사했다고 오해하지 않게 한다.

## 13. Proxy 영향과 대용량 Base64 검증

`server.py`는 JSON을 parse하거나 media를 업로드하지 않는다. 따라서 기능 구현을 위해 route, 저장소, multipart 처리, 임시 파일 관리 코드를 추가하지 않고 opaque proxy를 유지한다.

다만 현재 `_proxy()`는 다음 코드 경계에서 request 전체를 메모리에 올린다.

```text
Content-Length 읽기
  → self.rfile.read(length)
  → bytes body 보유
  → http.client로 upstream 전송
```

영향:

- raw media가 `N` bytes라면 Base64 payload만 대략 `4 * ceil(N / 3)` bytes다.
- 브라우저에는 FileReader 결과 string, DOM 보관 값, request serialization/fetch body가 동시에 존재할 수 있다.
- proxy는 in-flight request마다 적어도 `Content-Length` 크기의 body bytes를 보유한다.
- `ThreadingServer`이므로 동시 대용량 요청 수에 따라 메모리 사용량이 곱해진다.
- Python/http.client 및 OS buffer까지 포함한 정확한 peak는 플랫폼에 따라 달라지므로 단일 고정 배수로 보장하지 않는다.

이번 범위에서는 proxy upload streaming이나 request size limit을 도입하지 않는다. 대신 README에 위 특성과 “큰 영상은 브라우저·proxy·upstream 모두의 메모리 여유를 확인해 단계적으로 시험할 것”을 기록한다.

검증을 위해 Python stdlib만 사용하는 `tests/test_proxy.py`를 추가한다.

- loopback fake upstream과 `make_handler()`를 임시 port에서 실행
- 작은 canonical multimodal JSON body의 byte-for-byte 전달 검증
- 1 MiB 이상 deterministic binary fixture를 Base64 Data URL로 만든 body 전달 검증
- upstream이 받은 `Content-Length`, body 길이, SHA-256이 client 전송값과 같은지 검증
- response streaming/기존 route가 회귀하지 않는 최소 smoke test

이 테스트는 opaque 전달 정확성을 검증하며 특정 머신의 peak memory 수치를 합격 기준으로 삼지는 않는다. 메모리 관찰은 별도 수동 체크리스트로 기록한다.

## 14. Playwright E2E 계획

기존 테스트 파일을 확장하고 payload 경계 전용 spec을 추가한다. media fixture는 저장소에 큰 바이너리를 넣지 않고 Playwright의 in-memory file payload(`name`, `mimeType`, `buffer`)로 `setInputFiles()`한다.

### `tests/messages.spec.js`

- 기본 Message가 String format인지
- String 편집 값 유지
- Content Parts 전환 시 single text part 생성
- text/image/audio/video CRUD
- user/assistant의 네 add button 노출
- system/tool의 Text만 노출
- role 변경 후 기존 media 유지, warning 표시, SEND 비차단
- 무손실 parts → string 전환
- 손실 전환 취소와 “Text만 합쳐 전환” 결과
- 빈 array와 빈 string을 서로 다른 상태로 유지

### `tests/import-export.spec.js`

- string content Import → Export가 string 유지
- single text part array Import → Export가 array 유지
- mixed text/image/audio/video array의 순서 및 Data URL round-trip
- assistant array content와 tool role warning Import
- URL, `file://`, unknown part type, 잘못된 shape Import의 원자적 실패
- 기존 string Import 회귀 테스트 유지

### `tests/drag-sort.spec.js`

- Message 안 Content Parts의 up/down 정렬
- nested part drag가 parent Message drag를 오염시키지 않음
- part reorder가 request JSON 순서에 반영됨
- 기존 Message/Tool Call drag 테스트 유지

### 신규 `tests/multimodal-request.spec.js`

- in-memory image/audio/video 파일 선택 후 정확한 Data URL 생성
- 파일 교체 시 part 값과 metadata가 함께 교체됨
- MIME 불일치/read error에서 기존 값 유지
- route interception으로 SEND body에 full Base64가 포함되는지 검증
- compact Preview에는 full payload가 보이지 않는지 검증
- 축약 해제 Preview에는 full payload가 보이는지 검증
- COPY가 compact Preview와 무관하게 full Base64를 복사하는지 검증
- Export download JSON에도 full Base64가 포함되는지 검증
- supported multimodal assistant response를 Messages에 복사했을 때 array가 유지되는지 검증

Playwright의 request interception은 실제 vLLM 없이 `/v1/chat/completions` 요청 JSON을 관찰하고 deterministic response를 반환하는 데 사용한다. 브라우저 테스트에서 proxy의 메모리 특성까지 증명하려 하지 않는다.

## 15. 구현 단계

### Phase 1 — 계약과 회귀 테스트 고정

1. content union, canonical part shape, role capability 상수를 코드 주석으로 고정한다.
2. 기존 string 경로의 Playwright 회귀 테스트를 먼저 유지/보강한다.
3. 새 multimodal fixtures와 selector 명칭을 정한다.

### Phase 2 — Content Parts DOM 편집기

1. `tpl-message`에 format selector와 parts outer/items 구조를 추가한다.
2. `tpl-content-part`와 `createContentPartElement()`를 구현한다.
3. add/delete/clear/up/down/drag 동작을 기존 helper 계약에 연결한다.
4. role별 add button과 advisory warning을 구현한다.
5. `vllm-tester.css`에 content part 전용 스타일을 추가한다.

### Phase 3 — 변환과 직렬화

1. String → Parts 무손실 변환을 구현한다.
2. Parts → String 손실 확인 dialog와 deterministic text join을 구현한다.
3. `buildRequest()`를 `readMessageContent()` 기반 union 직렬화로 변경한다.
4. incomplete media의 blocking validation과 focus/scroll 안내를 연결한다.

### Phase 4 — 파일 읽기

1. modality별 accept/MIME 규칙을 적용한다.
2. `FileReader.readAsDataURL()` load/error 처리를 구현한다.
3. Data URL, 파일 metadata, size 표시를 DOM state에 반영한다.
4. 파일 교체 실패 시 기존 값을 보존한다.

### Phase 5 — Import/Export 및 response 복사

1. `contentToString()`을 제거하고 union parser를 구현한다.
2. Import의 지원 part/Data URL validation을 추가한다.
3. Export round-trip을 검증한다.
4. response 표시와 `copyResponseToMessage()`가 array content를 보존하게 한다.
5. streaming structured content 누적을 방어적으로 처리한다.

### Phase 6 — Preview/COPY 분리

1. preview-only compact serializer를 추가한다.
2. Base64 축약 checkbox와 full JSON copy 버튼을 추가한다.
3. SEND/COPY/EXPORT가 compact serializer를 참조하지 않는지 테스트한다.

### Phase 7 — Proxy 문서화와 전체 검증

1. stdlib loopback proxy 테스트를 추가한다.
2. README에 지원 범위, Data URL 예시, 메모리 특성, 제외 항목을 문서화한다.
3. Playwright 전체 suite와 Python proxy test를 실행한다.
4. `git diff --check`와 최종 변경 범위를 확인한다.

## 16. 예상 변경 파일

| 파일 | 변경 내용 |
|---|---|
| `vllm-tester.html` | templates, content editor, union serializer/parser, FileReader, role warning, conversion, preview/copy, response-copy 처리 |
| `vllm-tester.css` | format selector, content part 카드, media metadata, warning, preview toolbar/dialog 스타일 |
| `tests/messages.spec.js` | format/part CRUD, role capability, conversion E2E |
| `tests/import-export.spec.js` | union 및 Base64 round-trip, invalid import E2E |
| `tests/drag-sort.spec.js` | nested content part 정렬 E2E |
| `tests/multimodal-request.spec.js` | 파일 선택, SEND/COPY/EXPORT/Preview payload E2E |
| `tests/test_proxy.py` | Python stdlib opaque proxy와 큰 JSON body 검증 |
| `README.md` | 사용법, 지원/제외 범위, Base64 및 메모리 주의사항 |

`server.py`, `package.json`, `playwright.config.js`, `vllm-tester-base.css`는 기본적으로 변경하지 않는다. 테스트가 현재 공개 helper만으로 proxy를 띄우기 어렵다는 것이 확인될 때에만 `server.py`의 동작을 바꾸지 않는 범위에서 testability용 helper 추출을 별도 검토한다.

## 17. Definition of Done

- [ ] 각 Message에서 String과 Content Parts를 명시적으로 선택할 수 있다.
- [ ] String request와 single text-part array request가 서로 normalize되지 않는다.
- [ ] text/image/audio/video part를 순서대로 추가·편집·삭제·정렬할 수 있다.
- [ ] media 파일은 브라우저에서 Base64 Data URL로 변환되며 서버 업로드가 발생하지 않는다.
- [ ] URL, `file://`, 범용 파일 part를 UI와 Import가 지원하지 않는다.
- [ ] system/tool은 Text만 새로 추가할 수 있고 user/assistant는 네 타입을 모두 추가할 수 있다.
- [ ] role 변경 후 비권장 part는 유지되며 warning만 표시되고 전송 가능하다.
- [ ] String → Parts가 single text part로 무손실 변환된다.
- [ ] Parts → String의 유일한 무손실 조건이 “text part 정확히 하나”로 구현된다.
- [ ] 그 외 Parts → String은 사용자의 취소/확정 없이는 상태를 바꾸지 않는다.
- [ ] Import → Export가 지원 content의 union branch, part 순서, 전체 Data URL을 보존한다.
- [ ] compact Preview는 Base64를 축약하지만 SEND/COPY/EXPORT는 full Base64를 사용한다.
- [ ] assistant의 supported array response를 Messages로 복사해 후속 request에 재사용할 수 있다.
- [ ] content part drag가 parent Message 및 tool-call drag 동작을 깨뜨리지 않는다.
- [ ] `server.py`가 request JSON을 해석·저장하지 않는 opaque proxy로 유지된다.
- [ ] proxy의 full-body buffering 및 동시 요청 메모리 영향이 README에 명시된다.
- [ ] Python stdlib proxy 테스트에서 canonical/대형 Base64 body가 byte-for-byte 전달된다.
- [ ] 기존 Playwright E2E와 신규 multimodal E2E가 모두 통과한다.
- [ ] `git diff --check`가 통과하고 계획 밖 기능 변경이 없다.
