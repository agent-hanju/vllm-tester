# vLLM Tester — 개발 규칙

> 별도의 상태 관리 없이 DOM 자체를 상태로 사용하는 구조다. 아래 규칙은 이 stateless DOM 위에서 리스트 편집 UI를 일관되게 구현하기 위한 것이다.

## 빠른 시작

```bash
python server.py                  # http://localhost:8080, target: http://localhost:8000
python server.py --port 9000 --target http://10.0.0.5:8000 --api-key sk-xxx

npm install && npx playwright test # E2E 테스트 실행
```

## 파일 구조

| 파일 | 역할 |
|------|------|
| `server.py` | HTTP 서버 + vLLM 프록시 (Python 표준 라이브러리만 사용) |
| `vllm-tester.html` | 전체 UI — JS/HTML 모두 여기 |
| `vllm-tester.css` | 테마 스타일 (CSS 변수, 다크모드, 그림자 등) |
| `vllm-tester-base.css` | 기본 스타일 (CSS 변수 없음, `lite` 브랜치용) |
| `tests/` | Playwright E2E 테스트 |

CSS 수정 시: `main`/`dev` → `vllm-tester.css`, `lite` → `vllm-tester-base.css`

## 브랜치 구조

| 브랜치 | 설명 |
|--------|------|
| `main` | 안정판. `dev` 개발 후 머지 |
| `dev` | 기본 개발 라인 (모던 JS, 드래그 정렬, 다크모드) |
| `lite` | ES5 + XHR 호환 버전. 드래그·다크모드 없음 |

**`lite` 브랜치 제약**: arrow function/`const`/`let` 사용 금지, `fetch` 대신 `XMLHttpRequest`, 드래그 관련 코드 추가 금지.

---

## DOM 구조 규칙

### 리스트 컨테이너
동적 item을 추가·제거하는 모든 리스트는 다음 구조를 따른다:

```html
<div id="someList">
  <p class="empty">— 비어있음 —</p>
  <div class="items"></div>   <!-- template clone은 여기에만 -->
</div>
```

- `p.empty` : placeholder. `checkEmpty(container)`가 `.items` 안에 item이 없을 때 표시.
- `div.items` : 실제 item들이 들어오는 유일한 컨테이너. `innerHTML` 대신 `replaceChildren()` 또는 `appendChild`/`remove` 사용.
- `checkEmpty`는 항상 outer container(id가 있는 div)를 인자로 받는다. inner `.items`를 직접 넘기지 않는다.

### 구조용 클래스와 디자인 클래스 분리

`.item`, `.items`처럼 JS가 셀렉터로 사용하는 클래스는 구조용이다. 구조용 클래스에 CSS를 넣는다면, 해당 구조를 가진 **모든** 요소가 공유해야 하는 속성만 넣는다. 특정 타입에만 적용되는 시각 속성(색상, 간격 등)은 `tool-item`, `message-item`처럼 별도의 디자인 클래스에 넣고, 요소에 두 클래스를 함께 부여한다.

### 정적 요소는 HTML에 선언, hidden으로 제어
DOM, template 안에서 항상 하나만 존재하는 요소는 JS로 동적 생성하지 않고 HTML에 정적으로 선언한다. 표시 여부는 `hidden` 어트리뷰트로만 제어한다.

```html
<p class="r-status" hidden></p>
<div class="r-error" hidden><label>Error</label><pre></pre></div>
```

- N개가 동적으로 생성되는 요소만 `<template>`을 사용한다.

---

## 리스트 항목 편집 UI 구현 패턴

### `createXxxElement` 계약

각 항목 생성 함수는 **DOM 요소를 반환만** 한다. 리스트에 붙이는 것과 `checkEmpty` 호출은 호출부 책임이다. 함수 내부에서 `appendChild`나 `checkEmpty`를 직접 호출하지 않는다.

### `attachItemButtonListeners` — container 파라미터 규칙

`attachItemButtonListeners(div, container, onchange?)` 의 `container`는 반드시 **outer container**여야 한다. `.items` div를 넘기면 del 핸들러 안의 `checkEmpty(container)`가 `p.empty`를 찾지 못해 crash한다.

- 최상위 리스트: `$toolsList`, `$messagesList` 등 id가 있는 div.
- 중첩 리스트(tcContainer): `div.querySelector('.tc-container')` — 클로저로 캡처한 지역 변수로 전달한다. `parentNode`로 추론하지 않는다.

### 중첩 리스트 (tcContainer)

Message 항목 안의 Tool Calls처럼 리스트 안의 리스트도 **동일한 outer/items 구조**를 따른다:

```
div.item  (message)
  └─ div.tc-container          ← outer container
       ├─ p.empty
       └─ div.items
            └─ div.tool-call
```

### `checkEmpty` 계약

- 인자: outer container (`.tc-container` 포함). `.items`를 직접 넘기지 않는다.
- `.item` 클래스만 확인한다 (모든 항목 타입에 공통으로 부여됨).
- 항목 추가/삭제가 일어나는 모든 경로에서 반드시 호출한다.

### 목록 전체 교체 시 주의

목록을 통째로 바꿀 때는 `replaceChildren()`으로 `.items`만 비운다 (`.replaceChildren()`는 자식 노드를 모두 제거함). outer container는 건드리지 않는다. 보안 훅이 문자열 파싱 방식의 DOM 초기화를 차단하므로 DOM API만 사용한다.

### 드래그 정렬 (drag-sort) — `main`/`dev` 전용

모든 `.item`은 `attachItemButtonListeners` 안에서 `draggable="true"`와 `dragstart`/`dragend` 리스너가 붙는다. 컨테이너 쪽 `dragover`/`drop` 리스너는 `enableDragSort(container)`로 별도 설정한다.

- 최상위 리스트: 초기화 시점에 `enableDragSort($toolsList)`, `enableDragSort($messagesList)` 호출.
- 중첩 리스트(tcContainer): `createMessageElement` 안에서 `enableDragSort(tcContainer)` 호출.

**이벤트 전파 주의사항**

중첩 구조에서 `draggable="true"` 요소가 부모·자식 모두에 있으면 `dragstart`가 버블링되어 부모 핸들러가 `_dragSrc`를 덮어써 자식 드래그가 동작하지 않는다. 반드시 `dragstart` 핸들러 안에서 `e.stopPropagation()`을 호출해 전파를 차단한다.

```javascript
div.addEventListener('dragstart', (e) => {
  if (e.target.closest('input, textarea, select, button')) { e.preventDefault(); return; }
  e.stopPropagation(); // ← 부모 item의 dragstart 핸들러가 _dragSrc를 덮어쓰는 것을 막음
  _dragSrc = div;
  ...
});
```

input·textarea·select·button 위에서 드래그를 시작하면 `e.preventDefault()`로 취소한다. 단, `dragstart`의 `e.target`은 `draggable` 속성이 있는 div 자신이므로 클릭한 자식 요소를 가리키지 않는다. `mousedown`에서 실제 클릭 요소를 플래그로 기록하고 `dragstart`에서 읽어야 한다.

```javascript
div.addEventListener('mousedown', (e) => {
  div._dragLocked = !!e.target.closest('input, textarea, select, button');
});
div.addEventListener('dragstart', (e) => {
  if (div._dragLocked) { e.preventDefault(); return; }
  ...
});
```
