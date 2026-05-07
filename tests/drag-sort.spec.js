const { test, expect } = require('@playwright/test');

// HTML5 DnD: dragstart → dragover → drop → dragend 순서로 이벤트를 직접 발생시킨다.
// _dragSrc는 dragstart 핸들러에서 세팅되므로 dispatchEvent로 정상 트리거해야 한다.
async function dragTo(page, srcLocator, destLocator) {
  const dataTransfer = await page.evaluateHandle(() => new DataTransfer());

  // dragstart: _dragSrc 세팅 + placeholder 생성
  await srcLocator.dispatchEvent('dragstart', { dataTransfer });

  // dragover: placeholder를 target 위치로 이동 (clientY는 target 중앙)
  const destBox = await destLocator.boundingBox();
  const clientY = destBox.y + destBox.height * 0.75; // 아래쪽 — target 다음에 삽입
  await destLocator.dispatchEvent('dragover', { dataTransfer, clientY });

  // drop: _dragSrc를 placeholder 위치에 삽입
  await destLocator.dispatchEvent('drop', { dataTransfer });

  // dragend: 상태 정리
  await srcLocator.dispatchEvent('dragend', { dataTransfer });
}

test.describe('Drag & Drop Sorting — Tools', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/vllm-tester.html');
    await page.locator('#addTool').waitFor({ state: 'visible' });
  });

  test('Move down button is equivalent to moving item down', async ({ page }) => {
    await page.locator('#addTool').click();
    await page.locator('#addTool').click();
    await page.locator('#addTool').click();

    await page.locator('#toolsList .items .item:nth-child(1) .t-name').fill('A');
    await page.locator('#toolsList .items .item:nth-child(2) .t-name').fill('B');
    await page.locator('#toolsList .items .item:nth-child(3) .t-name').fill('C');

    await page.locator('#toolsList .items .item:nth-child(1) [data-act="down"]').click();

    const names = page.locator('#toolsList .items .item .t-name');
    await expect(names.nth(0)).toHaveValue('B');
    await expect(names.nth(1)).toHaveValue('A');
    await expect(names.nth(2)).toHaveValue('C');
  });

  test('Multiple move operations maintain correct order', async ({ page }) => {
    await page.locator('#addTool').click();
    await page.locator('#addTool').click();
    await page.locator('#addTool').click();

    await page.locator('#toolsList .items .item:nth-child(1) .t-name').fill('1');
    await page.locator('#toolsList .items .item:nth-child(2) .t-name').fill('2');
    await page.locator('#toolsList .items .item:nth-child(3) .t-name').fill('3');

    // Move 1 down twice: 1,2,3 -> 2,1,3 -> 2,3,1
    await page.locator('#toolsList .items .item:nth-child(1) [data-act="down"]').click();
    await page.locator('#toolsList .items .item:nth-child(2) [data-act="down"]').click();

    const names = page.locator('#toolsList .items .item .t-name');
    await expect(names.nth(0)).toHaveValue('2');
    await expect(names.nth(1)).toHaveValue('3');
    await expect(names.nth(2)).toHaveValue('1');
  });

  test('Up button at top does nothing', async ({ page }) => {
    await page.locator('#addTool').click();
    await page.locator('#addTool').click();

    await page.locator('#toolsList .items .item:nth-child(1) .t-name').fill('First');
    await page.locator('#toolsList .items .item:nth-child(2) .t-name').fill('Second');

    await page.locator('#toolsList .items .item:nth-child(1) [data-act="up"]').click();

    const names = page.locator('#toolsList .items .item .t-name');
    await expect(names.nth(0)).toHaveValue('First');
    await expect(names.nth(1)).toHaveValue('Second');
  });

  test('Down button at bottom does nothing', async ({ page }) => {
    await page.locator('#addTool').click();
    await page.locator('#addTool').click();

    await page.locator('#toolsList .items .item:nth-child(1) .t-name').fill('First');
    await page.locator('#toolsList .items .item:nth-child(2) .t-name').fill('Second');

    await page.locator('#toolsList .items .item:nth-child(2) [data-act="down"]').click();

    const names = page.locator('#toolsList .items .item .t-name');
    await expect(names.nth(0)).toHaveValue('First');
    await expect(names.nth(1)).toHaveValue('Second');
  });

  test('Drag: first item moves after second', async ({ page }) => {
    await page.locator('#addTool').click();
    await page.locator('#addTool').click();
    await page.locator('#addTool').click();

    await page.locator('#toolsList .items .item:nth-child(1) .t-name').fill('A');
    await page.locator('#toolsList .items .item:nth-child(2) .t-name').fill('B');
    await page.locator('#toolsList .items .item:nth-child(3) .t-name').fill('C');

    const src  = page.locator('#toolsList .items .item:nth-child(1)');
    const dest = page.locator('#toolsList .items .item:nth-child(2)');
    await dragTo(page, src, dest);

    const names = page.locator('#toolsList .items .item .t-name');
    await expect(names.nth(0)).toHaveValue('B');
    await expect(names.nth(1)).toHaveValue('A');
    await expect(names.nth(2)).toHaveValue('C');
  });
});

test.describe('Drag & Drop Sorting — Messages', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/vllm-tester.html');
    await page.locator('#addMessage').waitFor({ state: 'visible' });
    await page.locator('#clearMessages').click();
    await page.locator('#addMessage').click();
    await page.locator('#addMessage').click();
  });

  test('Message up/down buttons reorder correctly', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .f-content textarea').fill('SysMsg');
    await page.locator('#messagesList .items .item:nth-child(2) .f-content textarea').fill('UserMsg');

    await page.locator('#messagesList .items .item:nth-child(1) [data-act="down"]').click();

    const contents = page.locator('#messagesList .items .item .f-content textarea');
    await expect(contents.nth(0)).toHaveValue('UserMsg');
    await expect(contents.nth(1)).toHaveValue('SysMsg');
  });

  test('Drag: first message moves after second', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .f-content textarea').fill('First');
    await page.locator('#messagesList .items .item:nth-child(2) .f-content textarea').fill('Second');

    const src  = page.locator('#messagesList .items .item:nth-child(1)');
    const dest = page.locator('#messagesList .items .item:nth-child(2)');
    await dragTo(page, src, dest);

    const contents = page.locator('#messagesList .items .item .f-content textarea');
    await expect(contents.nth(0)).toHaveValue('Second');
    await expect(contents.nth(1)).toHaveValue('First');
  });

  test('Nested tool call up/down independent from parent message', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .role-select').selectOption('assistant');
    await page.locator('#messagesList .items .item:nth-child(1) .add-tc-btn').click();
    await page.locator('#messagesList .items .item:nth-child(1) .add-tc-btn').click();

    const tcNames = page.locator('#messagesList .items .item:nth-child(1) .tc-container .item .tc-name');
    await tcNames.nth(0).fill('FuncA');
    await tcNames.nth(1).fill('FuncB');

    await page.locator('#messagesList .items .item:nth-child(1) .tc-container .item:nth-child(1) [data-act="down"]').click();

    await expect(tcNames.nth(0)).toHaveValue('FuncB');
    await expect(tcNames.nth(1)).toHaveValue('FuncA');

    // Parent message should still be in original position (use direct child selector)
    await expect(page.locator('#messagesList > .items > .item')).toHaveCount(2);
  });
});
