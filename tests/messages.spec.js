const { test, expect } = require('@playwright/test');

test.describe('Messages List CRUD', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/vllm-tester.html');
    await page.locator('#addMessage').waitFor({ state: 'visible' });
  });

  test('Page load: has 2 default messages (system + user)', async ({ page }) => {
    await expect(page.locator('#messagesList .items > .item')).toHaveCount(2);
    await expect(page.locator('#messagesList > p.empty')).toBeHidden();
  });

  test('First message has role system', async ({ page }) => {
    await expect(page.locator('#messagesList .items .item:nth-child(1) .role-select')).toHaveValue('system');
  });

  test('Second message has role user', async ({ page }) => {
    await expect(page.locator('#messagesList .items .item:nth-child(2) .role-select')).toHaveValue('user');
  });

  test('Add message after user creates assistant', async ({ page }) => {
    await page.locator('#addMessage').click();
    await expect(page.locator('#messagesList .items > .item')).toHaveCount(3);
    await expect(page.locator('#messagesList .items .item:nth-child(3) .role-select')).toHaveValue('assistant');
  });

  test('Change role to assistant shows reasoning field', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .role-select').selectOption('assistant');
    await expect(page.locator('#messagesList .items .item:nth-child(1) .f-reasoning')).not.toHaveAttribute('hidden', '');
  });

  test('Change role to user hides reasoning field', async ({ page }) => {
    const roleSelect = page.locator('#messagesList .items .item:nth-child(1) .role-select');
    await roleSelect.selectOption('assistant');
    await roleSelect.selectOption('user');
    await expect(page.locator('#messagesList .items .item:nth-child(1) .f-reasoning')).toHaveAttribute('hidden', '');
  });

  test('Change role to tool shows tool_call_id field', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .role-select').selectOption('tool');
    await expect(page.locator('#messagesList .items .item:nth-child(1) .f-tool-call-id')).not.toHaveAttribute('hidden', '');
  });

  test('Change role to tool hides tool_calls section', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .role-select').selectOption('tool');
    await expect(page.locator('#messagesList .items .item:nth-child(1) .f-tool-calls')).toHaveAttribute('hidden', '');
  });

  test('Change role to assistant shows tool_calls section', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .role-select').selectOption('assistant');
    await expect(page.locator('#messagesList .items .item:nth-child(1) .f-tool-calls')).not.toHaveAttribute('hidden', '');
  });

  test('Edit message content persists', async ({ page }) => {
    const contentTA = page.locator('#messagesList .items .item:nth-child(2) .f-content textarea');
    await contentTA.fill('Hello, test message');
    await expect(contentTA).toHaveValue('Hello, test message');
  });

  test('Edit reasoning for assistant persists', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .role-select').selectOption('assistant');
    const reasoningTA = page.locator('#messagesList .items .item:nth-child(1) .f-reasoning textarea');
    await reasoningTA.fill('Thinking step by step...');
    await expect(reasoningTA).toHaveValue('Thinking step by step...');
  });

  test('Tool role: tool_call_id input is editable', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .role-select').selectOption('tool');
    const tcIdInput = page.locator('#messagesList .items .item:nth-child(1) .f-tool-call-id input');
    await tcIdInput.fill('call_abc123');
    await expect(tcIdInput).toHaveValue('call_abc123');
  });

  test('Delete message removes item', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) [data-act="del"]').click();
    await expect(page.locator('#messagesList .items > .item')).toHaveCount(1);
  });

  test('Clear all messages shows placeholder', async ({ page }) => {
    await page.locator('#clearMessages').click();
    await expect(page.locator('#messagesList .items > .item')).toHaveCount(0);
    await expect(page.locator('#messagesList > p.empty')).toBeVisible();
  });

  test('Move message down: reorders correctly', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .f-content textarea').fill('Msg1');
    await page.locator('#messagesList .items .item:nth-child(2) .f-content textarea').fill('Msg2');
    await page.locator('#messagesList .items .item:nth-child(1) [data-act="down"]').click();
    const contents = page.locator('#messagesList .items .item .f-content textarea');
    await expect(contents.nth(0)).toHaveValue('Msg2');
    await expect(contents.nth(1)).toHaveValue('Msg1');
  });

  test('Move message up: reorders correctly', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .f-content textarea').fill('Msg1');
    await page.locator('#messagesList .items .item:nth-child(2) .f-content textarea').fill('Msg2');
    await page.locator('#messagesList .items .item:nth-child(2) [data-act="up"]').click();
    const contents = page.locator('#messagesList .items .item .f-content textarea');
    await expect(contents.nth(0)).toHaveValue('Msg2');
    await expect(contents.nth(1)).toHaveValue('Msg1');
  });
});

test.describe('Nested Tool Calls in Messages', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/vllm-tester.html');
    await page.locator('#addMessage').waitFor({ state: 'visible' });
    // Switch first message to assistant to get tool calls section
    await page.locator('#messagesList .items .item:nth-child(1) .role-select').selectOption('assistant');
  });

  test('Tool calls placeholder visible initially', async ({ page }) => {
    const tcEmpty = page.locator('#messagesList .items .item:nth-child(1) .tc-container p.empty');
    await expect(tcEmpty).toBeVisible();
  });

  test('Add tool call: creates nested item and hides placeholder', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .add-tc-btn').click();
    const tcItems = page.locator('#messagesList .items .item:nth-child(1) .tc-container > .items > .item');
    await expect(tcItems).toHaveCount(1);
    await expect(page.locator('#messagesList .items .item:nth-child(1) .tc-container p.empty')).toBeHidden();
  });

  test('Tool call id auto-generated with call_ prefix', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .add-tc-btn').click();
    const tcId = page.locator('#messagesList .items .item:nth-child(1) .tc-container .item .tc-id');
    const value = await tcId.inputValue();
    expect(value).toMatch(/^call_/);
  });

  test('Tool call name is editable', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .add-tc-btn').click();
    const tcName = page.locator('#messagesList .items .item:nth-child(1) .tc-container .item .tc-name');
    await tcName.fill('get_weather');
    await expect(tcName).toHaveValue('get_weather');
  });

  test('Tool call args valid JSON shows v-ok', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .add-tc-btn').click();
    const tcArgs = page.locator('#messagesList .items .item:nth-child(1) .tc-container .item .tc-args');
    const validation = page.locator('#messagesList .items .item:nth-child(1) .tc-container .item .validation');
    await tcArgs.fill('{"location":"Seoul"}');
    await expect(validation).toHaveClass(/v-ok/);
  });

  test('Tool call args invalid JSON shows v-err', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .add-tc-btn').click();
    const tcArgs = page.locator('#messagesList .items .item:nth-child(1) .tc-container .item .tc-args');
    const validation = page.locator('#messagesList .items .item:nth-child(1) .tc-container .item .validation');
    await tcArgs.fill('not json {]');
    await expect(validation).toHaveClass(/v-err/);
  });

  test('Delete tool call restores placeholder', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .add-tc-btn').click();
    await page.locator('#messagesList .items .item:nth-child(1) .tc-container .item [data-act="del"]').click();
    const tcItems = page.locator('#messagesList .items .item:nth-child(1) .tc-container > .items > .item');
    await expect(tcItems).toHaveCount(0);
    await expect(page.locator('#messagesList .items .item:nth-child(1) .tc-container p.empty')).toBeVisible();
  });

  test('Clear all tool calls: empties list and shows placeholder', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .add-tc-btn').click();
    await page.locator('#messagesList .items .item:nth-child(1) .add-tc-btn').click();
    await page.locator('#messagesList .items .item:nth-child(1) .clear-tc-btn').click();
    const tcItems = page.locator('#messagesList .items .item:nth-child(1) .tc-container > .items > .item');
    await expect(tcItems).toHaveCount(0);
    await expect(page.locator('#messagesList .items .item:nth-child(1) .tc-container p.empty')).toBeVisible();
  });

  test('Move tool call down: reorders correctly', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .add-tc-btn').click();
    await page.locator('#messagesList .items .item:nth-child(1) .add-tc-btn').click();
    const tcNames = page.locator('#messagesList .items .item:nth-child(1) .tc-container .item .tc-name');
    await tcNames.nth(0).fill('FuncA');
    await tcNames.nth(1).fill('FuncB');
    await page.locator('#messagesList .items .item:nth-child(1) .tc-container .item:nth-child(1) [data-act="down"]').click();
    await expect(tcNames.nth(0)).toHaveValue('FuncB');
    await expect(tcNames.nth(1)).toHaveValue('FuncA');
  });

  test('Move tool call up: reorders correctly', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .add-tc-btn').click();
    await page.locator('#messagesList .items .item:nth-child(1) .add-tc-btn').click();
    const tcNames = page.locator('#messagesList .items .item:nth-child(1) .tc-container .item .tc-name');
    await tcNames.nth(0).fill('FuncA');
    await tcNames.nth(1).fill('FuncB');
    await page.locator('#messagesList .items .item:nth-child(1) .tc-container .item:nth-child(2) [data-act="up"]').click();
    await expect(tcNames.nth(0)).toHaveValue('FuncB');
    await expect(tcNames.nth(1)).toHaveValue('FuncA');
  });
});
