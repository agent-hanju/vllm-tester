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
    const contentTA = page.locator('#messagesList .items .item:nth-child(2) .content-string');
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
    await page.locator('#messagesList .items .item:nth-child(1) .content-string').fill('Msg1');
    await page.locator('#messagesList .items .item:nth-child(2) .content-string').fill('Msg2');
    await page.locator('#messagesList .items .item:nth-child(1) [data-act="down"]').click();
    const contents = page.locator('#messagesList .items .item .content-string');
    await expect(contents.nth(0)).toHaveValue('Msg2');
    await expect(contents.nth(1)).toHaveValue('Msg1');
  });

  test('Move message up: reorders correctly', async ({ page }) => {
    await page.locator('#messagesList .items .item:nth-child(1) .content-string').fill('Msg1');
    await page.locator('#messagesList .items .item:nth-child(2) .content-string').fill('Msg2');
    await page.locator('#messagesList .items .item:nth-child(2) [data-act="up"]').click();
    const contents = page.locator('#messagesList .items .item .content-string');
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

test.describe('Message Content Formats and Parts', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/vllm-tester.html');
    await page.locator('#addMessage').waitFor({ state: 'visible' });
  });

  test('Default messages preserve String content format', async ({ page }) => {
    const formats = page.locator('#messagesList > .items > .message-item .content-format');
    await expect(formats).toHaveCount(2);
    await expect(formats.nth(0)).toHaveValue('string');
    await expect(formats.nth(1)).toHaveValue('string');
  });

  test('String to Content Parts creates one lossless text part', async ({ page }) => {
    const message = page.locator('#messagesList > .items > .message-item').nth(1);
    await message.locator('.content-string').fill('Hello multimodal');
    await message.locator('.content-format').selectOption('parts');

    const parts = message.locator('.content-parts-container > .items > .content-part');
    await expect(parts).toHaveCount(1);
    await expect(parts.locator('.cp-type')).toHaveText('TEXT');
    await expect(parts.locator('.cp-text')).toHaveValue('Hello multimodal');
  });

  test('User and assistant expose all part buttons; system and tool expose Text only', async ({ page }) => {
    const system = page.locator('#messagesList > .items > .message-item').nth(0);
    const user = page.locator('#messagesList > .items > .message-item').nth(1);
    await system.locator('.content-format').selectOption('parts');
    await user.locator('.content-format').selectOption('parts');

    await expect(system.locator('[data-part="text"]')).toBeVisible();
    await expect(system.locator('[data-part="image"]')).toBeHidden();
    await expect(user.locator('[data-part="image"]')).toBeVisible();
    await expect(user.locator('[data-part="audio"]')).toBeVisible();
    await expect(user.locator('[data-part="video"]')).toBeVisible();

    await system.locator('.role-select').selectOption('assistant');
    await expect(system.locator('[data-part="image"]')).toBeVisible();
    await system.locator('.role-select').selectOption('tool');
    await expect(system.locator('[data-part="text"]')).toBeVisible();
    await expect(system.locator('[data-part="video"]')).toBeHidden();
  });

  test('Role change retains incompatible media and only warns', async ({ page }) => {
    const message = page.locator('#messagesList > .items > .message-item').nth(1);
    await message.locator('.content-format').selectOption('parts');
    await message.locator('[data-part="image"]').click();
    const image = message.locator('.content-part[data-part-kind="image"]');
    await image.locator('.cp-file').setInputFiles({
      name: 'pixel.png',
      mimeType: 'image/png',
      buffer: Buffer.from([0x89, 0x50, 0x4e, 0x47]),
    });
    await expect(image.locator('.cp-validation')).toHaveClass(/v-ok/);

    await message.locator('.role-select').selectOption('tool');
    await expect(image).toHaveCount(1);
    await expect(image).toHaveClass(/capability-warning/);
    await expect(message.locator('.content-role-warning')).toContainText('tool role의 비권장 part');

    const built = await page.evaluate(() => buildRequest().body.messages[1]);
    expect(built.role).toBe('tool');
    expect(built.content[1].type).toBe('image_url');
    expect(built.content[1].image_url.url).toBe('data:image/png;base64,iVBORw==');
  });

  test('Single text part converts back to String without warning', async ({ page }) => {
    const message = page.locator('#messagesList > .items > .message-item').nth(1);
    await message.locator('.content-string').fill('round trip');
    await message.locator('.content-format').selectOption('parts');
    await message.locator('.content-format').selectOption('string');
    await expect(message.locator('.content-format')).toHaveValue('string');
    await expect(message.locator('.content-string')).toHaveValue('round trip');
    await expect(page.locator('#contentConversionDialog')).not.toBeVisible();
  });

  test('Lossy conversion can be cancelled or confirmed with ordered text join', async ({ page }) => {
    const message = page.locator('#messagesList > .items > .message-item').nth(1);
    await message.locator('.content-format').selectOption('parts');
    await message.locator('.cp-text').fill('first');
    await message.locator('[data-part="image"]').click();
    await message.locator('[data-part="text"]').click();
    await message.locator('.content-part[data-part-kind="text"]').nth(1).locator('.cp-text').fill('second');

    await message.locator('.content-format').selectOption('string');
    const dialog = page.locator('#contentConversionDialog');
    await expect(dialog).toBeVisible();
    await dialog.getByRole('button', { name: '취소' }).click();
    await expect(dialog).toBeHidden();
    await expect(message.locator('.content-format')).toHaveValue('parts');
    await expect(message.locator('.content-part')).toHaveCount(3);

    await message.locator('.content-format').selectOption('string');
    await expect(dialog).toBeVisible();
    await dialog.getByRole('button', { name: 'Text만 합쳐 전환' }).click();
    await expect(dialog).toBeHidden();
    await expect(message.locator('.content-format')).toHaveValue('string');
    await expect(message.locator('.content-string')).toHaveValue('first\nsecond');
  });

  test('Empty Content Parts serializes as an array rather than an empty String', async ({ page }) => {
    const message = page.locator('#messagesList > .items > .message-item').nth(1);
    await message.locator('.content-format').selectOption('parts');
    await message.locator('.clear-content-parts').click();
    const content = await page.evaluate(() => buildRequest().body.messages[1].content);
    expect(content).toEqual([]);
  });
});
