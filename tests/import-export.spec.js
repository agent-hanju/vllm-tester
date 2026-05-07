const { test, expect } = require('@playwright/test');

test.describe('Import / Export', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/vllm-tester.html');
    await page.locator('#exportBtn').waitFor({ state: 'visible' });
  });

  test('Export triggers download with JSON filename', async ({ page }) => {
    const downloadPromise = page.waitForEvent('download');
    await page.locator('#exportBtn').click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/^chat-request-\d+\.json$/);
  });

  test('Export shows success status', async ({ page }) => {
    const downloadPromise = page.waitForEvent('download');
    await page.locator('#exportBtn').click();
    await downloadPromise;
    const statusText = await page.locator('#ioStatus').textContent();
    expect(statusText).toContain('✓');
  });

  test('Import valid JSON: messages loaded', async ({ page }) => {
    const testJson = JSON.stringify({
      model: '',
      messages: [
        { role: 'system', content: 'You are helpful' },
        { role: 'user', content: 'What is 2+2?' },
      ],
    });
    await page.locator('#importFile').setInputFiles({
      name: 'test.json',
      mimeType: 'application/json',
      buffer: Buffer.from(testJson),
    });
    await page.waitForTimeout(300);
    await expect(page.locator('#messagesList .items > .item')).toHaveCount(2);
    await expect(page.locator('#ioStatus')).toContainText('✓');
  });

  test('Import valid JSON: tools loaded', async ({ page }) => {
    const testJson = JSON.stringify({
      model: '',
      messages: [{ role: 'user', content: 'test' }],
      tools: [
        {
          type: 'function',
          function: {
            name: 'get_weather',
            description: 'Get weather',
            parameters: { type: 'object', properties: {} },
          },
        },
      ],
    });
    await page.locator('#importFile').setInputFiles({
      name: 'test.json',
      mimeType: 'application/json',
      buffer: Buffer.from(testJson),
    });
    await page.waitForTimeout(300);
    await expect(page.locator('#toolsList .items > .item')).toHaveCount(1);
    const toolName = page.locator('#toolsList .items .item .t-name');
    await expect(toolName).toHaveValue('get_weather');
  });

  test('Import with hyperparameters: values applied to inputs', async ({ page }) => {
    const testJson = JSON.stringify({
      model: '',
      messages: [{ role: 'user', content: 'test' }],
      temperature: 0.7,
      max_tokens: 512,
      top_p: 0.9,
    });
    await page.locator('#importFile').setInputFiles({
      name: 'test.json',
      mimeType: 'application/json',
      buffer: Buffer.from(testJson),
    });
    await page.waitForTimeout(300);
    await expect(page.locator('#hp-temperature')).toHaveValue('0.7');
    await expect(page.locator('#hp-max_tokens')).toHaveValue('512');
    await expect(page.locator('#hp-top_p')).toHaveValue('0.9');
  });

  test('Import with tool_calls in assistant message', async ({ page }) => {
    const testJson = JSON.stringify({
      model: '',
      messages: [
        {
          role: 'assistant',
          content: '',
          tool_calls: [
            { id: 'call_1', type: 'function', function: { name: 'func', arguments: '{}' } },
          ],
        },
      ],
    });
    await page.locator('#importFile').setInputFiles({
      name: 'test.json',
      mimeType: 'application/json',
      buffer: Buffer.from(testJson),
    });
    await page.waitForTimeout(300);
    const tcItems = page.locator('#messagesList .items .item:nth-child(1) .tc-container > .items > .item');
    await expect(tcItems).toHaveCount(1);
  });

  test('Import with tool role message', async ({ page }) => {
    const testJson = JSON.stringify({
      model: '',
      messages: [
        { role: 'user', content: 'Call it' },
        {
          role: 'assistant',
          content: '',
          tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'f', arguments: '{}' } }],
        },
        { role: 'tool', tool_call_id: 'call_1', content: 'result' },
      ],
    });
    await page.locator('#importFile').setInputFiles({
      name: 'test.json',
      mimeType: 'application/json',
      buffer: Buffer.from(testJson),
    });
    await page.waitForTimeout(300);
    await expect(page.locator('#messagesList > .items > .item')).toHaveCount(3);
    await expect(page.locator('#messagesList > .items > .item:nth-child(3) .role-select')).toHaveValue('tool');
  });

  test('Import invalid JSON shows error in status', async ({ page }) => {
    await page.locator('#importFile').setInputFiles({
      name: 'bad.json',
      mimeType: 'application/json',
      buffer: Buffer.from('not valid json {]'),
    });
    await page.waitForTimeout(300);
    await expect(page.locator('#ioStatus')).toContainText('✗');
  });

  test('Import missing messages field shows error', async ({ page }) => {
    const testJson = JSON.stringify({ model: 'test-model' });
    await page.locator('#importFile').setInputFiles({
      name: 'test.json',
      mimeType: 'application/json',
      buffer: Buffer.from(testJson),
    });
    await page.waitForTimeout(300);
    await expect(page.locator('#ioStatus')).toContainText('✗');
  });

  test('Import overwrites existing tools', async ({ page }) => {
    await page.locator('#addTool').click();
    await page.locator('#toolsList .items .item .t-name').fill('OldTool');

    const testJson = JSON.stringify({
      model: '',
      messages: [{ role: 'user', content: 'test' }],
      tools: [
        {
          type: 'function',
          function: { name: 'new_tool', description: '', parameters: { type: 'object', properties: {} } },
        },
      ],
    });
    await page.locator('#importFile').setInputFiles({
      name: 'test.json',
      mimeType: 'application/json',
      buffer: Buffer.from(testJson),
    });
    await page.waitForTimeout(300);
    await expect(page.locator('#toolsList .items > .item')).toHaveCount(1);
    await expect(page.locator('#toolsList .items .item .t-name')).toHaveValue('new_tool');
  });

  test('Import overwrites existing messages', async ({ page }) => {
    const testJson = JSON.stringify({
      model: '',
      messages: [{ role: 'user', content: 'Imported message' }],
    });
    await page.locator('#importFile').setInputFiles({
      name: 'test.json',
      mimeType: 'application/json',
      buffer: Buffer.from(testJson),
    });
    await page.waitForTimeout(300);
    await expect(page.locator('#messagesList .items > .item')).toHaveCount(1);
    const roleSelect = page.locator('#messagesList .items .item:nth-child(1) .role-select');
    await expect(roleSelect).toHaveValue('user');
  });
});
