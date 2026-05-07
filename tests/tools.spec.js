const { test, expect } = require('@playwright/test');

test.describe('Tools List CRUD', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/vllm-tester.html');
    await page.locator('#addTool').waitFor({ state: 'visible' });
  });

  test('Page load: empty state with placeholder visible', async ({ page }) => {
    await expect(page.locator('#toolsList p.empty')).toBeVisible();
    await expect(page.locator('#toolsList .items > .item')).toHaveCount(0);
  });

  test('Add tool: creates item and hides placeholder', async ({ page }) => {
    await page.locator('#addTool').click();
    await expect(page.locator('#toolsList .items > .item')).toHaveCount(1);
    await expect(page.locator('#toolsList p.empty')).toBeHidden();
  });

  test('Add multiple tools: all items present', async ({ page }) => {
    await page.locator('#addTool').click();
    await page.locator('#addTool').click();
    await page.locator('#addTool').click();
    await expect(page.locator('#toolsList .items > .item')).toHaveCount(3);
  });

  test('Edit tool name: value persists', async ({ page }) => {
    await page.locator('#addTool').click();
    const nameInput = page.locator('#toolsList .items .item .t-name').first();
    await nameInput.fill('get_weather');
    await expect(nameInput).toHaveValue('get_weather');
  });

  test('Default params have valid JSON schema on init', async ({ page }) => {
    await page.locator('#addTool').click();
    const paramsInput = page.locator('#toolsList .items .item .t-params').first();
    const value = await paramsInput.inputValue();
    expect(value).toContain('"type"');
    expect(value).toContain('"object"');
    expect(value).toContain('"properties"');
  });

  test('Default params show v-ok validation', async ({ page }) => {
    await page.locator('#addTool').click();
    const validationEl = page.locator('#toolsList .items .item .validation').first();
    await expect(validationEl).toHaveClass(/v-ok/);
  });

  test('Invalid JSON params shows v-err', async ({ page }) => {
    await page.locator('#addTool').click();
    const paramsInput = page.locator('#toolsList .items .item .t-params').first();
    const validationEl = page.locator('#toolsList .items .item .validation').first();
    await paramsInput.fill('not valid json {]');
    await expect(validationEl).toHaveClass(/v-err/);
  });

  test('Missing type field shows v-warn', async ({ page }) => {
    await page.locator('#addTool').click();
    const paramsInput = page.locator('#toolsList .items .item .t-params').first();
    const validationEl = page.locator('#toolsList .items .item .validation').first();
    await paramsInput.fill('{"properties":{}}');
    await expect(validationEl).toHaveClass(/v-warn/);
  });

  test('Valid JSON schema shows v-ok', async ({ page }) => {
    await page.locator('#addTool').click();
    const paramsInput = page.locator('#toolsList .items .item .t-params').first();
    const validationEl = page.locator('#toolsList .items .item .validation').first();
    await paramsInput.fill('{"type":"object","properties":{"loc":{"type":"string"}},"required":["loc"]}');
    await expect(validationEl).toHaveClass(/v-ok/);
  });

  test('Empty params shows v-err', async ({ page }) => {
    await page.locator('#addTool').click();
    const paramsInput = page.locator('#toolsList .items .item .t-params').first();
    const validationEl = page.locator('#toolsList .items .item .validation').first();
    await paramsInput.fill('');
    await expect(validationEl).toHaveClass(/v-err/);
  });

  test('Non-object properties value shows v-err', async ({ page }) => {
    await page.locator('#addTool').click();
    const paramsInput = page.locator('#toolsList .items .item .t-params').first();
    const validationEl = page.locator('#toolsList .items .item .validation').first();
    await paramsInput.fill('{"type":"object","properties":"invalid"}');
    await expect(validationEl).toHaveClass(/v-err/);
  });

  test('Required key not in properties shows v-warn', async ({ page }) => {
    await page.locator('#addTool').click();
    const paramsInput = page.locator('#toolsList .items .item .t-params').first();
    const validationEl = page.locator('#toolsList .items .item .validation').first();
    await paramsInput.fill('{"type":"object","properties":{"a":{}},"required":["a","missing"]}');
    await expect(validationEl).toHaveClass(/v-warn/);
  });

  test('Delete single tool: placeholder reappears', async ({ page }) => {
    await page.locator('#addTool').click();
    await expect(page.locator('#toolsList .items > .item')).toHaveCount(1);
    await page.locator('#toolsList .items .item [data-act="del"]').first().click();
    await expect(page.locator('#toolsList .items > .item')).toHaveCount(0);
    await expect(page.locator('#toolsList p.empty')).toBeVisible();
  });

  test('Delete first of two: second remains with correct name', async ({ page }) => {
    await page.locator('#addTool').click();
    await page.locator('#addTool').click();
    await page.locator('#toolsList .items .item:nth-child(1) .t-name').fill('Tool1');
    await page.locator('#toolsList .items .item:nth-child(2) .t-name').fill('Tool2');
    await page.locator('#toolsList .items .item:nth-child(1) [data-act="del"]').click();
    await expect(page.locator('#toolsList .items > .item')).toHaveCount(1);
    await expect(page.locator('#toolsList .items .item .t-name').first()).toHaveValue('Tool2');
  });

  test('Move tool down: reorders correctly', async ({ page }) => {
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

  test('Move tool up: reorders correctly', async ({ page }) => {
    await page.locator('#addTool').click();
    await page.locator('#addTool').click();
    await page.locator('#addTool').click();
    await page.locator('#toolsList .items .item:nth-child(1) .t-name').fill('A');
    await page.locator('#toolsList .items .item:nth-child(2) .t-name').fill('B');
    await page.locator('#toolsList .items .item:nth-child(3) .t-name').fill('C');
    await page.locator('#toolsList .items .item:nth-child(2) [data-act="up"]').click();
    const names = page.locator('#toolsList .items .item .t-name');
    await expect(names.nth(0)).toHaveValue('B');
    await expect(names.nth(1)).toHaveValue('A');
    await expect(names.nth(2)).toHaveValue('C');
  });

  test('Clear all tools: empties list and shows placeholder', async ({ page }) => {
    await page.locator('#addTool').click();
    await page.locator('#addTool').click();
    await expect(page.locator('#toolsList .items > .item')).toHaveCount(2);
    await page.locator('#clearTools').click();
    await expect(page.locator('#toolsList .items > .item')).toHaveCount(0);
    await expect(page.locator('#toolsList p.empty')).toBeVisible();
  });
});
