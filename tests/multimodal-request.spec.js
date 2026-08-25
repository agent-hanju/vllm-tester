const fs = require('fs');
const { test, expect } = require('@playwright/test');

async function openPage(page) {
  await page.goto('/vllm-tester.html');
  await page.locator('#addMessage').waitFor({ state: 'visible' });
}

async function userMessageInParts(page) {
  const message = page.locator('#messagesList > .items > .message-item').nth(1);
  await message.locator('.content-format').selectOption('parts');
  return message;
}

test.describe('Multimodal request payload', () => {
  test.beforeEach(async ({ page }) => openPage(page));

  test('Image, audio, and video files become canonical Base64 Data URLs', async ({ page }) => {
    const message = await userMessageInParts(page);
    await message.locator('[data-part="image"]').click();
    await message.locator('[data-part="audio"]').click();
    await message.locator('[data-part="video"]').click();

    await message.locator('.content-part[data-part-kind="image"] .cp-file').setInputFiles({
      name: 'image.png', mimeType: 'image/png', buffer: Buffer.from('image'),
    });
    await message.locator('.content-part[data-part-kind="audio"] .cp-file').setInputFiles({
      name: 'audio.wav', mimeType: 'audio/wav', buffer: Buffer.from('audio'),
    });
    await message.locator('.content-part[data-part-kind="video"] .cp-file').setInputFiles({
      name: 'video.mp4', mimeType: 'video/mp4', buffer: Buffer.from('video'),
    });

    const content = await page.evaluate(() => buildRequest().body.messages[1].content);
    expect(content).toEqual([
      { type: 'text', text: '' },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,aW1hZ2U=' } },
      { type: 'audio_url', audio_url: { url: 'data:audio/wav;base64,YXVkaW8=' } },
      { type: 'video_url', video_url: { url: 'data:video/mp4;base64,dmlkZW8=' } },
    ]);
  });

  test('Failed media replacement keeps the previous Data URL', async ({ page }) => {
    const message = await userMessageInParts(page);
    await message.locator('[data-part="image"]').click();
    const image = message.locator('.content-part[data-part-kind="image"]');
    await image.locator('.cp-file').setInputFiles({
      name: 'good.png', mimeType: 'image/png', buffer: Buffer.from('good'),
    });
    await expect(image.locator('.cp-data-url')).toHaveValue('data:image/png;base64,Z29vZA==');

    await image.locator('.cp-file').setInputFiles({
      name: 'wrong.wav', mimeType: 'audio/wav', buffer: Buffer.from('wrong'),
    });
    await expect(image.locator('.cp-validation')).toHaveClass(/v-err/);
    await expect(image.locator('.cp-data-url')).toHaveValue('data:image/png;base64,Z29vZA==');
  });

  test('Incomplete media blocks request construction', async ({ page }) => {
    const message = await userMessageInParts(page);
    await message.locator('[data-part="image"]').click();
    const result = await page.evaluate(() => {
      try {
        buildRequest();
        return null;
      } catch (error) {
        return { validation: error.isRequestValidation, message: error.message };
      }
    });
    expect(result).toEqual({ validation: true, message: 'image part: Base64 Data URL 형식이 아닙니다.' });
  });

  test('SEND uses full Base64 while Preview can compact it', async ({ page }) => {
    let sentBody;
    await page.route('**/v1/chat/completions', async route => {
      sentBody = route.request().postDataJSON();
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }] }),
      });
    });
    await page.locator('#baseUrl').fill('http://localhost:8080');
    const message = await userMessageInParts(page);
    await message.locator('[data-part="image"]').click();
    await message.locator('.content-part[data-part-kind="image"] .cp-file').setInputFiles({
      name: 'image.png', mimeType: 'image/png', buffer: Buffer.from('preview-data'),
    });

    await page.locator('#sendBtn').click();
    await expect(page.locator('#response .r-status')).toContainText('200');
    const fullDataUrl = 'data:image/png;base64,cHJldmlldy1kYXRh';
    expect(sentBody.messages[1].content[1].image_url.url).toBe(fullDataUrl);
    await expect(page.locator('#reqPreviewPre')).toContainText('<omitted:');
    await expect(page.locator('#reqPreviewPre')).not.toContainText(fullDataUrl);

    await page.locator('#reqPreviewDetails > summary').click();
    await page.locator('#compactBase64').uncheck();
    await expect(page.locator('#reqPreviewPre')).toContainText(fullDataUrl);
  });

  test('COPY and Export contain full Base64 independently of compact Preview', async ({ page, context }) => {
    await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: 'http://localhost:8080' });
    await page.route('**/v1/chat/completions', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }] }),
    }));
    await page.locator('#baseUrl').fill('http://localhost:8080');
    const message = await userMessageInParts(page);
    await message.locator('[data-part="audio"]').click();
    await message.locator('.content-part[data-part-kind="audio"] .cp-file').setInputFiles({
      name: 'audio.wav', mimeType: 'audio/wav', buffer: Buffer.from('copy-export'),
    });
    const fullDataUrl = 'data:audio/wav;base64,Y29weS1leHBvcnQ=';

    await page.locator('#sendBtn').click();
    await expect(page.locator('#copyRequestBtn')).toBeEnabled();
    await expect(page.locator('#reqPreviewPre')).not.toContainText(fullDataUrl);
    await page.locator('#reqPreviewDetails > summary').click();
    await page.locator('#copyRequestBtn').click();
    const clipboard = await page.evaluate(() => navigator.clipboard.readText());
    expect(JSON.parse(clipboard).messages[1].content[1].audio_url.url).toBe(fullDataUrl);

    const downloadPromise = page.waitForEvent('download');
    await page.locator('#exportBtn').click();
    const download = await downloadPromise;
    const exported = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
    expect(exported.messages[1].content[1].audio_url.url).toBe(fullDataUrl);
  });

  test('Supported assistant array response copies back as Content Parts', async ({ page }) => {
    const responseContent = [
      { type: 'text', text: 'generated' },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,iVBORw==' } },
    ];
    await page.route('**/v1/chat/completions', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        choices: [{ message: { role: 'assistant', content: responseContent }, finish_reason: 'stop' }],
      }),
    }));
    await page.locator('#baseUrl').fill('http://localhost:8080');
    await page.locator('#sendBtn').click();
    await expect(page.locator('#response .copy-btn')).toBeVisible();
    await page.locator('#response .copy-btn').click();

    const copied = page.locator('#messagesList > .items > .message-item').last();
    await expect(copied.locator('.role-select')).toHaveValue('assistant');
    await expect(copied.locator('.content-format')).toHaveValue('parts');
    expect(await page.evaluate(() => buildRequest().body.messages.at(-1).content)).toEqual(responseContent);
  });
});
