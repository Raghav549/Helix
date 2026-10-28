/* Optional end-to-end smoke test against a running HELIX server and Chromium.
 * Install puppeteer-core separately; set HELIX_BROWSER to a browser executable.
 * This uses real mouse/keyboard events and writes screenshots under out/.
 */
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer-core');

const browserPath = process.env.HELIX_BROWSER;
if (!browserPath) throw new Error('Set HELIX_BROWSER to the Chromium executable');
const base = process.env.HELIX_URL || 'http://127.0.0.1:8766/';
const screenshots = path.resolve('out/browser-smoke');
fs.mkdirSync(screenshots, { recursive: true });

(async () => {
  const errors = [];
  // Never disable the browser sandbox unless explicitly requested in an
  // isolated test environment. This is unrelated to the HELIX code sandbox.
  const args = process.env.HELIX_BROWSER_NO_SANDBOX === '1' ? ['--no-sandbox'] : [];
  const browser = await puppeteer.launch({ args, executablePath: browserPath, headless: true });
  try {
    const page = await browser.newPage();
    page.on('pageerror', error => errors.push(error.message));
    page.on('response', response => {
      if (response.status() >= 500) errors.push(`HTTP ${response.status()} ${response.url()}`);
    });
    const screenshot = name => page.screenshot({path: path.join(screenshots, `${name}.png`), fullPage: true});
    await page.setViewport({width: 1440, height: 900, deviceScaleFactor: 1});
    await page.goto(base, {waitUntil: 'networkidle2'});
    await page.waitForFunction(() => document.querySelector('#top-status').textContent.includes('4 of 4'));
    await screenshot('desktop-chat');
    await page.type('#chat-input', 'Explain recursion');
    await page.click('#chat-send');
    await page.waitForFunction(() => !document.querySelector('#chat-send').disabled && document.querySelector('.message.assistant'));
    await page.click('[data-view="image"]');
    await page.click('#image-submit');
    await page.waitForSelector('#image-output img');
    await screenshot('desktop-image');
    const upload = await page.$('#upload-image');
    await upload.uploadFile(path.resolve('artifacts/image-gen/physics-reference.png'));
    await page.click('#analyze-image');
    await page.waitForFunction(() => document.querySelector('#analysis-output').textContent.includes('synthetic_domain_caption'));
    await page.click('#edit-image');
    await page.waitForFunction(() => document.querySelector('#analysis-output').textContent.includes('edit generated'));
    await page.click('[data-view="code"]');
    await page.click('#run-code');
    await page.waitForFunction(() => document.querySelector('#code-output').textContent.includes('approval box'));
    await page.click('#code-approval');
    await page.click('#run-code');
    await page.waitForFunction(() => document.querySelector('#code-output').textContent.includes('Sum:'));
    await screenshot('desktop-code');
    await page.click('[data-view="research"]');
    await page.click('#math-submit');
    await page.waitForFunction(() => document.querySelector('#math-output').textContent.includes('3*x**2 + 2'));
    await page.click('#plot-csv');
    await page.waitForSelector('#csv-plot img');
    await screenshot('desktop-research');
    await page.setViewport({width: 390, height: 844, deviceScaleFactor: 1});
    await page.click('[data-view="chat"]');
    await screenshot('mobile-chat');
    const overflowing = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
    if (overflowing) errors.push('Mobile page has unintended horizontal overflow');
    console.log(JSON.stringify({browser: await browser.version(), checks: [
      'desktop and mobile layout', 'chat', 'physics render', 'image upload analysis/edit',
      'code approval and isolation', 'symbolic math', 'CSV plot'
    ], errors}, null, 2));
    if (errors.length) process.exitCode = 1;
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
