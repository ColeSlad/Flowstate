// NODE_PATH=/tmp/flowstate-browser/node_modules node tests/recorded_browser_tests.cjs
// Uses playwright-core and installed Chrome; serves only files under web/.
const { chromium } = require('playwright-core');
const assert = require('node:assert/strict');
const { readFile } = require('node:fs/promises');
const { once } = require('node:events');
const http = require('node:http');
const path = require('node:path');

(async () => {
  const root = path.resolve(__dirname, '../web');
  const data = JSON.parse(await readFile(path.join(root, 'data/demo.json'), 'utf8'));
  const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml' };
  const server = http.createServer(async (request, response) => {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    const target = path.resolve(root, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (!target.startsWith(root + path.sep)) { response.writeHead(403).end(); return; }
    try {
      const content = await readFile(target);
      response.writeHead(200, { 'Content-Type': types[path.extname(target)] || 'text/plain' });
      response.end(content);
    } catch { response.writeHead(404).end(); }
  });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  let browser;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1160 }, reducedMotion: 'reduce', colorScheme: 'light' });
    const errors = [], requests = [], consoleErrors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => requests.push(new URL(request.url()).pathname));
    page.on('console', message => { if (message.type() === 'error') consoleErrors.push(message.text()); });
    const url = `http://127.0.0.1:${server.address().port}`;
    const ready = () => page.waitForFunction(() => document.body.dataset.connection === 'recorded');
    const position = () => page.locator('#recording-position').inputValue();
    await page.goto(url); await ready();
    assert.equal(await page.locator('body').getAttribute('data-mode'), 'recorded');
    assert.equal(await page.locator('#controls').isVisible(), false);
    assert.equal(await page.locator('[data-traffic="low"]').isDisabled(), true);
    assert.equal(await page.locator('#recording').isVisible(), true);
    assert.match(await page.title(), /Recorded/);
    assert.equal(await page.locator('#comparison-rows tr').count(), 5);
    assert.equal(await page.locator('[data-stage]').count(), 4);
    assert.equal(await page.locator('#play-recording').textContent(), 'Play recording');

    // Every displayed frame must come directly from its recorded snapshot.
    for (let index = 0; index < data.samples.length; index++) {
      await page.locator('#recording-position').evaluate((element, value) => {
        element.value = value; element.dispatchEvent(new Event('input', { bubbles: true }));
      }, index);
      const displayed = await page.evaluate(() => Object.fromEntries(['p99', 'arrival', 'rejected', 'failures'].map(id => [id, document.getElementById(id).textContent])));
      const sample = data.samples[index];
      const fmt = (value, digits = 0) => value == null ? '—' : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
      assert.deepEqual(displayed, {p99: fmt(sample.p99_ms, 1), arrival: fmt(sample.arrival_rate), rejected: fmt(sample.rejected), failures: fmt(sample.failed)});
    }
    assert.equal(await page.locator('#play-recording').textContent(), 'Replay');
    await page.locator('[data-stage="quiet"]').click();
    assert.equal(await position(), '0');
    assert.equal(await page.locator('#transitions li').count(), 1);
    assert.equal(await page.locator('#play-recording').textContent(), 'Play recording');
    await page.locator('#play-recording').click();
    await page.waitForFunction(() => Number(document.getElementById('recording-position').value) > 0);
    await page.locator('#play-recording').click();
    const paused = await position();
    await page.waitForTimeout(2700); // Exceeds the live stale-connection threshold.
    assert.equal(await position(), paused);
    assert.equal(await page.locator('body').getAttribute('data-connection'), 'recorded');

    await page.locator('[data-stage="contention"]').click();
    assert.equal(Number(await position()), data.samples.findIndex(sample => sample.stage === 'contention'));
    await page.locator('#recording-position').focus();
    await page.keyboard.press('End');
    assert.equal(Number(await position()), data.samples.length - 1);
    await page.keyboard.press('ArrowLeft');
    await page.locator('#play-recording').click();
    await page.waitForFunction(() => document.getElementById('play-recording').textContent === 'Replay');
    assert.equal(Number(await position()), data.samples.length - 1);
    await page.locator('#play-recording').click(); // End -> restart at the first frame.
    assert.equal(await position(), '0');
    await page.evaluate(() => dispatchEvent(new PageTransitionEvent('pagehide', { persisted: true })));
    await page.evaluate(() => dispatchEvent(new PageTransitionEvent('pageshow', { persisted: true })));
    assert.equal(await page.locator('#play-recording').textContent(), 'Play recording');
    assert.equal(await page.locator('body').getAttribute('data-connection'), 'recorded');
    await page.locator('[data-stage="contention"]').click();
    await page.screenshot({ path: '/tmp/flowstate-recorded-desktop.png', fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForFunction(() => Number(document.getElementById('latency-chart').getAttribute('viewBox').split(' ')[2]) < 400);
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({ path: '/tmp/flowstate-recorded-mobile.png', fullPage: true });
    await page.emulateMedia({ colorScheme: 'dark' });
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await page.screenshot({ path: '/tmp/flowstate-recorded-dark.png', fullPage: true });
    assert.deepEqual(consoleErrors, []);

    // Missing or malformed recordings stay explicitly unavailable; no fake/live fallback.
    for (const payload of [null, {schema_version: 1, samples: []}, { ...data, samples: [...data.samples].reverse() }]) {
      await page.route('**/data/demo.json', route => route.fulfill({ status: payload ? 200 : 404, contentType: 'application/json', body: JSON.stringify(payload) }));
      await page.reload();
      await page.waitForFunction(() => document.body.dataset.connection === 'offline');
      assert.equal(await page.locator('#play-recording').isDisabled(), true);
      assert.equal(await page.locator('#recording-position').isDisabled(), true);
      assert.match(await page.locator('#recording-description').textContent(), /could not be loaded/);
      await page.unroute('**/data/demo.json');
    }
    await page.reload(); await ready();
    assert.equal(await page.locator('#play-recording').isDisabled(), false);
    assert.deepEqual(requests.filter(value => value.startsWith('/api/') || value === '/events'), []);
    assert.deepEqual(errors, []);
    console.log('Static replay: 76 exact frames, timeline, stages, pause/replay, keyboard, lifecycle, mobile, errors, and zero backend calls passed');
  } finally {
    await browser?.close();
    await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
