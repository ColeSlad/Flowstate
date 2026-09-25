// Optional browser checks: NODE_PATH=/tmp/flowstate-browser/node_modules node tests/browser_tests.cjs build-demo/flowstate_server
// Install playwright-core into that temporary directory; uses an isolated installed Chrome instance.
const { chromium } = require('playwright-core');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const http = require('node:http');
const net = require('node:net');
const path = require('node:path');
let server, serverExit;
async function startServer(binary, port) {
  server = spawn(binary, ['--vectors', '1000', '--dimension', '9', '--top-k', '20', '--cuda', 'off', '--port', String(port)], { stdio: ['ignore', 'pipe', 'pipe'] });
  let errors = '';
  server.stderr.on('data', value => errors += value);
  serverExit = once(server, 'exit');
  await new Promise((resolve, reject) => {
    server.stdout.once('data', resolve);
    server.once('error', reject);
    server.once('exit', code => reject(new Error(`Server startup exited ${code}: ${errors}`)));
  });
}
async function stopServer() {
  if (!server || server.exitCode !== null) return;
  server.kill('SIGTERM');
  const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error('Shutdown timed out')), 12000).unref());
  const [code] = await Promise.race([serverExit, timeout]);
  assert.equal(code, 0);
}
async function holdStream(url) {
  return new Promise((resolve, reject) => {
    const request = http.get(url, response => {
      assert.equal(response.statusCode, 200);
      response.once('data', () => resolve({ request, response }));
    });
    request.on('error', reject);
  });
}
(async () => {
  const probe = net.createServer(); probe.listen(0, '127.0.0.1'); await once(probe, 'listening');
  const port = probe.address().port; await new Promise(resolve => probe.close(resolve));
  const url = `http://127.0.0.1:${port}`;
  const binary = path.resolve(process.argv[2] || 'build-demo/flowstate_server');
  await startServer(binary, port);
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const held = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1080 }, reducedMotion: 'reduce' });
    const errors = [], consoleErrors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') consoleErrors.push(message.text()); });
    const live = () => page.waitForFunction(() => document.body.dataset.connection === 'live');
    await page.goto(url); await live();
    assert.equal(await page.locator('#top-k').inputValue(), '20');
    await page.locator('#top-k').selectOption('10');
    await page.waitForFunction(() => !document.querySelector('#controls').disabled);
    await page.locator('#top-k').selectOption('20');
    await page.waitForFunction(() => !document.querySelector('#controls').disabled);
    await page.locator('[data-traffic="burst"]').click();
    await page.waitForFunction(() => document.querySelector('[data-traffic="burst"]').getAttribute('aria-pressed') === 'true');
    assert.equal(await page.locator('#contention').isDisabled(), true);
    await page.locator('.comparison summary').click();
    assert.equal(await page.locator('#comparison-rows tr').count(), 5);
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.waitForFunction(() => Number(document.querySelector('#latency-chart').getAttribute('viewBox').split(' ')[2]) < 400);
    assert.deepEqual(consoleErrors, []);

    // The exact lifecycle events delivered to a restored back/forward-cache document.
    await page.evaluate(() => dispatchEvent(new PageTransitionEvent('pagehide', { persisted: true })));
    assert.equal(await page.locator('[data-traffic="low"]').isDisabled(), true);
    await page.evaluate(() => dispatchEvent(new PageTransitionEvent('pageshow', { persisted: true })));
    await live();

    await stopServer();
    await page.waitForFunction(() => document.body.dataset.connection === 'offline');
    assert.equal(await page.locator('[data-traffic="low"]').isDisabled(), true);
    await startServer(binary, port); await live();
    assert.equal(await page.locator('[data-traffic="low"]').isDisabled(), false);
    // Navigate away to release the browser stream, then fill all four server slots.
    await page.goto('about:blank');
    await new Promise(resolve => setTimeout(resolve, 1100));
    for (let i = 0; i < 4; i++) held.push(await holdStream(`${url}/events`));
    const rejected = page.waitForResponse(response => response.url().endsWith('/events') && response.status() === 503);
    await page.goto(url); await rejected;
    await page.waitForFunction(() => document.body.dataset.connection === 'offline');
    assert.equal(await page.locator('[data-traffic="low"]').isDisabled(), true);
    held.pop().response.destroy();
    await live();
    assert.equal(await page.locator('[data-traffic="low"]').isDisabled(), false);
    assert.deepEqual(errors, []);
    console.log('Browser controls, custom top-K, mobile layout, cached-page events, restart, and 503 recovery passed');
  } finally {
    for (const stream of held) { stream.response.destroy(); stream.request.destroy(); }
    await browser.close();
    try { await stopServer(); } finally { if (server?.exitCode === null) server.kill('SIGKILL'); }
  }
})().catch(error => { console.error(error); if (server?.exitCode === null) server.kill('SIGKILL'); process.exitCode = 1; });
