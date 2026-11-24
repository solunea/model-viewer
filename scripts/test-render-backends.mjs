import {strict as assert} from 'node:assert';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import {encode} from 'fast-png';
import {chromium} from 'playwright';

const origin = 'http://localhost:8765';
const root = new URL('../', import.meta.url);
const directories = new Map([
  ['viewer', new URL('packages/model-viewer/dist/', root)],
  ['effects', new URL('packages/model-viewer-effects/dist/', root)],
  ['three', new URL('node_modules/three/build/', root)],
  ['assets', new URL('packages/shared-assets/models/', root)]
]);
const html = `<!doctype html>
<script type="importmap">
{"imports":{"three":"/three/three.module.js","three/webgpu":"/three/three.webgpu.js","three/tsl":"/three/three.tsl.js"}}
</script>
<script type="module" src="/viewer/model-viewer-module.min.js"></script>
<style>model-viewer { display: inline-block; width: 300px; height: 240px }</style>
<model-viewer src="/assets/cube.gltf" shadow-intensity="1" view-helper camera-controls></model-viewer>
`;

async function start(t, settings = {}) {
  const browser = await chromium.launch({
    headless: true,
    channel: process.env.MODEL_VIEWER_TEST_CHROME_CHANNEL || 'chrome',
    args: ['--enable-unsafe-swiftshader']
  });
  t.after(() => browser.close());
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text());
  });
  if (settings.disableGPU) {
    await page.addInitScript(() =>
      Object.defineProperty(Navigator.prototype, 'gpu', {value: undefined}));
  } else {
    await page.addInitScript(() => {
      const original = GPUAdapter.prototype.requestDevice;
      GPUAdapter.prototype.requestDevice = async function (...args) {
        const device = await original.apply(this, args);
        globalThis.testGPUDevice = device;
        return device;
      };
    });
  }
  const markup = settings.backend == null ? html :
      html.replace('<model-viewer ', `<model-viewer render-backend="${settings.backend}" `);
  await page.route(`${origin}/**`, async route => {
    const {pathname} = new URL(route.request().url());
    if (pathname === '/') return route.fulfill({contentType: 'text/html', body: markup});
    const [, directory, ...path] = pathname.split('/');
    const base = directories.get(directory);
    if (!base || path.includes('..')) return route.fulfill({status: 404});
    try {
      const body = await readFile(new URL(path.join('/'), base));
      const contentType = pathname.endsWith('.js') ? 'text/javascript' :
          pathname.endsWith('.gltf') ? 'application/json' : 'application/octet-stream';
      await route.fulfill({body, contentType});
    } catch {
      await route.fulfill({status: 404});
    }
  });
  await page.goto(origin);
  return {page, errors};
}

function waitBackend(page, backend) {
  return page.waitForFunction(expected => {
    const viewer = document.querySelector('model-viewer');
    return viewer.loaded && viewer.activeRenderBackend === expected;
  }, backend, {timeout: 20000});
}

async function visiblePixels(page) {
  return page.evaluate(async () => {
    const blob = await document.querySelector('model-viewer').toBlob();
    const bitmap = await createImageBitmap(blob);
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext('2d');
    context.drawImage(bitmap, 0, 0);
    const data = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
    return data.filter((value, index) => index % 4 === 3 && value > 0).length;
  });
}

test('defaults to WebGL, can opt into WebGPU and recovers from GPU device loss', async t => {
  const {page, errors} = await start(t);
  await waitBackend(page, 'webgl');
  assert.ok(await visiblePixels(page) > 100);
  assert.deepEqual(await page.locator('model-viewer').evaluate(viewer => ({
    preference: viewer.renderBackend,
    attribute: viewer.getAttribute('render-backend'),
    deviceRequested: Boolean(globalThis.testGPUDevice)
  })), {preference: 'webgl', attribute: null, deviceRequested: false});

  await page.locator('model-viewer').evaluate(viewer => viewer.setAttribute('render-backend', 'webgpu'));
  await waitBackend(page, 'webgpu');
  assert.ok(await visiblePixels(page) > 100);
  const orbit = await page.evaluate(async () => {
    const viewer = document.querySelector('model-viewer');
    viewer.cameraOrbit = '20deg 65deg 4m';
    await viewer.updateComplete;
    viewer.jumpCameraToGoal();
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    viewer.removeAttribute('render-backend');
    return viewer.getCameraOrbit().toString();
  });
  await waitBackend(page, 'webgl');
  assert.equal(await page.locator('model-viewer').evaluate(viewer => viewer.renderBackend), 'webgl');
  assert.ok(await visiblePixels(page) > 100);
  await page.evaluate(() => document.querySelector('model-viewer').renderBackend = 'auto');
  await waitBackend(page, 'webgpu');
  await page.evaluate(() => globalThis.testGPUDevice.destroy());
  await waitBackend(page, 'webgl');
  assert.ok(await visiblePixels(page) > 100);
  assert.equal(await page.evaluate(() => document.querySelector('model-viewer')
      .getCameraOrbit().toString()), orbit);
  assert.deepEqual(errors, []);
});

test('loads the same model on WebGL when WebGPU is unavailable', async t => {
  const {page, errors} = await start(t, {backend: 'webgpu', disableGPU: true});
  await waitBackend(page, 'webgl');
  assert.equal(await page.locator('model-viewer').evaluate(viewer => viewer.renderBackend), 'webgpu');
  assert.ok(await visiblePixels(page) > 100);
  assert.deepEqual(errors, []);
});

test('keeps a skinned model renderable across WebGPU and WebGL', async t => {
  const {page, errors} = await start(t);
  await page.evaluate(async () => {
    const viewer = document.querySelector('model-viewer');
    const loaded = new Promise(resolve => viewer.addEventListener('load', resolve, {once: true}));
    viewer.src = '/assets/RobotExpressive.glb';
    await loaded;
    viewer.play();
  });
  for (const backend of ['webgpu', 'webgl', 'webgpu']) {
    await page.evaluate(value => document.querySelector('model-viewer').renderBackend = value, backend);
    await waitBackend(page, backend);
    assert.ok(await visiblePixels(page) > 100);
  }
  assert.deepEqual(errors, []);
});

test('draws an orbitable contact shadow on WebGPU', async t => {
  const {page, errors} = await start(t, {backend: 'webgpu'});
  await waitBackend(page, 'webgpu');
  await page.evaluate(async () => {
    const viewer = document.querySelector('model-viewer');
    viewer.shadowOrbit = '30deg 40deg';
    await viewer.updateComplete;
  });
  await page.waitForTimeout(500);
  assert.equal(await page.locator('model-viewer').evaluate(viewer => viewer.activeRenderBackend), 'webgpu');
  assert.ok(await visiblePixels(page) > 100);
  for (const backend of ['webgl', 'webgpu']) {
    await page.evaluate(value => document.querySelector('model-viewer').renderBackend = value, backend);
    await waitBackend(page, backend);
    assert.ok(await visiblePixels(page) > 100);
  }
  assert.deepEqual(errors, []);
});

test('renders each built-in effect on WebGPU', async t => {
  const {page, errors} = await start(t, {backend: 'webgpu'});
  await waitBackend(page, 'webgpu');
  await page.addScriptTag({type: 'module', url: origin + '/effects/model-viewer-effects.min.js'});
  await page.evaluate(() => customElements.whenDefined('effect-composer'));
  for (const effect of ['bloom', 'selective-bloom', 'outline', 'pixelate', 'smaa', 'ssao', 'color-grade', 'glitch']) {
    await t.test(effect, async () => {
      await page.evaluate(async name => {
        const viewer = document.querySelector('model-viewer');
        viewer.querySelector('effect-composer')?.remove();
        const composer = document.createElement('effect-composer');
        composer.append(document.createElement(name + '-effect'));
        viewer.append(composer);
        await viewer.updateComplete;
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        await new Promise(resolve => setTimeout(resolve, 350));
      }, effect);
      assert.equal(await page.locator('model-viewer').evaluate(viewer => viewer.activeRenderBackend), 'webgpu');
      assert.ok(await visiblePixels(page) > 100);
      assert.deepEqual(errors, []);
    });
  }
});

test('draws a panorama with a mask and a defocus map in the native backend', async t => {
  const {page, errors} = await start(t, {backend: 'webgpu'});
  await waitBackend(page, 'webgpu');
  const depth = Buffer.from(encode({
    width: 128, height: 64, channels: 1, data: new Uint8Array(128 * 64).fill(192)
  })).toString('base64');
  await page.evaluate(async depthImage => {
    const viewer = document.querySelector('model-viewer');
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 64;
    const context = canvas.getContext('2d');
    context.fillStyle = '#4368a3';
    context.fillRect(0, 0, 128, 64);
    context.fillStyle = '#fbab50';
    for (let x = 0; x < 128; x += 8) context.fillRect(x, 0, 4, 64);
    const skybox = canvas.toDataURL();
    context.fillStyle = '#c0c0c0';
    context.fillRect(0, 0, 128, 64);
    viewer.setAttribute('skybox-only', '');
    viewer.setAttribute('skybox-image', skybox);
    viewer.setAttribute('skybox-depth-image', 'data:image/png;base64,' + depthImage);
    viewer.setAttribute('skybox-dof-focus', '0.1');
    viewer.setPanoramaAreas([{
      name: 'sample',
      mask: {width: 8, height: 4, bounds: {x: 0, y: 0, width: 8, height: 4}, bits: new Uint8Array(4).fill(255)}
    }]);
    viewer.setPanoramaAreaHighlight('sample');
    await viewer.updateComplete;
  }, depth);
  await page.waitForFunction(() => {
    const viewer = document.querySelector('model-viewer');
    const scene = viewer[Object.getOwnPropertySymbols(viewer).find(symbol => symbol.description === 'scene')];
    return Boolean(scene.getSkyboxDofFrame());
  }, null, {timeout: 12000});
  await page.waitForTimeout(500);
  assert.equal(await page.locator('model-viewer').evaluate(viewer => viewer.activeRenderBackend), 'webgpu');
  assert.ok(await visiblePixels(page) > 100);
  assert.deepEqual(errors, []);
});
