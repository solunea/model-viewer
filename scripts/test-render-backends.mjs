import {strict as assert} from 'node:assert';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import {decode, encode} from 'fast-png';
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
<model-viewer src="/assets/cube.gltf" shadow-intensity="1" view-helper camera-controls interaction-prompt="none"></model-viewer>
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
    await page.addInitScript(failGPU => {
      globalThis.testGPUAdapterRequests = 0;
      globalThis.testGPUDeviceRequests = 0;
      const requestAdapter = GPU.prototype.requestAdapter;
      GPU.prototype.requestAdapter = function (...args) {
        ++globalThis.testGPUAdapterRequests;
        return requestAdapter.apply(this, args);
      };
      const original = GPUAdapter.prototype.requestDevice;
      GPUAdapter.prototype.requestDevice = async function (...args) {
        ++globalThis.testGPUDeviceRequests;
        if (failGPU) throw new Error('WebGPU device request failed in the test');
        const device = await original.apply(this, args);
        globalThis.testGPUDevice = device;
        return device;
      };
      globalThis.testViewerRenderer = viewer => {
        for (let owner = viewer; owner; owner = Object.getPrototypeOf(owner)) {
          const key = Object.getOwnPropertySymbols(owner).find(symbol => symbol.description === 'renderer');
          if (key) return viewer[key];
        }
        throw new Error('Missing model-viewer renderer');
      };
    }, settings.failGPU === true);
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

async function visiblePixels(page, selector = 'model-viewer', dataURL = false) {
  return page.locator(selector).first().evaluate(async (viewer, dataURL) => {
    const blob = dataURL ? await (await fetch(viewer.toDataURL())).blob() : await viewer.toBlob();
    const bitmap = await createImageBitmap(blob);
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext('2d');
    context.drawImage(bitmap, 0, 0);
    const data = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
    bitmap.close();
    return data.filter((value, index) => index % 4 === 3 && value > 0).length;
  }, dataURL);
}

test('opts into WebGPU, returns to WebGL in auto mode and recovers from GPU device loss', async t => {
  const {page, errors} = await start(t);
  await waitBackend(page, 'webgl');
  await page.locator('model-viewer').evaluate(viewer => viewer.setAttribute('render-backend', 'webgpu'));
  await waitBackend(page, 'webgpu');
  assert.deepEqual(await page.locator('model-viewer').evaluate(viewer => ({
    preference: viewer.renderBackend,
    attribute: viewer.getAttribute('render-backend'),
    deviceRequests: globalThis.testGPUDeviceRequests
  })), {preference: 'webgpu', attribute: 'webgpu', deviceRequests: 1});
  assert.equal(await page.locator('canvas#webgpu-canvas.show').count(), 1);
  assert.ok(await visiblePixels(page) > 100);
  assert.ok(await visiblePixels(page, 'model-viewer', true) > 100);
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
  assert.equal(await page.locator('canvas#webgpu-canvas').count(), 0);
  assert.equal(await page.locator('model-viewer').evaluate(viewer => viewer.renderBackend), 'auto');
  assert.ok(await visiblePixels(page) > 100);
  await page.locator('model-viewer').evaluate(viewer => viewer.renderBackend = 'webgpu');
  await waitBackend(page, 'webgpu');
  await page.evaluate(() => globalThis.testGPUDevice.destroy());
  await waitBackend(page, 'webgl');
  assert.equal(await page.locator('model-viewer').evaluate(viewer => viewer.renderBackend), 'webgpu');
  assert.equal(await page.locator('canvas#webgpu-canvas').count(), 0);
  assert.ok(await visiblePixels(page) > 100);
  assert.equal(await page.evaluate(() => document.querySelector('model-viewer')
      .getCameraOrbit().toString()), orbit);
  assert.deepEqual(errors, []);
});

for (const backend of [undefined, 'auto', 'webgl']) {
  test(`renders in WebGL with ${backend ?? 'no'} backend attribute without probing WebGPU`, async t => {
    const {page, errors} = await start(t, {backend});
    await waitBackend(page, 'webgl');
    assert.ok(await visiblePixels(page) > 100);
    assert.deepEqual(await page.locator('model-viewer').evaluate(viewer => ({
      preference: viewer.renderBackend,
      attribute: viewer.getAttribute('render-backend'),
      adapterRequests: globalThis.testGPUAdapterRequests,
      deviceRequests: globalThis.testGPUDeviceRequests
    })), {preference: backend ?? 'auto', attribute: backend ?? null, adapterRequests: 0, deviceRequests: 0});
    assert.deepEqual(errors, []);
  });
}

test('presents multiple WebGPU viewers with independently sized canvases', async t => {
  const {page, errors} = await start(t, {backend: 'webgpu'});
  await waitBackend(page, 'webgpu');
  await page.evaluate(() => {
    customElements.get('model-viewer').minimumRenderScale = 1;
    const first = document.querySelector('model-viewer');
    const second = first.cloneNode();
    first.id = 'first';
    second.id = 'second';
    second.style.width = '200px';
    second.style.height = '160px';
    first.after(second);
  });
  await page.waitForFunction(() => document.querySelector('#second').activeRenderBackend === 'webgpu');
  const screenshot = decode(await page.locator('#first').screenshot());
  const center = (Math.floor(screenshot.height / 2) * screenshot.width +
      Math.floor(screenshot.width / 2)) * screenshot.channels;
  assert.ok(screenshot.data.subarray(center, center + 3).some(value => value < 240));

  const measure = () => page.evaluate(async () => {
    return Promise.all([...document.querySelectorAll('model-viewer')].map(async viewer => {
      const [canvas] = viewer.shadowRoot.querySelectorAll('canvas.show');
      const bitmap = await createImageBitmap(await viewer.toBlob());
      const measurement = {
        canvas: [canvas.width, canvas.height],
        capture: [bitmap.width, bitmap.height],
        sharedDevice: canvas.getContext('webgpu').getConfiguration().device === globalThis.testGPUDevice
      };
      bitmap.close();
      return measurement;
    }));
  });
  const expected = (width, height) => ({
    canvas: [width, height], capture: [width, height], sharedDevice: true
  });
  assert.deepEqual(await measure(), [expected(300, 240), expected(200, 160)]);
  assert.ok(await visiblePixels(page, '#first') > 1000);
  assert.ok(await visiblePixels(page, '#second') > 1000);

  await page.locator('#second').evaluate(viewer => viewer.style.width = '240px');
  await page.waitForFunction(() => document.querySelector('#second')
      .shadowRoot.querySelector('#webgpu-canvas').width === 240);
  assert.deepEqual(await measure(), [expected(300, 240), expected(240, 160)]);

  await page.locator('#first').evaluate(viewer => viewer.renderBackend = 'webgl');
  await waitBackend(page, 'webgl');
  assert.equal(await page.locator('#second').evaluate(viewer => viewer.activeRenderBackend), 'webgpu');
  assert.ok(await visiblePixels(page, '#second') > 1000);

  assert.equal(await page.locator('#second').evaluate(viewer => {
    const canvas = viewer.shadowRoot.querySelector('#webgpu-canvas');
    viewer.remove();
    return canvas.parentNode == null;
  }), true);
  assert.ok(await visiblePixels(page, '#first') > 1000);
  assert.deepEqual(errors, []);
});

test('presents a lone WebGL canvas directly beside native WebGPU viewers', async t => {
  const {page, errors} = await start(t, {backend: 'webgl'});
  await waitBackend(page, 'webgl');
  await page.evaluate(() => {
    customElements.get('model-viewer').minimumRenderScale = 1;
    const first = document.querySelector('model-viewer');
    first.id = 'gl-view';
    first.viewHelper = false;
    const second = first.cloneNode();
    second.id = 'gpu-view';
    second.style.width = '200px';
    second.style.height = '160px';
    second.setAttribute('render-backend', 'webgpu');
    first.after(second);
    const scene = viewer => viewer[
      Object.getOwnPropertySymbols(viewer).find(key => key.description === 'scene')];
    const manager = globalThis.testViewerRenderer(first);
    window.mixedViews = {first, second, manager, firstScene: scene(first), secondScene: scene(second)};
  });
  const waitForPresentation = (backend, direct) => page.waitForFunction(({backend, direct}) => {
    const {first, second, manager, firstScene} = mixedViews;
    return first.activeRenderBackend === 'webgl' && second.activeRenderBackend === backend &&
        (manager.displayCanvas(firstScene) === manager.canvas3D) === direct;
  }, {backend, direct});
  await waitForPresentation('webgpu', true);
  const initial = await page.evaluate(() => {
    const {first, manager, firstScene, secondScene} = mixedViews;
    const scale = secondScene.scaleStep;
    const size = [manager.canvas3D.style.width, manager.canvas3D.style.height];
    secondScene.scaleStep = 1;
    manager.rescaleCanvas(secondScene);
    const sizeAfterGPUScale = [manager.canvas3D.style.width, manager.canvas3D.style.height];
    secondScene.scaleStep = scale;
    manager.rescaleCanvas(secondScene);

    const copy = manager.copyPixels;
    let copies = 0;
    manager.copyPixels = function (...args) {
      copies++;
      return copy.apply(this, args);
    };
    try {
      firstScene.queueRender();
      manager.render(performance.now());
    } finally {
      manager.copyPixels = copy;
    }
    return {copies, size, sizeAfterGPUScale, url: first.toDataURL()};
  });
  assert.equal(initial.copies, 0);
  assert.deepEqual(initial.size, initial.sizeAfterGPUScale);

  await page.locator('#gpu-view').evaluate(viewer => viewer.renderBackend = 'webgl');
  await waitForPresentation('webgl', false);
  const copied = await page.locator('#gl-view').evaluate(viewer => viewer.toDataURL());
  const pixels = url => decode(Buffer.from(url.split(',')[1], 'base64'));
  const first = pixels(initial.url);
  const second = pixels(copied);
  assert.deepEqual([first.width, first.height, first.channels],
    [second.width, second.height, second.channels]);
  let total = 0;
  for (let i = 0; i < first.data.length; ++i) {
    total += Math.abs(first.data[i] - second.data[i]);
  }
  assert.ok(total / first.data.length < .2, 'Direct and copied captures differ');
  assert.ok(await visiblePixels(page, '#gl-view') > 1000);

  // A static WebGL view must regain its shared canvas after its neighbour switches.
  await page.locator('#gpu-view').evaluate(viewer => viewer.renderBackend = 'webgpu');
  await waitForPresentation('webgpu', true);
  await page.evaluate(() => globalThis.testGPUDevice.destroy());
  await waitForPresentation('webgl', false);
  assert.ok(await visiblePixels(page, '#gl-view') > 1000);
  assert.ok(await visiblePixels(page, '#gpu-view') > 1000);
  assert.deepEqual(errors, []);
});

test('preloads panorama textures on the destination renderer', async t => {
  const {page, errors} = await start(t, {backend: 'webgl'});
  await waitBackend(page, 'webgl');
  let requests = 0;
  const panorama = origin + '/preload-panorama.png';
  const body = Buffer.from(encode({
    width: 4, height: 2, channels: 4, data: new Uint8Array(32).fill(255)
  }));
  await page.route(panorama, route => {
    ++requests;
    return route.fulfill({body, contentType: 'image/png'});
  });

  const result = await page.evaluate(async url => {
    const viewer = document.querySelector('model-viewer');
    const manager = globalThis.testViewerRenderer(viewer);
    const gpu = await manager.probeWebGPU();
    const uploads = {webgl: [], webgpu: []};
    let gpuWaits = 0;
    const {queue} = gpu.device;
    const onSubmittedWorkDone = queue.onSubmittedWorkDone.bind(queue);
    queue.onSubmittedWorkDone = () => {
      ++gpuWaits;
      return onSubmittedWorkDone();
    };
    for (const [backend, renderer] of [
      ['webgl', manager.threeRenderer], ['webgpu', gpu.renderer]
    ]) {
      const initTexture = renderer.initTexture;
      renderer.initTexture = function (texture) {
        uploads[backend].push(texture);
        return initTexture.call(this, texture);
      };
    }
    const warmWebGPU = () => viewer.preloadSkybox(url, null, {renderBackend: 'webgpu'});
    await Promise.all([warmWebGPU(), warmWebGPU()]);
    await warmWebGPU();
    const uploadedToWebGL = uploads.webgl.length;
    await viewer.preloadSkybox(url, null, {renderBackend: 'auto'});
    return {
      uploadedToWebGL,
      gpuWaits,
      webglUploads: uploads.webgl.length,
      webgpuUploads: uploads.webgpu.length,
      reusedTexture: uploads.webgl[0] === uploads.webgpu[0],
      activeRenderBackend: viewer.activeRenderBackend
    };
  }, panorama);
  assert.equal(requests, 1);
  assert.deepEqual(result, {
    uploadedToWebGL: 0, gpuWaits: 1, webglUploads: 1, webgpuUploads: 1,
    reusedTexture: true, activeRenderBackend: 'webgl'
  });
  assert.deepEqual(errors, []);
});

test('adapts WebGPU render scale and updates contact shadows at 30 Hz', async t => {
  const {page, errors} = await start(t, {backend: 'webgpu'});
  await waitBackend(page, 'webgpu');
  await page.evaluate(async () => {
    const viewer = document.querySelector('model-viewer');
    const loaded = new Promise(resolve => viewer.addEventListener('load', resolve, {once: true}));
    viewer.viewHelper = false;
    viewer.src = '/assets/RobotExpressive.glb';
    await loaded;
    viewer.animationName = 'Walking';
    await viewer.updateComplete;
    viewer.play();
  });
  await page.waitForFunction(() => document.querySelector('model-viewer').modelIsVisible);

  const result = await page.evaluate(() => {
    const viewer = document.querySelector('model-viewer');
    const elementClass = customElements.get('model-viewer');
    const renderer = globalThis.testViewerRenderer(viewer);
    renderer.threeRenderer.setAnimationLoop(null);
    const {shadows} = renderer.gpu;
    const setDepth = shadows.setDepth;
    let shadowUpdates = 0;
    shadows.setDepth = (...args) => {
      ++shadowUpdates;
      setDepth(...args);
    };
    let time = performance.now() + 50;
    const frame = (delta = 1000 / 60) => {
      shadowUpdates = 0;
      renderer.render(time += delta);
      return shadowUpdates;
    };
    const renderedWidth = () => viewer.shadowRoot.querySelector('#webgpu-canvas').width;

    elementClass.minimumRenderScale = 1;
    viewer.currentTime = 0.5;
    const updates = Array.from({length: 6}, () => frame());
    viewer.pause();
    const pauseUpdates = frame();
    const fullWidth = renderedWidth();

    viewer.play();
    frame();
    elementClass.minimumRenderScale = 0.5;
    for (let i = 0; i < 3; ++i) frame(25);
    const reducedWidth = renderedWidth();
    viewer.pause();
    // The last pose invalidates the shadow and the next idle frame restores
    // full resolution.
    frame();
    frame();
    return {updates, pauseUpdates, fullWidth, reducedWidth, idleWidth: renderedWidth()};
  });
  assert.deepEqual(result, {
    updates: [1, 0, 1, 0, 1, 0],
    pauseUpdates: 1,
    fullWidth: 300,
    reducedWidth: 237,
    idleWidth: 300
  });
  assert.ok(await visiblePixels(page) > 1000);
  assert.deepEqual(errors, []);
});

test('loads the same model on WebGL when WebGPU is unavailable', async t => {
  const {page, errors} = await start(t, {backend: 'webgpu', disableGPU: true});
  await waitBackend(page, 'webgl');
  assert.equal(await page.locator('model-viewer').evaluate(viewer => viewer.renderBackend), 'webgpu');
  assert.ok(await visiblePixels(page) > 100);
  assert.deepEqual(errors, []);
});

test('falls back to WebGL when an explicit WebGPU device request fails', async t => {
  const {page, errors} = await start(t, {backend: 'webgpu', failGPU: true});
  await waitBackend(page, 'webgl');
  assert.equal(await page.evaluate(() => globalThis.testGPUDeviceRequests), 1);
  assert.equal(await page.locator('model-viewer').evaluate(viewer => viewer.renderBackend), 'webgpu');
  assert.ok(await visiblePixels(page) > 100);
  assert.deepEqual(errors, []);
});

test('keeps a skinned model renderable across WebGPU and WebGL', async t => {
  const {page, errors} = await start(t, {backend: 'webgl'});
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

async function loadPanoramaWithDepth(page, backend) {
  const depth = Buffer.from(encode({
    width: 128, height: 64, channels: 1, data: new Uint8Array(128 * 64).fill(192)
  })).toString('base64');
  await page.evaluate(async ({depthImage, backend}) => {
    const viewer = document.createElement('model-viewer');
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
    viewer.setAttribute('loading', 'eager');
    viewer.setAttribute('render-backend', backend);
    viewer.setAttribute('skybox-image', skybox);
    viewer.setAttribute('skybox-depth-image', 'data:image/png;base64,' + depthImage);
    viewer.setAttribute('skybox-dof-focus', '0.1');
    viewer.setAttribute('skybox-dof-max-blur', '24');
    viewer.setPanoramaAreas([{
      name: 'sample',
      mask: {width: 8, height: 4, bounds: {x: 0, y: 0, width: 8, height: 4}, bits: new Uint8Array(4).fill(255)}
    }]);
    viewer.setPanoramaAreaHighlight('sample');
    document.querySelector('model-viewer').replaceWith(viewer);
    await viewer.updateComplete;
  }, {depthImage: depth, backend});
  await waitBackend(page, backend);
  await page.waitForFunction(() => {
    const viewer = document.querySelector('model-viewer');
    const scene = viewer[Object.getOwnPropertySymbols(viewer).find(symbol => symbol.description === 'scene')];
    return Boolean(scene.getSkyboxDofFrame());
  }, null, {timeout: 12000});
  await page.waitForTimeout(500);
}

test('draws a panorama with a mask and a defocus map in the native backend', async t => {
  const {page, errors} = await start(t, {backend: 'webgpu'});
  await loadPanoramaWithDepth(page, 'webgpu');
  const {passes, highlightChanged, matchesRefresh, sampleCounts, motionPasses, shine} = await page.evaluate(async () => {
    const viewer = document.querySelector('model-viewer');
    const renderer = globalThis.testViewerRenderer(viewer);
    const scene = viewer[Object.getOwnPropertySymbols(viewer).find(symbol => symbol.description === 'scene')];
    renderer.threeRenderer.setAnimationLoop(null);
    customElements.get('model-viewer').minimumRenderScale = 1;
    // Include the hotspot sphere in the camera's clip range.
    scene.camera.far = 20;
    scene.camera.updateProjectionMatrix();
    viewer.toDataURL();
    viewer.toDataURL();
    const dof = renderer.gpu.skyboxDof;
    const {queue} = globalThis.testGPUDevice;
    const submit = queue.submit.bind(queue);
    let submissions = 0;
    queue.submit = commands => {
      ++submissions;
      submit(commands);
    };
    const capture = () => {
      const before = submissions;
      return {url: viewer.toDataURL(), passes: submissions - before};
    };
    dof.blurInputs = null;
    const initial = capture();
    viewer.setPanoramaAreaHighlight('sample', '#ff6e20');
    await new Promise(resolve => requestAnimationFrame(resolve));
    const cached = capture();
    dof.blurInputs = null;
    const refreshed = capture();
    viewer.skyboxDofStrength = 2;
    await viewer.updateComplete;
    const changedStrength = capture();
    scene.getSkyboxDofFrame().depth.needsUpdate = true;
    const updatedDepth = capture();
    const yaw = scene.backgroundRotation.y + Math.PI / 3;
    let time = performance.now();
    const draw = angle => {
      scene.backgroundRotation.y = angle;
      scene.queueRender();
      const before = submissions;
      renderer.render(time += 1000 / 60);
      return submissions - before;
    };
    const moved = draw(yaw);
    const moving = dof.sampleCount.value;
    const reused = draw(yaw + .0005);
    const still = draw(yaw + .0005);
    const areas = scene.panoramaAreas;
    viewer.setPanoramaAreaHighlight(null);
    viewer.setPanoramaAreaShine(true);
    areas.tick(0);
    const idle = capture();
    const {x, y, width, height} = viewer.getBoundingClientRect();
    const idleHit = viewer.panoramaAreaFromPoint(x + width / 2, y + height / 2)?.name;
    const restoredVisibility = areas.visible && areas.children[0].visible;
    areas.tick(950);
    const sweeping = capture();
    areas.tick(1400);
    viewer.setPanoramaAreaFocus('sample');
    capture();
    const focused = capture();
    viewer.setPanoramaAreaFocus(null);
    viewer.setPanoramaAreaShine(false);
    const withoutOverlay = capture();
    queue.submit = submit;
    return {
      passes: [initial, cached, refreshed, changedStrength, updatedDepth].map(frame => frame.passes),
      highlightChanged: initial.url !== cached.url,
      matchesRefresh: cached.url === refreshed.url,
      sampleCounts: [moving, dof.sampleCount.value],
      motionPasses: [moved, reused, still],
      shine: {
        passes: [idle, sweeping, focused, withoutOverlay].map(frame => frame.passes),
        idleHit, restoredVisibility,
        idleMatchesBackground: idle.url === withoutOverlay.url,
        sweepVisible: sweeping.url !== idle.url
      }
    };
  });
  assert.deepEqual(passes, [4, 2, 4, 4, 4]);
  assert.ok(highlightChanged);
  assert.ok(matchesRefresh);
  assert.deepEqual(sampleCounts, [8, 24]);
  assert.deepEqual(motionPasses, [4, 2, 4]);
  assert.deepEqual(shine, {
    passes: [1, 2, 2, 1],
    idleHit: 'sample',
    restoredVisibility: true,
    idleMatchesBackground: true,
    sweepVisible: true
  });
  assert.equal(await page.locator('model-viewer').evaluate(viewer => viewer.activeRenderBackend), 'webgpu');
  assert.ok(await visiblePixels(page) > 100);
  assert.deepEqual(errors, []);
});

test('reuses WebGL panorama blur while recoloring a hotspot', async t => {
  const {page, errors} = await start(t, {backend: 'webgl'});
  await loadPanoramaWithDepth(page, 'webgl');
  const {passes, highlightChanged, matchesRefresh} = await page.evaluate(async () => {
    const viewer = document.querySelector('model-viewer');
    const manager = globalThis.testViewerRenderer(viewer);
    const renderer = manager.threeRenderer;
    const scene = viewer[Object.getOwnPropertySymbols(viewer).find(symbol => symbol.description === 'scene')];
    renderer.setAnimationLoop(null);
    customElements.get('model-viewer').minimumRenderScale = 1;
    scene.camera.far = 20;
    scene.camera.updateProjectionMatrix();

    const original = renderer.render;
    let draws = 0;
    renderer.render = function (...args) {
      ++draws;
      return original.apply(this, args);
    };
    let time = performance.now();
    const capture = () => {
      const before = draws;
      scene.queueRender();
      manager.render(time += 1000 / 60);
      return {url: viewer.toDataURL(), draws: draws - before};
    };
    try {
      const initial = capture();
      viewer.setPanoramaAreaHighlight('sample', '#ff6e20');
      const cached = capture();
      const yaw = scene.backgroundRotation.y;
      scene.backgroundRotation.y = yaw + Math.PI / 3;
      const moved = capture();
      scene.backgroundRotation.y = yaw;
      const refreshed = capture();
      viewer.skyboxDofFocus = .4;
      await viewer.updateComplete;
      const refocused = capture();
      scene.getSkyboxDofFrame().depth.needsUpdate = true;
      const updatedDepth = capture();
      const unchanged = capture();
      return {
        passes: [initial, cached, moved, refreshed, refocused, updatedDepth, unchanged].map(frame => frame.draws),
        highlightChanged: initial.url !== cached.url,
        matchesRefresh: cached.url === refreshed.url
      };
    } finally {
      renderer.render = original;
    }
  });
  assert.deepEqual(passes, [3, 1, 3, 3, 3, 3, 1]);
  assert.ok(highlightChanged);
  assert.ok(matchesRefresh);
  assert.deepEqual(errors, []);
});

test('draws a WebGPU skybox directly with the same colors and alpha as scene compositing', async t => {
  const {page, errors} = await start(t);
  const depth = Buffer.from(encode({
    width: 128, height: 64, channels: 1, data: new Uint8Array(128 * 64).fill(192)
  })).toString('base64');
  await page.evaluate(depthImage => {
    const panorama = document.createElement('canvas');
    panorama.width = 128;
    panorama.height = 64;
    const context = panorama.getContext('2d');
    for (let x = 0; x < panorama.width; x += 8) {
      context.fillStyle = x % 16 === 0 ? 'rgba(90, 150, 200, .7)' : 'rgba(170, 100, 50, .5)';
      context.fillRect(x, 0, 8, panorama.height);
    }
    const viewer = document.createElement('model-viewer');
    viewer.setAttribute('skybox-only', '');
    viewer.setAttribute('skybox-image', panorama.toDataURL());
    viewer.setAttribute('skybox-depth-image', 'data:image/png;base64,' + depthImage);
    viewer.setAttribute('skybox-dof-max-blur', '24');
    viewer.setAttribute('skybox-dof-focus', '0.2');
    viewer.setAttribute('skybox-dof-focus-mode', 'manual');
    viewer.setAttribute('loading', 'eager');
    viewer.setAttribute('render-backend', 'webgpu');
    document.querySelector('model-viewer').replaceWith(viewer);
  }, depth);
  await waitBackend(page, 'webgpu');
  await page.waitForFunction(() => {
    const viewer = document.querySelector('model-viewer');
    const scene = viewer[Object.getOwnPropertySymbols(viewer).find(symbol => symbol.description === 'scene')];
    return scene.getSkyboxDofFrame()?.maxBlur === 24;
  });

  const {direct, composited, anchorPasses, reused, refreshed, movedPasses, stillPasses,
    stillSamples, plainDirect, plainComposited, plainWithRaycastMesh} =
      await page.evaluate(async () => {
    const {Mesh, MeshBasicMaterial, PlaneGeometry} = await import('three');
    const viewer = document.querySelector('model-viewer');
    const scene = viewer[Object.getOwnPropertySymbols(viewer).find(symbol => symbol.description === 'scene')];
    const manager = globalThis.testViewerRenderer(viewer);
    manager.threeRenderer.setAnimationLoop(null);
    const {queue} = globalThis.testGPUDevice;
    const submit = queue.submit.bind(queue);
    let submissions = 0;
    queue.submit = commands => {
      ++submissions;
      submit(commands);
    };
    const capture = () => {
      // Let Three set the coordinate system and the blur return to full quality.
      viewer.toDataURL();
      viewer.toDataURL();
      viewer.toDataURL();
      const before = submissions;
      return {url: viewer.toDataURL(), passes: submissions - before};
    };
    const direct = capture();
    const mesh = new Mesh(new PlaneGeometry(), new MeshBasicMaterial());
    mesh.position.set(1000, 1000, 1000);
    scene.add(mesh);
    const composited = capture();
    mesh.removeFromParent();

    const dof = manager.gpu.skyboxDof;
    const canvas = viewer.shadowRoot.querySelector('#webgpu-canvas');
    const startYaw = scene.backgroundRotation.y;
    let time = performance.now();
    const draw = yaw => {
      scene.backgroundRotation.y = yaw;
      scene.queueRender();
      const before = submissions;
      manager.render(time += 1000 / 60);
      return {passes: submissions - before, url: canvas.toDataURL()};
    };
    const anchorPasses = draw(startYaw + .1).passes;
    const reused = draw(startYaw + .1005);
    draw(startYaw + .1);
    dof.blurInputs = null;
    const refreshed = draw(startYaw + .1005);
    const movedPasses = draw(startYaw + .4).passes;
    const stillPasses = draw(startYaw + .4).passes;
    const stillSamples = dof.sampleCount.value;
    viewer.skyboxDofMaxBlur = 0;
    await viewer.updateComplete;
    const plainDirect = capture();
    mesh.material.visible = false;
    scene.add(mesh);
    const plainWithRaycastMesh = capture();
    mesh.material.visible = true;
    const plainComposited = capture();
    mesh.removeFromParent();
    mesh.geometry.dispose();
    mesh.material.dispose();
    queue.submit = submit;
    return {direct, composited, anchorPasses, reused, refreshed, movedPasses, stillPasses,
      stillSamples, plainDirect, plainComposited, plainWithRaycastMesh};
  });
  assert.equal(direct.passes, 1);
  assert.equal(composited.passes, 2);
  assert.equal(plainDirect.passes, 1);
  assert.equal(plainWithRaycastMesh.passes, 1);
  assert.equal(plainComposited.passes, 2);
  assert.deepEqual([anchorPasses, reused.passes, refreshed.passes, movedPasses, stillPasses],
    [3, 1, 3, 3, 3]);
  assert.equal(stillSamples, 24);
  const image = ({url}) => decode(Buffer.from(url.split(',')[1], 'base64'));
  const difference = (a, b) => {
    assert.deepEqual([a.width, a.height, a.channels], [b.width, b.height, 4]);
    let total = 0;
    let max = 0;
    for (let i = 0; i < a.data.length; i++) {
      const delta = Math.abs(a.data[i] - b.data[i]);
      total += delta;
      max = Math.max(max, delta);
    }
    return {mean: total / a.data.length, max};
  };
  const fast = image(direct);
  assert.ok(fast.data.some((value, index) => index % 4 === 3 && value > 0 && value < 255));
  const colorError = difference(fast, image(composited));
  assert.ok(colorError.mean < .5 && colorError.max <= 10, JSON.stringify(colorError));
  const motionError = difference(image(reused), image(refreshed));
  assert.ok(motionError.mean < .5, JSON.stringify(motionError));
  const plain = image(plainDirect);
  assert.ok(plain.data.some((value, index) => index % 4 === 3 && value > 0));
  const plainColorError = difference(plain, image(plainComposited));
  assert.ok(plainColorError.mean < .5 && plainColorError.max <= 10, JSON.stringify(plainColorError));
  assert.deepEqual(errors, []);
});

test('composes a WebGPU panorama transition in one pass with the original colors', async t => {
  const {page, errors} = await start(t);
  await page.evaluate(() => {
    const viewer = document.createElement('model-viewer');
    viewer.setAttribute('skybox-only', '');
    viewer.setAttribute('loading', 'eager');
    viewer.setAttribute('render-backend', 'webgpu');
    const canvas = new OffscreenCanvas(128, 64);
    const context = canvas.getContext('2d');
    context.fillStyle = '#2458bc';
    context.fillRect(0, 0, 128, 64);
    return canvas.convertToBlob().then(blob => {
      viewer.skyboxImage = URL.createObjectURL(blob);
      document.querySelector('model-viewer').replaceWith(viewer);
    });
  });
  await waitBackend(page, 'webgpu');
  const captures = await page.evaluate(async () => {
    const {EquirectangularReflectionMapping, Mesh, MeshBasicMaterial, PlaneGeometry,
      SRGBColorSpace, TextureLoader} = await import('three');
    const viewer = document.querySelector('model-viewer');
    const scene = viewer[Object.getOwnPropertySymbols(viewer).find(symbol => symbol.description === 'scene')];
    const manager = globalThis.testViewerRenderer(viewer);
    manager.threeRenderer.setAnimationLoop(null);
    scene.camera.far = 20;
    scene.camera.updateProjectionMatrix();
    const overlay = new Mesh(new PlaneGeometry(), new MeshBasicMaterial());
    overlay.position.set(1000, 1000, 1000);
    const {queue} = globalThis.testGPUDevice;
    const submit = queue.submit.bind(queue);
    let submissions = 0;
    queue.submit = buffers => { submissions++; submit(buffers); };
    const capture = () => {
      viewer.toDataURL();
      const before = submissions;
      const url = viewer.toDataURL();
      return {url, passes: submissions - before};
    };
    const captures = [];
    for (const tint of ['rgba(240, 25, 15, .4)', 'rgba(10, 240, 70, .8)']) {
      const canvas = new OffscreenCanvas(128, 64);
      const context = canvas.getContext('2d');
      context.fillStyle = tint;
      context.fillRect(0, 0, 128, 64);
      context.fillStyle = 'rgba(180, 160, 90, .7)';
      context.fillRect(8, 12, 48, 18);
      const url = URL.createObjectURL(await canvas.convertToBlob());
      const next = await new TextureLoader().loadAsync(url);
      URL.revokeObjectURL(url);
      next.mapping = EquirectangularReflectionMapping;
      next.colorSpace = SRGBColorSpace;
      scene.setSkyboxInterpolationDecay(350);
      scene.setEnvironmentAndSkybox(null, next);
      scene.skyboxTransitionMesh.material.opacity = .5;
      const direct = capture();
      scene.add(overlay);
      const reference = capture();
      overlay.removeFromParent();
      captures.push({direct, reference});
      scene.setSkyboxInterpolationDecay(0);
    }
    queue.submit = submit;
    overlay.geometry.dispose();
    overlay.material.dispose();
    return captures;
  });
  assert.notEqual(captures[0].direct.url, captures[1].direct.url);
  for (const {direct, reference} of captures) {
    assert.equal(direct.passes, 1);
    assert.equal(reference.passes, 2);
    const read = frame => decode(Buffer.from(frame.url.split(',')[1], 'base64')).data;
    const current = read(direct);
    const expected = read(reference);
    assert.equal(current.length, expected.length);
    let max = 0;
    for (let i = 0; i < current.length; ++i) {
      max = Math.max(max, Math.abs(current[i] - expected[i]));
    }
    assert.ok(max <= 2, 'Maximum transition color difference was ' + max);
  }
  assert.deepEqual(errors, []);
});
