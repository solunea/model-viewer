import {strict as assert} from 'node:assert';
import {execFileSync} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {decode, encode} from 'fast-png';
import {chromium} from 'playwright';

const root = new URL('../', import.meta.url);
const origin = 'http://localhost:8766';
const viewerAssets = 'packages/model-viewer/dist/';
const models = new URL('packages/shared-assets/models/', root);
const three = new URL('node_modules/three/build/', root);
const files = ['model-viewer-module.min.js', 'model-viewer-webgpu-module.min.js'];
const layout = {
  model: {count: 1, width: 1280, height: 720},
  gallery: {count: 4, width: 640, height: 360},
  panorama: {count: 1, width: 1280, height: 720, dof: true}
};
const cases = (process.env.MODEL_VIEWER_BENCH_CASES ?? Object.keys(layout).join(',')).split(',');
const backends = (process.env.MODEL_VIEWER_BENCH_BACKENDS ?? 'webgl,webgpu').split(',');
const rounds = Number(process.env.MODEL_VIEWER_BENCH_ROUNDS ?? 3);
const seconds = Number(process.env.MODEL_VIEWER_BENCH_SECONDS ?? 2);
const baseline = process.argv.find(argument => argument.startsWith('--baseline='))?.slice(11);
assert.ok(cases.every(name => Object.hasOwn(layout, name)), 'Unknown benchmark scene');
assert.ok(backends.every(name => ['webgl', 'webgpu'].includes(name)), 'Unknown renderer');
assert.ok(Number.isInteger(rounds) && rounds > 0 && Number.isFinite(seconds) && seconds > 0);

const current = new Map(await Promise.all(files.map(async file =>
  [file, await readFile(new URL(viewerAssets + file, root))])));
const variants = new Map([['current', current]]);
if (baseline) {
  const revision = execFileSync('git', ['rev-parse', '--verify', '--end-of-options', baseline + '^{commit}'],
    {cwd: fileURLToPath(root), encoding: 'utf8'}).trim();
  variants.set('baseline', new Map(files.map(file => [file,
    execFileSync('git', ['show', revision + ':' + viewerAssets + file], {cwd: fileURLToPath(root)})])));
}

function createPanorama() {
  const width = 1024;
  const height = 512;
  const color = new Uint8Array(width * height * 3);
  const depth = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      const stripes = ((x >> 2) ^ (y >> 2)) & 1;
      color.set([stripes ? 240 : 56, 46 + (x >> 3), 24 + (y >> 2)], index * 3);
      depth[index] = ((x >> 5) + (y >> 5)) % 2 === 0 ? 32 : 224;
    }
  }
  return {
    color: Buffer.from(encode({data: color, width, height, channels: 3})),
    depth: Buffer.from(encode({data: depth, width, height, channels: 1}))
  };
}

const panorama = createPanorama();

function html({count, width, height, dof}, backend) {
  const attributes = dof ?
    'skybox-only skybox-image="/color.png" skybox-depth-image="/depth.png" skybox-dof-focus="0.5" skybox-dof-max-blur="24"' :
    'src="/models/RobotExpressive.glb" shadow-intensity="0.7" animation-name="Walking" autoplay';
  const viewers = Array.from({length: count}, (_, index) =>
    '<model-viewer loading="eager" auto-rotate auto-rotate-delay="0" rotation-per-second="20deg" ' +
    'camera-orbit="' + (index * 17) + 'deg 75deg auto" render-backend="' +
    backend + '" ' + attributes + '></model-viewer>').join('');
  return '<!doctype html><script type="importmap">' +
    '{"imports":{"three":"/three/three.module.js","three/webgpu":"/three/three.webgpu.js","three/tsl":"/three/three.tsl.js"}}' +
    '</script><script type="module">import "/viewer/model-viewer-module.min.js";' +
    'customElements.get("model-viewer").minimumRenderScale = 1;</script>' +
    '<style>body{margin:0;display:grid;grid-template-columns:repeat(' + (count > 1 ? 2 : 1) + ',' + width +
    'px)}model-viewer{display:block;width:' + width + 'px;height:' + height + 'px}</style>' + viewers;
}

async function benchmark(browser, name, backend, assets, capture) {
  const page = await browser.newPage({viewport: {width: 1280, height: 720}, deviceScaleFactor: 1});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text());
  });
  try {
    await page.route(origin + '/**', async route => {
      const {pathname} = new URL(route.request().url());
      if (pathname === '/') return route.fulfill({contentType: 'text/html', body: html(layout[name], backend)});
      if (pathname === '/color.png' || pathname === '/depth.png') {
        return route.fulfill({contentType: 'image/png', body: panorama[pathname.slice(1, -4)]});
      }
      const [, group, file] = pathname.split('/');
      if (group === 'viewer' && assets.has(file)) {
        return route.fulfill({contentType: 'text/javascript', body: assets.get(file)});
      }
      if (group === 'three' && /^three\.[a-z.]+\.js$/.test(file)) {
        return route.fulfill({contentType: 'text/javascript', body: await readFile(new URL(file, three))});
      }
      if (group === 'models' && file === 'RobotExpressive.glb') {
        return route.fulfill({contentType: 'model/gltf-binary', body: await readFile(new URL(file, models))});
      }
      return route.fulfill({status: 404});
    });

    await page.goto(origin);
    await page.waitForFunction(({backend, dof}) => Array.from(document.querySelectorAll('model-viewer')).every(viewer => {
      const scene = viewer[Object.getOwnPropertySymbols(viewer).find(key => key.description === 'scene')];
      return viewer.loaded && viewer.activeRenderBackend === backend && (!dof || scene.getSkyboxDofFrame());
    }), {backend, dof: layout[name].dof}, {timeout: 25000});
    const metrics = await page.evaluate(async ({seconds, dof}) => {
      const viewers = [...document.querySelectorAll('model-viewer')];
      const find = (object, description) => {
        for (let level = object; level; level = Object.getPrototypeOf(level)) {
          const key = Object.getOwnPropertySymbols(level).find(symbol => symbol.description === description);
          if (key) return object[key];
        }
        throw new Error('Missing model-viewer ' + description);
      };
      const scenes = viewers.map(viewer => find(viewer, 'scene'));
      const manager = find(viewers[0], 'renderer');
      const gl = manager.threeRenderer.getContext();
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      const gpu = manager.gpu?.renderer;
      const device = gpu?.backend.device;
      if (dof && !(device ? manager.gpu.hasSkyboxDof : manager.skyboxDof)) {
        throw new Error('The panorama depth-of-field pass is inactive');
      }
      const adapter = device?.adapterInfo;
      let submissions = 0;
      if (device) {
        const submit = device.queue.submit.bind(device.queue);
        device.queue.submit = buffers => {
          submissions++;
          return submit(buffers);
        };
      }
      const render = manager.render.bind(manager);
      const copyPixels = manager.copyPixels.bind(manager);
      let copies = 0;
      let copyCPU = 0;
      manager.copyPixels = (...args) => {
        copies++;
        const start = performance.now();
        const result = copyPixels(...args);
        copyCPU += performance.now() - start;
        return result;
      };
      const frames = [];
      const renderCPU = [];
      const copyCPUPerFrame = [];
      const intervals = [];
      let lastFrame = 0;
      let started = false;
      const count = () => scenes.reduce((sum, scene) => sum + scene.renderCount, 0);
      manager.render = (...args) => {
        const before = count();
        for (const scene of scenes) scene.queueRender();
        copyCPU = 0;
        const start = performance.now();
        const result = render(...args);
        const drawn = count() - before;
        if (started && drawn > 0) {
          frames.push(drawn);
          renderCPU.push(performance.now() - start);
          copyCPUPerFrame.push(copyCPU);
          if (lastFrame) intervals.push(start - lastFrame);
          lastFrame = start;
        }
        return result;
      };
      const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
      const drain = () => device ? device.queue.onSubmittedWorkDone() : gl.finish();
      await delay(1000);
      await drain();
      copies = submissions = 0;
      started = true;
      const start = performance.now();
      await delay(seconds * 1000);
      const elapsed = performance.now() - start;
      const totalCopies = copies;
      const totalSubmissions = submissions;
      started = false;
      await drain();
      const queueDrainMs = performance.now() - start - elapsed;
      const result = {
        frames: frames.length,
        framesPerSecond: frames.length * 1000 / (elapsed + queueDrainMs),
        scenesPerFrame: frames.reduce((sum, value) => sum + value, 0) / frames.length,
        frameMsP95: percentile(intervals, .95),
        cpuMsMedian: percentile(renderCPU, .5),
        copyMsMedian: percentile(copyCPUPerFrame, .5),
        queueDrainMs,
        copiesPerFrame: totalCopies / frames.length,
        submissionsPerFrame: device ? totalSubmissions / frames.length : null,
        adapter: adapter ? {vendor: adapter.vendor, architecture: adapter.architecture, fallback: adapter.isFallbackAdapter} :
          debugInfo && gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
      };
      // Time GPU passes separately so resolving queries does not affect the
      // throughput run. There is only one frame in flight in this phase.
      if (device?.features.has('timestamp-query')) {
        manager.threeRenderer.setAnimationLoop(null);
        const timestamps = [];
        gpu.backend.trackTimestamp = true;
        try {
          for (let frame = 0; frame < 12; frame++) {
            manager.render(performance.now());
            await gpu.resolveTimestampsAsync();
            // Three.js reports only its last internal frame. A viewer frame
            // can run multiple scenes and post-processing passes.
            const timings = gpu.backend.timestampQueryPool.render?.timestamps;
            if (timings?.size && frame >= 2) {
              timestamps.push(Array.from(timings.values()).reduce((sum, time) => sum + time, 0));
            }
          }
          result.gpuMsMedian = percentile(timestamps, .5);
        } finally {
          gpu.backend.trackTimestamp = false;
          manager.threeRenderer.setAnimationLoop((time, frame) => manager.render(time, frame));
        }
      }
      manager.render = render;
      manager.copyPixels = copyPixels;
      return result;

      function percentile(values, proportion) {
        values.sort((left, right) => left - right);
        return values[Math.ceil(values.length * proportion) - 1];
      }
    }, {seconds, dof: layout[name].dof});
    assert.ok(metrics.frames > 3, name + ' did not render enough frames');
    assert.equal(metrics.scenesPerFrame, layout[name].count, 'Some scenes stopped rendering');
    assert.equal(await page.locator('model-viewer').first().evaluate(viewer => viewer.activeRenderBackend), backend);
    let image;
    if (capture) {
      const bytes = await page.locator('model-viewer').evaluate(async viewer => {
        viewer.autoRotate = false;
        viewer.resetTurntableRotation();
        viewer.cameraOrbit = '0deg 75deg auto';
        if (viewer.availableAnimations.length) {
          viewer.pause();
          viewer.currentTime = 0.5;
        }
        await viewer.updateComplete;
        viewer.jumpCameraToGoal();
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        return Array.from(new Uint8Array(await (await viewer.toBlob()).arrayBuffer()));
      });
      image = decode(Uint8Array.from(bytes));
    }
    assert.deepEqual(errors, []);
    return {metrics, image};
  } finally {
    await page.close();
  }
}

function compareImages(first, second) {
  assert.equal(first.width, second.width);
  assert.equal(first.height, second.height);
  assert.equal(first.data.length, second.data.length);
  let difference = 0;
  let largest = 0;
  for (let index = 0; index < first.data.length; index++) {
    const value = Math.abs(first.data[index] - second.data[index]);
    difference += value;
    largest = Math.max(largest, value);
  }
  return {meanAbsoluteDifference: difference / first.data.length, maximumDifference: largest};
}

const browser = await chromium.launch({
  headless: true,
  channel: process.env.MODEL_VIEWER_TEST_CHROME_CHANNEL || 'chrome',
  args: ['--enable-unsafe-swiftshader', '--disable-frame-rate-limit', '--disable-gpu-vsync']
});
try {
  console.log('Chromium ' + browser.version() + ', DPR 1, fixed rendering scale, uncapped frame loop');
  const entries = [...variants];
  for (const name of cases) {
    for (const backend of backends) {
      const images = new Map();
      for (let round = 0; round < rounds; round++) {
        const order = round % 2 === 0 ? entries : [...entries].reverse();
        for (const [variant, assets] of order) {
          const {metrics, image} = await benchmark(browser, name, backend, assets,
            Boolean(baseline) && layout[name].count === 1 && round === 0);
          console.log(JSON.stringify({name, backend, variant, round: round + 1, ...metrics}));
          if (image) images.set(variant, image);
        }
      }
      if (images.size === 2) {
        console.log(JSON.stringify({name, backend, imageComparison:
          compareImages(images.get('baseline'), images.get('current'))}));
      }
    }
  }
} finally {
  await browser.close();
}
