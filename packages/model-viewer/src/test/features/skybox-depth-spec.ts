import '../renderer-gate.js';

import {expect} from 'chai';
import {encode} from 'fast-png';
import {EquirectangularReflectionMapping, SRGBColorSpace, TextureLoader} from 'three';

import {$scene} from '../../model-viewer-base.js';
import {ModelViewerElement} from '../../model-viewer.js';
import {Renderer} from '../../three-components/Renderer.js';
import {loadSkyboxDepth, sampleSkyboxDepth} from '../../three-components/SkyboxDepth.js';
import {SkyboxDof} from '../../three-components/SkyboxDof.js';
import {assetPath, rafPasses, until} from '../helpers.js';

suite('Skybox depth of field', () => {
  let element: ModelViewerElement;
  let urls: string[];

  suiteSetup(() => {
    Renderer.singleton.minScale = 1;
  });

  suiteTeardown(() => {
    Renderer.resetSingleton();
  });

  setup(() => {
    urls = [];
    element = new ModelViewerElement();
    element.skyboxOnly = true;
    element.style.cssText = 'display:block;width:320px;height:180px';
  });

  teardown(() => {
    element.remove();
    urls.forEach(url => URL.revokeObjectURL(url));
  });

  function png(
      width: number,
      height: number,
      data: Uint8Array|Uint16Array,
      channels: number) {
    const depth = data instanceof Uint16Array ? 16 : 8;
    const bytes =
        new Uint8Array(encode({width, height, channels, depth, data}));
    const url = URL.createObjectURL(new Blob([bytes], {type: 'image/png'}));
    urls.push(url);
    return url;
  }

  function stripes() {
    const width = 512;
    const height = 256;
    const pixels = new Uint8Array(width * height * 3);
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        const luminance = x % 8 < 4 ? 190 : 55;
        const offset = 3 * (y * width + x);
        pixels.set([luminance, luminance, luminance], offset);
      }
    }
    return png(width, height, pixels, 3);
  }

  async function canvasPixels() {
    const renderer = Renderer.singleton;
    const scene = element[$scene];
    await until(
        () => element.modelIsVisible && scene.width === 320 &&
            scene.height === 180);
    const previousRender = scene.renderCount;
    scene.queueRender();
    await until(
        () => scene.renderCount > previousRender && scene.scaleStep === 0);
    const {canvas3D} = renderer;
    const canvas = document.createElement('canvas');
    canvas.width = canvas3D.width;
    canvas.height = canvas3D.height;
    const context = canvas.getContext('2d')!;
    context.drawImage(canvas3D, 0, 0);
    return context;
  }

  async function capture() {
    const context = await canvasPixels();
    const {canvas} = context;
    const row = context
                    .getImageData(
                        Math.floor(canvas.width / 4),
                        Math.floor(canvas.height / 2),
                        Math.floor(canvas.width / 2),
                        1)
                    .data;
    const colors = Array.from(row).filter((_, index) => index % 4 === 0);
    return {
      mean: colors.reduce((a, b) => a + b, 0) / colors.length,
      contrast: Math.max(...colors) - Math.min(...colors),
    };
  }

  test(
      'decodes all sixteen bits and shares the panorama orientation with autofocus',
      async () => {
        const samples = new Uint16Array(
            [0x1234, 0x1234, 0x1234, 0x1234, 0xabcd, 0xabcd, 0xabcd, 0xabcd]);
        const url = png(4, 2, samples, 1);
        const depth =
            await loadSkyboxDepth(url, false, new AbortController().signal);
        expect(depth.bitDepth).to.equal(16);
        expect(sampleSkyboxDepth(depth, 0, 1, 0, true))
            .to.equal(0x1234 / 65535);
        expect(sampleSkyboxDepth(depth, 0, -1, 0, true))
            .to.equal(0xabcd / 65535);
        expect(sampleSkyboxDepth(depth, 0, 1, 0, false))
            .to.equal(0xabcd / 65535);
      });

  test(
      'disabling the effect never requests the available depth asset',
      async () => {
        const depthUrl = png(4, 2, new Uint8Array(8), 1);
        element.skyboxImage = stripes();
        element.skyboxDofMaxBlur = 0;
        element.skyboxDepthImage = depthUrl;
        const originalFetch = window.fetch;
        const requests: string[] = [];
        window.fetch = (input, init) => {
          requests.push(String(input));
          return originalFetch(input, init);
        };
        try {
          document.body.append(element);
          await until(() => element.loaded);
          await rafPasses();
          expect(requests).not.to.include(depthUrl);
          expect(element[$scene].getSkyboxDofFrame()).to.equal(null);
        } finally {
          window.fetch = originalFetch;
        }
      });

  test(
      'renders a defocused panorama and restores the native sharp background',
      async () => {
        element.skyboxImage = stripes();
        document.body.append(element);
        await until(() => element.loaded && element[$scene].background != null);
        const sharp = await capture();
        expect(sharp.contrast).to.be.greaterThan(70);

        element.skyboxDepthImage = png(8, 4, new Uint8Array(32), 1);
        await until(() => element[$scene].getSkyboxDofFrame() != null);
        const blurred = await capture();
        expect(blurred.contrast).to.be.lessThan(sharp.contrast * .85);
        expect(blurred.mean).to.be.closeTo(sharp.mean, 24);

        element.skyboxDofMaxBlur = 0;
        await element.updateComplete;
        const restored = await capture();
        expect(restored.contrast).to.be.closeTo(sharp.contrast, 2);
        expect(restored.mean).to.be.closeTo(sharp.mean, 2);
      });

  test(
      'keeps the native projection with an in-focus map at the seam and poles',
      async () => {
        const width = 512;
        const height = width / 2;
        const pixels = new Uint8Array(width * height * 3);
        for (let y = 0; y < height; ++y) {
          for (let x = 0; x < width; ++x) {
            const longitude = 2 * Math.PI * x / width;
            pixels.set(
                [
                  Math.round(128 + 110 * Math.cos(longitude)),
                  Math.round(128 + 110 * Math.sin(longitude)),
                  Math.round(30 + 180 * y / height),
                ],
                3 * (y * width + x));
          }
        }
        const color =
            await new TextureLoader().loadAsync(png(width, height, pixels, 3));
        color.mapping = EquirectangularReflectionMapping;
        color.colorSpace = SRGBColorSpace;
        const renderer = Renderer.singleton.threeRenderer;
        renderer.setSize(320, 180, false);
        const effect = new SkyboxDof();
        const scene = element[$scene];
        scene.setSize(320, 180);
        scene.setBackground(color);

        const canvas = document.createElement('canvas');
        canvas.width = 320;
        canvas.height = 180;
        const context = canvas.getContext('2d')!;
        const capture = () => {
          context.drawImage(renderer.domElement, 0, 0);
          return context.getImageData(60, 40, 200, 100).data;
        };

        try {
          for (const [theta, phi, fov] of
                   [[90, 90, 45], [0, 6, 60], [0, 174, 30]]) {
            const camera = scene.camera;
            camera.position.setFromSphericalCoords(
                1, phi * Math.PI / 180, theta * Math.PI / 180);
            camera.lookAt(0, 0, 0);
            camera.fov = fov;
            camera.updateProjectionMatrix();
            renderer.render(scene, camera);
            const sharp = capture();

            scene.setSkyboxDepth(color, {
              width: 8,
              height: 4,
              bitDepth: 8,
              samples: new Uint8Array(32).fill(255),
            });
            effect.render(
                renderer, scene, scene.getSkyboxDofFrame()!, 320, 180);
            const active = capture();
            let difference = 0;
            for (let i = 0; i < sharp.length; ++i) {
              difference += Math.abs(sharp[i] - active[i]);
            }
            expect(difference / sharp.length, `theta ${theta}, phi ${phi}`)
                .to.be.lessThan(4);
          }
        } finally {
          scene.clearSkyboxDepth();
          effect.dispose();
          color.dispose();
        }
      });

  test('keeps an HDR panorama visible with the effect enabled', async () => {
    const renderer = Renderer.singleton.threeRenderer;
    if (!renderer.extensions.has('EXT_color_buffer_float')) {
      return;
    }
    element.skyboxImage = assetPath('environments/spruit_sunrise_1k_HDR.hdr');
    document.body.append(element);
    await until(() => element.loaded && element[$scene].background != null);
    const sharp = await capture();
    element.skyboxDepthImage = png(8, 4, new Uint8Array(32), 1);
    await until(() => element[$scene].getSkyboxDofFrame() != null);
    const blurred = await capture();
    expect(sharp.mean).to.be.greaterThan(20);
    expect(blurred.mean).to.be.closeTo(sharp.mean, 25);
  });

  test('can focus the view center or an off-center hotspot', async () => {
    const hotspot = document.createElement('button');
    hotspot.slot = 'hotspot-high';
    hotspot.dataset.position = '0m 0.8m -1m';
    hotspot.dataset.normal = '0m -0.8m 1m';
    element.append(hotspot);
    element.cameraOrbit = '0deg 70deg auto';
    element.skyboxImage = stripes();
    const depth = new Uint8Array(32).fill(210);
    depth.fill(80, 0, 16);
    element.skyboxDepthImage = png(8, 4, depth, 1);
    document.body.append(element);
    await until(
        () => element.modelIsVisible &&
            element[$scene].getSkyboxDofFrame() != null &&
            element.queryHotspot(hotspot.slot) != null);
    await element.updateComplete;
    element.jumpCameraToGoal();
    await until(() => hotspot.getClientRects().length > 0);
    expect(element.queryHotspot(hotspot.slot)?.canvasPosition.z)
        .to.be.lessThan(1);

    const bounds = element.getBoundingClientRect();
    expect(element.focusSkyboxAtPoint(bounds.right + 1, bounds.top))
        .to.equal(null);
    expect(element.focusSkyboxOnHotspot('hotspot-missing')).to.equal(null);
    expect(element.focusSkyboxAtPoint(
               bounds.x + bounds.width / 2, bounds.y + bounds.height / 2))
        .to.equal(210 / 255);
    expect(element.focusSkyboxOnHotspot(hotspot.slot)).to.equal(80 / 255);
    expect(element.skyboxDofFocus).to.equal(80 / 255);
    expect(element.skyboxDofFocusMode).to.equal('manual');
  });

  test(
      'ignores a stale depth response after swapping both assets', async () => {
        const firstDepthUrl = png(4, 2, new Uint8Array(8), 1);
        const nextDepth = new Uint16Array(8).fill(0xfedc);
        const nextDepthUrl = png(4, 2, nextDepth, 1);
        const originalFetch = window.fetch;
        let release = () => {};
        const blocked = new Promise<void>(resolve => release = resolve);
        let requested = false;
        window.fetch = async (input, init) => {
          if (String(input) === firstDepthUrl) {
            requested = true;
            await blocked;
          }
          return originalFetch(input, init);
        };

        try {
          element.skyboxImage = stripes();
          element.skyboxDepthImage = firstDepthUrl;
          element.skyboxDofFocusMode = 'auto';
          element.skyboxDofFocusSmoothing = 0;
          document.body.append(element);
          await until(() => requested && element.loaded);
          element.skyboxImage = stripes();
          element.skyboxDepthImage = nextDepthUrl;
          await until(
              () => element[$scene].getSkyboxDofAutoFocusTarget() ===
                  0xfedc / 65535);
          release();
          await rafPasses();
          expect(element[$scene].getSkyboxDofFocus()).to.equal(0xfedc / 65535);
        } finally {
          release();
          window.fetch = originalFetch;
        }
      });
});
