/* @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import '../renderer-gate.js';

import {expect} from 'chai';
import {Cache, CubeReflectionMapping, EquirectangularReflectionMapping, Texture, WebGLRenderer} from 'three';

import {CachingGLTFLoader} from '../../three-components/CachingGLTFLoader.js';
import {Renderer} from '../../three-components/Renderer.js';
import {skyboxSourceFlipY} from '../../three-components/SkyboxDepth.js';
import TextureUtils from '../../three-components/TextureUtils.js';
import {assetPath} from '../helpers.js';

// Reuse the same canvas as to not stress the WebGL
// context limit
const canvas = document.createElement('canvas');
const EQUI_URL = assetPath('environments/spruit_sunrise_1k_LDR.jpg');
const HDR_EQUI_URL = assetPath('environments/spruit_sunrise_1k_HDR.hdr');
const KTX2_URL = assetPath('models/CesiumLogoFlat.ktx2');

suite('TextureUtils', () => {
  let threeRenderer: WebGLRenderer;

  suiteSetup(function() {
    if (!Renderer.singleton.canRender) {
      this.skip();
    }
    // The threeRenderer can retain state, so these tests have the possibility
    // of getting different results in different orders. However, our use of the
    // threeRenderer *should* always return its state to what it was before to
    // avoid this kind of problem (and many other headaches).
    threeRenderer = new WebGLRenderer({canvas});
    threeRenderer.debug.checkShaderErrors = true;
    CachingGLTFLoader.setKTX2TranscoderLocation(
        'https://www.gstatic.com/basis-universal/versioned/2021-04-15-ba1c3e4/');
    CachingGLTFLoader.initializeKTX2Loader(threeRenderer);
  });

  suiteTeardown(() => {
    // Ensure we free up memory from loading large environment maps:
    Cache.clear();
    threeRenderer.dispose();
  });

  let textureUtils: TextureUtils;

  setup(() => {
    textureUtils = new TextureUtils(threeRenderer);
  });

  teardown(async () => {
    await textureUtils.dispose();
  });


  suite('load', () => {
    test('loads a valid texture from URL', async () => {
      let texture = await textureUtils.loadEquirect(EQUI_URL);
      texture.dispose();
      expect(texture.isTexture).to.be.ok;
      expect(texture.name).to.be.eq(EQUI_URL);
      expect(texture.mapping).to.be.eq(EquirectangularReflectionMapping);
    });
    test('loads a LDR equirect without the HDR JPG gainmap path', async () => {
      const textureUtilsAny = textureUtils as any;
      const originalImageLoader = textureUtilsAny.imageLoader;
      let imageLoaderCalled = false;
      textureUtilsAny.imageLoader = function(...args: unknown[]) {
        imageLoaderCalled = true;
        return originalImageLoader.apply(this, args);
      };

      try {
        const texture = await textureUtils.loadEquirect(EQUI_URL);
        texture.dispose();
        expect(imageLoaderCalled).to.be.eq(false);
      } finally {
        textureUtilsAny.imageLoader = originalImageLoader;
      }
    });
    test('decodes a gainmap and disposes intermediate render targets', async () => {
      const GAINMAP_URL = assetPath('environments/spruit_sunrise_1k_HDR.jpg');
      const THREE = await import('three');
      let disposeCount = 0;
      const originalDispose = THREE.WebGLRenderTarget.prototype.dispose;
      THREE.WebGLRenderTarget.prototype.dispose = function() {
        disposeCount++;
        return originalDispose.call(this);
      };

      try {
        const texture = await textureUtils.loadEquirect(GAINMAP_URL);
        texture.dispose();
        expect(disposeCount).to.be.greaterThan(0);
      } finally {
        THREE.WebGLRenderTarget.prototype.dispose = originalDispose;
      }
    });
    test('loads a valid KTX2 texture from URL', async () => {
      let texture = await textureUtils.loadImage(KTX2_URL, false);
      texture.dispose();
      expect(texture.isTexture).to.be.ok;
      expect(texture.name).to.be.eq(KTX2_URL);
    });
    test('loads a valid HDR texture from URL', async () => {
      let texture = await textureUtils.loadEquirect(HDR_EQUI_URL);
      texture.dispose();
      expect(texture.isTexture).to.be.ok;
      expect(texture.name).to.be.eq(HDR_EQUI_URL);
      expect(texture.mapping).to.be.eq(EquirectangularReflectionMapping);
    });
    test('throws on invalid URL', async () => {
      try {
        await textureUtils.loadEquirect('');
        expect(false).to.be.ok;
      } catch (e) {
        expect(true).to.be.ok;
      }
    });
    test('throws if texture not found', async () => {
      try {
        await textureUtils.loadEquirect('./nope.png');
        expect(false).to.be.ok;
      } catch (e) {
        expect(true).to.be.ok;
      }
    });
  });

  suite('generating an environment map and skybox', () => {
    test('returns an environmentMap and skybox texture from url', async () => {
      const textures =
          await textureUtils.generateEnvironmentMapAndSkybox(EQUI_URL);

      const skybox = textures.skybox!;
      const environment = textures.environmentMap;

      expect(skybox.isTexture).to.be.ok;
      expect(environment.isTexture).to.be.ok;

      expect(skybox.name).to.be.eq(EQUI_URL);
      expect(skybox.mapping).to.be.eq(EquirectangularReflectionMapping);

      expect(environment.name).to.be.eq(EQUI_URL);
      expect(environment.mapping).to.be.eq(EquirectangularReflectionMapping);
    });

    test('retries a url whose earlier load failed', async () => {
      const realLoadEquirect = textureUtils.loadEquirect.bind(textureUtils);
      let calls = 0;
      textureUtils.loadEquirect = async (url: string) => {
        calls++;
        if (calls === 1) {
          throw new Error('network error');
        }
        return realLoadEquirect(url);
      };

      let failed = false;
      try {
        await textureUtils.generateEnvironmentMapAndSkybox(EQUI_URL);
      } catch (e) {
        failed = true;
      }
      expect(failed).to.be.eq(true);

      const textures =
          await textureUtils.generateEnvironmentMapAndSkybox(EQUI_URL);

      expect(textures.environmentMap.name).to.be.eq(EQUI_URL);
      expect(textures.skybox!.name).to.be.eq(EQUI_URL);
    });

    test(
        'returns an environmentMap and skybox texture from an HDR url',
        async () => {
          const textures =
              await textureUtils.generateEnvironmentMapAndSkybox(HDR_EQUI_URL);

          const skybox = textures.skybox!;
          const environment = textures.environmentMap;

          expect(skybox.isTexture).to.be.ok;
          expect(environment.isTexture).to.be.ok;

          expect(skybox.name).to.be.eq(HDR_EQUI_URL);
          expect(skybox.mapping).to.be.eq(EquirectangularReflectionMapping);

          expect(environment.name).to.be.eq(HDR_EQUI_URL);
          expect(environment.mapping)
              .to.be.eq(EquirectangularReflectionMapping);
        });

    test(
        'returns an environmentMap and skybox texture from two urls',
        async () => {
          const textures = await textureUtils.generateEnvironmentMapAndSkybox(
              EQUI_URL, HDR_EQUI_URL);

          const skybox = textures.skybox!;
          const environment = textures.environmentMap;

          expect(skybox.isTexture).to.be.ok;
          expect(environment.isTexture).to.be.ok;

          expect(skybox.name).to.be.eq(EQUI_URL);
          expect(skybox.mapping).to.be.eq(EquirectangularReflectionMapping);

          expect(environment.name).to.be.eq(HDR_EQUI_URL);
          expect(environment.mapping)
              .to.be.eq(EquirectangularReflectionMapping);
        });

    test(
        'replaces a cached transient skybox when the source blob changes',
        async () => {
          const textureUtilsAny = textureUtils as any;
          let loadCount = 0;
          let disposeCount = 0;

          textureUtilsAny.loadTransientEquirect = async (url: string) => {
            loadCount++;
            return {
              isTexture: true,
              name: url,
              mapping: EquirectangularReflectionMapping,
              userData: {},
              dispose() {
                disposeCount++;
              }
            };
          };

          const first = await textureUtils.generateEnvironmentMapAndSkybox(
              'blob:first', null, () => {}, false, 'panorama');
          const second = await textureUtils.generateEnvironmentMapAndSkybox(
              'blob:second', null, () => {}, false, 'panorama');

          expect(first.skybox).to.not.equal(second.skybox);
          expect(loadCount).to.be.eq(2);
          expect(disposeCount).to.be.eq(1);
          expect(textureUtilsAny.skyboxCache.size).to.be.eq(1);
        });

    test('retries a failed transient skybox with a custom cache key', async () => {
      const textureUtilsAny = textureUtils as any;
      let attempts = 0;
      textureUtilsAny.loadTransientEquirect = async (url: string) => {
        attempts++;
        if (attempts === 1) {
          throw new Error('network error');
        }
        const texture = new (await import('three')).Texture();
        texture.name = url;
        return texture;
      };

      try {
        await textureUtils.generateEnvironmentMapAndSkybox(
            'blob:retry', null, () => {}, false, 'panorama');
        expect.fail('The first load should fail');
      } catch (error) {
        expect((error as Error).message).to.equal('network error');
      }

      expect(textureUtilsAny.skyboxCache.size).to.equal(0);

      const textures = await textureUtils.generateEnvironmentMapAndSkybox(
          'blob:retry', null, () => {}, false, 'panorama');
      expect(textures.skybox!.name).to.equal('blob:retry');
      expect(attempts).to.equal(2);
      expect(textureUtilsAny.skyboxCache.size).to.equal(1);
    });

    test('caches the original skybox orientation at the requested size', async () => {
      const image = document.createElement('canvas');
      image.width = 8;
      image.height = 4;
      const source = image.toDataURL();
      const load = (width: number|null) =>
          textureUtils.generateEnvironmentMapAndSkybox(
              source, null, () => {}, false, 'panorama', width);

      const legacy = (await load(null)).skybox!;
      expect(legacy.flipY).to.equal(false);
      const fullSize = (await load(0)).skybox!;
      expect(fullSize.flipY).to.equal(true);
      expect((fullSize.image as HTMLImageElement).width).to.equal(8);
      const resized = (await load(4)).skybox!;
      expect(skyboxSourceFlipY(resized)).to.equal(true);
      const bitmap = resized.image as ImageBitmap|HTMLCanvasElement;
      expect([bitmap.width, bitmap.height]).to.deep.equal([4, 2]);
      expect((await load(4)).skybox).to.equal(resized);
    });

    test('keeps a replaced texture alive while it is still displayed', async () => {
      let displayed: Texture|null = null;
      textureUtils = new TextureUtils(
          threeRenderer, () => null, (texture) => texture === displayed);
      const image = document.createElement('canvas');
      image.width = image.height = 2;
      const load = (url: string) => textureUtils.generateEnvironmentMapAndSkybox(
          url, null, () => {}, false, 'panorama');

      displayed = (await load(image.toDataURL())).skybox!;
      let disposed = false;
      displayed.addEventListener('dispose', () => {
        disposed = true;
      });
      image.getContext('2d')!.fillRect(0, 0, 1, 1);
      await load(image.toDataURL());
      textureUtils.releaseRetiredSkyboxes();
      expect(disposed).to.equal(false);
      displayed = null;
      textureUtils.releaseRetiredSkyboxes();
      expect(disposed).to.equal(true);
    });

    test('evicts idle URL skyboxes before releasing displayed ones', async () => {
      const active = new Set<Texture>();
      textureUtils = new TextureUtils(
          threeRenderer, () => null, texture => active.has(texture));
      textureUtils.loadEquirect = async (url: string) => {
        const texture = new Texture({width: 4096, height: 4096});
        texture.name = url;
        texture.generateMipmaps = false;
        return texture;
      };
      const load = async (url: string) =>
          (await textureUtils.generateEnvironmentMapAndSkybox(url)).skybox!;

      const outgoing = await load('outgoing.jpg');
      active.add(outgoing);
      const idle = await load('idle.jpg');
      const incoming = await load('incoming.jpg');
      active.add(incoming);
      let idleDisposed = false;
      idle.addEventListener('dispose', () => { idleDisposed = true; });

      textureUtils.releaseRetiredSkyboxes();
      expect(idleDisposed).to.equal(true);
      expect(await load('outgoing.jpg')).to.equal(outgoing);
      expect(await load('incoming.jpg')).to.equal(incoming);
      expect(await load('idle.jpg')).to.not.equal(idle);
    });

    test('throws if given an invalid url', async () => {
      try {
        await textureUtils.generateEnvironmentMapAndSkybox();
        expect(false).to.be.ok;
      } catch (e) {
        expect(true).to.be.ok;
      }
    });
  });

  suite('dynamically generating environment maps', () => {
    test('creates a cubemap render target with PMREM by default', async () => {
      const environment =
          (await textureUtils.generateEnvironmentMapAndSkybox()).environmentMap;

      expect(environment.name).to.be.eq('neutral');
      expect(environment.mapping).to.be.eq(CubeReflectionMapping);
    });

    test('creates a cubemap render target with PMREM for neutral', async () => {
      const environment =
          (await textureUtils.generateEnvironmentMapAndSkybox(null, 'neutral'))
              .environmentMap;

      expect(environment.name).to.be.eq('neutral');
      expect(environment.mapping).to.be.eq(CubeReflectionMapping);
    });

    test('creates a cubemap render target with PMREM for legacy', async () => {
      const environment =
          (await textureUtils.generateEnvironmentMapAndSkybox(null, 'legacy'))
              .environmentMap;

      expect(environment.name).to.be.eq('legacy');
      expect(environment.mapping).to.be.eq(CubeReflectionMapping);
    });
  });
});
