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

import {Event, EventDispatcher, NeutralToneMapping, Vector2, WebGLRenderer} from 'three';
import type {Texture} from 'three';
import type {WebGPURenderer} from 'three/webgpu';

import {$updateEnvironment} from '../features/environment.js';
import {ModelViewerGlobalConfig} from '../features/loading.js';
import ModelViewerElementBase, {$canvas, $setRenderBackend, $tick, $updateSize, EffectComposerInterface, RendererInterface} from '../model-viewer-base.js';
import type {RenderBackendPreference} from '../model-viewer-base.js';
import {clamp, isDebugMode} from '../utilities.js';

import {ARRenderer} from './ARRenderer.js';
import {CachingGLTFLoader} from './CachingGLTFLoader.js';
import {ModelViewerGLTFInstance} from './gltf-instance/ModelViewerGLTFInstance.js';
import {ModelScene} from './ModelScene.js';
import {SkyboxDof, supportsSkyboxDof} from './SkyboxDof.js';
import {SkyboxPanorama} from './SkyboxPanorama.js';
import TextureUtils from './TextureUtils.js';
import {ModelViewerViewHelper, ViewHelperOrbit} from './ViewHelper.js';
import type {WebGPUContext} from './WebGPU.js';

export interface RendererOptions {
  powerPreference: string;
  debug?: boolean;
}

export interface ContextLostEvent extends Event {
  type: 'contextlost';
  sourceEvent: WebGLContextEvent;
}

// Between 0 and 1: larger means the average responds faster and is less smooth.
const DURATION_DECAY = 0.2;
const LOW_FRAME_DURATION_MS = 40;
const HIGH_FRAME_DURATION_MS = 60;
const MAX_AVG_CHANGE_MS = 5;
const SCALE_STEPS = [1, 0.79, 0.62, 0.5, 0.4, 0.31, 0.25];
const DEFAULT_LAST_STEP = 3;
// Allow for jitter around the 16.7 ms refresh interval on 60 Hz displays.
const WEBGPU_SLOW_FRAME_MS = 18;
const WEBGPU_FAST_FRAME_MS = 17;
const WEBGPU_DOWNSCALE_FRAMES = 3;
const WEBGPU_UPSCALE_FRAMES = 60;

export const DEFAULT_POWER_PREFERENCE: string = 'high-performance';
const COMMERCE_EXPOSURE = 1.3;

/**
 * Shares a renderer across the registered scenes. WebGL multiplexes a shared
 * canvas, and WebGPU draws directly to each scene's canvas target.
 */
export class Renderer extends
    EventDispatcher<{contextlost: {sourceEvent: WebGLContextEvent}}> {
  private static _singleton: Renderer;

  static get singleton() {
    if (!this._singleton) {
      this._singleton = new Renderer({
        powerPreference: (((self as any).ModelViewerElement || {}) as
                          ModelViewerGlobalConfig)
                             .powerPreference ||
            DEFAULT_POWER_PREFERENCE,
        debug: isDebugMode()
      });
    }
    return this._singleton;
  }

  static resetSingleton() {
    const elements = this._singleton.dispose();
    for (const element of elements) {
      element.disconnectedCallback();
    }

    this._singleton = new Renderer({
      powerPreference:
          (((self as any).ModelViewerElement || {}) as ModelViewerGlobalConfig)
              .powerPreference ||
          DEFAULT_POWER_PREFERENCE,
      debug: isDebugMode()
    });

    for (const element of elements) {
      element.connectedCallback();
    }
  }

  public threeRenderer!: WebGLRenderer;
  public canvas3D: HTMLCanvasElement;
  public textureUtils: TextureUtils|null;
  public arRenderer: ARRenderer;
  public loader = new CachingGLTFLoader(
      ModelViewerGLTFInstance, () => this.gpuProbe);
  public width = 0;
  public height = 0;
  public dpr = 1;

  private scenes: Set<ModelScene> = new Set();
  private multipleWebGLScenesVisible = false;
  private lastTick = performance.now();
  private renderedLastFrame = false;
  private renderedWebGPULastFrame = false;
  private scaleStep = 0;
  private webGPUScaleStep = 0;
  private slowWebGPUFrames = 0;
  private fastWebGPUFrames = 0;
  private lastStep = DEFAULT_LAST_STEP;
  private avgFrameDuration =
      (HIGH_FRAME_DURATION_MS + LOW_FRAME_DURATION_MS) / 2;
  private viewHelpers = new Map<ModelScene, ModelViewerViewHelper>();
  private skyboxDof: SkyboxDof|null = null;
  private skyboxPanorama: SkyboxPanorama|null = null;
  private gpu: WebGPUContext|null = null;
  private gpuProbe: Promise<WebGPUContext|null>|null = null;
  private gpuPending = false;
  private gpuUnavailable = false;
  private disposed = false;
  private readonly failedGPUScenes = new WeakSet<ModelScene>();
  private readonly gpuEffects = new WeakMap<EffectComposerInterface, boolean>();
  private readonly externalBackends = new WeakMap<RendererInterface, {
    context: WebGPUContext|null,
    ready: boolean,
    pending: Promise<void>
  }>();

  get canRender() {
    return this.threeRenderer != null;
  }

  get scaleFactor() {
    return SCALE_STEPS[this.scaleStep];
  }

  set minScale(scale: number) {
    let i = 1;
    while (i < SCALE_STEPS.length) {
      if (SCALE_STEPS[i] < scale) {
        break;
      }
      ++i;
    }
    this.lastStep = i - 1;
  }

  constructor(private readonly options: RendererOptions) {
    super();

    this.dpr = window.devicePixelRatio;

    this.canvas3D = document.createElement('canvas');
    this.canvas3D.id = 'webgl-canvas';
    this.canvas3D.classList.add('show');

    try {
      this.threeRenderer = new WebGLRenderer({
        canvas: this.canvas3D,
        alpha: true,
        antialias: true,
        powerPreference: options.powerPreference as WebGLPowerPreference,
        preserveDrawingBuffer: true,
      });
      this.threeRenderer.autoClear = true;
      this.threeRenderer.setPixelRatio(1);  // handle pixel ratio externally

      this.threeRenderer.debug.checkShaderErrors = !!options.debug;

      // ACESFilmicToneMapping appears to be the most "saturated",
      // and similar to Filament's gltf-viewer.
      this.threeRenderer.toneMapping = NeutralToneMapping;
    } catch (error) {
      console.warn(error);
    }

    this.arRenderer = this.canRender ? new ARRenderer(this) : null as any;
    this.textureUtils = this.createTextureUtils();
    if (this.canRender) {
      CachingGLTFLoader.initializeKTX2Loader(this.threeRenderer);
    }

    this.canvas3D.addEventListener('webglcontextlost', this.onWebGLContextLost);
    this.canvas3D.addEventListener(
        'webglcontextrestored', this.onWebGLContextRestored);

    this.updateRendererSize();
  }

  registerScene(scene: ModelScene) {
    this.scenes.add(scene);

    scene.forceRescale();
    this.updateBackend(scene);

    if (this.canRender) {
      const size = new Vector2();
      this.threeRenderer.getSize(size);
      scene.canvas.width = size.x;
      scene.canvas.height = size.y;

      if (this.scenes.size > 0) {
        this.threeRenderer.setAnimationLoop(
            (time: number, frame?: any) => this.render(time, frame));
      }
    }
  }

  unregisterScene(scene: ModelScene) {
    this.scenes.delete(scene);
    this.gpu?.releaseCanvas(scene);
    this.textureUtils?.releaseRetiredSkyboxes();
    this.failedGPUScenes.delete(scene);
    this.disposeViewHelper(scene);
    if (this.skyboxDof != null) {
      this.releaseUnusedSkyboxDof();
    }

    if (this.canvas3D.parentElement === scene.canvas.parentElement) {
      scene.canvas.parentElement!.removeChild(this.canvas3D);
    }

    if (this.canRender && this.scenes.size === 0) {
      this.threeRenderer.setAnimationLoop(null);
    }
  }

  displayCanvas(scene: ModelScene): HTMLCanvasElement {
    const externalCanvas = scene.externalRenderer?.captureCanvas?.();
    if (externalCanvas != null) return externalCanvas;
    if (scene.element.activeRenderBackend === 'webgpu') {
      return this.gpu?.displayCanvas(scene) ?? scene.canvas;
    }
    return this.canvas3D.parentElement != null &&
            this.canvas3D.parentElement === scene.canvas.parentElement ?
        this.canvas3D : scene.element[$canvas];
  }

  captureSize(scene: ModelScene) {
    const canvas = this.gpu?.displayCanvas(scene);
    if (canvas != null) return {width: canvas.width, height: canvas.height};
    const captureDpr = this.dpr * this.scaleFactor;
    return {width: scene.width * captureDpr, height: scene.height * captureDpr};
  }

  captureCanvas(scene: ModelScene): HTMLCanvasElement {
    const canvas = this.displayCanvas(scene);
    const context = this.gpu;
    if (context != null && canvas === context.displayCanvas(scene) &&
        context.renderer.domElement === context.canvas && !this.isPresenting) {
      // Some browsers expose a fresh drawing buffer after presenting a WebGPU
      // canvas. Draw the current pose in this task before reading its pixels.
      this.updateExposure(scene, context.renderer);
      scene.renderShadow(context.renderer, context.shadows);
      this.renderWebGPU(scene, context, canvas.width, canvas.height, 0);
    }
    return canvas;
  }

  private wantsWebGPU(
      scene: ModelScene, renderBackend = scene.element.renderBackend): boolean {
    const {effectRenderer, externalRenderer} = scene;
    return (renderBackend === 'webgpu' || renderBackend === 'auto') &&
        !this.gpuUnavailable && !this.failedGPUScenes.has(scene) &&
        (externalRenderer == null || externalRenderer.prepareRenderBackend != null) &&
        (effectRenderer == null ||
         (effectRenderer.prepareWebGPU != null &&
          effectRenderer.renderWebGPU != null &&
          effectRenderer.supportsWebGPU?.() !== false));
  }

  async prepareSkybox(
      scene: ModelScene, textures: readonly Texture[],
      backend?: RenderBackendPreference): Promise<void> {
    if (this.disposed || !this.canRender) return;
    const wantsGPU = () =>
        scene.externalRenderer == null && this.wantsWebGPU(scene, backend);
    const context = wantsGPU() ? await this.probeWebGPU() : null;
    if (this.disposed) return;
    if (context != null && context === this.gpu && wantsGPU()) {
      await context.prepareTextures(textures);
      return;
    }
    for (const texture of textures) {
      if (!texture.isRenderTargetTexture) this.threeRenderer.initTexture(texture);
    }
  }

  updateBackend(scene: ModelScene) {
    if (!this.scenes.has(scene)) return;
    this.failedGPUScenes.delete(scene);
    if (this.wantsWebGPU(scene)) void this.probeWebGPU();
    scene.queueRender();
  }

  private probeWebGPU(): Promise<WebGPUContext|null> {
    if (this.disposed || this.gpuUnavailable || !globalThis.isSecureContext ||
        !navigator.gpu) {
      return Promise.resolve(null);
    }
    if (this.gpuProbe != null) return this.gpuProbe;
    this.gpuPending = true;
    this.gpuProbe = import('./WebGPU.js')
                        .then(({WebGPUContext}) => WebGPUContext.create(
                            this.options.powerPreference, this.handleDeviceLoss))
                        .then(context => {
                          if (this.disposed || this.gpuUnavailable) {
                            context?.dispose();
                            return null;
                          }
                          if (context != null) {
                            this.gpu = context;
                            context.renderer.setSize(
                                this.canvas3D.width, this.canvas3D.height, false);
                            CachingGLTFLoader.initializeKTX2Loader(
                                this.threeRenderer, context.renderer);
                          } else {
                            this.gpuUnavailable = true;
                          }
                          return context;
                        })
                        .catch(error => {
                          console.warn('<model-viewer> WebGPU initialization failed.', error);
                          this.gpuUnavailable = true;
                          this.gpu?.dispose();
                          this.gpu = null;
                          return null;
                        })
                        .finally(() => {
                          this.gpuPending = false;
                          for (const scene of this.scenes) scene.queueRender();
                        });
    return this.gpuProbe;
  }

  async prepareExternalRenderer(scene: ModelScene, external: RendererInterface) {
    if (external.prepareRenderBackend == null) return;
    const context = this.wantsWebGPU(scene) ? await this.probeWebGPU() : null;
    await this.prepareExternalBackend(scene, external, context);
  }

  private prepareExternalBackend(
      scene: ModelScene, external: RendererInterface,
      context: WebGPUContext|null) {
    const previous = this.externalBackends.get(external);
    if (previous?.context === context) return previous.pending;
    const state = {context, ready: false, pending: Promise.resolve()};
    const onError = (error: unknown) => {
      if (this.externalBackends.get(external) === state) {
        this.fallbackScene(scene, error);
      }
    };
    state.pending = (previous?.pending.catch(() => {}) ?? Promise.resolve())
                        .then(() => external.prepareRenderBackend!(context, onError))
                        .catch(async error => {
                          if (context == null) throw error;
                          this.fallbackScene(scene, error);
                          state.context = null;
                          await external.prepareRenderBackend!(null, onError);
                        })
                        .then(() => {
                          state.ready = true;
                          scene.queueRender();
                        });
    this.externalBackends.set(external, state);
    return state.pending;
  }

  private fallbackScene(scene: ModelScene, error: unknown) {
    this.gpu?.releaseCanvas(scene);
    this.failedGPUScenes.add(scene);
    scene.queueRender();
    console.warn('<model-viewer> Using WebGL for this scene.', error);
  }

  private handleDeviceLoss = () => {
    if (this.gpuUnavailable || this.disposed) return;
    this.gpuUnavailable = true;
    const context = this.gpu;
    this.gpu = null;
    for (const scene of this.scenes) scene.queueRender();
    if (this.isPresenting) {
      void this.arRenderer.stopPresenting().finally(() => context?.dispose());
    } else {
      context?.dispose();
    }
  };

  webGPUForXR(scene: ModelScene, environmentEstimation: boolean) {
    if (environmentEstimation || !this.wantsWebGPU(scene) ||
        !('XRGPUBinding' in globalThis)) {
      return null;
    }
    return this.gpu;
  }

  private hasNativeWebGPUCanvas(scene: ModelScene): boolean {
    const parent = scene.canvas.parentElement;
    return parent != null && scene.element.activeRenderBackend === 'webgpu' &&
        this.gpu?.displayCanvas(scene)?.parentElement === parent &&
        this.wantsWebGPU(scene) &&
        scene.width > 0 && scene.height > 0 &&
        this.canvas3D.width > 0 && this.canvas3D.height > 0;
  }

  private snapshotWebGLCanvas(scene: ModelScene) {
    const {width, height} = this.sceneSize(scene);
    this.copyPixels(scene, width, height);
    this.canvas3D.remove();
  }

  /** A single visible WebGL scene can present the shared canvas directly. */
  private updateWebGLPresentation() {
    const {canvas3D} = this;
    let visibleScenes = 0;
    let visibleWebGLScene: ModelScene|null = null;
    let canvasOwner: ModelScene|null = null;
    const owner = canvas3D.parentElement;
    for (const scene of this.scenes) {
      const {element} = scene;
      if (element.modelIsVisible && scene.externalRenderer == null &&
          !this.hasNativeWebGPUCanvas(scene)) {
        ++visibleScenes;
        visibleWebGLScene = scene;
      }
      if (owner != null && owner === scene.canvas.parentElement) {
        canvasOwner = scene;
      }
    }
    const directScene = visibleScenes === 1 ? visibleWebGLScene : null;
    if (canvasOwner != null && canvasOwner !== directScene) {
      this.snapshotWebGLCanvas(canvasOwner);
    }
    if (directScene != null && directScene !== canvasOwner) directScene.queueRender();
    this.multipleWebGLScenesVisible = visibleScenes > 1;
  }

  /**
   * Updates the renderer's size based on the largest scene and any changes to
   * device pixel ratio.
   */
  private updateRendererSize() {
    const dpr = window.devicePixelRatio;
    if (dpr !== this.dpr) {
      // If the device pixel ratio has changed due to page zoom, elements
      // specified by % width do not fire a resize event even though their CSS
      // pixel dimensions change, so we force them to update their size here.
      for (const scene of this.scenes) {
        const {element} = scene;
        element[$updateSize](element.getBoundingClientRect());
      }
    }

    // Make the renderer the size of the largest scene
    let width = 0;
    let height = 0;
    for (const scene of this.scenes) {
      width = Math.max(width, scene.width);
      height = Math.max(height, scene.height);
    }

    if (width === this.width && height === this.height && dpr === this.dpr) {
      return;
    }
    this.width = width;
    this.height = height;
    this.dpr = dpr;
    width = Math.ceil(width * dpr);
    height = Math.ceil(height * dpr);

    if (this.canRender) {
      this.threeRenderer.setSize(width, height, false);
    }
    this.gpu?.renderer.setSize(width, height, false);

    // Each scene's canvas must match the renderer size. In general they can be
    // larger than the element that contains them, but the overflow is hidden
    // and only the portion that is shown is copied over.
    for (const scene of this.scenes) {
      const {canvas} = scene;
      canvas.width = width;
      canvas.height = height;
      scene.forceRescale();
      scene.effectRenderer?.setSize(width, height);
    }
  }

  private updateRendererScale(delta: number) {
    const scaleStep = this.scaleStep;

    this.avgFrameDuration += clamp(
        DURATION_DECAY * (delta - this.avgFrameDuration),
        -MAX_AVG_CHANGE_MS,
        MAX_AVG_CHANGE_MS);

    if (this.avgFrameDuration > HIGH_FRAME_DURATION_MS) {
      ++this.scaleStep;
    } else if (
        this.avgFrameDuration < LOW_FRAME_DURATION_MS && this.scaleStep > 0) {
      --this.scaleStep;
    }
    this.scaleStep = Math.min(this.scaleStep, this.lastStep);

    if (scaleStep !== this.scaleStep) {
      this.avgFrameDuration =
          (HIGH_FRAME_DURATION_MS + LOW_FRAME_DURATION_MS) / 2;
    }
  }

  private updateWebGPUScale(delta: number) {
    const sampled = this.renderedWebGPULastFrame && delta > 0 && delta <= 200;
    this.slowWebGPUFrames =
        sampled && delta > WEBGPU_SLOW_FRAME_MS ? this.slowWebGPUFrames + 1 : 0;
    this.fastWebGPUFrames =
        sampled && delta <= WEBGPU_FAST_FRAME_MS ? this.fastWebGPUFrames + 1 : 0;

    if (this.slowWebGPUFrames >= WEBGPU_DOWNSCALE_FRAMES) {
      ++this.webGPUScaleStep;
      this.slowWebGPUFrames = 0;
    } else if (this.fastWebGPUFrames >= WEBGPU_UPSCALE_FRAMES) {
      --this.webGPUScaleStep;
      this.fastWebGPUFrames = 0;
    }
    this.webGPUScaleStep = clamp(this.webGPUScaleStep, 0, this.lastStep);
    this.renderedWebGPULastFrame = false;
  }

  private shouldRender(scene: ModelScene, scaleStep: number): boolean {
    if (!scene.shouldRender()) {
      // The first frame we stop rendering the scene (because it stops moving),
      // trigger one extra render at full scale.
      if (scene.scaleStep != 0) {
        scene.scaleStep = 0;
        this.rescaleCanvas(scene);
      } else {
        return false;
      }
    } else if (scene.scaleStep != scaleStep) {
      // Update render scale
      scene.scaleStep = scaleStep;
      this.rescaleCanvas(scene);
    }
    return true;
  }

  private rescaleCanvas(scene: ModelScene) {
    const scale = SCALE_STEPS[scene.scaleStep];
    const width = Math.ceil(this.width / scale);
    const height = Math.ceil(this.height / scale);

    const {style} = scene.canvas;
    style.width = `${width}px`;
    style.height = `${height}px`;
    if (this.canvas3D.parentElement != null &&
        this.canvas3D.parentElement === scene.canvas.parentElement) {
      this.canvas3D.style.width = style.width;
      this.canvas3D.style.height = style.height;
    }

    const renderedDpr = this.dpr * scale;
    const reason = scale < 1                 ? 'GPU throttling' :
        this.dpr !== window.devicePixelRatio ? 'No meta viewport tag' :
                                               '';
    scene.element.dispatchEvent(new CustomEvent('render-scale', {
      detail: {
        reportedDpr: window.devicePixelRatio,
        renderedDpr: renderedDpr,
        minimumDpr: this.dpr * SCALE_STEPS[this.lastStep],
        pixelWidth: Math.ceil(scene.width * renderedDpr),
        pixelHeight: Math.ceil(scene.height * renderedDpr),
        reason: reason
      }
    }));
  }

  private sceneSize(scene: ModelScene) {
    const {dpr} = this;
    const scaleFactor = SCALE_STEPS[scene.scaleStep] ?? SCALE_STEPS[this.scaleStep];
    // We avoid using the Three.js PixelRatio and handle it ourselves here so
    // that we can do proper rounding and avoid white boundary pixels.
    const width = Math.min(
        Math.ceil(scene.width * scaleFactor * dpr), this.canvas3D.width);
    const height = Math.min(
        Math.ceil(scene.height * scaleFactor * dpr), this.canvas3D.height);
    return {width, height};
  }

  private disposeViewHelper(scene: ModelScene) {
    const viewHelper = this.viewHelpers.get(scene);
    if (viewHelper == null) {
      return;
    }
    viewHelper.dispose();
    this.viewHelpers.delete(scene);
  }

  private renderViewHelper(
      scene: ModelScene, x: number, y: number, width: number, height: number,
      renderer: WebGLRenderer|WebGPURenderer = this.threeRenderer) {
    if (!scene.element.viewHelper) {
      this.disposeViewHelper(scene);
      return;
    }

    let viewHelper = this.viewHelpers.get(scene);
    if (viewHelper == null) {
      viewHelper = new ModelViewerViewHelper(scene.camera);
      this.viewHelpers.set(scene, viewHelper);
    }

    viewHelper.render(renderer, {
      x,
      y,
      width,
      height,
      canvasWidth: renderer.domElement.width,
      canvasHeight: renderer.domElement.height,
      renderedDpr: this.dpr * (SCALE_STEPS[scene.scaleStep] ?? 1)
    });
  }

  hasViewHelperAtPoint(scene: ModelScene, event: PointerEvent): boolean {
    if (!scene.element.viewHelper) {
      return false;
    }

    return this.viewHelpers.get(scene)?.containsPoint(
        event, scene.element.getBoundingClientRect()) === true;
  }

  handleViewHelperClick(scene: ModelScene, event: PointerEvent):
      ViewHelperOrbit|null {
    if (!scene.element.viewHelper) {
      return null;
    }

    return this.viewHelpers.get(scene)?.handleClick(
               event, scene.element.getBoundingClientRect()) ||
        null;
  }

  private copyPixels(scene: ModelScene, width: number, height: number) {
    const context2D = scene.context;
    if (context2D == null) {
      console.log('could not acquire 2d context');
      return;
    }
    context2D.clearRect(0, 0, width, height);
    context2D.drawImage(
        this.canvas3D, 0, 0, width, height, 0, 0, width, height);
    scene.canvas.classList.add('show');
  }

  /**
   * Returns an array version of this.scenes where the non-visible ones are
   * first. This allows eager scenes to be rendered before they are visible,
   * without needing the multi-canvas render path.
   */
  private orderedScenes(): Array<ModelScene> {
    const scenes = [];
    for (const visible of [false, true]) {
      for (const scene of this.scenes) {
        if (scene.element.modelIsVisible === visible) {
          scenes.push(scene);
        }
      }
    }
    return scenes;
  }

  get isPresenting(): boolean {
    return (this as any).arRenderer?.isPresenting === true;
  }

  /**
   * This method takes care of updating the element and renderer state based on
   * the time that has passed since the last rendered frame.
   */
  preRender(
      scene: ModelScene, t: number, delta: number,
      renderer: WebGLRenderer|WebGPURenderer = this.threeRenderer) {
    scene.element[$tick](t, delta);
    this.updateExposure(scene, renderer);
  }

  private updateExposure(
      scene: ModelScene, renderer: WebGLRenderer|WebGPURenderer) {
    const {element, exposure, toneMapping} = scene;
    const exposureIsNumber =
        typeof exposure === 'number' && !Number.isNaN(exposure);
    const env = element.environmentImage;
    const sky = element.skyboxImage;
    const compensateExposure = toneMapping === NeutralToneMapping &&
        (env === 'neutral' || env === 'legacy' || (!env && !sky));
    renderer.toneMappingExposure =
        (exposureIsNumber ? exposure : 1.0) *
        (compensateExposure ? COMMERCE_EXPOSURE : 1.0);
  }

  private renderWebGPU(
      scene: ModelScene, context: WebGPUContext, width: number, height: number,
      delta: number) {
    const {renderer} = context;
    const {effectRenderer} = scene;
    let panoramaMoved = false;
    context.withCanvas(scene, width, height, () => {
      renderer.setRenderTarget(null);
      renderer.setViewport(0, 0, width, height);
      renderer.autoClear = true;
      renderer.toneMapping = scene.toneMapping;
      context.withScene(scene, () => {
        if (effectRenderer?.renderWebGPU != null &&
            this.gpuEffects.get(effectRenderer) === true) {
          effectRenderer.renderWebGPU(delta);
        } else {
          panoramaMoved = context.render(scene, width, height);
        }
      });
      this.renderViewHelper(scene, 0, 0, width, height, renderer);
    });
    return panoramaMoved;
  }

  render(t: number, frame?: XRFrame, forceScene?: ModelScene) {
    if (frame != null) {
      if (this.arRenderer != null) {
        this.arRenderer.onWebXRFrame(t, frame);
      }
      return;
    }

    const delta = t - this.lastTick;
    this.lastTick = t;

    if (!this.canRender || this.isPresenting) {
      return;
    }

    this.updateWebGLPresentation();
    this.updateRendererSize();
    if (this.renderedLastFrame) {
      this.updateRendererScale(delta);
      this.renderedLastFrame = false;
    }
    if (this.gpu != null) this.updateWebGPUScale(delta);

    const {canvas3D} = this;

    for (const scene of this.orderedScenes()) {
      const {element} = scene;
      if (!element.loaded ||
          (!element.modelIsVisible && scene.renderCount > 0 && scene !== forceScene)) {
        continue;
      }

      const wantsGPU = this.wantsWebGPU(scene);
      if (wantsGPU && this.gpuPending) continue;
      const context = wantsGPU ? this.gpu : null;
      const {externalRenderer, effectRenderer} = scene;
      if (externalRenderer?.prepareRenderBackend != null) {
        const state = this.externalBackends.get(externalRenderer);
        if (state == null || state.context !== context) {
          void this.prepareExternalBackend(scene, externalRenderer, context)
              .catch(error => console.error(error));
          continue;
        }
        if (!state.ready) continue;
      } else if (context != null && effectRenderer != null) {
        const ready = this.gpuEffects.get(effectRenderer);
        if (ready !== true) {
          if (ready == null) {
            this.gpuEffects.set(effectRenderer, false);
            void effectRenderer.prepareWebGPU!(context).then(() => {
              this.gpuEffects.set(effectRenderer, true);
              scene.queueRender();
            }, error => {
              this.gpuEffects.delete(effectRenderer);
              this.fallbackScene(scene, error);
            });
          }
          continue;
        }
      }
      this.preRender(scene, t, delta, context?.renderer);
      scene.updateSkyboxDof(delta);

      const nativeWebGPU = context != null && externalRenderer == null;
      if (!this.shouldRender(
              scene, nativeWebGPU ? this.webGPUScaleStep : this.scaleStep)) {
        continue;
      }

      if (externalRenderer != null) {
        const camera = scene.getCamera();
        camera.updateMatrix();
        const {matrix, projectionMatrix} = camera;
        const viewMatrix = matrix.elements.slice();
        const target = scene.getTarget();
        viewMatrix[12] += target.x;
        viewMatrix[13] += target.y;
        viewMatrix[14] += target.z;

        try {
          externalRenderer.render({
            viewMatrix: viewMatrix,
            projectionMatrix: projectionMatrix.elements
          });
          this.gpu?.releaseCanvas(scene);
          element[$setRenderBackend](context != null ? 'webgpu' : 'webgl');
        } catch (error) {
          if (context == null) throw error;
          this.fallbackScene(scene, error);
        }
        continue;
      }

      const {width, height} = this.sceneSize(scene);

      if (context != null && width > 0 && height > 0) {
        try {
          scene.renderShadow(context.renderer, context.shadows, t);
          const panoramaMoved = this.renderWebGPU(scene, context, width, height, delta);
          scene.canvas.classList.remove('show');
          if (canvas3D.parentElement === scene.canvas.parentElement) {
            canvas3D.remove();
          }
          element[$setRenderBackend]('webgpu');
          scene.hasRendered();
          if (panoramaMoved) scene.queueRender();
          ++scene.renderCount;
          this.renderedWebGPULastFrame ||= element.modelIsVisible;
        } catch (error) {
          this.fallbackScene(scene, error);
        }
        continue;
      }

      const canvasOwner = canvas3D.parentElement;
      if (canvasOwner != null && canvasOwner !== scene.canvas.parentElement) {
        // Rendering callbacks can change a later scene's backend during this frame.
        for (const previous of this.scenes) {
          if (previous.canvas.parentElement === canvasOwner) {
            this.snapshotWebGLCanvas(previous);
            if (element.modelIsVisible) this.multipleWebGLScenesVisible = true;
            break;
          }
        }
      }
      if (!element.modelIsVisible && !this.multipleWebGLScenesVisible) {
        // The shared WebGL canvas needs another draw for the visible scene.
        for (const visibleScene of this.scenes) {
          if (visibleScene.element.modelIsVisible &&
              visibleScene.externalRenderer == null &&
              !this.hasNativeWebGPUCanvas(visibleScene)) {
            visibleScene.queueRender();
          }
        }
      }
      const viewportY = Math.ceil(this.height * this.dpr) - height;
      scene.renderShadow(this.threeRenderer);

      // Need to set the render target in order to prevent
      // clearing the depth from a different buffer
      this.threeRenderer.setRenderTarget(null);
      this.threeRenderer.setViewport(0, viewportY, width, height);
      if (scene.effectRenderer != null) {
        scene.effectRenderer.render(delta);
      } else {
        this.threeRenderer.autoClear =
            true;  // this might get reset by the effectRenderer
        this.threeRenderer.toneMapping = scene.toneMapping;
        const dof = scene.getSkyboxDofFrame();
        if (dof != null && supportsSkyboxDof(this.threeRenderer, dof.skybox)) {
          const effect = this.skyboxDof ??= new SkyboxDof();
          effect.reportFrame(t, delta);
          effect.render(this.threeRenderer, scene, dof, width, height);
        } else if (!element.skyboxOnly ||
                   !(this.skyboxPanorama ??= new SkyboxPanorama())
                        .render(this.threeRenderer, scene)) {
          this.threeRenderer.render(scene, scene.camera);
        }
      }
      this.renderViewHelper(scene, 0, viewportY, width, height);
      if (this.multipleWebGLScenesVisible ||
          (!scene.element.modelIsVisible && (scene.renderCount === 0 || scene === forceScene))) {
        this.copyPixels(scene, width, height);
      } else if (canvas3D.parentElement !== scene.canvas.parentElement) {
        canvas3D.style.width = scene.canvas.style.width;
        canvas3D.style.height = scene.canvas.style.height;
        scene.canvas.parentElement!.appendChild(canvas3D);
        scene.canvas.classList.remove('show');
      }

      this.gpu?.releaseCanvas(scene);
      element[$setRenderBackend]('webgl');
      scene.hasRendered();
      ++scene.renderCount;
      this.renderedLastFrame = true;
    }
    if (this.skyboxDof != null || this.gpu?.hasSkyboxDof === true) {
      this.releaseUnusedSkyboxDof();
    }
    this.textureUtils?.releaseRetiredSkyboxes();
  }

  private createTextureUtils(): TextureUtils|null {
    return this.canRender ?
        new TextureUtils(
            this.threeRenderer,
            () => this.gpuUnavailable ? null : this.gpuProbe,
            (texture) => Array.from(this.scenes).some(
                (scene) => scene.isUsingSkyboxTexture(texture))) :
        null;
  }

  private releaseUnusedSkyboxDof() {
    for (const scene of this.scenes) {
      if (scene.hasSkyboxDepth()) {
        return;
      }
    }
    this.skyboxDof?.dispose();
    this.skyboxDof = null;
    this.gpu?.releaseSkyboxDof();
  }

  dispose(): Array<ModelViewerElementBase> {
    this.disposed = true;
    this.gpu?.dispose();
    this.gpu = null;
    this.skyboxDof?.dispose();
    this.skyboxDof = null;
    this.skyboxPanorama?.dispose();
    this.skyboxPanorama = null;
    if (this.textureUtils != null) {
      this.textureUtils.dispose();
    }

    if (this.threeRenderer != null) {
      this.threeRenderer.dispose();
    }

    this.textureUtils = null;
    (this as any).threeRenderer = null;

    const elements = [];
    for (const scene of this.scenes) {
      this.disposeViewHelper(scene);
      elements.push(scene.element);
    }

    this.canvas3D.removeEventListener(
        'webglcontextlost', this.onWebGLContextLost);
    this.canvas3D.removeEventListener(
        'webglcontextrestored', this.onWebGLContextRestored);

    return elements;
  }

  onWebGLContextLost = (event: Event) => {
    this.dispatchEvent(
        {type: 'contextlost', sourceEvent: event} as ContextLostEvent);
  };

  onWebGLContextRestored = () => {
    this.skyboxDof?.dispose();
    this.skyboxDof = null;
    this.textureUtils?.dispose();
    this.textureUtils = this.createTextureUtils();
    for (const scene of this.scenes) {
      (scene.element as any)[$updateEnvironment]();
    }
  };
}
