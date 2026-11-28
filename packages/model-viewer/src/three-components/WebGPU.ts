/// <reference types="@webgpu/types" preserve="true" />
/* @license Copyright 2026 Solunea. Licensed under the Apache License,
 * Version 2.0. */
import {HDRJPGLoader} from '@monogrid/gainmap-js/webgpu';
import {EquirectangularReflectionMapping, Texture} from 'three';
import type {Scene} from 'three';
import {CanvasTarget, DirectRenderPipeline, PMREMGenerator, WebGPURenderer} from 'three/webgpu';

import EnvironmentScene from './EnvironmentScene.js';
import type {ModelScene} from './ModelScene.js';
import {createWebGPUPanoramaMaterial} from './WebGPUPanoramaAreas.js';
import {createWebGPUShadowMaterials} from './WebGPUShadow.js';
import {canRenderSkyboxDirectly, WebGPUSkyboxDof} from './WebGPUSkyboxDof.js';
import {WebGPUSkyboxTransition} from './WebGPUSkyboxTransition.js';

function isNativeWebGPU(renderer: WebGPURenderer): boolean {
  return (renderer.backend as {isWebGPUBackend?: boolean}).isWebGPUBackend === true;
}

export class WebGPUContext {
  readonly shadows = createWebGPUShadowMaterials();
  private skyboxDof: WebGPUSkyboxDof|null = null;
  private panoramaPipeline: DirectRenderPipeline|null = null;
  private panoramaTransition: WebGPUSkyboxTransition|null = null;
  private readonly pmrem: PMREMGenerator;
  private readonly environments = new Map<string, ReturnType<PMREMGenerator['fromScene']>>();
  private readonly canvases =
      new Map<ModelScene, {canvas: HTMLCanvasElement, target: CanvasTarget}>();
  private readonly preparedTextures =
      new WeakMap<Texture, {version: number, ready: Promise<void>}>();
  private readonly device: GPUDevice;

  private readonly onGPUError = () => this.onLost();

  get hasSkyboxDof(): boolean {
    return this.skyboxDof != null;
  }

  private constructor(
      readonly renderer: WebGPURenderer, readonly canvas: HTMLCanvasElement,
      private readonly onLost: () => void) {
    this.pmrem = new PMREMGenerator(renderer);
    this.device = (renderer.backend as unknown as {device: GPUDevice}).device;
    this.device.addEventListener('uncapturederror', this.onGPUError);
  }

  static async create(
      powerPreference: string, onLost: () => void): Promise<WebGPUContext|null> {
    const canvas = document.createElement('canvas');
    const renderer = new WebGPURenderer({
      canvas, alpha: true, antialias: true,
      powerPreference: powerPreference as GPUPowerPreference
    });
    try {
      await renderer.init();
      if (!isNativeWebGPU(renderer)) {
        renderer.dispose();
        return null;
      }
      renderer.setPixelRatio(1);
      renderer.onDeviceLost = onLost;
      const {device} = renderer.backend as unknown as {device: GPUDevice};
      void device.lost.then(onLost);
      return new WebGPUContext(renderer, canvas, onLost);
    } catch (error) {
      renderer.dispose();
      throw error;
    }
  }

  /**
   * External renderers own their presentation canvases. Share the GPU device
   * with those renderers so they follow the same device-loss lifecycle.
   */
  async createRenderer(canvas: HTMLCanvasElement, antialias = true) {
    const renderer = new WebGPURenderer({
      canvas, device: this.device, alpha: true, antialias
    });
    try {
      await renderer.init();
      if (!isNativeWebGPU(renderer)) {
        throw new Error('WebGPU did not initialize on this canvas.');
      }
      renderer.onDeviceLost = this.onLost;
      return renderer;
    } catch (error) {
      renderer.dispose();
      throw error;
    }
  }

  async loadGainMap(
      url: string, withCredentials: boolean, onProgress: (event: ProgressEvent) => void) {
    const loader = new HDRJPGLoader(this.renderer).setWithCredentials(withCredentials);
    const quad = await loader.loadAsync(url, onProgress);
    try {
      const texture = await quad.toDataTexture();
      texture.needsUpdate = true;
      return texture;
    } finally {
      quad.dispose(true);
    }
  }

  async prepareTextures(textures: readonly Texture[]) {
    const uploads: Texture[] = [];
    const pending = new Set<Promise<void>>();
    for (const texture of textures) {
      if (texture.isRenderTargetTexture) continue;
      const prepared = this.preparedTextures.get(texture);
      if (prepared?.version === texture.version) {
        pending.add(prepared.ready);
      } else {
        this.renderer.initTexture(texture);
        uploads.push(texture);
      }
    }
    if (uploads.length > 0) {
      // Warmup and navigation can request the same texture concurrently.
      const ready = this.device.queue.onSubmittedWorkDone();
      for (const texture of uploads) {
        this.preparedTextures.set(texture, {version: texture.version, ready});
      }
      pending.add(ready);
    }
    await Promise.all(pending);
  }

  private environment(texture: Texture|null): Texture|null {
    if (texture == null || !texture.isRenderTargetTexture ||
        (texture.name !== 'legacy' && texture.name !== 'neutral')) {
      return texture;
    }
    const name = texture.name;
    let target = this.environments.get(name);
    if (target == null) {
      const environment = new EnvironmentScene(name);
      try {
        target = this.pmrem.fromScene(environment, 0.04, 0.1, 100);
        this.environments.set(name, target);
      } finally {
        environment.dispose();
      }
    }
    return target.texture;
  }

  withEnvironment(scene: Scene, render: () => void) {
    const environment = scene.environment;
    try {
      scene.environment = this.environment(environment);
      render();
    } finally {
      scene.environment = environment;
    }
  }

  withScene(scene: ModelScene, render: () => void) {
    const draw = () => this.withEnvironment(scene, () => {
      if (scene.panoramaAreas != null) {
        scene.panoramaAreas.withMaterial(createWebGPUPanoramaMaterial, render);
      } else {
        render();
      }
    });
    if (scene.shadow != null) scene.shadow.withWebGPUShadow(this.shadows, draw);
    else draw();
  }

  displayCanvas(scene: ModelScene): HTMLCanvasElement|null {
    return this.canvases.get(scene)?.canvas ?? null;
  }

  withCanvas(
      scene: ModelScene, width: number, height: number, render: () => void) {
    let entry = this.canvases.get(scene);
    if (entry == null) {
      const canvas = document.createElement('canvas');
      canvas.id = 'webgpu-canvas';
      canvas.classList.add('show');
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      entry = {canvas, target: new CanvasTarget(canvas)};
      this.canvases.set(scene, entry);
    }

    const {canvas, target} = entry;
    const previous = this.renderer.getCanvasTarget();
    this.renderer.setCanvasTarget(target);
    try {
      if (canvas.width !== width || canvas.height !== height) {
        target.setSize(width, height, false);
      }
      render();
      if (canvas.parentNode == null) scene.canvas.after(canvas);
    } finally {
      this.renderer.setCanvasTarget(previous);
    }
  }

  releaseCanvas(scene: ModelScene) {
    this.panoramaTransition?.release(scene);
    const entry = this.canvases.get(scene);
    if (entry == null) return;
    this.canvases.delete(scene);
    entry.target.dispose();
    if (entry.canvas.parentNode != null) {
      entry.canvas.getContext('webgpu')?.unconfigure();
      entry.canvas.remove();
    }
  }

  render(scene: ModelScene, width: number, height: number) {
    const {renderer} = this;
    const frame = scene.getSkyboxDofFrame();
    const transition = scene.skyboxTransitionMesh;
    if (frame == null && transition != null &&
        canRenderSkyboxDirectly(scene, renderer, transition)) {
      this.panoramaTransition ??= new WebGPUSkyboxTransition();
      this.panoramaPipeline ??= new DirectRenderPipeline(renderer);
      if (this.panoramaTransition.render(this.panoramaPipeline, scene, transition)) {
        return false;
      }
    }
    this.panoramaTransition?.release(scene);
    if (frame != null) {
      const dof = this.skyboxDof ??= new WebGPUSkyboxDof();
      scene.camera.updateWorldMatrix(true, false);
      return dof.render(renderer, scene, frame, width, height);
    }
    const {background, camera} = scene;
    if (background instanceof Texture &&
        background.mapping === EquirectangularReflectionMapping &&
        !('backgroundNode' in scene && scene.backgroundNode != null) &&
        canRenderSkyboxDirectly(scene, renderer)) {
      this.panoramaPipeline ??= new DirectRenderPipeline(renderer);
      this.panoramaPipeline.render(scene, camera);
    } else {
      renderer.render(scene, camera);
    }
    return false;
  }

  releaseSkyboxDof() {
    this.skyboxDof?.dispose();
    this.skyboxDof = null;
  }

  dispose() {
    this.device.removeEventListener('uncapturederror', this.onGPUError);
    for (const scene of this.canvases.keys()) this.releaseCanvas(scene);
    this.releaseSkyboxDof();
    this.panoramaPipeline?.dispose();
    this.shadows.depth.dispose();
    this.shadows.blur.dispose();
    for (const target of this.environments.values()) target.dispose();
    this.environments.clear();
    this.pmrem.dispose();
    this.renderer.dispose();
  }
}
