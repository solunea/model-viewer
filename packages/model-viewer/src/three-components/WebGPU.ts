/// <reference types="@webgpu/types" preserve="true" />
/* @license Copyright 2026 Solunea. Licensed under the Apache License,
 * Version 2.0. */
import {HDRJPGLoader} from '@monogrid/gainmap-js/webgpu';
import type {Scene, Texture} from 'three';
import {PMREMGenerator, WebGPURenderer} from 'three/webgpu';

import EnvironmentScene from './EnvironmentScene.js';
import type {ModelScene} from './ModelScene.js';
import {createWebGPUPanoramaMaterial} from './WebGPUPanoramaAreas.js';
import {createWebGPUShadowMaterials} from './WebGPUShadow.js';
import {WebGPUSkyboxDof} from './WebGPUSkyboxDof.js';

function isNativeWebGPU(renderer: WebGPURenderer): boolean {
  return (renderer.backend as {isWebGPUBackend?: boolean}).isWebGPUBackend === true;
}

export class WebGPUContext {
  readonly shadows = createWebGPUShadowMaterials();
  private skyboxDof: WebGPUSkyboxDof|null = null;
  private readonly pmrem: PMREMGenerator;
  private readonly environments = new Map<string, ReturnType<PMREMGenerator['fromScene']>>();
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

  render(scene: ModelScene, width: number, height: number) {
    const frame = scene.getSkyboxDofFrame();
    if (frame != null) {
      const dof = this.skyboxDof ??= new WebGPUSkyboxDof();
      scene.camera.updateWorldMatrix(true, false);
      dof.render(this.renderer, scene, frame, width, height);
    } else {
      this.renderer.render(scene, scene.camera);
    }
  }

  releaseSkyboxDof() {
    this.skyboxDof?.dispose();
    this.skyboxDof = null;
  }

  dispose() {
    this.device.removeEventListener('uncapturederror', this.onGPUError);
    this.releaseSkyboxDof();
    this.shadows.depth.dispose();
    this.shadows.blur.dispose();
    for (const target of this.environments.values()) target.dispose();
    this.environments.clear();
    this.pmrem.dispose();
    this.renderer.dispose();
  }
}
