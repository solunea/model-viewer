/* @license Copyright 2026 Solunea. Licensed under the Apache License,
 * Version 2.0. */
import {EquirectangularReflectionMapping, Matrix4, SRGBColorSpace, Texture} from 'three';
import type {Mesh} from 'three';
import {cameraPosition, float, mix, positionWorld, texture, uniform, varying, vec4} from 'three/tsl';
import {CubeMapNode, NodeMaterial} from 'three/webgpu';
import type {DirectRenderPipeline, Node} from 'three/webgpu';

import type {ModelScene} from './ModelScene.js';

const IDENTITY = new Matrix4();

interface TransitionMaterial {
  background: Texture;
  incoming: Texture;
  material: NodeMaterial;
}

function isPanorama(value: ModelScene['background']): value is Texture {
  return value instanceof Texture && !value.isRenderTargetTexture &&
      !value.isPMREMTexture && value.mapping === EquirectangularReflectionMapping &&
      value.colorSpace === SRGBColorSpace &&
      ((value.image as {height?: number}|null)?.height ?? 0) > 0;
}

/** Blend the existing transition sphere with the skybox before output conversion. */
export class WebGPUSkyboxTransition {
  private readonly materials = new Map<ModelScene, TransitionMaterial>();
  private readonly rotation = uniform(new Matrix4());
  private readonly intensity = uniform(1);
  private readonly opacity = uniform(0);

  render(
      pipeline: DirectRenderPipeline, scene: ModelScene,
      transition: NonNullable<ModelScene['skyboxTransitionMesh']>) {
    const {background, camera} = scene;
    const source = transition.material;
    const incoming = source.map;
    if (!isPanorama(background) || !isPanorama(incoming) ||
        scene.backgroundBlurriness !== 0 ||
        ('backgroundNode' in scene && scene.backgroundNode != null) ||
        scene.xrCamera != null || scene.parent != null || camera.parent != null ||
        camera.view?.enabled || camera.filmOffset !== 0 ||
        !transition.visible || !source.visible ||
        !transition.layers.test(camera.layers)) {
      return false;
    }
    // When the existing sphere can be clipped, the scene background must
    // still cover the pixels outside of it.
    scene.updateWorldMatrix(false, false);
    const radius = transition.scale.x;
    const halfHeight = Math.tan(camera.getEffectiveFOV() * Math.PI / 360);
    const corner = Math.hypot(1, halfHeight, halfHeight * camera.aspect);
    if (!scene.matrixWorld.equals(IDENTITY) || !camera.position.equals(transition.position) ||
        camera.near * corner >= radius * .98 || camera.far <= radius) {
      return false;
    }

    let entry = this.materials.get(scene);
    if (entry?.background !== background || entry.incoming !== incoming) {
      this.release(scene);
      const material = new NodeMaterial();
      material.setValues({depthTest: false, depthWrite: false, fog: false});
      const cube = new CubeMapNode(texture(background)) as unknown as Node<'vec4'>;
      const ray = varying(this.rotation.mul(vec4(positionWorld.sub(cameraPosition), 0)).xyz);
      const from = vec4(cube).context({getUV: () => ray, getTextureLevel: () => float(0)});
      const to = texture(incoming);
      material.colorNode = vec4(
          mix(from.rgb.mul(this.intensity), to.rgb, to.a.mul(this.opacity)), 1);
      entry = {background, incoming, material};
      this.materials.set(scene, entry);
    }
    this.rotation.value.makeRotationFromEuler(scene.backgroundRotation).transpose();
    this.intensity.value = scene.backgroundIntensity;
    this.opacity.value = source.opacity;
    const mesh: Mesh = transition;
    mesh.material = entry.material;
    try {
      pipeline.render(transition, camera);
    } finally {
      mesh.material = source;
    }
    return true;
  }

  release(scene: ModelScene) {
    const entry = this.materials.get(scene);
    if (entry == null) return;
    entry.material.dispose();
    this.materials.delete(scene);
  }
}
