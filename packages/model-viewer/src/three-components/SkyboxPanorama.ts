/* @license
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

import {EquirectangularReflectionMapping, Mesh, ShaderMaterial, SphereGeometry, SRGBColorSpace, Texture, WebGLRenderer} from 'three';

import type {ModelScene} from './ModelScene.js';

const vertexShader = /* glsl */ `
varying vec2 panoramaUv;

void main() {
  panoramaUv = uv;
  gl_Position = projectionMatrix * vec4(mat3(viewMatrix) * position, 0.);
  gl_Position.z = gl_Position.w;
}
`;

const fragmentShader = /* glsl */ `
varying vec2 panoramaUv;
uniform sampler2D backgroundMap;
uniform sampler2D transitionMap;
uniform float transitionOpacity;

void main() {
  vec4 backgroundColor = linearToOutputTexel(texture2D(backgroundMap, panoramaUv));
  if (transitionOpacity > 0.) {
    vec4 nextColor = linearToOutputTexel(texture2D(transitionMap, panoramaUv));
    float alpha = nextColor.a * transitionOpacity;
    // Match the original alpha blend in the output color space.
    backgroundColor.rgb = mix(backgroundColor.rgb, nextColor.rgb, alpha);
    backgroundColor.a = mix(backgroundColor.a, 1., alpha);
  }
  gl_FragColor = backgroundColor;
}
`;

function isPanorama(texture: ModelScene['background']): texture is Texture {
  return texture instanceof Texture &&
      texture.mapping === EquirectangularReflectionMapping &&
      texture.colorSpace === SRGBColorSpace && !texture.isRenderTargetTexture &&
      ((texture.image as {height?: number}|null)?.height ?? 0) > 0;
}

/** Draw the skybox directly from the source textures in one pass. */
export class SkyboxPanorama {
  private readonly uniforms = {
    backgroundMap: {value: null as Texture|null},
    transitionMap: {value: null as Texture|null},
    transitionOpacity: {value: 0},
  };
  private readonly mesh: Mesh<SphereGeometry, ShaderMaterial>;

  constructor() {
    const geometry = new SphereGeometry(1, 64, 32);
    geometry.scale(1, 1, -1);
    this.mesh = new Mesh(geometry, new ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader,
      fragmentShader,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    }));
    this.mesh.frustumCulled = false;
    this.mesh.layers.enableAll();
    this.mesh.renderOrder = -Infinity;
  }

  render(renderer: WebGLRenderer, scene: ModelScene): boolean {
    const background = scene.background;
    const transition = scene.skyboxTransitionMesh;
    const rotation = scene.backgroundRotation;
    if (!isPanorama(background) ||
        (transition != null && !isPanorama(transition.material.map)) ||
        scene.backgroundBlurriness !== 0 || scene.backgroundIntensity !== 1 ||
        rotation.x !== 0 || rotation.y !== 0 || rotation.z !== 0 ||
        scene.overrideMaterial != null || scene.fog != null) {
      return false;
    }

    this.uniforms.backgroundMap.value = background;
    this.uniforms.transitionMap.value = transition?.material.map ?? null;
    this.uniforms.transitionOpacity.value =
        transition?.visible ? transition.material.opacity : 0;
    scene.background = null;
    const transitionVisible = transition?.visible ?? false;
    if (transition != null) {
      transition.visible = false;
    }
    scene.add(this.mesh);
    try {
      renderer.render(scene, scene.camera);
    } finally {
      this.mesh.removeFromParent();
      scene.background = background;
      if (transition != null) {
        transition.visible = transitionVisible;
      }
      this.uniforms.backgroundMap.value = null;
      this.uniforms.transitionMap.value = null;
    }
    return true;
  }

  dispose() {
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
  }
}
