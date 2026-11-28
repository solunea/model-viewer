/* @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {BackSide, BasicShadowMap, Box3, DirectionalLight, DoubleSide, Material, Mesh, MeshBasicMaterial, MeshDepthMaterial, Object3D, OrthographicCamera, PCFShadowMap, PlaneGeometry, RGBAFormat, Scene, ShaderChunk, ShaderMaterial, ShadowMaterial, Texture, Vector3, WebGLRenderer, WebGLRenderTarget} from 'three';
import type {WebGPURenderer} from 'three/webgpu';

import {Damper} from './Damper.js';
import {ModelScene} from './ModelScene.js';
import {pcssBlockerOffsets, pcssDepthEpsilon} from './PCSS.js';

export type Side = 'back'|'bottom';
export type ShadowMode = 'basic'|'pcss';

export interface WebGPUShadowMaterials {
  depth: Material;
  blur: Material;
  setDepth(camera: OrthographicCamera, opacity: number): void;
  setBlur(source: Texture, x: number, y: number): void;
  setPCSS(
      light: DirectionalLight, scale: number, maxInverseDepth: number): void;
  withPCSSShadow(light: DirectionalLight, render: () => void): void;
}

const DEFAULT_SHADOW_THETA = 0;
const DEFAULT_SHADOW_PHI = 0;

// Softness maps to resolution between 2^LOG_MAX and 2^LOG_MIN
const LOG_MAX_RESOLUTION = 9;
const LOG_MIN_RESOLUTION = 6;
const DEFAULT_HARD_INTENSITY = 0.3;
const WEBGPU_ANIMATED_SHADOW_INTERVAL_MS = 1000 / 30;

// Fit the projection around the casters with room for the penumbra.
const PCSS_FRUSTUM_SCALE = 4;
const PCSS_MAP_SIZE = 2048;

// ─── Blur shaders (compatible with Three.js r183) ───────────────────
const BLUR_VERTEX = `varying vec2 vUv;
void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

const HORIZONTAL_BLUR_FRAGMENT = `uniform sampler2D tDiffuse; uniform float h; varying vec2 vUv;
void main(){
  vec4 sum = vec4(0.0);
  sum += texture2D(tDiffuse, vec2(vUv.x - 4.0*h, vUv.y)) * 0.051;
  sum += texture2D(tDiffuse, vec2(vUv.x - 3.0*h, vUv.y)) * 0.0918;
  sum += texture2D(tDiffuse, vec2(vUv.x - 2.0*h, vUv.y)) * 0.12245;
  sum += texture2D(tDiffuse, vec2(vUv.x - 1.0*h, vUv.y)) * 0.1531;
  sum += texture2D(tDiffuse, vec2(vUv.x, vUv.y)) * 0.1633;
  sum += texture2D(tDiffuse, vec2(vUv.x + 1.0*h, vUv.y)) * 0.1531;
  sum += texture2D(tDiffuse, vec2(vUv.x + 2.0*h, vUv.y)) * 0.12245;
  sum += texture2D(tDiffuse, vec2(vUv.x + 3.0*h, vUv.y)) * 0.0918;
  sum += texture2D(tDiffuse, vec2(vUv.x + 4.0*h, vUv.y)) * 0.051;
  gl_FragColor = sum;
}`;

const VERTICAL_BLUR_FRAGMENT = `uniform sampler2D tDiffuse; uniform float v; varying vec2 vUv;
void main(){
  vec4 sum = vec4(0.0);
  sum += texture2D(tDiffuse, vec2(vUv.x, vUv.y - 4.0*v)) * 0.051;
  sum += texture2D(tDiffuse, vec2(vUv.x, vUv.y - 3.0*v)) * 0.0918;
  sum += texture2D(tDiffuse, vec2(vUv.x, vUv.y - 2.0*v)) * 0.12245;
  sum += texture2D(tDiffuse, vec2(vUv.x, vUv.y - 1.0*v)) * 0.1531;
  sum += texture2D(tDiffuse, vec2(vUv.x, vUv.y)) * 0.1633;
  sum += texture2D(tDiffuse, vec2(vUv.x, vUv.y + 1.0*v)) * 0.1531;
  sum += texture2D(tDiffuse, vec2(vUv.x, vUv.y + 2.0*v)) * 0.12245;
  sum += texture2D(tDiffuse, vec2(vUv.x, vUv.y + 3.0*v)) * 0.0918;
  sum += texture2D(tDiffuse, vec2(vUv.x, vUv.y + 4.0*v)) * 0.051;
  gl_FragColor = sum;
}`;

// Keep the shader local to the floor material so that viewers can share a
// renderer and vary their softness independently without recompiling.
const glslOffsets = (offsets: ReadonlyArray<readonly [number, number]>) =>
    offsets.map(([x, y]) => `vec2(${x.toFixed(2)}, ${y.toFixed(2)})`).join(', ');

const PCSS_SHADOW_MASK = `
#if defined(USE_SHADOWMAP) && defined(SHADOWMAP_TYPE_BASIC) && NUM_DIR_LIGHT_SHADOWS > 0
uniform float pcssScale;
uniform float pcssMaxInverseDepth;
const vec2 pcssBlockerOffsets[${pcssBlockerOffsets.length}] =
    vec2[](${glslOffsets(pcssBlockerOffsets)});

float pcssDepth(sampler2D shadowMap, vec2 uv) {
  float depth = texture2D(shadowMap, uv).r;
  #ifdef USE_REVERSED_DEPTH_BUFFER
    depth = 1.0 - depth;
  #endif
  return depth;
}

// Interpolate the comparison results, rather than interpolating the depths.
float pcssCompare(sampler2D shadowMap, vec2 uv, float z, vec2 size, vec2 texel) {
  vec2 samplePoint = uv * size - 0.5;
  vec2 base = (floor(samplePoint) + 0.5) * texel;
  vec2 weight = fract(samplePoint);
  float a = step(z, pcssDepth(shadowMap, base));
  float b = step(z, pcssDepth(shadowMap, base + vec2(texel.x, 0.0)));
  float c = step(z, pcssDepth(shadowMap, base + vec2(0.0, texel.y)));
  float d = step(z, pcssDepth(shadowMap, base + texel));
  return mix(mix(a, b, weight.x), mix(c, d, weight.x), weight.y);
}

// Reconstruct coverage between samples on a fixed lattice so a broad penumbra
// has a continuous gradient even when its samples are several texels apart.
float pcssFilter(sampler2D shadowMap, vec2 point, float z, vec2 texel, float spacing) {
  float origin = floor(0.5 * spacing);
  vec2 grid = (point - origin) / spacing;
  vec2 cell = floor(grid + 0.5);
  vec2 offset = grid - cell;
  vec2 low = 0.5 - offset;
  vec2 high = 0.5 + offset;
  vec2 w0 = 0.5 * low * low;
  vec2 w1 = 0.75 - offset * offset;
  vec2 w2 = 0.5 * high * high;
  vec3 weightX = vec3(w0.x, w1.x, w2.x);
  vec3 weightY = vec3(w0.y, w1.y, w2.y);
  vec2 center = (cell * spacing + origin + 0.5) * texel;
  vec2 stepUV = spacing * texel;
  float visibility = 0.0;
  for (int y = 0; y < 3; ++y) {
    for (int x = 0; x < 3; ++x) {
      vec2 sampleUV = center + (vec2(x, y) - 1.0) * stepUV;
      visibility += weightX[x] * weightY[y] * step(z, pcssDepth(shadowMap, sampleUV));
    }
  }
  return visibility;
}

float pcssVisibility(sampler2D shadowMap, vec3 coord, vec2 size) {
  vec2 uv = coord.xy;
  float z = coord.z;
  if (z <= 0.0 || z > 1.0 || any(lessThan(uv, vec2(0))) || any(greaterThan(uv, vec2(1)))) {
    return 1.0;
  }

  vec2 texel = 1.0 / size;
  float texelWidth = max(texel.x, texel.y);
  float inverseReceiver = 1.0 / max(z, ${pcssDepthEpsilon});
  float searchRadius = pcssScale * (pcssMaxInverseDepth - inverseReceiver);
  if (searchRadius <= texelWidth) {
    return pcssCompare(shadowMap, uv, z, size, texel);
  }

  float blockerSum = 0.0;
  float blockerCount = 0.0;
  for (int i = 0; i < ${pcssBlockerOffsets.length}; ++i) {
    float depth = pcssDepth(shadowMap, uv + pcssBlockerOffsets[i] * searchRadius);
    float blocked = 1.0 - step(z, depth);
    blockerSum += blocked * depth;
    blockerCount += blocked;
  }
  if (blockerCount == 0.0) {
    return 1.0;
  }

  float inverseBlocker = blockerCount / max(blockerSum, ${pcssDepthEpsilon} * blockerCount);
  float radius = pcssScale * (inverseBlocker - inverseReceiver);
  if (radius <= texelWidth) {
    return pcssCompare(shadowMap, uv, z, size, texel);
  }

  // The filter variance grows with the square of the spacing. Interpolating
  // that variance gives a smooth radius change while favoring the denser grid.
  float radiusInTexels = radius / texelWidth;
  float spacing = exp2(floor(log2(radiusInTexels)));
  float ratio = radiusInTexels / spacing;
  float blend = (ratio * ratio - 1.0) / 3.0;
  vec2 point = uv * size - 0.5;
  float fine = spacing == 1.0 ? pcssCompare(shadowMap, uv, z, size, texel) :
      pcssFilter(shadowMap, point, z, texel, spacing);
  float coarse = pcssFilter(shadowMap, point, z, texel, spacing * 2.0);
  return mix(fine, coarse, blend);
}

float modelViewerPCSSShadow(
    sampler2D shadowMap, vec2 size, float intensity, float bias, float radius, vec4 coord) {
  vec3 projected = coord.xyz / coord.w;
  #ifdef USE_REVERSED_DEPTH_BUFFER
    projected.z = 1.0 - projected.z;
  #endif
  projected.z += bias;
  return mix(1.0, pcssVisibility(shadowMap, projected, size), intensity);
}

${ShaderChunk.shadowmask_pars_fragment.replace(
    'getShadow( directionalShadowMap', 'modelViewerPCSSShadow( directionalShadowMap')}
#else
#include <shadowmask_pars_fragment>
#endif
`;

const _center = new Vector3();
const _lightDirection = new Vector3();

/**
 * Dual-mode shadow:
 * - 'basic' (default): Orthographic camera + depth + gaussian blur. Very fast.
 * - 'pcss': DirectionalLight + PCSS shader. Higher quality, supports shadow-orbit.
 *
 * Mode is auto-selected: basic when orbit is (0,0), pcss when orbit is non-zero.
 */
export class Shadow extends Object3D {
  private lastRenderer: WebGLRenderer|WebGPURenderer|null = null;
  private lastRenderTime = -Infinity;
  private mode: ShadowMode = 'basic';

  // ─── Shared state ───
  private floor!: Mesh;
  private intensity = 0;
  private softness = 0;
  private boundingBox = new Box3();
  private size = new Vector3();
  private maxDimension = 0;
  private side: Side = 'bottom';
  private theta = DEFAULT_SHADOW_THETA;
  private phi = DEFAULT_SHADOW_PHI;
  private goalTheta = DEFAULT_SHADOW_THETA;
  private goalPhi = DEFAULT_SHADOW_PHI;
  private interpolationDecay = 0;
  private thetaDamper = new Damper();
  private phiDamper = new Damper();
  public needsUpdate = false;
  public animationNeedsUpdate = false;

  // ─── Basic mode state ───
  private basicCamera: OrthographicCamera|null = null;
  private renderTarget: WebGLRenderTarget|null = null;
  private renderTargetBlur: WebGLRenderTarget|null = null;
  private depthMaterial: MeshDepthMaterial|null = null;
  private horizontalBlurMaterial: ShaderMaterial|null = null;
  private verticalBlurMaterial: ShaderMaterial|null = null;
  private blurPlane: Mesh|null = null;

  // ─── PCSS mode state ───
  private light: DirectionalLight|null = null;
  private readonly pcssScale = {value: 0};
  private readonly pcssMaxInverseDepth = {value: 0};
  private castShadowSet = false;

  constructor(scene: ModelScene, softness: number, side: Side) {
    super();
    this.initBasicMode();
    scene.target.add(this);
    this.setScene(scene, softness, side);
  }

  // ─── Mode initialization ───

  private initBasicMode() {
    this.disposeMode();
    this.mode = 'basic';

    this.basicCamera = new OrthographicCamera(-0.5, 0.5, 0.5, -0.5, 0, 1);
    this.basicCamera.rotation.x = Math.PI / 2;
    this.basicCamera.updateProjectionMatrix();
    this.add(this.basicCamera);

    const plane = new PlaneGeometry();
    const floorMat = new MeshBasicMaterial({
      opacity: 1,
      transparent: true,
      side: BackSide,
    });
    this.floor = new Mesh(plane, floorMat);
    this.floor.userData.noHit = true;
    this.floor.name = 'ShadowFloor';
    this.basicCamera.add(this.floor);

    this.blurPlane = new Mesh(plane);
    this.blurPlane.visible = false;
    this.basicCamera.add(this.blurPlane);

    this.depthMaterial = new MeshDepthMaterial();
    this.depthMaterial.onBeforeCompile = function(shader) {
      if (shader && typeof shader.fragmentShader === 'string') {
        shader.fragmentShader = shader.fragmentShader.replace(
            /gl_FragColor\s*=\s*vec4\(\s*vec3\(\s*1\.0\s*-\s*fragCoordZ\s*\),\s*opacity\s*\);/,
            'gl_FragColor = vec4( vec3( 0.0 ), ( 1.0 - fragCoordZ ) * opacity );');
      }
    };
    this.depthMaterial.side = DoubleSide;

    this.horizontalBlurMaterial = new ShaderMaterial({
      uniforms: {tDiffuse: {value: null}, h: {value: 1.0 / 512.0}},
      vertexShader: BLUR_VERTEX,
      fragmentShader: HORIZONTAL_BLUR_FRAGMENT,
    });
    this.horizontalBlurMaterial.depthTest = false;
    this.horizontalBlurMaterial.depthWrite = false;

    this.verticalBlurMaterial = new ShaderMaterial({
      uniforms: {tDiffuse: {value: null}, v: {value: 1.0 / 512.0}},
      vertexShader: BLUR_VERTEX,
      fragmentShader: VERTICAL_BLUR_FRAGMENT,
    });
    this.verticalBlurMaterial.depthTest = false;
    this.verticalBlurMaterial.depthWrite = false;
  }

  private initPCSSMode() {
    this.disposeMode();
    this.mode = 'pcss';
    this.position.set(0, 0, 0);
    this.rotation.set(0, 0, 0);

    this.light = new DirectionalLight(0xffffff, 2);
    this.light.castShadow = true;
    this.light.shadow.camera.near = 0.25;
    this.light.shadow.camera.far = 100;
    this.light.shadow.bias = 0.001;
    this.light.shadow.normalBias = 0.01;
    this.light.shadow.mapSize.setScalar(PCSS_MAP_SIZE);
    this.light.name = 'ShadowLight';

    const plane = new PlaneGeometry(1, 1);
    const shadowMat = new ShadowMaterial({opacity: 0, transparent: true});
    shadowMat.onBeforeCompile = (shader) => {
      shader.uniforms.pcssScale = this.pcssScale;
      shader.uniforms.pcssMaxInverseDepth = this.pcssMaxInverseDepth;
      shader.fragmentShader = shader.fragmentShader.replace(
          '#include <shadowmask_pars_fragment>', PCSS_SHADOW_MASK);
    };
    this.floor = new Mesh(plane, shadowMat);
    this.floor.receiveShadow = true;
    this.floor.userData.noHit = true;
    this.floor.name = 'ShadowFloor';

    this.add(this.light);
    this.add(this.light.target);
    this.add(this.floor);

    this.castShadowSet = false;
  }

  private disposeMode() {
    // Dispose basic mode resources
    if (this.basicCamera != null) {
      this.basicCamera.removeFromParent();
      this.basicCamera = null;
    }
    if (this.renderTarget != null) {
      this.renderTarget.dispose();
      this.renderTarget = null;
    }
    if (this.renderTargetBlur != null) {
      this.renderTargetBlur.dispose();
      this.renderTargetBlur = null;
    }
    if (this.depthMaterial != null) {
      this.depthMaterial.dispose();
      this.depthMaterial = null;
    }
    if (this.horizontalBlurMaterial != null) {
      this.horizontalBlurMaterial.dispose();
      this.horizontalBlurMaterial = null;
    }
    if (this.verticalBlurMaterial != null) {
      this.verticalBlurMaterial.dispose();
      this.verticalBlurMaterial = null;
    }
    if (this.blurPlane != null) {
      this.blurPlane = null;
    }

    // Dispose PCSS mode resources
    if (this.light != null) {
      this.light.shadow.map?.dispose();
      this.light.target.removeFromParent();
      this.light.removeFromParent();
      this.light = null;
    }

    // Dispose floor
    if (this.floor != null) {
      (this.floor.material as any).dispose?.();
      this.floor.geometry.dispose();
      this.floor.removeFromParent();
    }
  }

  // ─── Public API (shared) ───

  setScene(scene: ModelScene, softness: number, side: Side) {
    this.side = side;
    this.boundingBox.copy(scene.boundingBox);
    this.boundingBox.getSize(this.size);
    this.maxDimension = Math.max(this.size.x, this.size.y, this.size.z);

    if (this.mode === 'basic') {
      this.setupBasicScene(side);
    } else {
      this.setupPCSSScene(side);
    }

    this.setSoftness(softness);
    this.needsUpdate = true;
  }

  setOrbit(theta: number, phi: number) {
    const isFlatOrbit = (phi === 0);
    const needMode = isFlatOrbit ? 'basic' : 'pcss';

    if (needMode === 'pcss' && this.mode !== 'pcss') {
      const savedSoftness = this.softness;
      const savedIntensity = this.intensity;
      this.initPCSSMode();
      this.setupPCSSScene(this.side);
      this.setSoftness(savedSoftness);
      this.setIntensity(savedIntensity);
    }

    this.goalTheta = theta;
    this.goalPhi = phi;
    if (this.interpolationDecay <= 0) {
      this.theta = theta;
      this.phi = phi;
      if (needMode !== this.mode) {
        const savedSoftness = this.softness;
        const savedIntensity = this.intensity;
        this.initBasicMode();
        this.setupBasicScene(this.side);
        this.setSoftness(savedSoftness);
        this.setIntensity(savedIntensity);
      } else if (this.mode === 'pcss') {
        this.updatePCSSLightPosition();
      }
    }
    this.needsUpdate = true;
  }

  update(delta: number): boolean {
    if (this.mode === 'basic') return false;
    if (this.theta === this.goalTheta && this.phi === this.goalPhi) {
      return false;
    }

    if (this.interpolationDecay <= 0) {
      this.theta = this.goalTheta;
      this.phi = this.goalPhi;
    } else {
      this.theta = this.thetaDamper.update(
          this.theta, this.goalTheta, delta, Math.PI * 2);
      this.phi =
          this.phiDamper.update(this.phi, this.goalPhi, delta, Math.PI);
    }

    if (this.phi === 0 && this.goalPhi === 0) {
      const savedSoftness = this.softness;
      const savedIntensity = this.intensity;
      this.initBasicMode();
      this.setupBasicScene(this.side);
      this.setSoftness(savedSoftness);
      this.setIntensity(savedIntensity);
    } else {
      this.updatePCSSLightPosition();
    }
    this.needsUpdate = true;
    return true;
  }

  setInterpolationDecay(decayMilliseconds: number) {
    this.interpolationDecay = decayMilliseconds;
    this.thetaDamper.setDecayTime(decayMilliseconds);
    this.phiDamper.setDecayTime(decayMilliseconds);
    if (decayMilliseconds <= 0 && this.mode === 'pcss') {
      this.theta = this.goalTheta;
      this.phi = this.goalPhi;
      if (this.phi === 0) {
        const savedSoftness = this.softness;
        const savedIntensity = this.intensity;
        this.initBasicMode();
        this.setupBasicScene(this.side);
        this.setSoftness(savedSoftness);
        this.setIntensity(savedIntensity);
      } else {
        this.updatePCSSLightPosition();
      }
      this.needsUpdate = true;
    }
  }

  setSoftness(softness: number) {
    this.softness = softness;
    if (this.mode === 'basic') {
      this.updateBasicSoftness();
    } else if (this.light != null) {
      const near = this.light.shadow.camera.near;
      this.pcssScale.value = Math.max(0, softness) * 0.05 * near / PCSS_FRUSTUM_SCALE;
    }
    this.needsUpdate = true;
  }

  setIntensity(intensity: number) {
    this.intensity = intensity;
    if (this.mode === 'basic') {
      if (intensity > 0) {
        this.visible = true;
        this.floor.visible = true;
        const lerped = DEFAULT_HARD_INTENSITY + (1 - DEFAULT_HARD_INTENSITY) * this.softness * this.softness;
        (this.floor.material as MeshBasicMaterial).opacity = intensity * lerped;
      } else {
        this.visible = false;
        this.floor.visible = false;
      }
    } else {
      const mat = this.floor.material as ShadowMaterial;
      if (intensity > 0) {
        this.visible = true;
        this.floor.visible = true;
        mat.opacity = intensity;
      } else {
        this.visible = false;
        this.floor.visible = false;
        mat.opacity = 0;
      }
    }
  }

  getIntensity(): number {
    return this.intensity;
  }

  setOffset(offset: number) {
    if (this.mode === 'basic') {
      this.floor.position.z = -offset + this.gap();
    } else {
      const normal = this.side === 'bottom' ? 'y' : 'z';
      this.floor.position[normal] =
          this.boundingBox.min[normal] - offset + this.gap();
      this.updatePCSSReceiver();
    }
  }

  gap() {
    return 0.001 * this.maxDimension;
  }

  render(
      renderer: WebGLRenderer|WebGPURenderer, scene: Scene,
      gpu?: WebGPUShadowMaterials, frameTime?: number) {
    const elapsed = (frameTime ?? Infinity) - this.lastRenderTime;
    // Allow for rounding of consecutive display timestamps at 60 Hz.
    const deferAnimation = this.mode === 'basic' && elapsed >= 0 &&
        elapsed < WEBGPU_ANIMATED_SHADOW_INTERVAL_MS - 0.5;
    if (!this.needsUpdate && this.lastRenderer === renderer &&
        (!this.animationNeedsUpdate || deferAnimation)) {
      return;
    }
    if (this.mode === 'basic') {
      this.renderBasic(renderer, scene, gpu);
    } else {
      this.renderPCSS(renderer, scene, gpu);
    }
    this.lastRenderer = renderer;
    this.lastRenderTime = frameTime ?? -Infinity;
    this.needsUpdate = false;
    this.animationNeedsUpdate = false;
  }

  invalidateCastShadow() {
    this.castShadowSet = false;
  }

  dispose() {
    this.disposeMode();
    this.removeFromParent();
  }

  withWebGPUShadow(materials: WebGPUShadowMaterials, render: () => void) {
    if (this.light == null) render();
    else materials.withPCSSShadow(this.light, render);
  }

  // ─── Basic mode internals ───

  private setupBasicScene(side: Side) {
    const {boundingBox, size, position} = this;
    boundingBox.getCenter(position);

    if (side === 'back') {
      const {min, max} = boundingBox;
      [min.y, min.z] = [min.z, min.y];
      [max.y, max.z] = [max.z, max.y];
      [size.y, size.z] = [size.z, size.y];
      this.rotation.set(Math.PI / 2, Math.PI, 0);
    } else {
      this.rotation.set(0, 0, 0);
    }

    if (side === 'bottom') {
      position.y = boundingBox.min.y;
    } else {
      position.z = boundingBox.min.y;
    }
  }

  private updateBasicSoftness() {
    if (this.basicCamera == null) return;
    const {size} = this;

    const resolution = Math.pow(2,
        LOG_MAX_RESOLUTION - this.softness * (LOG_MAX_RESOLUTION - LOG_MIN_RESOLUTION));
    this.setMapSize(resolution);

    const softFar = size.y / 2;
    const hardFar = size.y;
    this.basicCamera.near = 0;
    this.basicCamera.far = hardFar + (softFar - hardFar) * this.softness;
    if (this.depthMaterial != null && this.softness > 0) {
      this.depthMaterial.opacity = 1.0 / this.softness;
    }
    this.basicCamera.updateProjectionMatrix();
    this.setIntensity(this.intensity);
    this.setOffset(0);
  }

  private setMapSize(maxMapSize: number) {
    const {size} = this;
    const baseWidth = Math.floor(size.x > size.z ? maxMapSize : maxMapSize * size.x / size.z);
    const baseHeight = Math.floor(size.x > size.z ? maxMapSize * size.z / size.x : maxMapSize);
    const TAP_WIDTH = 10;
    const width = TAP_WIDTH + baseWidth;
    const height = TAP_WIDTH + baseHeight;

    if (this.renderTarget != null &&
        (this.renderTarget.width !== width || this.renderTarget.height !== height)) {
      this.renderTarget.dispose();
      this.renderTarget = null;
      this.renderTargetBlur!.dispose();
      this.renderTargetBlur = null;
    }

    if (this.renderTarget == null) {
      const params = {format: RGBAFormat};
      this.renderTarget = new WebGLRenderTarget(width, height, params);
      this.renderTargetBlur = new WebGLRenderTarget(width, height, params);
      (this.floor.material as MeshBasicMaterial).map = this.renderTarget.texture;
    }

    if (this.basicCamera != null) {
      const scaleW = size.x * (1 + TAP_WIDTH / baseWidth);
      const scaleH = size.z * (1 + TAP_WIDTH / baseHeight);
      this.basicCamera.left = -scaleW / 2;
      this.basicCamera.right = scaleW / 2;
      this.basicCamera.top = scaleH / 2;
      this.basicCamera.bottom = -scaleH / 2;
      this.basicCamera.updateProjectionMatrix();
      if (this.floor != null) this.floor.scale.set(scaleW, scaleH, 1);
      if (this.blurPlane != null) this.blurPlane.scale.set(scaleW, scaleH, 1);
    }
    this.needsUpdate = true;
  }

  private renderBasic(
      renderer: WebGLRenderer|WebGPURenderer, scene: Scene,
      gpu?: WebGPUShadowMaterials) {
    if (this.basicCamera == null || this.renderTarget == null) return;

    const xrEnabled = renderer.xr.enabled;
    const oldRenderTarget = renderer.getRenderTarget();
    const initialClearAlpha = renderer.getClearAlpha();
    const initialAutoClear = renderer.autoClear;
    const background = scene.background;
    const override = scene.overrideMaterial;
    const floorVisible = this.floor.visible;
    try {
      renderer.xr.enabled = false;
      renderer.setRenderTarget(this.renderTarget);
      renderer.setClearAlpha(0);
      if (gpu != null) {
        gpu.setDepth(this.basicCamera, this.depthMaterial!.opacity);
        scene.background = null;
        // Clear the attachment in the depth pass instead of submitting a
        // separate pass to the WebGPU queue.
        renderer.autoClear = true;
      } else {
        const state = (renderer as WebGLRenderer).state;
        state?.buffers.color.setMask(true);
        state?.buffers.depth.setMask(true);
        renderer.clear();
        renderer.autoClear = false;
      }
      scene.overrideMaterial = gpu?.depth ?? this.depthMaterial;
      this.floor.visible = false;
      renderer.render(scene, this.basicCamera);
      renderer.autoClear = initialAutoClear;
      this.blurShadow(renderer, gpu);
    } finally {
      scene.background = background;
      scene.overrideMaterial = override;
      this.floor.visible = floorVisible;
      renderer.autoClear = initialAutoClear;
      renderer.xr.enabled = xrEnabled;
      renderer.setRenderTarget(oldRenderTarget as WebGLRenderTarget|null);
      renderer.setClearAlpha(initialClearAlpha);
    }
  }

  private blurShadow(
      renderer: WebGLRenderer|WebGPURenderer, gpu?: WebGPUShadowMaterials) {
    if (this.basicCamera == null || this.blurPlane == null ||
        this.horizontalBlurMaterial == null || this.verticalBlurMaterial == null ||
        this.renderTarget == null || this.renderTargetBlur == null) return;

    this.blurPlane.visible = true;
    if (gpu != null) {
      try {
        this.blurPlane.material = gpu.blur;
        gpu.setBlur(this.renderTarget.texture, 1 / this.renderTarget.width, 0);
        renderer.setRenderTarget(this.renderTargetBlur);
        renderer.render(this.blurPlane, this.basicCamera);
        gpu.setBlur(this.renderTargetBlur.texture, 0, 1 / this.renderTarget.height);
        renderer.setRenderTarget(this.renderTarget);
        renderer.render(this.blurPlane, this.basicCamera);
      } finally {
        this.blurPlane.visible = false;
        this.blurPlane.material = this.horizontalBlurMaterial;
      }
      return;
    }

    this.blurPlane.material = this.horizontalBlurMaterial;
    this.horizontalBlurMaterial.uniforms.h.value = 1 / this.renderTarget.width;
    this.horizontalBlurMaterial.uniforms.tDiffuse.value = this.renderTarget.texture;
    renderer.setRenderTarget(this.renderTargetBlur);
    renderer.render(this.blurPlane, this.basicCamera);

    this.blurPlane.material = this.verticalBlurMaterial;
    this.verticalBlurMaterial.uniforms.v.value = 1 / this.renderTarget.height;
    this.verticalBlurMaterial.uniforms.tDiffuse.value = this.renderTargetBlur.texture;
    renderer.setRenderTarget(this.renderTarget);
    renderer.render(this.blurPlane, this.basicCamera);

    this.blurPlane.visible = false;
  }

  // ─── PCSS mode internals ───

  private setupPCSSScene(side: Side) {
    if (this.light == null) return;
    const bottom = side === 'bottom';
    this.floor.rotation.x = bottom ? -Math.PI / 2 : 0;
    const normal = bottom ? 'y' : 'z';
    this.floor.position[normal] = this.boundingBox.min[normal];
    this.updatePCSSLightPosition();
  }

  private updatePCSSLightPosition() {
    if (this.light == null) return;
    this.boundingBox.getCenter(_center);

    const radius = this.maxDimension * 2 + 1;
    const sinPhi = Math.sin(this.phi);
    const lx = _center.x + radius * sinPhi * Math.sin(this.theta);
    const ly = _center.y + radius * Math.cos(this.phi);
    const lz = _center.z + radius * sinPhi * Math.cos(this.theta);

    this.light.position.set(lx, ly, lz);
    this.light.target.position.copy(_center);
    this.light.updateMatrixWorld();
    this.light.target.updateMatrixWorld();

    const halfSize = this.maxDimension * PCSS_FRUSTUM_SCALE / 2;
    const camera = this.light.shadow.camera;
    camera.left = -halfSize;
    camera.right = halfSize;
    camera.top = halfSize;
    camera.bottom = -halfSize;
    camera.far = radius * 20;
    camera.updateProjectionMatrix();
    this.light.shadow.needsUpdate = true;

    // Bound the blocker search by the closest possible caster inside the model
    // bounds. This keeps the sparse search close to the actual penumbra.
    const nearestDepth =
        (radius - this.size.length() / 2 - camera.near) / (camera.far - camera.near);
    this.pcssMaxInverseDepth.value = 1 / Math.max(nearestDepth, pcssDepthEpsilon);
    this.updatePCSSReceiver();
  }

  private updatePCSSReceiver() {
    if (this.light == null) return;
    const bottom = this.side === 'bottom';
    const normal = bottom ? 'y' : 'z';
    const tangent = bottom ? 'z' : 'y';
    const center = this.light.target.position;
    const direction = _lightDirection.subVectors(this.light.position, center);
    const distance = direction.length();
    direction.divideScalar(distance);

    // Project the model bounds, with room for the penumbra, onto the receiver.
    // The light's far plane bounds the reach even at a grazing angle.
    const padding = this.maxDimension / 2;
    const height = Math.max(0, this.boundingBox.max[normal] - this.floor.position[normal]);
    const maxTravel = this.light.shadow.camera.far - distance + this.size.length() / 2;
    const travel = direction[normal] < 0 ? 0 : Math.min(
        (height + padding) / Math.max(direction[normal], pcssDepthEpsilon),
        maxTravel);
    const shiftX = -travel * direction.x;
    const shiftY = -travel * direction[tangent];

    const halfWidth = Math.max(this.size.x * 5, this.size.x / 2 + padding);
    const halfHeight = Math.max(this.size[tangent] * 5, this.size[tangent] / 2 + padding);
    const extraWidth = Math.max(0, Math.abs(shiftX) + this.size.x / 2 + padding - halfWidth);
    const extraHeight = Math.max(
        0, Math.abs(shiftY) + this.size[tangent] / 2 + padding - halfHeight);
    this.floor.position.x = center.x + Math.sign(shiftX) * extraWidth / 2;
    this.floor.position[tangent] = center[tangent] + Math.sign(shiftY) * extraHeight / 2;
    this.floor.scale.set(2 * halfWidth + extraWidth, 2 * halfHeight + extraHeight, 1);
  }

  private renderPCSS(
      renderer: WebGLRenderer|WebGPURenderer, scene: Scene,
      gpu?: WebGPUShadowMaterials) {
    if (this.light == null) return;
    gpu?.setPCSS(this.light, this.pcssScale.value, this.pcssMaxInverseDepth.value);

    if (!renderer.shadowMap.enabled) {
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type =
          'isWebGPURenderer' in renderer ? PCFShadowMap : BasicShadowMap;
    }

    if (!this.castShadowSet) {
      scene.traverse((object) => {
        if ((object as Mesh).isMesh && !object.userData.noHit) {
          object.castShadow = true;
        }
      });
      this.castShadowSet = true;
    }
  }
}
