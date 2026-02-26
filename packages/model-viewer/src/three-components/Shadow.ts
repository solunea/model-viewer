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

import {BasicShadowMap, Box3, DirectionalLight, DoubleSide, Mesh, MeshBasicMaterial, MeshDepthMaterial, Object3D, OrthographicCamera, PlaneGeometry, RGBAFormat, Scene, ShaderChunk, ShaderMaterial, ShadowMaterial, Vector3, WebGLRenderer, WebGLRenderTarget} from 'three';

import {ModelScene} from './ModelScene.js';
import {Damper} from './Damper.js';

export type Side = 'back'|'bottom';
export type ShadowMode = 'basic'|'pcss';

const DEFAULT_SHADOW_THETA = 0;
const DEFAULT_SHADOW_PHI = 0;

// Softness maps to resolution between 2^LOG_MAX and 2^LOG_MIN
const LOG_MAX_RESOLUTION = 9;
const LOG_MIN_RESOLUTION = 6;
const DEFAULT_HARD_INTENSITY = 0.3;

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

// ─── PCSS shader injection ──────────────────────────────────────────
const originalShadowChunk = ShaderChunk.shadowmap_pars_fragment;
let lastPCSS_lightSize = -1;
let lastPCSS_frustumWidth = -1;
let lastPCSS_nearPlane = -1;

function patchPCSS(lightSize: number, frustumWidth: number, nearPlane: number) {
  if (lightSize === lastPCSS_lightSize &&
      frustumWidth === lastPCSS_frustumWidth &&
      nearPlane === lastPCSS_nearPlane) {
    return false;
  }
  lastPCSS_lightSize = lightSize;
  lastPCSS_frustumWidth = frustumWidth;
  lastPCSS_nearPlane = nearPlane;

  const pcssShader = `
    #define LIGHT_WORLD_SIZE ${lightSize.toFixed(6)}
    #define LIGHT_FRUSTUM_WIDTH ${frustumWidth.toFixed(6)}
    #define LIGHT_SIZE_UV (LIGHT_WORLD_SIZE / LIGHT_FRUSTUM_WIDTH)
    #define NEAR_PLANE ${nearPlane.toFixed(6)}
    #define NUM_SAMPLES 10
    #define NUM_RINGS 5
    #define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
    vec2 poissonDisk[NUM_SAMPLES];
    void initPoissonSamples(const in vec2 randomSeed){
      float ANGLE_STEP = PI2 * float(NUM_RINGS) / float(NUM_SAMPLES);
      float INV_NUM_SAMPLES = 1.0 / float(NUM_SAMPLES);
      float angle = rand(randomSeed) * PI2;
      float radius = INV_NUM_SAMPLES;
      float radiusStep = radius;
      for(int i = 0; i < NUM_SAMPLES; i++){
        poissonDisk[i] = vec2(cos(angle), sin(angle)) * pow(radius, 0.75);
        radius += radiusStep; angle += ANGLE_STEP;
      }
    }
    float penumbraSize(const in float zReceiver, const in float zBlocker){
      return (zReceiver - zBlocker) / zBlocker;
    }
    float findBlocker(sampler2D shadowMap, const in vec2 uv, const in float zReceiver){
      float searchRadius = LIGHT_SIZE_UV * (zReceiver - NEAR_PLANE) / zReceiver;
      float blockerDepthSum = 0.0; int numBlockers = 0;
      for(int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; i++){
        float d = texture2D(shadowMap, uv + poissonDisk[i] * searchRadius).r;
        if(d < zReceiver){ blockerDepthSum += d; numBlockers++; }
      }
      if(numBlockers == 0) return -1.0;
      return blockerDepthSum / float(numBlockers);
    }
    float PCF_Filter(sampler2D shadowMap, vec2 uv, float zReceiver, float filterRadius){
      float sum = 0.0; float depth;
      #pragma unroll_loop_start
      for(int i = 0; i < 10; i++){
        depth = texture2D(shadowMap, uv + poissonDisk[i] * filterRadius).r;
        if(zReceiver <= depth) sum += 1.0;
      }
      #pragma unroll_loop_end
      #pragma unroll_loop_start
      for(int i = 0; i < 10; i++){
        depth = texture2D(shadowMap, uv + -poissonDisk[i].yx * filterRadius).r;
        if(zReceiver <= depth) sum += 1.0;
      }
      #pragma unroll_loop_end
      return sum / (2.0 * float(10));
    }
    float PCSS(sampler2D shadowMap, vec4 coords){
      vec2 uv = coords.xy; float zReceiver = coords.z;
      initPoissonSamples(uv);
      float avgBlockerDepth = findBlocker(shadowMap, uv, zReceiver);
      if(avgBlockerDepth == -1.0) return 1.0;
      float penumbraRatio = penumbraSize(zReceiver, avgBlockerDepth);
      float filterRadius = penumbraRatio * LIGHT_SIZE_UV * NEAR_PLANE / zReceiver;
      return PCF_Filter(shadowMap, uv, zReceiver, filterRadius);
    }
  `;
  let shader = originalShadowChunk;
  shader = shader.replace('#ifdef USE_SHADOWMAP', '#ifdef USE_SHADOWMAP' + pcssShader);
  shader = shader.replace(
      '\t\t\tif ( frustumTest ) {\n\t\t\t\tfloat depth = texture2D( shadowMap, shadowCoord.xy ).r;',
      '\t\t\tif ( frustumTest ) {\nreturn PCSS( shadowMap, shadowCoord );\n\t\t\t\tfloat depth = texture2D( shadowMap, shadowCoord.xy ).r;');
  ShaderChunk.shadowmap_pars_fragment = shader;
  return true;
}

function restoreShadowChunk() {
  ShaderChunk.shadowmap_pars_fragment = originalShadowChunk;
}

const _center = new Vector3();

/**
 * Dual-mode shadow:
 * - 'basic' (default): Orthographic camera + depth + gaussian blur. Very fast.
 * - 'pcss': DirectionalLight + PCSS shader. Higher quality, supports shadow-orbit.
 *
 * Mode is auto-selected: basic when orbit is (0,0), pcss when orbit is non-zero.
 */
export class Shadow extends Object3D {
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
  private thetaDamper = new Damper();
  private phiDamper = new Damper();
  public needsUpdate = false;

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
  private frustumWidth = 1;
  private nearPlane = 0.5;
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
    restoreShadowChunk();

    this.basicCamera = new OrthographicCamera(-0.5, 0.5, 0.5, -0.5, 0, 1);
    this.basicCamera.rotation.x = Math.PI / 2;  // look toward -Y (down)
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
    this.floor.rotation.x = Math.PI / 2; // lay flat in world XZ plane
    this.add(this.floor);  // child of Shadow, not camera

    this.blurPlane = new Mesh(plane);
    this.blurPlane.visible = false;
    this.basicCamera.add(this.blurPlane);

    this.depthMaterial = new MeshDepthMaterial();
    this.depthMaterial.onBeforeCompile = function(shader) {
      if (shader && typeof shader.fragmentShader === 'string') {
        shader.fragmentShader = shader.fragmentShader.replace(
            'gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );',
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

    this.light = new DirectionalLight(0xffffff, 2);
    this.light.castShadow = true;
    this.light.shadow.camera.near = 0.25;
    this.light.shadow.camera.far = 100;
    this.light.shadow.bias = 0.001;
    this.light.shadow.normalBias = 0.01;
    this.light.shadow.mapSize.set(1024, 1024);
    this.light.name = 'ShadowLight';

    const plane = new PlaneGeometry(1, 1);
    const shadowMat = new ShadowMaterial({opacity: 0, transparent: true});
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
      restoreShadowChunk();
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
    const isZero = (theta === 0 && phi === 0);
    const needMode = isZero ? 'basic' : 'pcss';

    if (needMode !== this.mode) {
      const savedSoftness = this.softness;
      const savedIntensity = this.intensity;
      if (needMode === 'pcss') {
        this.initPCSSMode();
        this.setupPCSSScene(this.side);
      } else {
        this.initBasicMode();
        this.setupBasicScene(this.side);
      }
      this.setSoftness(savedSoftness);
      this.setIntensity(savedIntensity);
    }

    this.goalTheta = theta;
    this.goalPhi = phi;
    this.needsUpdate = true;
  }

  update(delta: number): boolean {
    if (this.mode === 'basic') return false;
    if (this.theta === this.goalTheta && this.phi === this.goalPhi) {
      return false;
    }

    let dTheta = this.theta - this.goalTheta;
    if (Math.abs(dTheta) > Math.PI) {
      this.theta -= Math.sign(dTheta) * 2 * Math.PI;
    }

    this.theta = this.thetaDamper.update(this.theta, this.goalTheta, delta, Math.PI);
    this.phi = this.phiDamper.update(this.phi, this.goalPhi, delta, Math.PI / 2);

    this.updatePCSSLightPosition();
    this.needsUpdate = true;
    return true;
  }

  setSoftness(softness: number) {
    this.softness = softness;
    if (this.mode === 'basic') {
      this.updateBasicSoftness();
    } else {
      this.updatePCSSPatch();
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
      // floor.position.y is already set in setupBasicScene; offset shifts it
      this.floor.position.y += -offset + this.gap();
    } else {
      if (this.side === 'bottom') {
        this.floor.position.y = this.boundingBox.min.y - offset + this.gap();
      } else {
        this.floor.position.z = this.boundingBox.min.z - offset + this.gap();
      }
    }
  }

  gap() {
    return 0.001 * this.maxDimension;
  }

  render(renderer: WebGLRenderer, scene: Scene) {
    if (this.mode === 'basic') {
      this.renderBasic(renderer, scene);
    } else {
      this.renderPCSS(renderer, scene);
    }
    this.needsUpdate = false;
  }

  invalidateCastShadow() {
    this.castShadowSet = false;
  }

  dispose() {
    this.disposeMode();
    this.removeFromParent();
  }

  // ─── Basic mode internals ───

  private setupBasicScene(side: Side) {
    const {boundingBox, size} = this;
    boundingBox.getCenter(this.position);

    if (side === 'back') {
      const {min, max} = boundingBox;
      [min.y, min.z] = [min.z, min.y];
      [max.y, max.z] = [max.z, max.y];
      [size.y, size.z] = [size.z, size.y];
    }
    this.rotation.set(0, 0, 0);

    // Camera sits above the model, looking down (-Y)
    // Position in local space: center of model is at (0,0,0) relative to Shadow
    if (side === 'bottom') {
      // Shadow center = bbox center; camera above = max.y relative to Shadow
      this.basicCamera!.position.set(0, boundingBox.max.y - this.position.y, 0);
      // Floor at min.y
      this.floor.position.set(0, boundingBox.min.y - this.position.y, 0);
    } else {
      this.basicCamera!.position.set(0, boundingBox.max.z - this.position.z, 0);
      this.floor.position.set(0, boundingBox.min.z - this.position.z, 0);
    }
  }

  private updateBasicSoftness() {
    if (this.basicCamera == null) return;
    const {size} = this;

    const resolution = Math.pow(2,
        LOG_MAX_RESOLUTION - this.softness * (LOG_MAX_RESOLUTION - LOG_MIN_RESOLUTION));
    this.setMapSize(resolution);

    // Camera near=0, far = full height of model so depth captures everything
    this.basicCamera.near = 0;
    this.basicCamera.far = size.y;
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
      this.basicCamera.scale.set(
          size.x * (1 + TAP_WIDTH / baseWidth),
          size.z * (1 + TAP_WIDTH / baseHeight),
          1);
    }
    this.needsUpdate = true;
  }

  private renderBasic(renderer: WebGLRenderer, scene: Scene) {
    if (this.basicCamera == null || this.renderTarget == null) return;

    scene.overrideMaterial = this.depthMaterial;
    const initialClearAlpha = renderer.getClearAlpha();
    renderer.setClearAlpha(0);
    this.floor.visible = false;

    const xrEnabled = renderer.xr.enabled;
    renderer.xr.enabled = false;

    const oldRenderTarget = renderer.getRenderTarget();
    renderer.setRenderTarget(this.renderTarget);
    renderer.render(scene, this.basicCamera);

    scene.overrideMaterial = null;
    this.floor.visible = true;

    this.blurShadow(renderer);

    renderer.xr.enabled = xrEnabled;
    renderer.setRenderTarget(oldRenderTarget);
    renderer.setClearAlpha(initialClearAlpha);
  }

  private blurShadow(renderer: WebGLRenderer) {
    if (this.basicCamera == null || this.blurPlane == null ||
        this.horizontalBlurMaterial == null || this.verticalBlurMaterial == null ||
        this.renderTarget == null || this.renderTargetBlur == null) return;

    this.blurPlane.visible = true;

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
    this.position.set(0, 0, 0);
    this.rotation.set(0, 0, 0);
    this.boundingBox.getCenter(_center);
    const min = this.boundingBox.min;

    if (side === 'bottom') {
      this.floor.rotation.x = -Math.PI / 2;
      this.floor.position.set(_center.x, min.y, _center.z);
      this.floor.scale.set(this.size.x * 10, this.size.z * 10, 1);
    } else {
      this.floor.rotation.x = 0;
      this.floor.position.set(_center.x, _center.y, min.z);
      this.floor.scale.set(this.size.x * 10, this.size.y * 10, 1);
    }

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
    this.light.target.updateMatrixWorld();

    const halfSize = this.maxDimension * 3;
    const newFrustumWidth = halfSize * 2;
    const newNearPlane = this.light.shadow.camera.near;

    this.light.shadow.camera.left = -halfSize;
    this.light.shadow.camera.right = halfSize;
    this.light.shadow.camera.top = halfSize;
    this.light.shadow.camera.bottom = -halfSize;
    this.light.shadow.camera.far = radius * 20;
    this.light.shadow.camera.updateProjectionMatrix();

    if (newFrustumWidth !== this.frustumWidth || newNearPlane !== this.nearPlane) {
      this.frustumWidth = newFrustumWidth;
      this.nearPlane = newNearPlane;
      this.updatePCSSPatch();
    }
  }

  private updatePCSSPatch() {
    const lightSize = this.softness * this.maxDimension * 0.05;
    const changed = patchPCSS(lightSize, this.frustumWidth, this.nearPlane);
    if (!changed) return;

    const recompileId = Date.now();
    const forceRecompile = (m: any) => {
      m.defines = m.defines || {};
      m.defines.PCSS_VERSION = recompileId;
      m.needsUpdate = true;
    };
    forceRecompile(this.floor.material);

    if (this.parent != null) {
      this.parent.traverse((object) => {
        if ((object as Mesh).isMesh) {
          const meshMat = (object as Mesh).material;
          if (Array.isArray(meshMat)) {
            meshMat.forEach(forceRecompile);
          } else if (meshMat != null) {
            forceRecompile(meshMat);
          }
        }
      });
    }
  }

  private renderPCSS(renderer: WebGLRenderer, scene: Scene) {
    if (this.light == null) return;

    if (!renderer.shadowMap.enabled) {
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = BasicShadowMap;
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