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

import {BasicShadowMap, Box3, DirectionalLight, Mesh, Object3D, PlaneGeometry, Scene, ShaderChunk, ShadowMaterial, Vector3, WebGLRenderer} from 'three';

import {ModelScene} from './ModelScene.js';
import {Damper} from './Damper.js';

export type Side = 'back'|'bottom';

// Default shadow orbit: 0° azimuth, 0° polar (directly from above)
const DEFAULT_SHADOW_THETA = 0;
const DEFAULT_SHADOW_PHI = 0;

// ─── PCSS shader injection ──────────────────────────────────────────
// Percentage Closer Soft Shadows: shadows get softer the farther the
// receiver is from the blocker, like real-world contact shadows.
// 10 Poisson samples keeps it fast on mobile / integrated GPUs.
// ─────────────────────────────────────────────────────────────────────
const originalShadowChunk = ShaderChunk.shadowmap_pars_fragment;

// Cache last PCSS params to avoid redundant shader patching
let lastPCSS_lightSize = -1;
let lastPCSS_frustumWidth = -1;
let lastPCSS_nearPlane = -1;

// Pre-patch the PCSS shader at module load time with default values.
// This ensures the shader is compiled during the first general render,
// not at the moment the model becomes visible (which causes a visible freeze).
patchPCSS(0.05, 6.0, 0.25);

function patchPCSS(lightSize: number, frustumWidth: number, nearPlane: number) {
  // Skip if params haven't changed (avoids costly shader recompilation)
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

    void initPoissonSamples( const in vec2 randomSeed ) {
      float ANGLE_STEP = PI2 * float( NUM_RINGS ) / float( NUM_SAMPLES );
      float INV_NUM_SAMPLES = 1.0 / float( NUM_SAMPLES );
      float angle = rand( randomSeed ) * PI2;
      float radius = INV_NUM_SAMPLES;
      float radiusStep = radius;
      for( int i = 0; i < NUM_SAMPLES; i ++ ) {
        poissonDisk[i] = vec2( cos( angle ), sin( angle ) ) * pow( radius, 0.75 );
        radius += radiusStep;
        angle += ANGLE_STEP;
      }
    }

    float penumbraSize( const in float zReceiver, const in float zBlocker ) {
      return (zReceiver - zBlocker) / zBlocker;
    }

    float findBlocker( sampler2D shadowMap, const in vec2 uv, const in float zReceiver ) {
      float searchRadius = LIGHT_SIZE_UV * ( zReceiver - NEAR_PLANE ) / zReceiver;
      float blockerDepthSum = 0.0;
      int numBlockers = 0;
      for( int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; i++ ) {
        float shadowMapDepth = texture2D(shadowMap, uv + poissonDisk[i] * searchRadius).r;
        if ( shadowMapDepth < zReceiver ) {
          blockerDepthSum += shadowMapDepth;
          numBlockers ++;
        }
      }
      if( numBlockers == 0 ) return -1.0;
      return blockerDepthSum / float( numBlockers );
    }

    float PCF_Filter(sampler2D shadowMap, vec2 uv, float zReceiver, float filterRadius ) {
      float sum = 0.0;
      float depth;
      #pragma unroll_loop_start
      for( int i = 0; i < 10; i ++ ) {
        depth = texture2D( shadowMap, uv + poissonDisk[ i ] * filterRadius ).r;
        if( zReceiver <= depth ) sum += 1.0;
      }
      #pragma unroll_loop_end
      #pragma unroll_loop_start
      for( int i = 0; i < 10; i ++ ) {
        depth = texture2D( shadowMap, uv + -poissonDisk[ i ].yx * filterRadius ).r;
        if( zReceiver <= depth ) sum += 1.0;
      }
      #pragma unroll_loop_end
      return sum / ( 2.0 * float( 10 ) );
    }

    float PCSS ( sampler2D shadowMap, vec4 coords ) {
      vec2 uv = coords.xy;
      float zReceiver = coords.z;
      initPoissonSamples( uv );
      float avgBlockerDepth = findBlocker( shadowMap, uv, zReceiver );
      if( avgBlockerDepth == -1.0 ) return 1.0;
      float penumbraRatio = penumbraSize( zReceiver, avgBlockerDepth );
      float filterRadius = penumbraRatio * LIGHT_SIZE_UV * NEAR_PLANE / zReceiver;
      return PCF_Filter( shadowMap, uv, zReceiver, filterRadius );
    }
  `;

  const pcssGetShadow = `return PCSS( shadowMap, shadowCoord );`;

  let shader = originalShadowChunk;

  shader = shader.replace(
      '#ifdef USE_SHADOWMAP',
      '#ifdef USE_SHADOWMAP' + pcssShader);

  shader = shader.replace(
      '\t\t\tif ( frustumTest ) {\n\t\t\t\tfloat depth = texture2D( shadowMap, shadowCoord.xy ).r;',
      '\t\t\tif ( frustumTest ) {\n' + pcssGetShadow +
          '\n\t\t\t\tfloat depth = texture2D( shadowMap, shadowCoord.xy ).r;');

  ShaderChunk.shadowmap_pars_fragment = shader;
  return true;
}

/**
 * Real Three.js shadow implementation using DirectionalLight + ShadowMaterial
 * with PCSS (Percentage Closer Soft Shadows) for distance-dependent penumbra.
 *
 * shadow-intensity controls opacity of the shadow plane.
 * shadow-softness controls PCSS light size (penumbra spread).
 * shadow-orbit controls the light direction as spherical coords (theta, phi).
 */
// Reusable vector to avoid GC pressure in hot path
const _center = new Vector3();

export class Shadow extends Object3D {
  private light: DirectionalLight;
  private floor: Mesh;
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
  private frustumWidth = 1;
  private nearPlane = 0.5;
  private castShadowSet = false;
  public needsUpdate = false;

  constructor(scene: ModelScene, softness: number, side: Side) {
    super();

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

    scene.target.add(this);

    this.setScene(scene, softness, side);
  }

  /**
   * Update shadow geometry and light frustum to fit the scene's bounding box.
   */
  setScene(scene: ModelScene, softness: number, side: Side) {
    this.side = side;
    this.boundingBox.copy(scene.boundingBox);
    this.boundingBox.getSize(this.size);
    this.maxDimension = Math.max(this.size.x, this.size.y, this.size.z);

    const min = this.boundingBox.min;
    const center = _center;

    if (side === 'bottom') {
      this.floor.rotation.x = -Math.PI / 2;
      this.floor.position.set(center.x, min.y, center.z);
      this.floor.scale.set(this.size.x * 10, this.size.z * 10, 1);
    } else {
      this.floor.rotation.x = 0;
      this.floor.position.set(center.x, center.y, min.z);
      this.floor.scale.set(this.size.x * 10, this.size.y * 10, 1);
    }

    this.updateLightPosition();
    this.setSoftness(softness);
    this.needsUpdate = true;
  }

  /**
   * Set the shadow light direction goal using spherical coordinates.
   * theta = azimuth angle (radians, around Y axis, 0 = front)
   * phi = polar angle (radians, from Y axis, 0 = directly above)
   */
  setOrbit(theta: number, phi: number) {
    this.goalTheta = theta;
    this.goalPhi = phi;
    this.needsUpdate = true;
  }

  /**
   * Updates the shadow orbit based on damper progression.
   * Returns true if the shadow orbit changed during this update.
   */
  update(delta: number): boolean {
    if (this.theta === this.goalTheta && this.phi === this.goalPhi) {
      return false;
    }

    // Wrap theta to take the shortest path
    let dTheta = this.theta - this.goalTheta;
    if (Math.abs(dTheta) > Math.PI) {
      this.theta -= Math.sign(dTheta) * 2 * Math.PI;
    }

    this.theta = this.thetaDamper.update(this.theta, this.goalTheta, delta, Math.PI);
    this.phi = this.phiDamper.update(this.phi, this.goalPhi, delta, Math.PI / 2);

    this.updateLightPosition();
    this.needsUpdate = true;
    return true;
  }

  /**
   * Position the DirectionalLight using current theta/phi spherical coords.
   */
  private updateLightPosition() {
    this.boundingBox.getCenter(_center);

    // Distance from center to place the light
    const radius = this.maxDimension * 2 + 1;

    // Spherical to cartesian (Y-up):
    // x = r * sin(phi) * sin(theta)
    // y = r * cos(phi)
    // z = r * sin(phi) * cos(theta)
    const sinPhi = Math.sin(this.phi);
    const lx = _center.x + radius * sinPhi * Math.sin(this.theta);
    const ly = _center.y + radius * Math.cos(this.phi);
    const lz = _center.z + radius * sinPhi * Math.cos(this.theta);

    this.light.position.set(lx, ly, lz);
    this.light.target.position.copy(_center);
    this.light.target.updateMatrixWorld();

    // Fit shadow camera frustum to the bounding box (tight = sharper PCSS)
    const halfSize = this.maxDimension * 3;
    const newFrustumWidth = halfSize * 2;
    const newNearPlane = this.light.shadow.camera.near;

    this.light.shadow.camera.left = -halfSize;
    this.light.shadow.camera.right = halfSize;
    this.light.shadow.camera.top = halfSize;
    this.light.shadow.camera.bottom = -halfSize;
    this.light.shadow.camera.far = radius * 20;
    this.light.shadow.camera.updateProjectionMatrix();

    // Only re-patch PCSS shader if frustum dimensions actually changed
    if (newFrustumWidth !== this.frustumWidth || newNearPlane !== this.nearPlane) {
      this.frustumWidth = newFrustumWidth;
      this.nearPlane = newNearPlane;
      this.updatePCSSPatch();
    }
  }

  /**
   * Controls PCSS penumbra spread via the light size parameter.
   * softness=0 → sharp shadow, softness=1 → very soft penumbra.
   */
  setSoftness(softness: number) {
    this.softness = softness;
    this.updatePCSSPatch();
    this.needsUpdate = true;
  }

  /**
   * Re-patch the PCSS shader with current light size and frustum dimensions.
   * Only triggers material recompilation when the shader actually changed.
   */
  private updatePCSSPatch() {
    // LIGHT_WORLD_SIZE controls penumbra: 0 = sharp, scales with model size
    const lightSize = this.softness * this.maxDimension * 0.05;
    const changed = patchPCSS(lightSize, this.frustumWidth, this.nearPlane);

    if (!changed) return;

    // Force materials to recompile with the new ShaderChunk (single traversal)
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

  /**
   * Set the shadow's intensity (0 to 1) — controls floor opacity.
   */
  setIntensity(intensity: number) {
    this.intensity = intensity;
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

  getIntensity(): number {
    return this.intensity;
  }

  /**
   * Shift the floor vertically. Positive is up.
   */
  setOffset(offset: number) {
    if (this.side === 'bottom') {
      this.floor.position.y = this.boundingBox.min.y - offset + this.gap();
    } else {
      this.floor.position.z = this.boundingBox.min.z - offset + this.gap();
    }
  }

  gap() {
    return 0.001 * this.maxDimension;
  }

  /**
   * Enable shadow rendering on the WebGLRenderer and set castShadow on
   * all meshes. The scene traversal is done only once for performance.
   */
  render(renderer: WebGLRenderer, scene: Scene) {
    if (!renderer.shadowMap.enabled) {
      renderer.shadowMap.enabled = true;
      // PCSS requires BasicShadowMap to read raw depth values
      renderer.shadowMap.type = BasicShadowMap;
    }

    // Only traverse scene once to set castShadow
    if (!this.castShadowSet) {
      scene.traverse((object) => {
        if ((object as Mesh).isMesh && !object.userData.noHit) {
          object.castShadow = true;
        }
      });
      this.castShadowSet = true;
    }

    this.needsUpdate = false;
  }

  /**
   * Reset castShadow flag so next render will re-traverse.
   * Call when model geometry changes.
   */
  invalidateCastShadow() {
    this.castShadowSet = false;
  }

  dispose() {
    // Restore original shader chunk on dispose
    ShaderChunk.shadowmap_pars_fragment = originalShadowChunk;
    this.light.shadow.map?.dispose();
    (this.floor.material as ShadowMaterial).dispose();
    this.floor.geometry.dispose();
    this.removeFromParent();
    this.light.target.removeFromParent();
  }
}