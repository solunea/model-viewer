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

import {Box3, DirectionalLight, Mesh, Object3D, PCFSoftShadowMap, PlaneGeometry, Scene, ShadowMaterial, Vector3, WebGLRenderer} from 'three';

import {ModelScene} from './ModelScene.js';

export type Side = 'back'|'bottom';

// Default shadow orbit: 0° azimuth, 75° polar (from above, slightly angled)
const DEFAULT_SHADOW_THETA = 0;
const DEFAULT_SHADOW_PHI = Math.PI * 75 / 180;

/**
 * Real Three.js shadow implementation using DirectionalLight + ShadowMaterial.
 * Replaces the broken contact-shadow approach for Three.js r183+.
 *
 * shadow-intensity controls opacity of the shadow plane.
 * shadow-softness controls shadow map resolution (0=soft/512, 1=crisp/2048).
 * shadow-orbit controls the light direction as spherical coords (theta, phi).
 */
export class Shadow extends Object3D {
  private light: DirectionalLight;
  private floor: Mesh;
  private intensity = 0;
  private boundingBox = new Box3();
  private size = new Vector3();
  private maxDimension = 0;
  private side: Side = 'bottom';
  private theta = DEFAULT_SHADOW_THETA;
  private phi = DEFAULT_SHADOW_PHI;
  public needsUpdate = false;

  constructor(scene: ModelScene, softness: number, side: Side) {
    super();

    this.light = new DirectionalLight(0xffffff, 2);
    this.light.castShadow = true;
    this.light.shadow.camera.near = 0.1;
    this.light.shadow.camera.far = 100;
    this.light.shadow.bias = -0.002;
    this.light.shadow.normalBias = 0.02;
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
    const center = new Vector3();
    this.boundingBox.getCenter(center);

    if (side === 'bottom') {
      this.floor.rotation.x = -Math.PI / 2;
      this.floor.position.set(center.x, min.y, center.z);
      this.floor.scale.set(this.size.x * 3, this.size.z * 3, 1);
    } else {
      this.floor.rotation.x = 0;
      this.floor.position.set(center.x, center.y, min.z);
      this.floor.scale.set(this.size.x * 3, this.size.y * 3, 1);
    }

    this.updateLightPosition();
    this.setSoftness(softness);
    this.needsUpdate = true;
  }

  /**
   * Set the shadow light direction using spherical coordinates.
   * theta = azimuth angle (radians, around Y axis, 0 = front)
   * phi = polar angle (radians, from Y axis, 0 = directly above)
   */
  setOrbit(theta: number, phi: number) {
    this.theta = theta;
    this.phi = phi;
    this.updateLightPosition();
    this.needsUpdate = true;
  }

  /**
   * Position the DirectionalLight using current theta/phi spherical coords.
   */
  private updateLightPosition() {
    const center = new Vector3();
    this.boundingBox.getCenter(center);

    // Distance from center to place the light
    const radius = this.maxDimension * 2 + 1;

    // Spherical to cartesian (Y-up):
    // x = r * sin(phi) * sin(theta)
    // y = r * cos(phi)
    // z = r * sin(phi) * cos(theta)
    const sinPhi = Math.sin(this.phi);
    const lx = center.x + radius * sinPhi * Math.sin(this.theta);
    const ly = center.y + radius * Math.cos(this.phi);
    const lz = center.z + radius * sinPhi * Math.cos(this.theta);

    this.light.position.set(lx, ly, lz);
    this.light.target.position.copy(center);
    this.light.target.updateMatrixWorld();

    // Fit shadow camera frustum to the bounding box
    const halfSize = this.maxDimension * 1.5;
    this.light.shadow.camera.left = -halfSize;
    this.light.shadow.camera.right = halfSize;
    this.light.shadow.camera.top = halfSize;
    this.light.shadow.camera.bottom = -halfSize;
    this.light.shadow.camera.far = radius * 3;
    this.light.shadow.camera.updateProjectionMatrix();
  }

  /**
   * Controls shadow map resolution. softness=0 → 512 (soft), softness=1 → 2048 (crisp).
   */
  setSoftness(softness: number) {
    const mapSize = Math.round(512 + softness * 1536);
    this.light.shadow.mapSize.set(mapSize, mapSize);
    if (this.light.shadow.map) {
      this.light.shadow.map.dispose();
      (this.light.shadow as any).map = null;
    }
    this.needsUpdate = true;
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
   * Enable shadow rendering on the WebGLRenderer and traverse the scene to set
   * castShadow on all meshes. Called once per frame when needsUpdate is true.
   */
  render(renderer: WebGLRenderer, scene: Scene) {
    if (!renderer.shadowMap.enabled) {
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = PCFSoftShadowMap;
    }

    scene.traverse((object) => {
      if ((object as Mesh).isMesh && !object.userData.noHit) {
        object.castShadow = true;
      }
    });

    this.needsUpdate = false;
  }

  dispose() {
    this.light.shadow.map?.dispose();
    (this.floor.material as ShadowMaterial).dispose();
    this.floor.geometry.dispose();
    this.removeFromParent();
    this.light.target.removeFromParent();
  }
}