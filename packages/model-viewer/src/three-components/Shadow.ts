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

/**
 * Real Three.js shadow implementation using DirectionalLight + ShadowMaterial.
 * Replaces the broken contact-shadow approach for Three.js r183+.
 *
 * shadow-intensity controls opacity of the shadow plane.
 * shadow-softness controls shadow map resolution (0=harsh/512, 1=soft/2048).
 */
export class Shadow extends Object3D {
  private light: DirectionalLight;
  private floor: Mesh;
  private intensity = 0;
  private boundingBox = new Box3();
  private size = new Vector3();
  private maxDimension = 0;
  private side: Side = 'bottom';
  public needsUpdate = false;

  constructor(scene: ModelScene, softness: number, side: Side) {
    super();

    this.light = new DirectionalLight(0xffffff, 2);
    this.light.castShadow = true;
    this.light.shadow.camera.near = 0.1;
    this.light.shadow.camera.far = 100;
    this.light.shadow.bias = -0.001;
    this.light.name = 'ShadowLight';

    const plane = new PlaneGeometry(1, 1);
    const shadowMaterial = new ShadowMaterial({opacity: 0, transparent: true});
    this.floor = new Mesh(plane, shadowMaterial);
    this.floor.receiveShadow = true;
    this.floor.userData.noHit = true;
    this.floor.name = 'ShadowFloor';

    this.add(this.light);
    this.add(this.light.target); // Add light target to scene graph
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

    const {min, max} = this.boundingBox;
    const center = new Vector3();
    this.boundingBox.getCenter(center);

    if (side === 'bottom') {
      // Floor at bottom of model
      this.floor.rotation.x = -Math.PI / 2;
      this.floor.position.set(center.x, min.y, center.z);
      this.floor.scale.set(
          this.size.x * 3,
          this.size.z * 3,
          1,
      );

      // Light above the model, slightly offset
      const lightHeight = (max.y - min.y) * 2 + 1;
      this.light.position.set(
          center.x + this.size.x * 0.5,
          max.y + lightHeight,
          center.z + this.size.z * 0.5,
      );
      this.light.target.position.copy(center);
      this.light.target.updateMatrixWorld();

      const halfSize = Math.max(this.size.x, this.size.z) * 1.5;
      this.light.shadow.camera.left = -halfSize;
      this.light.shadow.camera.right = halfSize;
      this.light.shadow.camera.top = halfSize;
      this.light.shadow.camera.bottom = -halfSize;
      this.light.shadow.camera.far = lightHeight * 3;
      this.light.shadow.camera.updateProjectionMatrix();
    } else {
      // Wall shadow: floor behind the model
      this.floor.rotation.x = 0;
      this.floor.position.set(center.x, center.y, min.z);
      this.floor.scale.set(
          this.size.x * 3,
          this.size.y * 3,
          1,
      );

      const lightDepth = (max.z - min.z) * 2 + 1;
      this.light.position.set(
          center.x + this.size.x * 0.5,
          center.y + this.size.y * 0.5,
          max.z + lightDepth,
      );
      this.light.target.position.copy(center);
      this.light.target.updateMatrixWorld();

      const halfSize = Math.max(this.size.x, this.size.y) * 1.5;
      this.light.shadow.camera.left = -halfSize;
      this.light.shadow.camera.right = halfSize;
      this.light.shadow.camera.top = halfSize;
      this.light.shadow.camera.bottom = -halfSize;
      this.light.shadow.camera.far = lightDepth * 3;
      this.light.shadow.camera.updateProjectionMatrix();
    }

    this.setSoftness(softness);
    this.needsUpdate = true;
  }

  /**
   * Controls shadow map resolution. softness=0 → 2048 (crisp), softness=1 → 512 (soft).
   */
  setSoftness(softness: number) {
    const mapSize = Math.round(512 + (1 - softness) * 1536);
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