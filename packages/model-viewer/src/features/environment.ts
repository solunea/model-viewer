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

import {property} from 'lit/decorators.js';
import {ACESFilmicToneMapping, AgXToneMapping, CineonToneMapping, LinearToneMapping, NeutralToneMapping, NoToneMapping, ReinhardToneMapping, Texture} from 'three';

import ModelViewerElementBase, {$needsRender, $progressTracker, $renderer, $scene, $shouldAttemptPreload} from '../model-viewer-base.js';
import {clamp, Constructor, deserializeUrl} from '../utilities.js';

export const BASE_OPACITY = 0.5;
const DEFAULT_SHADOW_INTENSITY = 0.0;
const DEFAULT_SHADOW_SOFTNESS = 1.0;
const DEFAULT_SHADOW_ORBIT = '0deg 0deg';
const DEFAULT_SHADOW_INTERPOLATION_DECAY = 0;
const DEFAULT_SKYBOX_INTERPOLATION_DECAY = 0;
const DEFAULT_EXPOSURE = 1.0;

export type ToneMappingValue = 'auto'|'aces'|'agx'|'commerce'|'neutral'|
    'reinhard'|'cineon'|'linear'|'none';

export const $currentEnvironmentMap = Symbol('currentEnvironmentMap');
export const $currentBackground = Symbol('currentBackground');
export const $updateEnvironment = Symbol('updateEnvironment');
const $cancelEnvironmentUpdate = Symbol('cancelEnvironmentUpdate');

export declare interface EnvironmentInterface {
  environmentImage: string|null;
  skyboxImage: string|null;
  skyboxDepthImage: string|null;
  skyboxDofFocusMode: string;
  skyboxDofFocus: number;
  skyboxDofStrength: number;
  skyboxDofMaxBlur: number;
  skyboxDofFocusSmoothing: number;
  skyboxDepthInvert: boolean;
  skyboxHeight: string;
  shadowIntensity: number;
  shadowSoftness: number;
  shadowOrbit: string;
  shadowInterpolationDecay: number;
  skyboxInterpolationDecay: number;
  exposure: number;
  hasBakedShadow(): boolean;
  preloadSkybox(
      skyboxImage?: string|null, environmentImage?: string|null,
      skyboxDepthImage?: string|null): Promise<void>;
}

export const EnvironmentMixin = <T extends Constructor<ModelViewerElementBase>>(
    ModelViewerElement: T): Constructor<EnvironmentInterface>&T => {
  class EnvironmentModelViewerElement extends ModelViewerElement {
    @property({type: String, attribute: 'environment-image'})
    environmentImage: string|null = null;

    @property({type: String, attribute: 'skybox-image'})
    skyboxImage: string|null = null;

    @property({type: String, attribute: 'skybox-depth-image'})
    skyboxDepthImage: string|null = null;

    @property({type: String, attribute: 'skybox-dof-focus-mode'})
    skyboxDofFocusMode: string = 'manual';

    @property({type: Number, attribute: 'skybox-dof-focus'})
    skyboxDofFocus: number = 1;

    @property({type: Number, attribute: 'skybox-dof-strength'})
    skyboxDofStrength: number = 1;

    @property({type: Number, attribute: 'skybox-dof-max-blur'})
    skyboxDofMaxBlur: number = 12;

    @property({type: Number, attribute: 'skybox-dof-focus-smoothing'})
    skyboxDofFocusSmoothing: number = 140;

    @property({type: Boolean, attribute: 'skybox-depth-invert'})
    skyboxDepthInvert: boolean = false;

    @property(
        {type: Number, attribute: 'shadow-intensity', hasChanged: () => true})
    shadowIntensity: number = DEFAULT_SHADOW_INTENSITY;

    @property(
        {type: Number, attribute: 'shadow-softness', hasChanged: () => true})
    shadowSoftness: number = DEFAULT_SHADOW_SOFTNESS;

    @property({type: String, attribute: 'shadow-orbit', hasChanged: () => true})
    shadowOrbit: string = DEFAULT_SHADOW_ORBIT;

    @property({type: Number, attribute: 'shadow-interpolation-decay'})
    shadowInterpolationDecay: number = DEFAULT_SHADOW_INTERPOLATION_DECAY;

    @property({type: Number, attribute: 'skybox-interpolation-decay'})
    skyboxInterpolationDecay: number = DEFAULT_SKYBOX_INTERPOLATION_DECAY;

    @property({type: Number}) exposure: number = DEFAULT_EXPOSURE;

    @property({type: String, attribute: 'tone-mapping'})
    toneMapping: ToneMappingValue = 'auto';

    @property({type: String, attribute: 'skybox-height'})
    skyboxHeight: string = '0';

    protected[$currentEnvironmentMap]: Texture|null = null;
    protected[$currentBackground]: Texture|null = null;

    private[$cancelEnvironmentUpdate]: ((...args: any[]) => any)|null = null;

    updated(changedProperties: Map<string|number|symbol, unknown>) {
      super.updated(changedProperties);

      if (changedProperties.has('shadowIntensity')) {
        this[$scene].setShadowIntensity(this.shadowIntensity * BASE_OPACITY);
        this[$needsRender]();
      }

      if (changedProperties.has('shadowSoftness')) {
        this[$scene].setShadowSoftness(this.shadowSoftness);
        this[$needsRender]();
      }

      if (changedProperties.has('shadowInterpolationDecay')) {
        this[$scene].setShadowInterpolationDecay(this.shadowInterpolationDecay);
      }

      if (changedProperties.has('skyboxInterpolationDecay')) {
        this[$scene].setSkyboxInterpolationDecay(this.skyboxInterpolationDecay);
      }

      if (changedProperties.has('shadowOrbit')) {
        const orbitStr = this.shadowOrbit || '0deg 0deg';
        const parts = orbitStr.trim().split(/\s+/);
        const thetaDeg =
            parts.length > 0 && parts[0] !== 'auto' ? parseFloat(parts[0]) : 0;
        const phiDeg =
            parts.length > 1 && parts[1] !== 'auto' ? parseFloat(parts[1]) : 0;
        const theta = (isNaN(thetaDeg) ? 0 : thetaDeg) * Math.PI / 180;
        const phi = (isNaN(phiDeg) ? 0 : phiDeg) * Math.PI / 180;
        this[$scene].setShadowOrbit(theta, phi);
        this[$needsRender]();
      }

      if (changedProperties.has('exposure')) {
        this[$scene].exposure = this.exposure;
        this[$needsRender]();
      }

      if (changedProperties.has('toneMapping')) {
        const TONE_MAPPING = new Map([
          ['aces', ACESFilmicToneMapping],
          ['agx', AgXToneMapping],
          ['reinhard', ReinhardToneMapping],
          ['cineon', CineonToneMapping],
          ['linear', LinearToneMapping],
          ['none', NoToneMapping]
        ]);

        this[$scene].toneMapping =
            TONE_MAPPING.get(this.toneMapping) ?? NeutralToneMapping;
        this[$needsRender]();
      }

      if ((changedProperties.has('environmentImage') ||
           changedProperties.has('skyboxImage') ||
           changedProperties.has('skyboxDepthImage')) &&
          this[$shouldAttemptPreload]()) {
        this[$updateEnvironment]();
      }

      if (changedProperties.has('skyboxDofFocusMode') ||
          changedProperties.has('skyboxDofFocus') ||
          changedProperties.has('skyboxDofStrength') ||
          changedProperties.has('skyboxDofMaxBlur') ||
          changedProperties.has('skyboxDofFocusSmoothing') ||
          changedProperties.has('skyboxDepthInvert')) {
        this[$scene].setSkyboxDofOptions({
          focusMode: this.skyboxDofFocusMode,
          focus: this.skyboxDofFocus,
          strength: this.skyboxDofStrength,
          maxBlur: this.skyboxDofMaxBlur,
          focusSmoothing: this.skyboxDofFocusSmoothing,
          depthInvert: this.skyboxDepthInvert
        });
        this[$needsRender]();
      }

      if (changedProperties.has('skyboxHeight')) {
        this[$scene].setGroundedSkybox();
        this[$needsRender]();
      }
    }

    hasBakedShadow(): boolean {
      return this[$scene].bakedShadows.size > 0;
    }

    private getSkyboxCacheKey(skyboxImage: string|null): string|null {
      if (skyboxImage == null || !/^(blob:|data:)/i.test(skyboxImage)) {
        return null;
      }

      return this.getAttribute('data-skybox-cache-key') ||
          this.getAttribute('data-skybox-src');
    }

    async preloadSkybox(
        skyboxImage: string|null = this.skyboxImage,
        environmentImage: string|null = this.environmentImage,
        skyboxDepthImage: string|null = this.skyboxDepthImage): Promise<void> {
      const {textureUtils} = this[$renderer];
      if (textureUtils == null) {
        return;
      }

      await textureUtils.generateEnvironmentMapAndSkybox(
          deserializeUrl(skyboxImage), environmentImage, () => {},
          this.withCredentials, skyboxDepthImage,
          this.getSkyboxCacheKey(skyboxImage));
    }

    async[$updateEnvironment]() {
      const {skyboxDepthImage, skyboxImage, environmentImage} = this;

      if (this[$cancelEnvironmentUpdate] != null) {
        this[$cancelEnvironmentUpdate]!();
        this[$cancelEnvironmentUpdate] = null;
      }

      const {textureUtils} = this[$renderer];

      if (textureUtils == null) {
        return;
      }

      const updateEnvProgress =
          this[$progressTracker].beginActivity('environment-update');

      try {
        const {environmentMap, skybox, skyboxDepth} =
            await textureUtils.generateEnvironmentMapAndSkybox(
                deserializeUrl(skyboxImage),
                environmentImage,
                (progress: number) => updateEnvProgress(clamp(progress, 0, 1)),
                this.withCredentials,
                skyboxDepthImage,
                this.getSkyboxCacheKey(skyboxImage));

        if (this[$currentEnvironmentMap] !== environmentMap) {
          this[$currentEnvironmentMap] = environmentMap;
          this.dispatchEvent(new CustomEvent('environment-change'));
        }
        if (skybox != null) {
          // When using the same environment and skybox, use the environment as
          // it gives HDR filtering.
          this[$currentBackground] =
              skybox.name === environmentMap.name ? environmentMap : skybox;
        } else {
          this[$currentBackground] = null;
        }

        this[$scene].setSkyboxDofOptions({
          focusMode: this.skyboxDofFocusMode,
          focus: this.skyboxDofFocus,
          strength: this.skyboxDofStrength,
          maxBlur: this.skyboxDofMaxBlur,
          focusSmoothing: this.skyboxDofFocusSmoothing,
          depthInvert: this.skyboxDepthInvert
        });
        this[$scene].setEnvironmentAndSkybox(
            this[$currentEnvironmentMap], this[$currentBackground],
            skyboxDepth);
      } catch (errorOrPromise) {
        if (errorOrPromise instanceof Error) {
          this[$scene].setEnvironmentAndSkybox(null, null, null);
          throw errorOrPromise;
        }
      } finally {
        updateEnvProgress(1.0);
      }
    }
  }

  return EnvironmentModelViewerElement;
};
