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

import ModelViewerElementBase, {$needsRender, $progressTracker, $renderer, $scene, $shouldAttemptPreload, $updateSource} from '../model-viewer-base.js';
import {loadSkyboxDepth, SkyboxDepthImage} from '../three-components/SkyboxDepth.js';
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

interface SkyboxDepthRequest {
  color: string;
  url: string;
  credentials: boolean;
  controller: AbortController;
  image: SkyboxDepthImage|null;
}

export declare interface EnvironmentInterface {
  environmentImage: string|null;
  skyboxImage: string|null;
  skyboxDepthImage: string|null;
  skyboxDepthInvert: boolean;
  skyboxDofFocusMode: string;
  skyboxDofFocus: number;
  skyboxDofStrength: number;
  skyboxDofMaxBlur: number;
  skyboxDofFocusSmoothing: number;
  skyboxHeight: string;
  shadowIntensity: number;
  shadowSoftness: number;
  shadowOrbit: string;
  shadowInterpolationDecay: number;
  skyboxInterpolationDecay: number;
  exposure: number;
  hasBakedShadow(): boolean;
  preloadSkybox(
      skyboxImage?: string|null, environmentImage?: string|null): Promise<void>;
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

    @property({type: Boolean, attribute: 'skybox-depth-invert'})
    skyboxDepthInvert = false;

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

    private environmentRequest = 0;
    private backgroundUrl: string|null = null;
    private depthRequest: SkyboxDepthRequest|null = null;

    connectedCallback() {
      super.connectedCallback();
      this.refreshSkyboxDepth();
    }

    disconnectedCallback() {
      super.disconnectedCallback();
      this.depthRequest?.controller.abort();
      this.depthRequest = null;
      this[$scene].clearSkyboxDepth();
    }

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
           changedProperties.has('skyboxOnly')) &&
          this[$shouldAttemptPreload]()) {
        if (!this.loaded && this.src == null && this.panoramaSource != null) {
          void this[$updateSource]();
        } else {
          this[$updateEnvironment]();
        }
      }

      if (changedProperties.has('skyboxDepthInvert') ||
          changedProperties.has('skyboxDofFocusMode') ||
          changedProperties.has('skyboxDofFocus') ||
          changedProperties.has('skyboxDofStrength') ||
          changedProperties.has('skyboxDofMaxBlur') ||
          changedProperties.has('skyboxDofFocusSmoothing')) {
        this.updateSkyboxDofOptions();
        this[$needsRender]();
      }

      if (changedProperties.has('skyboxHeight')) {
        this[$scene].setGroundedSkybox();
        this[$needsRender]();
      }

      if (changedProperties.has('skyboxImage') ||
          changedProperties.has('skyboxDepthImage') ||
          changedProperties.has('skyboxDofMaxBlur') ||
          changedProperties.has('skyboxDofStrength') ||
          changedProperties.has('withCredentials')) {
        this.refreshSkyboxDepth();
      }
    }

    private updateSkyboxDofOptions() {
      this[$scene].setSkyboxDofOptions({
        focusMode: this.skyboxDofFocusMode,
        focus: this.skyboxDofFocus,
        strength: this.skyboxDofStrength,
        maxBlur: this.skyboxDofMaxBlur,
        focusSmoothing: this.skyboxDofFocusSmoothing,
        depthInvert: this.skyboxDepthInvert
      });
    }

    private applySkyboxDepth() {
      const request = this.depthRequest;
      if (request != null && request.color === this.backgroundUrl &&
          request.color === deserializeUrl(this.skyboxImage) &&
          request.url === deserializeUrl(this.skyboxDepthImage) &&
          request.credentials === this.withCredentials) {
        this[$scene].setSkyboxDepth(this[$currentBackground], request.image);
      }
    }

    private refreshSkyboxDepth() {
      const url = deserializeUrl(this.skyboxDepthImage);
      const previous = this.depthRequest;
      if (!url || !this.skyboxImage || !this.isConnected ||
          !this[$scene].skyboxDofEnabled) {
        previous?.controller.abort();
        this.depthRequest = null;
        this[$scene].clearSkyboxDepth();
        return;
      }

      const color = deserializeUrl(this.skyboxImage);
      const credentials = this.withCredentials;
      if (!color || !this[$renderer].canRender || !this[$shouldAttemptPreload]() ||
          (previous?.color === color && previous.url === url &&
           previous.credentials === credentials)) {
        return;
      }

      previous?.controller.abort();
      const controller = new AbortController();
      const request: SkyboxDepthRequest = {
        color, url, credentials, controller, image: null
      };
      this.depthRequest = request;
      this.applySkyboxDepth();
      void loadSkyboxDepth(url, credentials, controller.signal).then(image => {
        if (this.depthRequest === request) {
          request.image = image;
          this.applySkyboxDepth();
        }
      }, () => {
        // The color load is independent, including on HTTP or decode failure.
      });
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
        environmentImage: string|null = this.environmentImage): Promise<void> {
      const {textureUtils} = this[$renderer];
      if (textureUtils == null) {
        return;
      }

      await textureUtils.generateEnvironmentMapAndSkybox(
          deserializeUrl(skyboxImage), environmentImage, () => {},
          this.withCredentials, this.getSkyboxCacheKey(skyboxImage));
    }

    async[$updateEnvironment]() {
      const {skyboxImage, environmentImage} = this;
      const skyboxUrl = deserializeUrl(skyboxImage);
      const request = ++this.environmentRequest;
      this.updateSkyboxDofOptions();
      this.refreshSkyboxDepth();

      const {textureUtils} = this[$renderer];

      if (textureUtils == null) {
        return;
      }

      const updateEnvProgress =
          this[$progressTracker].beginActivity('environment-update');

      try {
        const {environmentMap, skybox} =
            await textureUtils.generateEnvironmentMapAndSkybox(
                skyboxUrl,
                environmentImage,
                (progress: number) => updateEnvProgress(clamp(progress, 0, 1)),
                this.withCredentials,
                this.getSkyboxCacheKey(skyboxImage));

        if (request !== this.environmentRequest || skyboxImage !== this.skyboxImage ||
            environmentImage !== this.environmentImage) {
          return;
        }
        const environmentChanged = this[$currentEnvironmentMap] !== environmentMap;
        this[$currentEnvironmentMap] = environmentMap;
        if (skybox != null) {
          // When using the same environment and skybox, use the environment as
          // it gives HDR filtering.
          this[$currentBackground] =
              skybox.name === environmentMap.name ? environmentMap : skybox;
        } else {
          this[$currentBackground] = null;
        }

        this[$scene].setEnvironmentAndSkybox(
            this[$currentEnvironmentMap], this[$currentBackground]);
        this.backgroundUrl = skyboxUrl;
        this.applySkyboxDepth();
        if (environmentChanged) {
          this.dispatchEvent(new CustomEvent('environment-change'));
        }
      } catch (errorOrPromise) {
        if (request === this.environmentRequest && skyboxImage === this.skyboxImage &&
            environmentImage === this.environmentImage && errorOrPromise instanceof Error) {
          this[$scene].setEnvironmentAndSkybox(null, null);
          throw errorOrPromise;
        }
      } finally {
        updateEnvProgress(1.0);
      }
    }
  }

  return EnvironmentModelViewerElement;
};
