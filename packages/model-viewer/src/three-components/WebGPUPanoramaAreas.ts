/* @license Copyright 2026 Solunea. Licensed under the Apache License,
 * Version 2.0. */
import {BackSide} from 'three';
import {Discard, Fn, If, asin, atan, clamp, float, floor, fract, mix, min, normalize, positionLocal, smoothstep, texture, uniform, varying, vec2, vec4, viewportUV} from 'three/tsl';
import {MeshBasicNodeMaterial} from 'three/webgpu';

import type {CreatePanoramaAreaMaterial} from './PanoramaAreas.js';

export const createWebGPUPanoramaMaterial: CreatePanoramaAreaMaterial =
    (mask, color, dimensions, bounds) => {
      const map = texture(mask);
      const tint = uniform(color);
      const area = uniform(bounds);
      const panorama = uniform(dimensions);
      const progress = uniform(-1);
      const direction = varying(positionLocal);
      const material = new MeshBasicNodeMaterial({
        side: BackSide,
        transparent: true,
        depthTest: false,
        depthWrite: false,
        toneMapped: false
      });
      material.fragmentNode = Fn(() => {
        const d = normalize(direction);
        const uv = vec2(
            fract(atan(d.x, d.z.negate()).div(2 * Math.PI).add(0.25)),
            float(0.5).sub(asin(clamp(d.y, -1, 1)).div(Math.PI)));
        const point = min(floor(uv.mul(panorama)), panorama.sub(1)).sub(area.xy);
        Discard(point.x.lessThan(0).or(point.y.lessThan(0))
                    .or(point.x.greaterThanEqual(area.z))
                    .or(point.y.greaterThanEqual(area.w)));
        Discard(map.sample(point.add(0.5).div(area.zw)).r.lessThan(0.5));
        const alpha = float(0.3).toVar();
        If(progress.greaterThanEqual(0), () => {
          Discard(progress.greaterThanEqual(1));
          const fraction = smoothstep(0, 1, progress);
          const center = mix(-0.65, 1.65, fraction)
                             .add(float(0.5).sub(viewportUV.y).mul(mix(0.36, 0.18, fraction)));
          alpha.assign(float(1).sub(viewportUV.x.sub(center).abs().mul(2))
                           .max(0).mul(0.3));
          Discard(alpha.lessThanEqual(0));
        });
        return vec4(tint, alpha);
      })();
      return {
        material,
        update(nextMask, shineProgress) {
          map.value = nextMask;
          progress.value = shineProgress;
        }
      };
    };
