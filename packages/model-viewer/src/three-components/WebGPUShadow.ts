/* @license Copyright 2026 Solunea. Licensed under the Apache License,
 * Version 2.0. */
import {DoubleSide, Texture, Vector2} from 'three';
import type {DepthTexture, DirectionalLight} from 'three';
import {Fn, If, clamp, float, ivec2, max, positionView, renderGroup, step, texture, textureLoad, textureSize, uniform, uv, vec2, vec4} from 'three/tsl';
import {MeshBasicNodeMaterial} from 'three/webgpu';
import type {Node} from 'three/webgpu';

import {pcssBlockerOffsets, pcssDepthEpsilon} from './PCSS.js';
import type {WebGPUShadowMaterials} from './Shadow.js';

const weights = [0.051, 0.0918, 0.12245, 0.1531, 0.1633, 0.1531, 0.12245, 0.0918, 0.051];

function addPCSSFilter(light: DirectionalLight) {
  const shadow = light.shadow.clone() as DirectionalLight['shadow'];
  // Three.js updates shared uniforms even when the receiver mesh is static.
  const scale = uniform(0).setGroup(renderGroup);
  const maxInverseDepth = uniform(0).setGroup(renderGroup);
  const filter = Fn(({depthTexture, shadowCoord, depthLayer}: {
    depthTexture: DepthTexture,
    shadowCoord: Node<'vec3'>,
    depthLayer: Node<'int'>
  }) => {
    const map = texture(depthTexture);
    const size = vec2(textureSize(map) as unknown as Node<'uvec2'>).toVar();
    const texelWidth = float(1).div(size.x.min(size.y)).toVar();
    const uv = shadowCoord.xy;
    const z = shadowCoord.z;
    const isArray = depthTexture.isArrayTexture;
    const compare = (point: Node<'vec2'>) => {
      const sample = map.sample(point);
      return (isArray ? sample.depth(depthLayer) : sample).compare(z);
    };
    const filterGrid = (point: Node<'vec2'>, spacing: Node<'float'>) => {
      const origin = spacing.mul(0.5).floor().toVar();
      const grid = point.sub(origin).div(spacing).toVar();
      const cell = grid.add(0.5).floor().toVar();
      const offset = grid.sub(cell).toVar();
      const low = float(0.5).sub(offset).toVar();
      const high = offset.add(0.5).toVar();
      const weights = [
        low.mul(low).mul(0.5).toVar(),
        float(0.75).sub(offset.mul(offset)).toVar(),
        high.mul(high).mul(0.5).toVar(),
      ];
      const center = cell.mul(spacing).add(origin).add(0.5).div(size).toVar();
      const stepUV = spacing.div(size).toVar();
      const visibility = float(0).toVar();
      for (const [y, wy] of weights.entries()) {
        for (const [x, wx] of weights.entries()) {
          const sample = center.add(vec2(x - 1, y - 1).mul(stepUV));
          visibility.addAssign(wx.x.mul(wy.y).mul(compare(sample)));
        }
      }
      return visibility;
    };
    const inverseReceiver = float(1).div(max(z, pcssDepthEpsilon)).toVar();
    const searchRadius = scale.mul(maxInverseDepth.sub(inverseReceiver)).toVar();
    const lit = float(1).toVar();
    If(searchRadius.lessThanEqual(texelWidth), () => {
      lit.assign(compare(uv));
    }).Else(() => {
      const sum = float(0).toVar();
      const count = float(0).toVar();
      for (const [x, y] of pcssBlockerOffsets) {
        const point = uv.add(vec2(x, y).mul(searchRadius));
        const sample = textureLoad(depthTexture, ivec2(
            clamp(point.mul(size), vec2(0), size.sub(1))));
        const load = isArray ? sample.depth(depthLayer) : sample;
        const depth = float(load as unknown as Node<'float'>).toVar();
        const blocked = float(1).sub(step(z, depth)).toVar();
        sum.addAssign(blocked.mul(depth));
        count.addAssign(blocked);
      }
      If(count.greaterThan(0), () => {
        const inverseBlocker = count.div(max(sum, count.mul(pcssDepthEpsilon)));
        const radius = scale.mul(inverseBlocker.sub(inverseReceiver)).toVar();
        If(radius.lessThanEqual(texelWidth), () => {
          lit.assign(compare(uv));
        }).Else(() => {
          const radiusInTexels = radius.div(texelWidth).toVar();
          const spacing = radiusInTexels.log2().floor().exp2().toVar();
          const ratio = radiusInTexels.div(spacing).toVar();
          const blend = ratio.mul(ratio).sub(1).div(3);
          const point = uv.mul(size).sub(0.5).toVar();
          If(spacing.equal(1), () => {
            lit.assign(compare(uv));
          }).Else(() => {
            lit.assign(filterGrid(point, spacing));
          });
          const coarse = filterGrid(point, spacing.mul(2));
          lit.assign(blend.mix(lit, coarse));
        });
      });
    });
    return lit;
  });
  (shadow as unknown as {filterNode: typeof filter}).filterNode = filter;
  return {shadow, scale, maxInverseDepth};
}

export function createWebGPUShadowMaterials(): WebGPUShadowMaterials {
  const pcss = new WeakMap<DirectionalLight, ReturnType<typeof addPCSSFilter>>();
  const getPCSS = (light: DirectionalLight) => {
    let filter = pcss.get(light);
    if (filter == null) {
      filter = addPCSSFilter(light);
      pcss.set(light, filter);
      const {shadow} = filter;
      const dispose = () => {
        light.removeEventListener('removed', dispose);
        pcss.delete(light);
        shadow.dispose();
      };
      light.addEventListener('removed', dispose);
    }
    return filter;
  };
  const range = uniform(new Vector2());
  const opacity = uniform(1);
  const direction = uniform(new Vector2());
  const input = texture(new Texture());
  const depth = new MeshBasicNodeMaterial({side: DoubleSide, toneMapped: false});
  depth.fragmentNode = vec4(
      0, 0, 0, clamp(float(1).sub(
          positionView.z.negate().sub(range.x).mul(range.y)), 0, 1).mul(opacity));
  const blur = new MeshBasicNodeMaterial({
    depthTest: false, depthWrite: false, toneMapped: false
  });
  blur.fragmentNode = Fn(() => {
    const sum = vec4(0).toVar();
    for (const [index, weight] of weights.entries()) {
      sum.addAssign(input.sample(uv().add(direction.mul(index - 4))).mul(weight));
    }
    return sum;
  })();
  return {
    depth,
    blur,
    setDepth(camera, value) {
      range.value.set(camera.near, 1 / (camera.far - camera.near));
      opacity.value = value;
    },
    setBlur(source, x, y) {
      input.value = source;
      direction.value.set(x, y);
    },
    setPCSS(light, scale, maxInverseDepth) {
      const filter = getPCSS(light);
      filter.scale.value = scale;
      filter.maxInverseDepth.value = maxInverseDepth;
    },
    withPCSSShadow(light, render) {
      // WebGL can render this light between WebGPU frames when capturing a
      // poster. Three.js must keep each renderer's shadow map independent.
      const webgl = light.shadow;
      const {shadow} = getPCSS(light);
      const camera = shadow.camera;
      const coordinateSystem = camera.coordinateSystem;
      camera.copy(webgl.camera);
      camera.coordinateSystem = coordinateSystem;
      camera.updateProjectionMatrix();
      light.shadow = shadow;
      try {
        render();
      } finally {
        light.shadow = webgl;
      }
    }
  };
}
