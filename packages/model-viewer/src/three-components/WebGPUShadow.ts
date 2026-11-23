/* @license Copyright 2026 Solunea. Licensed under the Apache License,
 * Version 2.0. */
import {DoubleSide, Texture, Vector2} from 'three';
import type {DepthTexture, DirectionalLight} from 'three';
import {Fn, If, clamp, float, ivec2, max, positionView, rand, step, texture, textureLoad, textureSize, uniform, uv, vec2, vec4} from 'three/tsl';
import {MeshBasicNodeMaterial} from 'three/webgpu';
import type {Node} from 'three/webgpu';

import type {WebGPUShadowMaterials} from './Shadow.js';

const weights = [0.051, 0.0918, 0.12245, 0.1531, 0.1633, 0.1531, 0.12245, 0.0918, 0.051];
const sampleCount = 10;

function addPCSSFilter(light: DirectionalLight) {
  const shadow = light.shadow.clone() as DirectionalLight['shadow'];
  const lightSize = uniform(0);
  const near = uniform(0.5);
  const filter = Fn(({depthTexture, shadowCoord, depthLayer}: {
    depthTexture: DepthTexture,
    shadowCoord: Node<'vec3'>,
    depthLayer: Node<'int'>
  }) => {
    const map = texture(depthTexture);
    const size = vec2(textureSize(map) as unknown as Node<'uvec2'>);
    const texel = (point: Node<'vec2'>) => {
      const value = textureLoad(depthTexture, ivec2(
          clamp(point.mul(size), vec2(0), size.sub(1))));
      if ('isArrayTexture' in depthTexture) value.depth(depthLayer);
      return float(value as unknown as Node<'float'>);
    };
    const uv = shadowCoord.xy;
    const z = shadowCoord.z;
    const rotation = rand(uv).mul(2 * Math.PI);
    const offsets = Array.from({length: sampleCount}, (_, i) => {
      const angle = rotation.add(i * Math.PI);
      return vec2(angle.cos(), angle.sin()).mul(((i + 1) / sampleCount) ** .75);
    });
    const radius = lightSize.mul(z.sub(near)).div(max(z, 1e-6));
    const sum = float(0).toVar();
    const count = float(0).toVar();
    for (const offset of offsets) {
      const depth = texel(uv.add(offset.mul(radius))).toVar();
      If(depth.lessThan(z), () => {
        sum.addAssign(depth);
        count.addAssign(1);
      });
    }
    const lit = float(1).toVar();
    If(count.greaterThan(0), () => {
      const blocker = max(sum.div(count), 1e-6);
      const penumbra = z.sub(blocker).div(blocker).mul(lightSize).mul(near).div(max(z, 1e-6));
      lit.assign(0);
      for (const offset of offsets) {
        lit.addAssign(step(z, texel(uv.add(offset.mul(penumbra)))));
        lit.addAssign(step(z, texel(uv.sub(offset.yx.mul(penumbra)))));
      }
      lit.divAssign(sampleCount * 2);
    });
    return lit;
  });
  (shadow as unknown as {filterNode: typeof filter}).filterNode = filter;
  return {shadow, lightSize, near};
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
    setPCSS(light, size, nearPlane) {
      const filter = getPCSS(light);
      filter.lightSize.value = size;
      filter.near.value = nearPlane;
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
