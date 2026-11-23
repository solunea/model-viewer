/* @license Copyright 2026 Solunea. Licensed under the Apache License,
 * Version 2.0. */
import {FloatType, NearestFilter} from 'three';
import type {Color} from 'three';
import {Fn, blendOverlay, blendScreen, clamp, dot, float, length, max, mix, rtt, screenSize, screenUV, select, smoothstep, step, vec2, vec3, vec4} from 'three/tsl';
import type {Node, TextureNode} from 'three/webgpu';

export interface WebGPUJFASettings {
  mode: number;
  thickness: number;
  inside: boolean;
  color: Color;
  blendMode: number;
}

const neighbors = [[-1, -1], [0, -1], [1, -1], [-1, 0],
                   [1, 0], [-1, 1], [0, 1], [1, 1]];

export function createJFAOutput(
    input: Node<'vec4'>, mask: TextureNode, options: WebGPUJFASettings) {
  const resources: Array<{dispose(): void}> = [];
  const target = (shader: Node<'vec4'>) => {
    const stage = rtt(shader, null, null, {
      type: FloatType, minFilter: NearestFilter, magFilter: NearestFilter,
      depthBuffer: false
    });
    resources.push({dispose: () => {
      stage.renderTarget?.dispose();
      stage.dispose();
    }});
    return stage;
  };
  const silhouette = (point: Node<'vec2'>) => mask.sample(
      point.div(screenSize)).a.greaterThan(.01);
  let field = target(Fn(() => {
    const point = screenUV.mul(screenSize);
    const filled = silhouette(point);
    const different = (x: number, y: number) => {
      const other = silhouette(point.add(vec2(x, y)));
      return filled.and(other.not()).or(filled.not().and(other));
    };
    const border = different(1, 0).or(different(-1, 0))
                       .or(different(0, 1)).or(different(0, -1));
    return vec4(select(border, point, vec2(-1)), 0, 1);
  })());

  for (let jump = Math.ceil(options.thickness); ; jump = Math.ceil(jump / 2)) {
    const source = field;
    const stride = jump;
    field = target(Fn(() => {
      const point = screenUV.mul(screenSize);
      const best = source.sample(screenUV).xy.toVar();
      const distance = select(best.x.greaterThanEqual(0), length(best.sub(point)), float(1e6)).toVar();
      for (const [x, y] of neighbors) {
        const candidate = source.sample(screenUV.add(vec2(x * stride, y * stride).div(screenSize))).xy;
        const delta = length(candidate.sub(point));
        const closer = candidate.x.greaterThanEqual(0).and(delta.lessThan(distance));
        best.assign(select(closer, candidate, best));
        distance.assign(select(closer, delta, distance));
      }
      return vec4(best, 0, 1);
    })());
    if (jump <= 1) break;
  }

  const distanceAt = (point: Node<'vec2'>) => {
    const seed = field.sample(point.div(screenSize)).xy;
    return select(seed.x.lessThan(0), float(1e6), length(seed.sub(point)));
  };
  const output = Fn(() => {
    const point = screenUV.mul(screenSize);
    const filled = silhouette(point);
    const distance = distanceAt(point);
    const inside = options.inside ? filled : filled.not();
    let draw = inside.and(distance.lessThan(options.thickness));
    if (options.mode === 6) {
      draw = draw.and(distance.div(15).fract().lessThan(.4));
    } else if (options.mode === 5) {
      const radius = 13 * Math.min(options.thickness, 40) * .02;
      const width = radius * 2;
      const nearest = point.div(width).floor().mul(width).add(radius);
      const alpha = float(0).toVar();
      for (const [x, y] of [[0, 0], ...neighbors]) {
        const center = nearest.add(vec2(x, y).mul(width));
        const d = distanceAt(center).add(radius * 1.3).div(1.5);
        const strength = clamp(d.div(options.thickness), 0, 1).pow(.75).oneMinus();
        const dotRadius = strength.mul(width);
        alpha.assign(max(alpha, smoothstep(
            dotRadius.sub(1), dotRadius, length(point.sub(center))).oneMinus()));
      }
      draw = draw.and(alpha.greaterThan(.5));
    }

    const tint = vec3(options.color.r, options.color.g, options.color.b);
    let model = input.rgb;
    switch (options.blendMode) {
      case 1: model = tint.mul(model); break;
      case 2: model = blendScreen(tint, model); break;
      case 3: model = blendOverlay(tint, model); break;
      case 4: {
        const low = tint.sub(model.mul(2).oneMinus().mul(tint).mul(tint.oneMinus()));
        const curve = mix(tint.mul(16).sub(12).mul(tint).add(4).mul(tint),
                          tint.sqrt(), step(vec3(.25), tint));
        const high = tint.add(model.mul(2).sub(1).mul(curve.sub(tint)));
        model = mix(low, high, step(vec3(.5), model));
        break;
      }
      case 5: {
        const luminance = vec3(.3, .59, .11);
        model = clamp(tint.add(dot(model, luminance).sub(dot(tint, luminance))), 0, 1);
        break;
      }
    }
    return select(draw, vec4(tint, 1),
                  vec4(select(filled, model, input.rgb), input.a));
  })();
  return {output, resources};
}
