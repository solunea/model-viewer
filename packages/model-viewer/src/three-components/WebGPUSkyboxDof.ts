/* @license Copyright 2026 Solunea. Licensed under the Apache License,
 * Version 2.0. */
import {BufferAttribute, BufferGeometry, ColorManagement, EquirectangularReflectionMapping, HalfFloatType, Matrix4, Mesh, NoBlending, RenderTarget, SRGBTransfer, Texture, Vector2} from 'three';
import {Fn, If, Loop, asin, atan, clamp, dot, float, fract, int, length, max, min, mix, normalize, positionGeometry, select, smoothstep, step, texture, textureSize, uniform, uniformArray, uv, vec2, vec4, viewportUV} from 'three/tsl';
import {NodeMaterial, QuadMesh, WebGPURenderer} from 'three/webgpu';
import type {Node} from 'three/webgpu';

import type {ModelScene, SkyboxDofFrame} from './ModelScene.js';
import {rings} from './SkyboxDof.js';

const MAX_WORK_PIXELS = 520000;

function target(name: string) {
  const result = new RenderTarget(1, 1, {type: HalfFloatType, depthBuffer: false});
  result.texture.name = name;
  return result;
}

function passMaterial() {
  const material = new NodeMaterial();
  material.setValues({
    depthTest: false, depthWrite: false, blending: NoBlending, toneMapped: false
  });
  return material;
}

export class WebGPUSkyboxDof {
  private readonly projection = target('projectedColorAndRadius');
  private readonly near = target('nearLayer');
  private readonly far = target('farLayer');
  private readonly layer = uniform(0);
  private readonly project = passMaterial();
  private readonly gather = passMaterial();
  private readonly composite = passMaterial();
  private readonly quad = new QuadMesh(this.project);
  private readonly background = new Mesh(
      new BufferGeometry().setAttribute('position', new BufferAttribute(
          new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3)),
      this.composite);
  private readonly cameraToSkybox = uniform(new Matrix4());
  private readonly projectionInverse = uniform(new Matrix4());
  private readonly skyboxRotation = new Matrix4();
  private readonly color = texture(new Texture());
  private readonly depth = texture(new Texture());
  private readonly flipDepthY = uniform(0);
  private readonly depthIs16Bit = uniform(0);
  private readonly invert = uniform(0);
  private readonly focus = uniform(0);
  private readonly strength = uniform(0);
  private readonly maxBlur = uniform(0);
  private readonly maximumNear = uniform(0);
  private readonly maximumFar = uniform(0);
  private readonly opacity = uniform(1);
  private readonly intensity = uniform(1);
  private readonly crop = uniform(new Vector2());
  private readonly texel = uniform(new Vector2());
  private readonly radiusScale = uniform(new Vector2());
  private readonly fullKernel = rings(9, 5, 2);
  private readonly wideKernel = rings(13, 8, 3);
  private readonly reducedKernel = rings(4, 3, 1);
  private readonly offsets = uniformArray<'vec2'>(this.fullKernel, 'vec2');
  private readonly sampleCount = uniform(16, 'int');

  constructor() {
    this.background.frustumCulled = false;
    this.background.renderOrder = -Infinity;
    this.background.userData.noHit = true;
    this.composite.vertexNode = vec4(positionGeometry.xy, 0, 1);

    const panoramaUV = (screen: Node<'vec2'>) => {
      const ray = this.projectionInverse.mul(vec4(screen.mul(2).sub(1), 1, 1));
      const direction = normalize(this.cameraToSkybox.mul(vec4(ray.xyz, 0)).xyz);
      return vec2(
          fract(float(.5).add(atan(direction.z, direction.x).div(2 * Math.PI))),
          float(.5).add(asin(clamp(direction.y, -1, 1)).div(Math.PI)));
    };
    const radiusAt = (panorama: Node<'vec2'>) => {
      const depthUV = vec2(panorama.x, mix(panorama.y, panorama.y.oneMinus(), this.flipDepthY));
      const value = vec4(this.depth.sample(depthUV)).toVar();
      const normalized = mix(value.r, value.r.add(value.g.mul(256)).div(257),
                             this.depthIs16Bit);
      return clamp(mix(normalized, normalized.oneMinus(), this.invert)
                       .sub(this.focus).mul(this.strength), -1, 1);
    };
    const colorAt = (panorama: Node<'vec2'>) => {
      const dx = panorama.dFdx();
      const dy = panorama.dFdy();
      const gradX = vec2(dx.x.sub(dx.x.round()), dx.y);
      const gradY = vec2(dy.x.sub(dy.x.round()), dy.y);
      const size = textureSize(this.color) as unknown as Node<'uvec2'>;
      const half = float(.5).div(size.x.toFloat());
      const left = half.oneMinus();
      const a = this.color.sample(vec2(left, panorama.y)).grad(gradX, gradY);
      const b = this.color.sample(vec2(half, panorama.y)).grad(gradX, gradY);
      const ordinary = this.color.sample(panorama).grad(gradX, gradY);
      const longitude = select(panorama.x.lessThan(.5), panorama.x, panorama.x.sub(1));
      return select(panorama.x.lessThan(half).or(panorama.x.greaterThan(left)),
                    mix(a, b, longitude.div(half.mul(2)).add(.5)), ordinary);
    };

    this.project.fragmentNode = Fn(() => {
      // QuadMesh UVs and WebGPU render-target rows run from top to bottom.
      const screen = vec2(uv().x, uv().y.oneMinus()).sub(.5).div(this.crop).add(.5);
      const delta = this.texel.mul(.25).div(this.crop);
      const coordinates = [[-1, -1], [1, -1], [-1, 1], [1, 1]].map(
          ([x, y]) => panoramaUV(screen.add(delta.mul(vec2(x, y)))));
      const radius = vec4(radiusAt(coordinates[0]), radiusAt(coordinates[1]),
                          radiusAt(coordinates[2]), radiusAt(coordinates[3]));
      const closest = min(min(radius.x, radius.y), min(radius.z, radius.w));
      const center = select(closest.lessThan(0), closest, radius.x);
      const weight = smoothstep(vec4(.5), vec4(1.5),
                                radius.sub(center).abs().mul(this.maxBlur)).oneMinus();
      const color = colorAt(coordinates[0]).rgb.mul(weight.x)
                        .add(colorAt(coordinates[1]).rgb.mul(weight.y))
                        .add(colorAt(coordinates[2]).rgb.mul(weight.z))
                        .add(colorAt(coordinates[3]).rgb.mul(weight.w));
      return vec4(color.div(dot(weight, vec4(1))), center);
    })();

    const projected = texture(this.projection.texture);
    this.gather.fragmentNode = Fn(() => {
      const screen = uv();
      const sample = projected.sample(screen).toVar();
      const radius = sample.a.mul(this.maxBlur);
      const center = sample.rgb;
      const fill = radius.lessThan(-.5);
      const search = select(fill, max(radius.negate(), this.maximumFar), radius);
      const taps = this.sampleCount.toFloat();
      const footprint = max(1, this.maximumNear.mul(.5).div(taps.sqrt()));
      const nearWeight = select(fill, max(1, radius.mul(radius)).reciprocal(), float(0)).toVar();
      const near = center.mul(nearWeight).toVar();
      const farWeight = select(radius.greaterThan(.5), float(1), float(0)).toVar();
      const far = center.mul(farWeight).toVar();
      const coverage = float(0).toVar();
      If(this.maximumNear.greaterThan(.5), () => {
        Loop({start: int(0), end: this.sampleCount, type: 'int'}, ({i}) => {
          const offset = this.offsets.element(i);
          const point = screen.add(offset.mul(this.maximumNear).mul(this.radiusScale));
          const neighbor = projected.sample(point).a.mul(this.maxBlur).negate();
          const overlap = smoothstep(footprint.negate(), footprint,
                                     neighbor.sub(this.maximumNear.mul(length(offset))));
          const weight = select(neighbor.greaterThan(.5), overlap.div(max(1, neighbor.mul(neighbor))), float(0));
          near.addAssign(projected.sample(point).rgb.mul(weight));
          nearWeight.addAssign(weight);
          coverage.addAssign(weight.mul(this.maximumNear.mul(this.maximumNear)).div(taps));
        });
      });
      const unresolved = smoothstep(.30, .70, radius.negate().div(max(.5, this.maximumNear))).oneMinus();
      const alpha = clamp(max(coverage, smoothstep(.5, 1.5, radius.negate()).mul(unresolved)), 0, 1);
      If(search.greaterThan(.5).and(fill.not().or(alpha.lessThan(.985))), () => {
        Loop({start: int(0), end: this.sampleCount, type: 'int'}, ({i}) => {
          const offset = this.offsets.element(i);
          const point = screen.add(offset.mul(search).mul(this.radiusScale));
          const neighbor = projected.sample(point).a.mul(this.maxBlur);
          const weight = select(fill, smoothstep(-1.5, -.5, neighbor),
                                step(.5, neighbor).mul(smoothstep(-1, 1, neighbor.sub(search.mul(length(offset))))));
          far.addAssign(projected.sample(point).rgb.mul(weight));
          farWeight.addAssign(weight);
        });
      });
      return select(this.layer.lessThan(.5),
                    vec4(near.mul(alpha.div(max(nearWeight, 1e-10))), alpha),
                    vec4(far.div(max(farWeight, 1e-10)), step(1e-10, farWeight)));
    })();

    const nearLayer = texture(this.near.texture);
    const farLayer = texture(this.far.texture);
    this.composite.fragmentNode = Fn(() => {
      const screen = viewportUV;
      const panorama = panoramaUV(vec2(screen.x, screen.y.oneMinus()));
      const sharp = colorAt(panorama);
      const radius = radiusAt(panorama).mul(this.maxBlur);
      const projected = screen.sub(.5).mul(this.crop).add(.5);
      const near = nearLayer.sample(projected);
      const far = farLayer.sample(projected);
      const defocus = smoothstep(.5, 1.5, radius.abs());
      const color = mix(sharp.rgb, far.rgb, far.a.mul(defocus))
                        .mul(near.a.oneMinus()).add(near.rgb);
      return vec4(color.mul(this.intensity), sharp.a.mul(this.opacity));
    })();
  }

  render(
      renderer: WebGPURenderer, scene: ModelScene, frame: SkyboxDofFrame,
      width: number, height: number) {
    if (frame.skybox.mapping !== EquirectangularReflectionMapping) {
      renderer.render(scene, scene.camera);
      return;
    }
    const reduced = scene.scaleStep > 0;
    const wide = frame.maxBlur > 16;
    const paddedWidth = Math.max(1, scene.width) + 2 * frame.maxBlur;
    const paddedHeight = Math.max(1, scene.height) + 2 * frame.maxBlur;
    const scale = Math.min(width / Math.max(1, scene.width) / (reduced ? 3 : 2),
                           height / Math.max(1, scene.height) / (reduced ? 3 : 2),
                           Math.sqrt(MAX_WORK_PIXELS / (paddedWidth * paddedHeight)));
    const workWidth = Math.max(1, Math.floor(paddedWidth * scale));
    const workHeight = Math.max(1, Math.floor(paddedHeight * scale));
    this.projection.setSize(workWidth, workHeight);
    this.near.setSize(workWidth, workHeight);
    this.far.setSize(workWidth, workHeight);
    this.cameraToSkybox.value.multiplyMatrices(
        this.skyboxRotation.makeRotationFromEuler(scene.backgroundRotation).transpose(),
        scene.camera.matrixWorld);
    this.projectionInverse.value.copy(scene.camera.projectionMatrixInverse);
    this.crop.value.set(scene.width / paddedWidth, scene.height / paddedHeight);
    this.texel.value.set(1 / workWidth, 1 / workHeight);
    this.radiusScale.value.set(1 / paddedWidth, 1 / paddedHeight);
    this.color.value = frame.skybox;
    this.depth.value = frame.depth;
    this.flipDepthY.value = frame.skybox.flipY ? 1 : 0;
    this.depthIs16Bit.value = frame.bitDepth === 16 ? 1 : 0;
    this.invert.value = frame.invert ? 1 : 0;
    this.focus.value = frame.focus;
    this.strength.value = frame.strength;
    this.maxBlur.value = frame.maxBlur;
    this.maximumNear.value = Math.min(1, frame.focus * frame.strength) * frame.maxBlur;
    this.maximumFar.value = Math.min(1, (1 - frame.focus) * frame.strength) * frame.maxBlur;
    this.opacity.value = frame.opacity;
    this.intensity.value = scene.backgroundIntensity;
    this.offsets.array = reduced ? this.reducedKernel : wide ? this.wideKernel : this.fullKernel;
    this.sampleCount.value = reduced ? 8 : wide ? 24 : 16;
    const hdr = ColorManagement.getTransfer(frame.skybox.colorSpace) !== SRGBTransfer;
    if (hdr !== this.composite.toneMapped) {
      this.composite.toneMapped = hdr;
      this.composite.needsUpdate = true;
    }

    const background = scene.background;
    const hidden = frame.hideDuringRender;
    const visible = hidden?.visible;
    const autoClear = renderer.autoClear;
    const target = renderer.getRenderTarget();
    const previousMRT = renderer.getMRT();
    try {
      renderer.setMRT(null);
      renderer.autoClear = false;
      renderer.setRenderTarget(this.projection);
      this.quad.material = this.project;
      this.quad.render(renderer);
      this.quad.material = this.gather;
      this.layer.value = 0;
      renderer.setRenderTarget(this.near);
      this.quad.render(renderer);
      this.layer.value = 1;
      renderer.setRenderTarget(this.far);
      this.quad.render(renderer);
      renderer.setRenderTarget(target);
      renderer.autoClear = autoClear;
      scene.background = null;
      if (hidden != null) hidden.visible = false;
      scene.add(this.background);
      renderer.render(scene, scene.camera);
    } finally {
      this.background.removeFromParent();
      scene.background = background;
      if (hidden != null) hidden.visible = visible!;
      renderer.setRenderTarget(target);
      renderer.setMRT(previousMRT);
      renderer.autoClear = autoClear;
    }
  }

  dispose() {
    this.background.geometry.dispose();
    this.projection.dispose();
    this.near.dispose();
    this.far.dispose();
    this.project.dispose();
    this.gather.dispose();
    this.composite.dispose();
  }
}
