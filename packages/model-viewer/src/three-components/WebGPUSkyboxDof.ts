/* @license Copyright 2026 Solunea. Licensed under the Apache License,
 * Version 2.0. */
import {BufferAttribute, BufferGeometry, ColorManagement, DataTexture, EquirectangularReflectionMapping, HalfFloatType, LinearFilter, Matrix4, Mesh, NoBlending, NoToneMapping, RedFormat, RenderTarget, SRGBTransfer, Texture, Vector2, Vector3, Vector4} from 'three';
import type {ToneMapping} from 'three';
import {Fn, If, Loop, asin, atan, clamp, dot, float, fract, int, max, min, mix, normalize, outputStruct, positionGeometry, property, renderOutput, select, smoothstep, step, texture, textureSize, uniform, uniformArray, uv, varying, vec2, vec4, viewportUV} from 'three/tsl';
import {NodeMaterial, QuadMesh, WebGPURenderer} from 'three/webgpu';
import type {Node} from 'three/webgpu';

import type {ModelScene, SkyboxDofFrame} from './ModelScene.js';
import {rings} from './SkyboxDof.js';

const MAX_WORK_PIXELS = 520000;
const CLIP_EDGES = [-1, 1];

interface SkyboxProjection {
  cameraToSkybox: Matrix4;
  projectionInverse: Matrix4;
}

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

function gatherKernel(outer: number, middle: number, inner: number) {
  // The uniform array already pads each entry to four floats.
  return rings(outer, middle, inner).map(
      offset => new Vector3(offset.x, offset.y, offset.length()));
}

export function canRenderSkyboxDirectly(
    scene: ModelScene, renderer: WebGPURenderer,
    transition: ModelScene['skyboxTransitionMesh'] = null): boolean {
  if (!scene.element.skyboxOnly || scene.element.viewHelper ||
      (scene.skyboxTransitionActive && transition == null) ||
      scene.overrideMaterial != null ||
      scene.fog != null || renderer.getRenderTarget() != null ||
      renderer.getOutputRenderTarget() != null) {
    return false;
  }
  let geometryVisible = false;
  scene.traverseVisible(object => {
    const {geometry, material} = object as Mesh;
    if (geometryVisible || object === transition || geometry == null ||
        !object.layers.test(scene.camera.layers)) return;
    // An invisible material can still provide a mesh for raycasting.
    geometryVisible = Array.isArray(material) ?
        material.some(entry => entry.visible) : material?.visible !== false;
  });
  return !geometryVisible;
}

export class WebGPUSkyboxDof {
  private readonly projection = target('projectedColorAndRadius');
  private readonly layers = new RenderTarget(1, 1, {
    count: 2, type: HalfFloatType, depthBuffer: false
  });
  private readonly project = passMaterial();
  private readonly gather = passMaterial();
  private readonly composite = passMaterial();
  private readonly directComposite = passMaterial();
  private readonly quad = new QuadMesh(this.project);
  private directToneMapping: ToneMapping|null = null;
  private directColorSpace: string|null = null;
  private readonly background = new Mesh(
      new BufferGeometry().setAttribute('position', new BufferAttribute(
          new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3)),
      this.composite);
  private readonly cameraToSkybox = uniform(new Matrix4());
  private readonly projectionInverse = uniform(new Matrix4());
  private readonly lastProjection = new WeakMap<ModelScene, SkyboxProjection>();
  private readonly skyboxRotation = new Matrix4();
  private readonly cameraRotation = new Matrix4();
  private readonly blurProjection: SkyboxProjection = {
    cameraToSkybox: new Matrix4(),
    projectionInverse: new Matrix4()
  };
  private readonly cachedSkyboxToClip = new Matrix4();
  private readonly blurReprojection = uniform(new Matrix4());
  private readonly reprojectBlur = uniform(0);
  private blurInputs: number[]|null = null;
  private readonly color = texture(new Texture());
  private readonly depth = texture(new Texture());
  private readonly cameraToWorld = uniform(new Matrix4());
  private readonly emptyMask = new DataTexture(new Uint8Array(1), 1, 1, RedFormat);
  private readonly focusMask = texture(this.emptyMask);
  private readonly focusDimensions = uniform(new Vector2());
  private readonly focusBounds = uniform(new Vector4());
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
  private readonly fullKernel = gatherKernel(9, 5, 2);
  private readonly wideKernel = gatherKernel(13, 8, 3);
  private readonly reducedKernel = gatherKernel(4, 3, 1);
  private readonly offsets = uniformArray<'vec3'>(this.fullKernel, 'vec3');
  private readonly sampleCount = uniform(16, 'int');

  constructor() {
    // A separate fallback prevents TSL from merging the mask and depth bindings.
    this.emptyMask.minFilter = this.emptyMask.magFilter = LinearFilter;
    this.emptyMask.needsUpdate = true;
    this.background.frustumCulled = false;
    this.background.renderOrder = -Infinity;
    this.background.userData.noHit = true;
    this.composite.vertexNode = vec4(positionGeometry.xy, 0, 1);

    const viewRay = (screen: Node<'vec2'>) =>
      vec4(this.projectionInverse.mul(vec4(screen.mul(2).sub(1), 1, 1)).xyz, 0);
    const directionUV = (ray: Node<'vec3'>) => {
      const direction = normalize(ray);
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
      const result = this.color.sample(panorama).grad(gradX, gradY).toVar();
      If(panorama.x.lessThan(half).or(panorama.x.greaterThan(left)), () => {
        const a = this.color.sample(vec2(left, panorama.y)).grad(gradX, gradY);
        const b = this.color.sample(vec2(half, panorama.y)).grad(gradX, gradY);
        const longitude = select(panorama.x.lessThan(.5), panorama.x, panorama.x.sub(1));
        result.assign(mix(a, b, longitude.div(half.mul(2)).add(.5)));
      });
      return result;
    };

    // The four sampling rays are affine over the fullscreen triangle.
    // QuadMesh UVs and WebGPU render-target rows run from top to bottom.
    const projectScreen = vec2(uv().x, uv().y.oneMinus()).sub(.5).div(this.crop).add(.5);
    const sampleDelta = this.texel.mul(.25).div(this.crop);
    const sampleRays = [[-1, -1], [1, -1], [-1, 1], [1, 1]].map(
        ([x, y]) => varying(this.cameraToSkybox.mul(
            viewRay(projectScreen.add(sampleDelta.mul(vec2(x, y))))).xyz));
    this.project.fragmentNode = Fn(() => {
      const coordinates = sampleRays.map(directionUV);
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

    const [nearTexture, farTexture] = this.layers.textures;
    nearTexture.name = 'nearLayer';
    farTexture.name = 'farLayer';
    const projected = texture(this.projection.texture);
    const farOutput = property('vec4');
    const nearOutput = Fn(() => {
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
        const scale = this.radiusScale.mul(this.maximumNear).toVar();
        const radiusSquared = this.maximumNear.mul(this.maximumNear).toVar();
        Loop({start: int(0), end: this.sampleCount, type: 'int'}, ({i}) => {
          const offset = this.offsets.element(i);
          const point = screen.add(offset.xy.mul(scale));
          const sample = projected.sample(point).toVar();
          const neighbor = sample.a.mul(this.maxBlur).negate();
          const overlap = smoothstep(footprint.negate(), footprint,
                                     neighbor.sub(this.maximumNear.mul(offset.z)));
          const weight = select(neighbor.greaterThan(.5), overlap.div(max(1, neighbor.mul(neighbor))), float(0));
          near.addAssign(sample.rgb.mul(weight));
          nearWeight.addAssign(weight);
          coverage.addAssign(weight.mul(radiusSquared).div(taps));
        });
      });
      const unresolved = smoothstep(.30, .70, radius.negate().div(max(.5, this.maximumNear))).oneMinus();
      const alpha = clamp(max(coverage, smoothstep(.5, 1.5, radius.negate()).mul(unresolved)), 0, 1);
      If(search.greaterThan(.5).and(fill.not().or(alpha.lessThan(.985))), () => {
        const scale = this.radiusScale.mul(search).toVar();
        Loop({start: int(0), end: this.sampleCount, type: 'int'}, ({i}) => {
          const offset = this.offsets.element(i);
          const point = screen.add(offset.xy.mul(scale));
          const sample = projected.sample(point).toVar();
          const neighbor = sample.a.mul(this.maxBlur);
          const weight = select(fill, smoothstep(-1.5, -.5, neighbor),
                                step(.5, neighbor).mul(smoothstep(-1, 1, neighbor.sub(search.mul(offset.z)))));
          far.addAssign(sample.rgb.mul(weight));
          farWeight.addAssign(weight);
        });
      });
      farOutput.assign(vec4(far.div(max(farWeight, 1e-10)), step(1e-10, farWeight)));
      return vec4(near.mul(alpha.div(max(nearWeight, 1e-10))), alpha);
    })();
    // The far blur uses the near coverage. Write both results from that
    // shared calculation while the projected samples are still available.
    this.gather.fragmentNode = outputStruct(nearOutput, farOutput);

    const nearLayer = texture(nearTexture);
    const farLayer = texture(farTexture);
    const clipPosition = vec4(positionGeometry.xy, 1, 1);
    const ray = vec4(this.projectionInverse.mul(clipPosition).xyz, 0);
    // These transforms are linear in screen coordinates across the triangle.
    const skyboxRay = varying(this.cameraToSkybox.mul(ray).xyz);
    const worldRay = varying(this.cameraToWorld.mul(ray).xyz);
    const blurClip = varying(this.blurReprojection.mul(clipPosition));
    this.composite.fragmentNode = Fn(() => {
      const screen = viewportUV;
      const panorama = directionUV(skyboxRay);
      const sharp = colorAt(panorama);
      const radius = radiusAt(panorama).mul(this.maxBlur);
      const blurScreen = screen.toVar();
      If(this.reprojectBlur.greaterThan(0), () => {
        blurScreen.assign(vec2(
            blurClip.x.div(blurClip.w).mul(.5).add(.5),
            blurClip.y.div(blurClip.w).mul(-.5).add(.5)));
      });
      const projected = blurScreen.sub(.5).mul(this.crop).add(.5);
      const near = nearLayer.sample(projected);
      const far = farLayer.sample(projected);
      const defocus = smoothstep(.5, 1.5, radius.abs());
      const color = mix(sharp.rgb, far.rgb, far.a.mul(defocus))
                        .mul(near.a.oneMinus()).add(near.rgb).toVar();
      If(this.focusBounds.z.greaterThan(0), () => {
        const worldUV = directionUV(worldRay);
        const maskUV = vec2(worldUV.x, worldUV.y.oneMinus()).mul(this.focusDimensions)
                           .sub(this.focusBounds.xy).div(this.focusBounds.zw);
        const size = textureSize(this.focusMask) as unknown as Node<'vec2'>;
        const amount = this.focusMask.sample(maskUV.mul(size.sub(2)).add(1).div(size)).r;
        color.assign(mix(color, sharp.rgb, amount));
      });
      return vec4(color.mul(this.intensity), sharp.a.mul(this.opacity));
    })();
  }

  private canReprojectBlur(
      cameraToSkybox: Matrix4, workWidth: number, workHeight: number) {
    const {value: projection} = this.blurReprojection;
    this.cameraRotation.copy(cameraToSkybox).setPosition(0, 0, 0);
    projection.copy(this.cachedSkyboxToClip).multiply(this.cameraRotation)
        .multiply(this.projectionInverse.value);
    const {x: cropX, y: cropY} = this.crop.value;
    // Bound the warp to one blur texel and keep every corner inside the padding.
    const maxX = Math.min(2 / (workWidth * cropX), 1 / cropX - 1);
    const maxY = Math.min(2 / (workHeight * cropY), 1 / cropY - 1);
    const matrix = projection.elements;
    for (const y of CLIP_EDGES) {
      for (const x of CLIP_EDGES) {
        const w = matrix[3] * x + matrix[7] * y + matrix[11] + matrix[15];
        const oldX = (matrix[0] * x + matrix[4] * y + matrix[8] + matrix[12]) / w;
        const oldY = (matrix[1] * x + matrix[5] * y + matrix[9] + matrix[13]) / w;
        if (!(w > 0 && Math.abs(oldX - x) <= maxX && Math.abs(oldY - y) <= maxY)) {
          return false;
        }
      }
    }
    return true;
  }

  private renderDirect(renderer: WebGPURenderer) {
    const {toneMapping, outputColorSpace} = renderer;
    if (this.directToneMapping !== toneMapping ||
        this.directColorSpace !== outputColorSpace) {
      // Match the premultiplied conversion of the framebuffer output pass.
      this.directComposite.fragmentNode =
          renderOutput(this.composite.fragmentNode!, toneMapping, outputColorSpace);
      this.directComposite.needsUpdate = true;
      this.directToneMapping = toneMapping;
      this.directColorSpace = outputColorSpace;
    }
    renderer.toneMapping = NoToneMapping;
    renderer.outputColorSpace = ColorManagement.workingColorSpace;
    this.quad.material = this.directComposite;
    try {
      this.quad.render(renderer);
    } finally {
      renderer.toneMapping = toneMapping;
      renderer.outputColorSpace = outputColorSpace;
    }
  }

  render(
      renderer: WebGPURenderer, scene: ModelScene, frame: SkyboxDofFrame,
      width: number, height: number) {
    if (frame.skybox.mapping !== EquirectangularReflectionMapping) {
      renderer.render(scene, scene.camera);
      return false;
    }
    const cameraToSkybox = this.cameraToSkybox.value.multiplyMatrices(
        this.skyboxRotation.makeRotationFromEuler(scene.backgroundRotation).transpose(),
        scene.camera.matrixWorld);
    const projectionInverse = this.projectionInverse.value.copy(
        scene.camera.projectionMatrixInverse);
    const previous = this.lastProjection.get(scene);
    const moved = previous != null &&
        (!previous.cameraToSkybox.equals(cameraToSkybox) ||
         !previous.projectionInverse.equals(projectionInverse));
    if (previous == null) {
      this.lastProjection.set(scene, {
        cameraToSkybox: cameraToSkybox.clone(),
        projectionInverse: projectionInverse.clone()
      });
    } else if (moved) {
      previous.cameraToSkybox.copy(cameraToSkybox);
      previous.projectionInverse.copy(projectionInverse);
    }
    const reduced = moved || scene.scaleStep > 0;
    const wide = frame.maxBlur > 16;
    const paddedWidth = Math.max(1, scene.width) + 2 * frame.maxBlur;
    const paddedHeight = Math.max(1, scene.height) + 2 * frame.maxBlur;
    const scale = Math.min(width / Math.max(1, scene.width) / (reduced ? 3 : 2),
                           height / Math.max(1, scene.height) / (reduced ? 3 : 2),
                           Math.sqrt(MAX_WORK_PIXELS / (paddedWidth * paddedHeight)));
    const workWidth = Math.max(1, Math.floor(paddedWidth * scale));
    const workHeight = Math.max(1, Math.floor(paddedHeight * scale));
    this.crop.value.set(scene.width / paddedWidth, scene.height / paddedHeight);
    this.color.value = frame.skybox;
    this.depth.value = frame.depth;
    const {focusMask} = frame;
    this.focusMask.value = focusMask?.texture ?? this.emptyMask;
    if (focusMask != null) {
      this.cameraToWorld.value.copy(scene.camera.matrixWorld);
      this.focusDimensions.value.copy(focusMask.dimensions);
      this.focusBounds.value.copy(focusMask.bounds);
    } else {
      this.focusBounds.value.set(0, 0, 0, 0);
    }
    this.flipDepthY.value = frame.skybox.flipY ? 1 : 0;
    this.depthIs16Bit.value = frame.bitDepth === 16 ? 1 : 0;
    this.invert.value = frame.invert ? 1 : 0;
    this.focus.value = frame.focus;
    this.strength.value = frame.strength;
    this.maxBlur.value = frame.maxBlur;
    this.opacity.value = frame.opacity;
    this.intensity.value = scene.backgroundIntensity;
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
    const direct = canRenderSkyboxDirectly(scene, renderer);
    const inputs = [
      scene.id, scene.width, scene.height, workWidth, workHeight, +reduced,
      frame.skybox.id, frame.skybox.version, frame.skybox.source.version, +frame.skybox.flipY,
      frame.depth.id, frame.depth.version, frame.depth.source.version, +frame.depth.flipY,
      frame.bitDepth, frame.focus, frame.strength, frame.maxBlur, +frame.invert,
    ];
    const previousInputs = this.blurInputs;
    const sameInputs = previousInputs != null &&
        inputs.every((input, index) => input === previousInputs[index]);
    // The mask and scene overlays affect only the final composition.
    const sameProjection = sameInputs &&
        this.blurProjection.cameraToSkybox.equals(cameraToSkybox) &&
        this.blurProjection.projectionInverse.equals(projectionInverse);
    const highlightedPanorama = scene.element.skyboxOnly &&
        scene.panoramaAreas?.visible && focusMask == null &&
        !scene.skyboxTransitionActive;
    const reproject = sameInputs && !sameProjection && moved &&
        (direct || highlightedPanorama) &&
        this.canReprojectBlur(cameraToSkybox, workWidth, workHeight);
    this.reprojectBlur.value = reproject ? 1 : 0;
    try {
      renderer.setMRT(null);
      if (!sameProjection && !reproject) {
        this.blurInputs = null;
        this.texel.value.set(1 / workWidth, 1 / workHeight);
        this.radiusScale.value.set(1 / paddedWidth, 1 / paddedHeight);
        this.maximumNear.value = Math.min(1, frame.focus * frame.strength) * frame.maxBlur;
        this.maximumFar.value = Math.min(1, (1 - frame.focus) * frame.strength) * frame.maxBlur;
        this.offsets.array = reduced ? this.reducedKernel : wide ? this.wideKernel : this.fullKernel;
        this.sampleCount.value = reduced ? 8 : wide ? 24 : 16;
        this.projection.setSize(workWidth, workHeight);
        this.layers.setSize(workWidth, workHeight);
        renderer.autoClear = false;
        renderer.setRenderTarget(this.projection);
        this.quad.material = this.project;
        this.quad.render(renderer);
        this.quad.material = this.gather;
        renderer.setRenderTarget(this.layers);
        this.quad.render(renderer);
        this.blurProjection.cameraToSkybox.copy(cameraToSkybox);
        this.blurProjection.projectionInverse.copy(projectionInverse);
        this.cameraRotation.copy(cameraToSkybox).setPosition(0, 0, 0).invert();
        this.cachedSkyboxToClip.copy(scene.camera.projectionMatrix)
            .multiply(this.cameraRotation);
        this.blurInputs = inputs;
      }
      renderer.setRenderTarget(target);
      renderer.autoClear = autoClear;
      if (direct) {
        this.renderDirect(renderer);
        return moved;
      }
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
      this.focusMask.value = this.emptyMask;
    }
    return moved;
  }

  dispose() {
    this.emptyMask.dispose();
    this.background.geometry.dispose();
    this.projection.dispose();
    this.layers.dispose();
    this.project.dispose();
    this.gather.dispose();
    this.composite.dispose();
    this.directComposite.dispose();
  }
}
