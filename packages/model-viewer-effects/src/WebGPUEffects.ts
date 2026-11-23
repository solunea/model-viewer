/* @license Copyright 2026 Solunea. Licensed under the Apache License,
 * Version 2.0. */
import {ACESFilmicToneMapping, CineonToneMapping, Color, HalfFloatType, LinearToneMapping, NoToneMapping, ReinhardToneMapping, UnsignedByteType, Vector4} from 'three';
import type {Camera, Object3D, Scene, ToneMapping} from 'three';
import {blendBurn, blendDodge, blendOverlay, blendScreen, clamp, convertToTexture, float, hue, max, min, mix, mrt, normalView, output, pass, rand, saturation, screenSize, screenUV, step, time, toneMapping, toneMappingExposure, vec2, vec4} from 'three/tsl';
import {RenderPipeline} from 'three/webgpu';
import type {Node, PassNode, WebGPURenderer} from 'three/webgpu';
import {bloom} from 'three/addons/tsl/display/BloomNode.js';
import {outline} from 'three/addons/tsl/display/OutlineNode.js';
import {smaa} from 'three/addons/tsl/display/SMAANode.js';
import {ssao} from 'three/addons/tsl/display/SSAONode.js';

import {createJFAOutput} from './WebGPUJFA.js';
import type {WebGPUJFASettings} from './WebGPUJFA.js';

export type EffectElement = HTMLElement & {
  blendMode?: string;
  selection?: Array<Object3D|string>;
  strength?: number;
  threshold?: number;
  radius?: number;
  smoothing?: number;
  color?: string|number;
  granularity?: number;
  quality?: string;
  mode?: string;
  tonemapping?: string;
  hue?: number;
  saturation?: number;
  contrast?: number;
  brightness?: number;
};

export interface WebGPUEffectsSettings {
  quality: boolean;
  samples: number;
  effects: EffectElement[];
  custom: WebGPUJFASettings[];
}

const tones: Record<string, ToneMapping> = {
  LINEAR: LinearToneMapping,
  REINHARD: ReinhardToneMapping,
  REINHARD2: ReinhardToneMapping,
  REINHARD2_ADAPTIVE: ReinhardToneMapping,
  OPTIMIZED_CINEON: CineonToneMapping,
  ACES_FILMIC: ACESFilmicToneMapping
};

function selection(scene: Scene, effect: EffectElement) {
  const choices = effect.selection ?? [];
  const selected: Object3D[] = [];
  scene.traverse(object => {
    if ('geometry' in object && !object.userData.noHit &&
        (choices.length === 0 || choices.includes(object) || choices.includes(object.name))) {
      selected.push(object);
    }
  });
  return selected;
}

export function colorPass(
    scene: Scene, camera: Camera, predicate: (object: Object3D) => boolean) {
  const mask = pass(scene, camera, {type: HalfFloatType});
  const draw = mask.updateBefore.bind(mask);
  mask.updateBefore = frame => {
    const hidden: Object3D[] = [];
    const background = scene.background;
    const backgroundNode = (scene as Scene & {backgroundNode?: Node}).backgroundNode;
    const {renderer} = frame;
    if (renderer == null) return;
    const alpha = renderer.getClearAlpha();
    try {
      scene.background = null;
      (scene as Scene & {backgroundNode?: Node|null}).backgroundNode = null;
      renderer.setClearAlpha(0);
      scene.traverseVisible(object => {
        if ('geometry' in object && !predicate(object)) {
          object.visible = false;
          hidden.push(object);
        }
      });
      return draw(frame);
    } finally {
      renderer.setClearAlpha(alpha);
      scene.background = background;
      (scene as Scene & {backgroundNode?: Node}).backgroundNode = backgroundNode;
      for (const object of hidden) object.visible = true;
    }
  };
  return mask;
}

function blend(base: Node<'vec3'>, overlay: Node<'vec3'>, mode: string) {
  switch (mode) {
    case 'ADD': return base.add(overlay);
    case 'MULTIPLY': return base.mul(overlay);
    case 'SCREEN': return blendScreen(base, overlay);
    case 'OVERLAY': return blendOverlay(base, overlay);
    case 'COLOR_DODGE': return blendDodge(base, overlay);
    case 'COLOR_BURN': return blendBurn(base, overlay);
    case 'DARKEN': return min(base, overlay);
    case 'LIGHTEN': return max(base, overlay);
    case 'DIFFERENCE': return base.sub(overlay).abs();
    case 'SUBTRACT': return base.sub(overlay);
    default: return overlay;
  }
}

function optionsOf(effect: EffectElement) {
  const {localName, blendMode, strength, threshold, radius, smoothing, color,
         granularity, quality, mode, tonemapping, hue, saturation, contrast, brightness} = effect;
  return [localName, blendMode, strength, threshold, radius, smoothing, color,
          granularity, quality, mode, tonemapping, hue, saturation, contrast, brightness];
}

export class WebGPUEffects {
  private readonly pipeline: RenderPipeline;
  private resources: Array<{dispose(): void}> = [];
  private scene: Scene|null = null;
  private camera: Camera|null = null;
  private viewport = new Vector4();
  private signature = '';
  private colorGrade = false;

  constructor(private readonly renderer: WebGPURenderer) {
    this.pipeline = new RenderPipeline(renderer);
  }

  private keep<T extends {dispose(): void}>(resource: T): T {
    this.resources.push(resource);
    return resource;
  }

  private build(scene: Scene, camera: Camera, settings: WebGPUEffectsSettings) {
    for (const resource of this.resources) resource.dispose();
    this.resources = [];
    const effects = settings.effects.filter(element => element.blendMode !== 'SKIP');
    this.colorGrade = effects.some(element => element.localName === 'color-grade-effect');
    const scenePass = this.keep(pass(scene, camera, {
      type: settings.quality ? HalfFloatType : UnsignedByteType,
      samples: settings.samples
    }));
    if (effects.some(element => element.localName === 'ssao-effect')) {
      scenePass.setMRT(mrt({output, normal: normalView}));
    }
    let result: Node<'vec4'> = scenePass.getTextureNode();
    for (const options of settings.custom) {
      const mask = this.keep(colorPass(scene, camera, object => !object.userData.noHit));
      const effect = createJFAOutput(result, mask.getTextureNode(), options);
      for (const target of effect.resources) this.keep(target);
      result = effect.output;
    }
    for (const element of effects) {
      result = this.apply(result, scenePass, scene, camera, element);
    }
    this.pipeline.outputNode = result;
    this.pipeline.needsUpdate = true;
  }

  private apply(
      input: Node<'vec4'>, scenePass: PassNode, scene: Scene, camera: Camera,
      effect: EffectElement): Node<'vec4'> {
    const mode = effect.blendMode ?? 'DEFAULT';
    const strength = effect.strength ?? 1;
    switch (effect.localName) {
      case 'selective-bloom-effect':
      case 'bloom-effect': {
        let source = input;
        if (effect.localName === 'selective-bloom-effect') {
          const selected = new Set(selection(scene, effect));
          source = this.keep(colorPass(scene, camera, mesh => selected.has(mesh))).getTextureNode();
        }
        const bloomNode = bloom(source, strength, effect.radius ?? .85, effect.threshold ?? .85);
        this.keep(bloomNode);
        bloomNode.smoothWidth.value = effect.smoothing ?? .025;
        const glow = bloomNode.rgb;
        return vec4(blend(input.rgb, glow, mode === 'DEFAULT' ? 'ADD' : mode), input.a);
      }
      case 'outline-effect': {
        const edge = this.keep(outline(scene, camera, {
          selectedObjects: selection(scene, effect),
          edgeThickness: float(effect.smoothing ?? 1)
        }));
        const alpha = clamp(edge.visibleEdge.mul(strength), 0, 1);
        const color = vec4(new Color(effect.color ?? '#ffffff'), 1);
        return vec4(mix(input.rgb, blend(input.rgb, color.rgb, mode), alpha),
                    max(input.a, alpha));
      }
      case 'ssao-effect': {
        const ambient = ssao(scenePass.getTextureNode('depth'),
                             scenePass.getTextureNode('normal'), camera);
        this.keep(ambient);
        ambient.intensity.value = strength;
        return vec4(input.rgb.mul(ambient.getTextureNode().r), input.a);
      }
      case 'smaa-effect': {
        const aa = this.keep(smaa(input));
        return aa.getTextureNode();
      }
      case 'pixelate-effect': {
        const texture = this.keep(convertToTexture(input));
        const stepSize = Math.max(1, effect.granularity ?? 10);
        const point = screenUV.mul(screenSize).div(stepSize).floor()
                          .add(.5).mul(stepSize).div(screenSize);
        return texture.sample(point);
      }
      case 'color-grade-effect': {
        const mapping = tones[effect.tonemapping ?? 'ACES_FILMIC'] ?? ACESFilmicToneMapping;
        const mapped = toneMapping(mapping, toneMappingExposure, input.rgb);
        const adjusted = saturation(hue(mapped, effect.hue ?? 0), 1 + (effect.saturation ?? 0));
        const rgb = adjusted.sub(.5).mul(1 + (effect.contrast ?? 0)).add(.5 + (effect.brightness ?? 0));
        return vec4(blend(input.rgb, rgb, mode), input.a);
      }
      case 'glitch-effect': {
        const source = this.keep(convertToTexture(input));
        const pulse = effect.mode === 'constant' ? float(1) : step(.72, time.div(2.5).fract());
        const amount = rand(vec2(time.mul(24).floor(), screenUV.y.mul(48).floor()))
                           .sub(.5).mul(strength * .18).mul(pulse);
        const offset = vec2(amount, 0);
        const color = source.sample(screenUV.add(offset));
        return vec4(source.sample(screenUV.add(offset.mul(1.1))).r, color.g,
                    source.sample(screenUV.add(offset.mul(.9))).b, color.a);
      }
      default: return input;
    }
  }

  render(scene: Scene, camera: Camera, settings: WebGPUEffectsSettings) {
    const viewport = this.renderer.getViewport(this.viewport);
    const signature = JSON.stringify([
      scene.uuid, camera.uuid, settings.quality, settings.samples, viewport.z, viewport.w,
      settings.effects.map(effect => [
        ...optionsOf(effect),
        effect.selection?.map(entry => typeof entry === 'string' ? entry : entry.uuid),
        effect.localName === 'outline-effect' || effect.localName === 'selective-bloom-effect' ?
            selection(scene, effect).map(mesh => mesh.uuid) : null
      ]),
      settings.custom.map(effect => [
        effect.mode, effect.thickness, effect.inside, effect.color.getHex(), effect.blendMode
      ])
    ]);
    if (this.scene !== scene || this.camera !== camera || signature !== this.signature) {
      this.build(scene, camera, settings);
      this.scene = scene;
      this.camera = camera;
      this.signature = signature;
    }
    this.renderer.getViewport(this.viewport);
    const toneMapping = this.renderer.toneMapping;
    if (this.colorGrade) this.renderer.toneMapping = NoToneMapping;
    try {
      this.pipeline.render();
    } finally {
      this.renderer.toneMapping = toneMapping;
    }
  }

  dispose() {
    for (const resource of this.resources) resource.dispose();
    this.resources = [];
    this.pipeline.dispose();
  }
}
