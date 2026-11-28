import {BufferAttribute, BufferGeometry, Camera, ColorManagement, EquirectangularReflectionMapping, Euler, GLSL3, HalfFloatType, LinearFilter, Matrix4, Mesh, NearestFilter, NoBlending, NoColorSpace, RGFormat, Scene, ShaderMaterial, SRGBColorSpace, SRGBTransfer, Texture, UnsignedByteType, Vector2, Vector4, WebGLRenderer, WebGLRenderTarget} from 'three';

import type {ModelScene, SkyboxDofFrame} from './ModelScene.js';
import {skyboxSourceFlipY} from './SkyboxDepth.js';

const MAX_WORK_PIXELS = 520000;

const vertex = `
out vec2 uvScreen;

void main() {
  uvScreen = position.xy * .5 + .5;
  gl_Position = vec4(position.xy, 0., 1.);
}
`;

const panorama = `
in vec2 uvScreen;
uniform sampler2D colorTexture;
uniform sampler2D depthTexture;
uniform mat4 cameraToSkybox;
uniform mat4 projectionInverse;
uniform float flipDepthY;
uniform float depthIs16Bit;
uniform float invertDepth;
uniform float focus;
uniform float strength;

vec3 viewRay(vec2 screen) {
  return (projectionInverse * vec4(screen * 2. - 1., 1., 1.)).xyz;
}

vec2 directionUv(vec3 ray) {
  vec3 direction = normalize(ray);
  float u = dot(direction.xz, direction.xz) < 1e-12 ?
      .5 : .5 + atan(direction.z, direction.x) * .15915494309;
  return vec2(fract(u), .5 + asin(clamp(direction.y, -1., 1.)) * .31830988618);
}

vec2 panoramaUv(vec2 screen) {
  return directionUv(mat3(cameraToSkybox) * viewRay(screen));
}

float circleOfConfusion(vec2 uv) {
  vec2 depthUv = vec2(uv.x, mix(uv.y, 1. - uv.y, flipDepthY));
  vec2 sampleValue = texture(depthTexture, depthUv).rg;
  float depth = depthIs16Bit == 1. ?
      dot(sampleValue, vec2(1., 256.)) / 257. : sampleValue.r;
  depth = mix(depth, 1. - depth, invertDepth);
  return clamp((depth - focus) * strength, -1., 1.);
}

vec4 panoramaColor(vec2 uv) {
  vec2 dx = dFdx(uv);
  vec2 dy = dFdy(uv);
  // Derivatives should represent the short way across the longitude seam.
  dx.x -= round(dx.x);
  dy.x -= round(dy.x);
  float halfTexel = .5 / float(textureSize(colorTexture, 0).x);
  if (uv.x < halfTexel || uv.x > 1. - halfTexel) {
    float longitude = uv.x < .5 ? uv.x : uv.x - 1.;
    vec4 left = textureGrad(colorTexture, vec2(1. - halfTexel, uv.y), dx, dy);
    vec4 right = textureGrad(colorTexture, vec2(halfTexel, uv.y), dx, dy);
    return mix(left, right, longitude / (2. * halfTexel) + .5);
  }
  return textureGrad(colorTexture, uv, dx, dy);
}
`;

const radii = `
// Keep the signed radius in a two-channel UNORM target on ordinary WebGL 2
// devices. Keep zero exactly representable and sample the target without filtering.
vec2 packRadius(float radius) {
  float value = floor(radius * 32767. + 32767.5);
  return vec2(mod(value, 256.), floor(value / 256.)) / 255.;
}

float unpackRadius(vec2 value) {
  return (dot(round(value * 255.), vec2(1., 256.)) - 32767.) / 32767.;
}
`;

const project = `
${panorama}
${radii}
uniform vec2 crop;
uniform vec2 texelSize;
uniform float maxBlur;
layout(location = 0) out vec4 projectedColor;
layout(location = 1) out vec4 projectedRadius;

void main() {
  vec2 screen = (uvScreen - .5) / crop + .5;
  vec2 delta = texelSize * .25 / crop;
  vec2 uv0 = panoramaUv(screen + vec2(-delta.x, -delta.y));
  vec2 uv1 = panoramaUv(screen + vec2( delta.x, -delta.y));
  vec2 uv2 = panoramaUv(screen + vec2(-delta.x,  delta.y));
  vec2 uv3 = panoramaUv(screen + vec2( delta.x,  delta.y));
  vec4 radius = vec4(circleOfConfusion(uv0), circleOfConfusion(uv1),
                     circleOfConfusion(uv2), circleOfConfusion(uv3));
  // Protect a foreground silhouette even if it covers only one corner.
  float closest = min(min(radius.x, radius.y), min(radius.z, radius.w));
  float representative = closest < 0. ? closest : radius.x;
  vec4 weight = 1. - smoothstep(vec4(.5), vec4(1.5),
                                abs(radius - representative) * maxBlur);
  vec3 color = panoramaColor(uv0).rgb * weight.x +
      panoramaColor(uv1).rgb * weight.y + panoramaColor(uv2).rgb * weight.z +
      panoramaColor(uv3).rgb * weight.w;
  projectedColor = vec4(color / dot(weight, vec4(1.)), 1.);
  projectedRadius = vec4(packRadius(representative), 0., 1.);
}
`;

const gather = `
in vec2 uvScreen;
uniform sampler2D projectedColor;
uniform sampler2D projectedRadius;
uniform vec2 radiusScale;
uniform vec2 usedArea;
uniform vec2 samples[24];
uniform int tapCount;
uniform float maxBlur;
uniform float maximumNear;
uniform float maximumFar;
${radii}
layout(location = 0) out vec4 nearLayer;
layout(location = 1) out vec4 farLayer;

vec2 targetUv(vec2 uv) {
  vec2 halfTexel = .5 / vec2(textureSize(projectedRadius, 0));
  return clamp(uv * usedArea, halfTexel, usedArea - halfTexel);
}

float radiusAt(vec2 uv) {
  return unpackRadius(texture(projectedRadius, targetUv(uv)).rg) * maxBlur;
}

void main() {
  float centerRadius = radiusAt(uvScreen);
  vec3 center = texture(projectedColor, targetUv(uvScreen)).rgb;
  bool fillsBehindNear = centerRadius < -.5;
  float farSearch = fillsBehindNear ? max(-centerRadius, maximumFar) : centerRadius;
  float nearFootprint = max(1., .50 * maximumNear / sqrt(float(tapCount)));
  vec3 nearSum = vec3(0.);
  vec3 farSum = vec3(0.);
  float nearWeight = 0.;
  float farWeight = 0.;
  float coverage = 0.;

  if (centerRadius > .5) {
    farSum = center;
    farWeight = 1.;
  }
  if (centerRadius < -.5) {
    float centerWeight = 1. / max(1., centerRadius * centerRadius);
    nearWeight = centerWeight;
    nearSum = center * centerWeight;
  }

  if (maximumNear > .5) {
    for (int i = 0; i < tapCount; ++i) {
      vec2 offset = samples[i];
      float fraction = length(offset);
      vec2 uv = uvScreen + offset * maximumNear * radiusScale;
      float neighborRadius = -radiusAt(uv);
      float overlap = smoothstep(-nearFootprint, nearFootprint,
                                 neighborRadius - maximumNear * fraction);
      if (neighborRadius > .5 && overlap > 0.) {
        float weight = overlap / max(1., neighborRadius * neighborRadius);
        nearSum += texture(projectedColor, targetUv(uv)).rgb * weight;
        nearWeight += weight;
        coverage += weight * maximumNear * maximumNear / float(tapCount);
      }
    }
  }
  float unresolved = 1. - smoothstep(.30, .70, -centerRadius / max(.5, maximumNear));
  float centerCoverage = smoothstep(.5, 1.5, -centerRadius) * unresolved;
  float alpha = clamp(max(coverage, centerCoverage), 0., 1.);

  if (farSearch > .5 && (!fillsBehindNear || alpha < .985)) {
    for (int i = 0; i < tapCount; ++i) {
      vec2 offset = samples[i];
      float fraction = length(offset);
      vec2 uv = uvScreen + offset * farSearch * radiusScale;
      float neighborRadius = radiusAt(uv);
      // A focused background texel can still fill the color hidden behind a
      // blurred foreground silhouette.
      float weight = fillsBehindNear ? smoothstep(-1.5, -.5, neighborRadius) :
          step(.5, neighborRadius) *
          smoothstep(-1., 1., neighborRadius - farSearch * fraction);
      if (weight > 0.) {
        farSum += texture(projectedColor, targetUv(uv)).rgb * weight;
        farWeight += weight;
      }
    }
  }
  // The near layer stores its coverage separately from its blurred color.
  nearLayer = nearWeight > 0. ? vec4(nearSum * (alpha / nearWeight), alpha) : vec4(0.);
  farLayer = farWeight > 0. ? vec4(farSum / farWeight, 1.) : vec4(0.);
}
`;

const composite = `
${panorama}
uniform sampler2D nearLayer;
uniform sampler2D farLayer;
uniform vec2 crop;
uniform vec2 usedArea;
uniform float opacity;
uniform float maxBlur;
uniform float backgroundIntensity;
uniform mat4 cameraToWorld;
uniform sampler2D focusMask;
uniform vec2 focusDimensions;
uniform vec4 focusBounds;

out vec4 result;
#define gl_FragColor result

void main() {
  vec3 ray = viewRay(uvScreen);
  vec2 uv = directionUv(mat3(cameraToSkybox) * ray);
  vec4 sharp = panoramaColor(uv);
  float radius = circleOfConfusion(uv) * maxBlur;
  vec2 halfTexel = .5 / vec2(textureSize(nearLayer, 0));
  vec2 projectedUv = clamp(((uvScreen - .5) * crop + .5) * usedArea,
                           halfTexel, usedArea - halfTexel);
  vec4 near = texture(nearLayer, projectedUv);
  vec4 far = texture(farLayer, projectedUv);
  vec3 color = sharp.rgb;
  if (far.a > .001) {
    float defocus = smoothstep(.5, 1.5, abs(radius));
    color = mix(sharp.rgb, far.rgb / far.a, far.a * defocus);
  }
  // The near layer is premultiplied and sits in front of the far layer.
  color = color * (1. - near.a) + near.rgb;
  if (focusBounds.z > 0.) {
    vec2 worldUv = directionUv(mat3(cameraToWorld) * ray);
    vec2 maskUv = (vec2(worldUv.x, 1. - worldUv.y) * focusDimensions - focusBounds.xy)
        / focusBounds.zw;
    // The padding lets filtering blend to zero just outside the mask.
    vec2 size = vec2(textureSize(focusMask, 0));
    float amount = texture(focusMask, (maskUv * (size - 2.) + 1.) / size).r;
    color = mix(color, sharp.rgb, amount);
  }

  gl_FragColor = vec4(color * backgroundIntensity, sharp.a * opacity);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <premultiplied_alpha_fragment>
}
`;

function makeMaterial(
    fragmentShader: string, uniforms: ShaderMaterial['uniforms']) {
  return new ShaderMaterial({
    glslVersion: GLSL3,
    vertexShader: vertex,
    fragmentShader,
    uniforms,
    depthTest: false,
    depthWrite: false,
    blending: NoBlending,
    toneMapped: false,
  });
}

export function rings(
    outerCount: number, middleCount: number, innerCount: number): Vector2[] {
  const offsets: Vector2[] = [];
  for (const [count, radius, rotation] of [
           [outerCount, .88, 0],
           [middleCount, .59, Math.PI / 4],
           [innerCount, .29, Math.PI / 7]]) {
    for (let i = 0; i < count; ++i) {
      const angle = rotation + i * Math.PI * 2 / count;
      offsets.push(
          new Vector2(Math.cos(angle), Math.sin(angle)).multiplyScalar(radius));
    }
  }
  // Preserve the uniform array layout for both quality levels.
  while (offsets.length < 24) {
    offsets.push(new Vector2());
  }
  return offsets;
}

interface SkyboxDofTargets {
  projection: WebGLRenderTarget;
  blur: WebGLRenderTarget;
}

export function supportsSkyboxDof(renderer: WebGLRenderer, skybox: Texture) {
  return skybox.mapping === EquirectangularReflectionMapping &&
      (ColorManagement.getTransfer(skybox.colorSpace) === SRGBTransfer ||
       renderer.extensions.has('EXT_color_buffer_float'));
}

/**
 * Scratch render targets are reused by the native renderer's sequential views.
 */
export class SkyboxDof {
  private camera = new Camera();
  private passes = new Scene();
  private geometry = new BufferGeometry().setAttribute(
      'position',
      new BufferAttribute(
          new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
  private skyboxRotation = new Matrix4();
  private skyboxEuler = new Euler();
  private fullKernel = rings(9, 5, 2);
  private wideKernel = rings(13, 8, 3);
  private reducedKernel = rings(4, 3, 1);
  private usedArea = {value: new Vector2()};
  private common = {
    colorTexture: {value: null as Texture | null},
    depthTexture: {value: null as Texture|null},
    cameraToSkybox: {value: new Matrix4()},
    projectionInverse: {value: new Matrix4()},
    crop: {value: new Vector2()},
    flipDepthY: {value: 0},
    depthIs16Bit: {value: 0},
    invertDepth: {value: 0},
    focus: {value: 1},
    strength: {value: 1},
    maxBlur: {value: 12},
  };
  private projection = makeMaterial(project, {
    ...this.common,
    texelSize: {value: new Vector2()},
  });
  private blur = makeMaterial(gather, {
    projectedColor: {value: null},
    projectedRadius: {value: null},
    radiusScale: {value: new Vector2()},
    usedArea: this.usedArea,
    samples: {value: this.fullKernel},
    tapCount: {value: 16},
    maxBlur: this.common.maxBlur,
    maximumNear: {value: 0},
    maximumFar: {value: 0},
  });
  private composite = makeMaterial(composite, {
    ...this.common,
    nearLayer: {value: null},
    farLayer: {value: null},
    usedArea: this.usedArea,
    opacity: {value: 1},
    backgroundIntensity: {value: 1},
    cameraToWorld: {value: new Matrix4()},
    focusMask: {value: null as Texture|null},
    focusDimensions: {value: new Vector2()},
    focusBounds: {value: new Vector4()},
  });
  private passMesh = new Mesh(this.geometry, this.projection);
  private backgroundMesh = new Mesh(this.geometry, this.composite);
  private targets = new Map<boolean, SkyboxDofTargets>();
  private lastFrameTime = -Infinity;
  private slowFrames = 0;
  private fastFrames = 0;
  private reducedQuality = false;

  constructor() {
    this.passMesh.frustumCulled = false;
    this.passes.add(this.passMesh);

    this.backgroundMesh.frustumCulled = false;
    this.backgroundMesh.renderOrder = -Infinity;
    this.backgroundMesh.userData.noHit = true;
    this.composite.premultipliedAlpha = true;
  }

  private getTargets(renderer: WebGLRenderer, hdr: boolean) {
    const {width: canvasWidth, height: canvasHeight} = renderer.domElement;
    const scale =
        Math.min(.5, Math.sqrt(MAX_WORK_PIXELS / (canvasWidth * canvasHeight)));
    const maxSize = renderer.capabilities.maxTextureSize;
    const width =
        Math.max(1, Math.min(maxSize, Math.floor(canvasWidth * scale)));
    const height =
        Math.max(1, Math.min(maxSize, Math.floor(canvasHeight * scale)));

    let targets = this.targets.get(hdr);
    if (targets == null) {
      const options = {
        count: 2,
        depthBuffer: false,
        type: hdr ? HalfFloatType : UnsignedByteType,
        colorSpace: hdr ? NoColorSpace : SRGBColorSpace,
        minFilter: LinearFilter,
        magFilter: LinearFilter,
      };
      const projection = new WebGLRenderTarget(width, height, options);
      const packedRadius = projection.textures[1];
      packedRadius.type = UnsignedByteType;
      packedRadius.colorSpace = NoColorSpace;
      packedRadius.format = RGFormat;
      packedRadius.minFilter = packedRadius.magFilter = NearestFilter;
      targets = {
        projection,
        blur: new WebGLRenderTarget(width, height, options)
      };
      this.targets.set(hdr, targets);
    }
    targets.projection.setSize(width, height);
    targets.blur.setSize(width, height);
    return targets;
  }

  reportFrame(time: number, delta: number) {
    if (time === this.lastFrameTime) {
      return;
    }

    const consecutive = Math.abs(time - this.lastFrameTime - delta) < .01;
    this.lastFrameTime = time;
    if (!consecutive || delta <= 0 || delta > 200) {
      this.slowFrames = this.fastFrames = 0;
      return;
    }

    // The shared renderer's existing throttling threshold is much lower than
    // the 60 Hz interaction target for this optional effect.
    this.slowFrames = delta > 19 ? this.slowFrames + 1 : 0;
    this.fastFrames = delta < 17.5 ? this.fastFrames + 1 : 0;
    if (this.slowFrames >= 3) {
      this.reducedQuality = true;
    } else if (this.fastFrames >= 120) {
      this.reducedQuality = false;
    }
  }

  render(
      renderer: WebGLRenderer, scene: ModelScene, frame: SkyboxDofFrame,
      width: number, height: number) {
    const hdr =
        ColorManagement.getTransfer(frame.skybox.colorSpace) !== SRGBTransfer;
    const divisor = this.reducedQuality || scene.scaleStep > 0 ? 3 : 2;
    const cssWidth = Math.max(1, scene.width);
    const cssHeight = Math.max(1, scene.height);
    const margin = 2 * frame.maxBlur;
    const paddedWidth = cssWidth + margin;
    const paddedHeight = cssHeight + margin;
    const targets = this.getTargets(renderer, hdr);
    const {width: targetWidth, height: targetHeight} = targets.projection;
    const scale = Math.min(
        width / cssWidth / divisor,
        height / cssHeight / divisor,
        targetWidth / paddedWidth,
        targetHeight / paddedHeight);
    const workWidth = Math.max(1, Math.floor(paddedWidth * scale));
    const workHeight = Math.max(1, Math.floor(paddedHeight * scale));
    targets.projection.viewport.set(0, 0, workWidth, workHeight);
    targets.blur.viewport.set(0, 0, workWidth, workHeight);
    this.usedArea.value.set(workWidth / targetWidth, workHeight / targetHeight);

    const camera = scene.camera;
    camera.updateWorldMatrix(true, false);
    this.skyboxRotation.makeRotationFromEuler(
        this.skyboxEuler.copy(scene.backgroundRotation)).transpose();
    this.common.cameraToSkybox.value.multiplyMatrices(
        this.skyboxRotation, camera.matrixWorld);
    this.common.projectionInverse.value.copy(camera.projectionMatrixInverse);
    this.common.colorTexture.value = frame.skybox;
    this.common.depthTexture.value = frame.depth;
    this.common.crop.value.set(
        cssWidth / paddedWidth, cssHeight / paddedHeight);
    this.common.flipDepthY.value = skyboxSourceFlipY(frame.skybox) ? 1 : 0;
    this.common.depthIs16Bit.value = frame.bitDepth === 16 ? 1 : 0;
    this.common.invertDepth.value = frame.invert ? 1 : 0;
    this.common.focus.value = frame.focus;
    this.common.strength.value = frame.strength;
    this.common.maxBlur.value = frame.maxBlur;

    this.projection.uniforms.texelSize.value.set(1 / workWidth, 1 / workHeight);
    this.blur.uniforms.projectedColor.value = targets.projection.textures[0];
    this.blur.uniforms.projectedRadius.value = targets.projection.textures[1];
    this.blur.uniforms.radiusScale.value.set(1 / paddedWidth, 1 / paddedHeight);
    const wide = frame.maxBlur > 16;
    this.blur.uniforms.samples.value = divisor === 3 ? this.reducedKernel :
        wide                                         ? this.wideKernel :
                                                       this.fullKernel;
    this.blur.uniforms.tapCount.value = divisor === 3 ? 8 : wide ? 24 : 16;
    this.blur.uniforms.maximumNear.value =
        Math.min(1, frame.focus * frame.strength) * frame.maxBlur;
    this.blur.uniforms.maximumFar.value =
        Math.min(1, (1 - frame.focus) * frame.strength) * frame.maxBlur;

    this.composite.uniforms.nearLayer.value = targets.blur.textures[0];
    this.composite.uniforms.farLayer.value = targets.blur.textures[1];
    this.composite.uniforms.opacity.value = frame.opacity;
    this.composite.uniforms.backgroundIntensity.value =
        scene.backgroundIntensity;
    const {focusMask} = frame;
    const uniforms = this.composite.uniforms;
    if (focusMask != null) {
      uniforms.cameraToWorld.value.copy(camera.matrixWorld);
      uniforms.focusMask.value = focusMask.texture;
      uniforms.focusDimensions.value.copy(focusMask.dimensions);
      uniforms.focusBounds.value.copy(focusMask.bounds);
    } else {
      uniforms.focusBounds.value.set(0, 0, 0, 0);
    }
    if (this.composite.toneMapped !== hdr) {
      this.composite.toneMapped = hdr;
      this.composite.needsUpdate = true;
    }

    const background = scene.background;
    const hidden = frame.hideDuringRender;
    const wasVisible = hidden?.visible;
    renderer.autoClear = false;
    try {
      this.passMesh.material = this.projection;
      renderer.setRenderTarget(targets.projection);
      renderer.render(this.passes, this.camera);

      this.passMesh.material = this.blur;
      renderer.setRenderTarget(targets.blur);
      renderer.render(this.passes, this.camera);

      renderer.setRenderTarget(null);
      renderer.autoClear = true;
      scene.background = null;
      if (hidden != null) {
        hidden.visible = false;
      }
      scene.add(this.backgroundMesh);
      renderer.render(scene, camera);
    } finally {
      scene.background = background;
      this.backgroundMesh.removeFromParent();
      if (hidden != null) {
        hidden.visible = wasVisible!;
      }
      renderer.setRenderTarget(null);
      renderer.autoClear = true;
      this.common.colorTexture.value = null;
      this.common.depthTexture.value = null;
      uniforms.focusMask.value = null;
    }
  }

  dispose() {
    for (const {projection, blur} of this.targets.values()) {
      projection.dispose();
      blur.dispose();
    }
    this.targets.clear();
    this.geometry.dispose();
    this.projection.dispose();
    this.blur.dispose();
    this.composite.dispose();
  }
}
