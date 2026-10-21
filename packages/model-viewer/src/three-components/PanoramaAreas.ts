/* @license Copyright 2026 Solunea. Licensed under the Apache License,
 * Version 2.0. */
import {BackSide, DataTexture, Group, Mesh, NearestFilter, Raycaster, RedFormat, ShaderMaterial, Sphere, SphereGeometry, Vector2, Vector3, Vector4} from 'three';

import {ModelScene} from './ModelScene.js';

export interface PanoramaAreaDefinition {
  readonly name: string;
  readonly mask: {
    readonly width: number;
    readonly height: number;
    readonly bounds: {readonly x: number; readonly y: number; readonly width: number; readonly height: number;};
    /** Row-major, least significant bit first, relative to bounds. */
    readonly bits: Uint8Array;
  };
}

export interface PanoramaAreaHit {
  name: string;
  u: number;
  v: number;
}

const radius = 10;
const sphere = new Sphere(new Vector3(), radius);
const raycaster = new Raycaster();
const intersection = new Vector3();
const ndc = new Vector2();

function validate(area: PanoramaAreaDefinition) {
  const {width, height, bounds, bits} = area.mask;
  if (!area.name || !Number.isInteger(width) || !Number.isInteger(height) ||
      width <= 0 || height <= 0 || width * height > 128 * 1024 * 1024 ||
      !['x', 'y', 'width', 'height'].every((key) => {
        const value = bounds[key as keyof typeof bounds];
        return Number.isInteger(value) && value >= 0;
      }) ||
      bounds.x + bounds.width > width || bounds.y + bounds.height > height ||
      !(bits instanceof Uint8Array) ||
      bits.length !== Math.ceil(bounds.width * bounds.height / 8)) {
    throw new Error('Invalid panorama area: ' + area.name);
  }
}

function equal(a: PanoramaAreaDefinition, b: PanoramaAreaDefinition): boolean {
  const x = a.mask;
  const y = b.mask;
  return x.width === y.width && x.height === y.height &&
      x.bounds.x === y.bounds.x && x.bounds.y === y.bounds.y &&
      x.bounds.width === y.bounds.width &&
      x.bounds.height === y.bounds.height && x.bits.length === y.bits.length &&
      x.bits.every((value, i) => value === y.bits[i]);
}

/** Internal scene-root overlay. Never added to the exportable model target. */
export class PanoramaAreas extends Group {
  private areas = new Map<string, PanoramaAreaDefinition>();
  private highlighted: string|null = null;
  private mesh: Mesh<SphereGeometry, ShaderMaterial>|null = null;
  private texture: DataTexture|null = null;
  private uploaded: PanoramaAreaDefinition|null = null;
  private active = false;
  private capturing = false;

  constructor(private scene: ModelScene, private textureLimit = 4096) {
    super();
    this.name = 'PanoramaAreas';
    this.userData.noHit = true;
    this.visible = false;
  }

  setAreas(areas: readonly PanoramaAreaDefinition[]) {
    const next = new Map<string, PanoramaAreaDefinition>();
    // Validate the complete replacement before changing the active collection.
    for (const area of areas) {
      validate(area);
      if (next.has(area.name)) {
        throw new Error('Duplicate panorama area: ' + area.name);
      }
      const previous = this.areas.get(area.name);
      next.set(area.name, previous && equal(previous, area) ? previous : {
        name: area.name,
        mask: {
          ...area.mask,
          bounds: {...area.mask.bounds},
          bits: area.mask.bits.slice()
        }
      });
    }
    this.areas = next;
    this.update();
  }

  setHighlight(name: string|null) {
    if (name === this.highlighted) {
      return;
    }
    this.highlighted = name;
    this.update();
  }

  setActive(active: boolean) {
    if (active === this.active) {
      return;
    }
    this.active = active;
    this.update();
  }

  setCapturing(capturing: boolean) {
    this.capturing = capturing;
    this.update();
  }

  fromPoint(clientX: number, clientY: number): PanoramaAreaHit|null {
    if (!this.active || !this.scene.element.isConnected || !this.areas.size ||
        !Number.isFinite(clientX) || !Number.isFinite(clientY)) {
      return null;
    }
    const rect = this.scene.element.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0 || clientX < rect.left ||
        clientX >= rect.right || clientY < rect.top || clientY >= rect.bottom) {
      return null;
    }
    ndc.set(
        (clientX - rect.left) / rect.width * 2 - 1,
        1 - (clientY - rect.top) / rect.height * 2);
    const camera = this.scene.getCamera();
    camera.updateMatrixWorld();
    raycaster.setFromCamera(ndc, camera);
    if (!raycaster.ray.intersectSphere(sphere, intersection)) {
      return null;
    }
    const u =
        ((0.25 + Math.atan2(intersection.x, -intersection.z) / (2 * Math.PI)) %
             1 +
         1) %
        1;
    const v = Math.max(
        0,
        Math.min(
            1,
            0.5 -
                Math.asin(Math.max(-1, Math.min(1, intersection.y / radius))) /
                    Math.PI));
    for (const {name, mask} of this.areas.values()) {
      const {width, height, bounds, bits} = mask;
      const x = Math.floor(u * width) - bounds.x;
      const y = Math.min(height - 1, Math.floor(v * height)) - bounds.y;
      if (x < 0 || y < 0 || x >= bounds.width || y >= bounds.height) {
        continue;
      }
      const p = y * bounds.width + x;
      if (bits[p >> 3] & (1 << (p & 7))) {
        return {name, u, v};
      }
    }
    return null;
  }

  private update() {
    const area =
        this.highlighted == null ? null : this.areas.get(this.highlighted);
    let changed = false;
    if (!this.active || !this.areas.size ||
        (area && (!area.mask.bounds.width || !area.mask.bounds.height)) ||
        (!area && this.uploaded &&
         this.areas.get(this.uploaded.name) !== this.uploaded)) {
      this.releaseGraphics();
    } else if (area && area.mask.bounds.width && area.mask.bounds.height) {
      if (this.mesh == null) {
        const material = new ShaderMaterial({
          transparent: true,
          depthWrite: false,
          depthTest: false,
          side: BackSide,
          toneMapped: false,
          uniforms: {
            mask: {value: null},
            panorama: {value: new Vector2()},
            bounds: {value: new Vector4()}
          },
          vertexShader: `varying vec3 direction;
            void main(){direction=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
          fragmentShader: `precision highp float;
            uniform sampler2D mask;uniform vec2 panorama;uniform vec4 bounds;varying vec3 direction;
            void main(){vec3 d=normalize(direction);
              vec2 uv=vec2(fract(.25+atan(d.x,-d.z)/6.28318530718),.5-asin(clamp(d.y,-1.,1.))/3.14159265359);
              vec2 p=min(floor(uv*panorama),panorama-1.)-bounds.xy;
              if(p.x<0.||p.y<0.||p.x>=bounds.z||p.y>=bounds.w)discard;
              float alpha=texture2D(mask,(p+.5)/bounds.zw).r;
              if(alpha<.5)discard;gl_FragColor=vec4(.5,.38,.9,.3);}`
        });
        this.mesh = new Mesh(new SphereGeometry(radius, 64, 32), material);
        this.mesh.name = 'PanoramaAreaHighlight';
        this.mesh.userData.noHit = true;
        this.mesh.raycast = () => {};
        this.mesh.renderOrder = 1000;
        this.mesh.frustumCulled = false;
        this.add(this.mesh);
      }
      if (this.uploaded !== area) {
        const {width, height, bounds, bits} = area.mask;
        // Only visual sampling is reduced; detection always uses the original
        // bits.
        const limit = Math.min(4096, this.textureLimit);
        const scale = Math.max(
            1, Math.ceil(Math.max(bounds.width, bounds.height) / limit));
        const tw = Math.ceil(bounds.width / scale);
        const th = Math.ceil(bounds.height / scale);
        const pixels = new Uint8Array(tw * th);
        for (let p = 0; p < bounds.width * bounds.height; p++) {
          if (bits[p >> 3] & (1 << (p & 7))) {
            pixels[Math.floor(Math.floor(p / bounds.width) / scale) * tw + Math.floor(p % bounds.width / scale)] = 255;
          }
        }
        this.texture?.dispose();
        this.texture = new DataTexture(pixels, tw, th, RedFormat);
        this.texture.minFilter = this.texture.magFilter = NearestFilter;
        this.texture.needsUpdate = true;
        const {uniforms} = this.mesh.material;
        uniforms.mask.value = this.texture;
        uniforms.panorama.value.set(width, height);
        uniforms.bounds.value.set(
            bounds.x, bounds.y, bounds.width, bounds.height);
        this.uploaded = area;
        changed = true;
      }
    }
    const visible = !!(this.active && area && this.mesh && !this.capturing);
    if (this.visible !== visible || changed) {
      this.scene.queueRender();
    }
    this.visible = visible;
  }

  private releaseGraphics() {
    this.texture?.dispose();
    this.texture = null;
    this.uploaded = null;
    if (this.mesh) {
      this.mesh.geometry.dispose();
      this.mesh.material.dispose();
      this.remove(this.mesh);
      this.mesh = null;
      this.scene.queueRender();
    }
  }

  dispose() {
    this.releaseGraphics();
    this.active = false;
    this.visible = false;
    this.removeFromParent();
  }
}
