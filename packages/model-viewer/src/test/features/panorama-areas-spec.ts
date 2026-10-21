import {expect} from 'chai';
import {Color, Mesh, Raycaster, ShaderMaterial, SphereGeometry, Vector2, Vector3} from 'three';

import {$scene} from '../../model-viewer-base.js';
import {ModelViewerElement} from '../../model-viewer.js';
import {PanoramaAreaDefinition, PanoramaAreas} from '../../three-components/PanoramaAreas.js';
import {Renderer} from '../../three-components/Renderer.js';
import {waitForEvent} from '../../utilities.js';
import {assetPath, rafPasses} from '../helpers.js';

const definition =
    (name = 'area', predicate = (_x: number, _y: number) => true):
        PanoramaAreaDefinition => {
          const width = 64;
          const height = 32;
          const bits = new Uint8Array(width * height / 8);
          for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
              const p = y * width + x;
              if (predicate(x, y)) {
                bits[p >> 3] |= 1 << (p & 7);
              }
            }
          }
          return {
            name,
            mask: {width, height, bounds: {x: 0, y: 0, width, height}, bits}
          };
        };

suite('PanoramaAreas geometry and lifecycle', () => {
  let element: ModelViewerElement;
  let areas: PanoramaAreas;
  setup(() => {
    element = new ModelViewerElement();
    element.style.cssText = 'display:block;width:640px;height:480px';
    document.body.appendChild(element);
    const scene = element[$scene];
    scene.camera.position.set(0, 0, 0.001);
    scene.camera.lookAt(0, 0, 0);
    scene.camera.aspect = 4 / 3;
    scene.camera.fov = 100;
    scene.camera.updateProjectionMatrix();
    scene.camera.updateMatrixWorld();
    areas = new PanoramaAreas(scene);
    scene.add(areas);
    areas.setActive(true);
  });
  teardown(() => {
    areas.dispose();
    element.remove();
  });
  const center = () => {
    const rect = element.getBoundingClientRect();
    return [rect.left + rect.width / 2, rect.top + rect.height / 2] as const;
  };
  const point = (u: number, v: number) => {
    const yaw = (u - 0.25) * 2 * Math.PI;
    const pitch = (v - 0.5) * Math.PI;
    const p = new Vector3(
                  Math.sin(yaw) * Math.cos(pitch),
                  -Math.sin(pitch),
                  -Math.cos(yaw) * Math.cos(pitch))
                  .multiplyScalar(10);
    p.project(element[$scene].camera);
    const rect = element.getBoundingClientRect();
    return [
      rect.left + (p.x + 1) * rect.width / 2,
      rect.top + (1 - p.y) * rect.height / 2
    ] as const;
  };
  test(
      'does not allocate graphics without a highlight and replaces/removes entries',
      () => {
        expect(areas.children.length).to.equal(0);
        areas.setAreas([definition()]);
        expect(areas.children.length).to.equal(0);
        const hit = areas.fromPoint(...center())!;
        expect(hit.name).to.equal('area');
        expect(hit.u).to.be.closeTo(0.25, 0.00001);
        expect(hit.v).to.be.closeTo(0.5, 0.00001);
        areas.setHighlight('area');
        expect(areas.children.length).to.equal(1);
        const mesh = areas.children[0] as Mesh<SphereGeometry, ShaderMaterial>;
        const texture = mesh.material.uniforms.mask.value;
        let disposed = 0;
        texture.addEventListener('dispose', () => disposed++);
        areas.setAreas([definition('area', () => false)]);
        expect(areas.fromPoint(...center())).to.equal(null);
        expect(disposed).to.equal(1);
        areas.setAreas([]);
        expect(areas.children.length).to.equal(0);
      });
  test(
      'preserves holes, disconnected islands and first entry overlap priority',
      () => {
        const shape = definition(
            'shape',
            (x, y) => (x >= 12 && x < 20 && y >= 12 && y < 20) &&
                    !(x >= 15 && x < 17 && y >= 15 && y < 17) ||
                x === 22 && y === 13);
        areas.setAreas([shape, definition('behind')]);
        expect(areas.fromPoint(...point(13.5 / 64, 13.5 / 32))!.name)
            .to.equal('shape');
        expect(areas.fromPoint(...point(16.5 / 64, 16.5 / 32))!.name)
            .to.equal('behind');
        expect(areas.fromPoint(...point(22.5 / 64, 13.5 / 32))!.name)
            .to.equal('shape');
        areas.setAreas([shape]);
        expect(areas.fromPoint(...point(16.5 / 64, 16.5 / 32))).to.equal(null);
      });
  test('changes highlight color without recreating the mask texture', () => {
    areas.setAreas([definition()]);
    areas.setHighlight('area', '#ff0000');
    const mesh = areas.children[0] as Mesh<SphereGeometry, ShaderMaterial>;
    const texture = mesh.material.uniforms.mask.value;
    expect(mesh.material.uniforms.highlightColor.value.equals(new Color('#ff0000'))).to.equal(true);
    areas.setHighlight('area', 'rgb(0, 255, 0)');
    expect(mesh.material.uniforms.highlightColor.value.equals(new Color('#00ff00'))).to.equal(true);
    expect(mesh.material.uniforms.mask.value).to.equal(texture);
  });
  test('handles seam, poles and CSS window coordinates after resize', () => {
    areas.setAreas([definition(
        'edges', (x, y) => x === 0 || x === 63 || y === 0 || y === 31)]);
    const camera = element[$scene].camera;
    camera.position.set(0.001, 0, 0);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    expect(areas.fromPoint(...point(0.5 / 64, 0.5))!.name).to.equal('edges');
    expect(areas.fromPoint(...point(63.5 / 64, 0.5))!.name).to.equal('edges');
    camera.position.set(0, -0.001, 0);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    expect(areas.fromPoint(...center())!.v).to.be.closeTo(0, 0.001);
    camera.position.set(0, 0.001, 0);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    expect(areas.fromPoint(...center())!.v).to.be.closeTo(1, 0.001);
    element.style.width = '320px';
    element.style.marginLeft = '75px';
    expect(areas.fromPoint(...center())!.name).to.equal('edges');
    expect(areas.fromPoint(0, -1)).to.equal(null);
  });
  test('uses cropped little-endian bits and snapshots caller data', () => {
    const area = {
      name: 'crop',
      mask: {
        width: 64,
        height: 32,
        bounds: {x: 16, y: 16, width: 2, height: 2},
        bits: new Uint8Array([1])
      }
    };
    areas.setAreas([area]);
    area.mask.bits[0] = 0;
    expect(areas.fromPoint(...point(16.5 / 64, 16.5 / 32))!.name)
        .to.equal('crop');
    expect(areas.fromPoint(...point(17.5 / 64, 16.5 / 32))).to.equal(null);
  });
  test('rejects invalid replacements atomically', () => {
    areas.setAreas([definition()]);
    expect(() => areas.setAreas([
      definition('duplicate'),
      definition('duplicate')
    ])).to.throw();
    expect(areas.fromPoint(...center())!.name).to.equal('area');
  });
  test('is excluded from ordinary raycasts and model target', () => {
    areas.setAreas([definition()]);
    areas.setHighlight('area');
    const ray = new Raycaster();
    ray.setFromCamera(new Vector2(), element[$scene].camera);
    expect(ray.intersectObject(areas, true)).to.have.length(0);
    expect(element[$scene].target.children).not.to.include(areas);
    expect(areas.children[0].castShadow).to.equal(false);
  });
  test(
      'disposes graphics on leaving panorama and recreates on reconnection',
      async () => {
        areas.dispose();
        element.skyboxOnly = true;
        await element.updateComplete;
        element.setPanoramaAreas([definition()]);
        element.setPanoramaAreaHighlight('area');
        const native = element[$scene].children.find(
            (child) => child.name === 'PanoramaAreas')!;
        expect(native.children.length).to.equal(1);
        element.remove();
        expect(native.children.length).to.equal(0);
        document.body.appendChild(element);
        await element.updateComplete;
        expect(native.children.length).to.equal(1);
        element.skyboxOnly = false;
        await element.updateComplete;
        expect(native.children.length).to.equal(0);
        expect(element.panoramaAreaFromPoint(...center())).to.equal(null);
      });
});

suite('PanoramaAreas native captures', () => {
  let element: ModelViewerElement;
  setup(async function() {
    this.timeout(30000);
    if (!Renderer.singleton.canRender) {
      this.skip();
      return;
    }
    element = new ModelViewerElement();
    element.style.cssText = 'display:block;width:256px;height:192px';
    element.skyboxOnly = true;
    document.body.appendChild(element);
    element.src = assetPath('models/cube.gltf');
    await waitForEvent(element, 'poster-dismissed');
    await rafPasses();
  });
  teardown(() => element?.remove());
  test(
      'excludes highlight from posters and serializes concurrent captures',
      async () => {
        const clean = await element.toBlob();
        element.setPanoramaAreas([definition()]);
        element.setPanoramaAreaHighlight('area');
        await rafPasses();
        const native = element[$scene].children.find(
            (child) => child.name === 'PanoramaAreas')!;
        expect(native.visible).to.equal(true);
        const captures =
            await Promise.all([element.toBlob(), element.toBlob()]);
        // Native adaptive resolution can change PNG dimensions while encoding.
        // Compare decoded source pixels, rather than the PNG representation.
        const centerPixel = async (blob: Blob) => {
          const url = URL.createObjectURL(blob);
          try {
            const image = new Image();
            image.src = url;
            await image.decode();
            const canvas = document.createElement('canvas');
            canvas.width = canvas.height = 1;
            const context = canvas.getContext('2d')!;
            context.drawImage(
                image,
                Math.floor(image.width / 2),
                Math.floor(image.height / 2),
                1,
                1,
                0,
                0,
                1,
                1);
            return Array.from(context.getImageData(0, 0, 1, 1).data);
          } finally {
            URL.revokeObjectURL(url);
          }
        };
        const pixel = await centerPixel(clean);
        for (const capture of captures) {
          expect(await centerPixel(capture)).to.eql(pixel);
        }
        expect(native.visible).to.equal(true);
        const exported = await element.exportScene({binary: false});
        expect(await exported.text()).not.to.contain('PanoramaArea');
      });
  test(
      'restores highlight after encoding failure and allows the next capture',
      async () => {
        element.setPanoramaAreas([definition()]);
        element.setPanoramaAreaHighlight('area');
        const native = element[$scene].children.find(
            (child) => child.name === 'PanoramaAreas')!;
        const original = HTMLCanvasElement.prototype.toBlob;
        HTMLCanvasElement.prototype.toBlob = function(callback) {
          callback(null);
        };
        try {
          let failed = false;
          try {
            await element.toBlob();
          } catch (_) {
            failed = true;
          }
          expect(failed).to.equal(true);
          expect(native.visible).to.equal(true);
        } finally {
          HTMLCanvasElement.prototype.toBlob = original;
        }
        expect((await element.toBlob()).size).to.be.greaterThan(0);
      });
  test('serializes captures and redraws cleanly while offscreen', async () => {
    element.setPanoramaAreas([definition()]);
    element.setPanoramaAreaHighlight('area');
    await rafPasses();
    element.style.position = 'fixed';
    element.style.left = '-10000px';
    await rafPasses();
    await rafPasses();
    expect(element.modelIsVisible).to.equal(false);
    const native =
        element[$scene].children.find((child) => child.name === 'PanoramaAreas')!;
    const original = HTMLCanvasElement.prototype.toBlob;
    let encoders = 0; let maximum = 0;
    HTMLCanvasElement.prototype.toBlob = function(callback, type, quality) {
      encoders++;
      maximum = Math.max(maximum, encoders);
      expect(native.visible).to.equal(false);
      setTimeout(() => original.call(this, (blob) => {
        encoders--;
        callback(blob);
      }, type, quality), 20);
    };
    try {
      const before = element[$scene].renderCount;
      const captures = await Promise.all([element.toBlob(), element.toBlob()]);
      expect(captures.every((blob) => blob.size > 0)).to.equal(true);
      expect(maximum).to.equal(1);
      expect(element[$scene].renderCount - before).to.be.at.least(4);
      expect(native.visible).to.equal(true);
    } finally {
      HTMLCanvasElement.prototype.toBlob = original;
    }
  });
});
