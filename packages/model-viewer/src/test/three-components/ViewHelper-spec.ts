import {expect} from 'chai';
import {Camera, Object3D, PerspectiveCamera, Vector4, WebGLRenderer} from 'three';

import {ModelViewerViewHelper, ViewHelperViewport} from '../../three-components/ViewHelper.js';

suite('ViewHelper', () => {
  let camera: PerspectiveCamera;
  let helper: ModelViewerViewHelper;
  let renderer: WebGLRenderer;
  let rendererViewport: Vector4;
  let renderedViewports: Vector4[];

  const elementRect = new DOMRect(20, 30, 400, 300);
  const createViewport = (renderedDpr: number): ViewHelperViewport => ({
    x: 80 * renderedDpr,
    y: 40 * renderedDpr,
    width: elementRect.width * renderedDpr,
    height: elementRect.height * renderedDpr,
    canvasWidth: 800 * renderedDpr,
    canvasHeight: 600 * renderedDpr,
    renderedDpr
  });
  const pointerAt = (x: number, y: number) => new PointerEvent(
      'pointerdown',
      {clientX: elementRect.left + x, clientY: elementRect.top + y});

  setup(() => {
    camera = new PerspectiveCamera();
    camera.position.set(0, 0, 5);
    camera.updateMatrixWorld();
    helper = new ModelViewerViewHelper(camera);
    rendererViewport = new Vector4(10, 20, 800, 600);
    renderedViewports = [];
    renderer = {
      autoClear: true,
      clearDepth() {},
      getViewport(target: Vector4) {
        return target.copy(rendererViewport);
      },
      setViewport(x: number, y: number, width: number, height: number) {
        rendererViewport.set(x, y, width, height);
      },
      render(scene: Object3D, viewCamera: Camera) {
        renderedViewports.push(rendererViewport.clone());
        scene.updateMatrixWorld(true);
        viewCamera.updateMatrixWorld(true);
      }
    } as unknown as WebGLRenderer;
  });

  teardown(() => helper.dispose());

  for (const renderedDpr of [1, 0.79, 0.5, 0.25, 2, 1.58]) {
    suite(`at rendered DPR ${renderedDpr}`, () => {
      setup(() => helper.render(renderer, createViewport(renderedDpr)));

      test('keeps the helper at 128 CSS pixels in its scene corner', () => {
        const viewport = renderedViewports[0];
        expect(viewport.x / renderedDpr).to.be.closeTo(80 + 400 - 128, 0.001);
        expect(viewport.y / renderedDpr).to.be.closeTo(40 + 300 - 128, 0.001);
        expect(viewport.z / renderedDpr).to.be.closeTo(128, 0.001);
        expect(viewport.w / renderedDpr).to.be.closeTo(128, 0.001);
      });

      test('keeps the clickable area at the displayed CSS size', () => {
        expect(helper.containsPoint(pointerAt(400 - 127, 1), elementRect))
            .to.be.true;
        expect(helper.containsPoint(pointerAt(400 - 129, 1), elementRect))
            .to.be.false;
        expect(helper.containsPoint(pointerAt(400 - 64, 129), elementRect))
            .to.be.false;
      });

      test(
          'selects the same orientation without moving the camera directly',
          () => {
            const position = camera.position.clone();
            const quaternion = camera.quaternion.clone();
            const orbit =
                helper.handleClick(pointerAt(400 - 32, 64), elementRect);
            expect(orbit).to.not.equal(null);
            expect(orbit!.theta).to.be.closeTo(Math.PI / 2, 0.001);
            expect(orbit!.phi).to.be.closeTo(Math.PI / 2, 0.001);
            expect(orbit!.radius).to.be.closeTo(5, 0.001);
            expect(camera.position.equals(position)).to.be.true;
            expect(camera.quaternion.equals(quaternion)).to.be.true;
          });

      test(
          'restores the shared renderer viewport and automatic clearing',
          () => {
            expect(rendererViewport.toArray())
                .to.deep.equal([10, 20, 800, 600]);
            expect(renderer.autoClear).to.be.true;
          });
    });
  }

  test('stays visible in a scene with a small rendering buffer', () => {
    helper.render(renderer, createViewport(0.25));
    expect(renderedViewports).to.have.lengthOf(1);
  });

  test('restores renderer state if rendering the helper fails', () => {
    renderer.autoClear = false;
    renderer.render = () => {
      throw new Error('render failure');
    };
    expect(() => helper.render(renderer, createViewport(0.79)))
        .to.throw('render failure');
    expect(rendererViewport.toArray()).to.deep.equal([10, 20, 800, 600]);
    expect(renderer.autoClear).to.be.false;
  });

  test('hides when the scene is smaller than the helper in CSS pixels', () => {
    const viewport = createViewport(2);
    viewport.width = 127 * viewport.renderedDpr;
    helper.render(renderer, viewport);
    expect(renderedViewports).to.be.empty;
    expect(helper.containsPoint(pointerAt(100, 64), elementRect)).to.be.false;
    expect(helper.handleClick(pointerAt(100, 64), elementRect)).to.equal(null);
  });
});
