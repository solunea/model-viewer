/* @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import '../renderer-gate.js';

import {expect} from 'chai';
import {Matrix4, Mesh, SphereGeometry, Texture, Vector3} from 'three';

import {$scene} from '../../model-viewer-base.js';
import {ModelViewerElement} from '../../model-viewer.js';
import {ModelScene} from '../../three-components/ModelScene.js';
import {assetPath} from '../helpers.js';

suite('ModelScene', () => {
  let element: ModelViewerElement;
  let scene: ModelScene;
  let dummyRadius: number;
  let dummyMesh: Mesh;

  setup(() => {
    // Set the radius of the sphere to 0.5 so that it's size is 1
    // for testing scaling.
    dummyRadius = 0.5;
    dummyMesh = new Mesh(new SphereGeometry(dummyRadius, 32, 32));
    element = new ModelViewerElement();
    scene = element[$scene];

    document.body.insertBefore(element, document.body.firstChild);
  });

  teardown(() => {
    document.body.removeChild(element);
  });

  suite('with a model', () => {
    setup(async () => {
      await scene.setSource(assetPath('models/soldier.glb'));
    });

    suite('setShadowIntensity', () => {
      test('can increase intensity and reset it to zero', () => {
        scene.setShadowIntensity(1);
        const shadow = scene.shadow!;
        expect(shadow).to.be.ok;
        expect(shadow.getIntensity()).to.be.equal(1);
        scene.setShadowIntensity(0);
        expect(shadow.getIntensity()).to.be.equal(0);
      });

      test('shadow is only created when intensity is greater than zero', () => {
        expect(scene.shadow).to.be.not.ok;
        scene.setShadowIntensity(1);
        expect(scene.shadow).to.be.ok;
      });
    });

    test('can append and play an animation', () => {
      expect(scene.animationNames.length).to.be.greaterThan(0);
      const animationName = scene.animationNames[0];
      scene.appendAnimation(animationName);
      expect(scene.appendedAnimations).to.include(animationName);
    });

    test('can detach an appended animation', () => {
      expect(scene.animationNames.length).to.be.greaterThan(0);
      const animationName = scene.animationNames[0];
      scene.appendAnimation(animationName);
      expect(scene.appendedAnimations).to.include(animationName);
      scene.detachAnimation(animationName, false);
      expect(scene.appendedAnimations).to.not.include(animationName);
    });
  });

  suite('shadow orbit interpolation', () => {
    setup(async () => {
      await scene.setSource(assetPath('models/soldier.glb'));
      scene.setShadowIntensity(1);
    });

    test('interpolates shadow orbit when decay is set', () => {
      scene.setShadowInterpolationDecay(200);
      scene.setShadowOrbit(0, Math.PI / 2);

      const shadow = scene.shadow! as any;
      expect(shadow.phi).to.be.equal(0);

      shadow.update(16);
      expect(shadow.phi).to.be.greaterThan(0);
      expect(shadow.phi).to.be.lessThan(Math.PI / 2);
    });

    test('marks the PCSS shadow map dirty while interpolating orbit', () => {
      scene.setShadowInterpolationDecay(200);
      scene.setShadowOrbit(0, Math.PI / 2);

      const shadow = scene.shadow! as any;
      shadow.light.shadow.needsUpdate = false;
      shadow.update(16);

      expect(shadow.light.shadow.needsUpdate).to.be.true;
    });

    test('interpolates initial shadow orbit when decay is already set', () => {
      scene.shadow!.dispose();
      scene.shadow = null;

      element.shadowInterpolationDecay = 200;
      element.shadowOrbit = '0deg 90deg';
      scene.setShadowIntensity(1);

      const shadow = scene.shadow! as any;
      expect(shadow.phi).to.be.equal(0);

      shadow.update(16);
      expect(shadow.phi).to.be.greaterThan(0);
      expect(shadow.phi).to.be.lessThan(Math.PI / 2);
    });

    test('changes shadow orbit immediately by default', () => {
      scene.setShadowOrbit(0, Math.PI / 2);

      expect((scene.shadow! as any).phi).to.be.equal(Math.PI / 2);
    });
  });

  suite('skybox interpolation', () => {
    test('fades in the next skybox when decay is set', () => {
      const firstSkybox = new Texture();
      const secondSkybox = new Texture();

      scene.setEnvironmentAndSkybox(null, firstSkybox);
      scene.setSkyboxInterpolationDecay(200);
      scene.setEnvironmentAndSkybox(null, secondSkybox);

      const transition = (scene as any).skyboxTransition;
      expect(scene.background).to.be.equal(firstSkybox);
      expect(transition.parent).to.be.equal(scene);
      expect(transition.material.map).to.be.equal(secondSkybox);
      expect(transition.material.opacity).to.be.equal(0);
      expect(transition.scale.x)
          .to.be.equal((scene.getCamera() as any).far * 0.5);

      scene.updateSkyboxTransition(16);

      expect(transition.material.opacity).to.be.greaterThan(0);
      expect(transition.material.opacity).to.be.lessThan(1);
    });

    test('changes skybox immediately by default', () => {
      const firstSkybox = new Texture();
      const secondSkybox = new Texture();

      scene.setEnvironmentAndSkybox(null, firstSkybox);
      scene.setEnvironmentAndSkybox(null, secondSkybox);

      expect(scene.background).to.be.equal(secondSkybox);
      expect((scene as any).skyboxTransition).to.be.null;
    });

    test('does not fade in the initial skybox in skybox-only mode', () => {
      const firstSkybox = new Texture();

      element.skyboxOnly = true;
      scene.setSkyboxInterpolationDecay(200);
      scene.setEnvironmentAndSkybox(null, firstSkybox);

      expect(scene.background).to.be.equal(firstSkybox);
      expect((scene as any).skyboxTransition).to.be.null;
    });
  });

  suite('setSize', () => {
    test('updates visual and buffer size', () => {
      scene.setSize(500, 200);
      expect(scene.width).to.be.equal(500);
      expect(scene.height).to.be.equal(200);
    });

    test('model is not scaled', async () => {
      dummyMesh.geometry.applyMatrix4(new Matrix4().makeScale(1, 3, 10));
      await scene.setObject(dummyMesh);

      scene.setSize(1000, 500);
      expect(scene.scale).to.be.eql(new Vector3(1, 1, 1));
    });

    test('idealCameraDistance is set correctly', async () => {
      await scene.setObject(dummyMesh);

      scene.framedFoVDeg = 25;
      const halfFov = (scene.framedFoVDeg / 2) * Math.PI / 180;
      const expectedDistance = dummyRadius / Math.sin(halfFov);
      expect(scene.idealCameraDistance())
          .to.be.closeTo(expectedDistance, 0.0001);
    });

    test('idealAspect is set correctly', async () => {
      scene.framedFoVDeg = 25;
      await scene.setObject(dummyMesh);

      expect(scene.idealAspect).to.be.closeTo(1, 0.001);
    });

    test('cannot set the canvas smaller than 1x1', () => {
      scene.setSize(0, 0);
      expect(scene.width).to.be.equal(1);
      expect(scene.height).to.be.equal(1);
    });
  });
});
