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

import type {Camera, WebGLRenderer} from 'three';
import {ViewHelper} from 'three/examples/jsm/helpers/ViewHelper.js';

const VIEW_HELPER_SIZE = 128;

export interface ViewHelperViewport {
  x: number;
  y: number;
  width: number;
  height: number;
  canvasWidth: number;
  canvasHeight: number;
}

interface ViewHelperDomElement {
  offsetWidth: number;
  offsetHeight: number;
  getBoundingClientRect(): DOMRect;
}

/**
 * Adapts Three.js ViewHelper to model-viewer's shared renderer viewport model.
 */
export class ModelViewerViewHelper {
  private readonly domElement: ViewHelperDomElement = {
    offsetWidth: 1,
    offsetHeight: 1,
    getBoundingClientRect() {
      return new DOMRect(0, 0, this.offsetWidth, this.offsetHeight);
    }
  };

  private readonly viewHelper: ViewHelper;

  constructor(camera: Camera) {
    this.viewHelper =
        new ViewHelper(camera, this.domElement as unknown as HTMLElement);
  }

  render(renderer: WebGLRenderer, viewport: ViewHelperViewport) {
    const {canvasWidth, canvasHeight, x, y, width, height} = viewport;
    if (width < VIEW_HELPER_SIZE || height < VIEW_HELPER_SIZE) {
      return;
    }

    this.domElement.offsetWidth = canvasWidth;
    this.domElement.offsetHeight = canvasHeight;

    this.viewHelper.location.left = null;
    this.viewHelper.location.top = null;
    this.viewHelper.location.right = Math.max(0, canvasWidth - x - width);
    this.viewHelper.location.bottom = Math.max(0, y);

    const autoClear = renderer.autoClear;
    renderer.autoClear = false;
    try {
      this.viewHelper.render(renderer);
    } finally {
      renderer.autoClear = autoClear;
    }
  }

  dispose() {
    this.viewHelper.dispose();
  }
}
