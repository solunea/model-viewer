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
  renderedDpr: number;
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
  private viewport: ViewHelperViewport|null = null;

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
    this.viewport = viewport;
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

  containsPoint(event: PointerEvent, elementRect: DOMRect): boolean {
    const pointer = this.eventToRendererPoint(event, elementRect);
    if (pointer == null) {
      return false;
    }

    const {viewport} = this;
    const helperLeft = viewport!.x + viewport!.width - VIEW_HELPER_SIZE;
    const helperRight = viewport!.x + viewport!.width;
    const helperTop =
        viewport!.canvasHeight - viewport!.y - VIEW_HELPER_SIZE;
    const helperBottom = viewport!.canvasHeight - viewport!.y;

    return pointer.clientX >= helperLeft && pointer.clientX <= helperRight &&
        pointer.clientY >= helperTop && pointer.clientY <= helperBottom;
  }

  handleClick(event: PointerEvent, elementRect: DOMRect): boolean {
    const pointer = this.eventToRendererPoint(event, elementRect);
    return pointer == null ? false :
                             this.viewHelper.handleClick(pointer as MouseEvent);
  }

  update(deltaMilliseconds: number): boolean {
    if (!this.viewHelper.animating) {
      return false;
    }

    this.viewHelper.update(deltaMilliseconds / 1000);
    return true;
  }

  get animating(): boolean {
    return this.viewHelper.animating;
  }

  dispose() {
    this.viewHelper.dispose();
  }

  private eventToRendererPoint(event: PointerEvent, elementRect: DOMRect):
      Pick<MouseEvent, 'clientX'|'clientY'>|null {
    const {viewport} = this;
    if (viewport == null || viewport.width < VIEW_HELPER_SIZE ||
        viewport.height < VIEW_HELPER_SIZE) {
      return null;
    }

    const {x, y, height, canvasHeight, renderedDpr} = viewport;
    const sceneTop = canvasHeight - y - height;
    return {
      clientX: x + (event.clientX - elementRect.left) * renderedDpr,
      clientY: sceneTop + (event.clientY - elementRect.top) * renderedDpr
    };
  }
}
