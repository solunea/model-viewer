/* @license Copyright 2026 Solunea. Licensed under the Apache License,
 * Version 2.0. */

import {TransformControls} from 'three/addons/controls/TransformControls.js';

import ModelViewerElementBase, {$needsRender, $onModelLoad, $scene, $userInputElement} from '../model-viewer-base.js';
import {Constructor} from '../utilities.js';

import {ExtraModelElement} from './extra-model.js';

export type ExtraModelTransformMode = 'translate'|'scale';

export interface ExtraModelTransformEventDetail {
  model: ExtraModelElement;
  mode: ExtraModelTransformMode;
  offset: string;
  scale: string;
}

export interface ExtraModelTransformInterface {
  setExtraModelTransform(
      model: ExtraModelElement|null, mode?: ExtraModelTransformMode): boolean;
}

const $transform = Symbol('extraModelTransform');

function controlPointer(event: PointerEvent, input: HTMLElement): PointerEvent {
  const {left, top, width, height} = input.getBoundingClientRect();
  // Three expects normalized x/y and the button in its pointer methods.
  return {
    x: (event.clientX - left) / width * 2 - 1,
    y: 1 - (event.clientY - top) / height * 2,
    button: event.button
  } as PointerEvent;
}

export const ExtraModelTransformMixin =
    <T extends Constructor<ModelViewerElementBase>>(
        ModelViewerElement: T): Constructor<ExtraModelTransformInterface>&T => {
      class ExtraModelTransformViewer extends ModelViewerElement {
        private[$transform]: (() => void)|null = null;

        disconnectedCallback() {
          this.setExtraModelTransform(null);
          super.disconnectedCallback();
        }

        [$onModelLoad]() {
          this.setExtraModelTransform(null);
          super[$onModelLoad]();
        }

        setExtraModelTransform(
            model: ExtraModelElement|null,
            mode: ExtraModelTransformMode = 'translate'): boolean {
          this[$transform]?.();
          this[$transform] = null;
          if (model == null) {
            return true;
          }
          const scene = this[$scene];
          const models = Array.from(this.querySelectorAll('extra-model'));
          const index = models.indexOf(model);
          const sources = models.map((extra) => extra.src);
          const object = scene.models[index + (this.src ? 1 : 0)];
          if ((mode !== 'translate' && mode !== 'scale') ||
              !this.isConnected || !this.loaded || scene.externalRenderer ||
              index < 0 || !object || this.src !== scene.url ||
              sources.length !== scene.extraUrls.length ||
              sources.some((src, i) => src !== scene.extraUrls[i])) {
            return false;
          }

          const input = this[$userInputElement];
          const controls = new TransformControls(scene.getCamera());
          controls.setMode(mode);
          const helper = controls.getHelper();
          scene.add(helper);
          controls.attach(object);
          const render = () => this[$needsRender]();
          controls.addEventListener('change', render);
          render();

          const listener = new AbortController();
          const eventOptions = {capture: true, signal: listener.signal};
          const originalTouchAction = input.style.touchAction;
          input.style.touchAction = 'none';
          const initialPosition = object.position.clone();
          const initialScale = object.scale.clone();
          let pointerId: number|null = null;

          const finish = (commit: boolean) => {
            if (pointerId == null) {
              return;
            }
            const id = pointerId;
            pointerId = null;
            const position = object.position.toArray();
            const size = object.scale.toArray();
            const valid = position.every(Number.isFinite) &&
                size.every((value) => Number.isFinite(value) && value !== 0);
            const changed = !object.position.equals(initialPosition) ||
                !object.scale.equals(initialScale);
            if (!commit || !valid) {
              controls.reset();
            }
            controls.pointerUp(null);
            if (input.hasPointerCapture(id)) {
              input.releasePointerCapture(id);
            }
            if (commit && valid && changed) {
              const offset = position.map((value) => value + 'm').join(' ');
              const scale = size.join(' ');
              model.removeAttribute('data-auto-offset');
              model.setAttribute('offset', offset);
              model.setAttribute('scale', scale);
              this.dispatchEvent(new CustomEvent<ExtraModelTransformEventDetail>(
                  'extra-model-transform', {
                    bubbles: true,
                    composed: true,
                    detail: {model, mode, offset, scale}
                  }));
            }
          };

          input.addEventListener('pointermove', (event) => {
            if (pointerId === event.pointerId) {
              event.stopImmediatePropagation();
              controls.pointerMove(controlPointer(event, input));
            } else if (pointerId == null && event.buttons === 0) {
              controls.pointerHover(controlPointer(event, input));
            }
          }, eventOptions);
          input.addEventListener('pointerdown', (event) => {
            if (pointerId != null) {
              event.stopImmediatePropagation();
              return;
            }
            if (!event.isPrimary || event.button !== 0) {
              return;
            }
            const pointer = controlPointer(event, input);
            helper.updateMatrixWorld(true);
            controls.pointerHover(pointer);
            controls.pointerDown(pointer);
            if (controls.dragging) {
              event.preventDefault();
              event.stopImmediatePropagation();
              pointerId = event.pointerId;
              initialPosition.copy(object.position);
              initialScale.copy(object.scale);
              input.focus({preventScroll: true});
              if (event.isTrusted) {
                input.setPointerCapture(pointerId);
              }
            }
          }, eventOptions);
          const finishPointer = (event: PointerEvent) => {
            if (pointerId === event.pointerId) {
              event.stopImmediatePropagation();
              finish(event.type === 'pointerup');
            }
          };
          input.addEventListener('pointerup', finishPointer, eventOptions);
          input.addEventListener('pointercancel', finishPointer, eventOptions);
          input.addEventListener('lostpointercapture', finishPointer, eventOptions);
          input.addEventListener('wheel', (event) => {
            if (pointerId != null) {
              event.preventDefault();
              event.stopImmediatePropagation();
            }
          }, eventOptions);
          input.addEventListener('keydown', (event) => {
            if (pointerId != null && event.key === 'Escape') {
              event.preventDefault();
              event.stopImmediatePropagation();
              finish(false);
            }
          }, eventOptions);

          const observer = new MutationObserver(() => {
            const current = Array.from(this.querySelectorAll('extra-model'));
            if (this.src !== scene.url || current.length !== models.length ||
                current.some((extra, i) =>
                    extra !== models[i] || extra.src !== sources[i])) {
              this.setExtraModelTransform(null);
            }
          });
          observer.observe(this, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['src']
          });
          this[$transform] = () => {
            finish(false);
            listener.abort();
            observer.disconnect();
            controls.detach();
            helper.removeFromParent();
            helper.dispose();
            if (input.style.touchAction === 'none') {
              input.style.touchAction = originalTouchAction;
            }
            render();
          };
          return true;
        }
      }
      return ExtraModelTransformViewer;
    };
