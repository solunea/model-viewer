/* @license Copyright 2026 Solunea. Licensed under the Apache License,
 * Version 2.0. */
import {PropertyValues} from 'lit';

import ModelViewerElementBase, {$needsRender, $renderer, $scene, ToBlobOptions} from '../model-viewer-base.js';
import {PanoramaAreaDefinition, PanoramaAreaHit, PanoramaAreas} from '../three-components/PanoramaAreas.js';
import {Constructor} from '../utilities.js';

export type{PanoramaAreaDefinition, PanoramaAreaHit} from '../three-components/PanoramaAreas.js';

export interface PanoramaAreasInterface {
  setPanoramaAreas(areas: readonly PanoramaAreaDefinition[]): void;
  setPanoramaAreaHighlight(name: string|null, color?: string): void;
  panoramaAreaFromPoint(clientX: number, clientY: number): PanoramaAreaHit|null;
}

const $areas = Symbol('panoramaAreas');
const $captures = Symbol('panoramaAreaCaptures');
const $sync = Symbol('syncPanoramaAreas');

export const PanoramaAreasMixin =
    <T extends Constructor<ModelViewerElementBase>>(
        ModelViewerElement: T): Constructor<PanoramaAreasInterface>&T => {
      class PanoramaAreasModelViewerElement extends ModelViewerElement {
        private[$areas] = new PanoramaAreas(
            this[$scene],
            this[$renderer].threeRenderer?.capabilities.maxTextureSize);
        private[$captures]: Promise<unknown> = Promise.resolve();

        connectedCallback() {
          super.connectedCallback();
          this[$scene].add(this[$areas]);
          this[$sync]();
        }

        disconnectedCallback() {
          this[$areas].dispose();
          super.disconnectedCallback();
        }

        updated(changes: PropertyValues) {
          super.updated(changes);
          this[$sync]();
        }

        private[$sync]() {
          this[$areas].setActive(
              this.isConnected && (this as ModelViewerElementBase & {
                                    skyboxOnly?: boolean
                                  }).skyboxOnly === true);
        }

        setPanoramaAreas(areas: readonly PanoramaAreaDefinition[]) {
          this[$areas].setAreas(areas);
        }

        setPanoramaAreaHighlight(name: string|null, color?: string) {
          this[$areas].setHighlight(name, color);
        }

        panoramaAreaFromPoint(clientX: number, clientY: number): PanoramaAreaHit
            |null {
          return this[$areas].fromPoint(clientX, clientY);
        }

        toBlob(options?: ToBlobOptions): Promise<Blob> {
          const capture = this[$captures].then(async () => {
            const redraw = this[$areas].visible;
            this[$areas].setCapturing(true);
            try {
              if (redraw) {
                this[$needsRender]();
                this[$renderer].render(
                    performance.now(), undefined, this[$scene]);
              }
              return await super.toBlob(options);
            } finally {
              this[$areas].setCapturing(false);
              if (redraw || this[$areas].visible) {
                this[$needsRender]();
                this[$renderer].render(
                    performance.now(), undefined, this[$scene]);
              }
            }
          });
          // A failed capture must not poison the queue for subsequent callers.
          this[$captures] = capture.catch(() => {});
          return capture;
        }
      }
      return PanoramaAreasModelViewerElement;
    };
