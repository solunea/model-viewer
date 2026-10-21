# Native panorama areas

This Solunea fork renders area highlights inside the existing Three.js scene.
There is no extra canvas or WebGL context. The feature is enabled by `skybox-only`.
Load a panorama and its usual source model before interacting with it.

## API

```ts
import type {PanoramaAreaDefinition} from '@solunea/model-viewer';

viewer.setPanoramaAreas(areas); // Replaces the ordered collection; [] clears it.
viewer.setPanoramaAreaHighlight('hotspot-rock'); // null clears the highlight.
viewer.setPanoramaAreaHighlight('hotspot-rock', '#ff6600'); // Optional CSS color.
const hit = viewer.panoramaAreaFromPoint(event.clientX, event.clientY);
// hit: {name, u, v} | null. Coordinates are CSS window coordinates.
```

`PanoramaAreaDefinition` has this shape:

```ts
{
  name: string;
  mask: {
    width: number; height: number; // Full equirectangular source dimensions.
    bounds: {x: number; y: number; width: number; height: number};
    bits: Uint8Array; // ceil(bounds.width * bounds.height / 8) bytes.
  };
}
```

Bits are row-major inside the cropped bounds, least significant bit first.
The element snapshots input bytes. To change a mask, submit the collection again.
Names must be nonempty and unique; invalid replacements throw atomically.
The first area containing a point wins overlaps, including when another area is
highlighted. Transparent holes and disconnected islands use the original bits.
No historical JSON format, codec or DOM schema is interpreted by this API.

UVs follow Thaleia's image-source convention. With
`yaw = (u - .25) * 2π`, `pitch = (v - .5) * π`, a point on the interior sphere is
`10 * [sin(yaw)cos(pitch), -sin(pitch), -cos(yaw)cos(pitch)]`.
U wraps at the 360° seam; V runs from top to bottom, including both poles.

The feature does not create hotspot elements. Keep native slotted hotspots for
keyboard focus, popover anchors and `queryHotspot()`. The application owns pointer
gestures, text editing, selected views and highlight priority. Call the hit API
from your own handlers and set the resulting highlight explicitly.

## Rendering and lifecycle

Only a highlighted area allocates a sphere, material and cropped red-channel
texture. Visual masks larger than the GPU limit (at most 4096 pixels on either
axis) are reduced with union sampling; CPU detection retains original precision.
At most one highlight is drawn. Resources are released when the collection is
cleared, the cached area is removed, panorama mode ends or the element disconnects.
Clearing the highlight retains at most its last texture for fast hover re-entry.
Definitions and highlight intent survive reconnection; GPU resources are rebuilt.
Ordinary raycasts, model bounds, shadows and `exportScene()` exclude the overlay.
Updates dirty the native scene; areas do not start an animation loop.
Changing the highlight color updates a uniform without reallocating its mask
texture. The application can pass its computed theme color, including an
inherited CSS custom property, without storing styling in each area definition.

`toBlob()` excludes areas, redraws before copying the image, then restores the
latest highlight even when encoding fails. Concurrent captures of the same
element are serialized. Each capture has its own temporary 2D canvas, so different
elements cannot overwrite each other's encoding buffer. `toDataURL()` remains a
snapshot of the currently displayed image, including highlights.

## Validation

```sh
npm run build --workspace packages/model-viewer
cd packages/model-viewer
npx web-test-runner --config web-test-panorama.config.mjs --static-logging
```

The dedicated configuration runs Chromium, Firefox and WebKit. Tests cover exact
mask detection, overlaps, holes, islands, seam, poles, CSS resizing, replacements,
disposal, reconnection, capture failure, concurrent captures and export exclusion.
See [the standalone example](examples/panorama-areas.html).
