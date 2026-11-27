import {ClampToEdgeWrapping, DataTexture, NearestFilter, RedFormat, RepeatWrapping, RGFormat, Texture, UnsignedByteType} from 'three';

export interface SkyboxDepthImage {
  width: number;
  height: number;
  bitDepth: 8|16;
  samples: Uint8Array;
}

export function skyboxSourceFlipY(texture: Texture): boolean {
  return texture.flipY || texture.userData.skyboxSourceFlipY === true;
}

export function depthTexture(image: SkyboxDepthImage): DataTexture {
  const {samples, width, height, bitDepth} = image;
  const texture = new DataTexture(
      samples,
      width,
      height,
      bitDepth === 16 ? RGFormat : RedFormat,
      UnsignedByteType,
      undefined,
      RepeatWrapping,
      ClampToEdgeWrapping,
      NearestFilter,
      NearestFilter);
  texture.needsUpdate = true;
  return texture;
}

export function sampleSkyboxDepth(
    image: SkyboxDepthImage,
    x: number,
    y: number,
    z: number,
    colorFlipY: boolean): number {
  const u = 0.5 + Math.atan2(z, x) / (2 * Math.PI);
  const v = 0.5 + Math.asin(Math.max(-1, Math.min(1, y))) / Math.PI;
  const column = Math.floor((u - Math.floor(u)) * image.width);
  const row = Math.min(
      image.height - 1, Math.floor((colorFlipY ? 1 - v : v) * image.height));
  const offset = row * image.width + column;
  return image.bitDepth === 16 ?
      (image.samples[2 * offset] + 256 * image.samples[2 * offset + 1]) /
          65535 :
      image.samples[offset] / 255;
}

function createDepthWorker() {
  // The distribution embeds the worker so CDN script tags can use it.
  return new Worker(
      new URL('./SkyboxDepthWorker.js', import.meta.url), {type: 'module'});
}

export async function loadSkyboxDepth(
    url: string, withCredentials: boolean, signal: AbortSignal):
    Promise<SkyboxDepthImage> {
  const response = await fetch(url, {
    signal,
    credentials: withCredentials ? 'include' : 'same-origin',
  });
  if (!response.ok) {
    throw new Error(`Could not load panorama depth: ${response.status}`);
  }
  const bytes = await response.arrayBuffer();
  signal.throwIfAborted();

  const worker = createDepthWorker();

  return new Promise((resolve, reject) => {
    const abort = () => finish(() => reject(signal.reason));
    function finish(complete: () => void) {
      worker.terminate();
      signal.removeEventListener('abort', abort);
      complete();
    }

    signal.addEventListener('abort', abort, {once: true});
    worker.onmessage =
        ({data}: MessageEvent<SkyboxDepthImage|{error: string}>) => finish(
            () => 'error' in data ? reject(new Error(data.error)) :
                                    resolve(data));
    worker.onerror = () => finish(
        () => reject(new Error('Could not decode the panorama depth map.')));
    worker.postMessage(bytes, [bytes]);
  });
}
