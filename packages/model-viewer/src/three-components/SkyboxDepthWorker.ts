import {decode} from 'fast-png';

// The depth map uses native PNG samples, independent of color-space metadata.
globalThis.onmessage = ({data: bytes}: MessageEvent<ArrayBuffer>) => {
  try {
    const header = new DataView(bytes);
    const width = header.getUint32(16);
    const height = header.getUint32(20);
    const bitDepth = header.getUint8(24);
    if (width !== 2 * height || height === 0 || width > 4096 ||
        (bitDepth !== 8 && bitDepth !== 16) || header.getUint8(25) !== 0) {
      throw new Error(
          'Expected an 8- or 16-bit grayscale 2:1 PNG up to 4096×2048.');
    }

    const {data} = decode(bytes, {checkCrc: true});
    const samples = new Uint8Array(width * height * (bitDepth / 8));
    if (data instanceof Uint16Array) {
      for (let i = 0; i < data.length; ++i) {
        samples[2 * i] = data[i] & 255;
        samples[2 * i + 1] = data[i] >> 8;
      }
    } else {
      samples.set(data);
    }

    // WorkerGlobalScope is deliberately absent from the window TypeScript lib.
    const worker = globalThis as unknown as {
      postMessage(message: unknown, transfer: Transferable[]): void;
    };
    worker.postMessage({width, height, bitDepth, samples}, [samples.buffer]);
  } catch (error) {
    globalThis.postMessage({error: String(error)});
  }
};
