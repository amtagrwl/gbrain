import { spawnSync } from 'node:child_process';
import {
  chmodSync,
  closeSync,
  constants as fsConstants,
  fstatSync,
  fsyncSync,
  lstatSync,
  mkdtempSync,
  openSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { extname, join } from 'node:path';
import {
  BOUNDED_IMAGE_LOCAL_MAX_PIXELS,
  BOUNDED_IMAGE_MAX_DIMENSION,
  type BoundedDecodedImageInfo,
  type PreparedBoundedDecodedImage,
} from './image-decode.ts';

const IMAGE_OCR_NATIVE_DECODER = '/usr/bin/sips';
const IMAGE_OCR_NATIVE_DECODE_TIMEOUT_MS = 30_000;
const IMAGE_OCR_NATIVE_MAX_PNG_BYTES = BOUNDED_IMAGE_LOCAL_MAX_PIXELS * 4 + 1024 * 1024;

type BoundedImageOcrFormat = BoundedDecodedImageInfo['format'];
type BoundedImageOcrInfo = BoundedDecodedImageInfo;
type PreparedBoundedImageOcrBytes = PreparedBoundedDecodedImage;

function validateBoundedImageDimensions(
  width: number,
  height: number,
): Omit<BoundedImageOcrInfo, 'format' | 'frameCount'> {
  if (
    !Number.isSafeInteger(width)
    || !Number.isSafeInteger(height)
    || width <= 0
    || height <= 0
  ) {
    throw new Error('Image OCR dimensions must be positive safe integers');
  }
  if (width > BOUNDED_IMAGE_MAX_DIMENSION || height > BOUNDED_IMAGE_MAX_DIMENSION) {
    throw new Error(`Image OCR dimension exceeds provider limit ${BOUNDED_IMAGE_MAX_DIMENSION}`);
  }
  const pixels = width * height;
  if (!Number.isSafeInteger(pixels) || pixels > BOUNDED_IMAGE_LOCAL_MAX_PIXELS) {
    throw new Error(`Image OCR decoded pixel count exceeds local ceiling ${BOUNDED_IMAGE_LOCAL_MAX_PIXELS}`);
  }
  return { width, height, pixels };
}

function checkedInfo(
  format: BoundedImageOcrFormat,
  width: number,
  height: number,
): BoundedImageOcrInfo {
  return { format, frameCount: 1, ...validateBoundedImageDimensions(width, height) };
}

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function inspectPng(bytes: Buffer): BoundedImageOcrInfo {
  const signature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  if (bytes.length < 45 || !bytes.subarray(0, 8).equals(signature)) {
    throw new Error('Malformed PNG signature');
  }
  let offset = 8;
  let width = 0;
  let height = 0;
  let sawHeader = false;
  let sawData = false;
  let sawEnd = false;
  while (offset < bytes.length) {
    if (offset + 12 > bytes.length) throw new Error('Malformed PNG chunk header');
    const length = bytes.readUInt32BE(offset);
    const chunkEnd = offset + 12 + length;
    if (!Number.isSafeInteger(chunkEnd) || chunkEnd > bytes.length) throw new Error('Malformed PNG chunk length');
    const type = bytes.toString('ascii', offset + 4, offset + 8);
    const storedCrc = bytes.readUInt32BE(offset + 8 + length);
    if (crc32(bytes.subarray(offset + 4, offset + 8 + length)) !== storedCrc) {
      throw new Error('Malformed PNG chunk CRC');
    }
    if (!sawHeader && type !== 'IHDR') throw new Error('Malformed PNG missing leading IHDR');
    if (type === 'IHDR') {
      if (sawHeader || length !== 13) throw new Error('Malformed PNG IHDR');
      width = bytes.readUInt32BE(offset + 8);
      height = bytes.readUInt32BE(offset + 12);
      const bitDepth = bytes[offset + 16];
      const colorType = bytes[offset + 17];
      const allowedDepths: Record<number, readonly number[]> = {
        0: [1, 2, 4, 8, 16], 2: [8, 16], 3: [1, 2, 4, 8], 4: [8, 16], 6: [8, 16],
      };
      if (
        !allowedDepths[colorType]?.includes(bitDepth)
        || bytes[offset + 18] !== 0
        || bytes[offset + 19] !== 0
        || (bytes[offset + 20] !== 0 && bytes[offset + 20] !== 1)
      ) throw new Error('Malformed PNG IHDR fields');
      sawHeader = true;
      // Enforce before a decoder can allocate an RGBA output buffer.
      validateBoundedImageDimensions(width, height);
    } else if (type === 'IDAT') {
      if (!sawHeader || sawEnd || length === 0) throw new Error('Malformed PNG IDAT');
      sawData = true;
    } else if (type === 'acTL' || type === 'fcTL' || type === 'fdAT') {
      throw new Error('Animated or multiframe PNG is not allowed');
    } else if (type === 'IEND') {
      if (length !== 0 || !sawData || chunkEnd !== bytes.length) throw new Error('Malformed PNG IEND');
      sawEnd = true;
    }
    offset = chunkEnd;
  }
  if (!sawHeader || !sawData || !sawEnd) throw new Error('Malformed PNG structure');
  return checkedInfo('png', width, height);
}

const JPEG_SOF_MARKERS = new Set([
  0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf,
]);

function inspectJpeg(bytes: Buffer): BoundedImageOcrInfo {
  if (bytes.length < 12 || bytes[0] !== 0xff || bytes[1] !== 0xd8) {
    throw new Error('Malformed JPEG signature');
  }
  let offset = 2;
  let width = 0;
  let height = 0;
  let sawSof = false;
  let sawScan = false;
  let sawEnd = false;
  while (offset < bytes.length) {
    if (bytes[offset] !== 0xff) {
      if (!sawScan) throw new Error('Malformed JPEG marker stream');
      offset++;
      continue;
    }
    while (offset < bytes.length && bytes[offset] === 0xff) offset++;
    if (offset >= bytes.length) throw new Error('Malformed JPEG terminal marker');
    const marker = bytes[offset++];
    if (marker === 0x00 || (marker >= 0xd0 && marker <= 0xd7)) continue;
    if (marker === 0xd9) {
      sawEnd = true;
      if (offset !== bytes.length) throw new Error('Malformed JPEG trailing bytes');
      break;
    }
    if (marker === 0xd8 || marker === 0x01) continue;
    if (offset + 2 > bytes.length) throw new Error('Malformed JPEG segment length');
    const length = bytes.readUInt16BE(offset);
    if (length < 2 || offset + length > bytes.length) throw new Error('Malformed JPEG segment');
    const dataStart = offset + 2;
    const dataEnd = offset + length;
    if (JPEG_SOF_MARKERS.has(marker)) {
      if (sawSof || length < 8) throw new Error('Malformed or multiframe JPEG');
      height = bytes.readUInt16BE(dataStart + 1);
      width = bytes.readUInt16BE(dataStart + 3);
      validateBoundedImageDimensions(width, height);
      sawSof = true;
    }
    if (marker === 0xe2 && bytes.toString('ascii', dataStart, Math.min(dataStart + 4, dataEnd)) === 'MPF\0') {
      throw new Error('Multiframe JPEG/MPO is not allowed');
    }
    if (marker === 0xda) sawScan = true;
    offset = dataEnd;
  }
  if (!sawSof || !sawScan || !sawEnd) throw new Error('Malformed JPEG structure');
  return checkedInfo('jpeg', width, height);
}

function skipGifSubBlocks(bytes: Buffer, start: number): number {
  let offset = start;
  let total = 0;
  while (true) {
    if (offset >= bytes.length) throw new Error('Malformed GIF sub-blocks');
    const length = bytes[offset++];
    if (length === 0) return offset;
    total += length;
    if (offset + length > bytes.length) throw new Error('Malformed GIF sub-block length');
    offset += length;
    if (total > bytes.length) throw new Error('Malformed GIF sub-block accounting');
  }
}

function inspectGif(bytes: Buffer): BoundedImageOcrInfo {
  const header = bytes.toString('ascii', 0, 6);
  if (bytes.length < 14 || (header !== 'GIF87a' && header !== 'GIF89a')) {
    throw new Error('Malformed GIF signature');
  }
  const canvasWidth = bytes.readUInt16LE(6);
  const canvasHeight = bytes.readUInt16LE(8);
  validateBoundedImageDimensions(canvasWidth, canvasHeight);
  const packed = bytes[10];
  let offset = 13;
  if (packed & 0x80) offset += 3 * (1 << ((packed & 0x07) + 1));
  if (offset > bytes.length) throw new Error('Malformed GIF global color table');
  let frameCount = 0;
  let frameWidth = 0;
  let frameHeight = 0;
  let sawTrailer = false;
  while (offset < bytes.length) {
    const introducer = bytes[offset++];
    if (introducer === 0x3b) {
      sawTrailer = true;
      if (offset !== bytes.length) throw new Error('Malformed GIF trailing bytes');
      break;
    }
    if (introducer === 0x2c) {
      if (offset + 9 > bytes.length) throw new Error('Malformed GIF image descriptor');
      frameCount++;
      if (frameCount !== 1) throw new Error('Animated or multiframe GIF is not allowed');
      frameWidth = bytes.readUInt16LE(offset + 4);
      frameHeight = bytes.readUInt16LE(offset + 6);
      validateBoundedImageDimensions(frameWidth, frameHeight);
      const imagePacked = bytes[offset + 8];
      offset += 9;
      if (imagePacked & 0x80) offset += 3 * (1 << ((imagePacked & 0x07) + 1));
      if (offset >= bytes.length) throw new Error('Malformed GIF local color table');
      const minimumCodeSize = bytes[offset++];
      if (minimumCodeSize < 2 || minimumCodeSize > 8) throw new Error('Malformed GIF LZW code size');
      offset = skipGifSubBlocks(bytes, offset);
      continue;
    }
    if (introducer === 0x21) {
      if (offset >= bytes.length) throw new Error('Malformed GIF extension');
      const label = bytes[offset++];
      if (label === 0xff || label === 0x01) {
        throw new Error('Animated or unsupported multiframe GIF extension');
      }
      if (label === 0xf9) {
        if (offset + 6 > bytes.length || bytes[offset] !== 4 || bytes[offset + 5] !== 0) {
          throw new Error('Malformed GIF graphic control extension');
        }
        offset += 6;
      } else {
        offset = skipGifSubBlocks(bytes, offset);
      }
      continue;
    }
    throw new Error('Malformed GIF block introducer');
  }
  if (!sawTrailer || frameCount !== 1) throw new Error('Malformed GIF structure');
  return checkedInfo('gif', frameWidth, frameHeight);
}

function readUInt24LE(bytes: Buffer, offset: number): number {
  return bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16);
}

function inspectWebp(bytes: Buffer): BoundedImageOcrInfo {
  if (
    bytes.length < 20
    || bytes.toString('ascii', 0, 4) !== 'RIFF'
    || bytes.toString('ascii', 8, 12) !== 'WEBP'
    || bytes.readUInt32LE(4) + 8 !== bytes.length
  ) throw new Error('Malformed WebP RIFF header');
  let offset = 12;
  let width = 0;
  let height = 0;
  let imageChunks = 0;
  let canvasDimensions: { width: number; height: number } | null = null;
  while (offset < bytes.length) {
    if (offset + 8 > bytes.length) throw new Error('Malformed WebP chunk header');
    const type = bytes.toString('ascii', offset, offset + 4);
    const length = bytes.readUInt32LE(offset + 4);
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    const paddedEnd = dataEnd + (length & 1);
    if (dataEnd > bytes.length || paddedEnd > bytes.length) throw new Error('Malformed WebP chunk length');
    if (type === 'ANIM' || type === 'ANMF') throw new Error('Animated or multiframe WebP is not allowed');
    if (type === 'VP8X') {
      if (length !== 10 || (bytes[dataStart] & 0x02) !== 0) throw new Error('Malformed or animated WebP VP8X');
      canvasDimensions = {
        width: readUInt24LE(bytes, dataStart + 4) + 1,
        height: readUInt24LE(bytes, dataStart + 7) + 1,
      };
      validateBoundedImageDimensions(canvasDimensions.width, canvasDimensions.height);
    } else if (type === 'VP8 ') {
      if (
        length < 10
        || bytes[dataStart + 3] !== 0x9d
        || bytes[dataStart + 4] !== 0x01
        || bytes[dataStart + 5] !== 0x2a
      ) throw new Error('Malformed WebP VP8 frame');
      imageChunks++;
      width = bytes.readUInt16LE(dataStart + 6) & 0x3fff;
      height = bytes.readUInt16LE(dataStart + 8) & 0x3fff;
      validateBoundedImageDimensions(width, height);
    } else if (type === 'VP8L') {
      if (length < 5 || bytes[dataStart] !== 0x2f) throw new Error('Malformed WebP VP8L frame');
      imageChunks++;
      const bits = bytes.readUInt32LE(dataStart + 1);
      width = (bits & 0x3fff) + 1;
      height = ((bits >>> 14) & 0x3fff) + 1;
      validateBoundedImageDimensions(width, height);
    }
    offset = paddedEnd;
  }
  if (offset !== bytes.length || imageChunks !== 1) throw new Error('Malformed or multiframe WebP structure');
  if (canvasDimensions && (canvasDimensions.width !== width || canvasDimensions.height !== height)) {
    throw new Error('WebP canvas and frame dimensions disagree');
  }
  return checkedInfo('webp', width, height);
}

interface IsoBox {
  type: string;
  start: number;
  payloadStart: number;
  end: number;
}

function isoBoxes(bytes: Buffer, start: number, end: number): IsoBox[] {
  const boxes: IsoBox[] = [];
  let offset = start;
  while (offset < end) {
    if (offset + 8 > end) throw new Error('Malformed ISO-BMFF box header');
    const size32 = bytes.readUInt32BE(offset);
    const type = bytes.toString('ascii', offset + 4, offset + 8);
    let size: number;
    let header = 8;
    if (size32 === 1) {
      if (offset + 16 > end) throw new Error('Malformed ISO-BMFF large box');
      const large = bytes.readBigUInt64BE(offset + 8);
      if (large > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('Unsafe ISO-BMFF box size');
      size = Number(large);
      header = 16;
    } else {
      size = size32 === 0 ? end - offset : size32;
    }
    if (size < header || offset + size > end) throw new Error('Malformed ISO-BMFF box length');
    boxes.push({ type, start: offset, payloadStart: offset + header, end: offset + size });
    offset += size;
  }
  return boxes;
}

const ISO_CONTAINER_BOXES = new Set(['meta', 'iprp', 'ipco', 'moov', 'trak', 'mdia', 'minf', 'stbl']);

function collectIsoDimensions(bytes: Buffer, boxes: IsoBox[], out: Array<{ width: number; height: number }>): void {
  for (const box of boxes) {
    if (box.type === 'ispe') {
      if (box.payloadStart + 12 > box.end) throw new Error('Malformed ISO-BMFF ispe box');
      const width = bytes.readUInt32BE(box.payloadStart + 4);
      const height = bytes.readUInt32BE(box.payloadStart + 8);
      validateBoundedImageDimensions(width, height);
      out.push({ width, height });
      continue;
    }
    if (ISO_CONTAINER_BOXES.has(box.type)) {
      const childStart = box.payloadStart + (box.type === 'meta' ? 4 : 0);
      if (childStart > box.end) throw new Error('Malformed ISO-BMFF container');
      collectIsoDimensions(bytes, isoBoxes(bytes, childStart, box.end), out);
    }
  }
}

function inspectIsoImage(bytes: Buffer, expected: 'heic' | 'avif'): {
  format: 'heic' | 'avif';
  dimensions: Array<{ width: number; height: number }>;
} {
  if (bytes.length < 24) throw new Error('Malformed ISO-BMFF image');
  const top = isoBoxes(bytes, 0, bytes.length);
  const ftyp = top.find(box => box.type === 'ftyp');
  if (!ftyp || ftyp.payloadStart + 8 > ftyp.end || (ftyp.end - ftyp.payloadStart) % 4 !== 0) {
    throw new Error('Malformed ISO-BMFF ftyp box');
  }
  const brands: string[] = [];
  brands.push(bytes.toString('ascii', ftyp.payloadStart, ftyp.payloadStart + 4));
  for (let offset = ftyp.payloadStart + 8; offset + 4 <= ftyp.end; offset += 4) {
    brands.push(bytes.toString('ascii', offset, offset + 4));
  }
  const avif = brands.some(brand => brand === 'avif' || brand === 'avis');
  const heic = brands.some(brand => ['heic', 'heix', 'hevc', 'hevx', 'mif1', 'msf1'].includes(brand));
  if (expected === 'avif' ? !avif : !heic || avif) throw new Error('Image extension does not match ISO-BMFF brand');
  if (brands.some(brand => ['avis', 'msf1', 'hevs'].includes(brand)) || top.some(box => box.type === 'moov')) {
    throw new Error('Animated or multiframe ISO-BMFF image is not allowed');
  }
  const dimensions: Array<{ width: number; height: number }> = [];
  collectIsoDimensions(bytes, top, dimensions);
  if (dimensions.length === 0) throw new Error('ISO-BMFF image has no bounded dimensions');
  return { format: expected, dimensions };
}

function asArrayBuffer(bytes: Buffer): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

function assertDecodedRgba(
  decoded: { data: ArrayLike<number>; width: number; height: number },
  expected: { width: number; height: number },
): void {
  if (
    decoded.width !== expected.width
    || decoded.height !== expected.height
    || decoded.data.length !== expected.width * expected.height * 4
  ) throw new Error('Decoded image dimensions disagree with bounded metadata');
}

async function encodePng(decoded: { data: ArrayLike<number>; width: number; height: number }): Promise<Buffer> {
  const encode = (await import('@jsquash/png/encode.js')).default;
  const output = await encode({
    data: new Uint8ClampedArray(decoded.data),
    width: decoded.width,
    height: decoded.height,
  });
  const bytes = Buffer.from(output);
  inspectPng(bytes);
  return bytes;
}

async function preparePng(bytes: Buffer): Promise<PreparedBoundedImageOcrBytes> {
  const info = inspectPng(bytes);
  const decode = (await import('@jsquash/png/decode.js')).default;
  const decoded = await decode(asArrayBuffer(bytes));
  assertDecodedRgba(decoded, info);
  return { buf: bytes, mime: 'image/png', info };
}

function decodeNativeCodecToPng(
  bytes: Buffer,
  format: 'jpeg' | 'gif' | 'webp',
): Buffer {
  const directory = mkdtempSync(join(tmpdir(), 'gbrain-image-ocr-decode-'));
  chmodSync(directory, 0o700);
  const extension = format === 'jpeg' ? 'jpg' : format;
  const inputPath = join(directory, `input.${extension}`);
  const outputPath = join(directory, 'output.png');
  const noFollow = typeof fsConstants.O_NOFOLLOW === 'number' ? fsConstants.O_NOFOLLOW : 0;
  try {
    const inputFd = openSync(
      inputPath,
      fsConstants.O_WRONLY | fsConstants.O_CREAT | fsConstants.O_EXCL | noFollow,
      0o600,
    );
    try {
      writeFileSync(inputFd, bytes);
      fsyncSync(inputFd);
    } finally {
      closeSync(inputFd);
    }

    const result = spawnSync(
      IMAGE_OCR_NATIVE_DECODER,
      ['-s', 'format', 'png', inputPath, '--out', outputPath],
      {
        encoding: 'buffer',
        env: { PATH: '/usr/bin:/bin', TMPDIR: directory, LANG: 'C' },
        maxBuffer: 64 * 1024,
        shell: false,
        stdio: ['ignore', 'pipe', 'pipe'],
        timeout: IMAGE_OCR_NATIVE_DECODE_TIMEOUT_MS,
        windowsHide: true,
      },
    );
    if (
      result.error
      || result.signal !== null
      || result.status !== 0
      || result.stderr.length !== 0
    ) {
      throw new Error(`Image OCR could not fully decode ${format}`);
    }

    const visibleOutput = lstatSync(outputPath);
    if (
      visibleOutput.isSymbolicLink()
      || !visibleOutput.isFile()
      || visibleOutput.size <= 0
      || visibleOutput.size > IMAGE_OCR_NATIVE_MAX_PNG_BYTES
    ) {
      throw new Error(`Image OCR native ${format} decoder produced invalid output`);
    }
    const outputFd = openSync(outputPath, fsConstants.O_RDONLY | noFollow);
    try {
      const before = fstatSync(outputFd);
      if (
        !before.isFile()
        || before.dev !== visibleOutput.dev
        || before.ino !== visibleOutput.ino
        || before.size !== visibleOutput.size
      ) {
        throw new Error(`Image OCR native ${format} output identity changed`);
      }
      const output = readFileSync(outputFd);
      const after = fstatSync(outputFd);
      if (
        after.dev !== before.dev
        || after.ino !== before.ino
        || after.size !== before.size
        || output.length !== before.size
      ) {
        throw new Error(`Image OCR native ${format} output changed while reading`);
      }
      return output;
    } finally {
      closeSync(outputFd);
    }
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

async function prepareNativeCodec(
  bytes: Buffer,
  info: BoundedImageOcrInfo,
  format: 'jpeg' | 'gif' | 'webp',
): Promise<PreparedBoundedImageOcrBytes> {
  const decodedPng = await preparePng(decodeNativeCodecToPng(bytes, format));
  if (
    decodedPng.info.width !== info.width
    || decodedPng.info.height !== info.height
    || decodedPng.info.pixels !== info.pixels
    || decodedPng.info.frameCount !== info.frameCount
  ) {
    throw new Error(`Decoded ${format} dimensions or frame count disagree with bounded metadata`);
  }
  return { buf: decodedPng.buf, mime: 'image/png', info };
}

interface HeicImageHandle {
  width: number;
  height: number;
  decode(): Promise<{ data: ArrayLike<number>; width: number; height: number }>;
}

interface HeicImageHandles extends Array<HeicImageHandle> {
  dispose(): void;
}

async function prepareHeic(bytes: Buffer): Promise<PreparedBoundedImageOcrBytes> {
  inspectIsoImage(bytes, 'heic');
  const decode = (await import('heic-decode')).default as unknown as {
    all(input: { buffer: Buffer }): Promise<HeicImageHandles>;
  };
  const images = await decode.all({ buffer: bytes });
  try {
    if (images.length !== 1) throw new Error('Animated or multiframe HEIC/HEIF is not allowed');
    const image = images[0];
    const info = checkedInfo('heic', image.width, image.height);
    const decoded = await image.decode();
    assertDecodedRgba(decoded, info);
    return { buf: await encodePng(decoded), mime: 'image/png', info };
  } finally {
    images.dispose();
  }
}

async function prepareAvif(bytes: Buffer): Promise<PreparedBoundedImageOcrBytes> {
  const metadata = inspectIsoImage(bytes, 'avif');
  // Every advertised ispe dimension is bounded before the decoder is allowed
  // to allocate. The decoded primary dimensions must then match one of them.
  const avifWasmModule = await import('@jsquash/avif/codec/dec/avif_dec.wasm', { with: { type: 'file' } });
  const avifMod = await import('@jsquash/avif/decode.js');
  const wasmBytes = readFileSync((avifWasmModule as { default: string }).default);
  const wasmModule = await WebAssembly.compile(asArrayBuffer(wasmBytes));
  await avifMod.init(wasmModule);
  const decoded = await avifMod.default(asArrayBuffer(bytes));
  if (!decoded) throw new Error('AVIF decoder returned no image');
  const matching = metadata.dimensions.find(
    dimensions => dimensions.width === decoded.width && dimensions.height === decoded.height,
  );
  if (!matching) throw new Error('Decoded AVIF dimensions disagree with bounded metadata');
  const info = checkedInfo('avif', decoded.width, decoded.height);
  assertDecodedRgba(decoded, info);
  return { buf: await encodePng(decoded), mime: 'image/png', info };
}

export async function prepareBoundedImageDecode(
  bytes: Buffer,
  filePathOrExtension: string,
): Promise<PreparedBoundedDecodedImage> {
  const extension = filePathOrExtension.startsWith('.')
    ? filePathOrExtension.toLowerCase()
    : extname(filePathOrExtension).toLowerCase();
  if (bytes.length === 0) throw new Error('Image OCR rejects zero-byte images');
  switch (extension) {
    case '.png':
      return preparePng(bytes);
    case '.jpg':
    case '.jpeg':
      return prepareNativeCodec(bytes, inspectJpeg(bytes), 'jpeg');
    case '.gif':
      return prepareNativeCodec(bytes, inspectGif(bytes), 'gif');
    case '.webp':
      return prepareNativeCodec(bytes, inspectWebp(bytes), 'webp');
    case '.heic':
    case '.heif':
      return prepareHeic(bytes);
    case '.avif':
      return prepareAvif(bytes);
    default:
      throw new Error(`Unsupported bounded image OCR extension: ${extension || '(none)'}`);
  }
}
