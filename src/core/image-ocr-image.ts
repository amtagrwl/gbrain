import { canonicalLookup } from './model-pricing.ts';
import {
  IMAGE_OCR_POLICY_MAX_OUTPUT_TOKENS,
  IMAGE_OCR_POLICY_MODEL,
} from './image-ocr-budget.ts';
import {
  BOUNDED_IMAGE_LOCAL_MAX_PIXELS,
  BOUNDED_IMAGE_MAX_DIMENSION,
  prepareBoundedImageDecode,
  type BoundedDecodedImageInfo,
  type PreparedBoundedDecodedImage,
} from './image-decode.ts';

export const IMAGE_OCR_PROVIDER_MAX_DIMENSION = BOUNDED_IMAGE_MAX_DIMENSION;
export const IMAGE_OCR_LOCAL_MAX_PIXELS = BOUNDED_IMAGE_LOCAL_MAX_PIXELS;
export const IMAGE_OCR_STANDARD_MAX_LONG_EDGE = 1_568;
export const IMAGE_OCR_STANDARD_MAX_PIXELS = 1_150_000;
export const IMAGE_OCR_TOKEN_PIXEL_DIVISOR = 750;
export const IMAGE_OCR_MAX_VISUAL_TOKENS = 1_568;
export const IMAGE_OCR_NONVISUAL_INPUT_TOKEN_ALLOWANCE = 500;

const pricing = canonicalLookup(`anthropic:${IMAGE_OCR_POLICY_MODEL}`);
if (!pricing) throw new Error(`Missing canonical pricing for ${IMAGE_OCR_POLICY_MODEL}`);
export const IMAGE_OCR_INPUT_USD_PER_MTOK = pricing.input;
export const IMAGE_OCR_OUTPUT_USD_PER_MTOK = pricing.output;
export const IMAGE_OCR_CURRENT_WORST_CASE_USD =
  ((IMAGE_OCR_MAX_VISUAL_TOKENS + IMAGE_OCR_NONVISUAL_INPUT_TOKEN_ALLOWANCE)
    * IMAGE_OCR_INPUT_USD_PER_MTOK
    + IMAGE_OCR_POLICY_MAX_OUTPUT_TOKENS * IMAGE_OCR_OUTPUT_USD_PER_MTOK)
  / 1_000_000;

export type BoundedImageOcrFormat = BoundedDecodedImageInfo['format'];

export interface BoundedImageOcrInfo extends BoundedDecodedImageInfo {
  scaledWidth: number;
  scaledHeight: number;
  visualTokens: number;
  worstCaseUsd: number;
}

export interface PreparedBoundedImageOcrBytes
  extends Omit<PreparedBoundedDecodedImage, 'info'> {
  info: BoundedImageOcrInfo;
}

export function validateAndPriceImageOcrDimensions(
  width: number,
  height: number,
): Omit<BoundedImageOcrInfo, 'format' | 'frameCount'> {
  if (
    !Number.isSafeInteger(width)
    || !Number.isSafeInteger(height)
    || width <= 0
    || height <= 0
  ) throw new Error('Image OCR dimensions must be positive safe integers');
  if (width > IMAGE_OCR_PROVIDER_MAX_DIMENSION || height > IMAGE_OCR_PROVIDER_MAX_DIMENSION) {
    throw new Error(`Image OCR dimension exceeds provider limit ${IMAGE_OCR_PROVIDER_MAX_DIMENSION}`);
  }
  const pixels = width * height;
  if (!Number.isSafeInteger(pixels) || pixels > IMAGE_OCR_LOCAL_MAX_PIXELS) {
    throw new Error(`Image OCR decoded pixel count exceeds local ceiling ${IMAGE_OCR_LOCAL_MAX_PIXELS}`);
  }

  const scale = Math.min(
    1,
    IMAGE_OCR_STANDARD_MAX_LONG_EDGE / Math.max(width, height),
    Math.sqrt(IMAGE_OCR_STANDARD_MAX_PIXELS / pixels),
  );
  const scaledWidth = Math.max(1, Math.ceil(width * scale));
  const scaledHeight = Math.max(1, Math.ceil(height * scale));
  const visualTokens = Math.ceil((scaledWidth * scaledHeight) / IMAGE_OCR_TOKEN_PIXEL_DIVISOR);
  if (visualTokens > IMAGE_OCR_MAX_VISUAL_TOKENS) {
    throw new Error('Image OCR visual token calculation exceeds the pinned policy ceiling');
  }
  const worstCaseUsd = (
    (visualTokens + IMAGE_OCR_NONVISUAL_INPUT_TOKEN_ALLOWANCE) * IMAGE_OCR_INPUT_USD_PER_MTOK
    + IMAGE_OCR_POLICY_MAX_OUTPUT_TOKENS * IMAGE_OCR_OUTPUT_USD_PER_MTOK
  ) / 1_000_000;
  return { width, height, pixels, scaledWidth, scaledHeight, visualTokens, worstCaseUsd };
}

/** Paid OCR wrapper: codec validation plus the existing pinned price envelope. */
export async function prepareBoundedImageOcrBytes(
  bytes: Buffer,
  filePathOrExtension: string,
): Promise<PreparedBoundedImageOcrBytes> {
  const prepared = await prepareBoundedImageDecode(bytes, filePathOrExtension);
  return {
    ...prepared,
    info: {
      ...prepared.info,
      ...validateAndPriceImageOcrDimensions(prepared.info.width, prepared.info.height),
    },
  };
}
