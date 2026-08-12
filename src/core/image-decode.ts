/**
 * Provider-independent bounded image decoding.
 *
 * This module intentionally has no pricing, OCR accounting, gateway, or
 * provider imports. The paid OCR wrapper supplies its own policy enrichment;
 * donor adoption consumes only this codec/physical validation result.
 */
export interface BoundedDecodedImageInfo {
  format: 'png' | 'jpeg' | 'gif' | 'webp' | 'heic' | 'avif';
  width: number;
  height: number;
  pixels: number;
  frameCount: 1;
}

export const BOUNDED_IMAGE_MAX_DIMENSION = 8_000;
export const BOUNDED_IMAGE_LOCAL_MAX_PIXELS = 25_000_000;
export const BOUNDED_IMAGE_MAX_BYTES = 20 * 1024 * 1024;
export const BOUNDED_IMAGE_EXTENSIONS = [
  '.png', '.jpg', '.jpeg', '.gif', '.webp', '.heic', '.heif', '.avif',
] as const;

export interface PreparedBoundedDecodedImage {
  buf: Buffer;
  mime: 'image/png' | 'image/jpeg' | 'image/gif' | 'image/webp';
  info: BoundedDecodedImageInfo;
}

export async function prepareBoundedImageDecode(
  bytes: Buffer,
  filePathOrExtension: string,
): Promise<PreparedBoundedDecodedImage> {
  const image = await import('./image-decode-impl.ts');
  return image.prepareBoundedImageDecode(bytes, filePathOrExtension);
}
