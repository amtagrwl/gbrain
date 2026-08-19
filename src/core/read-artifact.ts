import { createHash } from 'node:crypto';
import type { GBrainConfig } from './config.ts';
import type { BrainEngine, FileRow } from './engine.ts';
import { resolveFile } from './file-resolver.ts';
import { isValidSourceId } from './source-id.ts';
import {
  createStorage,
  StorageReadLimitError,
  type StorageBackend,
  type StorageConfig,
} from './storage.ts';

export const MAX_ARTIFACT_IMAGE_BYTES = 5 * 1024 * 1024;

export type ArtifactReadErrorCode =
  | 'invalid_params'
  | 'artifact_not_found'
  | 'artifact_mismatch'
  | 'artifact_stale'
  | 'unsupported_media_type'
  | 'artifact_too_large'
  | 'artifact_unavailable';

const ERROR_MESSAGES: Record<ArtifactReadErrorCode, string> = {
  invalid_params: 'read_artifact requires a valid exact source, page slug, and SHA-256 hash',
  artifact_not_found: 'No approved artifact exists at the requested source and page coordinate',
  artifact_mismatch: 'The approved page and file records do not identify the same image artifact',
  artifact_stale: 'The requested artifact hash or stored metadata is stale',
  unsupported_media_type: 'The requested artifact is not a supported v0 image type',
  artifact_too_large: `The requested artifact exceeds the ${MAX_ARTIFACT_IMAGE_BYTES}-byte read limit`,
  artifact_unavailable: 'The approved artifact bytes are unavailable',
};

export class ArtifactReadError extends Error {
  constructor(public readonly code: ArtifactReadErrorCode) {
    super(ERROR_MESSAGES[code]);
    this.name = 'ArtifactReadError';
  }
}

export interface ReadArtifactInput {
  source_id: string;
  page_slug: string;
  content_hash: string;
}

export interface ReadArtifactResult {
  source_id: string;
  page_slug: string;
  mime_type: string;
  size_bytes: number;
  content_hash: string;
  content_base64: string;
}

function isValidArtifactSlug(value: unknown): value is string {
  if (typeof value !== 'string' || value.length === 0 || value.length > 1024) return false;
  if (value.includes('\\') || /[\u0000-\u001f\u007f]/.test(value)) return false;
  const segments = value.split('/');
  return segments.every(segment => segment.length > 0 && segment !== '.' && segment !== '..');
}

function validateInput(input: ReadArtifactInput): void {
  if (!isValidSourceId(input.source_id)
    || !isValidArtifactSlug(input.page_slug)
    || !/^[a-f0-9]{64}$/.test(input.content_hash)) {
    throw new ArtifactReadError('invalid_params');
  }
}

function selectApprovedFile(files: FileRow[], input: ReadArtifactInput): FileRow {
  const coordinateMatches = files.filter(file =>
    file.source_id === input.source_id
    && file.page_slug === input.page_slug
  );
  if (coordinateMatches.length === 0) {
    throw new ArtifactReadError(files.length === 0 ? 'artifact_not_found' : 'artifact_mismatch');
  }
  const hashMatches = coordinateMatches.filter(file => file.content_hash === input.content_hash);
  if (hashMatches.length === 0) throw new ArtifactReadError('artifact_stale');
  if (hashMatches.length !== 1) throw new ArtifactReadError('artifact_mismatch');
  return hashMatches[0];
}

function sniffImageMime(data: Buffer): string | null {
  if (data.length >= 8
    && data.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) {
    return 'image/png';
  }
  if (data.length >= 3 && data[0] === 0xff && data[1] === 0xd8 && data[2] === 0xff) {
    return 'image/jpeg';
  }
  if (data.length >= 12
    && data.subarray(0, 4).toString('ascii') === 'RIFF'
    && data.subarray(8, 12).toString('ascii') === 'WEBP') {
    return 'image/webp';
  }
  return null;
}

async function resolveArtifactBytes(
  file: FileRow,
  sourceRoot: string | null,
  storage: StorageBackend | undefined,
): Promise<Buffer> {
  try {
    if (sourceRoot) {
      return (await resolveFile(file.storage_path, sourceRoot, storage, {
        maxBytes: MAX_ARTIFACT_IMAGE_BYTES,
      })).data;
    }
    if (storage) return await storage.download(file.storage_path, MAX_ARTIFACT_IMAGE_BYTES);
    throw new ArtifactReadError('artifact_unavailable');
  } catch (error) {
    if (error instanceof ArtifactReadError) throw error;
    if (error instanceof StorageReadLimitError) throw new ArtifactReadError('artifact_too_large');
    throw new ArtifactReadError('artifact_unavailable');
  }
}

export async function readArtifact(
  engine: BrainEngine,
  config: GBrainConfig,
  input: ReadArtifactInput,
): Promise<ReadArtifactResult> {
  validateInput(input);

  const page = await engine.getPage(input.page_slug, { sourceId: input.source_id });
  if (!page) throw new ArtifactReadError('artifact_not_found');
  if (page.source_id !== input.source_id || page.slug !== input.page_slug) {
    throw new ArtifactReadError('artifact_mismatch');
  }
  if (page.type !== 'image') throw new ArtifactReadError('unsupported_media_type');
  if (page.content_hash !== input.content_hash) {
    throw new ArtifactReadError('artifact_stale');
  }

  const file = selectApprovedFile(await engine.listFilesForPage(page.id), input);
  if (file.page_id !== page.id) throw new ArtifactReadError('artifact_mismatch');

  const mimeType = file.mime_type;
  if (mimeType !== 'image/png' && mimeType !== 'image/jpeg' && mimeType !== 'image/webp') {
    throw new ArtifactReadError('unsupported_media_type');
  }
  const sizeBytes = Number(file.size_bytes);
  if (!Number.isSafeInteger(sizeBytes) || sizeBytes < 0) {
    throw new ArtifactReadError('artifact_stale');
  }
  if (sizeBytes > MAX_ARTIFACT_IMAGE_BYTES) {
    throw new ArtifactReadError('artifact_too_large');
  }

  const source = (await engine.listAllSources({ includeArchived: false }))
    .find(candidate => candidate.id === input.source_id);
  if (!source) throw new ArtifactReadError('artifact_not_found');

  let storage: StorageBackend | undefined;
  try {
    storage = config.storage
      ? await createStorage(config.storage as StorageConfig)
      : undefined;
  } catch {
    throw new ArtifactReadError('artifact_unavailable');
  }
  const data = await resolveArtifactBytes(file, source.local_path, storage);
  if (data.length !== sizeBytes) throw new ArtifactReadError('artifact_stale');
  if (sniffImageMime(data) !== mimeType) throw new ArtifactReadError('artifact_mismatch');

  const actualHash = createHash('sha256').update(data).digest('hex');
  if (actualHash !== input.content_hash || actualHash !== file.content_hash) {
    throw new ArtifactReadError('artifact_stale');
  }

  return {
    source_id: input.source_id,
    page_slug: input.page_slug,
    mime_type: mimeType,
    size_bytes: data.length,
    content_hash: actualHash,
    content_base64: data.toString('base64'),
  };
}
