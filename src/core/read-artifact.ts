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
  invalid_params: 'read_artifact requires a valid exact source, child page slug, and canonical parent page slug',
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
  parent_page_slug: string;
  content_hash?: string;
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
    || !isValidArtifactSlug(input.parent_page_slug)
    || (input.content_hash !== undefined && !/^[a-f0-9]{64}$/.test(input.content_hash))) {
    throw new ArtifactReadError('invalid_params');
  }
}

/**
 * Imported email images use one stable relationship: `root/filename` is a
 * direct child of the canonical email page `root/root`. The caller must carry
 * that exact parent coordinate from search; runtime never guesses from report
 * ids, subjects, or answer text.
 */
function verifyDirectCanonicalParent(pageSlug: string, parentPageSlug: string): void {
  const child = pageSlug.split('/');
  if (child.length !== 2) throw new ArtifactReadError('artifact_mismatch');
  const [root, filename] = child;
  if (!root || !filename || root === filename || parentPageSlug !== `${root}/${root}`) {
    throw new ArtifactReadError('artifact_mismatch');
  }
}

function selectApprovedFile(files: FileRow[], pageId: number, pageSlug: string, ownerSourceId: string): FileRow {
  if (files.length === 0) throw new ArtifactReadError('artifact_not_found');
  if (files.length !== 1) throw new ArtifactReadError('artifact_mismatch');
  const file = files[0];
  if (file.page_id === null
    || String(file.page_id) !== String(pageId)
    || file.page_slug !== pageSlug
    || file.source_id !== ownerSourceId) {
    throw new ArtifactReadError('artifact_mismatch');
  }
  return file;
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
  verifyDirectCanonicalParent(input.page_slug, input.parent_page_slug);

  const source = (await engine.listAllSources({ includeArchived: false }))
    .find(candidate => candidate.id === input.source_id);
  if (!source) throw new ArtifactReadError('artifact_not_found');

  // Parent proof is always evaluated in the requested source. A same-slug row
  // in `default` (or any other source) cannot authorize an artifact read.
  const parent = await engine.getPage(input.parent_page_slug, { sourceId: input.source_id });
  if (!parent) throw new ArtifactReadError('artifact_not_found');
  if (parent.source_id !== input.source_id || parent.slug !== input.parent_page_slug) {
    throw new ArtifactReadError('artifact_mismatch');
  }
  const parentMessageId = parent.frontmatter?.message_id;
  if (parent.type === 'image'
    || typeof parentMessageId !== 'string'
    || !/^<[^<>\s@]+@[^<>\s@]+>$/.test(parentMessageId)) {
    throw new ArtifactReadError('artifact_mismatch');
  }

  let page = await engine.getPage(input.page_slug, { sourceId: input.source_id });
  if (!page && input.source_id !== 'default') {
    page = await engine.getPage(input.page_slug, { sourceId: 'default' });
  }
  if (!page) throw new ArtifactReadError('artifact_not_found');
  if (page.slug !== input.page_slug
    || (page.source_id !== input.source_id && page.source_id !== 'default')) {
    throw new ArtifactReadError('artifact_mismatch');
  }
  if (page.type !== 'image') throw new ArtifactReadError('unsupported_media_type');
  if (typeof page.content_hash !== 'string' || !/^[a-f0-9]{64}$/.test(page.content_hash)) {
    throw new ArtifactReadError('artifact_stale');
  }
  if (input.content_hash !== undefined && page.content_hash !== input.content_hash) {
    throw new ArtifactReadError('artifact_stale');
  }

  const file = selectApprovedFile(
    await engine.listFilesForPage(page.id),
    page.id,
    input.page_slug,
    page.source_id,
  );
  if (file.content_hash !== page.content_hash) throw new ArtifactReadError('artifact_stale');

  // Correctly owned rows agree on the requested source. Legacy imported image
  // rows may agree on `default`, but only after the exact requested-source
  // parent proof above. No mixed ownership and no third source are admitted.
  const correctlyOwned = page.source_id === input.source_id && file.source_id === input.source_id;
  const legacyOwned = input.source_id !== 'default'
    && page.source_id === 'default'
    && file.source_id === 'default';
  if (!correctlyOwned && !legacyOwned) throw new ArtifactReadError('artifact_mismatch');

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

  let storage: StorageBackend | undefined;
  try {
    storage = config.storage
      ? await createStorage(config.storage as StorageConfig)
      : undefined;
  } catch {
    throw new ArtifactReadError('artifact_unavailable');
  }

  // Deliberately resolve legacy metadata through the requested source's root or
  // configured storage. The `default` ownership marker never widens filesystem
  // scope and is not used to select physical storage.
  const data = await resolveArtifactBytes(file, source.local_path, storage);
  if (data.length !== sizeBytes) throw new ArtifactReadError('artifact_stale');
  if (sniffImageMime(data) !== mimeType) throw new ArtifactReadError('artifact_mismatch');

  const actualHash = createHash('sha256').update(data).digest('hex');
  if (actualHash !== page.content_hash || actualHash !== file.content_hash
    || (input.content_hash !== undefined && actualHash !== input.content_hash)) {
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
