/**
 * StorageBackend — pluggable interface for binary file storage.
 *
 * GBrain is agnostic about where files live. The setup skill picks
 * the backend (Supabase Storage or S3/R2/MinIO), gbrain doesn't care.
 */

import { closeSync, fstatSync, openSync, readFileSync, readSync } from 'node:fs';

export class StorageReadLimitError extends Error {
  constructor(public readonly limitBytes: number) {
    super(`Storage object exceeds read limit of ${limitBytes} bytes`);
    this.name = 'StorageReadLimitError';
  }
}

function validateReadLimit(maxBytes: number): void {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 0) {
    throw new Error('Storage read limit must be a non-negative safe integer');
  }
}

/** Read a local file while refusing content beyond maxBytes (plus one probe byte). */
export function readLocalFileBounded(path: string, maxBytes?: number): Buffer {
  if (maxBytes === undefined) return readFileSync(path);
  validateReadLimit(maxBytes);

  const fd = openSync(path, 'r');
  try {
    if (fstatSync(fd).size > maxBytes) throw new StorageReadLimitError(maxBytes);
    const chunks: Buffer[] = [];
    let total = 0;
    while (true) {
      const probeRemaining = maxBytes + 1 - total;
      if (probeRemaining <= 0) throw new StorageReadLimitError(maxBytes);
      const chunk = Buffer.allocUnsafe(Math.min(64 * 1024, probeRemaining));
      const count = readSync(fd, chunk, 0, chunk.length, null);
      if (count === 0) break;
      total += count;
      if (total > maxBytes) throw new StorageReadLimitError(maxBytes);
      chunks.push(chunk.subarray(0, count));
    }
    return Buffer.concat(chunks, total);
  } finally {
    closeSync(fd);
  }
}

/** Collect an SDK byte stream while refusing payload beyond the byte cap. */
export async function collectBytesBounded(
  stream: AsyncIterable<Uint8Array>,
  maxBytes?: number,
): Promise<Buffer> {
  if (maxBytes !== undefined) validateReadLimit(maxBytes);
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const value of stream) {
    const chunk = Buffer.from(value.buffer, value.byteOffset, value.byteLength);
    total += chunk.length;
    if (maxBytes !== undefined && total > maxBytes) {
      throw new StorageReadLimitError(maxBytes);
    }
    chunks.push(chunk);
  }
  return Buffer.concat(chunks, total);
}

/** Consume a fetch response body with a hard payload cap, even if Range is ignored. */
export async function readResponseBodyBounded(
  response: Response,
  maxBytes?: number,
): Promise<Buffer> {
  if (maxBytes === undefined) return Buffer.from(await response.arrayBuffer());
  validateReadLimit(maxBytes);
  const declaredLength = Number(response.headers.get('content-length'));
  if (Number.isFinite(declaredLength) && declaredLength > maxBytes) {
    await response.body?.cancel().catch(() => undefined);
    throw new StorageReadLimitError(maxBytes);
  }
  if (!response.body) return Buffer.alloc(0);

  const reader = response.body.getReader();
  const chunks: Buffer[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maxBytes) {
        await reader.cancel().catch(() => undefined);
        throw new StorageReadLimitError(maxBytes);
      }
      chunks.push(Buffer.from(value.buffer, value.byteOffset, value.byteLength));
    }
  } finally {
    reader.releaseLock();
  }
  return Buffer.concat(chunks, total);
}

export interface StorageBackend {
  upload(path: string, data: Buffer, mime?: string): Promise<void>;
  download(path: string, maxBytes?: number): Promise<Buffer>;
  delete(path: string): Promise<void>;
  exists(path: string): Promise<boolean>;
  list(prefix: string): Promise<string[]>;
  getUrl(path: string): Promise<string>;
}

export interface StorageConfig {
  backend: 's3' | 'supabase' | 'local';
  bucket: string;
  region?: string;
  endpoint?: string;
  // S3 credentials
  accessKeyId?: string;
  secretAccessKey?: string;
  // Supabase credentials
  projectUrl?: string;
  serviceRoleKey?: string;
  // Local (for testing)
  localPath?: string;
}

/**
 * Create a StorageBackend from config.
 */
export async function createStorage(config: StorageConfig): Promise<StorageBackend> {
  switch (config.backend) {
    case 's3': {
      const { S3Storage } = await import('./storage/s3.ts');
      return new S3Storage(config);
    }
    case 'supabase': {
      const { SupabaseStorage } = await import('./storage/supabase.ts');
      return new SupabaseStorage(config);
    }
    case 'local': {
      const { LocalStorage } = await import('./storage/local.ts');
      return new LocalStorage(config.localPath || '/tmp/gbrain-storage');
    }
    default:
      throw new Error(`Unknown storage backend: ${config.backend}`);
  }
}
