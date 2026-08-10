import {
  closeSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { dirname, join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { gbrainPath } from './config.ts';

const TOKEN_BRAND: unique symbol = Symbol('gbrain-image-import-fence');
const OWNER_FILE = 'owner.json';
const DEFAULT_TIMEOUT_MS = 10 * 60 * 1000;
const DEFAULT_POLL_MS = 50;

export interface ImageImportFenceToken {
  readonly [TOKEN_BRAND]: true;
  readonly ownerToken: string;
  readonly lockRoot: string;
}

export interface ImageImportFenceOptions {
  /** Override the named lock directory for hermetic tests. */
  lockRoot?: string;
  /** Total wait budget. Defaults above the bounded provider timeout. */
  timeoutMs?: number;
  pollMs?: number;
}

interface OwnerRecord {
  pid: number;
  owner_token: string;
  acquired_at: string;
}

export class ImageImportFenceError extends Error {
  constructor(readonly lockRoot: string) {
    super(`Timed out waiting for the shared image-import fence: ${lockRoot}`);
    this.name = 'ImageImportFenceError';
  }
}

export function defaultImageImportFenceRoot(): string {
  return gbrainPath('image-import-fence.lock');
}

function readOwner(lockRoot: string): OwnerRecord | null {
  try {
    const parsed = JSON.parse(readFileSync(join(lockRoot, OWNER_FILE), 'utf8')) as Partial<OwnerRecord>;
    if (
      !Number.isInteger(parsed.pid)
      || (parsed.pid ?? 0) <= 0
      || typeof parsed.owner_token !== 'string'
      || parsed.owner_token.length === 0
      || typeof parsed.acquired_at !== 'string'
    ) return null;
    return parsed as OwnerRecord;
  } catch {
    return null;
  }
}

function writeOwner(lockRoot: string, owner: OwnerRecord): void {
  const path = join(lockRoot, OWNER_FILE);
  const fd = openSync(path, 'wx', 0o600);
  try {
    writeFileSync(fd, `${JSON.stringify(owner)}\n`, 'utf8');
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
}

function tryAcquire(lockRoot: string): ImageImportFenceToken | null {
  mkdirSync(dirname(lockRoot), { recursive: true, mode: 0o700 });
  try {
    mkdirSync(lockRoot, { mode: 0o700 });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
    return null;
  }

  const ownerToken = `${process.pid}:${randomUUID()}`;
  try {
    writeOwner(lockRoot, {
      pid: process.pid,
      owner_token: ownerToken,
      acquired_at: new Date().toISOString(),
    });
  } catch (error) {
    try { rmSync(lockRoot, { recursive: true, force: true }); } catch { /* best effort */ }
    throw error;
  }
  return Object.freeze({
    [TOKEN_BRAND]: true as const,
    ownerToken,
    lockRoot,
  });
}

function release(token: ImageImportFenceToken): void {
  const owner = readOwner(token.lockRoot);
  if (!owner || owner.owner_token !== token.ownerToken || owner.pid !== process.pid) {
    return;
  }
  rmSync(token.lockRoot, { recursive: true, force: true });
}

export function assertImageImportFenceToken(token: ImageImportFenceToken): void {
  if (!token || token[TOKEN_BRAND] !== true) {
    throw new Error('Bounded image import requires the held shared image-import fence');
  }
}

/**
 * Cross-process mutex shared by routine image imports and the bounded OCR lane.
 * It is deliberately global per GBRAIN_HOME: simple over-serialization is safer
 * than allowing a source/page race around a paid provider attempt.
 */
export async function withImageImportFence<T>(
  fn: (token: ImageImportFenceToken) => Promise<T>,
  options: ImageImportFenceOptions = {},
): Promise<T> {
  const lockRoot = options.lockRoot ?? defaultImageImportFenceRoot();
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const pollMs = options.pollMs ?? DEFAULT_POLL_MS;
  const deadline = Date.now() + timeoutMs;
  let token = tryAcquire(lockRoot);
  while (!token && Date.now() < deadline) {
    await Bun.sleep(pollMs);
    token = tryAcquire(lockRoot);
  }
  if (!token) throw new ImageImportFenceError(lockRoot);
  try {
    return await fn(token);
  } finally {
    release(token);
  }
}
