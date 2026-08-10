import { createHash } from 'node:crypto';
import {
  closeSync,
  constants as fsConstants,
  fstatSync,
  lstatSync,
  openSync,
  readFileSync,
  realpathSync,
  statSync,
  type Stats,
} from 'node:fs';
import { isAbsolute, relative, resolve, sep } from 'node:path';
import type { BrainEngine } from './engine.ts';
import { gbrainPath } from './config.ts';
import {
  importImageFileWithBoundedOcrText,
  isImageFilePath,
  MAX_IMAGE_BYTES,
  prepareBoundedImageOcrInput,
  type BoundedImageOcrExpectedPageState,
} from './import-file.ts';
import type { ImageImportFenceToken } from './image-import-fence.ts';

export interface ImageOcrManifestEntry {
  source_id: string;
  slug: string;
  file_path: string;
  sha256: string;
}

export interface ValidatedImageOcrManifestEntry extends ImageOcrManifestEntry {
  registered_root: string;
  registered_root_identity: ImageOcrFilesystemIdentity;
  file_identity: ImageOcrFilesystemIdentity;
  image_format: 'png' | 'jpeg' | 'gif' | 'webp' | 'heic' | 'avif';
  image_width: number;
  image_height: number;
  image_pixels: number;
  visual_tokens: number;
  worst_case_usd: number;
}

export interface ImageOcrFilesystemIdentity {
  device: number;
  inode: number;
}

export interface ValidatedImageOcrManifest {
  manifestPath: string;
  manifestHash: string;
  entries: ValidatedImageOcrManifestEntry[];
}

export interface ImageOcrRunReport {
  utc_date: string;
  manifest_hash: string | null;
  requested: number;
  processed: number;
  succeeded: number;
  failed: number;
  skipped: number;
  reservations: number;
  provider_attempts: number;
  successful_provider_receipts: number;
  observed_input_tokens: number;
  observed_cache_creation_input_tokens: number;
  observed_cache_read_input_tokens: number;
  observed_output_tokens: number;
  observed_usd: number;
  persisted_imports: number;
  failures: number;
  daily_calls_before: number | null;
  daily_calls_after: number | null;
  daily_usd_reserved_before: number | null;
  daily_usd_reserved_after: number | null;
  cap: {
    max_images: number | null;
    max_usd: number | null;
    reserve_usd_per_call: number | null;
  };
  first_failing_entry: {
    index: number;
    source_id: string | null;
    slug: string | null;
    code: 'manifest_invalid' | 'import_failed' | 'post_reservation_failure' | 'image_cap' | 'usd_cap';
  } | null;
  status: 'completed' | 'cap_reached' | 'failed' | 'rejected' | 'locked';
  terminal_error?: 'arguments_invalid' | 'manifest_invalid' | 'budget_locked' | 'run_rejected';
}

export class ImageOcrManifestError extends Error {
  constructor(
    readonly index: number,
    readonly sourceId: string | null,
    readonly slug: string | null,
  ) {
    super(`Image OCR manifest entry ${index} is invalid`);
    this.name = 'ImageOcrManifestError';
  }
}

export function defaultImageOcrBudgetDirectory(): string {
  return gbrainPath('ocr-budget');
}

/** Production import adapter kept in the narrow core lane, not broad CLI import. */
export async function importValidatedImageOcrEntry(
  engine: BrainEngine,
  entry: ValidatedImageOcrManifestEntry,
  ocrText: string,
  expectedPageState: BoundedImageOcrExpectedPageState,
  imageImportFenceToken: ImageImportFenceToken,
): Promise<void> {
  const result = await importImageFileWithBoundedOcrText(
    engine,
    entry.file_path,
    entry.slug,
    entry.source_id,
    entry.registered_root,
    entry.sha256,
    ocrText,
    expectedPageState,
    imageImportFenceToken,
  );
  if (result.status !== 'imported') throw new Error('Bounded image import did not complete');
}

function hashBytes(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function filesystemIdentity(stat: Stats): ImageOcrFilesystemIdentity {
  return { device: stat.dev, inode: stat.ino };
}

function sameFilesystemIdentity(
  left: ImageOcrFilesystemIdentity,
  right: ImageOcrFilesystemIdentity,
): boolean {
  return left.device === right.device && left.inode === right.inode;
}

export function inspectImageOcrRegisteredRoot(path: string): {
  canonicalPath: string;
  identity: ImageOcrFilesystemIdentity;
} {
  const visiblePath = resolve(path);
  const visible = lstatSync(visiblePath);
  if (visible.isSymbolicLink()) throw new Error('Bounded image OCR registered source root may not be a symlink');
  const canonicalPath = realpathSync(visiblePath);
  const canonical = statSync(canonicalPath);
  if (!canonical.isDirectory()) throw new Error('Bounded image OCR registered source root is not a directory');
  return { canonicalPath, identity: filesystemIdentity(canonical) };
}

/**
 * Reopen through a no-follow descriptor and bind the pathname, descriptor,
 * source containment, filesystem identity, size, and SHA to one exact buffer.
 */
export function readImageOcrSourceFile(input: {
  filePath: string;
  imageSlug: string;
  registeredRoot: string;
  expectedHash: string;
  expectedFileIdentity: ImageOcrFilesystemIdentity;
}): Buffer {
  if (lstatSync(input.filePath).isSymbolicLink()) {
    throw new Error('Bounded image OCR file may not be a symlink');
  }
  const noFollow = typeof fsConstants.O_NOFOLLOW === 'number' ? fsConstants.O_NOFOLLOW : 0;
  const fd = openSync(input.filePath, fsConstants.O_RDONLY | noFollow);
  try {
    const before = fstatSync(fd);
    const beforeIdentity = filesystemIdentity(before);
    if (!before.isFile() || !sameFilesystemIdentity(beforeIdentity, input.expectedFileIdentity)) {
      throw new Error('Bounded image OCR file identity changed after manifest validation');
    }
    if (before.size > MAX_IMAGE_BYTES) {
      throw new Error(`Bounded image OCR file exceeds ${MAX_IMAGE_BYTES} bytes`);
    }

    const realRoot = realpathSync(input.registeredRoot);
    if (realRoot !== input.registeredRoot) {
      throw new Error('Bounded image OCR registered source root is no longer canonical');
    }
    const realFile = realpathSync(input.filePath);
    const realRelative = canonicalRelativePath(realRoot, realFile);
    if (realRelative.toLowerCase() !== input.imageSlug) {
      throw new Error('Bounded image OCR file is no longer inside its registered source root');
    }
    const pathBefore = statSync(input.filePath);
    if (!sameFilesystemIdentity(filesystemIdentity(pathBefore), beforeIdentity)) {
      throw new Error('Bounded image OCR pathname no longer identifies the opened file');
    }

    const bytes = readFileSync(fd);
    const after = fstatSync(fd);
    const pathAfter = statSync(input.filePath);
    const realFileAfter = realpathSync(input.filePath);
    if (
      !sameFilesystemIdentity(filesystemIdentity(after), beforeIdentity)
      || !sameFilesystemIdentity(filesystemIdentity(pathAfter), beforeIdentity)
      || after.size !== before.size
      || after.mtimeMs !== before.mtimeMs
      || after.ctimeMs !== before.ctimeMs
      || realFileAfter !== realFile
      || bytes.length !== before.size
    ) {
      throw new Error('Bounded image OCR file changed while it was being read');
    }
    if (hashBytes(bytes) !== input.expectedHash) {
      throw new Error('Bounded image OCR file hash changed after manifest validation');
    }
    return bytes;
  } finally {
    closeSync(fd);
  }
}

function canonicalRelativePath(root: string, filePath: string): string {
  const rel = relative(root, filePath);
  if (!rel || rel === '..' || rel.startsWith(`..${sep}`) || isAbsolute(rel)) {
    throw new Error(`Manifest file is outside registered source root: ${filePath}`);
  }
  const posix = rel.split(sep).join('/');
  if (posix.includes('\\') || posix.split('/').some(part => part === '' || part === '.' || part === '..')) {
    throw new Error(`Manifest path is not canonical source-relative: ${filePath}`);
  }
  return posix;
}

function parseManifestEntry(raw: unknown, line: number): ImageOcrManifestEntry {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error(`Manifest line ${line} must be a JSON object`);
  }
  const record = raw as Record<string, unknown>;
  const keys = Object.keys(record).sort();
  const expected = ['file_path', 'sha256', 'slug', 'source_id'];
  if (keys.join('\0') !== expected.join('\0')) {
    throw new Error(`Manifest line ${line} must contain exactly source_id, slug, file_path, sha256`);
  }
  for (const key of expected) {
    if (typeof record[key] !== 'string' || (record[key] as string).length === 0) {
      throw new Error(`Manifest line ${line} has invalid ${key}`);
    }
  }
  if (!/^[a-f0-9]{64}$/.test(record.sha256 as string)) {
    throw new Error(`Manifest line ${line} sha256 must be lowercase hexadecimal`);
  }
  return record as unknown as ImageOcrManifestEntry;
}

/** Full preflight. Completes for every entry before a caller may acquire budget. */
export async function parseAndValidateImageOcrManifest(
  engine: BrainEngine,
  manifestPath: string,
): Promise<ValidatedImageOcrManifest> {
  if (!isAbsolute(manifestPath) && resolve(manifestPath) !== manifestPath) {
    // Relative manifest paths are safe, but canonicalize their identity for hashing/reporting.
    manifestPath = resolve(manifestPath);
  }
  const bytes = readFileSync(manifestPath);
  const text = bytes.toString('utf8');
  if (Buffer.from(text, 'utf8').compare(bytes) !== 0) throw new Error('Manifest must be valid UTF-8');
  const split = text.split('\n');
  if (split.at(-1) === '') split.pop();
  if (split.length === 0 || split.some(line => line.trim().length === 0)) {
    throw new Error('Manifest must contain non-empty JSONL entries');
  }

  const sources = await engine.listAllSources({ includeArchived: false });
  const sourceById = new Map(sources.map(source => [source.id, source]));
  const entries: ValidatedImageOcrManifestEntry[] = [];
  const identities = new Set<string>();
  const paths = new Set<string>();

  for (let i = 0; i < split.length; i++) {
    let raw: unknown;
    let sourceId: string | null = null;
    let slug: string | null = null;
    try {
      raw = JSON.parse(split[i]);
      if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
        const candidate = raw as Record<string, unknown>;
        sourceId = typeof candidate.source_id === 'string' ? candidate.source_id : null;
        slug = typeof candidate.slug === 'string' ? candidate.slug : null;
      }
      const entry = parseManifestEntry(raw, i + 1);
      const identity = `${entry.source_id}\0${entry.slug}`;
      if (identities.has(identity) || paths.has(entry.file_path)) {
        throw new Error(`Manifest line ${i + 1} is a duplicate entry`);
      }
      identities.add(identity);
      paths.add(entry.file_path);

      const source = sourceById.get(entry.source_id);
      if (!source) throw new Error(`Manifest line ${i + 1} references unknown source_id`);
      if (!source.local_path) throw new Error(`Manifest line ${i + 1} source has no registered local_path`);
      if (!isAbsolute(entry.file_path) || resolve(entry.file_path) !== entry.file_path) {
        throw new Error(`Manifest line ${i + 1} file_path must be canonical and absolute`);
      }
      if (lstatSync(entry.file_path).isSymbolicLink()) {
        throw new Error(`Manifest line ${i + 1} file_path may not be a symlink`);
      }
      if (!statSync(entry.file_path).isFile()) throw new Error(`Manifest line ${i + 1} file_path is not a file`);
      const visibleRoot = resolve(source.local_path);
      const rootSnapshot = inspectImageOcrRegisteredRoot(source.local_path);
      const visibleRel = canonicalRelativePath(visibleRoot, entry.file_path);
      const realFile = realpathSync(entry.file_path);
      const realRoot = rootSnapshot.canonicalPath;
      const realRel = canonicalRelativePath(realRoot, realFile);
      if (visibleRel !== realRel) throw new Error(`Manifest line ${i + 1} source/file path normalization is ambiguous`);
      const canonicalSlug = visibleRel.toLowerCase();
      if (entry.slug !== canonicalSlug || entry.slug.includes('\\')) {
        throw new Error(`Manifest line ${i + 1} slug does not match canonical source-relative path`);
      }
      if (!isImageFilePath(entry.file_path) || !isImageFilePath(entry.slug)) {
        throw new Error(`Manifest line ${i + 1} is not a supported image type`);
      }
      const fileIdentity = filesystemIdentity(statSync(realFile));
      const sourceBytes = readImageOcrSourceFile({
        filePath: entry.file_path,
        imageSlug: entry.slug,
        registeredRoot: realRoot,
        expectedHash: entry.sha256,
        expectedFileIdentity: fileIdentity,
      });
      const existing = await engine.getPage(entry.slug, { sourceId: entry.source_id });
      if (existing?.content_hash === entry.sha256) {
        throw new Error(`Manifest line ${i + 1} is already imported with the same hash`);
      }
      // Decode and validate the exact provider payload before acquiring budget.
      const prepared = await prepareBoundedImageOcrInput(
        entry.file_path,
        entry.slug,
        realRoot,
        entry.sha256,
        sourceBytes,
      );
      entries.push(Object.freeze({
        ...entry,
        registered_root: realRoot,
        registered_root_identity: Object.freeze({ ...rootSnapshot.identity }),
        file_identity: Object.freeze({ ...fileIdentity }),
        image_format: prepared.info.format,
        image_width: prepared.info.width,
        image_height: prepared.info.height,
        image_pixels: prepared.info.pixels,
        visual_tokens: prepared.info.visualTokens,
        worst_case_usd: prepared.info.worstCaseUsd,
      }));
    } catch {
      throw new ImageOcrManifestError(i, sourceId, slug);
    }
  }

  return Object.freeze({
    manifestPath,
    manifestHash: hashBytes(bytes),
    entries: Object.freeze(entries) as unknown as ValidatedImageOcrManifestEntry[],
  });
}
