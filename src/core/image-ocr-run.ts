import { createHash } from 'node:crypto';
import {
  lstatSync,
  readFileSync,
  realpathSync,
  statSync,
} from 'node:fs';
import { isAbsolute, relative, resolve, sep } from 'node:path';
import type { BrainEngine } from './engine.ts';
import { gbrainPath } from './config.ts';
import {
  importImageFileWithBoundedOcrText,
  isImageFilePath,
  prepareBoundedImageOcrInput,
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
    imageImportFenceToken,
  );
  if (result.status !== 'imported') throw new Error('Bounded image import did not complete');
}

function hashBytes(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex');
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
      const visibleRel = canonicalRelativePath(visibleRoot, entry.file_path);
      const realFile = realpathSync(entry.file_path);
      const realRoot = realpathSync(source.local_path);
      const realRel = canonicalRelativePath(realRoot, realFile);
      if (visibleRel !== realRel) throw new Error(`Manifest line ${i + 1} source/file path normalization is ambiguous`);
      const canonicalSlug = visibleRel.toLowerCase();
      if (entry.slug !== canonicalSlug || entry.slug.includes('\\')) {
        throw new Error(`Manifest line ${i + 1} slug does not match canonical source-relative path`);
      }
      if (!isImageFilePath(entry.file_path) || !isImageFilePath(entry.slug)) {
        throw new Error(`Manifest line ${i + 1} is not a supported image type`);
      }
      const actualHash = hashBytes(readFileSync(entry.file_path));
      if (actualHash !== entry.sha256) throw new Error(`Manifest line ${i + 1} sha256 mismatch`);
      const existing = await engine.getPage(entry.slug, { sourceId: entry.source_id });
      if (existing?.content_hash === entry.sha256) {
        throw new Error(`Manifest line ${i + 1} is already imported with the same hash`);
      }
      // Decode and validate the exact provider payload before acquiring budget.
      await prepareBoundedImageOcrInput(
        entry.file_path,
        entry.slug,
        visibleRoot,
        entry.sha256,
      );
      entries.push(Object.freeze({ ...entry, registered_root: visibleRoot }));
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
