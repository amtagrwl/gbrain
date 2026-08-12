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
import { extname, isAbsolute, relative, resolve, sep } from 'node:path';
import type { BrainEngine } from './engine.ts';
import { gbrainPath } from './config.ts';
import {
  importImageFileWithExactHashDonor,
  type ExactHashDonorAdoptionResult,
  type ExactHashDonorImportOptions,
} from './image-donor-adopt.ts';
import type { BoundedImageOcrExpectedPageState } from './import-file.ts';
import {
  BOUNDED_IMAGE_EXTENSIONS,
  BOUNDED_IMAGE_MAX_BYTES,
  prepareBoundedImageDecode,
} from './image-decode.ts';
import type { ImageImportFenceToken } from './image-import-fence.ts';

export interface ImageOcrManifestEntry {
  source_id: string;
  slug: string;
  file_path: string;
  sha256: string;
}

export interface ValidatedImageOcrManifestEntry extends ImageOcrManifestEntry {
  /** Canonical, case-preserving path relative to the registered source root. */
  source_relative_path: string;
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
  size: number;
  mtime_ms: number;
  ctime_ms: number;
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

export class ImageDonorPhysicalStateError extends Error {
  readonly code = 'physical_state_changed';
  constructor() {
    super('Image donor physical source state changed after manifest validation');
    this.name = 'ImageDonorPhysicalStateError';
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
  const { importImageFileWithBoundedOcrText } = await import('./import-file.ts');
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

/**
 * Donor adoption shares paid OCR's path/hash/codec preflight, but deliberately
 * stops before constructing a provider request body. It also permits an
 * existing target through preflight so the transaction can distinguish an
 * exact prior adoption from an unrelated or drifted occupant under row lock.
 */
export async function parseAndValidateImageDonorManifest(
  engine: BrainEngine,
  manifestPath: string,
): Promise<ValidatedImageOcrManifest> {
  return parseAndValidateImageOcrManifest(engine, manifestPath, {
    allowExistingTarget: true,
    providerPayload: false,
  });
}

function donorMimeType(filePath: string): string {
  switch (extname(filePath).toLowerCase()) {
    case '.png': return 'image/png';
    case '.jpg':
    case '.jpeg': return 'image/jpeg';
    case '.gif': return 'image/gif';
    case '.webp': return 'image/webp';
    case '.heic':
      return 'image/heic';
    case '.heif': return 'image/heif';
    case '.avif': return 'image/avif';
    default: throw new Error('Image donor adoption received an unsupported image extension');
  }
}

function assertValidatedImageDonorPhysicalState(
  entry: ValidatedImageOcrManifestEntry,
  registeredLocalPath: string,
): void {
  const currentRoot = inspectImageOcrRegisteredRoot(registeredLocalPath);
  if (
    currentRoot.canonicalPath !== entry.registered_root
    || !sameImageOcrRegisteredRootIdentity(currentRoot.identity, entry.registered_root_identity)
  ) {
    throw new ImageDonorPhysicalStateError();
  }
  const exactBytes = readImageOcrSourceFile({
    filePath: entry.file_path,
    imageSlug: entry.slug,
    registeredRoot: entry.registered_root,
    expectedHash: entry.sha256,
    expectedFileIdentity: entry.file_identity,
    requireExpectedFileSnapshot: true,
  });
  if (exactBytes.length !== entry.file_identity.size) {
    throw new ImageDonorPhysicalStateError();
  }
  const exactRelative = canonicalRelativePath(
    entry.registered_root,
    realpathSync(entry.file_path),
  );
  if (exactRelative !== entry.source_relative_path) {
    throw new ImageDonorPhysicalStateError();
  }
}

export interface AdoptValidatedImageDonorOptions extends ExactHashDonorImportOptions {
  manifestHash?: string;
  manifestIndex?: number;
}

/** Storage-only adapter. It performs no embedding, OCR, gateway, or budget call. */
export async function adoptValidatedImageDonorEntry(
  engine: BrainEngine,
  entry: ValidatedImageOcrManifestEntry,
  imageImportFenceToken: ImageImportFenceToken,
  options: AdoptValidatedImageDonorOptions = {},
): Promise<ExactHashDonorAdoptionResult> {
  let sourceBytes: Buffer;
  try {
    sourceBytes = readImageOcrSourceFile({
      filePath: entry.file_path,
      imageSlug: entry.slug,
      registeredRoot: entry.registered_root,
      expectedHash: entry.sha256,
      expectedFileIdentity: entry.file_identity,
      requireExpectedFileSnapshot: true,
    });
    // Re-run the bounded full codec/decode validation. Unlike paid OCR this
    // never serializes a request payload and never crosses a provider boundary.
    const prepared = await prepareBoundedImageDecode(sourceBytes, entry.file_path);
    if (
      prepared.info.format !== entry.image_format
      || prepared.info.width !== entry.image_width
      || prepared.info.height !== entry.image_height
      || prepared.info.pixels !== entry.image_pixels
    ) throw new ImageDonorPhysicalStateError();
  } catch (error) {
    if (error instanceof ImageDonorPhysicalStateError) throw error;
    throw new ImageDonorPhysicalStateError();
  }

  const {
    manifestHash = entry.sha256,
    manifestIndex = 0,
    ...importOptions
  } = options;
  return importImageFileWithExactHashDonor(engine, {
    sourceId: entry.source_id,
    slug: entry.slug,
    sourceRelativePath: entry.source_relative_path,
    filePath: entry.file_path,
    registeredRoot: entry.registered_root,
    sha256: entry.sha256,
    mimeType: donorMimeType(entry.file_path),
    sizeBytes: sourceBytes.length,
    manifestHash,
    manifestIndex,
    validatePhysical(localPath) {
      try {
        assertValidatedImageDonorPhysicalState(entry, localPath);
      } catch (error) {
        if (error instanceof ImageDonorPhysicalStateError) throw error;
        throw new ImageDonorPhysicalStateError();
      }
    },
  }, imageImportFenceToken, importOptions);
}

function hashBytes(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function filesystemIdentity(stat: Stats): ImageOcrFilesystemIdentity {
  return {
    device: stat.dev,
    inode: stat.ino,
    size: stat.size,
    mtime_ms: stat.mtimeMs,
    ctime_ms: stat.ctimeMs,
  };
}

function sameFilesystemIdentity(
  left: ImageOcrFilesystemIdentity,
  right: ImageOcrFilesystemIdentity,
): boolean {
  return left.device === right.device && left.inode === right.inode;
}

function sameFilesystemSnapshot(
  left: ImageOcrFilesystemIdentity,
  right: ImageOcrFilesystemIdentity,
): boolean {
  return left.device === right.device
    && left.inode === right.inode
    && left.size === right.size
    && left.mtime_ms === right.mtime_ms
    && left.ctime_ms === right.ctime_ms;
}

/** Directory identity deliberately excludes mutable directory size/timestamps. */
export function sameImageOcrRegisteredRootIdentity(
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
  /** Donor adoption binds the entire preflight snapshot; paid OCR keeps its reviewed dev/inode contract. */
  requireExpectedFileSnapshot?: boolean;
}): Buffer {
  if (lstatSync(input.filePath).isSymbolicLink()) {
    throw new Error('Bounded image OCR file may not be a symlink');
  }
  const noFollow = typeof fsConstants.O_NOFOLLOW === 'number' ? fsConstants.O_NOFOLLOW : 0;
  const fd = openSync(input.filePath, fsConstants.O_RDONLY | noFollow);
  try {
    const before = fstatSync(fd);
    const beforeIdentity = filesystemIdentity(before);
    const expectedIdentityMatches = input.requireExpectedFileSnapshot
      ? sameFilesystemSnapshot(beforeIdentity, input.expectedFileIdentity)
      : sameFilesystemIdentity(beforeIdentity, input.expectedFileIdentity);
    if (!before.isFile() || !expectedIdentityMatches) {
      throw new Error('Bounded image OCR file identity changed after manifest validation');
    }
    if (before.size > BOUNDED_IMAGE_MAX_BYTES) {
      throw new Error(`Bounded image OCR file exceeds ${BOUNDED_IMAGE_MAX_BYTES} bytes`);
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
  options: { allowExistingTarget?: boolean; providerPayload?: boolean } = {},
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
      if (
        !(BOUNDED_IMAGE_EXTENSIONS as readonly string[]).includes(extname(entry.file_path).toLowerCase())
        || !(BOUNDED_IMAGE_EXTENSIONS as readonly string[]).includes(extname(entry.slug).toLowerCase())
      ) {
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
      if (!options.allowExistingTarget && existing?.content_hash === entry.sha256) {
        throw new Error(`Manifest line ${i + 1} is already imported with the same hash`);
      }
      // Decode and validate the exact provider payload before acquiring budget.
      let prepared: Awaited<ReturnType<typeof prepareBoundedImageDecode>>;
      let visualTokens = 0;
      let worstCaseUsd = 0;
      if (options.providerPayload === false) {
        prepared = await prepareBoundedImageDecode(sourceBytes, entry.file_path);
      } else {
        const { prepareBoundedImageOcrInput } = await import('./import-file.ts');
        const paidPrepared = await prepareBoundedImageOcrInput(
          entry.file_path,
          entry.slug,
          realRoot,
          entry.sha256,
          sourceBytes,
        );
        prepared = paidPrepared as Awaited<ReturnType<typeof prepareBoundedImageDecode>>;
        visualTokens = paidPrepared.info.visualTokens;
        worstCaseUsd = paidPrepared.info.worstCaseUsd;
      }
      entries.push(Object.freeze({
        ...entry,
        source_relative_path: visibleRel,
        registered_root: realRoot,
        registered_root_identity: Object.freeze({ ...rootSnapshot.identity }),
        file_identity: Object.freeze({ ...fileIdentity }),
        image_format: prepared.info.format,
        image_width: prepared.info.width,
        image_height: prepared.info.height,
        image_pixels: prepared.info.pixels,
        visual_tokens: visualTokens,
        worst_case_usd: worstCaseUsd,
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
