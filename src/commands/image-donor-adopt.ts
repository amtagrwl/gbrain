import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import type { BrainEngine } from '../core/engine.ts';
import {
  ExactHashDonorFileConflictError,
  ExactHashDonorPostconditionError,
  ExactHashDonorSchemaVersionError,
  ExactHashDonorTargetConflictError,
  ExactHashDonorUnavailableError,
  type ExactHashDonorAdoptionReceipt,
  type ExactHashDonorImportOptions,
} from '../core/image-donor-adopt.ts';
import {
  adoptValidatedImageDonorEntry,
  ImageDonorPhysicalStateError,
  ImageOcrManifestError,
  parseAndValidateImageDonorManifest,
  type ValidatedImageOcrManifestEntry,
} from '../core/image-ocr-run.ts';
import { withImageImportFence } from '../core/image-import-fence.ts';

export const IMAGE_DONOR_ADOPT_HELP = `Usage: gbrain image-donor-adopt <manifest.jsonl> --max-images N --yes

Creates absent source-local image pages from a qualifying exact-hash donor.
This command is storage-only: it performs no OCR, embedding, gateway, or
spend-ledger operation. --max-images is mandatory (1..1000), and --yes is an
explicit non-interactive confirmation.
`;

export interface ImageDonorAdoptArgs {
  manifestPath: string;
  maxImages: number;
  yes: true;
}

export type ImageDonorAdoptFailureCode =
  | 'arguments_invalid'
  | 'manifest_invalid'
  | 'donor_unavailable'
  | 'target_conflict'
  | 'file_conflict'
  | 'physical_state_changed'
  | 'schema_incompatible'
  | 'postcondition_failed'
  | 'run_rejected';

export interface ImageDonorAdoptResultRow {
  index: number;
  source_id: string;
  slug: string;
  status: 'adopted' | 'idempotent' | 'failed';
  donor_page_id: number | null;
  donor_chunk_id: number | null;
  donor_state_sha256: string | null;
  target_page_id: number | null;
  target_chunk_id: number | null;
  file_id: number | null;
  file_disposition: ExactHashDonorAdoptionReceipt['file_disposition'] | null;
  receipt: ExactHashDonorAdoptionReceipt | null;
  error_code: ImageDonorAdoptFailureCode | null;
}

export interface ImageDonorAdoptReport {
  manifest_hash: string | null;
  requested: number;
  processed: number;
  succeeded: number;
  skipped: number;
  failed: number;
  provider_attempts: 0;
  model_calls: 0;
  gateway_calls: 0;
  ocr_budget_reservations: 0;
  ocr_budget_usd: 0;
  cap: { max_images: number | null };
  results: ImageDonorAdoptResultRow[];
  first_failure: {
    index: number;
    source_id: string | null;
    slug: string | null;
    code: ImageDonorAdoptFailureCode;
  } | null;
  status: 'completed' | 'cap_reached' | 'failed' | 'rejected';
  terminal_error?: 'arguments_invalid' | 'manifest_invalid' | 'run_rejected';
}

function sha256(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function strictPositiveCap(raw: string): number {
  if (!/^(?:[1-9]|[1-9][0-9]{1,2}|1000)$/.test(raw)) {
    throw new Error('--max-images must be an integer from 1 through 1000');
  }
  return Number(raw);
}

export function parseImageDonorAdoptArgs(args: string[]): ImageDonorAdoptArgs {
  let manifestPath: string | null = null;
  let maxImages: number | null = null;
  let yes = false;

  for (let index = 0; index < args.length; index++) {
    const arg = args[index];
    if (arg === '--yes') {
      if (yes) throw new Error('--yes may appear only once');
      yes = true;
      continue;
    }
    if (arg === '--max-images' || arg.startsWith('--max-images=')) {
      if (maxImages !== null) throw new Error('--max-images may appear only once');
      const inline = arg.startsWith('--max-images=');
      const raw = inline ? arg.slice('--max-images='.length) : args[++index];
      if (!raw || raw.startsWith('-')) throw new Error('--max-images requires a value');
      maxImages = strictPositiveCap(raw);
      continue;
    }
    if (arg.startsWith('-')) throw new Error(`Unknown image-donor-adopt flag: ${arg}`);
    if (manifestPath !== null) throw new Error('Exactly one manifest JSONL path is required');
    manifestPath = arg;
  }

  if (!manifestPath) throw new Error('Exactly one manifest JSONL path is required');
  if (maxImages === null) throw new Error('--max-images is required');
  if (!yes) throw new Error('--yes explicit confirmation is required');
  return { manifestPath, maxImages, yes: true };
}

const RAW_VALUE_FLAGS = new Set(['--brain', '--max-images']);
const RAW_BOOLEAN_FLAGS = new Set(['--help', '--yes']);
const BRAIN_ID = /^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$/;

/** Strict full-argv gate, evaluated before cli.ts removes global flags. */
export function validateImageDonorAdoptRawArgv(rawArgv: string[]): string | null {
  let sawCommand = false;
  let sawHelp = false;
  const seenFlags = new Set<string>();
  const positionals: string[] = [];

  for (let index = 0; index < rawArgv.length; index++) {
    const arg = rawArgv[index];
    if (arg === 'image-donor-adopt' && !sawCommand) {
      sawCommand = true;
      continue;
    }
    if (arg.startsWith('--')) {
      const equals = arg.indexOf('=');
      const name = equals < 0 ? arg : arg.slice(0, equals);
      if (seenFlags.has(name)) return name;
      if (RAW_BOOLEAN_FLAGS.has(name)) {
        if (equals >= 0) return name;
        seenFlags.add(name);
        if (name === '--help') sawHelp = true;
        continue;
      }
      if (!RAW_VALUE_FLAGS.has(name)) return name;
      seenFlags.add(name);
      const value = equals >= 0 ? arg.slice(equals + 1) : rawArgv[++index];
      if (!value || value.startsWith('-') || value === 'image-donor-adopt') return name;
      if (name === '--brain' && !BRAIN_ID.test(value)) return name;
      continue;
    }
    if (arg.startsWith('-')) return arg;
    positionals.push(arg);
  }

  if (!sawCommand) return 'image-donor-adopt';
  if (sawHelp) return positionals.length <= 1 ? null : positionals[1];
  return positionals.length === 1 ? null : (positionals[1] ?? 'manifest');
}

function bestEffortManifestPath(args: string[]): string | undefined {
  const valueIndexes = new Set<number>();
  const valueFlags = new Set([
    '--brain', '--max-images', '--timeout', '--progress-interval', '--source',
    '--max-usd', '--reserve-usd-per-call',
  ]);
  for (let index = 0; index < args.length; index++) {
    const name = args[index];
    if (valueFlags.has(name)) valueIndexes.add(index + 1);
  }
  return args.find((arg, index) => (
    arg !== 'image-donor-adopt'
    && !arg.startsWith('-')
    && !valueIndexes.has(index)
  ));
}

function bestEffortManifestHash(path: string | undefined): string | null {
  if (!path) return null;
  try { return sha256(readFileSync(path)); }
  catch { return null; }
}

function bestEffortMaxImages(args: string[]): number | null {
  const inline = args.find(arg => arg.startsWith('--max-images='));
  const index = args.indexOf('--max-images');
  const raw = inline?.slice('--max-images='.length) ?? (index >= 0 ? args[index + 1] : undefined);
  if (!raw || !/^[0-9]+$/.test(raw)) return null;
  const value = Number(raw);
  return Number.isSafeInteger(value) ? value : null;
}

function emptyReport(input: {
  manifestHash: string | null;
  maxImages: number | null;
  terminalError: 'arguments_invalid' | 'manifest_invalid' | 'run_rejected';
  firstFailure?: ImageDonorAdoptReport['first_failure'];
}): ImageDonorAdoptReport {
  return {
    manifest_hash: input.manifestHash,
    requested: 0,
    processed: 0,
    succeeded: 0,
    skipped: 0,
    failed: 0,
    provider_attempts: 0,
    model_calls: 0,
    gateway_calls: 0,
    ocr_budget_reservations: 0,
    ocr_budget_usd: 0,
    cap: { max_images: input.maxImages },
    results: [],
    first_failure: input.firstFailure ?? null,
    status: 'rejected',
    terminal_error: input.terminalError,
  };
}

/** One report for failures that happen before runImageDonorAdopt owns stdout. */
export function emitImageDonorAdoptTerminalFailure(
  args: string[],
  terminalError: 'arguments_invalid' | 'run_rejected',
): ImageDonorAdoptReport {
  const report = emptyReport({
    manifestHash: bestEffortManifestHash(bestEffortManifestPath(args)),
    maxImages: bestEffortMaxImages(args),
    terminalError,
  });
  process.stdout.write(`${JSON.stringify(report)}\n`);
  return report;
}

function classifyFailure(error: unknown): ImageDonorAdoptFailureCode {
  if (error instanceof ExactHashDonorUnavailableError) return 'donor_unavailable';
  if (error instanceof ExactHashDonorTargetConflictError) return 'target_conflict';
  if (error instanceof ExactHashDonorFileConflictError) return 'file_conflict';
  if (error instanceof ExactHashDonorSchemaVersionError) return 'schema_incompatible';
  if (error instanceof ExactHashDonorPostconditionError) return 'postcondition_failed';
  if (error instanceof ImageDonorPhysicalStateError) return 'physical_state_changed';
  return 'postcondition_failed';
}

function receiptFromResult(
  result: Awaited<ReturnType<typeof adoptValidatedImageDonorEntry>>,
): ExactHashDonorAdoptionReceipt {
  if (result.status === 'adopted') return result.receipt;
  const { status: _status, ...receipt } = result;
  return receipt;
}

export interface ImageDonorAdoptCommandOptions {
  imageImportFenceRoot?: string;
  /** Test-only seam that runs after preflight and before the row attempt. */
  beforeEntry?: (entry: ValidatedImageOcrManifestEntry, index: number) => void | Promise<void>;
  /** Test-only transaction hooks. */
  importOptions?: ExactHashDonorImportOptions;
  /** Accepted for caller compatibility; report ordering/schema contain no wall clock. */
  now?: Date;
}

export async function runImageDonorAdopt(
  engine: BrainEngine,
  args: string[],
  options: ImageDonorAdoptCommandOptions = {},
): Promise<ImageDonorAdoptReport | null> {
  if (args.includes('--help')) {
    process.stdout.write(IMAGE_DONOR_ADOPT_HELP);
    return null;
  }
  void options.now;

  let parsed: ImageDonorAdoptArgs;
  try {
    parsed = parseImageDonorAdoptArgs(args);
  } catch {
    const report = emptyReport({
      manifestHash: bestEffortManifestHash(bestEffortManifestPath(args)),
      maxImages: bestEffortMaxImages(args),
      terminalError: 'arguments_invalid',
    });
    process.stdout.write(`${JSON.stringify(report)}\n`);
    return report;
  }

  let manifest: Awaited<ReturnType<typeof parseAndValidateImageDonorManifest>>;
  try {
    manifest = await parseAndValidateImageDonorManifest(engine, parsed.manifestPath);
  } catch (error) {
    const manifestError = error instanceof ImageOcrManifestError ? error : null;
    const report = emptyReport({
      manifestHash: bestEffortManifestHash(parsed.manifestPath),
      maxImages: parsed.maxImages,
      terminalError: 'manifest_invalid',
      firstFailure: {
        index: manifestError?.index ?? 0,
        source_id: manifestError?.sourceId ?? null,
        slug: manifestError?.slug ?? null,
        code: 'manifest_invalid',
      },
    });
    process.stdout.write(`${JSON.stringify(report)}\n`);
    return report;
  }

  const results: ImageDonorAdoptResultRow[] = [];
  let succeeded = 0;
  let idempotent = 0;
  let failed = 0;
  let firstFailure: ImageDonorAdoptReport['first_failure'] = null;
  const attemptCount = Math.min(manifest.entries.length, parsed.maxImages);

  for (let index = 0; index < attemptCount; index++) {
    const entry = manifest.entries[index];
    try {
      await options.beforeEntry?.(entry, index);
      const result = await withImageImportFence(
        token => adoptValidatedImageDonorEntry(engine, entry, token, {
          ...options.importOptions,
          manifestHash: manifest.manifestHash,
          manifestIndex: index,
        }),
        { lockRoot: options.imageImportFenceRoot },
      );
      const receipt = receiptFromResult(result);
      if (result.status === 'adopted') succeeded++;
      else idempotent++;
      results.push({
        index,
        source_id: entry.source_id,
        slug: entry.slug,
        status: result.status,
        donor_page_id: receipt.donor_page_id,
        donor_chunk_id: receipt.donor_chunk_id,
        donor_state_sha256: receipt.donor_state_sha256,
        target_page_id: receipt.target_page_id,
        target_chunk_id: receipt.target_chunk_id,
        file_id: receipt.file_id,
        file_disposition: receipt.file_disposition,
        receipt,
        error_code: null,
      });
    } catch (error) {
      failed++;
      const code = classifyFailure(error);
      firstFailure ??= { index, source_id: entry.source_id, slug: entry.slug, code };
      results.push({
        index,
        source_id: entry.source_id,
        slug: entry.slug,
        status: 'failed',
        donor_page_id: null,
        donor_chunk_id: null,
        donor_state_sha256: null,
        target_page_id: null,
        target_chunk_id: null,
        file_id: null,
        file_disposition: null,
        receipt: null,
        error_code: code,
      });
    }
  }

  const capSkipped = manifest.entries.length - attemptCount;
  const report: ImageDonorAdoptReport = {
    manifest_hash: manifest.manifestHash,
    requested: manifest.entries.length,
    processed: attemptCount,
    succeeded,
    skipped: idempotent + capSkipped,
    failed,
    provider_attempts: 0,
    model_calls: 0,
    gateway_calls: 0,
    ocr_budget_reservations: 0,
    ocr_budget_usd: 0,
    cap: { max_images: parsed.maxImages },
    results,
    first_failure: firstFailure,
    status: failed > 0 ? 'failed' : capSkipped > 0 ? 'cap_reached' : 'completed',
  };
  process.stdout.write(`${JSON.stringify(report)}\n`);
  return report;
}
