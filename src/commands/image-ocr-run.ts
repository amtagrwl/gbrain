import type { BrainEngine } from '../core/engine.ts';
import { prepareBoundedImageOcrInput } from '../core/import-file.ts';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  validateImageOcrCaps,
  type ImageOcrCaps,
  OcrBudgetCapError,
  OcrBudgetLockError,
  OcrBudgetLedger,
  type OcrReservation,
} from '../core/image-ocr-budget.ts';
import {
  defaultImageOcrBudgetDirectory,
  ImageOcrManifestError,
  importValidatedImageOcrEntry,
  parseAndValidateImageOcrManifest,
  type ValidatedImageOcrManifestEntry,
  type ImageOcrRunReport,
  type ValidatedImageOcrManifest,
} from '../core/image-ocr-run.ts';
import {
  buildBoundedImageOcrRequestBody,
  loadBoundedImageOcrProviderConfig,
  requestBoundedImageOcr,
  type BoundedImageOcrProviderConfig,
} from '../core/image-ocr-provider.ts';
import {
  withImageImportFence,
  type ImageImportFenceToken,
} from '../core/image-import-fence.ts';

const HELP = `Usage: gbrain image-ocr-run <manifest.jsonl> \\
  --max-images N --max-usd USD --reserve-usd-per-call USD --yes

Runs paid image OCR only for the exact, prevalidated JSONL entries. All caps
are mandatory and shared with prior runs on the same UTC date. --yes is an
explicit non-interactive confirmation; no env-only override exists.
`;

export interface ImageOcrRunArgs extends ImageOcrCaps {
  manifestPath: string;
  yes: true;
}

function flagValue(args: string[], name: string): string | undefined {
  const inline = args.find(arg => arg.startsWith(`${name}=`));
  if (inline) return inline.slice(name.length + 1);
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
}

function parseRequiredNumber(args: string[], name: string): number {
  const raw = flagValue(args, name);
  if (raw === undefined || raw.startsWith('--') || raw.trim() === '') {
    throw new Error(`${name} is required`);
  }
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`${name} must be finite`);
  return value;
}

function manifestHashBestEffort(path: string | undefined): string | null {
  if (!path || path.startsWith('--')) return null;
  try { return createHash('sha256').update(readFileSync(path)).digest('hex'); }
  catch { return null; }
}

export function parseImageOcrRunArgs(args: string[]): ImageOcrRunArgs {
  const allowed = new Set(['--max-images', '--max-usd', '--reserve-usd-per-call', '--yes']);
  for (const arg of args) {
    if (arg.startsWith('--') && !allowed.has(arg.split('=', 1)[0])) {
      throw new Error(`Unknown image-ocr-run flag: ${arg}`);
    }
  }
  if (!args.includes('--yes')) throw new Error('--yes explicit confirmation is required');
  const valueIndexes = new Set<number>();
  for (const name of ['--max-images', '--max-usd', '--reserve-usd-per-call']) {
    const index = args.indexOf(name);
    if (index >= 0) valueIndexes.add(index + 1);
  }
  const positionals = args.filter((arg, index) => !arg.startsWith('--') && !valueIndexes.has(index));
  if (positionals.length !== 1) throw new Error('Exactly one manifest JSONL path is required');
  const caps = validateImageOcrCaps({
    maxImages: parseRequiredNumber(args, '--max-images'),
    maxUsd: parseRequiredNumber(args, '--max-usd'),
    reserveUsdPerCall: parseRequiredNumber(args, '--reserve-usd-per-call'),
  });
  return { manifestPath: positionals[0], yes: true, ...caps };
}

const IMAGE_OCR_RAW_VALUE_FLAGS = new Set([
  '--brain', '--max-images', '--max-usd', '--reserve-usd-per-call',
]);
const IMAGE_OCR_RAW_BOOLEAN_FLAGS = new Set(['--help', '--yes']);

/** Strict full-argv gate run before cli.ts strips global flags. */
export function validateImageOcrRunRawArgv(rawArgv: string[]): string | null {
  let sawCommand = false;
  let help = false;
  const positionals: string[] = [];

  for (let index = 0; index < rawArgv.length; index++) {
    const arg = rawArgv[index];
    if (arg === 'image-ocr-run' && !sawCommand) {
      sawCommand = true;
      continue;
    }
    if (arg.startsWith('--')) {
      const equals = arg.indexOf('=');
      const name = equals >= 0 ? arg.slice(0, equals) : arg;
      if (IMAGE_OCR_RAW_BOOLEAN_FLAGS.has(name)) {
        if (equals >= 0) return name;
        if (name === '--help') help = true;
        continue;
      }
      if (!IMAGE_OCR_RAW_VALUE_FLAGS.has(name)) return name;
      if (equals >= 0) {
        const value = arg.slice(equals + 1);
        if (value.length === 0) return name;
        if (name === '--brain' && !/^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$/.test(value)) return name;
        continue;
      }
      const value = rawArgv[index + 1];
      if (value === undefined || value.startsWith('--') || value === 'image-ocr-run') return name;
      if (name === '--brain' && !/^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$/.test(value)) return name;
      index++;
      continue;
    }
    if (arg.startsWith('-')) return arg;
    positionals.push(arg);
  }

  if (!sawCommand) return 'image-ocr-run';
  if (help) return positionals.length <= 1 ? null : positionals[1];
  return positionals.length === 1 ? null : (positionals[1] ?? 'manifest');
}

function bestEffortManifestPath(args: string[]): string | undefined {
  const valueIndexes = new Set<number>();
  const valueFlags = new Set([
    '--brain', '--max-images', '--max-usd', '--reserve-usd-per-call',
    '--timeout', '--progress-interval', '--source',
  ]);
  for (let index = 0; index < args.length; index++) {
    if (valueFlags.has(args[index])) valueIndexes.add(index + 1);
  }
  return args.find((arg, index) => (
    arg !== 'image-ocr-run'
    && !arg.startsWith('-')
    && !valueIndexes.has(index)
  ));
}

function bestEffortCap(args: string[], name: string): number | null {
  const raw = flagValue(args, name);
  if (raw === undefined || raw.trim() === '') return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

/** One response-body-free report for failures before runImageOcrRun owns stdout. */
export function emitImageOcrRunTerminalFailure(
  args: string[],
  terminalError: 'arguments_invalid' | 'run_rejected',
  now = new Date(),
): ImageOcrRunReport {
  const manifestPath = bestEffortManifestPath(args);
  const report: ImageOcrRunReport = {
    utc_date: now.toISOString().slice(0, 10),
    manifest_hash: manifestHashBestEffort(manifestPath),
    requested: 0,
    processed: 0,
    succeeded: 0,
    failed: 0,
    skipped: 0,
    daily_calls_before: null,
    daily_calls_after: null,
    daily_usd_reserved_before: null,
    daily_usd_reserved_after: null,
    cap: {
      max_images: bestEffortCap(args, '--max-images'),
      max_usd: bestEffortCap(args, '--max-usd'),
      reserve_usd_per_call: bestEffortCap(args, '--reserve-usd-per-call'),
    },
    first_failing_entry: null,
    status: 'rejected',
    terminal_error: terminalError,
  };
  process.stdout.write(`${JSON.stringify(report)}\n`);
  return report;
}

export interface ImageOcrCommandOptions {
  ledgerDirectory?: string;
  now?: Date;
  /** Fresh wall-clock sample taken immediately before each paid-attempt reservation. */
  attemptClock?: () => Date;
  afterReserve?: (entry: ValidatedImageOcrManifestEntry, reservation: OcrReservation) => void | Promise<void>;
  importEntry?: (
    entry: ValidatedImageOcrManifestEntry,
    beforeProviderAttempt: () => Promise<OcrReservation>,
    imageImportFenceToken: ImageImportFenceToken,
  ) => void | Promise<void>;
  ocrProvider?: (imgBuf: Buffer, mime: string) => Promise<string>;
  imageImportFenceRoot?: string;
}

async function callConfirmedImageOcrProvider(
  engine: BrainEngine,
  entry: ValidatedImageOcrManifestEntry,
  providerConfig: BoundedImageOcrProviderConfig | null,
  beforeProviderAttempt: () => Promise<OcrReservation>,
  provider?: ImageOcrCommandOptions['ocrProvider'],
  attempt?: { attempted: boolean; succeeded: boolean },
): Promise<string> {
  const input = await prepareBoundedImageOcrInput(
    entry.file_path,
    entry.slug,
    entry.registered_root,
    entry.sha256,
  );
  const currentSource = (await engine.listAllSources({ includeArchived: false }))
    .find(source => source.id === entry.source_id);
  if (!currentSource?.local_path || resolve(currentSource.local_path) !== entry.registered_root) {
    throw new Error('Registered source root changed after manifest validation');
  }
  const existing = await engine.getPage(entry.slug, { sourceId: entry.source_id });
  if (existing?.content_hash === entry.sha256) {
    throw new Error('Source-scoped image already exists with the manifest hash');
  }
  await beforeProviderAttempt();
  if (attempt) attempt.attempted = true;
  try {
    let text: string;
    if (provider) {
      text = await provider(input.buf, input.mime);
    } else {
      text = await requestBoundedImageOcr({
        apiKey: providerConfig!.apiKey,
        providerBaseUrls: providerConfig!.providerBaseUrls,
        body: buildBoundedImageOcrRequestBody(input.buf, input.mime),
      });
    }
    if (text.trim().length === 0) throw new Error('empty OCR response');
    if (attempt) attempt.succeeded = true;
    return text;
  } catch {
    throw new Error('Image OCR provider call failed');
  }
}

async function bumpOcrCounter(engine: BrainEngine, key: string): Promise<void> {
  try {
    const current = parseInt((await engine.getConfig(key)) ?? '0', 10);
    await engine.setConfig(key, String((Number.isFinite(current) ? current : 0) + 1));
  } catch { /* best-effort; the durable reservation remains authoritative */ }
}

/** Private executor: reachable only after runImageOcrRun has parsed --yes. */
async function executeConfirmedImageOcrRun(input: {
  engine: BrainEngine;
  manifest: ValidatedImageOcrManifest;
  caps: ImageOcrCaps;
  ledgerDirectory: string;
  attemptClock: () => Date;
  fallbackNow: Date;
  afterReserve?: ImageOcrCommandOptions['afterReserve'];
  importEntry: NonNullable<ImageOcrCommandOptions['importEntry']>;
  imageImportFenceRoot?: string;
}): Promise<ImageOcrRunReport> {
  const caps = validateImageOcrCaps(input.caps);
  let processed = 0;
  let succeeded = 0;
  let failed = 0;
  let skipped = 0;
  let firstFailure: ImageOcrRunReport['first_failing_entry'] = null;
  let status: ImageOcrRunReport['status'] = 'completed';
  let terminalError: ImageOcrRunReport['terminal_error'];
  let before: ReturnType<OcrBudgetLedger['snapshot']> | null = null;
  let after: ReturnType<OcrBudgetLedger['snapshot']> | null = null;

  for (let index = 0; index < input.manifest.entries.length; index++) {
    const entry = input.manifest.entries[index];
    let attemptNow: Date | null = null;
    let ledger: OcrBudgetLedger | null = null;
    let callbackInvocations = 0;
    let reservationMade = false;
    let postReservationFailure = false;
    let callbackError: unknown;
    let runError: unknown;
    let closeFailed = false;
    let stop = false;

    try {
      await withImageImportFence(async (fenceToken) => {
        let fenceHeld = true;
        const beforeProviderAttempt = async (): Promise<OcrReservation> => {
          callbackInvocations++;
          if (!fenceHeld) {
            throw new Error('beforeProviderAttempt must run while the image-import fence is held');
          }
          if (callbackInvocations !== 1) {
            throw new Error('beforeProviderAttempt must be invoked exactly once');
          }

          try {
            // The clock and date-ledger reservation intentionally happen only
            // after the adapter's final source/page/path/hash revalidation.
            attemptNow = input.attemptClock();
            ledger = OcrBudgetLedger.acquire({
              directory: input.ledgerDirectory,
              now: attemptNow,
              caps,
              manifestHash: input.manifest.manifestHash,
            });
            const attemptBefore = ledger.snapshot();
            if (before?.utcDate !== attemptBefore.utcDate) before = attemptBefore;
            after = attemptBefore;

            const exceeded = ledger.nextCapExceeded();
            if (exceeded) throw new OcrBudgetCapError(exceeded);
            const reservation = ledger.reserve({
              index,
              sourceId: entry.source_id,
              slug: entry.slug,
              filePath: entry.file_path,
              registeredRoot: entry.registered_root,
              sha256: entry.sha256,
            });
            reservationMade = true;
            processed++;
            after = ledger.snapshot();
            try {
              await input.afterReserve?.(entry, reservation);
            } catch (error) {
              postReservationFailure = true;
              throw error;
            }
            return reservation;
          } catch (error) {
            callbackError = error;
            throw error;
          }
        };

        try {
          await input.importEntry(entry, beforeProviderAttempt, fenceToken);
          if (callbackInvocations !== 1) {
            throw new Error('Provider adapter did not invoke beforeProviderAttempt exactly once');
          }
          if (callbackError) throw callbackError;
          if (!reservationMade) throw new Error('Provider adapter reached no durable OCR reservation');
        } finally {
          fenceHeld = false;
        }
      }, { lockRoot: input.imageImportFenceRoot });
      succeeded++;
    } catch (error) {
      runError = error;
    } finally {
      const heldLedger = ledger as OcrBudgetLedger | null;
      if (heldLedger) {
        after = heldLedger.snapshot();
        try {
          heldLedger.close();
        } catch {
          closeFailed = true;
        }
      }
    }

    if (runError) {
      // A clock/ledger acquisition failure happened inside the held fence but
      // before any provider request. Preserve the historical fail-closed report.
      if (callbackInvocations === 1 && !ledger && !reservationMade) {
        const sampledNow = attemptNow as Date | null;
        const reportNow = sampledNow && Number.isFinite(sampledNow.getTime())
          ? sampledNow
          : input.fallbackNow;
        const locked = runError instanceof OcrBudgetLockError;
        return {
          utc_date: reportNow.toISOString().slice(0, 10),
          manifest_hash: input.manifest.manifestHash,
          requested: input.manifest.entries.length,
          processed,
          succeeded,
          failed,
          skipped: input.manifest.entries.length - index,
          daily_calls_before: null,
          daily_calls_after: null,
          daily_usd_reserved_before: null,
          daily_usd_reserved_after: null,
          cap: {
            max_images: caps.maxImages,
            max_usd: caps.maxUsd,
            reserve_usd_per_call: caps.reserveUsdPerCall,
          },
          first_failing_entry: firstFailure,
          status: locked ? 'locked' : 'rejected',
          terminal_error: locked ? 'budget_locked' : 'run_rejected',
        };
      }

      if (runError instanceof OcrBudgetCapError) {
        skipped = input.manifest.entries.length - index;
        firstFailure = {
          index,
          source_id: entry.source_id,
          slug: entry.slug,
          code: runError.cap === 'images' ? 'image_cap' : 'usd_cap',
        };
        status = 'cap_reached';
      } else if (postReservationFailure) {
        failed++;
        skipped = input.manifest.entries.length - index - 1;
        firstFailure = {
          index,
          source_id: entry.source_id,
          slug: entry.slug,
          code: 'post_reservation_failure',
        };
        status = 'failed';
      } else if (callbackInvocations === 1 && ledger && !reservationMade) {
        status = 'failed';
        terminalError = 'run_rejected';
        skipped = input.manifest.entries.length - index;
      } else {
        failed++;
        skipped = input.manifest.entries.length - index - 1;
        firstFailure = {
          index,
          source_id: entry.source_id,
          slug: entry.slug,
          code: 'import_failed',
        };
        status = 'failed';
      }
      stop = true;
    }

    if (closeFailed) {
      // The durable reservation and completed entry counters are still real.
      // Report them rather than letting close() replace the run with zeroes.
      status = 'failed';
      terminalError = 'run_rejected';
      skipped = input.manifest.entries.length - index - 1;
      stop = true;
    }
    if (stop) break;
  }

  const finalBefore = before as ReturnType<OcrBudgetLedger['snapshot']> | null;
  const finalAfter = after as ReturnType<OcrBudgetLedger['snapshot']> | null;
  const reportCaps = finalAfter?.caps ?? caps;
  return {
    utc_date: finalAfter?.utcDate ?? input.fallbackNow.toISOString().slice(0, 10),
    manifest_hash: input.manifest.manifestHash,
    requested: input.manifest.entries.length,
    processed,
    succeeded,
    failed,
    skipped,
    daily_calls_before: finalBefore?.callsReserved ?? null,
    daily_calls_after: finalAfter?.callsReserved ?? null,
    daily_usd_reserved_before: finalBefore?.usdReserved ?? null,
    daily_usd_reserved_after: finalAfter?.usdReserved ?? null,
    cap: {
      max_images: reportCaps.maxImages,
      max_usd: reportCaps.maxUsd,
      reserve_usd_per_call: reportCaps.reserveUsdPerCall,
    },
    first_failing_entry: firstFailure,
    status,
    ...(terminalError ? { terminal_error: terminalError } : {}),
  };
}

export async function runImageOcrRun(
  engine: BrainEngine,
  args: string[],
  options: ImageOcrCommandOptions = {},
): Promise<ImageOcrRunReport | null> {
  if (args.includes('--help') || args.includes('-h')) {
    process.stdout.write(HELP);
    return null;
  }
  const now = options.now ?? new Date();
  const attemptClock = options.attemptClock
    ?? (options.now ? () => new Date(now.getTime()) : () => new Date());
  let stage: 'arguments' | 'manifest' | 'run' = 'arguments';
  let parsed: ImageOcrRunArgs | null = null;
  let manifest: Awaited<ReturnType<typeof parseAndValidateImageOcrManifest>> | null = null;
  let report: ImageOcrRunReport;
  try {
    parsed = parseImageOcrRunArgs(args);
    stage = 'manifest';
    // The complete manifest preflight happens before the budget lock and before
    // any reservation/provider access. This command never enumerates sources or
    // files beyond the entries named by that manifest.
    manifest = await parseAndValidateImageOcrManifest(engine, parsed.manifestPath);
    stage = 'run';
    // Resolve credentials and reject every mutable Anthropic base URL before
    // the first budget lock/reservation. Injected offline providers skip this
    // production-only configuration boundary.
    const providerConfig = options.ocrProvider || options.importEntry
      ? null
      : await loadBoundedImageOcrProviderConfig(engine);
    report = await executeConfirmedImageOcrRun({
      engine,
      manifest,
      caps: parsed,
      ledgerDirectory: options.ledgerDirectory ?? defaultImageOcrBudgetDirectory(),
      attemptClock,
      fallbackNow: now,
      afterReserve: options.afterReserve,
      imageImportFenceRoot: options.imageImportFenceRoot,
      importEntry: options.importEntry
        ?? (async (entry, beforeProviderAttempt, fenceToken) => {
          const attempt = { attempted: false, succeeded: false };
          try {
            const ocrText = await callConfirmedImageOcrProvider(
              engine,
              entry,
              providerConfig,
              beforeProviderAttempt,
              options.ocrProvider,
              attempt,
            );
            await importValidatedImageOcrEntry(engine, entry, ocrText, fenceToken);
          } finally {
            if (attempt.attempted) {
              await bumpOcrCounter(engine, 'ocr_attempted');
              await bumpOcrCounter(engine, attempt.succeeded ? 'ocr_succeeded' : 'ocr_failed_other');
            }
          }
        }),
    });
  } catch (error) {
    const locked = error instanceof OcrBudgetLockError;
    report = {
      utc_date: now.toISOString().slice(0, 10),
      manifest_hash: manifest?.manifestHash ?? manifestHashBestEffort(parsed?.manifestPath ?? args[0]),
      requested: manifest?.entries.length ?? 0,
      processed: 0,
      succeeded: 0,
      failed: 0,
      skipped: manifest?.entries.length ?? 0,
      daily_calls_before: null,
      daily_calls_after: null,
      daily_usd_reserved_before: null,
      daily_usd_reserved_after: null,
      cap: {
        max_images: parsed?.maxImages ?? null,
        max_usd: parsed?.maxUsd ?? null,
        reserve_usd_per_call: parsed?.reserveUsdPerCall ?? null,
      },
      first_failing_entry: error instanceof ImageOcrManifestError ? {
        index: error.index,
        source_id: error.sourceId,
        slug: error.slug,
        code: 'manifest_invalid',
      } : null,
      status: locked ? 'locked' : 'rejected',
      terminal_error: locked
        ? 'budget_locked'
        : stage === 'arguments'
          ? 'arguments_invalid'
          : stage === 'manifest'
            ? 'manifest_invalid'
            : 'run_rejected',
    };
  }
  process.stdout.write(`${JSON.stringify(report)}\n`);
  return report;
}
