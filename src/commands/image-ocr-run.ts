import type { BrainEngine } from '../core/engine.ts';
import {
  captureBoundedImageOcrExpectedPageState,
  prepareBoundedImageOcrInput,
  type BoundedImageOcrExpectedPageState,
} from '../core/import-file.ts';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import {
  IMAGE_OCR_POLICY_MODEL,
  validateImageOcrCaps,
  type ImageOcrCaps,
  OcrBudgetCapError,
  OcrBudgetLockError,
  OcrBudgetLedger,
  type OcrFailureOutcome,
  type OcrFailureStage,
  type OcrInvalidProviderObservationAccounting,
  type OcrReservation,
} from '../core/image-ocr-budget.ts';
import {
  defaultImageOcrBudgetDirectory,
  ImageOcrManifestError,
  importValidatedImageOcrEntry,
  inspectImageOcrRegisteredRoot,
  parseAndValidateImageOcrManifest,
  readImageOcrSourceFile,
  type ValidatedImageOcrManifestEntry,
  type ImageOcrRunReport,
  type ValidatedImageOcrManifest,
} from '../core/image-ocr-run.ts';
import {
  buildBoundedImageOcrRequestBody,
  BoundedImageOcrReceiptError,
  IMAGE_OCR_NONVISUAL_INPUT_TOKEN_ALLOWANCE,
  loadBoundedImageOcrProviderConfig,
  parseBoundedImageOcrReceipt,
  requestBoundedImageOcr,
  type BoundedImageOcrReceipt,
  type BoundedImageOcrProviderConfig,
} from '../core/image-ocr-provider.ts';
import {
  withImageImportFence,
  type ImageImportFenceToken,
} from '../core/image-import-fence.ts';

const HELP = `Usage: gbrain image-ocr-run <manifest.jsonl> \\
  --max-images N --max-usd USD --reserve-usd-per-call USD --yes \\
  [--recurring-strict-absence]

Runs paid image OCR only for the exact, prevalidated JSONL entries. All caps
are mandatory and shared with prior runs on the same UTC date. --yes is an
explicit non-interactive confirmation; no env-only override exists.
`;

export interface ImageOcrRunArgs extends ImageOcrCaps {
  manifestPath: string;
  yes: true;
  recurringStrictAbsence: boolean;
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
  const allowed = new Set([
    '--max-images', '--max-usd', '--reserve-usd-per-call', '--yes',
    '--recurring-strict-absence',
  ]);
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
  return {
    manifestPath: positionals[0],
    yes: true,
    recurringStrictAbsence: args.includes('--recurring-strict-absence'),
    ...caps,
  };
}

const IMAGE_OCR_RAW_VALUE_FLAGS = new Set([
  '--brain', '--max-images', '--max-usd', '--reserve-usd-per-call',
]);
const IMAGE_OCR_RAW_BOOLEAN_FLAGS = new Set([
  '--help', '--yes', '--recurring-strict-absence',
]);

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
    reservations: 0,
    provider_attempts: 0,
    successful_provider_receipts: 0,
    observed_input_tokens: 0,
    observed_cache_creation_input_tokens: 0,
    observed_cache_read_input_tokens: 0,
    observed_output_tokens: 0,
    observed_usd: 0,
    persisted_imports: 0,
    failures: 0,
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
    lifecycle: ImageOcrReservationLifecycle,
  ) => void | Promise<void>;
  ocrProvider?: (
    imgBuf: Buffer,
    mime: string,
  ) => Promise<BoundedImageOcrReceipt>;
  imageImportFenceRoot?: string;
}

interface RecurringStrictAbsenceCounts {
  target_page_count: number | string;
  quality_donor_count: number | string;
  global_hash_page_count: number | string;
  file_row_count: number | string;
}

function strictZeroCount(value: number | string): boolean {
  return value === 0 || value === '0';
}

/** Re-prove recurring-only absence predicates under the shared import fence. */
async function revalidateRecurringStrictAbsence(
  engine: BrainEngine,
  entry: ValidatedImageOcrManifestEntry,
): Promise<void> {
  const currentSource = (await engine.listAllSources({ includeArchived: false }))
    .find(source => source.id === entry.source_id);
  if (!currentSource?.local_path) throw new Error('Recurring OCR source disappeared');
  const currentRoot = inspectImageOcrRegisteredRoot(currentSource.local_path);
  if (
    currentRoot.canonicalPath !== entry.registered_root
    || currentRoot.identity.device !== entry.registered_root_identity.device
    || currentRoot.identity.inode !== entry.registered_root_identity.inode
  ) {
    throw new Error('Recurring OCR source root changed');
  }
  readImageOcrSourceFile({
    filePath: entry.file_path,
    imageSlug: entry.slug,
    registeredRoot: entry.registered_root,
    expectedHash: entry.sha256,
    expectedFileIdentity: entry.file_identity,
  });

  const rows = await engine.executeRaw<RecurringStrictAbsenceCounts>(
    `SELECT
       (SELECT count(*) FROM pages target
         WHERE target.source_id = $1 AND target.slug = $2) AS target_page_count,
       (SELECT count(*)
          FROM pages p
          JOIN content_chunks cc ON cc.page_id = p.id
         WHERE p.deleted_at IS NULL
           AND p.page_kind = 'image'
           AND p.content_hash = $3
           AND (p.contextual_retrieval_mode IS NULL OR p.contextual_retrieval_mode = 'none')
           AND p.corpus_generation IS NULL
           AND cc.chunk_index = 0
           AND cc.chunk_source = 'image_asset'
           AND cc.modality = 'image'
           AND cc.chunk_text = p.compiled_truth
           AND length(p.compiled_truth) >= 120
           AND btrim(p.compiled_truth) <> ''
           AND p.compiled_truth ~ '[[:alnum:]]'
           AND cc.embedding IS NOT NULL
           AND btrim(cc.model) <> ''
           AND vector_dims(cc.embedding) > 0
           AND (cc.token_count IS NULL OR cc.token_count >= 0)
           AND (p.embedding_signature IS NULL
                OR p.embedding_signature = cc.model || ':' || vector_dims(cc.embedding)::text)
           AND (SELECT count(*) FROM content_chunks one_chunk WHERE one_chunk.page_id = p.id) = 1
       ) AS quality_donor_count,
       (SELECT count(*) FROM pages hashed WHERE hashed.content_hash = $3) AS global_hash_page_count,
       (SELECT count(*) FROM files stored WHERE stored.storage_path = $2) AS file_row_count`,
    [entry.source_id, entry.slug, entry.sha256],
  );
  if (
    rows.length !== 1
    || !strictZeroCount(rows[0].target_page_count)
    || !strictZeroCount(rows[0].quality_donor_count)
    || !strictZeroCount(rows[0].global_hash_page_count)
    || !strictZeroCount(rows[0].file_row_count)
  ) {
    throw new Error('Recurring OCR strict absence predicate changed');
  }
}

type ImageOcrLifecycleState =
  | 'unreserved'
  | 'reserved'
  | 'transport_attempted'
  | 'receipt_validated'
  | 'invalid_provider_observation'
  | 'persisted'
  | 'failed';

export interface ImageOcrReservationLifecycle {
  readonly state: ImageOcrLifecycleState;
  readonly transportAttempted: boolean;
  readonly receiptValidated: boolean;
  readonly persistenceSucceeded: boolean;
  recordTransportAttempt(): void;
  recordProviderReceipt(receipt: BoundedImageOcrReceipt): void;
  recordInvalidProviderObservation(observation: OcrInvalidProviderObservationAccounting): void;
  recordPersistenceSuccess(): void;
  recordFailure(stage: OcrFailureStage, outcome: OcrFailureOutcome): void;
}

function normalizeInjectedProviderReceipt(
  value: BoundedImageOcrReceipt,
  limits: { maxInputTokens: number; reservedUsd: number },
): BoundedImageOcrReceipt {
  const validated = parseBoundedImageOcrReceipt({
    id: value?.requestId,
    type: 'message',
    model: value?.model,
    stop_reason: value?.stopReason,
    content: [{ type: 'text', text: value?.text }],
    usage: {
      input_tokens: value?.inputTokens,
      cache_creation_input_tokens: value?.cacheCreationInputTokens,
      cache_read_input_tokens: value?.cacheReadInputTokens,
      output_tokens: value?.outputTokens,
    },
  }, limits);
  if (value.actualUsd !== validated.actualUsd) throw new BoundedImageOcrReceiptError();
  return validated;
}

async function callConfirmedImageOcrProvider(
  engine: BrainEngine,
  entry: ValidatedImageOcrManifestEntry,
  providerConfig: BoundedImageOcrProviderConfig | null,
  beforeProviderAttempt: () => Promise<OcrReservation>,
  lifecycle: ImageOcrReservationLifecycle,
  provider?: ImageOcrCommandOptions['ocrProvider'],
): Promise<{
  receipt: BoundedImageOcrReceipt;
  expectedPageState: BoundedImageOcrExpectedPageState;
}> {
  const reservation = await beforeProviderAttempt();
  const expectedMaxInputTokens = entry.visual_tokens + IMAGE_OCR_NONVISUAL_INPUT_TOKEN_ALLOWANCE;
  if (reservation.maxInputTokens !== expectedMaxInputTokens) {
    throw new Error('Durable OCR reservation has the wrong request-specific input token ceiling');
  }
  const receiptLimits = {
    maxInputTokens: reservation.maxInputTokens,
    reservedUsd: reservation.reservedUsd,
  };

  // Decode/build from a descriptor-bound post-reservation snapshot first. Any
  // async codec work completes before the final source/page re-read below.
  const sourceBytes = readImageOcrSourceFile({
    filePath: entry.file_path,
    imageSlug: entry.slug,
    registeredRoot: entry.registered_root,
    expectedHash: entry.sha256,
    expectedFileIdentity: entry.file_identity,
  });
  const prepared = await prepareBoundedImageOcrInput(
    entry.file_path,
    entry.slug,
    entry.registered_root,
    entry.sha256,
    sourceBytes,
  );
  if (
    prepared.info.format !== entry.image_format
    || prepared.info.width !== entry.image_width
    || prepared.info.height !== entry.image_height
    || prepared.info.pixels !== entry.image_pixels
    || prepared.info.visualTokens !== entry.visual_tokens
    || prepared.info.worstCaseUsd !== entry.worst_case_usd
  ) {
    throw new Error('Bounded image OCR format, dimensions, tokens, or cost changed after preflight');
  }
  // These are the final awaits before transport dispatch.
  const currentSource = (await engine.listAllSources({ includeArchived: false }))
    .find(source => source.id === entry.source_id);
  if (!currentSource?.local_path) throw new Error('Registered source disappeared after reservation');
  const currentRoot = inspectImageOcrRegisteredRoot(currentSource.local_path);
  if (
    currentRoot.canonicalPath !== entry.registered_root
    || currentRoot.identity.device !== entry.registered_root_identity.device
    || currentRoot.identity.inode !== entry.registered_root_identity.inode
  ) {
    throw new Error('Registered source root changed after manifest validation');
  }
  const existing = await engine.getPage(entry.slug, { sourceId: entry.source_id });
  if (existing?.content_hash === entry.sha256) {
    throw new Error('Source-scoped image already exists with the manifest hash');
  }
  const expectedPageState = captureBoundedImageOcrExpectedPageState(existing);

  // Reopen once more after the final await. The exact final buffer must match
  // the decoded/request-body source byte-for-byte. From here to invoking the
  // raw transport there is no await point.
  const finalRoot = inspectImageOcrRegisteredRoot(entry.registered_root);
  if (
    finalRoot.canonicalPath !== entry.registered_root
    || finalRoot.identity.device !== entry.registered_root_identity.device
    || finalRoot.identity.inode !== entry.registered_root_identity.inode
  ) {
    throw new Error('Registered source root changed during final validation');
  }
  const finalSourceBytes = readImageOcrSourceFile({
    filePath: entry.file_path,
    imageSlug: entry.slug,
    registeredRoot: entry.registered_root,
    expectedHash: entry.sha256,
    expectedFileIdentity: entry.file_identity,
  });
  if (!finalSourceBytes.equals(sourceBytes)) {
    throw new Error('Bounded image OCR source bytes changed during final validation');
  }
  const requestBody = buildBoundedImageOcrRequestBody(prepared.buf, prepared.mime);

  let receipt: BoundedImageOcrReceipt;
  if (provider) {
    lifecycle.recordTransportAttempt();
    let providerValue: Awaited<ReturnType<NonNullable<ImageOcrCommandOptions['ocrProvider']>>>;
    try {
      providerValue = await provider(prepared.buf, prepared.mime);
    } catch {
      lifecycle.recordFailure('provider_transport', 'ambiguous');
      throw new Error('Image OCR provider transport failed');
    }
    try {
      receipt = normalizeInjectedProviderReceipt(providerValue, receiptLimits);
    } catch (error) {
      if (error instanceof BoundedImageOcrReceiptError && error.observation) {
        lifecycle.recordInvalidProviderObservation(error.observation);
      } else {
        lifecycle.recordFailure('provider_receipt', 'ambiguous');
      }
      throw new Error('Image OCR provider receipt was invalid');
    }
  } else {
    try {
      receipt = await requestBoundedImageOcr({
        apiKey: providerConfig!.apiKey,
        providerBaseUrls: providerConfig!.providerBaseUrls,
        body: requestBody,
        maxInputTokens: receiptLimits.maxInputTokens,
        reservedUsd: receiptLimits.reservedUsd,
        onTransportAttempt: () => lifecycle.recordTransportAttempt(),
      });
    } catch (error) {
      if (lifecycle.state === 'transport_attempted') {
        if (error instanceof BoundedImageOcrReceiptError && error.observation) {
          lifecycle.recordInvalidProviderObservation(error.observation);
        } else {
          lifecycle.recordFailure(
            error instanceof BoundedImageOcrReceiptError ? 'provider_receipt' : 'provider_transport',
            'ambiguous',
          );
        }
      }
      throw new Error('Image OCR provider call failed');
    }
  }
  lifecycle.recordProviderReceipt(receipt);
  return { receipt, expectedPageState };
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
  recurringStrictAbsence: boolean;
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
  let providerAttempts = 0;
  let successfulProviderReceipts = 0;
  let observedInputTokens = 0;
  let observedCacheCreationInputTokens = 0;
  let observedCacheReadInputTokens = 0;
  let observedOutputTokens = 0;
  let observedUsdMicros = 0;
  let persistedImports = 0;

  for (let index = 0; index < input.manifest.entries.length; index++) {
    const entry = input.manifest.entries[index];
    let attemptNow: Date | null = null;
    let ledger: OcrBudgetLedger | null = null;
    let reservation: OcrReservation | null = null;
    let lifecycleState: ImageOcrLifecycleState = 'unreserved';
    let transportAttempted = false;
    let receiptValidated = false;
    let persistenceSucceeded = false;
    let callbackInvocations = 0;
    let reservationMade = false;
    let postReservationFailure = false;
    let callbackError: unknown;
    let runError: unknown;
    let closeFailed = false;
    let stop = false;

    const activeReservation = (): { ledger: OcrBudgetLedger; reservation: OcrReservation } => {
      const heldLedger = ledger as OcrBudgetLedger | null;
      const heldReservation = reservation as OcrReservation | null;
      if (!heldLedger || !heldReservation || !reservationMade) {
        throw new Error('OCR lifecycle event occurred without a durable reservation');
      }
      return { ledger: heldLedger, reservation: heldReservation };
    };
    const lifecycle: ImageOcrReservationLifecycle = {
      get state() { return lifecycleState; },
      get transportAttempted() { return transportAttempted; },
      get receiptValidated() { return receiptValidated; },
      get persistenceSucceeded() { return persistenceSucceeded; },
      recordTransportAttempt() {
        if (lifecycleState !== 'reserved') throw new Error('Invalid OCR lifecycle transport transition');
        const active = activeReservation();
        active.ledger.recordTransportAttempt(active.reservation);
        lifecycleState = 'transport_attempted';
        transportAttempted = true;
        providerAttempts++;
      },
      recordProviderReceipt(receiptValue) {
        if (lifecycleState !== 'transport_attempted') throw new Error('Invalid OCR lifecycle receipt transition');
        const active = activeReservation();
        if (active.reservation.maxInputTokens === null) {
          throw new Error('OCR reservation lacks a request-specific input token ceiling');
        }
        const receipt = normalizeInjectedProviderReceipt(receiptValue, {
          maxInputTokens: active.reservation.maxInputTokens,
          reservedUsd: active.reservation.reservedUsd,
        });
        active.ledger.recordProviderReceipt(active.reservation, receipt);
        lifecycleState = 'receipt_validated';
        receiptValidated = true;
        successfulProviderReceipts++;
        observedInputTokens += receipt.inputTokens;
        observedCacheCreationInputTokens += receipt.cacheCreationInputTokens;
        observedCacheReadInputTokens += receipt.cacheReadInputTokens;
        observedOutputTokens += receipt.outputTokens;
        observedUsdMicros += Math.round(receipt.actualUsd * 1_000_000);
      },
      recordInvalidProviderObservation(observation) {
        if (lifecycleState !== 'transport_attempted') {
          throw new Error('Invalid OCR lifecycle invalid-observation transition');
        }
        const active = activeReservation();
        active.ledger.recordInvalidProviderObservation(active.reservation, observation);
        lifecycleState = 'invalid_provider_observation';
        observedInputTokens += observation.inputTokens;
        observedCacheCreationInputTokens += observation.cacheCreationInputTokens;
        observedCacheReadInputTokens += observation.cacheReadInputTokens;
        observedOutputTokens += observation.outputTokens;
        observedUsdMicros += Math.round(observation.actualUsd * 1_000_000);
      },
      recordPersistenceSuccess() {
        if (lifecycleState !== 'receipt_validated') throw new Error('Invalid OCR lifecycle persistence transition');
        const active = activeReservation();
        active.ledger.recordPersistenceSuccess(active.reservation);
        lifecycleState = 'persisted';
        persistenceSucceeded = true;
        persistedImports++;
      },
      recordFailure(stage, outcome) {
        if (!['reserved', 'transport_attempted', 'receipt_validated'].includes(lifecycleState)) {
          throw new Error('Invalid OCR lifecycle failure transition');
        }
        const active = activeReservation();
        active.ledger.recordFailure(active.reservation, stage, outcome);
        lifecycleState = 'failed';
      },
    };

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
            if (input.recurringStrictAbsence) {
              await revalidateRecurringStrictAbsence(input.engine, entry);
            }
            // The clock and date-ledger reservation intentionally happen while
            // the shared import fence is held. The adapter then performs the
            // required post-reservation source/page/file revalidation.
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
            const madeReservation = ledger.reserve({
              index,
              sourceId: entry.source_id,
              slug: entry.slug,
              filePath: entry.file_path,
              registeredRoot: entry.registered_root,
              sha256: entry.sha256,
              maxInputTokens: entry.visual_tokens + IMAGE_OCR_NONVISUAL_INPUT_TOKEN_ALLOWANCE,
            });
            reservation = madeReservation;
            reservationMade = true;
            lifecycleState = 'reserved';
            processed++;
            after = ledger.snapshot();
            try {
              await input.afterReserve?.(entry, madeReservation);
            } catch (error) {
              postReservationFailure = true;
              throw error;
            }
            return madeReservation;
          } catch (error) {
            callbackError = error;
            throw error;
          }
        };

        try {
          await input.importEntry(entry, beforeProviderAttempt, fenceToken, lifecycle);
          if (callbackInvocations !== 1) {
            throw new Error('Provider adapter did not invoke beforeProviderAttempt exactly once');
          }
          if (callbackError) throw callbackError;
          if (!reservationMade) throw new Error('Provider adapter reached no durable OCR reservation');
          if (lifecycle.state !== 'persisted') {
            throw new Error('Provider adapter did not reconcile OCR persistence');
          }
        } finally {
          fenceHeld = false;
        }
      }, { lockRoot: input.imageImportFenceRoot });
      succeeded++;
    } catch (error) {
      runError = error;
      if (
        reservationMade
        && lifecycle.state !== 'persisted'
        && lifecycle.state !== 'failed'
        && lifecycle.state !== 'invalid_provider_observation'
      ) {
        const stage: OcrFailureStage = lifecycle.state === 'receipt_validated'
          ? 'persistence'
          : lifecycle.state === 'transport_attempted'
            ? 'provider_transport'
            : 'post_reservation_validation';
        const outcome: OcrFailureOutcome = (
          lifecycle.state === 'transport_attempted' || lifecycle.state === 'receipt_validated'
        )
          ? 'ambiguous'
          : 'failed';
        try {
          lifecycle.recordFailure(stage, outcome);
        } catch (auditError) {
          runError = auditError;
        }
      }
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
          reservations: processed,
          provider_attempts: providerAttempts,
          successful_provider_receipts: successfulProviderReceipts,
          observed_input_tokens: observedInputTokens,
          observed_cache_creation_input_tokens: observedCacheCreationInputTokens,
          observed_cache_read_input_tokens: observedCacheReadInputTokens,
          observed_output_tokens: observedOutputTokens,
          observed_usd: observedUsdMicros / 1_000_000,
          persisted_imports: persistedImports,
          failures: failed,
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
    reservations: processed,
    provider_attempts: providerAttempts,
    successful_provider_receipts: successfulProviderReceipts,
    observed_input_tokens: observedInputTokens,
    observed_cache_creation_input_tokens: observedCacheCreationInputTokens,
    observed_cache_read_input_tokens: observedCacheReadInputTokens,
    observed_output_tokens: observedOutputTokens,
    observed_usd: observedUsdMicros / 1_000_000,
    persisted_imports: persistedImports,
    failures: failed,
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
    const underReservedIndex = manifest.entries.findIndex(
      entry => entry.worst_case_usd > parsed!.reserveUsdPerCall,
    );
    if (underReservedIndex >= 0) {
      const entry = manifest.entries[underReservedIndex];
      throw new ImageOcrManifestError(underReservedIndex, entry.source_id, entry.slug);
    }
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
      recurringStrictAbsence: parsed.recurringStrictAbsence,
      importEntry: options.importEntry
        ?? (async (entry, beforeProviderAttempt, fenceToken, lifecycle) => {
          try {
            const providerResult = await callConfirmedImageOcrProvider(
              engine,
              entry,
              providerConfig,
              beforeProviderAttempt,
              lifecycle,
              options.ocrProvider,
            );
            try {
              await importValidatedImageOcrEntry(
                engine,
                entry,
                providerResult.receipt.text,
                providerResult.expectedPageState,
                fenceToken,
              );
            } catch (error) {
              lifecycle.recordFailure('persistence', 'failed');
              throw error;
            }
            lifecycle.recordPersistenceSuccess();
          } finally {
            if (lifecycle.transportAttempted) await bumpOcrCounter(engine, 'ocr_provider_attempts');
            if (lifecycle.receiptValidated) await bumpOcrCounter(engine, 'ocr_successful_provider_receipts');
            if (lifecycle.persistenceSucceeded) await bumpOcrCounter(engine, 'ocr_persisted_imports');
            if (lifecycle.state === 'failed' || lifecycle.state === 'invalid_provider_observation') {
              await bumpOcrCounter(engine, 'ocr_failures');
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
      reservations: 0,
      provider_attempts: 0,
      successful_provider_receipts: 0,
      observed_input_tokens: 0,
      observed_cache_creation_input_tokens: 0,
      observed_cache_read_input_tokens: 0,
      observed_output_tokens: 0,
      observed_usd: 0,
      persisted_imports: 0,
      failures: 0,
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
