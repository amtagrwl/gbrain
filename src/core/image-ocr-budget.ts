import {
  closeSync,
  existsSync,
  fstatSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  rmdirSync,
  unlinkSync,
  writeFileSync,
  type Stats,
} from 'node:fs';
import { randomBytes } from 'node:crypto';
import { hostname } from 'node:os';
import { dirname, join } from 'node:path';
import { canonicalLookup } from './model-pricing.ts';

export const IMAGE_OCR_POLICY_MAX_IMAGES = 1000;
export const IMAGE_OCR_POLICY_MAX_USD = 10;
export const IMAGE_OCR_POLICY_MIN_RESERVE_USD = 0.01;
// Fixed Anthropic model id used verbatim in the raw Messages API request.
export const IMAGE_OCR_POLICY_MODEL = 'claude-haiku-4-5-20251001';
export const IMAGE_OCR_POLICY_MAX_OUTPUT_TOKENS = 1024;
const USD_MICROS = 1_000_000;

export interface ImageOcrCaps {
  maxImages: number;
  maxUsd: number;
  reserveUsdPerCall: number;
}

interface LedgerAuditEntry {
  reservation_id: string;
  reserved_at: string;
  manifest_hash: string;
  entry_index: number;
  source_id: string;
  slug: string;
  file_path: string;
  registered_root: string;
  sha256: string;
  reserved_usd_micros: number;
  max_input_tokens: number | null;
  state: 'reserved' | 'transport_attempted' | 'receipt_validated' | 'invalid_provider_observation' | 'persisted' | 'failed';
  transport_attempted: boolean;
  provider_receipt: LedgerProviderReceipt | null;
  invalid_provider_observation: LedgerInvalidProviderObservation | null;
  persistence_succeeded: boolean;
  outcome: 'pending' | 'ambiguous' | 'invalid' | 'over_limit' | 'persisted' | 'failed';
  failure_stage: OcrFailureStage | null;
}

interface LedgerProviderReceipt {
  request_id: string;
  model: string;
  stop_reason: string;
  input_tokens: number;
  cache_creation_input_tokens: number;
  cache_read_input_tokens: number;
  output_tokens: number;
  actual_usd_micros: number;
}

interface LedgerInvalidProviderObservation extends LedgerProviderReceipt {
  outcome: OcrInvalidProviderObservationOutcome;
  limit_violations: OcrProviderLimitViolation[];
}

interface LedgerFile {
  schema_version: 2;
  utc_date: string;
  daily_max_images: number;
  daily_max_usd_micros: number;
  reserve_usd_per_call_micros: number;
  calls_reserved: number;
  usd_reserved_micros: number;
  audit: LedgerAuditEntry[];
}

interface BudgetLockOwner {
  schema_version: 1;
  pid: number;
  ppid: number;
  process_started_at_ms: number;
  hostname: string;
  owner_token: string;
  utc_date: string;
  acquired_at: string;
}

interface FileIdentity {
  dev: number;
  ino: number;
}

interface BudgetLockHandle {
  directoryFd: number;
  directoryIdentity: FileIdentity;
  lockPath: string;
  ownerFd: number;
  ownerIdentity: FileIdentity;
  ownerPath: string;
  owner: BudgetLockOwner;
}

const PROCESS_STARTED_AT_MS = Math.max(1, Math.round(Date.now() - process.uptime() * 1000));

export interface OcrBudgetSnapshot {
  utcDate: string;
  callsReserved: number;
  usdReserved: number;
  caps: ImageOcrCaps;
}

export class OcrBudgetLockError extends Error {
  constructor(readonly lockPath: string) {
    super(`Image OCR budget is locked or ambiguous: ${lockPath}. Refusing to break the lock.`);
    this.name = 'OcrBudgetLockError';
  }
}

export class OcrBudgetCapError extends Error {
  constructor(readonly cap: 'images' | 'usd') {
    super(`Daily image OCR ${cap} cap reached`);
    this.name = 'OcrBudgetCapError';
  }
}

export interface OcrReservation {
  readonly reservationId: string;
  readonly utcDate: string;
  readonly sourceId: string;
  readonly slug: string;
  readonly filePath: string;
  readonly registeredRoot: string;
  readonly sha256: string;
  readonly maxInputTokens: number | null;
  readonly reservedUsd: number;
}

export type OcrFailureStage =
  | 'post_reservation_validation'
  | 'provider_transport'
  | 'provider_receipt'
  | 'persistence'
  | 'adapter';

export type OcrFailureOutcome = 'failed' | 'ambiguous';

export interface OcrReceiptAccounting {
  readonly requestId: string;
  readonly model: string;
  readonly stopReason: string;
  readonly inputTokens: number;
  readonly cacheCreationInputTokens: number;
  readonly cacheReadInputTokens: number;
  readonly outputTokens: number;
  readonly actualUsd: number;
}

export type OcrInvalidProviderObservationOutcome = 'invalid' | 'over_limit';

export type OcrProviderLimitViolation = 'input_tokens' | 'output_tokens' | 'actual_usd';

export interface OcrInvalidProviderObservationAccounting extends OcrReceiptAccounting {
  readonly outcome: OcrInvalidProviderObservationOutcome;
  readonly limitViolations: readonly OcrProviderLimitViolation[];
}

function usdToMicros(value: number): number {
  return Math.round(value * USD_MICROS);
}

function assertExactUsdMicros(value: number, flag: string): void {
  if (Math.abs(microsToUsd(usdToMicros(value)) - value) > Number.EPSILON) {
    throw new Error(`${flag} supports at most 6 decimal places`);
  }
}

function microsToUsd(value: number): number {
  return value / USD_MICROS;
}

export function validateImageOcrCaps(input: ImageOcrCaps): ImageOcrCaps {
  const { maxImages, maxUsd, reserveUsdPerCall } = input;
  if (!Number.isFinite(maxImages) || !Number.isInteger(maxImages) || maxImages <= 0) {
    throw new Error('--max-images must be a positive finite integer');
  }
  if (!Number.isFinite(maxUsd) || maxUsd <= 0) {
    throw new Error('--max-usd must be a positive finite number');
  }
  if (!Number.isFinite(reserveUsdPerCall) || reserveUsdPerCall <= 0) {
    throw new Error('--reserve-usd-per-call must be a positive finite number');
  }
  assertExactUsdMicros(maxUsd, '--max-usd');
  assertExactUsdMicros(reserveUsdPerCall, '--reserve-usd-per-call');
  if (maxImages > IMAGE_OCR_POLICY_MAX_IMAGES) {
    throw new Error(`--max-images may not exceed policy ceiling ${IMAGE_OCR_POLICY_MAX_IMAGES}`);
  }
  if (maxUsd > IMAGE_OCR_POLICY_MAX_USD) {
    throw new Error(`--max-usd may not exceed policy ceiling ${IMAGE_OCR_POLICY_MAX_USD}`);
  }
  if (reserveUsdPerCall < IMAGE_OCR_POLICY_MIN_RESERVE_USD) {
    throw new Error(`--reserve-usd-per-call may not be below policy floor ${IMAGE_OCR_POLICY_MIN_RESERVE_USD}`);
  }
  return { maxImages, maxUsd, reserveUsdPerCall };
}

function utcDate(now: Date): string {
  if (!Number.isFinite(now.getTime())) throw new Error('Invalid OCR budget clock');
  return now.toISOString().slice(0, 10);
}

function fsyncDirectory(path: string): void {
  const fd = openSync(path, 'r');
  try { fsyncSync(fd); } finally { closeSync(fd); }
}

function fileIdentity(stat: Stats): FileIdentity {
  return { dev: stat.dev, ino: stat.ino };
}

function sameIdentity(left: FileIdentity, right: FileIdentity): boolean {
  return left.dev === right.dev && left.ino === right.ino;
}

function ownerMatches(actual: unknown, expected: BudgetLockOwner): boolean {
  if (!actual || typeof actual !== 'object' || Array.isArray(actual)) return false;
  const owner = actual as Partial<BudgetLockOwner>;
  return owner.schema_version === expected.schema_version
    && owner.pid === expected.pid
    && owner.ppid === expected.ppid
    && owner.process_started_at_ms === expected.process_started_at_ms
    && owner.hostname === expected.hostname
    && owner.owner_token === expected.owner_token
    && owner.utc_date === expected.utc_date
    && owner.acquired_at === expected.acquired_at;
}

function verifyBudgetLockOwnership(lock: BudgetLockHandle): boolean {
  let currentDirectoryFd: number | null = null;
  let currentOwnerFd: number | null = null;
  try {
    const heldDirectory = fstatSync(lock.directoryFd);
    const heldOwner = fstatSync(lock.ownerFd);
    if (
      !heldDirectory.isDirectory()
      || !heldOwner.isFile()
      || !sameIdentity(fileIdentity(heldDirectory), lock.directoryIdentity)
      || !sameIdentity(fileIdentity(heldOwner), lock.ownerIdentity)
    ) return false;

    currentDirectoryFd = openSync(lock.lockPath, 'r');
    const currentDirectory = fstatSync(currentDirectoryFd);
    if (
      !currentDirectory.isDirectory()
      || !sameIdentity(fileIdentity(currentDirectory), lock.directoryIdentity)
    ) return false;

    currentOwnerFd = openSync(lock.ownerPath, 'r');
    const currentOwner = fstatSync(currentOwnerFd);
    if (!currentOwner.isFile() || !sameIdentity(fileIdentity(currentOwner), lock.ownerIdentity)) {
      return false;
    }
    const parsed = JSON.parse(readFileSync(currentOwnerFd, 'utf8')) as unknown;
    return ownerMatches(parsed, lock.owner);
  } catch {
    return false;
  } finally {
    if (currentOwnerFd !== null) closeSync(currentOwnerFd);
    if (currentDirectoryFd !== null) closeSync(currentDirectoryFd);
  }
}

function closeBudgetLockDescriptors(lock: BudgetLockHandle): void {
  closeSync(lock.ownerFd);
  closeSync(lock.directoryFd);
}

function releaseBudgetLock(lock: BudgetLockHandle): void {
  const owned = verifyBudgetLockOwnership(lock);
  closeBudgetLockDescriptors(lock);
  if (!owned) throw new OcrBudgetLockError(lock.lockPath);

  // The owner filename contains 256 bits of randomness. A late owner can only
  // remove its own file; if the fixed directory was externally replaced, its
  // token-named path is absent and a replacement owner's file keeps rmdir from
  // succeeding. Never recursively remove this directory.
  unlinkSync(lock.ownerPath);
  rmdirSync(lock.lockPath);
  fsyncDirectory(dirname(lock.lockPath));
}

function acquireBudgetLock(directory: string, date: string, now: Date): BudgetLockHandle {
  const lockPath = join(directory, `${date}.lock`);
  try {
    mkdirSync(lockPath, { mode: 0o700 });
    fsyncDirectory(directory);
  } catch {
    throw new OcrBudgetLockError(lockPath);
  }

  const ownerToken = randomBytes(32).toString('hex');
  const owner: BudgetLockOwner = {
    schema_version: 1,
    pid: process.pid,
    ppid: process.ppid,
    process_started_at_ms: PROCESS_STARTED_AT_MS,
    hostname: hostname(),
    owner_token: ownerToken,
    utc_date: date,
    acquired_at: now.toISOString(),
  };
  const ownerPath = join(lockPath, `owner-${ownerToken}.json`);
  let directoryFd: number | null = null;
  let ownerFd: number | null = null;
  try {
    directoryFd = openSync(lockPath, 'r');
    ownerFd = openSync(ownerPath, 'wx', 0o600);
    writeFileSync(ownerFd, `${JSON.stringify(owner)}\n`, 'utf8');
    fsyncSync(ownerFd);
    fsyncSync(directoryFd);
    const directoryStat = fstatSync(directoryFd);
    const ownerStat = fstatSync(ownerFd);
    if (!directoryStat.isDirectory() || !ownerStat.isFile()) {
      throw new Error('Invalid image OCR budget lock filesystem objects');
    }
    return {
      directoryFd,
      directoryIdentity: fileIdentity(directoryStat),
      lockPath,
      ownerFd,
      ownerIdentity: fileIdentity(ownerStat),
      ownerPath,
      owner,
    };
  } catch {
    if (ownerFd !== null) closeSync(ownerFd);
    if (directoryFd !== null) closeSync(directoryFd);
    // Leave any partial lock in place. Its state is ambiguous and requires
    // explicit operator recovery after confirming that no owner remains.
    throw new OcrBudgetLockError(lockPath);
  }
}

function writeLedgerDurably(path: string, ledger: LedgerFile): void {
  const tmp = `${path}.tmp-${process.pid}`;
  const bytes = `${JSON.stringify(ledger)}\n`;
  const fd = openSync(tmp, 'wx', 0o600);
  try {
    writeFileSync(fd, bytes, 'utf8');
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
  renameSync(tmp, path);
  fsyncDirectory(dirname(path));
}

function providerAccountingIsValid(receipt: LedgerProviderReceipt | null): receipt is LedgerProviderReceipt {
  const price = canonicalLookup(IMAGE_OCR_POLICY_MODEL);
  return !!receipt && !!price
    && typeof receipt.request_id === 'string' && receipt.request_id.length > 0
    && receipt.model === IMAGE_OCR_POLICY_MODEL
    && receipt.stop_reason === 'end_turn'
    && Number.isSafeInteger(receipt.input_tokens) && receipt.input_tokens >= 0
    && receipt.cache_creation_input_tokens === 0
    && receipt.cache_read_input_tokens === 0
    && Number.isSafeInteger(receipt.output_tokens) && receipt.output_tokens >= 0
    && Number.isSafeInteger(receipt.actual_usd_micros) && receipt.actual_usd_micros >= 0
    && receipt.actual_usd_micros === Math.round(
      receipt.input_tokens * price.input + receipt.output_tokens * price.output,
    );
}

function providerReceiptIsValid(receipt: LedgerProviderReceipt | null, entry: LedgerAuditEntry): boolean {
  return providerAccountingIsValid(receipt)
    && receipt.output_tokens <= IMAGE_OCR_POLICY_MAX_OUTPUT_TOKENS
    && (entry.max_input_tokens === null || receipt.input_tokens <= entry.max_input_tokens)
    && receipt.actual_usd_micros <= entry.reserved_usd_micros;
}

function invalidProviderObservationIsValid(
  observation: LedgerInvalidProviderObservation | null,
  entry: LedgerAuditEntry,
): boolean {
  if (!providerAccountingIsValid(observation)) return false;
  if (!['invalid', 'over_limit'].includes(observation.outcome)) return false;
  if (
    !Array.isArray(observation.limit_violations)
    || observation.limit_violations.some(value => !['input_tokens', 'output_tokens', 'actual_usd'].includes(value))
    || new Set(observation.limit_violations).size !== observation.limit_violations.length
  ) return false;
  const expectedViolations: OcrProviderLimitViolation[] = [];
  if (entry.max_input_tokens !== null && observation.input_tokens > entry.max_input_tokens) {
    expectedViolations.push('input_tokens');
  }
  if (observation.output_tokens > IMAGE_OCR_POLICY_MAX_OUTPUT_TOKENS) {
    expectedViolations.push('output_tokens');
  }
  if (observation.actual_usd_micros > entry.reserved_usd_micros) {
    expectedViolations.push('actual_usd');
  }
  if (observation.outcome === 'over_limit') {
    return expectedViolations.length > 0
      && observation.limit_violations.join('\0') === expectedViolations.join('\0');
  }
  return observation.limit_violations.length === 0;
}

function auditLifecycleIsValid(entry: LedgerAuditEntry): boolean {
  const failureStages: ReadonlySet<string> = new Set([
    'post_reservation_validation',
    'provider_transport',
    'provider_receipt',
    'persistence',
    'adapter',
  ]);
  if (
    typeof entry.transport_attempted !== 'boolean'
    || typeof entry.persistence_succeeded !== 'boolean'
    || !(entry.max_input_tokens === null || (Number.isSafeInteger(entry.max_input_tokens) && entry.max_input_tokens >= 0))
    || !['reserved', 'transport_attempted', 'receipt_validated', 'invalid_provider_observation', 'persisted', 'failed'].includes(entry.state)
    || !['pending', 'ambiguous', 'invalid', 'over_limit', 'persisted', 'failed'].includes(entry.outcome)
    || !(entry.failure_stage === null || failureStages.has(entry.failure_stage))
  ) return false;

  switch (entry.state) {
    case 'reserved':
      return !entry.transport_attempted && entry.provider_receipt === null
        && entry.invalid_provider_observation === null
        && !entry.persistence_succeeded && entry.outcome === 'pending' && entry.failure_stage === null;
    case 'transport_attempted':
      return entry.transport_attempted && entry.provider_receipt === null
        && entry.invalid_provider_observation === null
        && !entry.persistence_succeeded && entry.outcome === 'ambiguous' && entry.failure_stage === null;
    case 'receipt_validated':
      return entry.transport_attempted && providerReceiptIsValid(entry.provider_receipt, entry)
        && entry.invalid_provider_observation === null
        && !entry.persistence_succeeded && entry.outcome === 'pending' && entry.failure_stage === null;
    case 'invalid_provider_observation':
      return entry.transport_attempted && entry.provider_receipt === null
        && invalidProviderObservationIsValid(entry.invalid_provider_observation, entry)
        && !entry.persistence_succeeded
        && entry.outcome === entry.invalid_provider_observation?.outcome
        && entry.failure_stage === 'provider_receipt';
    case 'persisted':
      return entry.transport_attempted && providerReceiptIsValid(entry.provider_receipt, entry)
        && entry.invalid_provider_observation === null
        && entry.persistence_succeeded && entry.outcome === 'persisted' && entry.failure_stage === null;
    case 'failed':
      if (
        entry.persistence_succeeded
        || entry.invalid_provider_observation !== null
        || entry.failure_stage === null
        || (entry.outcome !== 'failed' && entry.outcome !== 'ambiguous')
        || (entry.provider_receipt && (!entry.transport_attempted || !providerReceiptIsValid(entry.provider_receipt, entry)))
      ) return false;
      if (entry.failure_stage === 'post_reservation_validation') {
        return !entry.transport_attempted && entry.provider_receipt === null && entry.outcome === 'failed';
      }
      if (entry.failure_stage === 'provider_transport' || entry.failure_stage === 'provider_receipt') {
        return entry.transport_attempted && entry.provider_receipt === null && entry.outcome === 'ambiguous';
      }
      if (entry.failure_stage === 'persistence') {
        return entry.transport_attempted && providerReceiptIsValid(entry.provider_receipt, entry);
      }
      return !entry.persistence_succeeded
        && entry.failure_stage !== null
        && (entry.outcome === 'failed' || entry.outcome === 'ambiguous')
        && (!entry.provider_receipt || (entry.transport_attempted && providerReceiptIsValid(entry.provider_receipt, entry)));
  }
}

function normalizeLedgerAuditEntry(value: unknown, expectedDate: string): LedgerAuditEntry | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const raw = value as Record<string, unknown>;
  const entry = {
    ...raw,
    max_input_tokens: raw.max_input_tokens ?? null,
    invalid_provider_observation: raw.invalid_provider_observation ?? null,
  } as unknown as LedgerAuditEntry;
  if (
    !/^[a-f0-9]{64}$/.test(entry.reservation_id)
    || typeof entry.reserved_at !== 'string'
    || entry.reserved_at.slice(0, 10) !== expectedDate
    || !/^[a-f0-9]{64}$/.test(entry.manifest_hash)
    || !Number.isSafeInteger(entry.entry_index) || entry.entry_index < 0
    || typeof entry.source_id !== 'string' || entry.source_id.length === 0
    || typeof entry.slug !== 'string' || entry.slug.length === 0
    || typeof entry.file_path !== 'string' || !entry.file_path.startsWith('/')
    || typeof entry.registered_root !== 'string' || !entry.registered_root.startsWith('/')
    || !/^[a-f0-9]{64}$/.test(entry.sha256)
    || !Number.isSafeInteger(entry.reserved_usd_micros)
    || entry.reserved_usd_micros < usdToMicros(IMAGE_OCR_POLICY_MIN_RESERVE_USD)
    || !(entry.invalid_provider_observation === null
      || (typeof entry.invalid_provider_observation === 'object'
        && !Array.isArray(entry.invalid_provider_observation)))
    || !auditLifecycleIsValid(entry)
  ) return null;
  return entry;
}

function parseLedger(path: string, expectedDate: string): LedgerFile {
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as Partial<LedgerFile>;
  const normalizedAudit = Array.isArray(parsed.audit)
    ? parsed.audit.map(entry => normalizeLedgerAuditEntry(entry, expectedDate))
    : [];
  const auditValid = Array.isArray(parsed.audit) && normalizedAudit.every(entry => entry !== null);
  const audit = auditValid ? normalizedAudit as LedgerAuditEntry[] : [];
  const auditUsd = audit.reduce((sum, entry) => sum + Math.max(
    entry.reserved_usd_micros,
    entry.invalid_provider_observation?.actual_usd_micros ?? 0,
  ), 0);
  if (
    parsed.schema_version !== 2
    || parsed.utc_date !== expectedDate
    || !Number.isSafeInteger(parsed.daily_max_images)
    || (parsed.daily_max_images ?? 0) <= 0
    || (parsed.daily_max_images ?? Infinity) > IMAGE_OCR_POLICY_MAX_IMAGES
    || !Number.isSafeInteger(parsed.daily_max_usd_micros)
    || (parsed.daily_max_usd_micros ?? 0) <= 0
    || (parsed.daily_max_usd_micros ?? Infinity) > usdToMicros(IMAGE_OCR_POLICY_MAX_USD)
    || !Number.isSafeInteger(parsed.reserve_usd_per_call_micros)
    || (parsed.reserve_usd_per_call_micros ?? 0) < usdToMicros(IMAGE_OCR_POLICY_MIN_RESERVE_USD)
    || !Number.isSafeInteger(parsed.calls_reserved)
    || (parsed.calls_reserved ?? -1) < 0
    || !Number.isSafeInteger(parsed.usd_reserved_micros)
    || (parsed.usd_reserved_micros ?? -1) < 0
    || !auditValid
    || audit.length !== parsed.calls_reserved
    || auditUsd !== parsed.usd_reserved_micros
  ) {
    throw new Error(`Invalid or ambiguous image OCR budget ledger: ${path}`);
  }
  return { ...parsed, audit } as LedgerFile;
}

export class OcrBudgetLedger {
  private closed = false;

  private constructor(
    private readonly lock: BudgetLockHandle,
    private readonly ledgerPath: string,
    private caps: ImageOcrCaps,
    private readonly manifestHash: string,
    private readonly now: Date,
    private ledger: LedgerFile,
  ) {}

  static acquire(input: {
    directory: string;
    now: Date;
    caps: ImageOcrCaps;
    manifestHash: string;
  }): OcrBudgetLedger {
    const caps = validateImageOcrCaps(input.caps);
    const date = utcDate(input.now);
    mkdirSync(input.directory, { recursive: true, mode: 0o700 });
    const lock = acquireBudgetLock(input.directory, date, input.now);

    const ledgerPath = join(input.directory, `${date}.json`);
    let ledger: LedgerFile;
    const hadLedger = existsSync(ledgerPath);
    try {
      ledger = hadLedger
        ? parseLedger(ledgerPath, date)
        : {
          schema_version: 2 as const,
          utc_date: date,
          daily_max_images: caps.maxImages,
          daily_max_usd_micros: usdToMicros(caps.maxUsd),
          reserve_usd_per_call_micros: usdToMicros(caps.reserveUsdPerCall),
          calls_reserved: 0,
          usd_reserved_micros: 0,
          audit: [],
        };
    } catch (error) {
      closeBudgetLockDescriptors(lock);
      // The lock is intentionally retained: ledger ambiguity must require operator review.
      throw error;
    }

    try {
      if (hadLedger) {
        if (!verifyBudgetLockOwnership(lock)) throw new OcrBudgetLockError(lock.lockPath);
        if (
          caps.maxImages > ledger.daily_max_images
          || usdToMicros(caps.maxUsd) > ledger.daily_max_usd_micros
          || usdToMicros(caps.reserveUsdPerCall) < ledger.reserve_usd_per_call_micros
        ) {
          throw new Error('Same-day image OCR caps may only become stricter, never looser');
        }
        ledger = {
          ...ledger,
          daily_max_images: Math.min(ledger.daily_max_images, caps.maxImages),
          daily_max_usd_micros: Math.min(ledger.daily_max_usd_micros, usdToMicros(caps.maxUsd)),
          reserve_usd_per_call_micros: Math.max(ledger.reserve_usd_per_call_micros, usdToMicros(caps.reserveUsdPerCall)),
        };
        writeLedgerDurably(ledgerPath, ledger);
      }
      const effectiveCaps = {
        maxImages: ledger.daily_max_images,
        maxUsd: microsToUsd(ledger.daily_max_usd_micros),
        reserveUsdPerCall: microsToUsd(ledger.reserve_usd_per_call_micros),
      };
      return new OcrBudgetLedger(lock, ledgerPath, effectiveCaps, input.manifestHash, input.now, ledger);
    } catch (error) {
      releaseBudgetLock(lock);
      throw error;
    }
  }

  snapshot(): OcrBudgetSnapshot {
    return {
      utcDate: this.ledger.utc_date,
      callsReserved: this.ledger.calls_reserved,
      usdReserved: microsToUsd(this.ledger.usd_reserved_micros),
      caps: { ...this.caps },
    };
  }

  nextCapExceeded(): 'images' | 'usd' | null {
    if (this.ledger.calls_reserved + 1 > this.caps.maxImages) return 'images';
    if (this.ledger.usd_reserved_micros + usdToMicros(this.caps.reserveUsdPerCall) > usdToMicros(this.caps.maxUsd)) return 'usd';
    return null;
  }

  reserve(entry: {
    index: number;
    sourceId: string;
    slug: string;
    filePath: string;
    registeredRoot: string;
    sha256: string;
    maxInputTokens?: number;
  }): OcrReservation {
    if (this.closed) throw new Error('Image OCR budget ledger is closed');
    if (!verifyBudgetLockOwnership(this.lock)) throw new OcrBudgetLockError(this.lock.lockPath);
    const exceeded = this.nextCapExceeded();
    if (exceeded) throw new OcrBudgetCapError(exceeded);
    const reserveMicros = usdToMicros(this.caps.reserveUsdPerCall);
    const maxInputTokens = entry.maxInputTokens ?? null;
    if (maxInputTokens !== null && (!Number.isSafeInteger(maxInputTokens) || maxInputTokens < 0)) {
      throw new Error('Invalid OCR request-specific input token ceiling');
    }
    const reservationId = randomBytes(32).toString('hex');
    const reservation: OcrReservation = Object.freeze({
      reservationId,
      utcDate: this.ledger.utc_date,
      sourceId: entry.sourceId,
      slug: entry.slug,
      filePath: entry.filePath,
      registeredRoot: entry.registeredRoot,
      sha256: entry.sha256,
      maxInputTokens,
      reservedUsd: microsToUsd(reserveMicros),
    });
    this.ledger = {
      ...this.ledger,
      calls_reserved: this.ledger.calls_reserved + 1,
      usd_reserved_micros: this.ledger.usd_reserved_micros + reserveMicros,
      audit: [...this.ledger.audit, {
        reservation_id: reservationId,
        reserved_at: this.now.toISOString(),
        manifest_hash: this.manifestHash,
        entry_index: entry.index,
        source_id: entry.sourceId,
        slug: entry.slug,
        file_path: entry.filePath,
        registered_root: entry.registeredRoot,
        sha256: entry.sha256,
        reserved_usd_micros: reserveMicros,
        max_input_tokens: maxInputTokens,
        state: 'reserved',
        transport_attempted: false,
        provider_receipt: null,
        invalid_provider_observation: null,
        persistence_succeeded: false,
        outcome: 'pending',
        failure_stage: null,
      }],
    };
    writeLedgerDurably(this.ledgerPath, this.ledger);
    return reservation;
  }

  recordTransportAttempt(reservation: OcrReservation): void {
    this.updateReservation(reservation, (entry) => {
      if (entry.state !== 'reserved') throw new Error('Invalid OCR audit transition to transport_attempted');
      return {
        ...entry,
        state: 'transport_attempted',
        transport_attempted: true,
        outcome: 'ambiguous',
      };
    });
  }

  recordProviderReceipt(reservation: OcrReservation, receipt: OcrReceiptAccounting): void {
    this.updateReservation(reservation, (entry) => {
      if (entry.state !== 'transport_attempted') throw new Error('Invalid OCR audit transition to receipt_validated');
      const actualUsdMicros = usdToMicros(receipt.actualUsd);
      const providerReceipt: LedgerProviderReceipt = {
        request_id: receipt.requestId,
        model: receipt.model,
        stop_reason: receipt.stopReason,
        input_tokens: receipt.inputTokens,
        cache_creation_input_tokens: receipt.cacheCreationInputTokens,
        cache_read_input_tokens: receipt.cacheReadInputTokens,
        output_tokens: receipt.outputTokens,
        actual_usd_micros: actualUsdMicros,
      };
      if (!providerReceiptIsValid(providerReceipt, entry)) throw new Error('Invalid OCR provider receipt accounting');
      return {
        ...entry,
        state: 'receipt_validated',
        provider_receipt: providerReceipt,
        outcome: 'pending',
      };
    });
  }

  recordInvalidProviderObservation(
    reservation: OcrReservation,
    observation: OcrInvalidProviderObservationAccounting,
  ): void {
    this.updateReservation(reservation, (entry) => {
      if (entry.state !== 'transport_attempted') {
        throw new Error('Invalid OCR audit transition to invalid_provider_observation');
      }
      const invalidObservation: LedgerInvalidProviderObservation = {
        request_id: observation.requestId,
        model: observation.model,
        stop_reason: observation.stopReason,
        input_tokens: observation.inputTokens,
        cache_creation_input_tokens: observation.cacheCreationInputTokens,
        cache_read_input_tokens: observation.cacheReadInputTokens,
        output_tokens: observation.outputTokens,
        actual_usd_micros: usdToMicros(observation.actualUsd),
        outcome: observation.outcome,
        limit_violations: [...observation.limitViolations],
      };
      if (!invalidProviderObservationIsValid(invalidObservation, entry)) {
        throw new Error('Invalid OCR provider observation accounting');
      }
      return {
        ...entry,
        state: 'invalid_provider_observation',
        invalid_provider_observation: invalidObservation,
        outcome: observation.outcome,
        failure_stage: 'provider_receipt',
      };
    });
  }

  recordPersistenceSuccess(reservation: OcrReservation): void {
    this.updateReservation(reservation, (entry) => {
      if (entry.state !== 'receipt_validated') throw new Error('Invalid OCR audit transition to persisted');
      return {
        ...entry,
        state: 'persisted',
        persistence_succeeded: true,
        outcome: 'persisted',
      };
    });
  }

  recordFailure(
    reservation: OcrReservation,
    stage: OcrFailureStage,
    outcome: OcrFailureOutcome,
  ): void {
    this.updateReservation(reservation, (entry) => {
      if (
        entry.state === 'persisted'
        || entry.state === 'invalid_provider_observation'
        || entry.state === 'failed'
      ) {
        throw new Error('Invalid OCR audit transition to failed');
      }
      return {
        ...entry,
        state: 'failed',
        outcome,
        failure_stage: stage,
      };
    });
  }

  private updateReservation(
    reservation: OcrReservation,
    update: (entry: LedgerAuditEntry) => LedgerAuditEntry,
  ): void {
    if (this.closed) throw new Error('Image OCR budget ledger is closed');
    if (!verifyBudgetLockOwnership(this.lock)) throw new OcrBudgetLockError(this.lock.lockPath);
    const index = this.ledger.audit.findIndex(entry => entry.reservation_id === reservation.reservationId);
    if (index < 0) throw new Error('Image OCR reservation is not present in the held ledger');
    const audit = [...this.ledger.audit];
    audit[index] = update(audit[index]);
    const usdReservedMicros = audit.reduce((sum, entry) => sum + Math.max(
      entry.reserved_usd_micros,
      entry.invalid_provider_observation?.actual_usd_micros ?? 0,
    ), 0);
    this.ledger = { ...this.ledger, usd_reserved_micros: usdReservedMicros, audit };
    writeLedgerDurably(this.ledgerPath, this.ledger);
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    releaseBudgetLock(this.lock);
  }
}
