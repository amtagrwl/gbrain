import {
  closeSync,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import { dirname, join } from 'node:path';

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
  reserved_at: string;
  manifest_hash: string;
  entry_index: number;
  source_id: string;
  slug: string;
  file_path: string;
  registered_root: string;
  sha256: string;
  reserved_usd_micros: number;
}

interface LedgerFile {
  schema_version: 1;
  utc_date: string;
  daily_max_images: number;
  daily_max_usd_micros: number;
  reserve_usd_per_call_micros: number;
  calls_reserved: number;
  usd_reserved_micros: number;
  audit: LedgerAuditEntry[];
}

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
  readonly utcDate: string;
  readonly sourceId: string;
  readonly slug: string;
  readonly filePath: string;
  readonly registeredRoot: string;
  readonly sha256: string;
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

function parseLedger(path: string, expectedDate: string): LedgerFile {
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as Partial<LedgerFile>;
  const auditValid = Array.isArray(parsed.audit) && parsed.audit.every((entry): entry is LedgerAuditEntry => (
    !!entry
    && typeof entry === 'object'
    && typeof entry.reserved_at === 'string'
    && entry.reserved_at.slice(0, 10) === expectedDate
    && /^[a-f0-9]{64}$/.test(entry.manifest_hash)
    && Number.isSafeInteger(entry.entry_index) && entry.entry_index >= 0
    && typeof entry.source_id === 'string' && entry.source_id.length > 0
    && typeof entry.slug === 'string' && entry.slug.length > 0
    && typeof entry.file_path === 'string' && entry.file_path.startsWith('/')
    && typeof entry.registered_root === 'string' && entry.registered_root.startsWith('/')
    && /^[a-f0-9]{64}$/.test(entry.sha256)
    && Number.isSafeInteger(entry.reserved_usd_micros)
    && entry.reserved_usd_micros >= usdToMicros(IMAGE_OCR_POLICY_MIN_RESERVE_USD)
  ));
  const audit = auditValid ? parsed.audit as LedgerAuditEntry[] : [];
  const auditUsd = audit.reduce((sum, entry) => sum + entry.reserved_usd_micros, 0);
  if (
    parsed.schema_version !== 1
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
  return parsed as LedgerFile;
}

export class OcrBudgetLedger {
  private closed = false;

  private constructor(
    private readonly lockFd: number,
    private readonly lockPath: string,
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
    const lockPath = join(input.directory, `${date}.lock`);
    let lockFd: number;
    try {
      lockFd = openSync(lockPath, 'wx', 0o600);
      writeFileSync(lockFd, `${JSON.stringify({ pid: process.pid, utc_date: date, acquired_at: input.now.toISOString() })}\n`);
      fsyncSync(lockFd);
      fsyncDirectory(input.directory);
    } catch {
      throw new OcrBudgetLockError(lockPath);
    }

    const ledgerPath = join(input.directory, `${date}.json`);
    let ledger: LedgerFile;
    const hadLedger = existsSync(ledgerPath);
    try {
      ledger = hadLedger
        ? parseLedger(ledgerPath, date)
        : {
          schema_version: 1 as const,
          utc_date: date,
          daily_max_images: caps.maxImages,
          daily_max_usd_micros: usdToMicros(caps.maxUsd),
          reserve_usd_per_call_micros: usdToMicros(caps.reserveUsdPerCall),
          calls_reserved: 0,
          usd_reserved_micros: 0,
          audit: [],
        };
    } catch (error) {
      closeSync(lockFd);
      // The lock is intentionally retained: ledger ambiguity must require operator review.
      throw error;
    }

    try {
      if (hadLedger) {
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
      return new OcrBudgetLedger(lockFd, lockPath, ledgerPath, effectiveCaps, input.manifestHash, input.now, ledger);
    } catch (error) {
      closeSync(lockFd);
      unlinkSync(lockPath);
      fsyncDirectory(input.directory);
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

  reserve(entry: { index: number; sourceId: string; slug: string; filePath: string; registeredRoot: string; sha256: string }): OcrReservation {
    if (this.closed) throw new Error('Image OCR budget ledger is closed');
    const exceeded = this.nextCapExceeded();
    if (exceeded) throw new OcrBudgetCapError(exceeded);
    const reserveMicros = usdToMicros(this.caps.reserveUsdPerCall);
    const reservation: OcrReservation = Object.freeze({
      utcDate: this.ledger.utc_date,
      sourceId: entry.sourceId,
      slug: entry.slug,
      filePath: entry.filePath,
      registeredRoot: entry.registeredRoot,
      sha256: entry.sha256,
    });
    this.ledger = {
      ...this.ledger,
      calls_reserved: this.ledger.calls_reserved + 1,
      usd_reserved_micros: this.ledger.usd_reserved_micros + reserveMicros,
      audit: [...this.ledger.audit, {
        reserved_at: this.now.toISOString(),
        manifest_hash: this.manifestHash,
        entry_index: entry.index,
        source_id: entry.sourceId,
        slug: entry.slug,
        file_path: entry.filePath,
        registered_root: entry.registeredRoot,
        sha256: entry.sha256,
        reserved_usd_micros: reserveMicros,
      }],
    };
    writeLedgerDurably(this.ledgerPath, this.ledger);
    return reservation;
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    closeSync(this.lockFd);
    unlinkSync(this.lockPath);
    fsyncDirectory(dirname(this.lockPath));
  }
}
