import { afterEach, describe, expect, test } from 'bun:test';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import type { BrainEngine } from '../src/core/engine.ts';
import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import {
  IMAGE_OCR_POLICY_MAX_OUTPUT_TOKENS,
  IMAGE_OCR_POLICY_MODEL,
  OcrBudgetLedger,
  OcrBudgetLockError,
  validateImageOcrCaps,
} from '../src/core/image-ocr-budget.ts';
import {
  parseAndValidateImageOcrManifest,
} from '../src/core/image-ocr-run.ts';
import {
  importImageFile,
  maybeOcr,
} from '../src/core/import-file.ts';
import { withImageImportFence } from '../src/core/image-import-fence.ts';
import {
  parseImageOcrRunArgs,
  runImageOcrRun,
  validateImageOcrRunRawArgv,
} from '../src/commands/image-ocr-run.ts';
import { rawArgvTargetsImageOcrRun } from '../src/cli.ts';
import { CLI_FLAG_REGISTRY } from '../src/core/cli-flag-registry.generated.ts';
import {
  BOUNDED_IMAGE_OCR_ENDPOINT,
  BOUNDED_IMAGE_OCR_PROMPT,
  IMAGE_OCR_CURRENT_WORST_CASE_USD,
  IMAGE_OCR_INPUT_USD_PER_MTOK,
  IMAGE_OCR_MAX_VISUAL_TOKENS,
  IMAGE_OCR_NONVISUAL_INPUT_TOKEN_ALLOWANCE,
  IMAGE_OCR_OUTPUT_USD_PER_MTOK,
  buildBoundedImageOcrRequestBody,
  parseBoundedImageOcrReceipt,
  requestBoundedImageOcr,
  type BoundedImageOcrReceipt,
} from '../src/core/image-ocr-provider.ts';
import { withEnv } from './helpers/with-env.ts';

const UTC_DAY_1 = new Date('2026-08-10T23:59:59.000Z');
const UTC_DAY_2 = new Date('2026-08-11T00:00:00.000Z');
const TINY_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  'base64',
);
const TINY_JPEG = Buffer.from(
  '/9j/4AAQSkZJRgABAgAAAQABAAD//gAQTGF2YzYyLjI4LjEwMgD/2wBDAAgEBAQEBAUFBQUFBQYGBgYGBgYGBgYGBgYHBwcICAgHBwcGBgcHCAgICAkJCQgICAgJCQoKCgwMCwsODg4RERT/xABLAAEBAAAAAAAAAAAAAAAAAAAABwEBAAAAAAAAAAAAAAAAAAAAABABAAAAAAAAAAAAAAAAAAAAABEBAAAAAAAAAAAAAAAAAAAAAP/AABEIAAIAAgMBIgACEQADEQD/2gAMAwEAAhEDEQA/AL+AD//Z',
  'base64',
);
const TINY_GIF = Buffer.from('R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==', 'base64');
const TINY_WEBP = Buffer.from('UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoCAAIAAgA0JaQAA3AA/vtdAAA=', 'base64');

function anthropicMessageResponse(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: 'msg_injected_123',
    type: 'message',
    model: IMAGE_OCR_POLICY_MODEL,
    stop_reason: 'end_turn',
    content: [{ type: 'text', text: 'verbatim text' }],
    usage: {
      input_tokens: 100,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
      output_tokens: 10,
    },
    ...overrides,
  };
}

function injectedReceipt(text = 'verbatim text'): BoundedImageOcrReceipt {
  return {
    text,
    model: IMAGE_OCR_POLICY_MODEL,
    stopReason: 'end_turn',
    requestId: 'msg_injected_123',
    inputTokens: 100,
    cacheCreationInputTokens: 0,
    cacheReadInputTokens: 0,
    outputTokens: 10,
    actualUsd: 0.00015,
  };
}

function injectedReceiptWithUsage(
  inputTokens: number,
  outputTokens: number,
  text = 'verbatim text',
): BoundedImageOcrReceipt {
  return {
    ...injectedReceipt(text),
    inputTokens,
    outputTokens,
    actualUsd: (
      inputTokens * IMAGE_OCR_INPUT_USD_PER_MTOK
      + outputTokens * IMAGE_OCR_OUTPUT_USD_PER_MTOK
    ) / 1_000_000,
  };
}

function sha256(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function pngCrc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngWithDimensions(width: number, height: number): Buffer {
  const bytes = Buffer.from(TINY_PNG);
  bytes.writeUInt32BE(width, 16);
  bytes.writeUInt32BE(height, 20);
  bytes.writeUInt32BE(pngCrc32(bytes.subarray(12, 29)), 29);
  return bytes;
}

function animatedGif(): Buffer {
  const trailer = TINY_GIF.length - 1;
  const imageStart = TINY_GIF.indexOf(0x2c);
  return Buffer.concat([
    TINY_GIF.subarray(0, trailer),
    TINY_GIF.subarray(imageStart, trailer),
    TINY_GIF.subarray(trailer),
  ]);
}

function jpegWithTruncatedEntropyStream(): Buffer {
  const startOfScan = TINY_JPEG.indexOf(Buffer.from([0xff, 0xda]));
  if (startOfScan < 0) throw new Error('Tiny JPEG fixture lacks SOS');
  const scanDataStart = startOfScan + 2 + TINY_JPEG.readUInt16BE(startOfScan + 2);
  return Buffer.concat([
    TINY_JPEG.subarray(0, scanDataStart),
    Buffer.from([0xff, 0xd9]),
  ]);
}

function gifWithTruncatedLzwPayload(): Buffer {
  const bytes = Buffer.from(TINY_GIF);
  const imageDescriptor = bytes.indexOf(0x2c);
  if (imageDescriptor < 0) throw new Error('Tiny GIF fixture lacks an image descriptor');
  bytes.writeUInt16LE(1_000, 6);
  bytes.writeUInt16LE(1_000, 8);
  bytes.writeUInt16LE(1_000, imageDescriptor + 5);
  bytes.writeUInt16LE(1_000, imageDescriptor + 7);
  return bytes;
}

function webpWithTruncatedVp8Payload(): Buffer {
  const vp8DataStart = 20;
  const bytes = Buffer.from(TINY_WEBP.subarray(0, vp8DataStart + 10));
  bytes.writeUInt32LE(bytes.length - 8, 4);
  bytes.writeUInt32LE(10, 16);
  bytes.writeUInt16LE(1_000, vp8DataStart + 6);
  bytes.writeUInt16LE(1_000, vp8DataStart + 8);
  return bytes;
}

function reservationFields(filePath = '/tmp/image.png', hash = 'a'.repeat(64), registeredRoot = dirname(filePath)) {
  return { filePath, registeredRoot, sha256: hash };
}

async function waitForFile(path: string, timeoutMs = 5_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!existsSync(path) && Date.now() < deadline) await Bun.sleep(10);
  if (!existsSync(path)) throw new Error(`Timed out waiting for ${path}`);
}

function fixture(count = 1) {
  const root = mkdtempSync(join(tmpdir(), 'gbrain-bounded-ocr-root-'));
  const ledgerDir = mkdtempSync(join(tmpdir(), 'gbrain-bounded-ocr-ledger-'));
  const lines: string[] = [];
  for (let i = 0; i < count; i++) {
    const rel = `images/${String(i).padStart(4, '0')}.png`;
    const filePath = join(root, rel);
    mkdirSync(join(root, 'images'), { recursive: true });
    const bytes = Buffer.from(TINY_PNG);
    writeFileSync(filePath, bytes);
    lines.push(JSON.stringify({
      source_id: 'source-a',
      slug: rel,
      file_path: filePath,
      sha256: sha256(bytes),
    }));
  }
  const manifestPath = join(root, 'manifest.jsonl');
  writeFileSync(manifestPath, `${lines.join('\n')}\n`);
  const engine = {
    listAllSources: async () => [{
      id: 'source-a',
      name: 'Source A',
      local_path: root,
      last_sync_at: null,
      config: {},
    }],
    getPage: async () => null,
  } as unknown as BrainEngine;
  return { root, ledgerDir, manifestPath, engine };
}

function replaceFixtureImage(
  f: ReturnType<typeof fixture>,
  bytes: Buffer,
  extension = '.png',
): { file_path: string; sha256: string; slug: string; source_id: string } {
  const slug = `images/0000${extension}`;
  const filePath = join(f.root, slug);
  writeFileSync(filePath, bytes);
  const entry = {
    source_id: 'source-a',
    slug,
    file_path: filePath,
    sha256: sha256(bytes),
  };
  writeFileSync(f.manifestPath, `${JSON.stringify(entry)}\n`);
  return entry;
}

function statefulImageEngine(root: string) {
  let registeredRoot = root;
  let syntheticPageDrift = false;
  const pages = new Map<string, Record<string, unknown>>();
  const chunks = new Map<string, Array<Record<string, unknown>>>();
  let nextId = 1;
  const engine = {
    kind: 'postgres',
    listAllSources: async () => [{
      id: 'source-a',
      name: 'Source A',
      local_path: registeredRoot,
      last_sync_at: null,
      config: {},
    }],
    getPage: async (slug: string) => syntheticPageDrift
      ? { id: 'drift', content_hash: 'f'.repeat(64) }
      : pages.get(slug) ?? null,
    getConfig: async () => '0',
    setConfig: async () => {},
    transaction: async (fn: (tx: BrainEngine) => Promise<unknown>) => fn(engine as unknown as BrainEngine),
    executeRaw: async (sql: string, params: unknown[] = []) => {
      if (/INSERT INTO pages[\s\S]*ON CONFLICT \(source_id, slug\) DO NOTHING/.test(sql)) {
        return pages.has(String(params[1])) ? [] : [{ id: -1 }];
      }
      if (/FROM pages[\s\S]*FOR UPDATE/.test(sql)) {
        const current = pages.get(String(params[1]));
        if (!current) return [];
        const updatedAt = current.updated_at instanceof Date ? current.updated_at.toISOString() : null;
        const deletedAt = current.deleted_at instanceof Date ? current.deleted_at.toISOString() : null;
        return current.id === params[2]
          && current.generation === params[3]
          && updatedAt === params[4]
          && (current.content_hash ?? null) === params[5]
          && deletedAt === params[6]
          ? [{ id: current.id }]
          : [];
      }
      throw new Error('Unexpected statefulImageEngine raw SQL');
    },
    createVersion: async () => {},
    putPage: async (slug: string, page: Record<string, unknown>) => {
      pages.set(slug, { id: `page-${nextId++}`, ...page });
    },
    upsertFile: async () => {},
    upsertChunks: async (slug: string, value: Array<Record<string, unknown>>) => {
      chunks.set(slug, value);
    },
    deleteChunks: async (slug: string) => { chunks.delete(slug); },
    addLink: async () => {},
  } as unknown as BrainEngine;
  return {
    engine,
    pages,
    chunks,
    get registeredRoot() { return registeredRoot; },
    set registeredRoot(value: string) { registeredRoot = value; },
    get syntheticPageDrift() { return syntheticPageDrift; },
    set syntheticPageDrift(value: boolean) { syntheticPageDrift = value; },
  };
}

function commandArgs(f: ReturnType<typeof fixture>, caps = { maxImages: 1, maxUsd: 1, reserveUsdPerCall: 0.01 }) {
  return [
    f.manifestPath,
    '--max-images', String(caps.maxImages),
    '--max-usd', String(caps.maxUsd),
    '--reserve-usd-per-call', String(caps.reserveUsdPerCall),
    '--yes',
  ];
}

function recurringStrictCommandArgs(
  f: ReturnType<typeof fixture>,
  caps = { maxImages: 1, maxUsd: 1, reserveUsdPerCall: 0.01 },
) {
  return [...commandArgs(f, caps), '--recurring-strict-absence'];
}

async function runCommand(
  f: ReturnType<typeof fixture>,
  caps = { maxImages: 1, maxUsd: 1, reserveUsdPerCall: 0.01 },
  options: NonNullable<Parameters<typeof runImageOcrRun>[2]> = {},
) {
  const originalWrite = process.stdout.write.bind(process.stdout);
  process.stdout.write = (() => true) as typeof process.stdout.write;
  try {
    return await runImageOcrRun(f.engine, commandArgs(f, caps), {
      ledgerDirectory: f.ledgerDir,
      imageImportFenceRoot: join(f.ledgerDir, 'image-import-fence'),
      now: UTC_DAY_1,
      ...options,
    });
  } finally {
    process.stdout.write = originalWrite;
  }
}

async function runRecurringStrictCommand(
  f: ReturnType<typeof fixture>,
  strictCounts: Partial<{
    target_page_count: number;
    quality_donor_count: number;
    global_hash_page_count: number;
    file_row_count: number;
  }> = {},
) {
  let providerCalls = 0;
  let mutations = 0;
  let strictSql = '';
  f.engine = {
    ...f.engine,
    executeRaw: async (sql: string) => {
      strictSql = sql;
      return [{
        target_page_count: 0,
        quality_donor_count: 0,
        global_hash_page_count: 0,
        file_row_count: 0,
        ...strictCounts,
      }];
    },
  } as unknown as BrainEngine;
  const originalWrite = process.stdout.write.bind(process.stdout);
  process.stdout.write = (() => true) as typeof process.stdout.write;
  try {
    const report = await runImageOcrRun(f.engine, recurringStrictCommandArgs(f), {
      ledgerDirectory: f.ledgerDir,
      imageImportFenceRoot: join(f.ledgerDir, 'image-import-fence'),
      now: UTC_DAY_1,
      importEntry: async (_entry, beforeProviderAttempt, _fenceToken, lifecycle) => {
        await beforeProviderAttempt();
        providerCalls++;
        lifecycle.recordTransportAttempt();
        lifecycle.recordProviderReceipt(injectedReceipt());
        mutations++;
        lifecycle.recordPersistenceSuccess();
      },
    });
    return { report, providerCalls, mutations, strictSql };
  } finally {
    process.stdout.write = originalWrite;
  }
}

type TestImportEntry = NonNullable<NonNullable<Parameters<typeof runImageOcrRun>[2]>['importEntry']>;

function reservedImport(
  run: (entry: Parameters<TestImportEntry>[0]) => void | Promise<void>,
): TestImportEntry {
  return async (entry, beforeProviderAttempt, _fenceToken, lifecycle) => {
    await beforeProviderAttempt();
    lifecycle.recordTransportAttempt();
    try {
      await run(entry);
      lifecycle.recordProviderReceipt(injectedReceipt());
      lifecycle.recordPersistenceSuccess();
    } catch (error) {
      lifecycle.recordFailure('provider_transport', 'ambiguous');
      throw error;
    }
  };
}

afterEach(() => {
  delete process.env.GBRAIN_EMBEDDING_IMAGE_OCR;
});

describe('bounded OCR authorization', () => {
  test('env boolean without a ledger reservation makes zero provider calls', async () => {
    process.env.GBRAIN_EMBEDDING_IMAGE_OCR = 'true';

    const text = await maybeOcr({} as BrainEngine, Buffer.from('x'), 'image/png');
    expect(text).toBe('');
    const importerSource = readFileSync('src/core/import-file.ts', 'utf8');
    expect(importerSource).not.toContain("import('./ai/gateway.ts')");
    expect(importerSource).not.toContain('generateOcrText');
  });

  test('strict OCR failure occurs before page, chunk, file, or stub writes', async () => {
    const f = fixture();
    const writes: string[] = [];
    const engine = new Proxy({
      listAllSources: f.engine.listAllSources.bind(f.engine),
      getPage: async () => null,
      getConfig: async () => '0',
      setConfig: async () => {},
    }, {
      get(target, prop, receiver) {
        if (Reflect.has(target, prop)) return Reflect.get(target, prop, receiver);
        if (typeof prop === 'string') return async () => { writes.push(prop); throw new Error(`unexpected write ${prop}`); };
      },
    }) as unknown as BrainEngine;
    f.engine = engine;
    const report = await runCommand(f, undefined, {
      ocrProvider: async () => { throw new Error('provider body'); },
    });
    expect(report).toMatchObject({
      status: 'failed',
      reservations: 1,
      provider_attempts: 1,
      successful_provider_receipts: 0,
      observed_input_tokens: 0,
      observed_cache_creation_input_tokens: 0,
      observed_cache_read_input_tokens: 0,
      observed_output_tokens: 0,
      observed_usd: 0,
      persisted_imports: 0,
      failures: 1,
    });
    expect(writes).toEqual([]);
    expect(JSON.parse(readFileSync(join(f.ledgerDir, '2026-08-10.json'), 'utf8')).audit[0]).toMatchObject({
      state: 'failed',
      transport_attempted: true,
      provider_receipt: null,
      persistence_succeeded: false,
      outcome: 'ambiguous',
      failure_stage: 'provider_transport',
    });
  });

  test('empty OCR output is a strict failure and never becomes a filename stub', async () => {
    const f = fixture();
    f.engine = {
      ...f.engine,
      getConfig: async () => '0',
      setConfig: async () => {},
    } as unknown as BrainEngine;
    const report = await runCommand(f, undefined, { ocrProvider: async () => injectedReceipt('   ') });
    expect(report?.status).toBe('failed');
    expect(report?.succeeded).toBe(0);
  });
});

describe('cap parsing and policy ceilings', () => {
  test('requires every cap plus explicit --yes', () => {
    for (const args of [
      ['manifest.jsonl', '--max-usd', '1', '--reserve-usd-per-call', '0.01', '--yes'],
      ['manifest.jsonl', '--max-images', '100', '--reserve-usd-per-call', '0.01', '--yes'],
      ['manifest.jsonl', '--max-images', '100', '--max-usd', '1', '--yes'],
      ['manifest.jsonl', '--max-images', '100', '--max-usd', '1', '--reserve-usd-per-call', '0.01'],
    ]) {
      expect(() => parseImageOcrRunArgs(args)).toThrow();
    }
    expect(() => parseImageOcrRunArgs([
      'manifest.jsonl', '--max-images', '1', '--max-usd', '1',
      '--reserve-usd-per-call', '0.01', '--yes', '--full',
    ])).toThrow(/Unknown/);
  });

  test('accepts only the explicit recurring strict-absence contract flag', () => {
    const f = fixture();
    expect(parseImageOcrRunArgs(recurringStrictCommandArgs(f))).toMatchObject({
      manifestPath: f.manifestPath,
      recurringStrictAbsence: true,
    });
    expect(validateImageOcrRunRawArgv([
      '--brain', 'host', 'image-ocr-run', ...recurringStrictCommandArgs(f),
    ])).toBeNull();
  });

  test('rejects zero, negative, non-finite, and policy-loosening values', () => {
    const invalid = [
      { maxImages: 0, maxUsd: 1, reserveUsdPerCall: 0.01 },
      { maxImages: -1, maxUsd: 1, reserveUsdPerCall: 0.01 },
      { maxImages: 1.5, maxUsd: 1, reserveUsdPerCall: 0.01 },
      { maxImages: 1, maxUsd: Number.NaN, reserveUsdPerCall: 0.01 },
      { maxImages: 1, maxUsd: 0, reserveUsdPerCall: 0.01 },
      { maxImages: 1, maxUsd: 1, reserveUsdPerCall: 0 },
      { maxImages: 1001, maxUsd: 10, reserveUsdPerCall: 0.01 },
      { maxImages: 1000, maxUsd: 10.000001, reserveUsdPerCall: 0.01 },
      { maxImages: 1000, maxUsd: 10, reserveUsdPerCall: 0.009999 },
    ];
    for (const caps of invalid) expect(() => validateImageOcrCaps(caps)).toThrow();
    expect(validateImageOcrCaps({ maxImages: 1000, maxUsd: 10, reserveUsdPerCall: 0.01 }))
      .toEqual({ maxImages: 1000, maxUsd: 10, reserveUsdPerCall: 0.01 });
  });
});

describe('recurring strict absence provider boundary', () => {
  test.each([
    ['soft-deleted target key', { target_page_count: 1 }],
    ['different-hash target created after wrapper classification', { target_page_count: 1 }],
    ['qualifying exact-hash quality donor', { quality_donor_count: 1 }],
    ['global hash collision', { global_hash_page_count: 1 }],
    ['global files.storage_path collision', { file_row_count: 1 }],
  ] as const)('%s stops before reservation, provider, or mutation', async (_name, counts) => {
    const f = fixture();
    const result = await runRecurringStrictCommand(f, counts);
    expect(result.report).toMatchObject({
      reservations: 0,
      provider_attempts: 0,
      persisted_imports: 0,
      status: 'rejected',
      terminal_error: 'run_rejected',
    });
    expect(result.providerCalls).toBe(0);
    expect(result.mutations).toBe(0);
    expect(readdirSync(f.ledgerDir).filter(name => name.endsWith('.json'))).toEqual([]);
  });

  test('re-proves exact target, deterministic min-120 donor, global hash, and file absence', async () => {
    const f = fixture();
    const result = await runRecurringStrictCommand(f);
    expect(result.report).toMatchObject({ status: 'completed', reservations: 1, provider_attempts: 1 });
    expect(result.strictSql).toContain('p.deleted_at IS NULL');
    expect(result.strictSql).toContain("p.page_kind = 'image'");
    expect(result.strictSql).toContain("cc.chunk_source = 'image_asset'");
    expect(result.strictSql).toContain("cc.modality = 'image'");
    expect(result.strictSql).toContain('length(p.compiled_truth) >= 120');
    expect(result.strictSql).toContain('cc.embedding IS NOT NULL');
    expect(result.strictSql).toContain('vector_dims(cc.embedding) > 0');
    expect(result.strictSql).toContain('vector_dims(cc.embedding)');
    expect(result.strictSql).toContain('FROM files');
    expect(result.providerCalls).toBe(1);
    expect(result.mutations).toBe(1);
  });
});

describe('manifest preflight', () => {
  test('accepts a deterministic valid JSONL manifest', async () => {
    const f = fixture(2);
    const result = await parseAndValidateImageOcrManifest(f.engine, f.manifestPath);
    expect(result.entries).toHaveLength(2);
    expect(result.manifestHash).toBe(sha256(readFileSync(f.manifestPath)));
  });

  test('rejects duplicates, traversal, outside-root, slug, hash, and type mismatches', async () => {
    const mutations: Array<(entry: Record<string, string>, f: ReturnType<typeof fixture>) => void> = [
      (entry) => { entry.slug = '../images/0000.png'; },
      (entry, f) => { entry.file_path = join(f.root, '..', 'outside.png'); writeFileSync(entry.file_path, 'x'); entry.sha256 = sha256(Buffer.from('x')); },
      (entry) => { entry.slug = 'images/different.png'; },
      (entry) => { entry.sha256 = '0'.repeat(64); },
      (entry, f) => { const p = join(f.root, 'images/0000.txt'); writeFileSync(p, 'x'); entry.file_path = p; entry.slug = 'images/0000.txt'; entry.sha256 = sha256(Buffer.from('x')); },
    ];
    for (const mutate of mutations) {
      const f = fixture();
      const entry = JSON.parse(readFileSync(f.manifestPath, 'utf8').trim());
      mutate(entry, f);
      writeFileSync(f.manifestPath, `${JSON.stringify(entry)}\n`);
      await expect(parseAndValidateImageOcrManifest(f.engine, f.manifestPath)).rejects.toThrow();
    }

    const f = fixture();
    const line = readFileSync(f.manifestPath, 'utf8').trim();
    writeFileSync(f.manifestPath, `${line}\n${line}\n`);
    await expect(parseAndValidateImageOcrManifest(f.engine, f.manifestPath)).rejects.toThrow(/entry 1 is invalid/i);
  });

  test('invalid entry is identified in the machine-readable report before provider access', async () => {
    const f = fixture();
    const entry = JSON.parse(readFileSync(f.manifestPath, 'utf8').trim());
    entry.sha256 = '0'.repeat(64);
    writeFileSync(f.manifestPath, `${JSON.stringify(entry)}\n`);
    const report = await runCommand(f);
    expect(report).toMatchObject({
      status: 'rejected',
      terminal_error: 'manifest_invalid',
      first_failing_entry: {
        index: 0,
        source_id: 'source-a',
        slug: 'images/0000.png',
        code: 'manifest_invalid',
      },
    });
  });

  test('rejects a symlinked registered source root before reservation or transport', async () => {
    const f = fixture();
    const linkedRoot = join(dirname(f.root), `${f.root.split('/').at(-1)}-link`);
    symlinkSync(f.root, linkedRoot, 'dir');
    const entry = JSON.parse(readFileSync(f.manifestPath, 'utf8').trim());
    entry.file_path = join(linkedRoot, 'images/0000.png');
    writeFileSync(f.manifestPath, `${JSON.stringify(entry)}\n`);
    let providerCalls = 0;
    f.engine = {
      ...f.engine,
      listAllSources: async () => [{
        id: 'source-a', name: 'Source A', local_path: linkedRoot, last_sync_at: null, config: {},
      }],
      getConfig: async () => '0',
      setConfig: async () => {},
    } as unknown as BrainEngine;

    const report = await runCommand(f, undefined, {
      ocrProvider: async () => { providerCalls++; return injectedReceipt('text'); },
    });
    expect(providerCalls).toBe(0);
    expect(report).toMatchObject({ status: 'rejected', processed: 0, terminal_error: 'manifest_invalid' });
  });

  test('malformed and zero-byte images are rejected before reservation or transport', async () => {
    for (const bytes of [Buffer.alloc(0), Buffer.from('not a png')]) {
      const f = fixture();
      replaceFixtureImage(f, bytes);
      let providerCalls = 0;
      const report = await runCommand(f, undefined, {
        ocrProvider: async () => { providerCalls++; return injectedReceipt('text'); },
      });
      expect(providerCalls).toBe(0);
      expect(report).toMatchObject({ status: 'rejected', processed: 0, terminal_error: 'manifest_invalid' });
    }
  });

  for (const codec of [
    { name: 'JPEG', extension: '.jpg', bytes: jpegWithTruncatedEntropyStream() },
    { name: 'GIF', extension: '.gif', bytes: gifWithTruncatedLzwPayload() },
    { name: 'WebP', extension: '.webp', bytes: webpWithTruncatedVp8Payload() },
  ]) {
    test(`corrupt ${codec.name} compressed payload fails before reservation or transport`, async () => {
      const f = fixture();
      replaceFixtureImage(f, codec.bytes, codec.extension);
      let providerCalls = 0;
      const report = await runCommand(f, undefined, {
        ocrProvider: async () => {
          providerCalls++;
          return injectedReceipt('MUST NOT RUN');
        },
      });
      expect(providerCalls).toBe(0);
      expect(report).toMatchObject({
        status: 'rejected',
        processed: 0,
        reservations: 0,
        provider_attempts: 0,
        terminal_error: 'manifest_invalid',
      });
      expect(existsSync(join(f.ledgerDir, '2026-08-10.json'))).toBe(false);
    });
  }

  test('animated or multiframe content is rejected before reservation or transport', async () => {
    const f = fixture();
    replaceFixtureImage(f, animatedGif(), '.gif');
    let providerCalls = 0;
    const report = await runCommand(f, undefined, {
      ocrProvider: async () => { providerCalls++; return injectedReceipt('text'); },
    });
    expect(providerCalls).toBe(0);
    expect(report).toMatchObject({ status: 'rejected', processed: 0, terminal_error: 'manifest_invalid' });
  });

  test('provider-dimension and decoded-pixel bombs are rejected before decode or transport', async () => {
    for (const bytes of [pngWithDimensions(8_001, 1), pngWithDimensions(5_001, 5_000)]) {
      const f = fixture();
      replaceFixtureImage(f, bytes);
      let providerCalls = 0;
      const report = await runCommand(f, undefined, {
        ocrProvider: async () => { providerCalls++; return injectedReceipt('text'); },
      });
      expect(providerCalls).toBe(0);
      expect(report).toMatchObject({ status: 'rejected', processed: 0, terminal_error: 'manifest_invalid' });
    }
  });

  test('all allowed codecs have validated single-frame dimensions before wire preparation', async () => {
    const { prepareBoundedImageOcrBytes } = await import('../src/core/image-ocr-image.ts');
    const codecs = [
      { extension: '.png', bytes: TINY_PNG, format: 'png', width: 1, height: 1 },
      { extension: '.jpg', bytes: TINY_JPEG, format: 'jpeg', width: 2, height: 2 },
      { extension: '.gif', bytes: TINY_GIF, format: 'gif', width: 1, height: 1 },
      { extension: '.webp', bytes: TINY_WEBP, format: 'webp', width: 2, height: 2 },
      { extension: '.heic', bytes: readFileSync('test/fixtures/images/tiny.heic'), format: 'heic', width: 356, height: 356 },
      { extension: '.heif', bytes: readFileSync('test/fixtures/images/tiny.heic'), format: 'heic', width: 356, height: 356 },
      { extension: '.avif', bytes: readFileSync('test/fixtures/images/tiny.avif'), format: 'avif', width: 8, height: 8 },
    ];
    for (const codec of codecs) {
      const prepared = await prepareBoundedImageOcrBytes(codec.bytes, codec.extension);
      expect(prepared.info).toMatchObject({
        format: codec.format,
        width: codec.width,
        height: codec.height,
        frameCount: 1,
      });
      expect(prepared.info.visualTokens).toBeGreaterThan(0);
      expect(prepared.info.worstCaseUsd).toBeLessThanOrEqual(0.01);
      expect(prepared.buf.length).toBeGreaterThan(0);
    }
  });

  test('dimension, pixel, token, and cost enforcement is exact at every boundary', async () => {
    const {
      IMAGE_OCR_LOCAL_MAX_PIXELS,
      IMAGE_OCR_PROVIDER_MAX_DIMENSION,
      validateAndPriceImageOcrDimensions,
    } = await import('../src/core/image-ocr-image.ts');
    expect(IMAGE_OCR_PROVIDER_MAX_DIMENSION).toBe(8_000);
    expect(IMAGE_OCR_LOCAL_MAX_PIXELS).toBe(25_000_000);
    expect(validateAndPriceImageOcrDimensions(8_000, 1)).toMatchObject({ width: 8_000, height: 1 });
    expect(validateAndPriceImageOcrDimensions(750, 1)).toMatchObject({
      visualTokens: 1,
      worstCaseUsd: 0.005621,
    });
    expect(validateAndPriceImageOcrDimensions(751, 1)).toMatchObject({
      visualTokens: 2,
      worstCaseUsd: 0.005622,
    });
    const pixelBoundary = validateAndPriceImageOcrDimensions(5_000, 5_000);
    expect(pixelBoundary.pixels).toBe(25_000_000);
    expect(pixelBoundary.visualTokens).toBeLessThanOrEqual(1_568);
    expect(pixelBoundary.worstCaseUsd).toBeLessThanOrEqual(0.01);
    expect(() => validateAndPriceImageOcrDimensions(8_001, 1)).toThrow(/dimension/i);
    expect(() => validateAndPriceImageOcrDimensions(5_001, 5_000)).toThrow(/pixel/i);
    expect(() => validateAndPriceImageOcrDimensions(0, 1)).toThrow(/dimension/i);
  });
});

describe('persistent daily ledger', () => {
  test('the named image-import fence excludes a second OS process', async () => {
    const f = fixture();
    const lockRoot = join(f.ledgerDir, 'image-import-fence');
    const releasePath = join(f.ledgerDir, 'release-child');
    const code = `
      import { existsSync } from 'node:fs';
      import { withImageImportFence } from './src/core/image-import-fence.ts';
      await withImageImportFence(async () => {
        process.stdout.write('READY\\n');
        while (!existsSync(${JSON.stringify(releasePath)})) await Bun.sleep(10);
      }, { lockRoot: ${JSON.stringify(lockRoot)}, timeoutMs: 5000, pollMs: 10 });
    `;
    const child = Bun.spawn(['bun', '-e', code], {
      cwd: process.cwd(),
      stdout: 'pipe',
      stderr: 'pipe',
    });
    const reader = child.stdout.getReader();
    const ready = await reader.read();
    expect(new TextDecoder().decode(ready.value)).toContain('READY');

    let acquired = false;
    const waiter = withImageImportFence(async () => { acquired = true; }, {
      lockRoot,
      timeoutMs: 5000,
      pollMs: 10,
    });
    await Bun.sleep(40);
    expect(acquired).toBe(false);
    writeFileSync(releasePath, 'release');
    await waiter;
    expect(acquired).toBe(true);
    expect(await child.exited).toBe(0);
  });

  test('two OS-process waiters never reap or take over a pre-existing stale-owner fence', async () => {
    const f = fixture();
    const lockRoot = join(f.ledgerDir, 'image-import-fence');
    const ownerPath = join(lockRoot, 'owner.json');
    const staleOwner = `${JSON.stringify({
      pid: 2_147_483_647,
      owner_token: 'stale-owner-token',
      acquired_at: '2026-08-01T00:00:00.000Z',
    })}\n`;
    mkdirSync(lockRoot);
    writeFileSync(ownerPath, staleOwner);

    const spawnWaiter = (name: string) => {
      const enteredPath = join(f.ledgerDir, `${name}-entered`);
      const code = `
        import { writeFileSync } from 'node:fs';
        import { withImageImportFence } from './src/core/image-import-fence.ts';
        try {
          await withImageImportFence(async () => {
            writeFileSync(${JSON.stringify(enteredPath)}, new Date().toISOString());
            await Bun.sleep(250);
          }, { lockRoot: ${JSON.stringify(lockRoot)}, timeoutMs: 120, pollMs: 5 });
          process.stdout.write('ENTERED\\n');
        } catch (error) {
          process.stdout.write((error instanceof Error ? error.name : 'unknown') + '\\n');
        }
      `;
      return {
        enteredPath,
        child: Bun.spawn(['bun', '-e', code], {
          cwd: process.cwd(),
          stdout: 'pipe',
          stderr: 'pipe',
        }),
      };
    };

    const waiterA = spawnWaiter('waiter-a');
    const waiterB = spawnWaiter('waiter-b');
    const [stdoutA, stdoutB, exitA, exitB] = await Promise.all([
      new Response(waiterA.child.stdout).text(),
      new Response(waiterB.child.stdout).text(),
      waiterA.child.exited,
      waiterB.child.exited,
    ]);

    expect(exitA).toBe(0);
    expect(exitB).toBe(0);
    expect(stdoutA.trim()).toBe('ImageImportFenceError');
    expect(stdoutB.trim()).toBe('ImageImportFenceError');
    expect(existsSync(waiterA.enteredPath)).toBe(false);
    expect(existsSync(waiterB.enteredPath)).toBe(false);
    expect(readFileSync(ownerPath, 'utf8')).toBe(staleOwner);
  });

  test('persists reservations across same-day runs and rolls over on UTC date', () => {
    const f = fixture();
    const caps = validateImageOcrCaps({ maxImages: 2, maxUsd: 1, reserveUsdPerCall: 0.01 });
    const first = OcrBudgetLedger.acquire({ directory: f.ledgerDir, now: UTC_DAY_1, caps, manifestHash: 'a'.repeat(64) });
    first.reserve({ index: 0, sourceId: 'source-a', slug: 'images/a.png', ...reservationFields('/tmp/a.png') });
    first.close();
    const second = OcrBudgetLedger.acquire({ directory: f.ledgerDir, now: UTC_DAY_1, caps, manifestHash: 'b'.repeat(64) });
    expect(second.snapshot()).toMatchObject({ callsReserved: 1, usdReserved: 0.01 });
    second.reserve({ index: 0, sourceId: 'source-a', slug: 'images/b.png', ...reservationFields('/tmp/b.png') });
    expect(second.nextCapExceeded()).toBe('images');
    second.close();
    const nextDay = OcrBudgetLedger.acquire({ directory: f.ledgerDir, now: UTC_DAY_2, caps, manifestHash: 'c'.repeat(64) });
    expect(nextDay.snapshot()).toMatchObject({ utcDate: '2026-08-11', callsReserved: 0, usdReserved: 0 });
    nextDay.close();
  });

  test('samples and reserves after a fence-delayed UTC rollover, enforcing the new day ceilings', async () => {
    const f = fixture(3);
    const state = statefulImageEngine(f.root);
    f.engine = state.engine;
    const caps = validateImageOcrCaps({ maxImages: 2, maxUsd: 0.02, reserveUsdPerCall: 0.01 });
    const seeded = OcrBudgetLedger.acquire({
      directory: f.ledgerDir,
      now: UTC_DAY_2,
      caps,
      manifestHash: 'a'.repeat(64),
    });
    seeded.reserve({
      index: 0,
      sourceId: 'source-a',
      slug: 'seed/already-reserved.png',
      ...reservationFields('/tmp/already-reserved.png'),
    });
    seeded.close();

    const lockRoot = join(f.ledgerDir, 'image-import-fence');
    let releaseFence!: () => void;
    const fenceRelease = new Promise<void>(resolve => { releaseFence = resolve; });
    let signalFenceHeld!: () => void;
    const fenceHeld = new Promise<void>(resolve => { signalFenceHeld = resolve; });
    const holder = withImageImportFence(async () => {
      signalFenceHeld();
      await fenceRelease;
    }, { lockRoot, timeoutMs: 5_000, pollMs: 5 });
    await fenceHeld;

    let logicalNow = UTC_DAY_1;
    let clockCalls = 0;
    const reservationDates: string[] = [];
    const providerDates: string[] = [];
    const run = runCommand(f, caps, {
      attemptClock: () => {
        clockCalls++;
        return logicalNow;
      },
      afterReserve: (_entry, reservation) => { reservationDates.push(reservation.utcDate); },
      ocrProvider: async () => {
        providerDates.push(logicalNow.toISOString().slice(0, 10));
        return injectedReceipt('post-fence OCR');
      },
    });

    await Bun.sleep(40);
    const clockCallsBeforeRelease = clockCalls;
    const oldDateLedgerBeforeRelease = existsSync(join(f.ledgerDir, '2026-08-10.json'));
    logicalNow = UTC_DAY_2;
    releaseFence();
    await holder;
    const report = await run;

    expect(clockCallsBeforeRelease).toBe(0);
    expect(oldDateLedgerBeforeRelease).toBe(false);
    expect(reservationDates).toEqual(['2026-08-11']);
    expect(providerDates).toEqual(['2026-08-11']);
    expect(report).toMatchObject({
      status: 'cap_reached',
      utc_date: '2026-08-11',
      processed: 1,
      succeeded: 1,
      skipped: 2,
      daily_calls_before: 1,
      daily_calls_after: 2,
      daily_usd_reserved_before: 0.01,
      daily_usd_reserved_after: 0.02,
    });
    expect(existsSync(join(f.ledgerDir, '2026-08-10.json'))).toBe(false);
    expect(JSON.parse(readFileSync(join(f.ledgerDir, '2026-08-11.json'), 'utf8'))).toMatchObject({
      calls_reserved: 2,
      usd_reserved_micros: 20_000,
    });
  });

  test('samples a fresh clock and ledger date for every provider attempt', async () => {
    const f = fixture(2);
    const samples = [UTC_DAY_1, UTC_DAY_2];
    let clockCalls = 0;
    const report = await runCommand(f, { maxImages: 2, maxUsd: 1, reserveUsdPerCall: 0.01 }, {
      attemptClock: () => samples[clockCalls++],
      importEntry: reservedImport(async () => {}),
    });

    expect(clockCalls).toBe(2);
    expect(report).toMatchObject({
      status: 'completed',
      utc_date: '2026-08-11',
      processed: 2,
      succeeded: 2,
      daily_calls_before: 0,
      daily_calls_after: 1,
    });
    expect(JSON.parse(readFileSync(join(f.ledgerDir, '2026-08-10.json'), 'utf8')).calls_reserved).toBe(1);
    expect(JSON.parse(readFileSync(join(f.ledgerDir, '2026-08-11.json'), 'utf8')).calls_reserved).toBe(1);
  });

  test('a later attempt-date lock preserves earlier processed counters', async () => {
    const f = fixture(2);
    const caps = validateImageOcrCaps({ maxImages: 2, maxUsd: 1, reserveUsdPerCall: 0.01 });
    const held = OcrBudgetLedger.acquire({
      directory: f.ledgerDir,
      now: UTC_DAY_2,
      caps,
      manifestHash: 'f'.repeat(64),
    });
    let clockCalls = 0;
    let providerCalls = 0;
    try {
      const report = await runCommand(f, caps, {
        attemptClock: () => [UTC_DAY_1, UTC_DAY_2][clockCalls++],
        importEntry: reservedImport(async () => { providerCalls++; }),
      });
      expect(providerCalls).toBe(1);
      expect(report).toMatchObject({
        status: 'locked',
        terminal_error: 'budget_locked',
        utc_date: '2026-08-11',
        processed: 1,
        succeeded: 1,
        skipped: 1,
        daily_calls_before: null,
        daily_calls_after: null,
      });
    } finally {
      held.close();
    }
  });

  test('midnight crossing cannot bypass the 1000-image UTC-day cap', async () => {
    const f = fixture();
    const caps = validateImageOcrCaps({ maxImages: 1000, maxUsd: 10, reserveUsdPerCall: 0.01 });
    const ledger = OcrBudgetLedger.acquire({
      directory: f.ledgerDir,
      now: UTC_DAY_2,
      caps,
      manifestHash: 'd'.repeat(64),
    });
    for (let i = 0; i < 1000; i++) {
      ledger.reserve({
        index: i,
        sourceId: 'source-a',
        slug: `seed/images/${i}.png`,
        ...reservationFields(`/tmp/seed-${i}.png`),
      });
    }
    ledger.close();
    let providerCalls = 0;
    const report = await runCommand(f, caps, {
      attemptClock: () => UTC_DAY_2,
      importEntry: reservedImport(async () => { providerCalls++; }),
    });
    expect(providerCalls).toBe(0);
    expect(report).toMatchObject({
      status: 'cap_reached',
      utc_date: '2026-08-11',
      processed: 0,
      daily_calls_before: 1000,
      daily_calls_after: 1000,
      first_failing_entry: { code: 'image_cap' },
    });
  });

  test('midnight crossing cannot bypass the $10 UTC-day cap', async () => {
    const f = fixture();
    const caps = validateImageOcrCaps({ maxImages: 1000, maxUsd: 10, reserveUsdPerCall: 0.02 });
    const ledger = OcrBudgetLedger.acquire({
      directory: f.ledgerDir,
      now: UTC_DAY_2,
      caps,
      manifestHash: 'e'.repeat(64),
    });
    for (let i = 0; i < 500; i++) {
      ledger.reserve({
        index: i,
        sourceId: 'source-a',
        slug: `seed/usd/${i}.png`,
        ...reservationFields(`/tmp/seed-usd-${i}.png`),
      });
    }
    ledger.close();
    let providerCalls = 0;
    const report = await runCommand(f, caps, {
      attemptClock: () => UTC_DAY_2,
      importEntry: reservedImport(async () => { providerCalls++; }),
    });
    expect(providerCalls).toBe(0);
    expect(report).toMatchObject({
      status: 'cap_reached',
      utc_date: '2026-08-11',
      processed: 0,
      daily_usd_reserved_before: 10,
      daily_usd_reserved_after: 10,
      first_failing_entry: { code: 'usd_cap' },
    });
  });

  test('same-day caps may become stricter but can never be loosened later', () => {
    const f = fixture();
    const firstCaps = validateImageOcrCaps({ maxImages: 100, maxUsd: 1, reserveUsdPerCall: 0.01 });
    const first = OcrBudgetLedger.acquire({ directory: f.ledgerDir, now: UTC_DAY_1, caps: firstCaps, manifestHash: 'a'.repeat(64) });
    first.reserve({ index: 0, sourceId: 'source-a', slug: 'images/a.png', ...reservationFields('/tmp/a.png') });
    first.close();
    expect(() => OcrBudgetLedger.acquire({
      directory: f.ledgerDir,
      now: UTC_DAY_1,
      caps: { maxImages: 101, maxUsd: 1, reserveUsdPerCall: 0.01 },
      manifestHash: 'b'.repeat(64),
    })).toThrow(/never looser/);
    const stricter = OcrBudgetLedger.acquire({
      directory: f.ledgerDir,
      now: UTC_DAY_1,
      caps: { maxImages: 50, maxUsd: 0.5, reserveUsdPerCall: 0.02 },
      manifestHash: 'c'.repeat(64),
    });
    expect(stricter.snapshot().caps).toEqual({ maxImages: 50, maxUsd: 0.5, reserveUsdPerCall: 0.02 });
    stricter.close();
  });

  test('contradictory audit counters make the ledger ambiguous and fail closed', () => {
    const f = fixture();
    const caps = validateImageOcrCaps({ maxImages: 2, maxUsd: 1, reserveUsdPerCall: 0.01 });
    const first = OcrBudgetLedger.acquire({ directory: f.ledgerDir, now: UTC_DAY_1, caps, manifestHash: 'a'.repeat(64) });
    first.reserve({ index: 0, sourceId: 'source-a', slug: 'images/a.png', ...reservationFields('/tmp/a.png') });
    first.close();
    const ledgerPath = join(f.ledgerDir, '2026-08-10.json');
    const corrupt = JSON.parse(readFileSync(ledgerPath, 'utf8'));
    corrupt.calls_reserved = 0;
    writeFileSync(ledgerPath, `${JSON.stringify(corrupt)}\n`);
    expect(() => OcrBudgetLedger.acquire({ directory: f.ledgerDir, now: UTC_DAY_1, caps, manifestHash: 'b'.repeat(64) }))
      .toThrow(/ambiguous/);
  });

  test('an internally inconsistent provider receipt makes the audit ambiguous', () => {
    const f = fixture();
    const caps = validateImageOcrCaps({ maxImages: 2, maxUsd: 1, reserveUsdPerCall: 0.01 });
    const first = OcrBudgetLedger.acquire({ directory: f.ledgerDir, now: UTC_DAY_1, caps, manifestHash: 'a'.repeat(64) });
    const reservation = first.reserve({
      index: 0,
      sourceId: 'source-a',
      slug: 'images/a.png',
      ...reservationFields('/tmp/a.png'),
    });
    first.recordTransportAttempt(reservation);
    first.recordProviderReceipt(reservation, injectedReceipt());
    first.close();
    const ledgerPath = join(f.ledgerDir, '2026-08-10.json');
    const corrupt = JSON.parse(readFileSync(ledgerPath, 'utf8'));
    corrupt.audit[0].provider_receipt.model = 'unexpected-model';
    writeFileSync(ledgerPath, `${JSON.stringify(corrupt)}\n`);
    expect(() => OcrBudgetLedger.acquire({ directory: f.ledgerDir, now: UTC_DAY_1, caps, manifestHash: 'b'.repeat(64) }))
      .toThrow(/ambiguous/);
  });

  test('an existing or ambiguous lock fails closed and is never broken', () => {
    const f = fixture();
    const caps = validateImageOcrCaps({ maxImages: 1, maxUsd: 1, reserveUsdPerCall: 0.01 });
    const first = OcrBudgetLedger.acquire({ directory: f.ledgerDir, now: UTC_DAY_1, caps, manifestHash: 'a'.repeat(64) });
    expect(() => OcrBudgetLedger.acquire({ directory: f.ledgerDir, now: UTC_DAY_1, caps, manifestHash: 'b'.repeat(64) }))
      .toThrow(OcrBudgetLockError);
    first.close();
  });

  test('a lock held by a second process fails closed', async () => {
    const f = fixture();
    const code = `
      import { OcrBudgetLedger } from './src/core/image-ocr-budget.ts';
      const ledger = OcrBudgetLedger.acquire({
        directory: ${JSON.stringify(f.ledgerDir)},
        now: new Date('2026-08-10T23:59:59.000Z'),
        caps: { maxImages: 1, maxUsd: 1, reserveUsdPerCall: 0.01 },
        manifestHash: 'a'.repeat(64),
      });
      process.stdout.write('READY\\n');
      await new Promise(() => {});
    `;
    const child = Bun.spawn(['bun', '-e', code], { cwd: process.cwd(), stdout: 'pipe', stderr: 'pipe' });
    const reader = child.stdout.getReader();
    const ready = await reader.read();
    expect(new TextDecoder().decode(ready.value)).toContain('READY');
    const caps = validateImageOcrCaps({ maxImages: 1, maxUsd: 1, reserveUsdPerCall: 0.01 });
    expect(() => OcrBudgetLedger.acquire({ directory: f.ledgerDir, now: UTC_DAY_1, caps, manifestHash: 'b'.repeat(64) }))
      .toThrow(OcrBudgetLockError);
    child.kill();
    await child.exited;
  });

  test('a replaced budget lock cannot be unlinked by its old owner or admit two reservations', async () => {
    const f = fixture();
    const lockPath = join(f.ledgerDir, '2026-08-10.lock');
    const aReady = join(f.ledgerDir, 'a-ready');
    const aClose = join(f.ledgerDir, 'a-close');
    const aClosed = join(f.ledgerDir, 'a-closed');
    const bReady = join(f.ledgerDir, 'b-ready');
    const bReserve = join(f.ledgerDir, 'b-reserve');
    const bReserved = join(f.ledgerDir, 'b-reserved');
    const bClose = join(f.ledgerDir, 'b-close');

    const ownerCode = (name: 'a' | 'b') => `
      import { existsSync, writeFileSync } from 'node:fs';
      import { OcrBudgetLedger } from './src/core/image-ocr-budget.ts';
      const ledger = OcrBudgetLedger.acquire({
        directory: ${JSON.stringify(f.ledgerDir)},
        now: new Date('2026-08-10T23:59:59.000Z'),
        caps: { maxImages: 1, maxUsd: 0.01, reserveUsdPerCall: 0.01 },
        manifestHash: '${name}'.repeat(64),
      });
      writeFileSync(${JSON.stringify(name === 'a' ? aReady : bReady)}, 'ready');
      ${name === 'b' ? `
        while (!existsSync(${JSON.stringify(bReserve)})) await Bun.sleep(10);
        ledger.reserve({
          index: 0,
          sourceId: 'source-a',
          slug: 'images/b.png',
          filePath: '/tmp/b.png',
          registeredRoot: '/tmp',
          sha256: 'b'.repeat(64),
        });
        writeFileSync(${JSON.stringify(bReserved)}, 'reserved');
      ` : ''}
      while (!existsSync(${JSON.stringify(name === 'a' ? aClose : bClose)})) await Bun.sleep(10);
      try { ledger.close(); } catch {}
      ${name === 'a' ? `writeFileSync(${JSON.stringify(aClosed)}, 'closed');` : ''}
    `;

    const ownerA = Bun.spawn(['bun', '-e', ownerCode('a')], {
      cwd: process.cwd(), stdout: 'pipe', stderr: 'pipe',
    });
    await waitForFile(aReady);
    const [ownerMetadataFile] = readdirSync(lockPath);
    expect(JSON.parse(readFileSync(join(lockPath, ownerMetadataFile), 'utf8'))).toMatchObject({
      pid: expect.any(Number),
      process_started_at_ms: expect.any(Number),
      hostname: expect.any(String),
      owner_token: expect.stringMatching(/^[a-f0-9]{64}$/),
    });

    // Simulate an operator incorrectly removing a lock believed to be stale,
    // then let a new process create a replacement at the same pathname.
    rmSync(lockPath, { recursive: true });
    const ownerB = Bun.spawn(['bun', '-e', ownerCode('b')], {
      cwd: process.cwd(), stdout: 'pipe', stderr: 'pipe',
    });
    await waitForFile(bReady);

    writeFileSync(aClose, 'close');
    await waitForFile(aClosed);

    const ownerCCode = `
      import { OcrBudgetLedger, OcrBudgetLockError } from './src/core/image-ocr-budget.ts';
      try {
        const ledger = OcrBudgetLedger.acquire({
          directory: ${JSON.stringify(f.ledgerDir)},
          now: new Date('2026-08-10T23:59:59.000Z'),
          caps: { maxImages: 1, maxUsd: 0.01, reserveUsdPerCall: 0.01 },
          manifestHash: 'c'.repeat(64),
        });
        ledger.reserve({
          index: 0,
          sourceId: 'source-a',
          slug: 'images/c.png',
          filePath: '/tmp/c.png',
          registeredRoot: '/tmp',
          sha256: 'c'.repeat(64),
        });
        process.stdout.write('RESERVED\\n');
        ledger.close();
      } catch (error) {
        process.stdout.write(error instanceof OcrBudgetLockError ? 'LOCKED\\n' : 'ERROR\\n');
      }
    `;
    const ownerC = Bun.spawn(['bun', '-e', ownerCCode], {
      cwd: process.cwd(), stdout: 'pipe', stderr: 'pipe',
    });
    const [ownerCOutput, ownerCExit] = await Promise.all([
      new Response(ownerC.stdout).text(),
      ownerC.exited,
    ]);

    writeFileSync(bReserve, 'reserve');
    await waitForFile(bReserved);
    writeFileSync(bClose, 'close');
    const [ownerAExit, ownerBExit] = await Promise.all([ownerA.exited, ownerB.exited]);

    expect(ownerAExit).toBe(0);
    expect(ownerBExit).toBe(0);
    expect(ownerCExit).toBe(0);
    expect(ownerCOutput.trim()).toBe('LOCKED');
    expect(JSON.parse(readFileSync(join(f.ledgerDir, '2026-08-10.json'), 'utf8'))).toMatchObject({
      calls_reserved: 1,
      audit: [{ slug: 'images/b.png' }],
    });
  });
});

describe('bounded run behavior', () => {
  test('pins the raw paid boundary and its current official-price cost proof', async () => {
    expect(IMAGE_OCR_POLICY_MODEL).toBe('claude-haiku-4-5-20251001');
    expect(IMAGE_OCR_POLICY_MAX_OUTPUT_TOKENS).toBe(1024);
    expect(BOUNDED_IMAGE_OCR_ENDPOINT).toBe('https://api.anthropic.com/v1/messages');
    expect(BOUNDED_IMAGE_OCR_PROMPT.length).toBeGreaterThan(0);
    expect(IMAGE_OCR_INPUT_USD_PER_MTOK).toBe(1);
    expect(IMAGE_OCR_OUTPUT_USD_PER_MTOK).toBe(5);
    expect(IMAGE_OCR_MAX_VISUAL_TOKENS).toBe(1568);
    expect(IMAGE_OCR_NONVISUAL_INPUT_TOKEN_ALLOWANCE).toBe(500);
    expect(IMAGE_OCR_CURRENT_WORST_CASE_USD).toBe(0.007188);
    expect(IMAGE_OCR_CURRENT_WORST_CASE_USD).toBeLessThanOrEqual(0.01);

    const commandSource = readFileSync('src/commands/image-ocr-run.ts', 'utf8');
    const gatewaySource = readFileSync('src/core/ai/gateway.ts', 'utf8');
    const gatewayModule = await import('../src/core/ai/gateway.ts');
    expect(gatewayModule).not.toHaveProperty('generateOcrText');
    expect(gatewaySource).not.toContain('export async function generateOcrText');
    expect(commandSource).not.toContain("import('../core/ai/gateway.ts')");
    expect(commandSource).toContain("from '../core/image-ocr-provider.ts'");
    expect(commandSource).not.toContain("isAvailable('expansion')");

    const importerSource = readFileSync('src/core/import-file.ts', 'utf8');
    const boundedAdapter = importerSource.slice(
      importerSource.indexOf('export async function importImageFileWithBoundedOcrText'),
      importerSource.indexOf('async function importImageFileInternal'),
    );
    expect(boundedAdapter).toContain('noEmbed: true');
  });

  test('serializes one exact canonical Anthropic request and performs one fetch with no retry', async () => {
    const body = buildBoundedImageOcrRequestBody(Buffer.from('image-bytes'), 'image/png');
    const calls: Array<{ url: string; init: RequestInit }> = [];
    let transportAttempts = 0;
    const receipt = await requestBoundedImageOcr({
      apiKey: 'test-key',
      body,
      maxInputTokens: 600,
      reservedUsd: 0.01,
      onTransportAttempt: () => { transportAttempts++; },
      fetchImpl: async (url, init) => {
        calls.push({ url: String(url), init: init ?? {} });
        return new Response(JSON.stringify(anthropicMessageResponse()), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        });
      },
    });

    expect(receipt).toEqual(injectedReceipt());
    expect(transportAttempts).toBe(1);
    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe(BOUNDED_IMAGE_OCR_ENDPOINT);
    expect(calls[0].init).toEqual({
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'content-length': String(body.length),
        'anthropic-version': '2023-06-01',
        'x-api-key': expect.any(String),
      },
      body: body as unknown as BodyInit,
      redirect: 'error',
      signal: expect.anything(),
    });
    expect(JSON.parse(body.toString('utf8'))).toEqual({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 1024,
      messages: [{
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/png', data: Buffer.from('image-bytes').toString('base64') } },
          { type: 'text', text: BOUNDED_IMAGE_OCR_PROMPT },
        ],
      }],
    });
  });

  test('rejects unexpected, truncated, ambiguous, or unbillable provider receipts', async () => {
    const body = buildBoundedImageOcrRequestBody(Buffer.from('image-bytes'), 'image/png');
    const invalidResponses = [
      anthropicMessageResponse({ type: 'error' }),
      anthropicMessageResponse({ model: 'claude-other-model' }),
      anthropicMessageResponse({ stop_reason: 'max_tokens' }),
      anthropicMessageResponse({ stop_reason: null }),
      anthropicMessageResponse({ id: '' }),
      anthropicMessageResponse({ content: [
        { type: 'text', text: 'first' },
        { type: 'text', text: 'second' },
      ] }),
      anthropicMessageResponse({ content: [{ type: 'text', text: '   ' }] }),
      anthropicMessageResponse({ usage: undefined }),
      anthropicMessageResponse({ usage: {
        input_tokens: -1,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        output_tokens: 1,
      } }),
      anthropicMessageResponse({ usage: {
        input_tokens: 1.5,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        output_tokens: 1,
      } }),
      anthropicMessageResponse({ usage: {
        input_tokens: 1,
        cache_creation_input_tokens: 1,
        cache_read_input_tokens: 0,
        output_tokens: 1,
      } }),
    ];

    for (const responseBody of invalidResponses) {
      await expect(requestBoundedImageOcr({
        apiKey: 'test-key',
        body,
        maxInputTokens: 600,
        reservedUsd: 0.01,
        fetchImpl: async () => new Response(JSON.stringify(responseBody), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
      })).rejects.toThrow();
    }
  });

  test('enforces the exact request input, output, and durable reservation envelope', () => {
    const exactBoundary = parseBoundedImageOcrReceipt(anthropicMessageResponse({
      usage: {
        input_tokens: 501,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        output_tokens: IMAGE_OCR_POLICY_MAX_OUTPUT_TOKENS,
      },
    }), {
      maxInputTokens: 501,
      reservedUsd: 0.01,
    });
    expect(exactBoundary).toMatchObject({
      inputTokens: 501,
      outputTokens: IMAGE_OCR_POLICY_MAX_OUTPUT_TOKENS,
      actualUsd: 0.005621,
    });

    expect(() => parseBoundedImageOcrReceipt(anthropicMessageResponse({
      usage: {
        input_tokens: 502,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        output_tokens: 0,
      },
    }), { maxInputTokens: 501, reservedUsd: 0.01 })).toThrow();

    expect(() => parseBoundedImageOcrReceipt(anthropicMessageResponse({
      usage: {
        input_tokens: 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        output_tokens: IMAGE_OCR_POLICY_MAX_OUTPUT_TOKENS + 1,
      },
    }), { maxInputTokens: 501, reservedUsd: 0.01 })).toThrow();

    expect(() => parseBoundedImageOcrReceipt(anthropicMessageResponse(), {
      maxInputTokens: 501,
      reservedUsd: 0.000149,
    })).toThrow();
  });

  test('accounts an over-limit received response without accepting or persisting it', async () => {
    const f = fixture(2);
    const state = statefulImageEngine(f.root);
    f.engine = state.engine;
    let providerCalls = 0;
    const overLimit = injectedReceiptWithUsage(
      1_000_000,
      IMAGE_OCR_POLICY_MAX_OUTPUT_TOKENS + 1,
      'MUST NOT PERSIST',
    );

    const report = await runCommand(
      f,
      { maxImages: 2, maxUsd: 1, reserveUsdPerCall: 0.01 },
      {
        ocrProvider: async () => {
          providerCalls++;
          return overLimit;
        },
      },
    );

    expect(providerCalls).toBe(1);
    expect(report).toMatchObject({
      status: 'failed',
      requested: 2,
      processed: 1,
      failed: 1,
      skipped: 1,
      reservations: 1,
      provider_attempts: 1,
      successful_provider_receipts: 0,
      observed_input_tokens: 1_000_000,
      observed_cache_creation_input_tokens: 0,
      observed_cache_read_input_tokens: 0,
      observed_output_tokens: IMAGE_OCR_POLICY_MAX_OUTPUT_TOKENS + 1,
      observed_usd: 1.005125,
      persisted_imports: 0,
      failures: 1,
    });
    expect(state.pages.size).toBe(0);
    expect(state.chunks.size).toBe(0);
    const audit = JSON.parse(readFileSync(join(f.ledgerDir, '2026-08-10.json'), 'utf8')).audit[0];
    expect(audit).toMatchObject({
      reserved_usd_micros: 10_000,
      max_input_tokens: 501,
      state: 'invalid_provider_observation',
      transport_attempted: true,
      provider_receipt: null,
      persistence_succeeded: false,
      outcome: 'over_limit',
      failure_stage: 'provider_receipt',
      invalid_provider_observation: {
        outcome: 'over_limit',
        request_id: 'msg_injected_123',
        model: IMAGE_OCR_POLICY_MODEL,
        stop_reason: 'end_turn',
        input_tokens: 1_000_000,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        output_tokens: IMAGE_OCR_POLICY_MAX_OUTPUT_TOKENS + 1,
        actual_usd_micros: 1_005_125,
      },
    });
    expect(JSON.stringify(audit)).not.toContain('MUST NOT PERSIST');
  });

  test('an over-reservation provider observation consumes same-day USD capacity before a later run', async () => {
    const f = fixture();
    const state = statefulImageEngine(f.root);
    f.engine = state.engine;
    const caps = { maxImages: 2, maxUsd: 0.02, reserveUsdPerCall: 0.01 };
    let providerCalls = 0;

    const first = await runCommand(f, caps, {
      ocrProvider: async () => {
        providerCalls++;
        return injectedReceiptWithUsage(20_000, 0, 'MUST NOT PERSIST');
      },
    });
    expect(first).toMatchObject({ status: 'failed' });
    expect(providerCalls).toBe(1);

    const second = await runCommand(f, caps, {
      ocrProvider: async () => {
        providerCalls++;
        return injectedReceipt('MUST NOT RUN');
      },
    });

    expect(providerCalls).toBe(1);
    expect(second).toMatchObject({
      status: 'cap_reached',
      processed: 0,
      succeeded: 0,
      skipped: 1,
      daily_calls_before: 1,
      daily_calls_after: 1,
      daily_usd_reserved_before: 0.02,
      daily_usd_reserved_after: 0.02,
      first_failing_entry: { index: 0, code: 'usd_cap' },
    });
    expect(state.pages.size).toBe(0);
    expect(state.chunks.size).toBe(0);
    expect(JSON.parse(readFileSync(join(f.ledgerDir, '2026-08-10.json'), 'utf8'))).toMatchObject({
      calls_reserved: 1,
      usd_reserved_micros: 20_000,
    });
  });

  test('native fetch rejects redirects without sending the OCR request to the redirect target', async () => {
    const paths: string[] = [];
    const server = createServer((request, response) => {
      paths.push(request.url ?? '');
      request.resume();
      if (request.url === '/start') {
        response.writeHead(307, { location: '/final' });
        response.end();
        return;
      }
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ content: [{ type: 'text', text: 'must not be reached' }] }));
    });
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject);
      server.listen(0, '127.0.0.1', resolve);
    });
    const address = server.address();
    if (!address || typeof address === 'string') throw new Error('Expected an ephemeral TCP server');

    try {
      const body = buildBoundedImageOcrRequestBody(Buffer.from('image-bytes'), 'image/png');
      await expect(requestBoundedImageOcr({
        apiKey: 'test-key',
        body,
        maxInputTokens: 600,
        reservedUsd: 0.01,
        fetchImpl: (_url, init) => fetch(`http://127.0.0.1:${address.port}/start`, init),
      })).rejects.toThrow();
      expect(paths).toEqual(['/start']);
    } finally {
      await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
    }
  });

  test('rejects a configured Anthropic provider base URL before any reservation', async () => {
    const f = fixture();
    const home = mkdtempSync(join(tmpdir(), 'gbrain-bounded-ocr-config-'));
    writeFileSync(join(home, 'config.json'), JSON.stringify({
      engine: 'pglite',
      database_path: join(home, 'brain.pglite'),
      anthropic_api_key: 'file-key',
      provider_base_urls: { anthropic: 'https://proxy.example.invalid/v1' },
    }));
    f.engine = {
      ...f.engine,
      getConfig: async () => null,
      listConfigKeys: async () => [],
    } as unknown as BrainEngine;

    await withEnv({
      GBRAIN_HOME: home,
      ANTHROPIC_API_KEY: undefined,
      ANTHROPIC_BASE_URL: undefined,
    }, async () => {
      const report = await runCommand(f);
      expect(report).toMatchObject({ status: 'rejected', processed: 0 });
      expect(() => readFileSync(join(f.ledgerDir, '2026-08-10.json'), 'utf8')).toThrow();
    });
  });

  test('fails closed when a provider adapter returns without invoking beforeProviderAttempt', async () => {
    const f = fixture();
    let adapterCalls = 0;
    const report = await runCommand(f, undefined, {
      importEntry: async () => { adapterCalls++; },
    });

    expect(adapterCalls).toBe(1);
    expect(report).toMatchObject({
      status: 'failed',
      processed: 0,
      succeeded: 0,
      failed: 1,
      first_failing_entry: { code: 'import_failed' },
    });
    expect(existsSync(join(f.ledgerDir, '2026-08-10.json'))).toBe(false);
  });

  test('fails closed after a second beforeProviderAttempt invocation without double reservation', async () => {
    const f = fixture();
    let completedCallbacks = 0;
    const report = await runCommand(f, undefined, {
      importEntry: async (_entry, callbackValue) => {
        const beforeProviderAttempt = callbackValue as unknown as () => Promise<unknown>;
        await beforeProviderAttempt();
        completedCallbacks++;
        await beforeProviderAttempt();
        completedCallbacks++;
      },
    });

    expect(completedCallbacks).toBe(1);
    expect(report).toMatchObject({
      status: 'failed',
      processed: 1,
      succeeded: 0,
      failed: 1,
      first_failing_entry: { code: 'import_failed' },
    });
    expect(JSON.parse(readFileSync(join(f.ledgerDir, '2026-08-10.json'), 'utf8')).calls_reserved).toBe(1);
  });

  test('reserves before an injected crash and preserves the audit record', async () => {
    const f = fixture();
    await expect(runCommand(f, undefined, {
      afterReserve: () => { throw new Error('injected crash'); },
      importEntry: reservedImport(async () => { throw new Error('provider must not run'); }),
    })).resolves.toMatchObject({ status: 'failed', failed: 1 });
    const ledger = JSON.parse(readFileSync(join(f.ledgerDir, '2026-08-10.json'), 'utf8'));
    expect(ledger.calls_reserved).toBe(1);
    expect(ledger.usd_reserved_micros).toBe(10_000);
    expect(ledger.audit).toHaveLength(1);
  });

  test('preserves processed counters when ledger close fails afterward', async () => {
    const f = fixture();
    const originalClose = OcrBudgetLedger.prototype.close;
    OcrBudgetLedger.prototype.close = function closeThenFail() {
      originalClose.call(this);
      throw new Error('injected ledger close failure');
    };
    try {
      const report = await runCommand(f, undefined, { importEntry: reservedImport(async () => {}) });
      expect(report).toMatchObject({
        status: 'failed',
        terminal_error: 'run_rejected',
        processed: 1,
        succeeded: 1,
        failed: 0,
        skipped: 0,
        daily_calls_before: 0,
        daily_calls_after: 1,
        daily_usd_reserved_before: 0,
        daily_usd_reserved_after: 0.01,
      });
    } finally {
      OcrBudgetLedger.prototype.close = originalClose;
    }
  });

  test('the paid provider boundary is internal and reachable only from the confirmed command', () => {
    const commandSource = readFileSync('src/commands/image-ocr-run.ts', 'utf8');
    const importerSource = readFileSync('src/core/import-file.ts', 'utf8');
    const providerSource = readFileSync('src/core/image-ocr-provider.ts', 'utf8');
    const packageJson = JSON.parse(readFileSync('package.json', 'utf8'));
    expect(commandSource).not.toContain('export async function executeConfirmedImageOcrRun');
    expect(commandSource).toContain('requestBoundedImageOcr({');
    expect(providerSource).toContain("'https://api.anthropic.com/v1/messages'");
    expect(providerSource).not.toContain('generateText(');
    expect(Object.values(packageJson.exports)).not.toContain('./src/core/image-ocr-provider.ts');
    expect(importerSource).not.toContain('generateOcrText');
    expect(importerSource).not.toContain("import('./ai/gateway.ts')");
  });

  test('allows exact image and USD boundaries, then stops before the next call', async () => {
    const f = fixture(3);
    const calls: string[] = [];
    const report = await runCommand(f, { maxImages: 2, maxUsd: 0.02, reserveUsdPerCall: 0.01 }, {
      importEntry: reservedImport(async (entry) => { calls.push(entry.slug); }),
    });
    expect(calls).toHaveLength(2);
    expect(report).toMatchObject({
      requested: 3,
      processed: 2,
      succeeded: 2,
      failed: 0,
      skipped: 1,
      status: 'cap_reached',
      daily_calls_before: 0,
      daily_calls_after: 2,
      daily_usd_reserved_before: 0,
      daily_usd_reserved_after: 0.02,
      first_failing_entry: { index: 2, source_id: 'source-a', slug: 'images/0002.png', code: 'image_cap' },
    });
  });

  test('USD cap independently stops the next call', async () => {
    const f = fixture(3);
    let calls = 0;
    const report = await runCommand(f, { maxImages: 1000, maxUsd: 0.02, reserveUsdPerCall: 0.01 }, {
      importEntry: reservedImport(async () => { calls++; }),
    });
    expect(calls).toBe(2);
    expect(report?.first_failing_entry?.code).toBe('usd_cap');
  });

  test('enforces the literal 1000-call and $10 daily outer boundary', () => {
    const f = fixture();
    const caps = validateImageOcrCaps({ maxImages: 1000, maxUsd: 10, reserveUsdPerCall: 0.01 });
    const ledger = OcrBudgetLedger.acquire({ directory: f.ledgerDir, now: UTC_DAY_1, caps, manifestHash: 'd'.repeat(64) });
    for (let index = 0; index < 1000; index++) {
      ledger.reserve({
        index,
        sourceId: 'source-a',
        slug: `images/${index}.png`,
        ...reservationFields(`/tmp/${index}.png`, 'a'.repeat(64), '/tmp'),
      });
    }
    expect(ledger.snapshot()).toMatchObject({ callsReserved: 1000, usdReserved: 10 });
    expect(ledger.nextCapExceeded()).toBe('images');
    ledger.close();
  });

  test('post-reservation file mutation consumes the reservation and makes zero transport attempts', async () => {
    const f = fixture();
    let providerCalls = 0;
    let providerBytes = '';
    const engine = {
      ...f.engine,
      getPage: async () => null,
      getConfig: async () => '0',
      setConfig: async () => {},
    } as unknown as BrainEngine;
    f.engine = engine;
    const report = await runCommand(f, undefined, {
      afterReserve: (entry) => { writeFileSync(entry.file_path, 'changed-after-preflight'); },
      ocrProvider: async (bytes) => {
        providerCalls++;
        providerBytes = bytes.toString('utf8');
        return injectedReceipt('text');
      },
    });
    expect(providerCalls).toBe(0);
    expect(providerBytes).toBe('');
    expect(report).toMatchObject({ status: 'failed', processed: 1, succeeded: 0, failed: 1 });
  });

  test('post-reservation registered-root replacement consumes the reservation and makes zero transport attempts', async () => {
    const f = fixture();
    let providerCalls = 0;
    f.engine = {
      ...f.engine,
      getPage: async () => null,
      getConfig: async () => '0',
      setConfig: async () => {},
    } as unknown as BrainEngine;
    const report = await runCommand(f, undefined, {
      afterReserve: (entry) => {
        const movedRoot = `${f.root}-moved`;
        renameSync(f.root, movedRoot);
        mkdirSync(join(f.root, 'images'), { recursive: true });
        writeFileSync(entry.file_path, readFileSync(join(movedRoot, 'images/0000.png')));
      },
      ocrProvider: async () => { providerCalls++; return injectedReceipt('text'); },
    });
    expect(providerCalls).toBe(0);
    expect(report).toMatchObject({ status: 'failed', processed: 1, succeeded: 0, failed: 1 });
  });

  test('reconciles a valid provider receipt through durable persistence', async () => {
    const f = fixture();
    const state = statefulImageEngine(f.root);
    f.engine = state.engine;

    const report = await runCommand(f, undefined, {
      ocrProvider: async () => injectedReceipt('PAID OCR RESULT'),
    });

    expect(report).toMatchObject({
      status: 'completed',
      reservations: 1,
      provider_attempts: 1,
      successful_provider_receipts: 1,
      observed_input_tokens: 100,
      observed_cache_creation_input_tokens: 0,
      observed_cache_read_input_tokens: 0,
      observed_output_tokens: 10,
      observed_usd: 0.00015,
      persisted_imports: 1,
      failures: 0,
    });
    const audit = JSON.parse(readFileSync(join(f.ledgerDir, '2026-08-10.json'), 'utf8')).audit[0];
    expect(audit).toMatchObject({
      state: 'persisted',
      transport_attempted: true,
      persistence_succeeded: true,
      outcome: 'persisted',
      failure_stage: null,
      provider_receipt: {
        request_id: 'msg_injected_123',
        model: IMAGE_OCR_POLICY_MODEL,
        stop_reason: 'end_turn',
        input_tokens: 100,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        output_tokens: 10,
        actual_usd_micros: 150,
      },
    });
  });

  test('keeps a valid paid receipt but records persistence failure without importing text', async () => {
    const f = fixture();
    const state = statefulImageEngine(f.root);
    f.engine = {
      ...state.engine,
      transaction: async () => { throw new Error('injected persistence failure'); },
    } as unknown as BrainEngine;

    const report = await runCommand(f, undefined, {
      ocrProvider: async () => injectedReceipt('MUST NOT PERSIST'),
    });

    expect(report).toMatchObject({
      status: 'failed',
      reservations: 1,
      provider_attempts: 1,
      successful_provider_receipts: 1,
      observed_input_tokens: 100,
      observed_output_tokens: 10,
      observed_usd: 0.00015,
      persisted_imports: 0,
      failures: 1,
    });
    expect(state.pages.size).toBe(0);
    expect(state.chunks.size).toBe(0);
    expect(JSON.parse(readFileSync(join(f.ledgerDir, '2026-08-10.json'), 'utf8')).audit[0]).toMatchObject({
      state: 'failed',
      transport_attempted: true,
      persistence_succeeded: false,
      outcome: 'failed',
      failure_stage: 'persistence',
      provider_receipt: { request_id: 'msg_injected_123', actual_usd_micros: 150 },
    });
  });

  test('routine image import waits behind the paid attempt through OCR persistence', async () => {
    const f = fixture();
    const state = statefulImageEngine(f.root);
    f.engine = state.engine;
    const entry = JSON.parse(readFileSync(f.manifestPath, 'utf8').trim()) as {
      file_path: string;
      slug: string;
    };
    let releaseProvider!: () => void;
    const providerRelease = new Promise<void>(resolve => { releaseProvider = resolve; });
    let signalProviderStarted!: () => void;
    const providerStarted = new Promise<void>(resolve => { signalProviderStarted = resolve; });

    const bounded = runCommand(f, undefined, {
      ocrProvider: async () => {
        signalProviderStarted();
        await providerRelease;
        return injectedReceipt('PAID OCR RESULT');
      },
    });
    await providerStarted;

    let routineSettled = false;
    const routine = importImageFile(state.engine, entry.file_path, entry.slug, {
      noEmbed: true,
      sourceId: 'source-a',
      imageImportFenceRoot: join(f.ledgerDir, 'image-import-fence'),
    }).then(result => {
      routineSettled = true;
      return result;
    });
    await Bun.sleep(30);
    expect(routineSettled).toBe(false);

    releaseProvider();
    expect(await bounded).toMatchObject({ status: 'completed', succeeded: 1 });
    expect(await routine).toMatchObject({ status: 'skipped' });
    expect(state.chunks.get(entry.slug)?.[0]?.chunk_text).toBe('PAID OCR RESULT');
  });

  test('a generic page writer during provider latency wins the persistence compare-and-set', async () => {
    const f = fixture();
    const engine = new PGLiteEngine();
    await engine.connect({ type: 'pglite' } as never);
    await engine.initSchema();
    await engine.executeRaw(
      `INSERT INTO sources (id, name, local_path) VALUES ($1, $2, $3)`,
      ['source-a', 'Source A', f.root],
    );
    f.engine = engine;
    const entry = JSON.parse(readFileSync(f.manifestPath, 'utf8').trim()) as { slug: string };
    let signalProviderStarted!: () => void;
    const providerStarted = new Promise<void>(resolve => { signalProviderStarted = resolve; });
    let releaseProvider!: () => void;
    const providerRelease = new Promise<void>(resolve => { releaseProvider = resolve; });

    try {
      const bounded = runCommand(f, undefined, {
        ocrProvider: async () => {
          signalProviderStarted();
          await providerRelease;
          return injectedReceipt('PAID RESULT MUST NOT OVERWRITE');
        },
      });
      await providerStarted;
      await engine.putPage(entry.slug, {
        type: 'image',
        page_kind: 'image',
        title: 'generic-writer.png',
        content_hash: 'b'.repeat(64),
        compiled_truth: 'GENERIC WRITER CONTENT',
      }, { sourceId: 'source-a' });
      releaseProvider();

      expect(await bounded).toMatchObject({
        status: 'failed',
        reservations: 1,
        provider_attempts: 1,
        successful_provider_receipts: 1,
        persisted_imports: 0,
        failures: 1,
      });
      expect(await engine.getPage(entry.slug, { sourceId: 'source-a' })).toMatchObject({
        content_hash: 'b'.repeat(64),
        compiled_truth: 'GENERIC WRITER CONTENT',
      });
      expect(JSON.parse(readFileSync(join(f.ledgerDir, '2026-08-10.json'), 'utf8')).audit[0]).toMatchObject({
        state: 'failed',
        transport_attempted: true,
        persistence_succeeded: false,
        outcome: 'failed',
        failure_stage: 'persistence',
        provider_receipt: { request_id: 'msg_injected_123', actual_usd_micros: 150 },
      });
    } finally {
      releaseProvider();
      await engine.disconnect();
    }
  });

  test('no source/page await point exists between final revalidation and provider dispatch', async () => {
    const f = fixture();
    const state = statefulImageEngine(f.root);
    const changedRoot = mkdtempSync(join(tmpdir(), 'gbrain-bounded-ocr-post-check-root-'));
    let pageReads = 0;
    let mutateOnConfigRead = false;
    f.engine = new Proxy(state.engine, {
      get(target, prop, receiver) {
        if (prop === 'getPage') {
          return async (...args: unknown[]) => {
            pageReads++;
            const result = await (target.getPage as (...inner: unknown[]) => Promise<unknown>)(...args);
            if (pageReads >= 2) mutateOnConfigRead = true;
            return result;
          };
        }
        if (prop === 'getConfig') {
          return async () => {
            if (mutateOnConfigRead) {
              state.registeredRoot = changedRoot;
              state.syntheticPageDrift = true;
            }
            return '0';
          };
        }
        return Reflect.get(target, prop, receiver);
      },
    }) as BrainEngine;
    let providerSawStableSourceAndPage = false;
    const report = await runCommand(f, undefined, {
      ocrProvider: async () => {
        providerSawStableSourceAndPage =
          state.registeredRoot === f.root && state.syntheticPageDrift === false;
        return injectedReceipt('stable OCR');
      },
    });
    expect(report).toMatchObject({ status: 'completed', succeeded: 1 });
    expect(providerSawStableSourceAndPage).toBe(true);
  });

  test('source registration drift after preflight cannot reach the provider', async () => {
    const f = fixture();
    const changedRoot = mkdtempSync(join(tmpdir(), 'gbrain-bounded-ocr-moved-source-'));
    let sourceReads = 0;
    let providerCalls = 0;
    f.engine = {
      ...f.engine,
      listAllSources: async () => [{
        id: 'source-a', name: 'Source A',
        local_path: ++sourceReads === 1 ? f.root : changedRoot,
        last_sync_at: null, config: {},
      }],
      getConfig: async () => '0',
      setConfig: async () => {},
    } as unknown as BrainEngine;
    const report = await runCommand(f, undefined, {
      ocrProvider: async () => { providerCalls++; return injectedReceipt('text'); },
    });
    expect(sourceReads).toBeGreaterThanOrEqual(2);
    expect(providerCalls).toBe(0);
    expect(report?.status).toBe('failed');
  });

  test('an existing source-scoped same-hash page makes zero provider calls', async () => {
    const f = fixture();
    const entry = JSON.parse(readFileSync(f.manifestPath, 'utf8').trim());
    let providerCalls = 0;
    f.engine = {
      ...f.engine,
      getPage: async () => ({ content_hash: entry.sha256 }),
      getConfig: async () => '0',
      setConfig: async () => {},
    } as unknown as BrainEngine;
    const report = await runCommand(f, undefined, {
      ocrProvider: async () => { providerCalls++; return injectedReceipt('text'); },
    });
    expect(providerCalls).toBe(0);
    expect(report?.status).toBe('rejected');
    expect(report?.processed).toBe(0);
  });

  test('an exact serialized request body above 10 MiB is rejected before reservation', async () => {
    const f = fixture();
    const entry = JSON.parse(readFileSync(f.manifestPath, 'utf8').trim());
    // The old data-URL-only check passes this input; JSON request framing pushes
    // the exact UTF-8 body over the provider boundary.
    const bytes = Buffer.alloc(Math.floor(((10 * 1024 * 1024) - 100) * 3 / 4), 1);
    expect(Buffer.byteLength(`data:image/png;base64,${bytes.toString('base64')}`)).toBeLessThanOrEqual(10 * 1024 * 1024);
    expect(() => buildBoundedImageOcrRequestBody(bytes, 'image/png')).toThrow(/serialized request body exceeds/);
    writeFileSync(entry.file_path, bytes);
    entry.sha256 = sha256(bytes);
    writeFileSync(f.manifestPath, `${JSON.stringify(entry)}\n`);
    let providerCalls = 0;
    const report = await runCommand(f, undefined, {
      ocrProvider: async () => { providerCalls++; return injectedReceipt('text'); },
    });
    expect(providerCalls).toBe(0);
    expect(report?.status).toBe('rejected');
    expect(report?.processed).toBe(0);
  });

  test('an oversized image is rejected before provider access', async () => {
    const f = fixture();
    const entry = JSON.parse(readFileSync(f.manifestPath, 'utf8').trim());
    const bytes = Buffer.alloc(20 * 1024 * 1024 + 1, 1);
    writeFileSync(entry.file_path, bytes);
    entry.sha256 = sha256(bytes);
    writeFileSync(f.manifestPath, `${JSON.stringify(entry)}\n`);
    let providerCalls = 0;
    f.engine = {
      ...f.engine,
      getConfig: async () => '0',
      setConfig: async () => {},
    } as unknown as BrainEngine;
    const report = await runCommand(f, undefined, {
      ocrProvider: async () => { providerCalls++; return injectedReceipt('text'); },
    });
    expect(providerCalls).toBe(0);
    expect(report?.status).toBe('rejected');
    expect(report?.processed).toBe(0);
  });

  test('post-reservation path escape consumes the reservation and makes zero transport attempts', async () => {
    const f = fixture();
    let providerCalls = 0;
    let providerBytes = '';
    const engine = {
      ...f.engine,
      getPage: async () => null,
      getConfig: async () => '0',
      setConfig: async () => {},
    } as unknown as BrainEngine;
    f.engine = engine;
    const report = await runCommand(f, undefined, {
      afterReserve: (entry) => {
        const originalDir = join(f.root, 'images');
        const movedDir = join(f.root, 'images-original');
        const outsideDir = mkdtempSync(join(tmpdir(), 'gbrain-bounded-ocr-outside-'));
        renameSync(originalDir, movedDir);
        writeFileSync(join(outsideDir, '0000.png'), readFileSync(join(movedDir, '0000.png')));
        symlinkSync(outsideDir, originalDir, 'dir');
        expect(entry.file_path).toBe(join(originalDir, '0000.png'));
      },
      ocrProvider: async (bytes) => {
        providerCalls++;
        providerBytes = bytes.toString('utf8');
        return injectedReceipt('text');
      },
    });
    expect(providerCalls).toBe(0);
    expect(providerBytes).toBe('');
    expect(report).toMatchObject({ status: 'failed', processed: 1, succeeded: 0, failed: 1 });
  });

  test('provider/import failure stops immediately without processing later entries', async () => {
    const f = fixture(3);
    let calls = 0;
    const report = await runCommand(f, { maxImages: 3, maxUsd: 1, reserveUsdPerCall: 0.01 }, {
      importEntry: reservedImport(async () => { calls++; throw new Error('secret provider body'); }),
    });
    expect(calls).toBe(1);
    expect(report?.status).toBe('failed');
    expect(report?.first_failing_entry).toEqual({ index: 0, source_id: 'source-a', slug: 'images/0000.png', code: 'import_failed' });
    expect(JSON.stringify(report)).not.toContain('secret provider body');
  });

  test('report is byte-deterministic for a fixed manifest, clock, and result', async () => {
    const f = fixture();
    const run = async (ledgerDirectory: string) => {
      return JSON.stringify(await runCommand(f, undefined, {
        ledgerDirectory,
        importEntry: reservedImport(async () => {}),
      }));
    };
    expect(await run(f.ledgerDir)).toBe(await run(mkdtempSync(join(tmpdir(), 'gbrain-bounded-ocr-ledger-'))));
  });
});

test('command source contains no broad lifecycle lane', () => {
  const source = readFileSync('src/commands/image-ocr-run.ts', 'utf8');
  for (const forbidden of ['runSync', 'runImport', 'runExtract', 'runDream', 'runJobs', 'retry-failed', '--full', 'bookmark']) {
    expect(source).not.toContain(forbidden);
  }
});

test('safety-critical CLI flag registry is an exact narrow allowlist', () => {
  expect(CLI_FLAG_REGISTRY['image-ocr-run']).toEqual([
    '--brain', '--help', '--max-images', '--max-usd', '--recurring-strict-absence',
    '--reserve-usd-per-call', '--yes',
  ]);
  expect(CLI_FLAG_REGISTRY['image-ocr-run']).not.toContain('--full');
  expect(CLI_FLAG_REGISTRY['image-ocr-run']).not.toContain('--all');
});

test('raw image-ocr-run argv rejects every stripped global and unknown flag', () => {
  const base = [
    'image-ocr-run', 'manifest.jsonl',
    '--max-images', '1', '--max-usd', '1',
    '--reserve-usd-per-call', '0.01', '--yes',
  ];
  const forbidden = [
    ['--timeout', '1s'],
    ['--quiet'],
    ['--progress-json'],
    ['--progress-interval', '1'],
    ['--progress-interval=1'],
    ['--explain'],
    ['--json'],
    ['--source', 'source-a'],
    ['--unknown-image-ocr-flag'],
    ['-h'],
  ];
  for (const extra of forbidden) {
    expect(validateImageOcrRunRawArgv([...base, ...extra])).not.toBeNull();
  }

  expect(validateImageOcrRunRawArgv(base)).toBeNull();
  expect(validateImageOcrRunRawArgv(['--brain', 'host', ...base])).toBeNull();
  expect(validateImageOcrRunRawArgv(['--brain=host', ...base])).toBeNull();
  expect(validateImageOcrRunRawArgv(['image-ocr-run', '--help'])).toBeNull();
  expect(rawArgvTargetsImageOcrRun(['--quiet', ...base])).toBe(true);
  expect(rawArgvTargetsImageOcrRun(['import', 'image-ocr-run'])).toBe(false);
});

function runImageOcrCli(args: string[]) {
  const home = mkdtempSync(join(tmpdir(), 'gbrain-bounded-ocr-cli-home-'));
  return Bun.spawnSync(['bun', 'src/cli.ts', ...args], {
    cwd: process.cwd(),
    env: { ...process.env, GBRAIN_HOME: home, GBRAIN_SKIP_STARTUP_HOOKS: '1' },
    stdout: 'pipe',
    stderr: 'pipe',
  });
}

test('raw flag rejection emits exactly one JSON report before engine connect', () => {
  const result = runImageOcrCli([
    'image-ocr-run', '/does/not/exist.jsonl',
    '--max-images', '1', '--max-usd', '1',
    '--reserve-usd-per-call', '0.01', '--yes',
    '--timeout', '1s',
  ]);
  expect(result.exitCode).toBe(1);
  const lines = result.stdout.toString().trim().split('\n');
  expect(lines).toHaveLength(1);
  expect(JSON.parse(lines[0])).toMatchObject({
    status: 'rejected',
    terminal_error: 'arguments_invalid',
    requested: 0,
    processed: 0,
    cap: { max_images: 1, max_usd: 1, reserve_usd_per_call: 0.01 },
  });
  expect(result.stderr.toString()).toContain('--timeout');
  expect(result.stderr.toString()).not.toContain('No brain configured');
});

test('connectEngine failure emits exactly one accurate JSON report', () => {
  const result = runImageOcrCli([
    'image-ocr-run', '/does/not/exist.jsonl',
    '--max-images', '1', '--max-usd', '1',
    '--reserve-usd-per-call', '0.01', '--yes',
  ]);
  expect(result.exitCode).toBe(1);
  const lines = result.stdout.toString().trim().split('\n');
  expect(lines).toHaveLength(1);
  expect(JSON.parse(lines[0])).toMatchObject({
    manifest_hash: null,
    status: 'rejected',
    terminal_error: 'run_rejected',
    requested: 0,
    processed: 0,
    cap: { max_images: 1, max_usd: 1, reserve_usd_per_call: 0.01 },
  });
  expect(result.stderr.toString()).toContain('No brain configured');
});

test('argument rejection still emits a complete machine-readable terminal report', async () => {
  const writes: string[] = [];
  const originalWrite = process.stdout.write.bind(process.stdout);
  process.stdout.write = ((chunk: string | Uint8Array) => { writes.push(String(chunk)); return true; }) as typeof process.stdout.write;
  try {
    const report = await runImageOcrRun({} as BrainEngine, ['manifest.jsonl'], { now: UTC_DAY_1 });
    expect(report).toMatchObject({
      utc_date: '2026-08-10',
      manifest_hash: null,
      requested: 0,
      processed: 0,
      succeeded: 0,
      failed: 0,
      skipped: 0,
      status: 'rejected',
      terminal_error: 'arguments_invalid',
    });
    expect(JSON.parse(writes.join(''))).toEqual(report);
  } finally {
    process.stdout.write = originalWrite;
  }
});

test('locked command emits a complete machine-readable terminal report', async () => {
  const f = fixture();
  const caps = validateImageOcrCaps({ maxImages: 1, maxUsd: 1, reserveUsdPerCall: 0.01 });
  const ledger = OcrBudgetLedger.acquire({ directory: f.ledgerDir, now: UTC_DAY_1, caps, manifestHash: 'a'.repeat(64) });
  const writes: string[] = [];
  const originalWrite = process.stdout.write.bind(process.stdout);
  process.stdout.write = ((chunk: string | Uint8Array) => { writes.push(String(chunk)); return true; }) as typeof process.stdout.write;
  try {
    const report = await runImageOcrRun(f.engine, [
      f.manifestPath,
      '--max-images', '1',
      '--max-usd', '1',
      '--reserve-usd-per-call', '0.01',
      '--yes',
    ], {
      ledgerDirectory: f.ledgerDir,
      now: UTC_DAY_1,
      ocrProvider: async () => { throw new Error('provider must not run'); },
    });
    expect(report).toMatchObject({
      manifest_hash: sha256(readFileSync(f.manifestPath)),
      requested: 1,
      status: 'locked',
      terminal_error: 'budget_locked',
      daily_calls_before: null,
      daily_calls_after: null,
    });
    expect(JSON.parse(writes.join(''))).toEqual(report);
  } finally {
    process.stdout.write = originalWrite;
    ledger.close();
  }
});
