import { afterAll, beforeAll, beforeEach, describe, expect, test } from 'bun:test';
import { createHash } from 'node:crypto';
import {
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  renameSync,
  symlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { BrainEngine } from '../src/core/engine.ts';
import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import { resetPgliteState } from './helpers/reset-pglite.ts';
import {
  IMAGE_DONOR_ADOPTION_POLICY_VERSION,
  ExactHashDonorFileConflictError,
  ExactHashDonorRollbackConflictError,
  ExactHashDonorSchemaVersionError,
  ExactHashDonorTargetConflictError,
  ExactHashDonorUnavailableError,
  importImageFileWithExactHashDonor,
  rollbackImageFileExactHashDonor,
  type ExactHashDonorAdoptionReceipt,
} from '../src/core/image-donor-adopt.ts';
import {
  adoptValidatedImageDonorEntry,
  parseAndValidateImageDonorManifest,
  readImageOcrSourceFile,
} from '../src/core/image-ocr-run.ts';
import { withImageImportFence } from '../src/core/image-import-fence.ts';
import {
  IMAGE_DONOR_ADOPT_HELP,
  parseImageDonorAdoptArgs,
  runImageDonorAdopt,
  validateImageDonorAdoptRawArgv,
} from '../src/commands/image-donor-adopt.ts';
import { rawArgvTargetsImageDonorAdopt } from '../src/cli.ts';
import { CLI_FLAG_REGISTRY } from '../src/core/cli-flag-registry.generated.ts';

const TINY_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  'base64',
);
const TEXT_DIMS = 1536;
const IMAGE_DIMS = 1024;

let engine: PGLiteEngine;
let fenceRoot: string;

function sha256(value: Buffer | string): string {
  return createHash('sha256').update(value).digest('hex');
}

function vector(dimensions: number, value: number): string {
  return `[${Array.from({ length: dimensions }, () => value).join(',')}]`;
}

function dateText(value: unknown): string {
  return value instanceof Date ? value.toISOString() : String(value);
}

interface Fixture {
  root: string;
  filePath: string;
  slug: string;
  hash: string;
  manifestPath: string;
  sourceId: string;
}

async function fixture(suffix = ''): Promise<Fixture> {
  const root = mkdtempSync(join(tmpdir(), `gbrain-donor-adopt-${suffix}`));
  const slug = 'images/target.png';
  const filePath = join(root, slug);
  mkdirSync(join(root, 'images'), { recursive: true });
  writeFileSync(filePath, TINY_PNG);
  const hash = sha256(TINY_PNG);
  const sourceId = `source-${suffix || 'target'}`.replace(/[^a-z0-9-]/g, '-');
  await engine.executeRaw(
    `INSERT INTO sources (id, name, local_path, config)
     VALUES ($1, $2, $3, '{}'::jsonb)`,
    [sourceId, sourceId, root],
  );
  const manifestPath = join(root, 'manifest.jsonl');
  writeFileSync(manifestPath, `${JSON.stringify({
    source_id: sourceId,
    slug,
    file_path: filePath,
    sha256: hash,
  })}\n`);
  return { root, filePath, slug, hash, manifestPath, sourceId };
}

interface SeedDonorOptions {
  sourceId?: string;
  slug?: string;
  text?: string;
  chunkText?: string;
  pageKind?: string;
  deleted?: boolean;
  contextualRetrievalMode?: string | null;
  corpusGeneration?: number | null;
  chunkIndex?: number;
  chunkSource?: string;
  modality?: string;
  embedding?: string | null;
  model?: string;
  tokenCount?: number | null;
  embeddedAt?: string | null;
  embeddingSignature?: string | null;
  embeddingImage?: string | null;
  embeddingMultimodal?: string | null;
  extraChunk?: boolean;
}

async function seedDonor(
  hash: string,
  opts: SeedDonorOptions = {},
  donorEngine: BrainEngine = engine,
) {
  const sourceId = opts.sourceId ?? 'default';
  const slug = opts.slug ?? `donors/${Math.random().toString(16).slice(2)}.png`;
  const text = opts.text ?? `Donor OCR ${'quality text 123 '.repeat(10)}`;
  const model = opts.model ?? 'test:model';
  const embedding = opts.embedding === undefined ? vector(TEXT_DIMS, 0.01) : opts.embedding;
  const embeddingImage = opts.embeddingImage === undefined ? vector(IMAGE_DIMS, 0.02) : opts.embeddingImage;
  const embeddingMultimodal = opts.embeddingMultimodal === undefined
    ? vector(IMAGE_DIMS, 0.03)
    : opts.embeddingMultimodal;
  const signature = opts.embeddingSignature === undefined
    ? `${model}:${TEXT_DIMS}`
    : opts.embeddingSignature;
  const pages = await donorEngine.executeRaw<{ id: number }>(
    `INSERT INTO pages
       (source_id, slug, type, page_kind, title, compiled_truth, timeline,
        frontmatter, content_hash, embedding_signature, deleted_at,
        contextual_retrieval_mode, corpus_generation)
     VALUES ($1, $2, 'image', $3, 'donor.png', $4, '', '{}'::jsonb, $5, $6,
             CASE WHEN $7 THEN now() ELSE NULL END, $8, $9)
     RETURNING id`,
    [
      sourceId,
      slug,
      opts.pageKind ?? 'image',
      text,
      hash,
      signature,
      opts.deleted ?? false,
      opts.contextualRetrievalMode ?? null,
      opts.corpusGeneration ?? null,
    ],
  );
  const pageId = Number(pages[0].id);
  const chunks = await donorEngine.executeRaw<{ id: number }>(
    `INSERT INTO content_chunks
       (page_id, chunk_index, chunk_text, chunk_source, embedding, model,
        token_count, embedded_at, modality, embedding_image, embedding_multimodal)
     VALUES ($1, $2, $3, $4, $5::vector, $6, $7,
             $8::timestamptz, $9, $10::vector, $11::vector)
     RETURNING id`,
    [
      pageId,
      opts.chunkIndex ?? 0,
      opts.chunkText ?? text,
      opts.chunkSource ?? 'image_asset',
      embedding,
      model,
      opts.tokenCount === undefined ? 42 : opts.tokenCount,
      opts.embeddedAt === undefined ? '2026-08-12T00:00:00.000Z' : opts.embeddedAt,
      opts.modality ?? 'image',
      embeddingImage,
      embeddingMultimodal,
    ],
  );
  if (opts.extraChunk) {
    await donorEngine.executeRaw(
      `INSERT INTO content_chunks
         (page_id, chunk_index, chunk_text, chunk_source, modality)
       VALUES ($1, 1, 'extra', 'image_asset', 'image')`,
      [pageId],
    );
  }
  return { pageId, chunkId: Number(chunks[0].id), sourceId, slug, text };
}

async function validated(f: Fixture) {
  const manifest = await parseAndValidateImageDonorManifest(engine, f.manifestPath);
  expect(manifest.entries).toHaveLength(1);
  return manifest.entries[0];
}

async function adopt(f: Fixture, options: Parameters<typeof importImageFileWithExactHashDonor>[3] = {}) {
  const entry = await validated(f);
  return withImageImportFence(
    token => adoptValidatedImageDonorEntry(engine, entry, token, options),
    { lockRoot: fenceRoot },
  );
}

async function pageState(pageId: number) {
  const rows = await engine.executeRaw<Record<string, unknown>>(
    `SELECT id, source_id, slug, type, page_kind, title, compiled_truth, timeline,
            frontmatter, content_hash, embedding_signature, import_filename,
            source_path, generation, created_at, updated_at, deleted_at
       FROM pages WHERE id=$1`,
    [pageId],
  );
  return rows[0] ?? null;
}

async function chunkState(chunkId: number) {
  const rows = await engine.executeRaw<Record<string, unknown>>(
    `SELECT id, page_id, chunk_index, chunk_text, chunk_source,
            CASE WHEN embedding IS NULL THEN NULL ELSE embedding::text END AS embedding,
            model, token_count, embedded_at, modality,
            CASE WHEN embedding_image IS NULL THEN NULL ELSE embedding_image::text END AS embedding_image,
            CASE WHEN embedding_multimodal IS NULL THEN NULL ELSE embedding_multimodal::text END AS embedding_multimodal
       FROM content_chunks WHERE id=$1`,
    [chunkId],
  );
  return rows[0] ?? null;
}

async function fileState(storagePath: string) {
  const rows = await engine.executeRaw<Record<string, unknown>>(
    `SELECT id, source_id, page_slug, page_id, filename, storage_path, mime_type,
            size_bytes, content_hash, metadata, created_at
       FROM files WHERE storage_path=$1`,
    [storagePath],
  );
  return rows[0] ?? null;
}

async function captureStdout<T>(fn: () => Promise<T>): Promise<{ result: T; output: string }> {
  const writes: string[] = [];
  const original = process.stdout.write.bind(process.stdout);
  process.stdout.write = ((chunk: string | Uint8Array) => {
    writes.push(String(chunk));
    return true;
  }) as typeof process.stdout.write;
  try {
    return { result: await fn(), output: writes.join('') };
  } finally {
    process.stdout.write = original;
  }
}

function donorCliEnvironment(home: string): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value === undefined) continue;
    if (/(?:API_KEY|ACCESS_KEY|TOKEN|SECRET|PASSWORD|DATABASE_URL)/.test(key)) continue;
    env[key] = value;
  }
  env.GBRAIN_HOME = home;
  env.GBRAIN_BRAIN_ID = 'host';
  env.GBRAIN_MOUNTS_PATH = join(home, '.gbrain', 'absent-mounts.json');
  env.GBRAIN_SKIP_STARTUP_HOOKS = '1';
  env.NODE_ENV = 'test';
  return env;
}

function runImageDonorCli(args: string[], home = mkdtempSync(join(tmpdir(), 'gbrain-donor-cli-home-'))) {
  return Bun.spawnSync(['bun', '--no-env-file', 'run', 'src/cli.ts', ...args], {
    cwd: process.cwd(),
    env: donorCliEnvironment(home),
    stdout: 'pipe',
    stderr: 'pipe',
  });
}

async function resetDonorTestState(): Promise<void> {
  await resetPgliteState(engine);
  await engine.setConfig('version', '125');
}

beforeAll(async () => {
  engine = new PGLiteEngine();
  await engine.connect({});
  await engine.initSchema();
  fenceRoot = join(mkdtempSync(join(tmpdir(), 'gbrain-donor-adopt-fence-')), 'fence.lock');
});

afterAll(async () => {
  await engine.disconnect();
});

beforeEach(async () => {
  await resetDonorTestState();
});

describe('donor-only CLI authorization and report surface', () => {
  test('requires a finite bounded integer cap plus explicit confirmation', () => {
    expect(() => parseImageDonorAdoptArgs(['manifest.jsonl'])).toThrow(/--max-images/);
    expect(() => parseImageDonorAdoptArgs(['manifest.jsonl', '--max-images', '1'])).toThrow(/--yes/);
    for (const value of ['0', '-1', '1.5', 'Infinity', '1001']) {
      expect(() => parseImageDonorAdoptArgs([
        'manifest.jsonl', '--max-images', value, '--yes',
      ])).toThrow();
    }
    expect(parseImageDonorAdoptArgs([
      'manifest.jsonl', '--max-images=1000', '--yes',
    ])).toEqual({ manifestPath: 'manifest.jsonl', maxImages: 1000, yes: true });
  });

  test('uses an exact narrow raw/global flag allowlist', () => {
    const base = ['image-donor-adopt', 'manifest.jsonl', '--max-images', '1', '--yes'];
    expect(CLI_FLAG_REGISTRY['image-donor-adopt']).toEqual([
      '--brain', '--help', '--max-images', '--yes',
    ]);
    expect(validateImageDonorAdoptRawArgv(base)).toBeNull();
    expect(validateImageDonorAdoptRawArgv(['--brain=host', ...base])).toBeNull();
    expect(validateImageDonorAdoptRawArgv(['image-donor-adopt', '--help'])).toBeNull();
    expect(validateImageDonorAdoptRawArgv([
      'image-donor-adopt', 'manifest.jsonl', '--help',
    ])).toBeNull();
    for (const extra of [
      ['--timeout', '1s'], ['--source', 'x'], ['--json'], ['--quiet'],
      ['--progress-json'], ['--max-usd', '1'], ['--unknown'], ['-h'],
    ]) {
      expect(validateImageDonorAdoptRawArgv([...base, ...extra])).not.toBeNull();
    }
    expect(rawArgvTargetsImageDonorAdopt(['--quiet', ...base])).toBe(true);
    expect(rawArgvTargetsImageDonorAdopt(['--timeout', 'nope', ...base])).toBe(true);
    expect(rawArgvTargetsImageDonorAdopt(['--brain', 'INVALID', ...base])).toBe(true);
    expect(rawArgvTargetsImageDonorAdopt(['import', 'image-donor-adopt'])).toBe(false);
  });

  test('detailed and top-level help are reachable without a configured brain', () => {
    for (const args of [
      ['image-donor-adopt', '--help'],
      ['image-donor-adopt', 'manifest.jsonl', '--help'],
    ]) {
      const result = runImageDonorCli(args);
      expect(result.exitCode).toBe(0);
      expect(result.stdout.toString()).toBe(IMAGE_DONOR_ADOPT_HELP);
      expect(result.stderr.toString()).not.toContain('brain configured');
    }

    const topLevel = runImageDonorCli(['--help']);
    expect(topLevel.exitCode).toBe(0);
    expect(topLevel.stdout.toString()).toContain('image-donor-adopt <manifest>');
  });

  test('raw rejection emits one accurate report before storage connection', () => {
    const home = mkdtempSync(join(tmpdir(), 'gbrain-donor-raw-reject-'));
    const manifestPath = join(home, 'manifest.jsonl');
    const manifestBytes = Buffer.from('{}\n');
    writeFileSync(manifestPath, manifestBytes);
    const result = runImageDonorCli([
      '--timeout', '1s', 'image-donor-adopt', manifestPath,
      '--max-images', '1', '--yes',
    ], home);
    expect(result.exitCode).toBe(1);
    const lines = result.stdout.toString().trim().split('\n');
    expect(lines).toHaveLength(1);
    expect(JSON.parse(lines[0])).toMatchObject({
      manifest_hash: sha256(manifestBytes),
      requested: 0,
      processed: 0,
      status: 'rejected',
      terminal_error: 'arguments_invalid',
      cap: { max_images: 1 },
    });
    expect(result.stderr.toString()).toContain('--timeout');
    expect(result.stderr.toString()).not.toContain('brain configured');

    const malformed = runImageDonorCli([
      '--timeout', 'nope', 'image-donor-adopt', manifestPath,
      '--max-images', '1', '--yes',
    ], home);
    expect(malformed.exitCode).toBe(1);
    expect(malformed.stdout.toString().trim().split('\n')).toHaveLength(1);
    expect(JSON.parse(malformed.stdout.toString())).toMatchObject({
      manifest_hash: sha256(manifestBytes),
      terminal_error: 'arguments_invalid',
      provider_attempts: 0,
      ocr_budget_reservations: 0,
    });
    expect(malformed.stderr.toString()).toContain('--timeout');
  });

  test('storage connection and thin-client refusal each emit exactly one report', () => {
    const missing = runImageDonorCli([
      'image-donor-adopt', '/does/not/exist.jsonl', '--max-images', '1', '--yes',
    ]);
    expect(missing.exitCode).toBe(1);
    const missingLines = missing.stdout.toString().trim().split('\n');
    expect(missingLines).toHaveLength(1);
    expect(JSON.parse(missingLines[0])).toMatchObject({
      manifest_hash: null,
      status: 'rejected',
      terminal_error: 'run_rejected',
      cap: { max_images: 1 },
    });
    expect(missing.stderr.toString()).toContain('No host brain configured');

    const thinHome = mkdtempSync(join(tmpdir(), 'gbrain-donor-thin-'));
    mkdirSync(join(thinHome, '.gbrain'), { recursive: true });
    writeFileSync(join(thinHome, '.gbrain', 'config.json'), JSON.stringify({
      engine: 'postgres',
      remote_mcp: {
        issuer_url: 'https://brain-host.example',
        mcp_url: 'https://brain-host.example/mcp',
        oauth_client_id: 'example-client',
        oauth_client_secret: 'example-secret',
      },
    }));
    const refused = runImageDonorCli([
      'image-donor-adopt', '/does/not/exist.jsonl', '--max-images', '1', '--yes',
    ], thinHome);
    expect(refused.exitCode).toBe(1);
    const refusedLines = refused.stdout.toString().trim().split('\n');
    expect(refusedLines).toHaveLength(1);
    expect(JSON.parse(refusedLines[0])).toMatchObject({
      status: 'rejected',
      terminal_error: 'run_rejected',
      provider_attempts: 0,
      gateway_calls: 0,
      ocr_budget_reservations: 0,
    });
    expect(refused.stderr.toString()).toContain('requires the host source registry');
  });

  test('CLI cap is a nonzero partial-completion verdict with no OCR budget access', async () => {
    const home = mkdtempSync(join(tmpdir(), 'gbrain-donor-cap-cli-'));
    const dotGbrain = join(home, '.gbrain');
    const databasePath = join(dotGbrain, 'brain.pglite');
    const sourceRoot = join(home, 'source-root');
    mkdirSync(join(sourceRoot, 'images'), { recursive: true });
    mkdirSync(dotGbrain, { recursive: true });
    const hash = sha256(TINY_PNG);
    const manifestPath = join(home, 'manifest.jsonl');
    const entries = [0, 1].map(index => {
      const slug = `images/${index}.png`;
      const filePath = join(sourceRoot, slug);
      writeFileSync(filePath, TINY_PNG);
      return { source_id: 'source-cli-cap', slug, file_path: filePath, sha256: hash };
    });
    writeFileSync(manifestPath, `${entries.map(entry => JSON.stringify(entry)).join('\n')}\n`);

    const seedEngine = new PGLiteEngine();
    await seedEngine.connect({ engine: 'pglite', database_path: databasePath });
    await seedEngine.initSchema();
    await seedEngine.executeRaw(
      `INSERT INTO sources (id, name, local_path, config)
       VALUES ('source-cli-cap', 'source-cli-cap', $1, '{}'::jsonb)`,
      [sourceRoot],
    );
    await seedDonor(hash, { slug: 'donors/cli-cap.png' }, seedEngine);
    await seedEngine.disconnect();
    writeFileSync(join(dotGbrain, 'config.json'), JSON.stringify({
      engine: 'pglite',
      database_path: databasePath,
    }));

    const result = runImageDonorCli([
      'image-donor-adopt', manifestPath, '--max-images', '1', '--yes',
    ], home);
    expect(result.exitCode).toBe(1);
    const lines = result.stdout.toString().trim().split('\n');
    expect(lines).toHaveLength(1);
    expect(JSON.parse(lines[0])).toMatchObject({
      requested: 2,
      processed: 1,
      succeeded: 1,
      skipped: 1,
      failed: 0,
      status: 'cap_reached',
      provider_attempts: 0,
      model_calls: 0,
      gateway_calls: 0,
      ocr_budget_reservations: 0,
      ocr_budget_usd: 0,
      cap: { max_images: 1 },
    });
    expect(existsSync(join(dotGbrain, 'ocr-budget'))).toBe(false);

    const verifyEngine = new PGLiteEngine();
    await verifyEngine.connect({ engine: 'pglite', database_path: databasePath });
    expect(await verifyEngine.getPage('images/0.png', { sourceId: 'source-cli-cap' })).not.toBeNull();
    expect(await verifyEngine.getPage('images/1.png', { sourceId: 'source-cli-cap' })).toBeNull();
    await verifyEngine.disconnect();
  });

  test('contains no provider, model, gateway, budget, or broad lifecycle lane', () => {
    const command = readFileSync('src/commands/image-donor-adopt.ts', 'utf8');
    const donorCore = readFileSync('src/core/image-donor-adopt.ts', 'utf8');
    const decoder = readFileSync('src/core/image-decode.ts', 'utf8');
    const decoderImpl = readFileSync('src/core/image-decode-impl.ts', 'utf8');
    const cli = readFileSync('src/cli.ts', 'utf8');
    for (const forbidden of [
      'image-ocr-provider', 'image-ocr-budget', 'requestBoundedImageOcr',
      'loadBoundedImageOcrProviderConfig', 'fetch(', 'runSync', 'runImport',
      'runExtract', 'runDream', 'runJobs', 'retry-failed', '--full',
    ]) expect(command).not.toContain(forbidden);
    for (const source of [donorCore, decoder, decoderImpl]) {
      for (const forbidden of [
        'image-ocr-provider', 'image-ocr-budget', 'model-pricing', 'ai/gateway',
        'import-file', 'upsertFile(', 'upsertChunks(', 'embedMultimodal(',
        'maybeOcr(', 'fetch(', 'canonicalLookup(',
      ]) expect(source).not.toContain(forbidden);
    }
    const storageConnection = cli.slice(
      cli.indexOf('async function connectStorageOnlyEngine'),
      cli.indexOf('async function connectMountEngine'),
    );
    for (const forbidden of ['configureGateway(', 'initSchema(', 'tryRunPendingMigrations(', 'connectEngine(']) {
      expect(storageConnection).not.toContain(forbidden);
    }
  });
});

describe('exact-hash donor qualification and adoption transaction', () => {
  test('fails closed before writes for stale, missing, or malformed schema version state', async () => {
    const f = await fixture('schema-version');
    const donor = await seedDonor(f.hash);
    const donorPageBefore = await pageState(donor.pageId);
    const donorChunkBefore = await chunkState(donor.chunkId);

    await engine.setConfig('version', '124');
    const { result: staleReport } = await captureStdout(() => runImageDonorAdopt(engine, [
      f.manifestPath, '--max-images', '1', '--yes',
    ], { imageImportFenceRoot: fenceRoot }));
    expect(staleReport).toMatchObject({
      status: 'failed',
      first_failure: { code: 'schema_incompatible' },
      results: [{ status: 'failed', error_code: 'schema_incompatible' }],
      provider_attempts: 0,
      ocr_budget_reservations: 0,
    });

    for (const version of [null, 'not-an-integer'] as const) {
      if (version === null) await engine.unsetConfig('version');
      else await engine.setConfig('version', version);
      await expect(adopt(f)).rejects.toBeInstanceOf(ExactHashDonorSchemaVersionError);
      expect(await engine.getPage(f.slug, {
        sourceId: f.sourceId,
        includeDeleted: true,
      })).toBeNull();
      expect(await fileState(f.slug)).toBeNull();
      expect(await pageState(donor.pageId)).toEqual(donorPageBefore);
      expect(await chunkState(donor.chunkId)).toEqual(donorChunkBefore);
    }
  });

  test('creates fresh source-local page/chunk identities and copies only derived OCR/vector data', async () => {
    const f = await fixture('happy');
    const donor = await seedDonor(f.hash, { slug: 'donors/happy.png' });
    await engine.setConfig('ocr_provider_attempts', '17');
    await engine.setConfig('ocr_successful_provider_receipts', '13');
    await engine.setConfig('ocr_persisted_imports', '11');
    const donorPageBefore = await pageState(donor.pageId);
    const donorChunkBefore = await chunkState(donor.chunkId);

    const result = await adopt(f);
    expect(result.status).toBe('adopted');
    if (result.status !== 'adopted') throw new Error('expected adoption');
    expect(result.receipt.policy_version).toBe(IMAGE_DONOR_ADOPTION_POLICY_VERSION);
    expect(result.receipt.target_page_id).not.toBe(donor.pageId);
    expect(result.receipt.target_chunk_id).not.toBe(donor.chunkId);
    expect(result.receipt.donor_page_id).toBe(donor.pageId);
    expect(result.receipt.donor_chunk_id).toBe(donor.chunkId);
    expect(result.receipt.ocr_sha256).toBe(sha256(donor.text));
    expect(result.receipt.file_disposition).toBe('created');

    const targetPage = await pageState(result.receipt.target_page_id);
    const targetChunk = await chunkState(result.receipt.target_chunk_id);
    expect(targetPage).toMatchObject({
      source_id: f.sourceId,
      slug: f.slug,
      type: 'image',
      page_kind: 'image',
      title: 'target.png',
      compiled_truth: donor.text,
      timeline: '',
      content_hash: f.hash,
      embedding_signature: 'test:model:1536',
      import_filename: 'target.png',
      source_path: f.slug,
    });
    const fm = targetPage!.frontmatter as Record<string, unknown>;
    expect(fm.type).toBe('image');
    expect(fm.title).toBe('target.png');
    expect(fm.image_donor_adoption).toMatchObject({
      policy_version: IMAGE_DONOR_ADOPTION_POLICY_VERSION,
      donor_page_id: donor.pageId,
      donor_chunk_id: donor.chunkId,
      image_sha256: f.hash,
      ocr_sha256: sha256(donor.text),
    });
    expect(JSON.stringify(fm)).not.toContain(donor.slug);
    expect(targetChunk).toMatchObject({
      page_id: result.receipt.target_page_id,
      chunk_index: 0,
      chunk_text: donor.text,
      chunk_source: 'image_asset',
      model: 'test:model',
      token_count: 42,
      modality: 'image',
    });
    expect(targetChunk!.embedding).toBe(donorChunkBefore!.embedding);
    expect(targetChunk!.embedding_image).toBe(donorChunkBefore!.embedding_image);
    expect(targetChunk!.embedding_multimodal).toBe(donorChunkBefore!.embedding_multimodal);
    expect(dateText(targetChunk!.embedded_at)).toBe(dateText(donorChunkBefore!.embedded_at));
    expect(await pageState(donor.pageId)).toEqual(donorPageBefore);
    expect(await chunkState(donor.chunkId)).toEqual(donorChunkBefore);
    expect(await engine.getConfig('ocr_provider_attempts')).toBe('17');
    expect(await engine.getConfig('ocr_successful_provider_receipts')).toBe('13');
    expect(await engine.getConfig('ocr_persisted_imports')).toBe('11');
  });

  test('chooses the lowest page/chunk IDs deterministically and binds all reusable digests', async () => {
    const f = await fixture('deterministic');
    const first = await seedDonor(f.hash, { slug: 'z-donor.png' });
    await seedDonor(f.hash, { slug: 'a-donor.png', text: `Other ${'quality 456 '.repeat(11)}` });
    const result = await adopt(f);
    expect(result.status).toBe('adopted');
    if (result.status !== 'adopted') return;
    expect(result.receipt.donor_page_id).toBe(first.pageId);
    expect(result.receipt.donor_chunk_id).toBe(first.chunkId);
    expect(result.receipt.text_embedding_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(result.receipt.image_embedding_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(result.receipt.multimodal_embedding_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(result.receipt.donor_state_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(result.receipt.target_poststate_sha256).toMatch(/^[a-f0-9]{64}$/);
  });

  test('preserves honest nullable optional provenance while requiring a compatible text vector', async () => {
    const f = await fixture('nullable');
    await seedDonor(f.hash, {
      tokenCount: null,
      embeddedAt: null,
      embeddingSignature: null,
      embeddingImage: null,
      embeddingMultimodal: null,
    });
    const result = await adopt(f);
    expect(result.status).toBe('adopted');
    if (result.status !== 'adopted') return;
    const page = await pageState(result.receipt.target_page_id);
    const chunk = await chunkState(result.receipt.target_chunk_id);
    expect(page!.embedding_signature).toBeNull();
    expect(chunk).toMatchObject({ token_count: null, embedded_at: null, embedding_image: null, embedding_multimodal: null });
    expect(chunk!.embedding).not.toBeNull();
  });

  test('fails closed for missing or nonqualifying donors without creating a target', async () => {
    const badCases: Array<[string, SeedDonorOptions | null]> = [
      ['missing', null],
      ['short', { text: 'too short 123' }],
      ['blank', { text: ' '.repeat(130) }],
      ['punctuation', { text: '!'.repeat(130) }],
      ['deleted', { deleted: true }],
      ['not-image-page', { pageKind: 'markdown' }],
      ['wrong-index', { chunkIndex: 1 }],
      ['wrong-source', { chunkSource: 'compiled_truth' }],
      ['wrong-modality', { modality: 'text' }],
      ['text-mismatch', { chunkText: `Different ${'quality text 123 '.repeat(10)}` }],
      ['missing-text-vector', { embedding: null }],
      ['blank-model', { model: ' ' }],
      ['negative-token-count', { tokenCount: -1 }],
      ['bad-signature', { embeddingSignature: 'other:model:1536' }],
      ['contextual-title', { contextualRetrievalMode: 'title' }],
      ['contextual-synopsis', { contextualRetrievalMode: 'per_chunk_synopsis' }],
      ['corpus-generation', { corpusGeneration: 1 }],
      ['two-chunks', { extraChunk: true }],
    ];
    for (const [name, donorOpts] of badCases) {
      await resetDonorTestState();
      const f = await fixture(`bad-${name}`);
      if (donorOpts) await seedDonor(f.hash, donorOpts);
      await expect(adopt(f)).rejects.toBeInstanceOf(ExactHashDonorUnavailableError);
      const targets = await engine.executeRaw<{ n: number }>(
        `SELECT count(*)::int AS n FROM pages WHERE source_id=$1 AND slug=$2`,
        [f.sourceId, f.slug],
      );
      expect(Number(targets[0].n)).toBe(0);
    }
  });

  test('binds one donor and refuses its disappearance or qualification drift', async () => {
    for (const kind of ['deleted', 'text', 'vector', 'signature', 'extra-chunk'] as const) {
      await resetDonorTestState();
      const f = await fixture(`donor-drift-${kind}`);
      const donor = await seedDonor(f.hash, { slug: `donors/${kind}.png` });
      const pageBefore = await pageState(donor.pageId);
      const chunkBefore = await chunkState(donor.chunkId);
      await expect(adopt(f, {
        afterDonorLockForTest: async tx => {
          if (kind === 'deleted') {
            await tx.executeRaw(`UPDATE pages SET deleted_at=now() WHERE id=$1`, [donor.pageId]);
          } else if (kind === 'text') {
            await tx.executeRaw(`UPDATE content_chunks SET chunk_text='drift' WHERE id=$1`, [donor.chunkId]);
          } else if (kind === 'vector') {
            await tx.executeRaw(`UPDATE content_chunks SET embedding=$1::vector WHERE id=$2`, [vector(TEXT_DIMS, 0.04), donor.chunkId]);
          } else if (kind === 'signature') {
            await tx.executeRaw(`UPDATE pages SET embedding_signature='wrong:1536' WHERE id=$1`, [donor.pageId]);
          } else {
            await tx.executeRaw(
              `INSERT INTO content_chunks (page_id, chunk_index, chunk_text, chunk_source, modality)
               VALUES ($1, 1, 'extra', 'image_asset', 'image')`,
              [donor.pageId],
            );
          }
        },
      })).rejects.toBeInstanceOf(ExactHashDonorUnavailableError);
      expect(await pageState(donor.pageId)).toEqual(pageBefore);
      expect(await chunkState(donor.chunkId)).toEqual(chunkBefore);
      expect(await engine.getPage(f.slug, { sourceId: f.sourceId, includeDeleted: true })).toBeNull();
    }
  });

  test('refuses active, soft-deleted, and conditional-claim target conflicts without updates', async () => {
    for (const deleted of [false, true]) {
      await resetDonorTestState();
      const f = await fixture(`target-${deleted}`);
      await seedDonor(f.hash);
      const rows = await engine.executeRaw<{ id: number }>(
        `INSERT INTO pages
           (source_id, slug, type, page_kind, title, compiled_truth, frontmatter,
            content_hash, deleted_at)
         VALUES ($1, $2, 'image', 'image', 'existing', 'owned', '{}'::jsonb,
                 'different', CASE WHEN $3 THEN now() ELSE NULL END)
         RETURNING id`,
        [f.sourceId, f.slug, deleted],
      );
      const before = await pageState(Number(rows[0].id));
      await expect(adopt(f)).rejects.toBeInstanceOf(ExactHashDonorTargetConflictError);
      expect(await pageState(Number(rows[0].id))).toEqual(before);
    }

    await resetDonorTestState();
    const f = await fixture('claim-race');
    await seedDonor(f.hash);
    await expect(adopt(f, { forceTargetClaimConflictForTest: true }))
      .rejects.toBeInstanceOf(ExactHashDonorTargetConflictError);
    expect(await engine.getPage(f.slug, { sourceId: f.sourceId, includeDeleted: true })).toBeNull();
  });

  test('treats an exact matching prior adoption as idempotent and every other same-hash target as conflict', async () => {
    const f = await fixture('idempotent');
    await seedDonor(f.hash);
    const first = await adopt(f);
    expect(first.status).toBe('adopted');
    const second = await adopt(f);
    expect(second.status).toBe('idempotent');
    if (first.status !== 'adopted' || second.status !== 'idempotent') return;
    expect(second.target_page_id).toBe(first.receipt.target_page_id);
    expect(second.target_chunk_id).toBe(first.receipt.target_chunk_id);
    expect(second.donor_page_id).toBe(first.receipt.donor_page_id);
    expect(second.donor_chunk_id).toBe(first.receipt.donor_chunk_id);

    const replacementEmbedding = vector(TEXT_DIMS, 0.04);
    await engine.executeRaw(
      `UPDATE content_chunks
          SET embedding=$1::vector, embedded_at='2026-08-12T01:00:00.000Z'
        WHERE id IN ($2, $3)`,
      [replacementEmbedding, first.receipt.donor_chunk_id, first.receipt.target_chunk_id],
    );
    await expect(adopt(f)).rejects.toBeInstanceOf(ExactHashDonorTargetConflictError);

    const frontmatter = await fixture('idempotent-frontmatter');
    await seedDonor(frontmatter.hash);
    const frontmatterAdoption = await adopt(frontmatter);
    expect(frontmatterAdoption.status).toBe('adopted');
    if (frontmatterAdoption.status !== 'adopted') return;
    await engine.executeRaw(
      `UPDATE pages
          SET frontmatter = frontmatter || '{"unexpected":"drift"}'::jsonb
        WHERE id=$1`,
      [frontmatterAdoption.receipt.target_page_id],
    );
    await expect(adopt(frontmatter)).rejects.toBeInstanceOf(ExactHashDonorTargetConflictError);
  });

  test('replays exact adoption after get retrieval telemetry without rebasing rollback CAS', async () => {
    const f = await fixture('idempotent-after-get');
    await seedDonor(f.hash);
    const first = await adopt(f);
    expect(first.status).toBe('adopted');
    if (first.status !== 'adopted') return;

    await engine.executeRaw(
      `UPDATE pages
          SET last_retrieved_at='2026-08-12T04:29:09.344Z'::timestamptz
        WHERE id=$1`,
      [first.receipt.target_page_id],
    );
    const telemetry = await engine.executeRaw<Record<string, unknown>>(
      `SELECT last_retrieved_at, salience_touched_at,
              updated_at = created_at AS timestamps_unchanged
         FROM pages WHERE id=$1`,
      [first.receipt.target_page_id],
    );
    expect(telemetry[0]?.last_retrieved_at).not.toBeNull();
    expect(telemetry[0]?.salience_touched_at).toBeNull();
    expect(telemetry[0]?.timestamps_unchanged).toBe(true);

    const replay = await adopt(f);
    expect(replay.status).toBe('idempotent');
    if (replay.status !== 'idempotent') return;
    const { status: _status, ...replayReceipt } = replay;
    expect(replayReceipt).toEqual(first.receipt);
    const replayTelemetry = await engine.executeRaw<Record<string, unknown>>(
      `SELECT last_retrieved_at FROM pages WHERE id=$1`,
      [first.receipt.target_page_id],
    );
    expect(dateText(replayTelemetry[0]?.last_retrieved_at)).toBe('2026-08-12T04:29:09.344Z');
    await expect(withImageImportFence(
      token => rollbackImageFileExactHashDonor(engine, replayReceipt, token),
      { lockRoot: fenceRoot },
    )).rejects.toBeInstanceOf(ExactHashDonorRollbackConflictError);
    expect(await engine.getPage(f.slug, { sourceId: f.sourceId })).not.toBeNull();
  });

  test('creates, preserves, or rejects the global files row without unsafe ownership rewrites', async () => {
    const f = await fixture('files-created');
    await seedDonor(f.hash);
    const created = await adopt(f);
    expect(created.status).toBe('adopted');
    if (created.status !== 'adopted') return;
    expect(created.receipt.file_disposition).toBe('created');
    expect(await fileState(f.slug)).toMatchObject({
      id: created.receipt.file_id,
      source_id: f.sourceId,
      page_slug: f.slug,
      page_id: created.receipt.target_page_id,
      filename: 'target.png',
      storage_path: f.slug,
      content_hash: f.hash,
    });

    await resetDonorTestState();
    const same = await fixture('files-same');
    await seedDonor(same.hash);
    const existingRows = await engine.executeRaw<{ id: number }>(
      `INSERT INTO files
         (source_id, page_slug, page_id, filename, storage_path, mime_type,
          size_bytes, content_hash, metadata)
       VALUES ('default', 'legacy.png', NULL, 'legacy-name.png', $1,
               'application/x-legacy', 999, $2, '{"legacy":true}'::jsonb)
       RETURNING id`,
      [same.slug, same.hash],
    );
    const fileBefore = await fileState(same.slug);
    const preserved = await adopt(same);
    expect(preserved.status).toBe('adopted');
    if (preserved.status !== 'adopted') return;
    expect(preserved.receipt.file_disposition).toBe('preserved_existing');
    expect(preserved.receipt.file_id).toBe(Number(existingRows[0].id));
    expect(await fileState(same.slug)).toEqual(fileBefore);

    await resetDonorTestState();
    const different = await fixture('files-different');
    await seedDonor(different.hash);
    await engine.executeRaw(
      `INSERT INTO files (source_id, filename, storage_path, content_hash)
       VALUES ('default', 'legacy.png', $1, $2)`,
      [different.slug, 'f'.repeat(64)],
    );
    const conflictingFile = await fileState(different.slug);
    await expect(adopt(different)).rejects.toBeInstanceOf(ExactHashDonorFileConflictError);
    expect(await engine.getPage(different.slug, { sourceId: different.sourceId, includeDeleted: true })).toBeNull();
    expect(await fileState(different.slug)).toEqual(conflictingFile);

    await resetDonorTestState();
    const racedSame = await fixture('files-race-same');
    await seedDonor(racedSame.hash);
    const sameRace = await adopt(racedSame, {
      beforeFileInsertForTest: tx => tx.executeRaw(
        `INSERT INTO files (source_id, filename, storage_path, content_hash, metadata)
         VALUES ('default', 'racer.png', $1, $2, '{"racer":true}'::jsonb)`,
        [racedSame.slug, racedSame.hash],
      ).then(() => undefined),
    });
    expect(sameRace.status).toBe('adopted');
    if (sameRace.status === 'adopted') {
      expect(sameRace.receipt.file_disposition).toBe('preserved_existing');
      expect(await fileState(racedSame.slug)).toMatchObject({
        source_id: 'default', filename: 'racer.png', metadata: { racer: true },
      });
    }

    await resetDonorTestState();
    const racedDifferent = await fixture('files-race-different');
    await seedDonor(racedDifferent.hash);
    await expect(adopt(racedDifferent, {
      beforeFileInsertForTest: tx => tx.executeRaw(
        `INSERT INTO files (source_id, filename, storage_path, content_hash)
         VALUES ('default', 'racer.png', $1, $2)`,
        [racedDifferent.slug, 'e'.repeat(64)],
      ).then(() => undefined),
    })).rejects.toBeInstanceOf(ExactHashDonorFileConflictError);
    expect(await fileState(racedDifferent.slug)).toBeNull();
    expect(await engine.getPage(racedDifferent.slug, { sourceId: racedDifferent.sourceId, includeDeleted: true })).toBeNull();
  });

  test('refuses pre-existing dangling aliases before target adoption', async () => {
    const f = await fixture('pre-existing-alias');
    const donor = await seedDonor(f.hash);
    const donorPageBefore = await pageState(donor.pageId);
    const donorChunkBefore = await chunkState(donor.chunkId);
    await engine.executeRaw(
      `INSERT INTO page_aliases (source_id, alias_norm, slug)
       VALUES ($1, 'target alias', $2)`,
      [f.sourceId, f.slug],
    );

    await expect(adopt(f)).rejects.toBeInstanceOf(ExactHashDonorTargetConflictError);
    expect(await engine.getPage(f.slug, { sourceId: f.sourceId, includeDeleted: true })).toBeNull();
    expect(await fileState(f.slug)).toBeNull();
    expect(await engine.executeRaw(
      `SELECT alias_norm FROM page_aliases WHERE source_id=$1 AND slug=$2`,
      [f.sourceId, f.slug],
    )).toEqual([{ alias_norm: 'target alias' }]);
    expect(await pageState(donor.pageId)).toEqual(donorPageBefore);
    expect(await chunkState(donor.chunkId)).toEqual(donorChunkBefore);
  });

  test('rolls back the whole row transaction on a post-write failure', async () => {
    const f = await fixture('tx-rollback');
    const donor = await seedDonor(f.hash);
    const donorPageBefore = await pageState(donor.pageId);
    const donorChunkBefore = await chunkState(donor.chunkId);
    await expect(adopt(f, { afterWritesForTest: () => { throw new Error('forced'); } })).rejects.toThrow('forced');
    expect(await engine.getPage(f.slug, { sourceId: f.sourceId, includeDeleted: true })).toBeNull();
    expect(await fileState(f.slug)).toBeNull();
    expect(await pageState(donor.pageId)).toEqual(donorPageBefore);
    expect(await chunkState(donor.chunkId)).toEqual(donorChunkBefore);
  });
});

describe('append-only receipt rollback contract', () => {
  async function rollback(receipt: ExactHashDonorAdoptionReceipt) {
    return withImageImportFence(
      token => rollbackImageFileExactHashDonor(engine, receipt, token),
      { lockRoot: fenceRoot },
    );
  }

  test('CAS-deletes only the newly-created file and target page; donor remains byte-identical', async () => {
    const f = await fixture('rollback-created');
    const donor = await seedDonor(f.hash);
    const donorPageBefore = await pageState(donor.pageId);
    const donorChunkBefore = await chunkState(donor.chunkId);
    const beforeClock = await engine.executeRaw<{ value: number }>(
      `SELECT last_value AS value FROM page_generation_clock_seq`,
    );
    const adopted = await adopt(f);
    expect(adopted.status).toBe('adopted');
    if (adopted.status !== 'adopted') return;
    const rolled = await rollback(adopted.receipt);
    expect(rolled).toEqual({
      status: 'rolled_back',
      target_page_id: adopted.receipt.target_page_id,
      target_chunk_id: adopted.receipt.target_chunk_id,
      deleted_file_id: adopted.receipt.file_id,
      sequence_advancement_restored: false,
    });
    expect(await pageState(adopted.receipt.target_page_id)).toBeNull();
    expect(await chunkState(adopted.receipt.target_chunk_id)).toBeNull();
    expect(await fileState(f.slug)).toBeNull();
    expect(await pageState(donor.pageId)).toEqual(donorPageBefore);
    expect(await chunkState(donor.chunkId)).toEqual(donorChunkBefore);
    const afterClock = await engine.executeRaw<{ value: number }>(
      `SELECT last_value AS value FROM page_generation_clock_seq`,
    );
    expect(Number(afterClock[0].value)).toBeGreaterThan(Number(beforeClock[0].value));
  });

  test('never deletes a pre-existing same-hash files row', async () => {
    const f = await fixture('rollback-preserve');
    await seedDonor(f.hash);
    await engine.executeRaw(
      `INSERT INTO files (source_id, filename, storage_path, content_hash, metadata)
       VALUES ('default', 'legacy.png', $1, $2, '{"keep":true}'::jsonb)`,
      [f.slug, f.hash],
    );
    const before = await fileState(f.slug);
    const adopted = await adopt(f);
    expect(adopted.status).toBe('adopted');
    if (adopted.status !== 'adopted') return;
    expect(adopted.receipt.file_disposition).toBe('preserved_existing');
    const rolled = await rollback(adopted.receipt);
    expect(rolled.deleted_file_id).toBeNull();
    expect(await fileState(f.slug)).toEqual(before);
  });

  test('refuses rollback when an alias is inserted after adoption', async () => {
    const f = await fixture('rollback-alias-drift');
    await seedDonor(f.hash);
    const adopted = await adopt(f);
    expect(adopted.status).toBe('adopted');
    if (adopted.status !== 'adopted') return;
    await engine.executeRaw(
      `INSERT INTO page_aliases (source_id, alias_norm, slug)
       VALUES ($1, 'late alias', $2)`,
      [f.sourceId, f.slug],
    );

    await expect(rollback(adopted.receipt))
      .rejects.toBeInstanceOf(ExactHashDonorRollbackConflictError);
    expect(await pageState(adopted.receipt.target_page_id)).not.toBeNull();
    expect(await fileState(f.slug)).not.toBeNull();
    expect(await engine.executeRaw(
      `SELECT alias_norm FROM page_aliases WHERE source_id=$1 AND slug=$2`,
      [f.sourceId, f.slug],
    )).toEqual([{ alias_norm: 'late alias' }]);
  });

  test('refuses rollback after page, chunk, created-file, or dependent drift', async () => {
    for (const kind of ['page', 'chunk', 'file', 'file_ledger', 'raw_data', 'extra_file'] as const) {
      await resetDonorTestState();
      const f = await fixture(`rollback-drift-${kind}`);
      await seedDonor(f.hash);
      const adopted = await adopt(f);
      expect(adopted.status).toBe('adopted');
      if (adopted.status !== 'adopted') continue;
      if (kind === 'page') {
        await engine.executeRaw(`UPDATE pages SET title='drift' WHERE id=$1`, [adopted.receipt.target_page_id]);
      } else if (kind === 'chunk') {
        await engine.executeRaw(`UPDATE content_chunks SET chunk_text='drift' WHERE id=$1`, [adopted.receipt.target_chunk_id]);
      } else if (kind === 'file') {
        await engine.executeRaw(`UPDATE files SET filename='drift' WHERE id=$1`, [adopted.receipt.file_id]);
      } else if (kind === 'file_ledger') {
        await engine.executeRaw(
          `CREATE TABLE IF NOT EXISTS file_migration_ledger (
             file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
             storage_path_old TEXT NOT NULL,
             storage_path_new TEXT NOT NULL,
             status TEXT NOT NULL
           )`,
        );
        await engine.executeRaw(
          `INSERT INTO file_migration_ledger
             (file_id, storage_path_old, storage_path_new, status)
           VALUES ($1, $2, $3, 'pending')`,
          [adopted.receipt.file_id, f.slug, `${f.slug}.migrated`],
        );
      } else if (kind === 'raw_data') {
        await engine.executeRaw(
          `INSERT INTO raw_data (page_id, source, data) VALUES ($1, 'drift', '{}'::jsonb)`,
          [adopted.receipt.target_page_id],
        );
      } else {
        await engine.executeRaw(
          `INSERT INTO files
             (source_id, page_slug, page_id, filename, storage_path, content_hash)
           VALUES ($1, $2, $3, 'extra.png', $4, $5)`,
          [
            f.sourceId,
            f.slug,
            adopted.receipt.target_page_id,
            `${f.slug}.extra`,
            f.hash,
          ],
        );
      }
      await expect(rollback(adopted.receipt)).rejects.toBeInstanceOf(ExactHashDonorRollbackConflictError);
      expect(await pageState(adopted.receipt.target_page_id)).not.toBeNull();
    }
  });
});

describe('bounded command orchestration', () => {
  test('donor binds metadata snapshot without changing paid OCR dev/inode behavior', async () => {
    const f = await fixture('metadata-drift');
    await seedDonor(f.hash);
    const entry = await validated(f);
    const changedTime = new Date(entry.file_identity.mtime_ms + 60_000);
    utimesSync(f.filePath, changedTime, changedTime);

    expect(readImageOcrSourceFile({
      filePath: entry.file_path,
      imageSlug: entry.slug,
      registeredRoot: entry.registered_root,
      expectedHash: entry.sha256,
      expectedFileIdentity: entry.file_identity,
    })).toEqual(TINY_PNG);
    await expect(withImageImportFence(
      token => adoptValidatedImageDonorEntry(engine, entry, token),
      { lockRoot: fenceRoot },
    )).rejects.toThrow(/physical source state changed/i);
  });

  test('revalidates physical/root identity after preflight and fails with zero provider/budget counters', async () => {
    const f = await fixture('physical-drift');
    await seedDonor(f.hash);
    const { result: report, output } = await captureStdout(() => runImageDonorAdopt(engine, [
      f.manifestPath, '--max-images', '1', '--yes',
    ], {
      imageImportFenceRoot: fenceRoot,
      beforeEntry: () => writeFileSync(f.filePath, Buffer.from('changed')),
      now: new Date('2026-08-12T00:00:00.000Z'),
    }));
    expect(JSON.parse(output)).toEqual(report);
    expect(report).toMatchObject({
      requested: 1,
      processed: 1,
      succeeded: 0,
      skipped: 0,
      failed: 1,
      provider_attempts: 0,
      model_calls: 0,
      gateway_calls: 0,
      ocr_budget_reservations: 0,
      ocr_budget_usd: 0,
      status: 'failed',
    });
    expect(await engine.getPage(f.slug, { sourceId: f.sourceId, includeDeleted: true })).toBeNull();
  });

  test('rejects registered-root replacement after preflight and never reaches network transport', async () => {
    const f = await fixture('root-drift');
    await seedDonor(f.hash);
    const movedRoot = `${f.root}-moved`;
    let fetchCalls = 0;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () => {
      fetchCalls++;
      throw new Error('network transport must be unreachable');
    }) as unknown as typeof fetch;
    try {
      const { result: report } = await captureStdout(() => runImageDonorAdopt(engine, [
        f.manifestPath, '--max-images', '1', '--yes',
      ], {
        imageImportFenceRoot: fenceRoot,
        beforeEntry: () => {
          renameSync(f.root, movedRoot);
          mkdirSync(f.root);
        },
      }));
      expect(report).toMatchObject({
        processed: 1,
        failed: 1,
        provider_attempts: 0,
        model_calls: 0,
        gateway_calls: 0,
        ocr_budget_reservations: 0,
        ocr_budget_usd: 0,
      });
      expect(report?.first_failure?.code).toBe('physical_state_changed');
      expect(fetchCalls).toBe(0);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('preflights the entire manifest, including rows beyond the cap, before writes', async () => {
    const f = await fixture('full-preflight');
    await seedDonor(f.hash);
    writeFileSync(f.manifestPath, `${JSON.stringify({
      source_id: f.sourceId,
      slug: f.slug,
      file_path: f.filePath,
      sha256: f.hash,
    })}\n${JSON.stringify({
      source_id: f.sourceId,
      slug: 'images/missing.png',
      file_path: join(f.root, 'images/missing.png'),
      sha256: f.hash,
    })}\n`);
    const { result: report } = await captureStdout(() => runImageDonorAdopt(engine, [
      f.manifestPath, '--max-images', '1', '--yes',
    ], { imageImportFenceRoot: fenceRoot }));
    expect(report).toMatchObject({
      requested: 0,
      processed: 0,
      succeeded: 0,
      failed: 0,
      status: 'rejected',
      terminal_error: 'manifest_invalid',
    });
    expect(await engine.getPage(f.slug, { sourceId: f.sourceId, includeDeleted: true })).toBeNull();
  });

  test('stops at max-images and reports a deterministic manifest-ordered schema', async () => {
    const f = await fixture('cap');
    const lines: string[] = [];
    for (let index = 0; index < 3; index++) {
      const slug = `images/${index}.png`;
      const path = join(f.root, slug);
      writeFileSync(path, TINY_PNG);
      lines.push(JSON.stringify({ source_id: f.sourceId, slug, file_path: path, sha256: f.hash }));
    }
    writeFileSync(f.manifestPath, `${lines.join('\n')}\n`);
    const makeRun = async () => {
      await seedDonor(f.hash, { slug: 'donors/cap.png' });
      return captureStdout(() => runImageDonorAdopt(engine, [
        f.manifestPath, '--max-images', '2', '--yes',
      ], { imageImportFenceRoot: fenceRoot, now: new Date('2026-08-12T00:00:00.000Z') }));
    };
    const first = await makeRun();
    if (!first.result) throw new Error('expected donor adoption report');
    expect(first.result).toMatchObject({
      requested: 3,
      processed: 2,
      succeeded: 2,
      skipped: 1,
      failed: 0,
      provider_attempts: 0,
      model_calls: 0,
      gateway_calls: 0,
      ocr_budget_reservations: 0,
      ocr_budget_usd: 0,
      status: 'cap_reached',
    });
    expect(first.result.results).toHaveLength(2);
    expect(first.result.results.every(row => row.status === 'adopted')).toBe(true);
    expect(first.result.results.every(row => row.file_disposition === 'created')).toBe(true);
    expect(first.result.manifest_hash).toBe(sha256(readFileSync(f.manifestPath)));

    await resetDonorTestState();
    await engine.executeRaw(
      `INSERT INTO sources (id, name, local_path, config) VALUES ($1, $1, $2, '{}'::jsonb)`,
      [f.sourceId, f.root],
    );
    const second = await makeRun();
    if (!second.result) throw new Error('expected second donor adoption report');
    const stableSummary = (report: typeof first.result) => ({
      ...report,
      results: report.results.map(({ receipt: _receipt, ...row }) => row),
    });
    expect(stableSummary(first.result)).toEqual(stableSummary(second.result));
    expect(first.output).toBe(`${JSON.stringify(first.result)}\n`);
  });

  test('strict manifest rejects symlink/root/hash drift before any row transaction', async () => {
    const f = await fixture('manifest-drift');
    await seedDonor(f.hash);
    const real = join(f.root, 'images/real.png');
    renameSync(f.filePath, real);
    symlinkSync(real, f.filePath);
    expect(lstatSync(f.filePath).isSymbolicLink()).toBe(true);
    await expect(parseAndValidateImageDonorManifest(engine, f.manifestPath)).rejects.toThrow();
    expect(await engine.getPage(f.slug, { sourceId: f.sourceId, includeDeleted: true })).toBeNull();
  });
});
