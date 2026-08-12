/**
 * Exact-hash image donor adoption lives in a provider-free module.
 * Keep this dependency graph storage-only: no broad importer, embedding,
 * gateway, OCR accounting, pricing, or provider imports.
 */
import { basename } from 'node:path';
import { createHash } from 'node:crypto';
import type { BrainEngine } from './engine.ts';
import {
  assertImageImportFenceToken,
  type ImageImportFenceToken,
} from './image-import-fence.ts';

export const IMAGE_DONOR_ADOPTION_POLICY_VERSION =
  'exact-hash-source-local-image-donor-min120-v1';
export const IMAGE_DONOR_MIN_OCR_CHARACTERS = 120;
export const IMAGE_DONOR_REQUIRED_SCHEMA_VERSION = 125;

export type ExactHashDonorFileDisposition = 'created' | 'preserved_existing';

export interface ExactHashDonorTarget {
  sourceId: string;
  slug: string;
  sourceRelativePath: string;
  filePath: string;
  registeredRoot: string;
  sha256: string;
  mimeType: string;
  sizeBytes: number;
  manifestHash: string;
  manifestIndex: number;
  validatePhysical(localPath: string): void;
}

export interface ExactHashDonorAdoptionReceipt {
  policy_version: typeof IMAGE_DONOR_ADOPTION_POLICY_VERSION;
  manifest_hash: string;
  manifest_index: number;
  source_id: string;
  slug: string;
  storage_path: string;
  image_sha256: string;
  ocr_sha256: string;
  donor_page_id: number;
  donor_chunk_id: number;
  donor_state_sha256: string;
  text_embedding_sha256: string;
  image_embedding_sha256: string | null;
  multimodal_embedding_sha256: string | null;
  target_page_id: number;
  target_chunk_id: number;
  target_generation: number;
  target_poststate_sha256: string;
  file_id: number;
  file_disposition: ExactHashDonorFileDisposition;
  file_poststate_sha256: string;
  receipt_sha256: string;
}

export type ExactHashDonorAdoptionResult =
  | { status: 'adopted'; receipt: ExactHashDonorAdoptionReceipt }
  | ({ status: 'idempotent' } & ExactHashDonorAdoptionReceipt);

export interface ExactHashDonorImportOptions {
  forceTargetClaimConflictForTest?: boolean;
  afterDonorLockForTest?: (
    tx: BrainEngine,
    donor: { pageId: number; chunkId: number },
  ) => void | Promise<void>;
  beforeFileInsertForTest?: (tx: BrainEngine) => void | Promise<void>;
  afterWritesForTest?: () => void | Promise<void>;
}

export class ExactHashDonorUnavailableError extends Error {
  readonly code = 'donor_unavailable';
  constructor() {
    super('No qualifying exact-hash image donor is available');
    this.name = 'ExactHashDonorUnavailableError';
  }
}

export class ExactHashDonorTargetConflictError extends Error {
  readonly code = 'target_conflict';
  constructor() {
    super('Image donor adoption target already exists or lost its conditional claim');
    this.name = 'ExactHashDonorTargetConflictError';
  }
}

export class ExactHashDonorFileConflictError extends Error {
  readonly code = 'file_conflict';
  constructor() {
    super('Global image storage path belongs to different physical bytes');
    this.name = 'ExactHashDonorFileConflictError';
  }
}

export class ExactHashDonorPostconditionError extends Error {
  readonly code = 'postcondition_failed';
  constructor() {
    super('Exact-hash image donor adoption postcondition failed');
    this.name = 'ExactHashDonorPostconditionError';
  }
}

export class ExactHashDonorSchemaVersionError extends Error {
  readonly code = 'schema_incompatible';
  constructor() {
    super(`Image donor adoption requires schema version ${IMAGE_DONOR_REQUIRED_SCHEMA_VERSION} or newer`);
    this.name = 'ExactHashDonorSchemaVersionError';
  }
}

export class ExactHashDonorRollbackConflictError extends Error {
  readonly code = 'rollback_conflict';
  constructor() {
    super('Exact-hash image donor adoption has drifted; rollback refused');
    this.name = 'ExactHashDonorRollbackConflictError';
  }
}

interface ExactHashDonorSqlRow extends Record<string, unknown> {
  donor_page_id: number;
  donor_chunk_id: number;
  donor_source_id: string;
  donor_slug: string;
  compiled_truth: string;
  embedding_signature: string | null;
  contextual_retrieval_mode: string | null;
  corpus_generation: string | null;
  chunk_text: string;
  model: string;
  token_count: number | null;
  embedded_at: Date | string | null;
  embedding: string;
  embedding_dims: number;
  embedding_image: string | null;
  embedding_multimodal: string | null;
}

interface ExactHashFileSqlRow extends Record<string, unknown> {
  id: number;
  source_id: string;
  page_slug: string | null;
  page_id: number | null;
  filename: string;
  storage_path: string;
  mime_type: string | null;
  size_bytes: number | string | null;
  content_hash: string;
  metadata: Record<string, unknown>;
  created_at: Date | string;
}

function normalizedDigestValue(value: unknown): unknown {
  if (value instanceof Date) return value.toISOString();
  if (typeof value === 'bigint') return value.toString();
  if (Array.isArray(value)) return value.map(normalizedDigestValue);
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    return Object.fromEntries(
      Object.keys(record).sort().map(key => [key, normalizedDigestValue(record[key])]),
    );
  }
  return value;
}

function exactHashStateDigest(value: unknown): string {
  return createHash('sha256')
    .update(JSON.stringify(normalizedDigestValue(value)))
    .digest('hex');
}

function vectorDigest(value: string | null): string | null {
  return value === null ? null : createHash('sha256').update(value).digest('hex');
}

function safeInteger(value: unknown): number {
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number <= 0) throw new ExactHashDonorPostconditionError();
  return number;
}

function donorState(row: ExactHashDonorSqlRow): Record<string, unknown> {
  return {
    donor_page_id: safeInteger(row.donor_page_id),
    donor_chunk_id: safeInteger(row.donor_chunk_id),
    donor_source_id: row.donor_source_id,
    donor_slug: row.donor_slug,
    compiled_truth: row.compiled_truth,
    embedding_signature: row.embedding_signature,
    contextual_retrieval_mode: row.contextual_retrieval_mode,
    corpus_generation: row.corpus_generation,
    chunk_text: row.chunk_text,
    model: row.model,
    token_count: row.token_count,
    embedded_at: row.embedded_at,
    embedding: row.embedding,
    embedding_dims: Number(row.embedding_dims),
    embedding_image: row.embedding_image,
    embedding_multimodal: row.embedding_multimodal,
  };
}

function donorQualifies(row: ExactHashDonorSqlRow): boolean {
  const text = String(row.compiled_truth ?? '');
  const model = String(row.model ?? '');
  const dims = Number(row.embedding_dims);
  const signature = row.embedding_signature;
  return row.chunk_text === text
    && text.length >= IMAGE_DONOR_MIN_OCR_CHARACTERS
    && text.trim().length > 0
    && /[\p{L}\p{N}]/u.test(text)
    && model.trim().length > 0
    && Number.isSafeInteger(dims)
    && dims > 0
    && (row.token_count === null || (Number.isSafeInteger(Number(row.token_count)) && Number(row.token_count) >= 0))
    && (signature === null || signature === `${model}:${dims}`)
    && (row.contextual_retrieval_mode === null || row.contextual_retrieval_mode === 'none')
    && row.corpus_generation === null;
}

async function selectExactHashDonor(
  tx: BrainEngine,
  imageHash: string,
  excludePageId: number | null,
  bound?: { pageId: number; chunkId: number },
): Promise<ExactHashDonorSqlRow | null> {
  const params: unknown[] = [imageHash];
  let boundSql = '';
  if (excludePageId !== null) {
    params.push(excludePageId);
    boundSql += ` AND p.id <> $${params.length}`;
  }
  if (bound) {
    params.push(bound.pageId);
    boundSql += ` AND p.id = $${params.length}`;
    params.push(bound.chunkId);
    boundSql += ` AND cc.id = $${params.length}`;
  }
  const rows = await tx.executeRaw<ExactHashDonorSqlRow>(
    `SELECT
       p.id AS donor_page_id,
       cc.id AS donor_chunk_id,
       p.source_id AS donor_source_id,
       p.slug AS donor_slug,
       p.compiled_truth,
       p.embedding_signature,
       p.contextual_retrieval_mode,
       p.corpus_generation,
       cc.chunk_text,
       cc.model,
       cc.token_count,
       cc.embedded_at,
       cc.embedding::text AS embedding,
       vector_dims(cc.embedding)::int AS embedding_dims,
       CASE WHEN cc.embedding_image IS NULL THEN NULL ELSE cc.embedding_image::text END AS embedding_image,
       CASE WHEN cc.embedding_multimodal IS NULL THEN NULL ELSE cc.embedding_multimodal::text END AS embedding_multimodal
     FROM pages p
     JOIN content_chunks cc ON cc.page_id = p.id
     WHERE p.deleted_at IS NULL
       AND p.page_kind = 'image'
       AND p.content_hash = $1
       AND (p.contextual_retrieval_mode IS NULL OR p.contextual_retrieval_mode = 'none')
       AND p.corpus_generation IS NULL
       AND cc.chunk_index = 0
       AND cc.chunk_source = 'image_asset'
       AND cc.modality = 'image'
       AND cc.chunk_text = p.compiled_truth
       AND length(p.compiled_truth) >= ${IMAGE_DONOR_MIN_OCR_CHARACTERS}
       AND btrim(p.compiled_truth) <> ''
       AND cc.embedding IS NOT NULL
       AND btrim(cc.model) <> ''
       AND (cc.token_count IS NULL OR cc.token_count >= 0)
       AND (p.embedding_signature IS NULL
            OR p.embedding_signature = cc.model || ':' || vector_dims(cc.embedding)::text)
       AND (SELECT count(*) FROM content_chunks one_chunk WHERE one_chunk.page_id = p.id) = 1
       ${boundSql}
     ORDER BY p.id ASC, cc.id ASC
     FOR UPDATE OF p, cc`,
    params,
  );
  return rows.find(donorQualifies) ?? null;
}

async function selectGlobalFile(
  tx: BrainEngine,
  storagePath: string,
): Promise<ExactHashFileSqlRow | null> {
  const rows = await tx.executeRaw<ExactHashFileSqlRow>(
    `SELECT id, source_id, page_slug, page_id, filename, storage_path, mime_type,
            size_bytes, content_hash, metadata, created_at
       FROM files
      WHERE storage_path = $1
      FOR UPDATE`,
    [storagePath],
  );
  return rows[0] ?? null;
}

async function readExactTargetState(
  tx: BrainEngine,
  pageId: number,
  chunkId: number,
  lock: boolean,
): Promise<Record<string, unknown> | null> {
  const pages = await tx.executeRaw<Record<string, unknown>>(
    `SELECT id, source_id, slug, type, page_kind, title, compiled_truth, timeline,
            frontmatter, content_hash, emotional_weight,
            emotional_weight_recomputed_at, created_at, updated_at, deleted_at,
            effective_date, effective_date_source, import_filename,
            salience_touched_at, last_retrieved_at, links_extracted_at,
            contextual_retrieval_mode, corpus_generation, generation,
            chunker_version, source_path, ingested_via, ingested_at,
            source_uri, source_kind, embedding_signature
            , search_vector::text AS search_vector
       FROM pages WHERE id = $1${lock ? ' FOR UPDATE' : ''}`,
    [pageId],
  );
  if (pages.length !== 1) return null;
  const chunks = await tx.executeRaw<Record<string, unknown>>(
    `SELECT id, page_id, chunk_index, chunk_text, chunk_source,
            CASE WHEN embedding IS NULL THEN NULL ELSE embedding::text END AS embedding,
            model, token_count, embedded_at, created_at, language, symbol_name,
            symbol_type, start_line, end_line, parent_symbol_path, doc_comment,
            symbol_name_qualified, search_vector::text AS search_vector,
            edges_backfilled_at, modality,
            CASE WHEN embedding_image IS NULL THEN NULL ELSE embedding_image::text END AS embedding_image,
            CASE WHEN embedding_multimodal IS NULL THEN NULL ELSE embedding_multimodal::text END AS embedding_multimodal
       FROM content_chunks WHERE id = $1 AND page_id = $2${lock ? ' FOR UPDATE' : ''}`,
    [chunkId, pageId],
  );
  if (chunks.length !== 1) return null;
  const counts = await tx.executeRaw<Record<string, unknown>>(
    `SELECT
       (SELECT count(*)::int FROM content_chunks WHERE page_id=$1) AS chunks,
       (SELECT count(*)::int FROM links WHERE from_page_id=$1 OR to_page_id=$1 OR origin_page_id=$1) AS links,
       (SELECT count(*)::int FROM tags WHERE page_id=$1) AS tags,
       (SELECT count(*)::int FROM raw_data WHERE page_id=$1) AS raw_data,
       (SELECT count(*)::int FROM files WHERE page_id=$1) AS attached_files,
       (SELECT count(*)::int FROM timeline_entries WHERE page_id=$1 OR event_page_id=$1) AS timeline_entries,
       (SELECT count(*)::int FROM page_versions WHERE page_id=$1) AS versions,
       (SELECT count(*)::int FROM takes WHERE page_id=$1) AS takes,
       (SELECT count(*)::int FROM synthesis_evidence WHERE synthesis_page_id=$1) AS synthesis_evidence,
       (SELECT count(*)::int FROM code_edges_chunk WHERE from_chunk_id=$2 OR to_chunk_id=$2) AS code_edges_chunk,
       (SELECT count(*)::int FROM code_edges_symbol WHERE from_chunk_id=$2) AS code_edges_symbol`,
    [pageId, chunkId],
  );
  return { page: pages[0], chunk: chunks[0], related_counts: counts[0] };
}

function adoptionProvenance(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const frontmatter = value as Record<string, unknown>;
  const nested = frontmatter.image_donor_adoption;
  return nested && typeof nested === 'object' && !Array.isArray(nested)
    ? nested as Record<string, unknown>
    : null;
}

function finalizeReceipt(
  receipt: Omit<ExactHashDonorAdoptionReceipt, 'receipt_sha256'>,
): ExactHashDonorAdoptionReceipt {
  return Object.freeze({
    ...receipt,
    receipt_sha256: exactHashStateDigest(receipt),
  });
}

function validateReceipt(receipt: ExactHashDonorAdoptionReceipt): void {
  const { receipt_sha256: supplied, ...body } = receipt;
  if (supplied !== exactHashStateDigest(body)) throw new ExactHashDonorRollbackConflictError();
}

function canonicalFrontmatter(
  target: ExactHashDonorTarget,
  donor: ExactHashDonorSqlRow,
  file: ExactHashFileSqlRow,
  fileDisposition: ExactHashDonorFileDisposition,
  baseline: {
    targetGeneration: number;
    pageCreatedAt: unknown;
    chunkCreatedAt: unknown;
    pageSearchVector: string;
    chunkSearchVector: string;
  },
): Record<string, unknown> {
  return {
    type: 'image',
    title: basename(target.sourceRelativePath),
    mime_type: target.mimeType,
    bytes: target.sizeBytes,
    image_donor_adoption: {
      policy_version: IMAGE_DONOR_ADOPTION_POLICY_VERSION,
      manifest_hash: target.manifestHash,
      manifest_index: target.manifestIndex,
      donor_page_id: safeInteger(donor.donor_page_id),
      donor_chunk_id: safeInteger(donor.donor_chunk_id),
      donor_state_sha256: exactHashStateDigest(donorState(donor)),
      image_sha256: target.sha256,
      ocr_sha256: createHash('sha256').update(donor.compiled_truth).digest('hex'),
      text_embedding_sha256: vectorDigest(donor.embedding),
      image_embedding_sha256: vectorDigest(donor.embedding_image),
      multimodal_embedding_sha256: vectorDigest(donor.embedding_multimodal),
      file_id: safeInteger(file.id),
      file_disposition: fileDisposition,
      file_poststate_sha256: exactHashStateDigest(file),
      target_generation: baseline.targetGeneration,
      target_page_created_at: normalizedDigestValue(baseline.pageCreatedAt),
      target_chunk_created_at: normalizedDigestValue(baseline.chunkCreatedAt),
      target_page_search_vector_sha256: createHash('sha256').update(baseline.pageSearchVector).digest('hex'),
      target_chunk_search_vector_sha256: createHash('sha256').update(baseline.chunkSearchVector).digest('hex'),
    },
  };
}

function targetMatchesDonor(
  state: Record<string, unknown>,
  target: ExactHashDonorTarget,
  donor: ExactHashDonorSqlRow,
  file: ExactHashFileSqlRow,
  fileDisposition: ExactHashDonorFileDisposition,
): boolean {
  const page = state.page as Record<string, unknown>;
  const chunk = state.chunk as Record<string, unknown>;
  const counts = state.related_counts as Record<string, unknown>;
  if (typeof page.search_vector !== 'string' || typeof chunk.search_vector !== 'string') return false;
  const expectedFrontmatter = canonicalFrontmatter(target, donor, file, fileDisposition, {
    targetGeneration: Number(page.generation),
    pageCreatedAt: page.created_at,
    chunkCreatedAt: chunk.created_at,
    pageSearchVector: page.search_vector,
    chunkSearchVector: chunk.search_vector,
  });
  return page.deleted_at === null
    && page.source_id === target.sourceId
    && page.slug === target.slug
    && page.type === 'image'
    && page.page_kind === 'image'
    && page.title === basename(target.sourceRelativePath)
    && page.compiled_truth === donor.compiled_truth
    && page.timeline === ''
    && exactHashStateDigest(page.frontmatter) === exactHashStateDigest(expectedFrontmatter)
    && page.content_hash === target.sha256
    && Number(page.emotional_weight) === 0
    && page.emotional_weight_recomputed_at === null
    && page.effective_date === null
    && page.effective_date_source === null
    && page.salience_touched_at === null
    && page.last_retrieved_at === null
    && page.links_extracted_at === null
    && page.embedding_signature === donor.embedding_signature
    && page.contextual_retrieval_mode === donor.contextual_retrieval_mode
    && page.corpus_generation === null
    && page.import_filename === basename(target.sourceRelativePath)
    && Number(page.chunker_version) === 1
    && page.source_path === target.sourceRelativePath
    && page.ingested_via === null
    && page.ingested_at === null
    && page.source_uri === null
    && page.source_kind === null
    && normalizedDigestValue(page.updated_at) === normalizedDigestValue(page.created_at)
    && chunk.chunk_index === 0
    && chunk.chunk_text === donor.chunk_text
    && chunk.chunk_source === 'image_asset'
    && chunk.embedding === donor.embedding
    && chunk.model === donor.model
    && Number(chunk.token_count) === Number(donor.token_count)
    && (chunk.token_count === null) === (donor.token_count === null)
    && normalizedDigestValue(chunk.embedded_at) === normalizedDigestValue(donor.embedded_at)
    && chunk.language === null
    && chunk.symbol_name === null
    && chunk.symbol_type === null
    && chunk.start_line === null
    && chunk.end_line === null
    && chunk.parent_symbol_path === null
    && chunk.doc_comment === null
    && chunk.symbol_name_qualified === null
    && page.search_vector.trim().length > 0
    && chunk.search_vector.trim().length > 0
    && chunk.edges_backfilled_at === null
    && chunk.modality === 'image'
    && chunk.embedding_image === donor.embedding_image
    && chunk.embedding_multimodal === donor.embedding_multimodal
    && Number(counts.chunks) === 1
    && Number(counts.links) === 0
    && Number(counts.tags) === 0
    && Number(counts.raw_data) === 0
    && Number(counts.attached_files) === (fileDisposition === 'created' ? 1 : 0)
    && Number(counts.timeline_entries) === 0
    && Number(counts.versions) === 0
    && Number(counts.takes) === 0
    && Number(counts.synthesis_evidence) === 0
    && Number(counts.code_edges_chunk) === 0
    && Number(counts.code_edges_symbol) === 0;
}

function targetStateForIdempotentReplay(
  state: Record<string, unknown>,
): Record<string, unknown> {
  return {
    ...state,
    page: {
      ...state.page as Record<string, unknown>,
      last_retrieved_at: null,
    },
  };
}

async function exactPriorAdoption(
  tx: BrainEngine,
  target: ExactHashDonorTarget,
  existingPage: Record<string, unknown>,
): Promise<ExactHashDonorAdoptionResult | null> {
  if (existingPage.deleted_at !== null) return null;
  const provenance = adoptionProvenance(existingPage.frontmatter);
  if (
    provenance?.policy_version !== IMAGE_DONOR_ADOPTION_POLICY_VERSION
    || provenance.manifest_hash !== target.manifestHash
    || Number(provenance.manifest_index) !== target.manifestIndex
    || provenance.image_sha256 !== target.sha256
    || !Number.isSafeInteger(Number(provenance.donor_page_id))
    || !Number.isSafeInteger(Number(provenance.donor_chunk_id))
    || !Number.isSafeInteger(Number(provenance.file_id))
    || typeof provenance.donor_state_sha256 !== 'string'
    || typeof provenance.text_embedding_sha256 !== 'string'
  ) return null;
  const fileDisposition = provenance.file_disposition;
  if (fileDisposition !== 'created' && fileDisposition !== 'preserved_existing') return null;
  const donor = await selectExactHashDonor(tx, target.sha256, Number(existingPage.id), {
    pageId: Number(provenance.donor_page_id),
    chunkId: Number(provenance.donor_chunk_id),
  });
  if (!donor) return null;
  if (
    provenance.donor_state_sha256 !== exactHashStateDigest(donorState(donor))
    || provenance.text_embedding_sha256 !== vectorDigest(donor.embedding)
    || provenance.image_embedding_sha256 !== vectorDigest(donor.embedding_image)
    || provenance.multimodal_embedding_sha256 !== vectorDigest(donor.embedding_multimodal)
  ) return null;
  const chunkRows = await tx.executeRaw<{ id: number }>(
    `SELECT id FROM content_chunks WHERE page_id=$1 ORDER BY id FOR UPDATE`,
    [existingPage.id],
  );
  if (chunkRows.length !== 1) return null;
  const targetChunkId = safeInteger(chunkRows[0].id);
  const file = await selectGlobalFile(tx, target.sourceRelativePath);
  if (
    !file
    || file.content_hash !== target.sha256
    || Number(file.id) !== Number(provenance.file_id)
    || exactHashStateDigest(file) !== provenance.file_poststate_sha256
  ) return null;
  const state = await readExactTargetState(tx, safeInteger(existingPage.id), targetChunkId, true);
  if (!state) return null;
  const replayState = targetStateForIdempotentReplay(state);
  if (!targetMatchesDonor(replayState, target, donor, file, fileDisposition)) return null;
  const page = replayState.page as Record<string, unknown>;
  const receipt = finalizeReceipt({
    policy_version: IMAGE_DONOR_ADOPTION_POLICY_VERSION,
    manifest_hash: target.manifestHash,
    manifest_index: target.manifestIndex,
    source_id: target.sourceId,
    slug: target.slug,
    storage_path: target.sourceRelativePath,
    image_sha256: target.sha256,
    ocr_sha256: createHash('sha256').update(donor.compiled_truth).digest('hex'),
    donor_page_id: safeInteger(donor.donor_page_id),
    donor_chunk_id: safeInteger(donor.donor_chunk_id),
    donor_state_sha256: exactHashStateDigest(donorState(donor)),
    text_embedding_sha256: vectorDigest(donor.embedding)!,
    image_embedding_sha256: vectorDigest(donor.embedding_image),
    multimodal_embedding_sha256: vectorDigest(donor.embedding_multimodal),
    target_page_id: safeInteger(existingPage.id),
    target_chunk_id: targetChunkId,
    target_generation: safeInteger(page.generation),
    target_poststate_sha256: exactHashStateDigest(replayState),
    file_id: safeInteger(file.id),
    file_disposition: fileDisposition,
    file_poststate_sha256: exactHashStateDigest(file),
  });
  return { status: 'idempotent', ...receipt };
}

export async function importImageFileWithExactHashDonor(
  engine: BrainEngine,
  target: ExactHashDonorTarget,
  imageImportFenceToken: ImageImportFenceToken,
  options: ExactHashDonorImportOptions = {},
): Promise<ExactHashDonorAdoptionResult> {
  assertImageImportFenceToken(imageImportFenceToken);
  return engine.transaction(async (tx) => {
    const versions = await tx.executeRaw<{ value: string }>(
      `SELECT value FROM config WHERE key='version' FOR SHARE`,
    );
    const rawVersion = versions.length === 1 ? versions[0].value : null;
    if (
      typeof rawVersion !== 'string'
      || !/^[1-9][0-9]*$/.test(rawVersion)
      || !Number.isSafeInteger(Number(rawVersion))
      || Number(rawVersion) < IMAGE_DONOR_REQUIRED_SCHEMA_VERSION
    ) throw new ExactHashDonorSchemaVersionError();

    const sources = await tx.executeRaw<{ local_path: string | null }>(
      `SELECT local_path FROM sources WHERE id=$1 FOR UPDATE`,
      [target.sourceId],
    );
    if (sources.length !== 1 || !sources[0].local_path) throw new ExactHashDonorPostconditionError();
    target.validatePhysical(sources[0].local_path);

    const existing = await tx.executeRaw<Record<string, unknown>>(
      `SELECT id, deleted_at, frontmatter FROM pages
        WHERE source_id=$1 AND slug=$2 FOR UPDATE`,
      [target.sourceId, target.slug],
    );
    if (existing.length > 0) {
      const idempotent = await exactPriorAdoption(tx, target, existing[0]);
      if (idempotent) {
        target.validatePhysical(sources[0].local_path);
        return idempotent;
      }
      throw new ExactHashDonorTargetConflictError();
    }

    const donor = await selectExactHashDonor(tx, target.sha256, null);
    if (!donor) throw new ExactHashDonorUnavailableError();
    const donorBefore = exactHashStateDigest(donorState(donor));
    await options.afterDonorLockForTest?.(tx, {
      pageId: safeInteger(donor.donor_page_id),
      chunkId: safeInteger(donor.donor_chunk_id),
    });
    const reboundDonor = await selectExactHashDonor(tx, target.sha256, null, {
      pageId: safeInteger(donor.donor_page_id),
      chunkId: safeInteger(donor.donor_chunk_id),
    });
    if (!reboundDonor || exactHashStateDigest(donorState(reboundDonor)) !== donorBefore) {
      throw new ExactHashDonorUnavailableError();
    }

    const preExistingFile = await selectGlobalFile(tx, target.sourceRelativePath);
    if (preExistingFile && preExistingFile.content_hash !== target.sha256) {
      throw new ExactHashDonorFileConflictError();
    }
    if (options.forceTargetClaimConflictForTest) {
      await tx.executeRaw(
        `INSERT INTO pages
           (source_id, slug, type, page_kind, title, compiled_truth, timeline,
            frontmatter, content_hash)
         VALUES ($1, $2, 'image', 'image', 'competing writer', 'owned', '',
                 '{}'::jsonb, 'competing-writer')`,
        [target.sourceId, target.slug],
      );
    }

    const filename = basename(target.sourceRelativePath);
    const pageRows = await tx.executeRaw<{
      id: number;
      generation: number;
      created_at: Date | string;
      search_vector: string;
    }>(
      `INSERT INTO pages
         (source_id, slug, type, page_kind, title, compiled_truth, timeline,
          frontmatter, content_hash, embedding_signature,
          contextual_retrieval_mode, corpus_generation, import_filename, source_path)
       VALUES ($1, $2, 'image', 'image', $3, $4, '', '{}'::jsonb, $5,
               $6, $7, NULL, $8, $9)
       ON CONFLICT (source_id, slug) DO NOTHING
       RETURNING id, generation, created_at, search_vector::text AS search_vector`,
      [
        target.sourceId,
        target.slug,
        filename,
        reboundDonor.compiled_truth,
        target.sha256,
        reboundDonor.embedding_signature,
        reboundDonor.contextual_retrieval_mode,
        filename,
        target.sourceRelativePath,
      ],
    );
    if (pageRows.length !== 1) throw new ExactHashDonorTargetConflictError();
    const targetPageId = safeInteger(pageRows[0].id);

    const chunkRows = await tx.executeRaw<{
      id: number;
      created_at: Date | string;
      search_vector: string;
    }>(
      `INSERT INTO content_chunks
         (page_id, chunk_index, chunk_text, chunk_source, embedding, model,
          token_count, embedded_at, modality, embedding_image, embedding_multimodal)
       SELECT $1, 0, cc.chunk_text, 'image_asset', cc.embedding, cc.model,
              cc.token_count, cc.embedded_at, 'image', cc.embedding_image,
              cc.embedding_multimodal
         FROM pages p
         JOIN content_chunks cc ON cc.page_id=p.id
        WHERE p.id=$2 AND cc.id=$3
          AND p.deleted_at IS NULL
          AND p.page_kind='image'
          AND p.content_hash=$4
          AND cc.chunk_index=0
          AND cc.chunk_source='image_asset'
          AND cc.modality='image'
          AND cc.chunk_text=p.compiled_truth
          AND cc.embedding IS NOT NULL
          AND (SELECT count(*) FROM content_chunks one_chunk WHERE one_chunk.page_id=p.id)=1
       RETURNING id, created_at, search_vector::text AS search_vector`,
      [targetPageId, reboundDonor.donor_page_id, reboundDonor.donor_chunk_id, target.sha256],
    );
    if (chunkRows.length !== 1) throw new ExactHashDonorUnavailableError();
    const targetChunkId = safeInteger(chunkRows[0].id);
    if (targetPageId === safeInteger(reboundDonor.donor_page_id)
      || targetChunkId === safeInteger(reboundDonor.donor_chunk_id)) {
      throw new ExactHashDonorPostconditionError();
    }

    let file = preExistingFile;
    let fileDisposition: ExactHashDonorFileDisposition = 'preserved_existing';
    if (!file) {
      await options.beforeFileInsertForTest?.(tx);
      const inserted = await tx.executeRaw<ExactHashFileSqlRow>(
        `INSERT INTO files
           (source_id, page_slug, page_id, filename, storage_path, mime_type,
            size_bytes, content_hash, metadata)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, '{}'::jsonb)
         ON CONFLICT (storage_path) DO NOTHING
         RETURNING id, source_id, page_slug, page_id, filename, storage_path,
                   mime_type, size_bytes, content_hash, metadata, created_at`,
        [
          target.sourceId,
          target.slug,
          targetPageId,
          filename,
          target.sourceRelativePath,
          target.mimeType,
          target.sizeBytes,
          target.sha256,
        ],
      );
      if (inserted.length === 1) {
        file = inserted[0];
        fileDisposition = 'created';
      } else {
        file = await selectGlobalFile(tx, target.sourceRelativePath);
        if (!file || file.content_hash !== target.sha256) throw new ExactHashDonorFileConflictError();
      }
    }
    if (!file || file.content_hash !== target.sha256) throw new ExactHashDonorFileConflictError();
    const fileId = safeInteger(file.id);
    const filePoststateSha256 = exactHashStateDigest(file);
    const predictedFinalGeneration = safeInteger(pageRows[0].generation) + 1;
    const finalFrontmatter = canonicalFrontmatter(target, reboundDonor, file, fileDisposition, {
      targetGeneration: predictedFinalGeneration,
      pageCreatedAt: pageRows[0].created_at,
      chunkCreatedAt: chunkRows[0].created_at,
      pageSearchVector: pageRows[0].search_vector,
      chunkSearchVector: chunkRows[0].search_vector,
    });
    await tx.executeRaw(
      `UPDATE pages SET frontmatter=$1::text::jsonb, updated_at=created_at WHERE id=$2`,
      [JSON.stringify(finalFrontmatter), targetPageId],
    );

    await options.afterWritesForTest?.();
    const donorAfter = await selectExactHashDonor(tx, target.sha256, targetPageId, {
      pageId: safeInteger(reboundDonor.donor_page_id),
      chunkId: safeInteger(reboundDonor.donor_chunk_id),
    });
    const targetState = await readExactTargetState(tx, targetPageId, targetChunkId, true);
    const fileAfter = await selectGlobalFile(tx, target.sourceRelativePath);
    if (
      !donorAfter
      || exactHashStateDigest(donorState(donorAfter)) !== donorBefore
      || !targetState
      || !fileAfter
      || exactHashStateDigest(fileAfter) !== filePoststateSha256
      || !targetMatchesDonor(targetState, target, donorAfter, fileAfter, fileDisposition)
    ) throw new ExactHashDonorPostconditionError();
    if (fileDisposition === 'created' && (
      fileAfter.source_id !== target.sourceId
      || fileAfter.page_slug !== target.slug
      || Number(fileAfter.page_id) !== targetPageId
      || fileAfter.filename !== filename
      || fileAfter.mime_type !== target.mimeType
      || Number(fileAfter.size_bytes) !== target.sizeBytes
    )) throw new ExactHashDonorPostconditionError();

    target.validatePhysical(sources[0].local_path);
    const targetPage = targetState.page as Record<string, unknown>;
    const receipt = finalizeReceipt({
      policy_version: IMAGE_DONOR_ADOPTION_POLICY_VERSION,
      manifest_hash: target.manifestHash,
      manifest_index: target.manifestIndex,
      source_id: target.sourceId,
      slug: target.slug,
      storage_path: target.sourceRelativePath,
      image_sha256: target.sha256,
      ocr_sha256: createHash('sha256').update(donorAfter.compiled_truth).digest('hex'),
      donor_page_id: safeInteger(donorAfter.donor_page_id),
      donor_chunk_id: safeInteger(donorAfter.donor_chunk_id),
      donor_state_sha256: donorBefore,
      text_embedding_sha256: vectorDigest(donorAfter.embedding)!,
      image_embedding_sha256: vectorDigest(donorAfter.embedding_image),
      multimodal_embedding_sha256: vectorDigest(donorAfter.embedding_multimodal),
      target_page_id: targetPageId,
      target_chunk_id: targetChunkId,
      target_generation: safeInteger(targetPage.generation),
      target_poststate_sha256: exactHashStateDigest(targetState),
      file_id: fileId,
      file_disposition: fileDisposition,
      file_poststate_sha256: filePoststateSha256,
    });
    return { status: 'adopted', receipt };
  });
}

export interface ExactHashDonorRollbackResult {
  status: 'rolled_back';
  target_page_id: number;
  target_chunk_id: number;
  deleted_file_id: number | null;
  sequence_advancement_restored: false;
}

export async function rollbackImageFileExactHashDonor(
  engine: BrainEngine,
  receipt: ExactHashDonorAdoptionReceipt,
  imageImportFenceToken: ImageImportFenceToken,
): Promise<ExactHashDonorRollbackResult> {
  assertImageImportFenceToken(imageImportFenceToken);
  validateReceipt(receipt);
  return engine.transaction(async (tx) => {
    const targetState = await readExactTargetState(
      tx,
      receipt.target_page_id,
      receipt.target_chunk_id,
      true,
    );
    if (!targetState || exactHashStateDigest(targetState) !== receipt.target_poststate_sha256) {
      throw new ExactHashDonorRollbackConflictError();
    }
    const page = targetState.page as Record<string, unknown>;
    const provenance = adoptionProvenance(page.frontmatter);
    if (
      page.source_id !== receipt.source_id
      || page.slug !== receipt.slug
      || page.content_hash !== receipt.image_sha256
      || Number(page.generation) !== receipt.target_generation
      || provenance?.policy_version !== receipt.policy_version
      || Number(provenance.donor_page_id) !== receipt.donor_page_id
      || Number(provenance.donor_chunk_id) !== receipt.donor_chunk_id
      || provenance.donor_state_sha256 !== receipt.donor_state_sha256
      || provenance.text_embedding_sha256 !== receipt.text_embedding_sha256
      || provenance.image_embedding_sha256 !== receipt.image_embedding_sha256
      || provenance.multimodal_embedding_sha256 !== receipt.multimodal_embedding_sha256
    ) throw new ExactHashDonorRollbackConflictError();

    const file = await selectGlobalFile(tx, receipt.storage_path);
    if (
      !file
      || Number(file.id) !== receipt.file_id
      || exactHashStateDigest(file) !== receipt.file_poststate_sha256
    ) throw new ExactHashDonorRollbackConflictError();

    let deletedFileId: number | null = null;
    if (receipt.file_disposition === 'created') {
      const ledgerRelations = await tx.executeRaw<{ present: boolean }>(
        `SELECT to_regclass('public.file_migration_ledger') IS NOT NULL AS present`,
      );
      if (ledgerRelations[0]?.present) {
        const ledgerRows = await tx.executeRaw<{ count: number }>(
          `SELECT count(*)::int AS count
             FROM file_migration_ledger
            WHERE file_id=$1`,
          [receipt.file_id],
        );
        if (Number(ledgerRows[0]?.count) !== 0) {
          throw new ExactHashDonorRollbackConflictError();
        }
      }
      const deleted = await tx.executeRaw<{ id: number }>(
        `DELETE FROM files WHERE id=$1 RETURNING id`,
        [receipt.file_id],
      );
      if (deleted.length !== 1) throw new ExactHashDonorRollbackConflictError();
      deletedFileId = safeInteger(deleted[0].id);
    }
    const deletedPage = await tx.executeRaw<{ id: number }>(
      `DELETE FROM pages WHERE id=$1 RETURNING id`,
      [receipt.target_page_id],
    );
    if (deletedPage.length !== 1) throw new ExactHashDonorRollbackConflictError();
    return {
      status: 'rolled_back',
      target_page_id: receipt.target_page_id,
      target_chunk_id: receipt.target_chunk_id,
      deleted_file_id: deletedFileId,
      sequence_advancement_restored: false,
    };
  });
}
