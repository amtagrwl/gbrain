import { afterAll, beforeAll, beforeEach, describe, expect, test } from 'bun:test';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { OperationError, operationsByName, type OperationContext } from '../src/core/operations.ts';
import { MAX_ARTIFACT_IMAGE_BYTES } from '../src/core/read-artifact.ts';
import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import { resetPgliteState } from './helpers/reset-pglite.ts';

const SOURCE_ID = 'dept-a';
const PAGE_SLUG = 'assets/chart.png';
const PARENT_PAGE_SLUG = 'assets/assets';
const PNG_BYTES = Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  Buffer.from('non-ocr-visual-fixture'),
]);

let engine: PGLiteEngine;
let sourceRoot: string;
const sourceRoots: string[] = [];

beforeAll(async () => {
  engine = new PGLiteEngine();
  await engine.connect({});
  await engine.initSchema();
});

afterAll(async () => {
  await engine.disconnect();
  for (const root of sourceRoots) rmSync(root, { recursive: true, force: true });
});

beforeEach(async () => {
  await resetPgliteState(engine);
  sourceRoot = mkdtempSync(join(tmpdir(), 'gbrain-read-artifact-'));
  sourceRoots.push(sourceRoot);
});

async function seedImage(bytes = PNG_BYTES): Promise<string> {
  const hash = createHash('sha256').update(bytes).digest('hex');
  const filePath = join(sourceRoot, PAGE_SLUG);
  mkdirSync(dirname(filePath), { recursive: true });
  writeFileSync(filePath, bytes);
  await engine.executeRaw(
    `INSERT INTO sources (id, name, local_path, config, archived, created_at)
     VALUES ($1, $1, $2, '{}'::jsonb, false, NOW())`,
    [SOURCE_ID, sourceRoot],
  );
  await engine.putPage(PARENT_PAGE_SLUG, {
    type: 'concept',
    title: 'Synthetic parent email',
    compiled_truth: 'Synthetic parent body.',
    timeline: '',
    frontmatter: { message_id: '<synthetic-parent@example.invalid>' },
  }, { sourceId: SOURCE_ID });
  await engine.putPage(PAGE_SLUG, {
    type: 'image',
    page_kind: 'image',
    title: 'chart.png',
    compiled_truth: '',
    timeline: '',
    frontmatter: { type: 'image', mime_type: 'image/png' },
    content_hash: hash,
  }, { sourceId: SOURCE_ID });
  const page = await engine.getPage(PAGE_SLUG, { sourceId: SOURCE_ID });
  if (!page) throw new Error('failed to seed image page');
  await engine.upsertFile({
    source_id: SOURCE_ID,
    page_slug: PAGE_SLUG,
    page_id: page.id,
    filename: 'chart.png',
    storage_path: PAGE_SLUG,
    mime_type: 'image/png',
    size_bytes: bytes.length,
    content_hash: hash,
  });
  return hash;
}

async function seedLegacyImage(bytes = PNG_BYTES): Promise<string> {
  const hash = createHash('sha256').update(bytes).digest('hex');
  const filePath = join(sourceRoot, PAGE_SLUG);
  mkdirSync(dirname(filePath), { recursive: true });
  writeFileSync(filePath, bytes);
  await engine.executeRaw(
    `INSERT INTO sources (id, name, local_path, config, archived, created_at)
     VALUES ($1, $1, $2, '{}'::jsonb, false, NOW())`,
    [SOURCE_ID, sourceRoot],
  );
  await engine.putPage(PARENT_PAGE_SLUG, {
    type: 'concept',
    title: 'Synthetic parent email',
    compiled_truth: 'Synthetic parent body.',
    timeline: '',
    frontmatter: { message_id: '<synthetic-parent@example.invalid>' },
  }, { sourceId: SOURCE_ID });
  await engine.putPage(PAGE_SLUG, {
    type: 'image',
    page_kind: 'image',
    title: 'chart.png',
    compiled_truth: '',
    timeline: '',
    frontmatter: { type: 'image', mime_type: 'image/png' },
    content_hash: hash,
  }, { sourceId: 'default' });
  const page = await engine.getPage(PAGE_SLUG, { sourceId: 'default' });
  if (!page) throw new Error('failed to seed legacy image page');
  await engine.upsertFile({
    source_id: 'default',
    page_slug: PAGE_SLUG,
    page_id: page.id,
    filename: 'chart.png',
    storage_path: PAGE_SLUG,
    mime_type: 'image/png',
    size_bytes: bytes.length,
    content_hash: hash,
  });
  return hash;
}

function ctx(overrides: Partial<OperationContext> = {}): OperationContext {
  return {
    engine,
    config: {} as OperationContext['config'],
    logger: console as OperationContext['logger'],
    dryRun: false,
    remote: true,
    sourceId: SOURCE_ID,
    auth: {
      token: 'test-token',
      clientId: 'test-client',
      scopes: ['read'],
      allowedSources: [SOURCE_ID],
    },
    ...overrides,
  };
}

async function addSource(sourceId: string): Promise<string> {
  const root = mkdtempSync(join(tmpdir(), 'gbrain-read-artifact-source-'));
  sourceRoots.push(root);
  await engine.executeRaw(
    `INSERT INTO sources (id, name, local_path, config, archived, created_at)
     VALUES ($1, $1, $2, '{}'::jsonb, false, NOW())`,
    [sourceId, root],
  );
  return root;
}

async function expectArtifactError(
  params: Record<string, unknown>,
  code: string,
  context = ctx(),
): Promise<OperationError> {
  try {
    await operationsByName.read_artifact.handler(context, {
      parent_page_slug: PARENT_PAGE_SLUG,
      ...params,
    });
    throw new Error(`expected ${code}`);
  } catch (error) {
    expect(error).toBeInstanceOf(OperationError);
    expect((error as OperationError).code).toBe(code);
    expect((error as Error).message).not.toContain(sourceRoot);
    return error as OperationError;
  }
}

describe('read_artifact', () => {
  test('publishes exact source, child, parent, and optional hash coordinates as a read-scoped operation', () => {
    const op = operationsByName.read_artifact;
    expect(op.scope).toBe('read');
    expect(op.localOnly).not.toBe(true);
    expect(Object.keys(op.params)).toEqual([
      'source_id',
      'page_slug',
      'parent_page_slug',
      'content_hash',
    ]);
    expect(op.params.source_id.required).toBe(true);
    expect(op.params.page_slug.required).toBe(true);
    expect(op.params.parent_page_slug.required).toBe(true);
    expect(op.params.content_hash.required).not.toBe(true);
  });

  test('returns bounded original image bytes for an exact authorized page and hash', async () => {
    const hash = await seedImage();
    const op = operationsByName.read_artifact;

    expect(op).toBeDefined();
    const result = await op.handler(ctx(), {
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      parent_page_slug: PARENT_PAGE_SLUG,
      content_hash: hash,
    });

    expect(result).toEqual({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      mime_type: 'image/png',
      size_bytes: PNG_BYTES.length,
      content_hash: hash,
      content_base64: PNG_BYTES.toString('base64'),
    });
    expect(result).not.toHaveProperty('storage_path');
    expect(result).not.toHaveProperty('url');
  });

  test('verifies the stored page hash when the caller has no search-time content hash', async () => {
    const hash = await seedImage();

    const result = await operationsByName.read_artifact.handler(ctx(), {
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      parent_page_slug: PARENT_PAGE_SLUG,
    });

    expect(result).toMatchObject({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
      content_base64: PNG_BYTES.toString('base64'),
    });
  });

  test('bridges jointly default-owned legacy child records through the requested source root', async () => {
    const hash = await seedLegacyImage();

    const result = await operationsByName.read_artifact.handler(ctx(), {
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      parent_page_slug: PARENT_PAGE_SLUG,
    });

    expect(result).toMatchObject({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
      content_base64: PNG_BYTES.toString('base64'),
    });
    expect(result).not.toHaveProperty('storage_path');
    expect(result).not.toHaveProperty('url');
  });

  test('rejects a wrong canonical parent coordinate', async () => {
    const hash = await seedImage();
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      parent_page_slug: 'other/other',
      content_hash: hash,
    }, 'artifact_mismatch');
  });

  test('rejects a canonical parent that exists only in another source', async () => {
    const hash = await seedLegacyImage();
    await engine.executeRaw(
      'DELETE FROM pages WHERE source_id = $1 AND slug = $2',
      [SOURCE_ID, PARENT_PAGE_SLUG],
    );
    await engine.putPage(PARENT_PAGE_SLUG, {
      type: 'concept',
      title: 'Wrong-source parent',
      compiled_truth: '',
      timeline: '',
      frontmatter: { message_id: '<wrong-source@example.invalid>' },
    }, { sourceId: 'default' });

    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'artifact_not_found');
  });

  test('rejects a forged direct-convention parent without canonical email identity', async () => {
    const hash = await seedLegacyImage();
    await engine.executeRaw(
      `UPDATE pages SET frontmatter = '{"message_id":"not-canonical"}'::jsonb
       WHERE source_id = $1 AND slug = $2`,
      [SOURCE_ID, PARENT_PAGE_SLUG],
    );

    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'artifact_mismatch');
  });

  test('rejects a non-direct child even when the named parent exists', async () => {
    const hash = await seedImage();
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: 'assets/nested/chart.png',
      parent_page_slug: PARENT_PAGE_SLUG,
      content_hash: hash,
    }, 'artifact_mismatch');
  });

  test('rejects legacy child page and file records with mismatched ownership', async () => {
    const hash = await seedLegacyImage();
    await engine.executeRaw(
      'UPDATE files SET source_id = $1 WHERE source_id = $2 AND page_slug = $3',
      [SOURCE_ID, 'default', PAGE_SLUG],
    );
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'artifact_mismatch');
  });

  test('rejects an ambiguous second file record for the same child page', async () => {
    const hash = await seedImage();
    const page = await engine.getPage(PAGE_SLUG, { sourceId: SOURCE_ID });
    if (!page) throw new Error('failed to reload image page');
    await engine.upsertFile({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      page_id: page.id,
      filename: 'chart-copy.png',
      storage_path: 'assets/chart-copy.png',
      mime_type: 'image/png',
      size_bytes: PNG_BYTES.length,
      content_hash: hash,
    });
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'artifact_mismatch');
  });

  test('rejects a source outside the caller federated grant even with a citation coordinate', async () => {
    const hash = await seedImage();
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
      citation: `${SOURCE_ID}:${PAGE_SLUG}`,
    }, 'permission_denied', ctx({
      sourceId: 'dept-b',
      auth: {
        token: 'test-token',
        clientId: 'test-client',
        scopes: ['read'],
        allowedSources: ['dept-b'],
      },
    }));
  });

  test('falls back to the scalar source grant and fails closed when it does not match', async () => {
    const hash = await seedImage();
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'permission_denied', ctx({
      sourceId: 'dept-b',
      auth: {
        token: 'test-token',
        clientId: 'test-client',
        scopes: ['read'],
        allowedSources: [],
      },
    }));
  });

  test('fails closed when a remote caller has no source grant', async () => {
    const hash = await seedImage();
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'permission_denied', ctx({ sourceId: undefined, auth: undefined }));
  });

  test('uses the scalar source grant when federated_read is unavailable', async () => {
    const hash = await seedImage();
    const result = await operationsByName.read_artifact.handler(ctx({
      auth: {
        token: 'test-token',
        clientId: 'test-client',
        scopes: ['read'],
        allowedSources: undefined,
      },
    }), {
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      parent_page_slug: PARENT_PAGE_SLUG,
      content_hash: hash,
    });

    expect(result).toMatchObject({ source_id: SOURCE_ID, page_slug: PAGE_SLUG, content_hash: hash });
  });

  test('treats an undefined remote marker as untrusted', async () => {
    const hash = await seedImage();
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'permission_denied', ctx({
      remote: undefined as unknown as boolean,
      sourceId: undefined,
      auth: undefined,
    }));
  });

  test('does not resolve the same slug from a different authorized source', async () => {
    const hash = await seedImage();
    await addSource('dept-b');
    await expectArtifactError({
      source_id: 'dept-b',
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'artifact_not_found', ctx({
      auth: {
        token: 'test-token',
        clientId: 'test-client',
        scopes: ['read'],
        allowedSources: [SOURCE_ID, 'dept-b'],
      },
    }));
  });

  test('rejects file metadata linked to the page but belonging to another source', async () => {
    const hash = await seedImage();
    await addSource('dept-b');
    await engine.executeRaw(
      'UPDATE files SET source_id = $1 WHERE source_id = $2 AND page_slug = $3',
      ['dept-b', SOURCE_ID, PAGE_SLUG],
    );
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'artifact_mismatch');
  });

  test('rejects a stale requested content hash', async () => {
    await seedImage();
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: '0'.repeat(64),
    }, 'artifact_stale');
  });

  test.each([null, ''])('rejects an image page whose content hash is %p', async (pageHash) => {
    const hash = await seedImage();
    await engine.executeRaw(
      'UPDATE pages SET content_hash = $1 WHERE source_id = $2 AND slug = $3',
      [pageHash, SOURCE_ID, PAGE_SLUG],
    );
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'artifact_stale');
  });

  test('rejects bytes whose hash no longer matches the approved record', async () => {
    const hash = await seedImage();
    writeFileSync(join(sourceRoot, PAGE_SLUG), Buffer.concat([PNG_BYTES, Buffer.from('changed')]));
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'artifact_stale');
  });

  test('rejects missing physical bytes without exposing the host path', async () => {
    const hash = await seedImage();
    rmSync(join(sourceRoot, PAGE_SLUG));
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'artifact_unavailable');
  });

  test('rejects a source file symlink that escapes the approved source root', async () => {
    const hash = await seedImage();
    const externalRoot = mkdtempSync(join(tmpdir(), 'gbrain-read-artifact-external-'));
    sourceRoots.push(externalRoot);
    const externalPath = join(externalRoot, 'unrelated.png');
    writeFileSync(externalPath, PNG_BYTES);
    rmSync(join(sourceRoot, PAGE_SLUG));
    symlinkSync(externalPath, join(sourceRoot, PAGE_SLUG));
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'artifact_unavailable');
  });

  test('rejects unsupported stored MIME types before returning bytes', async () => {
    const hash = await seedImage();
    await engine.executeRaw(
      'UPDATE files SET mime_type = $1 WHERE source_id = $2 AND page_slug = $3',
      ['image/gif', SOURCE_ID, PAGE_SLUG],
    );
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'unsupported_media_type');
  });

  test('rejects a non-image page even when it has file metadata', async () => {
    const hash = await seedImage();
    await engine.executeRaw(
      'UPDATE pages SET page_kind = $1, type = $2 WHERE source_id = $3 AND slug = $4',
      ['markdown', 'note', SOURCE_ID, PAGE_SLUG],
    );
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'unsupported_media_type');
  });

  test('rejects MIME metadata that disagrees with the image signature', async () => {
    const hash = await seedImage();
    await engine.executeRaw(
      'UPDATE files SET mime_type = $1 WHERE source_id = $2 AND page_slug = $3',
      ['image/jpeg', SOURCE_ID, PAGE_SLUG],
    );
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'artifact_mismatch');
  });

  test('rejects oversized metadata without reading the artifact', async () => {
    const hash = await seedImage();
    await engine.executeRaw(
      'UPDATE files SET size_bytes = $1 WHERE source_id = $2 AND page_slug = $3',
      [MAX_ARTIFACT_IMAGE_BYTES + 1, SOURCE_ID, PAGE_SLUG],
    );
    rmSync(join(sourceRoot, PAGE_SLUG));
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'artifact_too_large');
  });

  test('bounds the physical read even when stored size metadata understates the file', async () => {
    const bytes = Buffer.concat([
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
      Buffer.alloc(MAX_ARTIFACT_IMAGE_BYTES, 0x61),
    ]);
    const hash = await seedImage(bytes);
    await engine.executeRaw(
      'UPDATE files SET size_bytes = $1 WHERE source_id = $2 AND page_slug = $3',
      [MAX_ARTIFACT_IMAGE_BYTES, SOURCE_ID, PAGE_SLUG],
    );
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'artifact_too_large');
  });

  test('rejects stored size metadata that does not match the resolved bytes', async () => {
    const hash = await seedImage();
    await engine.executeRaw(
      'UPDATE files SET size_bytes = $1 WHERE source_id = $2 AND page_slug = $3',
      [PNG_BYTES.length - 1, SOURCE_ID, PAGE_SLUG],
    );
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
    }, 'artifact_stale');
  });

  test('reads through the configured storage abstraction when the source has no local root', async () => {
    const hash = await seedImage();
    await engine.executeRaw(
      'UPDATE sources SET local_path = NULL WHERE id = $1',
      [SOURCE_ID],
    );
    const result = await operationsByName.read_artifact.handler(ctx({
      config: {
        storage: {
          backend: 'local',
          bucket: 'test',
          localPath: sourceRoot,
        },
      } as OperationContext['config'],
    }), {
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      parent_page_slug: PARENT_PAGE_SLUG,
      content_hash: hash,
    });

    expect(result).toMatchObject({
      source_id: SOURCE_ID,
      page_slug: PAGE_SLUG,
      content_hash: hash,
      content_base64: PNG_BYTES.toString('base64'),
    });
  });

  test('rejects an unsafe storage object key from the approved file row before fetch', async () => {
    const hash = await seedImage();
    await engine.executeRaw('UPDATE sources SET local_path = NULL WHERE id = $1', [SOURCE_ID]);
    await engine.executeRaw(
      'UPDATE files SET storage_path = $1 WHERE source_id = $2 AND page_slug = $3',
      ['../other-bucket/secret.png', SOURCE_ID, PAGE_SLUG],
    );
    const originalFetch = globalThis.fetch;
    let fetched = false;
    globalThis.fetch = (async (_input, _init) => {
      fetched = true;
      return new Response(PNG_BYTES, { status: 200 });
    }) as typeof fetch;
    try {
      await expectArtifactError({
        source_id: SOURCE_ID,
        page_slug: PAGE_SLUG,
        content_hash: hash,
      }, 'artifact_unavailable', ctx({
        config: {
          storage: {
            backend: 'supabase',
            bucket: 'test',
            projectUrl: 'https://storage.invalid',
            serviceRoleKey: 'test-service-role-key',
          },
        } as OperationContext['config'],
      }));
      expect(fetched).toBe(false);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('rejects an unsafe object key from redirect metadata before fetch', async () => {
    const hash = await seedImage();
    rmSync(join(sourceRoot, PAGE_SLUG));
    writeFileSync(
      join(sourceRoot, `${PAGE_SLUG}.redirect.yaml`),
      'storage_path: safe/%2e%2e/secret.png\n',
    );
    const originalFetch = globalThis.fetch;
    let fetched = false;
    globalThis.fetch = (async (_input, _init) => {
      fetched = true;
      return new Response(PNG_BYTES, { status: 200 });
    }) as typeof fetch;
    try {
      await expectArtifactError({
        source_id: SOURCE_ID,
        page_slug: PAGE_SLUG,
        content_hash: hash,
      }, 'artifact_unavailable', ctx({
        config: {
          storage: {
            backend: 'supabase',
            bucket: 'test',
            projectUrl: 'https://storage.invalid',
            serviceRoleKey: 'test-service-role-key',
          },
        } as OperationContext['config'],
      }));
      expect(fetched).toBe(false);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test('rejects malformed deterministic coordinates', async () => {
    await expectArtifactError({
      source_id: SOURCE_ID,
      page_slug: '../outside.png',
      content_hash: 'not-a-sha256',
    }, 'invalid_params');
  });
});
