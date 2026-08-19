import { describe, test, expect, beforeAll, afterAll, afterEach } from 'bun:test';
import { mkdtempSync, rmSync, existsSync, readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { LocalStorage } from '../src/core/storage/local.ts';
import { S3Storage } from '../src/core/storage/s3.ts';
import { SupabaseStorage } from '../src/core/storage/supabase.ts';
import { createStorage, StorageReadLimitError } from '../src/core/storage.ts';

describe('LocalStorage', () => {
  let storage: LocalStorage;
  let tmpDir: string;

  beforeAll(() => {
    tmpDir = mkdtempSync(join(tmpdir(), 'gbrain-storage-test-'));
    storage = new LocalStorage(tmpDir);
  });

  afterAll(() => {
    rmSync(tmpDir, { recursive: true });
  });

  test('upload creates file', async () => {
    await storage.upload('test/file.txt', Buffer.from('hello'));
    expect(existsSync(join(tmpDir, 'test/file.txt'))).toBe(true);
  });

  test('download returns uploaded data', async () => {
    await storage.upload('test/roundtrip.bin', Buffer.from('binary data'));
    const data = await storage.download('test/roundtrip.bin');
    expect(data.toString()).toBe('binary data');
  });

  test('download enforces an optional byte limit', async () => {
    await storage.upload('test/bounded.bin', Buffer.from('1234'));
    await expect(storage.download('test/bounded.bin', 3)).rejects.toThrow('read limit');
  });

  test('download throws for missing file', async () => {
    expect(storage.download('nonexistent.txt')).rejects.toThrow('not found');
  });

  test('exists returns true for uploaded file', async () => {
    await storage.upload('test/exists.txt', Buffer.from('x'));
    expect(await storage.exists('test/exists.txt')).toBe(true);
  });

  test('exists returns false for missing file', async () => {
    expect(await storage.exists('nope.txt')).toBe(false);
  });

  test('delete removes file', async () => {
    await storage.upload('test/deleteme.txt', Buffer.from('x'));
    await storage.delete('test/deleteme.txt');
    expect(await storage.exists('test/deleteme.txt')).toBe(false);
  });

  test('delete is idempotent (missing file is ok)', async () => {
    await storage.delete('already-gone.txt');
    // No throw
  });

  test('list returns uploaded files', async () => {
    await storage.upload('listdir/a.txt', Buffer.from('a'));
    await storage.upload('listdir/b.txt', Buffer.from('b'));
    await storage.upload('listdir/sub/c.txt', Buffer.from('c'));
    const files = await storage.list('listdir');
    expect(files.length).toBe(3);
    expect(files).toContain('listdir/a.txt');
    expect(files).toContain('listdir/b.txt');
    expect(files).toContain('listdir/sub/c.txt');
  });

  test('list returns empty for missing prefix', async () => {
    const files = await storage.list('nonexistent-prefix');
    expect(files.length).toBe(0);
  });

  test('getUrl returns file:// URL', async () => {
    const url = await storage.getUrl('test/file.txt');
    expect(url.startsWith('file://')).toBe(true);
  });
});

// --- Path traversal containment ---

describe('LocalStorage path traversal', () => {
  test('blocks upload path traversal via ../', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'gbrain-traversal-'));
    try {
      const storage = new LocalStorage(tmpDir);
      await expect(storage.upload('../../etc/evil', Buffer.from('pwned'))).rejects.toThrow('Path traversal blocked');
      await expect(storage.upload('../sibling/file', Buffer.from('x'))).rejects.toThrow('Path traversal blocked');
    } finally {
      rmSync(tmpDir, { recursive: true });
    }
  });

  test('blocks download path traversal via ../', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'gbrain-traversal-'));
    try {
      const storage = new LocalStorage(tmpDir);
      await expect(storage.download('../../etc/passwd')).rejects.toThrow('Path traversal blocked');
    } finally {
      rmSync(tmpDir, { recursive: true });
    }
  });

  test('blocks delete path traversal via ../', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'gbrain-traversal-'));
    try {
      const storage = new LocalStorage(tmpDir);
      await expect(storage.delete('../../../tmp/important')).rejects.toThrow('Path traversal blocked');
    } finally {
      rmSync(tmpDir, { recursive: true });
    }
  });

  test('blocks list path traversal via ../', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'gbrain-traversal-'));
    try {
      const storage = new LocalStorage(tmpDir);
      await expect(storage.list('../../etc')).rejects.toThrow('Path traversal blocked');
    } finally {
      rmSync(tmpDir, { recursive: true });
    }
  });

  test('blocks getUrl path traversal via ../', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'gbrain-traversal-'));
    try {
      const storage = new LocalStorage(tmpDir);
      await expect(storage.getUrl('../../etc/passwd')).rejects.toThrow('Path traversal blocked');
    } finally {
      rmSync(tmpDir, { recursive: true });
    }
  });

  test('allows legitimate nested paths', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'gbrain-traversal-'));
    try {
      const storage = new LocalStorage(tmpDir);
      await storage.upload('pages/people/elon/avatar.png', Buffer.from('img'));
      const data = await storage.download('pages/people/elon/avatar.png');
      expect(data.toString()).toBe('img');
    } finally {
      rmSync(tmpDir, { recursive: true });
    }
  });
});

describe('createStorage', () => {
  test('creates LocalStorage for backend: local', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'gbrain-factory-test-'));
    try {
      const storage = await createStorage({ backend: 'local', bucket: 'test', localPath: tmpDir });
      await storage.upload('test.txt', Buffer.from('hello'));
      expect(await storage.exists('test.txt')).toBe(true);
    } finally {
      rmSync(tmpDir, { recursive: true });
    }
  });

  test('throws for unknown backend', async () => {
    expect(createStorage({ backend: 'unknown' as any, bucket: 'test' })).rejects.toThrow('Unknown storage backend');
  });

  test('S3Storage requires credentials', async () => {
    expect(createStorage({ backend: 's3', bucket: 'test' })).rejects.toThrow('accessKeyId');
  });

  test('SupabaseStorage requires projectUrl', async () => {
    expect(createStorage({ backend: 'supabase', bucket: 'test' })).rejects.toThrow('projectUrl');
  });
});

describe('bounded remote storage downloads', () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test('S3 requests only max + one probe byte and caps a streamed response', async () => {
    const storage = new S3Storage({
      backend: 's3',
      bucket: 'test',
      accessKeyId: 'test-access-key',
      secretAccessKey: 'test-secret-key',
    });
    let range: string | undefined;
    async function* body(): AsyncGenerator<Uint8Array> {
      yield Buffer.from('12');
      yield Buffer.from('34');
    }
    (storage as unknown as { client: { send(command: { input: { Range?: string } }): Promise<unknown> } }).client = {
      send: async command => {
        range = command.input.Range;
        return { Body: body() };
      },
    };

    await expect(storage.download('image.png', 3)).rejects.toBeInstanceOf(StorageReadLimitError);
    expect(range).toBe('bytes=0-3');
  });

  test('S3 rejects a non-streaming body instead of buffering past the cap', async () => {
    const storage = new S3Storage({
      backend: 's3',
      bucket: 'test',
      accessKeyId: 'test-access-key',
      secretAccessKey: 'test-secret-key',
    });
    let transformed = false;
    (storage as unknown as { client: { send(): Promise<unknown> } }).client = {
      send: async () => ({
        Body: {
          transformToByteArray: async () => {
            transformed = true;
            return new Uint8Array(1024 * 1024);
          },
        },
      }),
    };

    await expect(storage.download('image.png', 3)).rejects.toBeInstanceOf(StorageReadLimitError);
    expect(transformed).toBe(false);
  });

  test('Supabase caps a response even when the server ignores Range', async () => {
    let range = '';
    globalThis.fetch = (async (_input, init) => {
      range = new Headers(init?.headers).get('range') ?? '';
      return new Response(Buffer.from('1234'), { status: 200 });
    }) as typeof fetch;
    const storage = new SupabaseStorage({
      backend: 'supabase',
      bucket: 'test',
      projectUrl: 'https://storage.invalid',
      serviceRoleKey: 'test-service-role-key',
    });

    await expect(storage.download('image.png', 3)).rejects.toBeInstanceOf(StorageReadLimitError);
    expect(range).toBe('bytes=0-3');
  });

  test.each([
    '../other-bucket/secret.png',
    '/other-bucket/secret.png',
    'safe/../secret.png',
    'safe/%2e%2e/secret.png',
    'safe\\secret.png',
    'safe/secret.png?download=1',
    'safe/secret.png#fragment',
  ])('Supabase rejects unsafe object key %s before issuing a request', async (path) => {
    let fetched = false;
    globalThis.fetch = (async (_input, _init) => {
      fetched = true;
      return new Response(Buffer.from('should-not-read'));
    }) as typeof fetch;
    const storage = new SupabaseStorage({
      backend: 'supabase',
      bucket: 'test',
      projectUrl: 'https://storage.invalid',
      serviceRoleKey: 'test-service-role-key',
    });

    await expect(storage.download(path, 3)).rejects.toThrow('Invalid storage object key');
    expect(fetched).toBe(false);
  });

  test('Supabase safely encodes each legitimate object-key segment', async () => {
    let requestedUrl = '';
    globalThis.fetch = (async input => {
      requestedUrl = String(input);
      return new Response(Buffer.from('123'), { status: 200 });
    }) as typeof fetch;
    const storage = new SupabaseStorage({
      backend: 'supabase',
      bucket: 'test bucket',
      projectUrl: 'https://storage.invalid/',
      serviceRoleKey: 'test-service-role-key',
    });

    await expect(storage.download('images/chart one.png', 3)).resolves.toEqual(Buffer.from('123'));
    expect(requestedUrl).toBe('https://storage.invalid/storage/v1/object/test%20bucket/images/chart%20one.png');
  });
});
