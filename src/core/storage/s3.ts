import {
  S3Client,
  PutObjectCommand,
  GetObjectCommand,
  DeleteObjectCommand,
  HeadObjectCommand,
  ListObjectsV2Command,
} from '@aws-sdk/client-s3';
import {
  collectBytesBounded,
  readResponseBodyBounded,
  StorageReadLimitError,
  type StorageBackend,
  type StorageConfig,
} from '../storage.ts';

type DownloadBody = AsyncIterable<Uint8Array> & {
  transformToByteArray(): Promise<Uint8Array>;
  transformToWebStream?: () => ReadableStream<Uint8Array>;
  destroy?: () => void;
  cancel?: () => Promise<void>;
};

async function cancelDownloadBody(body: DownloadBody): Promise<void> {
  try {
    if (typeof body.destroy === 'function') body.destroy();
    else if (typeof body.cancel === 'function') await body.cancel();
  } catch {
    // Best-effort resource cleanup while preserving the typed limit error.
  }
}

/**
 * S3-compatible storage — works with AWS S3, Cloudflare R2, MinIO, etc.
 * Uses @aws-sdk/client-s3 for proper authentication and request signing.
 */
export class S3Storage implements StorageBackend {
  private client: S3Client;
  private bucket: string;

  constructor(config: StorageConfig) {
    this.bucket = config.bucket;
    const region = config.region || 'us-east-1';

    if (!config.accessKeyId || !config.secretAccessKey) {
      throw new Error('S3 storage requires accessKeyId and secretAccessKey in config');
    }

    this.client = new S3Client({
      region,
      ...(config.endpoint ? {
        endpoint: config.endpoint,
        forcePathStyle: true, // Required for R2, MinIO, and custom endpoints
      } : {}),
      credentials: {
        accessKeyId: config.accessKeyId,
        secretAccessKey: config.secretAccessKey,
      },
    });
  }

  async upload(path: string, data: Buffer, mime?: string): Promise<void> {
    await this.client.send(new PutObjectCommand({
      Bucket: this.bucket,
      Key: path,
      Body: data,
      ContentType: mime || 'application/octet-stream',
    }));
  }

  async download(path: string, maxBytes?: number): Promise<Buffer> {
    const res = await this.client.send(new GetObjectCommand({
      Bucket: this.bucket,
      Key: path,
      ...(maxBytes === undefined ? {} : { Range: `bytes=0-${maxBytes}` }),
    }));
    if (!res.Body) throw new Error(`S3 download returned empty body: ${path}`);
    const body = res.Body as unknown as DownloadBody;
    if (maxBytes !== undefined) {
      const totalMatch = res.ContentRange?.match(/\/(\d+)$/);
      const totalSize = totalMatch ? Number(totalMatch[1]) : undefined;
      if ((totalSize !== undefined && totalSize > maxBytes)
        || (res.ContentLength !== undefined && res.ContentLength > maxBytes)) {
        await cancelDownloadBody(body);
        throw new StorageReadLimitError(maxBytes);
      }
    }

    if (typeof body[Symbol.asyncIterator] === 'function') {
      return collectBytesBounded(body, maxBytes);
    }
    if (maxBytes !== undefined) {
      if (typeof body.transformToWebStream === 'function') {
        return readResponseBodyBounded(new Response(body.transformToWebStream()), maxBytes);
      }
      if (typeof Blob !== 'undefined' && body instanceof Blob) {
        if (body.size > maxBytes) throw new StorageReadLimitError(maxBytes);
        const data = Buffer.from(await body.arrayBuffer());
        if (data.length > maxBytes) throw new StorageReadLimitError(maxBytes);
        return data;
      }
      await cancelDownloadBody(body);
      throw new StorageReadLimitError(maxBytes);
    }
    const data = Buffer.from(await body.transformToByteArray());
    return data;
  }

  async delete(path: string): Promise<void> {
    await this.client.send(new DeleteObjectCommand({
      Bucket: this.bucket,
      Key: path,
    }));
  }

  async exists(path: string): Promise<boolean> {
    try {
      await this.client.send(new HeadObjectCommand({
        Bucket: this.bucket,
        Key: path,
      }));
      return true;
    } catch (e: any) {
      if (e.name === 'NotFound' || e.$metadata?.httpStatusCode === 404) return false;
      throw e;
    }
  }

  async list(prefix: string): Promise<string[]> {
    const res = await this.client.send(new ListObjectsV2Command({
      Bucket: this.bucket,
      Prefix: prefix,
    }));
    return (res.Contents || []).map(obj => obj.Key!).filter(Boolean);
  }

  async getUrl(path: string): Promise<string> {
    // For custom endpoints (R2, MinIO), use the endpoint URL
    const endpoint = (this.client.config as any).endpoint;
    if (endpoint) {
      const base = typeof endpoint === 'function' ? (await endpoint()).url.toString() : endpoint;
      return `${base}/${this.bucket}/${path}`;
    }
    const region = await this.client.config.region();
    return `https://${this.bucket}.s3.${region}.amazonaws.com/${path}`;
  }

  async getContentHash(path: string): Promise<string | null> {
    try {
      const res = await this.client.send(new HeadObjectCommand({
        Bucket: this.bucket,
        Key: path,
      }));
      // ETag is typically the MD5 hash (quoted), but for multipart uploads it's different
      return res.ETag?.replace(/"/g, '') || null;
    } catch {
      return null;
    }
  }
}
