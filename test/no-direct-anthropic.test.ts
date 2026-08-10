import { afterEach, describe, expect, test } from 'bun:test';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

const roots: string[] = [];

function fixtureSource(source: string): string {
  const root = mkdtempSync(join(tmpdir(), 'gbrain-no-direct-anthropic-'));
  roots.push(root);
  const path = join(root, 'candidate.ts');
  writeFileSync(path, source);
  return path;
}

function runGuard(scanFiles?: string) {
  return Bun.spawnSync(['bash', 'scripts/check-gateway-routed-no-direct-anthropic.sh'], {
    cwd: process.cwd(),
    env: {
      ...process.env,
      ...(scanFiles === undefined ? {} : { GBRAIN_DIRECT_ANTHROPIC_SCAN_FILES: scanFiles }),
    },
    stdout: 'pipe',
    stderr: 'pipe',
  });
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe('direct Anthropic transport guard', () => {
  test('accepts the candidate and the sole private provider exception', () => {
    expect(runGuard().exitCode).toBe(0);
    expect(runGuard('src/core/image-ocr-provider.ts').exitCode).toBe(0);
  });

  test('rejects a direct Anthropic endpoint/fetch in every other file', () => {
    const path = fixtureSource(`
const endpoint = 'https://api.anthropic.com/v1/messages';
await fetch(endpoint, { method: 'POST' });
`);
    const result = runGuard(path);
    expect(result.exitCode).toBe(1);
    expect(result.stdout.toString()).toContain('Only src/core/image-ocr-provider.ts');
  });

  test('rejects a runtime Anthropic SDK path in every other file', () => {
    const path = fixtureSource(`
import Anthropic from '@anthropic-ai/sdk';
const client = new Anthropic();
void client;
`);
    const result = runGuard(path);
    expect(result.exitCode).toBe(1);
    expect(result.stdout.toString()).toContain('runtime @anthropic-ai/sdk import');
  });

  test('allows type-only Anthropic SDK imports', () => {
    const path = fixtureSource(`
import type Anthropic from '@anthropic-ai/sdk';
export type Message = Anthropic.Message;
`);
    expect(runGuard(path).exitCode).toBe(0);
  });
});
