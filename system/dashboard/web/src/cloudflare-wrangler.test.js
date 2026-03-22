import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const SCRIPT = join(process.cwd(), 'scripts', 'cloudflare-wrangler.mjs');

function writeEnvFile(contents) {
  const dir = mkdtempSync(join(tmpdir(), 'fenlie-cf-runner-'));
  const path = join(dir, '.env.cloudflare.local');
  writeFileSync(path, contents, 'utf8');
  return path;
}

describe('cloudflare wrangler wrapper', () => {
  it('injects CLOUDFLARE_API_TOKEN from a local env file before spawning wrangler', () => {
    const envFile = writeEnvFile('CLOUDFLARE_API_TOKEN=token-from-file\n');
    const stdout = execFileSync('node', [SCRIPT, 'whoami'], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        CI: '1',
        CLOUDFLARE_API_TOKEN: '',
        FENLIE_CLOUDFLARE_ENV_FILE: envFile,
        FENLIE_CLOUDFLARE_WRANGLER_ECHO: '1',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
      encoding: 'utf8',
    });

    const payload = JSON.parse(stdout);
    expect(payload.ok).toBe(true);
    expect(payload.tokenSource).toBe('env file');
    expect(payload.args).toEqual(['whoami']);
  });
});
