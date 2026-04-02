import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const SCRIPT = join(process.cwd(), 'scripts', 'cloudflare-preflight.mjs');

function missingEnvFilePath() {
  const dir = mkdtempSync(join(tmpdir(), 'fenlie-cf-env-missing-'));
  return join(dir, '.env.cloudflare.local');
}

function runPreflight(env = {}) {
  try {
    const stdout = execFileSync(process.execPath, [SCRIPT], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        CLOUDFLARE_API_TOKEN: '',
        FENLIE_CLOUDFLARE_ENV_FILE: missingEnvFilePath(),
        ...env,
      },
      stdio: ['ignore', 'pipe', 'pipe'],
      encoding: 'utf8',
    });
    return { ok: true, stdout };
  } catch (error) {
    return {
      ok: false,
      status: error.status,
      stdout: error.stdout?.toString() || '',
      stderr: error.stderr?.toString() || '',
    };
  }
}

function writeWranglerConfig(contents) {
  const dir = mkdtempSync(join(tmpdir(), 'fenlie-wrangler-'));
  const path = join(dir, 'default.toml');
  writeFileSync(path, contents, 'utf8');
  return path;
}

function writeEnvFile(contents) {
  const dir = mkdtempSync(join(tmpdir(), 'fenlie-cf-env-'));
  const path = join(dir, '.env.cloudflare.local');
  writeFileSync(path, contents, 'utf8');
  return path;
}

function missingWranglerConfigPath() {
  const dir = mkdtempSync(join(tmpdir(), 'fenlie-wrangler-missing-'));
  return join(dir, 'default.toml');
}

describe('cloudflare preflight', () => {
  it('passes immediately when CLOUDFLARE_API_TOKEN is present', () => {
    const result = runPreflight({
      CLOUDFLARE_API_TOKEN: 'token-test-value',
      CI: '1',
      FENLIE_WRANGLER_CONFIG_PATH: missingWranglerConfigPath(),
    });

    expect(result.ok).toBe(true);
    expect(result.stdout).toContain('CLOUDFLARE_API_TOKEN');
    expect(result.stdout).toContain('preflight ok');
  });

  it('fails fast in non-interactive mode when CLOUDFLARE_API_TOKEN is missing', () => {
    const result = runPreflight({
      CI: '1',
      CLOUDFLARE_API_TOKEN: '',
      FENLIE_WRANGLER_CONFIG_PATH: missingWranglerConfigPath(),
    });

    expect(result.ok).toBe(false);
    expect(result.status).toBe(1);
    expect(`${result.stdout}\n${result.stderr}`).toContain('CLOUDFLARE_API_TOKEN');
    expect(`${result.stdout}\n${result.stderr}`).toContain('non-interactive');
  });

  it('passes in non-interactive mode when an unexpired Wrangler OAuth cache exists', () => {
    const configPath = writeWranglerConfig([
      'oauth_token = "cached-oauth-token"',
      'refresh_token = "cached-refresh-token"',
      'expiration_time = "2099-01-01T00:00:00.000Z"',
    ].join('\n'));

    const result = runPreflight({
      CI: '1',
      CLOUDFLARE_API_TOKEN: '',
      FENLIE_WRANGLER_CONFIG_PATH: configPath,
    });

    expect(result.ok).toBe(true);
    expect(result.stdout).toContain('Wrangler OAuth cache');
    expect(result.stdout).toContain('not expired');
  });

  it('fails with an explicit expired-oauth message when cached Wrangler OAuth is stale', () => {
    const configPath = writeWranglerConfig([
      'oauth_token = "cached-oauth-token"',
      'refresh_token = "cached-refresh-token"',
      'expiration_time = "2000-01-01T00:00:00.000Z"',
    ].join('\n'));

    const result = runPreflight({
      CI: '1',
      CLOUDFLARE_API_TOKEN: '',
      FENLIE_WRANGLER_CONFIG_PATH: configPath,
    });

    expect(result.ok).toBe(false);
    expect(result.status).toBe(1);
    expect(`${result.stdout}\n${result.stderr}`).toContain('expired Wrangler OAuth cache');
    expect(`${result.stdout}\n${result.stderr}`).toContain('CLOUDFLARE_API_TOKEN');
  });

  it('passes when CLOUDFLARE_API_TOKEN is provided by a local env file', () => {
    const envFile = writeEnvFile([
      '# local only',
      'CLOUDFLARE_API_TOKEN=token-from-file',
    ].join('\n'));

    const result = runPreflight({
      CI: '1',
      CLOUDFLARE_API_TOKEN: '',
      FENLIE_CLOUDFLARE_ENV_FILE: envFile,
      FENLIE_WRANGLER_CONFIG_PATH: missingWranglerConfigPath(),
    });

    expect(result.ok).toBe(true);
    expect(result.stdout).toContain('CLOUDFLARE_API_TOKEN');
    expect(result.stdout).toContain('env file');
  });
});
