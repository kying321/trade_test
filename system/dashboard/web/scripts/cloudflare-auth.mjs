import { existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

export function log(message, method = 'log') {
  console[method](`[fenlie-cloudflare] ${message}`);
}

export function isNonInteractive(env) {
  if (String(env.CI || '').trim()) return true;
  return !(process.stdin.isTTY && process.stdout.isTTY);
}

function parseTomlScalar(text, key) {
  const pattern = new RegExp(`^\\s*${key}\\s*=\\s*"([^"]*)"\\s*$`, 'm');
  const match = text.match(pattern);
  return match ? match[1] : '';
}

export function candidateWranglerConfigPaths(env) {
  const explicit = String(env.FENLIE_WRANGLER_CONFIG_PATH || '').trim();
  if (explicit) return [explicit];
  const home = homedir();
  return [
    join(home, 'Library', 'Preferences', '.wrangler', 'config', 'default.toml'),
    join(home, '.wrangler', 'config', 'default.toml'),
    join(home, '.wrangler', 'default.toml'),
    join(home, '.config', '.wrangler', 'default.toml'),
  ];
}

export function readWranglerOAuthCache(env) {
  for (const path of candidateWranglerConfigPaths(env)) {
    if (!path || !existsSync(path)) continue;
    const text = readFileSync(path, 'utf8');
    const oauthToken = parseTomlScalar(text, 'oauth_token');
    const refreshToken = parseTomlScalar(text, 'refresh_token');
    const expirationTime = parseTomlScalar(text, 'expiration_time');
    if (!oauthToken && !refreshToken && !expirationTime) continue;
    let expired = true;
    if (expirationTime) {
      const expiration = Date.parse(expirationTime);
      if (Number.isFinite(expiration)) expired = Date.now() >= expiration;
    }
    return {
      path,
      oauthTokenPresent: Boolean(oauthToken),
      refreshTokenPresent: Boolean(refreshToken),
      expirationTime,
      expired,
    };
  }
  return null;
}

function candidateEnvFiles(env, cwd = process.cwd()) {
  const explicit = String(env.FENLIE_CLOUDFLARE_ENV_FILE || '').trim();
  if (explicit) return [explicit];
  return [join(cwd, '.env.cloudflare.local')];
}

export function readCloudflareTokenFromEnvFile(env, cwd = process.cwd()) {
  for (const path of candidateEnvFiles(env, cwd)) {
    if (!path || !existsSync(path)) continue;
    const text = readFileSync(path, 'utf8');
    for (const rawLine of text.split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith('#')) continue;
      const [key, ...rest] = line.split('=');
      if (key?.trim() !== 'CLOUDFLARE_API_TOKEN') continue;
      const value = rest.join('=').trim().replace(/^['"]|['"]$/g, '');
      if (value) {
        return {
          value,
          path,
        };
      }
    }
  }
  return null;
}

export function resolveCloudflareAuth(env, cwd = process.cwd()) {
  const directToken = String(env.CLOUDFLARE_API_TOKEN || '').trim();
  const envFileToken = directToken ? null : readCloudflareTokenFromEnvFile(env, cwd);
  const oauthCache = readWranglerOAuthCache(env);
  const nonInteractive = isNonInteractive(env);
  return {
    nonInteractive,
    oauthCache,
    token: directToken || envFileToken?.value || '',
    tokenSource: directToken ? 'env' : envFileToken ? 'env file' : '',
    envFileToken,
  };
}

export function evaluatePreflight(env, cwd = process.cwd()) {
  const auth = resolveCloudflareAuth(env, cwd);
  if (auth.token) {
    return {
      ok: true,
      auth,
      message: `Cloudflare preflight ok: CLOUDFLARE_API_TOKEN detected via ${auth.tokenSource}.`,
    };
  }
  if (auth.nonInteractive) {
    if (auth.oauthCache && !auth.oauthCache.expired) {
      return {
        ok: true,
        auth,
        message: `Cloudflare preflight ok: Wrangler OAuth cache is not expired (${auth.oauthCache.path}).`,
      };
    }
    const oauthHint = auth.oauthCache?.expired
      ? ` Detected expired Wrangler OAuth cache at ${auth.oauthCache.path}.`
      : '';
    return {
      ok: false,
      auth,
      message: `Cloudflare preflight failed: non-interactive environment without CLOUDFLARE_API_TOKEN.${oauthHint} Export CLOUDFLARE_API_TOKEN and retry.`,
    };
  }
  return {
    ok: true,
    auth,
    message: 'Cloudflare preflight ok: interactive session detected, OAuth whoami can run if needed.',
  };
}
