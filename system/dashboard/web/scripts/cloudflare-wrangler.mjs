import { spawnSync } from 'node:child_process';
import { evaluatePreflight, log } from './cloudflare-auth.mjs';

const check = evaluatePreflight(process.env, process.cwd());
if (!check.ok) {
  log(check.message, 'error');
  process.exit(1);
}

const env = { ...process.env };
if (!env.CLOUDFLARE_API_TOKEN && check.auth.token) {
  env.CLOUDFLARE_API_TOKEN = check.auth.token;
}

if (String(process.env.FENLIE_CLOUDFLARE_WRANGLER_ECHO || '').trim()) {
  const source = check.auth.tokenSource || (check.auth.oauthCache && !check.auth.oauthCache.expired ? 'oauth-cache' : 'none');
  console.log(JSON.stringify({
    ok: true,
    tokenSource: source,
    args: process.argv.slice(2),
  }));
  process.exit(0);
}

const result = spawnSync('npx', ['wrangler', ...process.argv.slice(2)], {
  stdio: 'inherit',
  env,
  shell: process.platform === 'win32',
});

if (typeof result.status === 'number') {
  process.exit(result.status);
}

process.exit(result.error ? 1 : 0);
