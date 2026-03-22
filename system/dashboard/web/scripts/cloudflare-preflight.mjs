import { evaluatePreflight, log } from './cloudflare-auth.mjs';

const check = evaluatePreflight(process.env, process.cwd());
if (!check.ok) {
  log(check.message, 'error');
  process.exit(1);
}

log(check.message);
