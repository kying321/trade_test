import { spawnSync } from 'node:child_process';
import { constants, existsSync, accessSync } from 'node:fs';
import { join } from 'node:path';

export function resolvePythonExecutable(env = process.env) {
  const condaPrefix = env.CONDA_PREFIX?.trim();
  if (condaPrefix) {
    const candidate = join(condaPrefix, 'bin', 'python3');
    if (existsSync(candidate)) {
      try {
        accessSync(candidate, constants.X_OK);
        return candidate;
      } catch {
        // fall through to PATH python
      }
    }
  }

  const probe = spawnSync('python3', ['-c', 'import sys; print(sys.executable)'], {
    env,
    encoding: 'utf8',
  });
  if (probe.status === 0) {
    const resolved = probe.stdout.trim();
    if (resolved) {
      return resolved;
    }
  }
  return 'python3';
}

const args = process.argv.slice(2);
const printOnly = args[0] === '--print-python';
if (printOnly) {
  args.shift();
}

const python = resolvePythonExecutable(process.env);
if (printOnly) {
  process.stdout.write(`${python}\n`);
  process.exit(0);
}

const child = spawnSync(python, args, {
  env: process.env,
  stdio: 'inherit',
});

if (child.error) {
  throw child.error;
}

process.exit(child.status ?? 1);
