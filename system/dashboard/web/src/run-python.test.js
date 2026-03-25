import { execFile } from 'node:child_process';
import { mkdtempSync, writeFileSync, chmodSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import { describe, expect, it } from 'vitest';

const SCRIPT = join(process.cwd(), 'scripts', 'run-python.mjs');
const execFileAsync = promisify(execFile);

async function printResolvedPython(env = {}) {
  const { stdout } = await execFileAsync('node', [SCRIPT, '--print-python'], {
    cwd: process.cwd(),
    env: { ...process.env, ...env },
    encoding: 'utf8',
  });
  return stdout.trim();
}

describe('run-python launcher', () => {
  it('prefers CONDA_PREFIX/bin/python3 when it is executable', async () => {
    const condaRoot = mkdtempSync(join(tmpdir(), 'fenlie-conda-'));
    const binDir = join(condaRoot, 'bin');
    const fakePython = join(binDir, 'python3');
    mkdirSync(binDir, { recursive: true });
    writeFileSync(fakePython, '#!/bin/sh\nexit 0\n', 'utf8');
    chmodSync(fakePython, 0o755);

    const resolved = await printResolvedPython({ CONDA_PREFIX: condaRoot });

    expect(resolved).toBe(fakePython);
  });

  it('falls back to the PATH python executable when CONDA_PREFIX is unusable', async () => {
    const expected = (
      await execFileAsync('python3', ['-c', 'import sys; print(sys.executable)'], {
        cwd: process.cwd(),
        env: process.env,
        encoding: 'utf8',
      })
    ).stdout.trim();

    const resolved = await printResolvedPython({ CONDA_PREFIX: join(tmpdir(), 'fenlie-missing-conda') });

    expect(resolved).toBe(expected);
  });
});
