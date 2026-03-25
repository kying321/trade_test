import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, chmodSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const SCRIPT = join(process.cwd(), 'scripts', 'run-python.mjs');

function runLauncher(args = [], env = {}) {
  return spawnSync(process.execPath, [SCRIPT, ...args], {
    cwd: process.cwd(),
    env: { ...process.env, ...env },
    encoding: 'utf8',
  });
}

function printResolvedPython(env = {}) {
  const result = runLauncher(['--print-python'], env);
  if (result.error) {
    throw result.error;
  }
  return result.stdout.trim();
}

describe('run-python launcher', () => {
  it('prefers CONDA_PREFIX/bin/python3 when it is executable', () => {
    const condaRoot = mkdtempSync(join(tmpdir(), 'fenlie-conda-'));
    const binDir = join(condaRoot, 'bin');
    const fakePython = join(binDir, 'python3');
    mkdirSync(binDir, { recursive: true });
    writeFileSync(fakePython, '#!/bin/sh\nexit 0\n', 'utf8');
    chmodSync(fakePython, 0o755);

    const resolved = printResolvedPython({ CONDA_PREFIX: condaRoot });

    expect(resolved).toBe(fakePython);
  });

  it('prefers CONDA_PREFIX/Scripts/python.exe when it is executable', () => {
    const condaRoot = mkdtempSync(join(tmpdir(), 'fenlie-conda-win-'));
    const scriptsDir = join(condaRoot, 'Scripts');
    const fakePython = join(scriptsDir, 'python.exe');
    mkdirSync(scriptsDir, { recursive: true });
    writeFileSync(fakePython, '#!/bin/sh\nexit 0\n', 'utf8');
    chmodSync(fakePython, 0o755);

    const resolved = printResolvedPython({ CONDA_PREFIX: condaRoot });

    expect(resolved).toBe(fakePython);
  });

  it('falls back to the PATH python executable when only python is available', () => {
    const binDir = mkdtempSync(join(tmpdir(), 'fenlie-path-python-'));
    const fakePython = join(binDir, 'python');
    writeFileSync(fakePython, '#!/bin/sh\nprintf \"%s\\n\" \"$0\"\n', 'utf8');
    chmodSync(fakePython, 0o755);

    const resolved = printResolvedPython({
      CONDA_PREFIX: join(tmpdir(), 'fenlie-missing-conda'),
      PATH: binDir,
    });

    expect(resolved).toBe(fakePython);
  });

  it('maps child signal termination into a conventional non-zero exit code', () => {
    const script = join(mkdtempSync(join(tmpdir(), 'fenlie-signal-')), 'sigterm.py');
    writeFileSync(script, 'import os, signal\nos.kill(os.getpid(), signal.SIGTERM)\n', 'utf8');

    const result = runLauncher([script]);

    expect(result.status).toBe(143);
  });
});
