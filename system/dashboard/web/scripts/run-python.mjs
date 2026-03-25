import { spawnSync } from 'node:child_process';
import { constants as fsConstants, existsSync, accessSync } from 'node:fs';
import { constants as osConstants } from 'node:os';
import { join } from 'node:path';

function findExecutable(candidates) {
  for (const candidate of candidates) {
    if (!existsSync(candidate)) {
      continue;
    }
    try {
      accessSync(candidate, fsConstants.X_OK);
      return candidate;
    } catch {
      // try next candidate
    }
  }
  return null;
}

function probePythonExecutable(command, env = process.env) {
  const probe = spawnSync(command, ['-c', 'import sys; print(sys.executable)'], {
    env,
    encoding: 'utf8',
  });
  if (probe.status !== 0) {
    return null;
  }
  return probe.stdout.trim() || command;
}

export function resolvePythonExecutable(env = process.env, platform = process.platform) {
  const condaPrefix = env.CONDA_PREFIX?.trim();
  if (condaPrefix) {
    const condaCandidates = [
      join(condaPrefix, 'python.exe'),
      join(condaPrefix, 'Scripts', 'python.exe'),
      join(condaPrefix, 'bin', 'python3'),
      join(condaPrefix, 'bin', 'python'),
      join(condaPrefix, 'bin', 'python.exe'),
    ];
    const condaPython = findExecutable(condaCandidates);
    if (condaPython) {
      return condaPython;
    }
  }

  for (const candidate of ['python3', 'python']) {
    const resolved = probePythonExecutable(candidate, env);
    if (resolved) {
      return resolved;
    }
  }

  if (platform === 'win32') {
    return 'python';
  }

  return 'python3';
}

function signalExitCode(signal) {
  const signalNumber = osConstants.signals[signal];
  if (typeof signalNumber === 'number') {
    return 128 + signalNumber;
  }
  return 1;
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

if (child.signal) {
  process.exit(signalExitCode(child.signal));
}

process.exit(child.status ?? 1);
