import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('legacy entry wiring', () => {
  it('points legacy bootstrap and app shell to explicit legacy modules', () => {
    const mainLegacy = readFileSync(join(process.cwd(), 'src/main.legacy.jsx'), 'utf8');
    const appLegacy = readFileSync(join(process.cwd(), 'src/App.legacy.jsx'), 'utf8');

    expect(mainLegacy).toContain('import App from "./App.legacy"');
    expect(appLegacy).toContain("from './components/ErrorBoundary.legacy'");
  });
});
