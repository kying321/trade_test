import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('legacy entry wiring', () => {
  it('keeps production entry pinned to main.tsx -> App.tsx and removes the legacy entry pair', () => {
    const indexHtml = readFileSync(join(process.cwd(), 'index.html'), 'utf8');
    const mainActive = readFileSync(join(process.cwd(), 'src/main.tsx'), 'utf8');

    expect(indexHtml).toContain('/src/main.tsx');
    expect(mainActive).toContain("import App from './App'");
    expect(existsSync(join(process.cwd(), 'src/main.legacy.jsx'))).toBe(false);
    expect(existsSync(join(process.cwd(), 'src/App.legacy.jsx'))).toBe(false);
  });
});
