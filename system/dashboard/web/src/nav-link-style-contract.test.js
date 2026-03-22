import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('nav link style contract', () => {
  it('keeps stack/sidebar copy left-aligned inside shared buttons', () => {
    const css = readFileSync(join(process.cwd(), 'src/index.css'), 'utf8');
    const match = css.match(/\.nav-link-copy\s*\{([^}]*)\}/);

    expect(match?.[1]).toBeTruthy();
    expect(match?.[1]).toMatch(/text-align:\s*left;/);
  });

  it('keeps artifact archive buttons left-aligned instead of inheriting centered stack-button layout', () => {
    const css = readFileSync(join(process.cwd(), 'src/index.css'), 'utf8');
    const match = css.match(/\.artifact-button\s*\{([^}]*)\}/);

    expect(match?.[1]).toBeTruthy();
    expect(match?.[1]).toMatch(/justify-content:\s*flex-start;/);
    expect(match?.[1]).toMatch(/align-items:\s*stretch;/);
    expect(match?.[1]).toMatch(/text-align:\s*left;/);
  });
});
