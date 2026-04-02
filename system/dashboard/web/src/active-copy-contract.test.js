import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const FILES = [
  'src/components/ErrorBoundary.tsx',
  'src/utils/dictionary.ts',
  'src/utils/formatters.ts',
];

describe('active copy contract', () => {
  it('keeps support-layer modules free of accidental inline Chinese copy', () => {
    const hanRegex = /[\u4e00-\u9fff]/g;

    for (const file of FILES) {
      const content = readFileSync(join(process.cwd(), file), 'utf8');
      const matches = [...content.matchAll(hanRegex)].map((match) => match[0]);
      expect(matches, file).toEqual([]);
    }
  });
});
