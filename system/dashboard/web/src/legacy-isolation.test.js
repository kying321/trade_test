import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { LEGACY_FILES, collectReachableSourceFiles, findLegacyIntrusions } from './legacy-isolation.js';

describe('legacy isolation', () => {
  it('keeps the active runtime import graph free of legacy sidecar files', () => {
    const reachable = collectReachableSourceFiles(process.cwd(), ['src/main.tsx']);

    expect(reachable).toContain('src/main.tsx');
    expect(reachable).toContain('src/App.tsx');
    expect(findLegacyIntrusions(reachable)).toEqual([]);
  });

  it('removes the legacy hook/lib data chain files from the repo', () => {
    const legacyDataChain = LEGACY_FILES.filter((file) => (
      file.startsWith('src/hooks/')
      || file.startsWith('src/lib/')
    ));

    for (const relPath of legacyDataChain) {
      expect(existsSync(join(process.cwd(), relPath)), relPath).toBe(false);
    }
  });

  it('removes the legacy component subtree from the repo', () => {
    const legacyComponents = [
      ...LEGACY_FILES.filter((file) => file.startsWith('src/components/')),
      'src/components/explorer/ArtifactListPane.jsx',
      'src/components/explorer/ArtifactInspectorPane.jsx',
      'src/components/explorer/ExplorerBreadcrumbs.jsx',
      'src/components/explorer/ExplorerHero.jsx',
      'src/components/explorer/BreadcrumbBar.jsx',
      'src/components/explorer/DomainRail.jsx',
      'src/components/explorer/ExplorerDomainPane.jsx',
    ];

    for (const relPath of legacyComponents) {
      expect(existsSync(join(process.cwd(), relPath)), relPath).toBe(false);
    }
  });
});
