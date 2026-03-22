import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const FILES = [
  'src/components/OverviewPanel.jsx',
  'src/components/HeroSidebar.jsx',
  'src/components/BacktestWorkbench.jsx',
  'src/components/charts.jsx',
  'src/components/ui.jsx',
  'src/components/InterfacePanel.jsx',
  'src/components/ArtifactDetail.jsx',
  'src/components/FieldExplorer.jsx',
  'src/components/explorer/ArtifactInspectorPane.jsx',
  'src/components/explorer/ArtifactListPane.jsx',
  'src/components/explorer/BreadcrumbBar.jsx',
  'src/components/explorer/ExplorerBreadcrumbs.jsx',
  'src/components/explorer/DomainRail.jsx',
  'src/components/explorer/ExplorerDomainPane.jsx',
  'src/lib/display-model.js',
];

describe('legacy copy contract', () => {
  it('keeps legacy user-facing copy in the shared term contract', () => {
    const hanRegex = /[\u4e00-\u9fff]/g;

    for (const file of FILES) {
      const content = readFileSync(join(process.cwd(), file), 'utf8');
      const matches = [...content.matchAll(hanRegex)].map((match) => match[0]);
      expect(matches, file).toEqual([]);
    }
  });
});
