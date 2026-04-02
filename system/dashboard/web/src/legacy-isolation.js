import { existsSync, readFileSync } from 'node:fs';
import { dirname, posix } from 'node:path';

const SOURCE_EXTENSIONS = ['.ts', '.tsx', '.js', '.jsx'];

export const LEGACY_FILES = [
  'src/main.legacy.jsx',
  'src/App.legacy.jsx',
  'src/hooks/useSnapshot.js',
  'src/hooks/useBacktestSelection.js',
  'src/hooks/useExplorerState.js',
  'src/hooks/usePageNavigation.js',
  'src/lib/display-model.js',
  'src/lib/snapshot-normalizers.js',
  'src/lib/navigation.js',
  'src/lib/dashboard-data.js',
  'src/lib/snapshot-selectors.js',
  'src/lib/formatters.js',
  'src/lib/i18n.js',
  'src/components/ErrorBoundary.legacy.jsx',
  'src/components/HeroSidebar.jsx',
  'src/components/OverviewPanel.jsx',
  'src/components/BacktestWorkbench.jsx',
  'src/components/ExplorerPanel.jsx',
  'src/components/InterfacePanel.jsx',
  'src/components/PageHeader.jsx',
  'src/components/ui.jsx',
  'src/components/charts.jsx',
  'src/components/ArtifactDetail.jsx',
  'src/components/FieldExplorer.jsx',
];

const LEGACY_PREFIXES = [
  'src/components/explorer/',
];

function toPosixPath(value) {
  return String(value || '').replace(/\\/g, '/');
}

function readImports(absPath) {
  const content = readFileSync(absPath, 'utf8');
  const matches = new Set();
  const patterns = [
    /import\s+(?:[^'"]+?\s+from\s+)?["']([^"']+)["']/g,
    /import\(\s*["']([^"']+)["']\s*\)/g,
  ];
  for (const pattern of patterns) {
    for (const match of content.matchAll(pattern)) {
      const specifier = String(match[1] || '').trim();
      if (specifier.startsWith('.')) matches.add(specifier);
    }
  }
  return [...matches];
}

function resolveImport(rootDir, fromRelPath, specifier) {
  const fromDir = dirname(fromRelPath);
  const basePath = posix.normalize(posix.join(fromDir, specifier));
  const candidates = [
    basePath,
    ...SOURCE_EXTENSIONS.map((ext) => `${basePath}${ext}`),
    ...SOURCE_EXTENSIONS.map((ext) => `${basePath}/index${ext}`),
  ];
  for (const relPath of candidates) {
    const absPath = posix.join(rootDir, relPath);
    if (existsSync(absPath)) return toPosixPath(relPath);
  }
  return null;
}

export function collectReachableSourceFiles(rootDir, entryFiles) {
  const normalizedRoot = toPosixPath(rootDir);
  const queue = [...entryFiles.map((entry) => toPosixPath(entry))];
  const seen = new Set();

  while (queue.length) {
    const relPath = queue.shift();
    if (!relPath || seen.has(relPath)) continue;
    const absPath = posix.join(normalizedRoot, relPath);
    if (!existsSync(absPath)) continue;
    seen.add(relPath);

    for (const specifier of readImports(absPath)) {
      const resolved = resolveImport(normalizedRoot, relPath, specifier);
      if (resolved && !seen.has(resolved)) queue.push(resolved);
    }
  }

  return [...seen].sort();
}

export function findLegacyIntrusions(reachableFiles) {
  return reachableFiles.filter((file) => (
    LEGACY_FILES.includes(file)
    || LEGACY_PREFIXES.some((prefix) => file.startsWith(prefix))
  ));
}
