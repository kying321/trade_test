import { PRIMARY_NAV, OPS_SURFACE_NAV, RESEARCH_LEGACY_NAV } from '../navigation/nav-config';
import { buildCpaLink, buildOpsLegacyLink, buildOverviewLink, buildResearchLegacyLink } from '../navigation/route-contract';
import { getWorkspacePageSections } from '../navigation/workspace-sections';
import type { TerminalReadModel } from '../types/contracts';
import { buildTerminalLink, buildWorkspaceLink, buildWorkspacePageLink } from '../utils/focus-links';
import { safeDisplayValue } from '../utils/formatters';
import { labelFor } from '../utils/dictionary';
import type { SearchCatalogEntry, SearchResultEntry, SearchScope } from './types';

const CATEGORY_ORDER: Record<'module' | 'route' | 'artifact', number> = {
  module: 0,
  route: 1,
  artifact: 2,
};

function normalizeText(value: unknown): string {
  return safeDisplayValue(value).trim().toLowerCase();
}

function uniqueEntries(entries: SearchCatalogEntry[]): SearchCatalogEntry[] {
  const seen = new Set<string>();
  return entries.filter((entry) => {
    const key = `${entry.category}::${entry.id}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function createRouteEntries(): SearchCatalogEntry[] {
  const primary = PRIMARY_NAV.map((item) => ({
    id: `route:${item.id}`,
    category: 'route' as const,
    title: item.label,
    subtitle: item.description,
    keywords: [item.id, item.label, item.description || '', item.to, '路由', '入口'],
    destination: item.id === 'overview'
      ? buildOverviewLink({})
      : item.id === 'ops'
        ? buildOpsLegacyLink('public', {})
        : item.id === 'cpa'
          ? buildCpaLink({})
        : buildResearchLegacyLink('artifacts', {}),
  }));
  const ops = OPS_SURFACE_NAV.map((item) => ({
    id: `route:${item.id}`,
    category: 'route' as const,
    title: item.label,
    subtitle: item.description,
    keywords: [item.id, item.label, item.description || '', item.to, '终端', 'surface'],
    destination: item.id === 'terminal-internal' ? buildTerminalLink('internal', {}) : buildTerminalLink('public', {}),
  }));
  const research = RESEARCH_LEGACY_NAV.map((item) => ({
    id: `route:${item.id}`,
    category: 'route' as const,
    title: item.label,
    subtitle: item.description,
    keywords: [item.id, item.label, item.description || '', item.to, item.section, '工作区'],
    destination: buildResearchLegacyLink(item.section, {}),
  }));
  return [...primary, ...ops, ...research];
}

function createStaticModuleEntries(): SearchCatalogEntry[] {
  return [
    {
      id: 'module:overview-mainline-summary',
      category: 'module',
      title: '研究主线摘要',
      subtitle: '总览 / source-owned / canonical',
      keywords: ['总览', '主线', 'overview', 'mainline', 'hold_selection_handoff'],
      destination: buildOverviewLink({ anchor: 'overview-mainline-summary' }),
    },
    {
      id: 'module:overview-commodity-reasoning',
      category: 'module',
      title: '国内商品推理线',
      subtitle: '总览 / 国内商品 / reasoning',
      keywords: ['总览', '国内商品', 'commodity', 'reasoning', 'BU2606'],
      destination: buildOverviewLink({ anchor: 'overview-commodity-reasoning' }),
    },
    {
      id: 'module:overview-jin10-sidecar',
      category: 'module',
      title: 'Jin10 研究侧边车',
      subtitle: '总览 / 日历 / 快讯 / quote watch',
      keywords: ['总览', 'jin10', '财经日历', '快讯', 'quote watch', 'xauusd', 'usoil'],
      destination: buildOverviewLink({ anchor: 'overview-jin10-sidecar' }),
    },
    {
      id: 'module:overview-axios-sidecar',
      category: 'module',
      title: 'Axios 研究侧边车',
      subtitle: '总览 / 新闻条目 / 关键词',
      keywords: ['总览', 'axios', '新闻', '关键词', 'local', 'national'],
      destination: buildOverviewLink({ anchor: 'overview-axios-sidecar' }),
    },
    {
      id: 'module:overview-polymarket-sidecar',
      category: 'module',
      title: 'Polymarket 情绪侧边车',
      subtitle: '总览 / prediction market / sentiment',
      keywords: ['总览', 'polymarket', 'prediction market', 'sentiment', 'yes price', 'crypto', 'economy'],
      destination: buildOverviewLink({ anchor: 'overview-polymarket-sidecar' }),
    },
    {
      id: 'module:overview-external-intelligence',
      category: 'module',
      title: '外部情报带',
      subtitle: '总览 / Jin10 + Axios / aggregated',
      keywords: ['总览', '外部情报', 'jin10', 'axios', 'polymarket', 'calendar', 'flash', 'news'],
      destination: buildOverviewLink({ anchor: 'overview-external-intelligence' }),
    },
    {
      id: 'module:overview-alignment-projection',
      category: 'module',
      title: '方向对齐投射',
      subtitle: '总览 / internal feedback',
      keywords: ['总览', 'alignment', 'feedback', '方向对齐'],
      destination: buildOverviewLink({ anchor: 'overview-alignment-projection', view: 'internal' }),
    },
    {
      id: 'module:terminal-orchestration',
      category: 'module',
      title: labelFor('terminal_panel_orchestration_title'),
      subtitle: '终端 / 总线调度与全局门禁',
      keywords: ['终端', 'orchestration', '总线调度', '门禁'],
      destination: buildTerminalLink('public', { anchor: 'terminal-orchestration', panel: 'orchestration' }),
    },
    {
      id: 'module:terminal-data-regime',
      category: 'module',
      title: labelFor('terminal_panel_data_regime_title'),
      subtitle: '终端 / 数据质量与 regime',
      keywords: ['终端', 'data regime', '数据面', 'regime'],
      destination: buildTerminalLink('public', { anchor: 'terminal-data-regime', panel: 'data-regime' }),
    },
    {
      id: 'module:terminal-signal-risk',
      category: 'module',
      title: labelFor('terminal_panel_signal_risk_title'),
      subtitle: '终端 / 信号与风险',
      keywords: ['终端', 'signal risk', '信号发生器', '风险节流阀'],
      destination: buildTerminalLink('public', { anchor: 'terminal-signal-risk', panel: 'signal-risk' }),
    },
    {
      id: 'module:terminal-commodity-reasoning',
      category: 'module',
      title: '国内商品推理线',
      subtitle: '终端 / reasoning / visible-priority',
      keywords: ['终端', '国内商品', 'commodity', 'reasoning', 'BU2606'],
      destination: buildTerminalLink('public', { anchor: 'terminal-commodity-reasoning' }),
    },
    {
      id: 'module:terminal-lab-review',
      category: 'module',
      title: labelFor('terminal_panel_lab_review_title'),
      subtitle: '终端 / 回测与实验室',
      keywords: ['终端', 'lab review', '回测', '研究头'],
      destination: buildTerminalLink('public', { anchor: 'terminal-lab-review', panel: 'lab-review' }),
    },
    {
      id: 'module:workspace-artifacts-search',
      category: 'module',
      title: labelFor('workspace_artifacts_search_panel_title'),
      subtitle: '工件池 / 检索定位',
      keywords: ['工件池', '检索定位', 'artifact search', 'search panel'],
      destination: buildWorkspaceLink('artifacts', { anchor: 'workspace-artifacts-search-panel' }),
    },
    {
      id: 'module:workspace-artifacts-focus',
      category: 'module',
      title: labelFor('workspace_artifacts_focus_title'),
      subtitle: '工件池 / 当前焦点',
      keywords: ['工件池', '当前焦点', 'focus', 'artifact focus'],
      destination: buildWorkspaceLink('artifacts', { anchor: 'workspace-artifacts-focus-panel' }),
    },
    {
      id: 'module:workspace-artifacts-target-pool',
      category: 'module',
      title: labelFor('workspace_artifacts_target_pool_title'),
      subtitle: '工件池 / 目标池',
      keywords: ['工件池', '目标池', 'artifact target pool'],
      destination: buildWorkspaceLink('artifacts', { anchor: 'workspace-artifacts-target-pool' }),
    },
  ];
}

function createWorkspaceSectionEntries(model: TerminalReadModel): SearchCatalogEntry[] {
  const sections = ['alignment', 'backtests', 'contracts', 'raw'] as const;
  return sections.flatMap((section) =>
    getWorkspacePageSections(section, model).map((item) => ({
      id: `module:${section}:${item.id}`,
      category: 'module' as const,
      title: item.label,
      subtitle: `${labelFor(section)} / ${item.groupLabel || '阶段内目录'}`,
      keywords: [section, item.id, item.label, item.groupId || '', item.groupLabel || '', 'workspace section'],
      destination: buildWorkspacePageLink(section, {}, item.id),
    })),
  );
}

function createArtifactEntries(model: TerminalReadModel): SearchCatalogEntry[] {
  const artifactEntries = model.workspace.artifactRows.map((row) => {
    const artifactId = safeDisplayValue(row.id || row.payload_key);
    const displayTitle = artifactId === 'jin10_mcp_snapshot'
      ? 'Jin10 研究侧边车'
      : artifactId === 'axios_site_snapshot'
        ? 'Axios 研究侧边车'
        : artifactId === 'polymarket_gamma_snapshot'
          ? 'Polymarket 情绪侧边车'
        : artifactId === 'external_intelligence_snapshot'
          ? '外部情报带'
          : labelFor(safeDisplayValue(row.label || row.payload_key || row.id));
    const payloadEntry = row.payload_key ? model.workspace.artifactPayloads[row.payload_key] : undefined;
    const payloadSummary = payloadEntry?.summary as Record<string, unknown> | undefined;
    const payload = payloadEntry?.payload as Record<string, unknown> | undefined;
    const nestedSummary = payload?.summary as Record<string, unknown> | undefined;
    const summaryKeywords = [
      safeDisplayValue(payloadSummary?.recommended_brief),
      safeDisplayValue(payloadSummary?.takeaway),
      safeDisplayValue(payload?.recommended_brief),
      safeDisplayValue(payload?.takeaway),
      ...(Array.isArray(nestedSummary?.high_importance_titles) ? nestedSummary.high_importance_titles.map((value) => safeDisplayValue(value)) : []),
      ...(Array.isArray(nestedSummary?.latest_flash_briefs) ? nestedSummary.latest_flash_briefs.map((value) => safeDisplayValue(value)) : []),
      ...(Array.isArray(nestedSummary?.quote_watch)
        ? nestedSummary.quote_watch.flatMap((value) => {
          const row = value as Record<string, unknown>;
          return [safeDisplayValue(row.name), safeDisplayValue(row.code)];
        })
        : []),
      ...(Array.isArray(nestedSummary?.top_categories) ? nestedSummary.top_categories.map((value) => safeDisplayValue(value)) : []),
      ...(Array.isArray(nestedSummary?.top_titles) ? nestedSummary.top_titles.map((value) => safeDisplayValue(value)) : []),
    ];
    return {
      id: `artifact:catalog:${artifactId}`,
      category: 'artifact' as const,
      title: displayTitle,
      subtitle: safeDisplayValue(row.path || row.category || row.artifact_group || '工件'),
      keywords: [
        artifactId,
        displayTitle,
        safeDisplayValue(row.payload_key),
        safeDisplayValue(row.label),
        safeDisplayValue(row.path),
        safeDisplayValue(row.category),
        safeDisplayValue(row.artifact_group),
        safeDisplayValue(row.research_decision),
        ...summaryKeywords,
      ],
      destination: buildWorkspaceLink('artifacts', { artifact: artifactId }),
    };
  });

  const sourceHeadEntries = model.workspace.sourceHeads.map((row) => ({
    id: `artifact:source-head:${row.id}`,
    category: 'artifact' as const,
    title: labelFor(safeDisplayValue(row.label || row.id)),
    subtitle: safeDisplayValue(row.summary || row.path || 'source head'),
    keywords: [
      row.id,
      safeDisplayValue(row.label),
      safeDisplayValue(row.summary),
      safeDisplayValue(row.path),
      safeDisplayValue(row.research_decision),
      'source head',
    ],
    destination: buildWorkspacePageLink('contracts', { artifact: row.id }, 'contracts-source-heads'),
  }));

  const acceptanceChecks = model.workspace.publicAcceptance.checks.map((row) => ({
    id: `artifact:acceptance-check:${row.id}`,
    category: 'artifact' as const,
    title: row.label,
    subtitle: safeDisplayValue(row.status || row.headline || '验收检查'),
    keywords: [row.id, row.label, safeDisplayValue(row.status), safeDisplayValue(row.headline), safeDisplayValue(row.report_path), 'acceptance', 'contracts'],
    destination: buildWorkspacePageLink('contracts', {}, 'contracts-acceptance-checks'),
  }));

  const subcommands = model.workspace.publicAcceptance.subcommands.map((row) => ({
    id: `artifact:acceptance-subcommand:${row.id}`,
    category: 'artifact' as const,
    title: row.label,
    subtitle: safeDisplayValue(row.cmd || row.cwd || '子命令证据'),
    keywords: [row.id, row.label, safeDisplayValue(row.cmd), safeDisplayValue(row.cwd), 'subcommand', 'contracts'],
    destination: buildWorkspacePageLink('contracts', {}, 'contracts-acceptance-subcommands'),
  }));

  return [...artifactEntries, ...sourceHeadEntries, ...acceptanceChecks, ...subcommands];
}

export function buildSearchCatalog(model?: TerminalReadModel | null): SearchCatalogEntry[] {
  const staticEntries = [...createRouteEntries(), ...createStaticModuleEntries()];
  if (!model) return uniqueEntries(staticEntries);
  return uniqueEntries([
    ...staticEntries,
    ...createWorkspaceSectionEntries(model),
    ...createArtifactEntries(model),
  ]);
}

function scoreMatches(entry: SearchCatalogEntry, query: string): SearchResultEntry | null {
  const normalizedQuery = normalizeText(query);
  if (!normalizedQuery) return null;
  const title = normalizeText(entry.title);
  const subtitle = normalizeText(entry.subtitle);
  const destination = normalizeText(entry.destination);
  const keywords = entry.keywords.map((keyword) => normalizeText(keyword)).filter(Boolean);
  const matchedFields: string[] = [];
  let score = 0;

  if (title === normalizedQuery) {
    matchedFields.push('title');
    score = Math.max(score, 120);
  } else if (title.startsWith(normalizedQuery)) {
    matchedFields.push('title');
    score = Math.max(score, 110);
  } else if (title.includes(normalizedQuery)) {
    matchedFields.push('title');
    score = Math.max(score, 100);
  }

  if (subtitle.includes(normalizedQuery)) {
    matchedFields.push('subtitle');
    score = Math.max(score, 70);
  }

  if (destination.includes(normalizedQuery)) {
    matchedFields.push('destination');
    score = Math.max(score, 50);
  }

  const keywordMatch = keywords.find((keyword) => keyword === normalizedQuery || keyword.includes(normalizedQuery));
  if (keywordMatch) {
    matchedFields.push('keyword');
    score = Math.max(score, keywordMatch === normalizedQuery ? 95 : 80);
  }

  if (!matchedFields.length) return null;
  return {
    ...entry,
    matchedFields: Array.from(new Set(matchedFields)),
    score,
  };
}

export function searchCatalog(
  entries: SearchCatalogEntry[],
  query: string,
  scope: SearchScope = 'all',
): SearchResultEntry[] {
  const normalizedQuery = normalizeText(query);
  if (!normalizedQuery) return [];
  return entries
    .filter((entry) => scope === 'all' || entry.category === scope)
    .map((entry) => scoreMatches(entry, normalizedQuery))
    .filter((entry): entry is SearchResultEntry => Boolean(entry))
    .sort((left, right) =>
      CATEGORY_ORDER[left.category] - CATEGORY_ORDER[right.category]
      || right.score - left.score
      || left.title.localeCompare(right.title, 'zh-Hans-CN'),
    );
}

export function groupSearchResults(results: SearchResultEntry[]): Record<'module' | 'route' | 'artifact', SearchResultEntry[]> {
  return {
    module: results.filter((result) => result.category === 'module'),
    route: results.filter((result) => result.category === 'route'),
    artifact: results.filter((result) => result.category === 'artifact'),
  };
}
