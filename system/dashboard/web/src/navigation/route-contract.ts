import { safeDisplayValue } from '../utils/formatters';

export const SHARED_QUERY_KEYS = [
  'artifact',
  'domain',
  'tone',
  'anchor',
  'search',
  'search_scope',
  'panel',
  'section',
  'row',
  'group',
  'view',
  'symbol',
  'window',
] as const;

export type SharedQueryKey = (typeof SHARED_QUERY_KEYS)[number];
export type SharedQueryState = Partial<Record<SharedQueryKey, string | undefined>>;

function buildQuery(focus: SharedQueryState = {}): string {
  const query = new URLSearchParams();
  SHARED_QUERY_KEYS.forEach((key) => {
    const raw = safeDisplayValue(focus[key]).trim();
    if (raw && raw !== '—') query.set(key, raw);
  });
  return query.toString();
}

export function preserveSharedQuery(focus: SharedQueryState = {}): string {
  return buildQuery(focus);
}

function buildPreservedLink(basePath: string, focus: SharedQueryState = {}): string {
  const query = preserveSharedQuery(focus);
  return `${basePath}${query ? `?${query}` : ''}`;
}

export type OpsPageId = 'overview' | 'risk' | 'audits' | 'runbooks' | 'workflow' | 'traces';

export function buildOpsPageLink(page: OpsPageId, focus: SharedQueryState = {}): string {
  return buildPreservedLink(`/ops/${page}`, focus);
}

export function buildOpsLegacyLink(surface: 'public' | 'internal', focus: SharedQueryState = {}): string {
  return surface === 'public'
    ? buildOpsPageLink('risk', focus)
    : buildPreservedLink('/terminal/internal', focus);
}

export function buildResearchLegacyLink(
  section: 'artifacts' | 'alignment' | 'backtests' | 'contracts' | 'raw',
  focus: SharedQueryState = {},
): string {
  return buildPreservedLink(`/workspace/${section}`, focus);
}

export function buildOverviewLink(focus: SharedQueryState = {}): string {
  return buildOpsPageLink('overview', focus);
}

export function buildGraphHomeLink(focus: SharedQueryState = {}): string {
  return buildPreservedLink('/graph-home', focus);
}

export function buildCpaLink(focus: SharedQueryState = {}): string {
  return buildPreservedLink('/cpa', focus);
}
