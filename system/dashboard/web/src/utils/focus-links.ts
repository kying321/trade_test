import type { SurfaceKey } from '../types/contracts';
import { preserveSharedQuery } from '../navigation/route-contract';
import { safeDisplayValue } from './formatters';

export type TerminalFocusState = {
  panel?: string;
  section?: string;
  row?: string;
};

export type WorkspaceSection = 'artifacts' | 'contracts' | 'raw' | 'backtests';

export type WorkspaceFocusState = {
  artifact?: string;
  domain?: string;
  tone?: string;
  search?: string;
  group?: string;
  view?: string;
  symbol?: string;
  window?: string;
};

export type SharedFocusState = TerminalFocusState & WorkspaceFocusState;

export function buildTerminalFocusKey(focus: TerminalFocusState = {}): string {
  return [focus.panel || '', focus.section || '', focus.row || ''].join('|');
}

function buildQuery(params: Record<string, string | undefined>): string {
  return preserveSharedQuery(params);
}

export function buildTerminalLink(surface: SurfaceKey, focus: SharedFocusState = {}): string {
  const qs = buildQuery({
    artifact: focus.artifact,
    domain: focus.domain,
    tone: focus.tone,
    search: focus.search,
    panel: focus.panel,
    section: focus.section,
    row: focus.row,
    group: focus.group,
    view: focus.view,
    symbol: focus.symbol,
    window: focus.window,
  });
  return `/terminal/${surface}${qs ? `?${qs}` : ''}`;
}

export function buildWorkspaceLink(section: WorkspaceSection, focus: SharedFocusState = {}): string {
  const qs = buildQuery({
    artifact: focus.artifact,
    domain: focus.domain,
    tone: focus.tone,
    search: focus.search,
    panel: focus.panel,
    section: focus.section,
    row: focus.row,
    group: focus.group,
    view: focus.view,
    symbol: focus.symbol,
    window: focus.window,
  });
  return `/workspace/${section}${qs ? `?${qs}` : ''}`;
}

const DEFAULT_ROW_IDENTITY_KEYS = ['id', 'slot', 'stage', 'label', 'symbol', 'action', 'target'];

export function buildDrilldownRowId(row: Record<string, unknown>, rowIdentityKeys?: string[]): string {
  const collect = (keys: string[]) => keys
    .map((key) => safeDisplayValue(row[key]).trim())
    .filter((value) => value && value !== '—');

  const preferredValues = rowIdentityKeys?.length ? collect(rowIdentityKeys) : [];
  if (preferredValues.length) return preferredValues.join('::');
  return collect(DEFAULT_ROW_IDENTITY_KEYS).join('::');
}
