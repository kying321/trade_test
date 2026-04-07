import type { ReactNode } from 'react';
import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { findWorkspacePageSection, getDefaultWorkspacePageSection, getWorkspacePageSections } from '../navigation/workspace-sections';
import { buildSearchLink } from '../search/links';
import type { OperatorAlertLink, PublicAcceptanceResearchAuditCase, TerminalReadModel, ViewFieldSchema } from '../types/contracts';
import { labelFor } from '../utils/dictionary';
import { buildTerminalLink, buildWorkspaceLink, buildWorkspacePageLink, type SharedFocusState, type WorkspaceSection } from '../utils/focus-links';
import { compactPath, safeDisplayValue, statusTone } from '../utils/formatters';
import { ActionButton, ActionLink, EntityRowButton } from './control-primitives';
import { AccordionCard, Badge, DenseTable, DrilldownSection, JsonBlock, KeyValueGrid, PanelCard, PathText, ValueText, toneClass } from './ui-kit';
import { useShellBreakpoint } from '../hooks/use-shell-breakpoint';

const t = (key: string) => labelFor(key);
const c = (key: string, labelKey?: string) => ({ key, label: t(labelKey || key) });

type WorkspaceFocus = SharedFocusState;
type ArtifactRowState = {
  id?: string;
  payload_key?: string | null;
  artifact_layer?: string;
  artifact_group?: string;
  category?: string;
  research_decision?: string;
  status?: string;
  path?: string;
  label?: string;
};

type ArtifactGroupId =
  | 'research_mainline'
  | 'system_anchor'
  | 'research_exit_risk'
  | 'research_hold_transfer'
  | 'research_cross_section'
  | 'archive';

type ArtifactSearchScope = 'title' | 'artifact' | 'path';
type TopologyTier = 'primary' | 'fallback' | 'probe';
const TOPOLOGY_TIER_ORDER: TopologyTier[] = ['primary', 'fallback', 'probe'];

const ARTIFACT_GROUP_ORDER: ArtifactGroupId[] = [
  'research_mainline',
  'system_anchor',
  'research_exit_risk',
  'research_hold_transfer',
  'research_cross_section',
  'archive',
];

const ARTIFACT_STATUS_ORDER = ['positive', 'warning', 'negative', 'neutral'] as const;

const ARTIFACT_GROUP_LABEL_KEYS: Record<ArtifactGroupId, string> = {
  research_mainline: 'workspace_artifacts_map_mainline_label',
  system_anchor: 'workspace_artifacts_map_anchor_label',
  research_exit_risk: 'workspace_artifacts_map_exit_label',
  research_hold_transfer: 'workspace_artifacts_map_hold_label',
  research_cross_section: 'workspace_artifacts_map_cross_label',
  archive: 'workspace_artifacts_map_archive_label',
};

function classifyTopologyTier(role: string | undefined): TopologyTier {
  const raw = safeDisplayValue(role || '').toLowerCase();
  if (raw.includes('探针')) return 'probe';
  if (raw.includes('回退')) return 'fallback';
  return 'primary';
}

function topologyTierLabel(tier: TopologyTier): string {
  if (tier === 'probe') return t('workspace_contracts_topology_tier_probe');
  if (tier === 'fallback') return t('workspace_contracts_topology_tier_fallback');
  return t('workspace_contracts_topology_tier_primary');
}

function topologySectionLabel(tier: TopologyTier): string {
  if (tier === 'probe') return t('workspace_contracts_topology_section_probe');
  if (tier === 'fallback') return t('workspace_contracts_topology_section_fallback');
  return t('workspace_contracts_topology_section_primary');
}

function shouldRenderTopologyFlow(tier: TopologyTier): boolean {
  return tier === 'primary' || tier === 'fallback';
}

function SectionAnchor({
  id,
  children,
}: {
  id: string;
  children: ReactNode;
}) {
  return (
    <div className="workspace-section-anchor" data-workspace-section={id} id={id}>
      {children}
    </div>
  );
}

function WorkspaceQuickNav({
  section,
  focus,
  pageSections,
  activeSectionId,
}: {
  section: WorkspaceSection;
  focus?: WorkspaceFocus;
  pageSections: ReturnType<typeof getWorkspacePageSections>;
  activeSectionId?: string;
}) {
  if (!pageSections.length) return null;
  return (
    <nav className="workspace-quick-nav workspace-quick-nav-mobile" aria-label="workspace-mobile-quick-nav">
      {pageSections.map((item) => (
        <Link
          key={item.id}
          aria-current={activeSectionId === item.id ? 'location' : undefined}
          className={`chip-button workspace-quick-nav-link level-${item.level} ${activeSectionId === item.id ? 'active' : ''}`.trim()}
          to={buildWorkspacePageLink(section, focus, item.id)}
        >
          {item.label}
        </Link>
      ))}
    </nav>
  );
}

function scrollToWorkspaceSection(sectionId: string) {
  if (typeof document === 'undefined') return;
  const node = document.querySelector<HTMLElement>(`[data-workspace-section="${sectionId}"]`);
  node?.scrollIntoView?.({ block: 'start', behavior: 'auto' });
}

function bindFieldValues(fields: ViewFieldSchema[], values: Record<string, unknown>) {
  return fields.map((field) => ({
    ...field,
    value: values[field.key],
  }));
}

function normalizeFocusValue(value: unknown): string {
  return safeDisplayValue(value).trim().toLowerCase();
}

function matchesArtifactFocus(row: ArtifactRowState, focus: string | undefined): boolean {
  const target = normalizeFocusValue(focus);
  if (!target || target === '—') return false;
  const candidates = [
    row.id,
    row.payload_key,
    row.label,
    row.path,
  ].map((value) => safeDisplayValue(value));

  return candidates.some((candidate) => {
    const raw = normalizeFocusValue(candidate);
    if (!raw || raw === '—') return false;
    return raw === target || raw.includes(target) || target.includes(raw);
  });
}

function resolveFocusedArtifactId(rows: ArtifactRowState[], focus: WorkspaceFocus | undefined): string {
  const target = normalizeFocusValue(focus?.artifact);
  if (!target || target === '—') return '';

  const exactIdMatch = rows.find((row) => normalizeFocusValue(row.id) === target);
  if (exactIdMatch) return safeDisplayValue(exactIdMatch.id);

  const exactPayloadMatch = rows.find((row) => normalizeFocusValue(row.payload_key) === target);
  if (exactPayloadMatch) return safeDisplayValue(exactPayloadMatch.id);

  const exactLabelOrPathMatch = rows.find((row) => [row.label, row.path]
    .map((value) => normalizeFocusValue(value))
    .some((value) => value && value !== '—' && value === target));
  if (exactLabelOrPathMatch) return safeDisplayValue(exactLabelOrPathMatch.id);

  return safeDisplayValue(rows.find((row) => matchesArtifactFocus(row, focus?.artifact))?.id);
}

function artifactLayer(row: ArtifactRowState): 'canonical' | 'archive' {
  const explicit = normalizeFocusValue(row.artifact_layer);
  if (explicit === 'canonical' || explicit === 'archive') return explicit;
  const id = normalizeFocusValue(row.id);
  const payloadKey = normalizeFocusValue(row.payload_key);
  if (payloadKey && id && payloadKey === id) return 'canonical';
  return 'archive';
}

function artifactGroup(row: ArtifactRowState): 'research_mainline' | 'system_anchor' | 'research_exit_risk' | 'research_hold_transfer' | 'research_cross_section' | 'archive' {
  const explicit = normalizeFocusValue(row.artifact_group);
  if (
    explicit === 'research_mainline'
    || explicit === 'system_anchor'
    || explicit === 'research_exit_risk'
    || explicit === 'research_hold_transfer'
    || explicit === 'research_cross_section'
    || explicit === 'archive'
  ) {
    return explicit;
  }
  return artifactLayer(row) === 'archive' ? 'archive' : 'research_hold_transfer';
}

function normalizeArtifactGroup(value: unknown): ArtifactGroupId | null {
  const raw = normalizeFocusValue(value);
  return ARTIFACT_GROUP_ORDER.find((group) => group === raw) || null;
}

function normalizeSearchScope(value: unknown): ArtifactSearchScope {
  const raw = normalizeFocusValue(value);
  if (raw === 'artifact' || raw === 'path') return raw;
  return 'title';
}

function artifactSearchText(row: ArtifactRowState, scope: ArtifactSearchScope): string {
  if (scope === 'artifact') {
    return [row.id, row.payload_key].map((value) => normalizeFocusValue(value)).join(' ');
  }
  if (scope === 'path') {
    return normalizeFocusValue(row.path);
  }
  return [row.payload_key, row.label, row.id].map((value) => normalizeFocusValue(value)).join(' ');
}

function artifactPath(row: ArtifactRowState, payloadPath?: unknown): string {
  const rowPath = safeDisplayValue(row.path);
  if (rowPath !== '—') return rowPath;
  const fallbackPath = safeDisplayValue(payloadPath);
  return fallbackPath !== '—' ? fallbackPath : '—';
}

function artifactGroupLabel(group: ArtifactGroupId): string {
  return t(ARTIFACT_GROUP_LABEL_KEYS[group]);
}

function artifactGroupClass(group: ArtifactGroupId): string {
  if (group === 'research_mainline') return 'artifact-layer-mainline';
  if (group === 'system_anchor') return 'artifact-layer-anchor';
  if (group === 'research_exit_risk') return 'artifact-layer-exit';
  if (group === 'research_hold_transfer') return 'artifact-layer-hold';
  if (group === 'research_cross_section') return 'artifact-layer-cross';
  return 'artifact-layer-archive';
}

function renderArtifactButton({
  row,
  selected,
  onSelect,
  showRawTitle,
  showRawPath,
  titleFor,
}: {
  row: ArtifactRowState;
  selected: boolean;
  onSelect: (id: string) => void;
  showRawTitle: boolean;
  showRawPath: boolean;
  titleFor: (row: ArtifactRowState | null) => string;
}) {
  const id = safeDisplayValue(row.id);
  const layer = artifactLayer(row);
  return (
    <EntityRowButton
      key={id}
      className={`artifact-button artifact-button-${layer} ${selected ? 'active' : ''}`.trim()}
      title={<ValueText value={titleFor(row)} showRaw={showRawTitle} expandable={false} />}
      subtitle={labelFor(row.category as string | undefined)}
      active={selected}
      onClick={() => onSelect(id)}
    >
      <Badge value={row.research_decision || row.status} />
      <small className="control-entity-row-subtitle">
        <PathText value={row.path} showRaw={showRawPath} expandable={false} />
      </small>
    </EntityRowButton>
  );
}

function ArtifactHandoffs({
  links,
  title,
}: {
  links: OperatorAlertLink[];
  title: string;
}) {
  if (!links.length) return null;
  return (
    <DrilldownSection title={title} summary={`${links.length} ${t('workspace_handoff_summary_suffix')}`} defaultOpen>
      <div className="button-row workspace-handoff-links">
        {links.map((link) => (
          <Link key={link.id} className={`button ${link.tone === 'negative' ? 'button-primary' : ''}`.trim()} to={link.to}>
            {link.label}
          </Link>
        ))}
      </div>
    </DrilldownSection>
  );
}

function jin10SnapshotSummary(payload: Record<string, unknown>) {
  const summary = (payload.summary as Record<string, unknown> | undefined) || {};
  const highImportanceTitles = Array.isArray(summary.high_importance_titles)
    ? summary.high_importance_titles.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  const latestFlashBriefs = Array.isArray(summary.latest_flash_briefs)
    ? summary.latest_flash_briefs.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  const quoteWatch = Array.isArray(summary.quote_watch)
    ? summary.quote_watch.map((row) => {
      const record = (row || {}) as Record<string, unknown>;
      const name = safeDisplayValue(record.name || record.code);
      const close = safeDisplayValue(record.close);
      const ups = safeDisplayValue(record.ups_percent);
      return [name, close !== '—' ? close : '', ups !== '—' ? `${ups}%` : ''].filter(Boolean).join(' ');
    }).filter((value) => value && value !== '—')
    : [];
  return {
    recommendedBrief: safeDisplayValue(payload.recommended_brief),
    takeaway: safeDisplayValue(payload.takeaway),
    calendarTotal: safeDisplayValue(summary.calendar_total),
    highImportanceCount: safeDisplayValue(summary.high_importance_count),
    flashTotal: safeDisplayValue(summary.flash_total),
    quoteWatchTotal: Array.isArray(summary.quote_watch) ? String(summary.quote_watch.length) : '0',
    highImportanceTitles,
    latestFlashBriefs,
    quoteWatch,
  };
}

function axiosSnapshotSummary(payload: Record<string, unknown>) {
  const summary = (payload.summary as Record<string, unknown> | undefined) || {};
  const topTitles = Array.isArray(summary.top_titles)
    ? summary.top_titles.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  const topKeywords = Array.isArray(summary.top_keywords)
    ? summary.top_keywords.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  return {
    recommendedBrief: safeDisplayValue(payload.recommended_brief),
    takeaway: safeDisplayValue(payload.takeaway),
    newsTotal: safeDisplayValue(summary.news_total),
    localTotal: safeDisplayValue(summary.local_total),
    nationalTotal: safeDisplayValue(summary.national_total),
    topTitles,
    topKeywords,
  };
}

function polymarketSnapshotSummary(payload: Record<string, unknown>) {
  const summary = (payload.summary as Record<string, unknown> | undefined) || {};
  const topTitles = Array.isArray(summary.top_titles)
    ? summary.top_titles.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  const topCategories = Array.isArray(summary.top_categories)
    ? summary.top_categories.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  const yesPriceAvg = typeof summary.yes_price_avg === 'number'
    ? `${(Number(summary.yes_price_avg) * 100).toFixed(1)}%`
    : safeDisplayValue(summary.yes_price_avg);
  return {
    recommendedBrief: safeDisplayValue(payload.recommended_brief),
    takeaway: safeDisplayValue(payload.takeaway),
    marketsTotal: safeDisplayValue(summary.markets_total),
    bullishCount: safeDisplayValue(summary.bullish_count),
    bearishCount: safeDisplayValue(summary.bearish_count),
    yesPriceAvg,
    topTitles,
    topCategories,
  };
}

function externalIntelligenceSummary(payload: Record<string, unknown>) {
  const summary = (payload.summary as Record<string, unknown> | undefined) || {};
  const topTitles = Array.isArray(summary.top_titles)
    ? summary.top_titles.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  const topKeywords = Array.isArray(summary.top_keywords)
    ? summary.top_keywords.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  const activeSources = Array.isArray(summary.active_sources)
    ? summary.active_sources.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  const polymarketTopCategories = Array.isArray(summary.polymarket_top_categories)
    ? summary.polymarket_top_categories.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  const quoteWatch = Array.isArray(summary.quote_watch)
    ? summary.quote_watch.map((value) => safeDisplayValue(value)).filter((value) => value && value !== '—')
    : [];
  const polymarketYesPriceAvg = typeof summary.polymarket_yes_price_avg === 'number'
    ? `${(Number(summary.polymarket_yes_price_avg) * 100).toFixed(1)}%`
    : safeDisplayValue(summary.polymarket_yes_price_avg);
  return {
    recommendedBrief: safeDisplayValue(payload.recommended_brief),
    takeaway: safeDisplayValue(payload.takeaway),
    sourcesTotal: safeDisplayValue(summary.sources_total),
    activeSources,
    calendarTotal: safeDisplayValue(summary.calendar_total),
    highImportanceCount: safeDisplayValue(summary.high_importance_count),
    flashTotal: safeDisplayValue(summary.flash_total),
    newsTotal: safeDisplayValue(summary.axios_news_total),
    localTotal: safeDisplayValue(summary.axios_local_total),
    nationalTotal: safeDisplayValue(summary.axios_national_total),
    polymarketMarketsTotal: safeDisplayValue(summary.polymarket_markets_total),
    polymarketYesPriceAvg,
    polymarketBullishCount: safeDisplayValue(summary.polymarket_bullish_count),
    polymarketBearishCount: safeDisplayValue(summary.polymarket_bearish_count),
    topTitles,
    topKeywords,
    polymarketTopCategories,
    quoteWatch,
  };
}

function buildResearchAuditCaseLinks(
  cases: PublicAcceptanceResearchAuditCase[] | undefined,
  idPrefix: string,
): OperatorAlertLink[] {
  if (!cases?.length) return [];
  const links: OperatorAlertLink[] = [];
  cases.forEach((row, index) => {
    const query = safeDisplayValue(row.query);
    const resultArtifact = safeDisplayValue(row.result_artifact);
    const rawPath = safeDisplayValue(row.raw_path);
    const searchRoute = safeDisplayValue(row.search_route) || (query !== '—' ? buildSearchLink(query, 'artifact') : '');
    const workspaceRoute = safeDisplayValue(row.workspace_route)
      || (resultArtifact !== '—' ? buildWorkspaceLink('artifacts', { artifact: resultArtifact }) : '');
    const rawRoute = rawPath !== '—' ? buildWorkspaceLink('raw', { artifact: rawPath }) : '';
    if (searchRoute) {
      links.push({
        id: `${idPrefix}-${index}-search`,
        label: `检索 / ${query !== '—' ? query : safeDisplayValue(row.case_id)}`,
        to: searchRoute,
      });
    }
    if (workspaceRoute) {
      links.push({
        id: `${idPrefix}-${index}-artifact`,
        label: `工件 / ${resultArtifact !== '—' ? resultArtifact : safeDisplayValue(row.case_id)}`,
        to: workspaceRoute,
      });
    }
    if (rawRoute) {
      links.push({
        id: `${idPrefix}-${index}-raw`,
        label: `原始层 / ${rawPath !== '—' ? compactPath(rawPath) : safeDisplayValue(row.case_id)}`,
        to: rawRoute,
      });
    }
  });
  return links;
}

function aggregateResearchAuditCasesFromChecks(
  checks: Array<{ research_audit_cases?: PublicAcceptanceResearchAuditCase[] }>,
): PublicAcceptanceResearchAuditCase[] {
  const seen = new Set<string>();
  return checks.flatMap((row) => row.research_audit_cases || []).filter((row) => {
    const key = [
      safeDisplayValue(row.case_id),
      safeDisplayValue(row.search_route),
      safeDisplayValue(row.workspace_route),
      safeDisplayValue(row.raw_path),
    ].join('|');
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function preferredContractsAccordionId(rows: Array<{ id?: string }>, preferredId: string): string {
  const preferred = rows.find((row) => safeDisplayValue(row.id) === preferredId);
  if (preferred) return safeDisplayValue(preferred.id);
  return safeDisplayValue(rows[0]?.id);
}

function preferredFeedbackEventId(
  feedback: TerminalReadModel['feedback']['projection'] | null | undefined,
): string {
  const actionFeedbackId = safeDisplayValue(feedback?.actions.find((row) => safeDisplayValue(row.feedback_id) !== '—')?.feedback_id);
  if (actionFeedbackId !== '—' && feedback?.events.some((row) => safeDisplayValue(row.feedback_id) === actionFeedbackId)) {
    return actionFeedbackId;
  }
  return safeDisplayValue(feedback?.events[0]?.feedback_id);
}

export function WorkspacePanels({
  model,
  section,
  focus,
  pageSectionId,
}: {
  model: TerminalReadModel;
  section: 'artifacts' | 'alignment' | 'backtests' | 'contracts' | 'raw';
  focus?: WorkspaceFocus;
  pageSectionId?: string;
}) {
  if (section === 'alignment') return <AlignmentWorkspace model={model} pageSectionId={pageSectionId} />;
  if (section === 'backtests') return <BacktestsWorkspace model={model} focus={focus} pageSectionId={pageSectionId} />;
  if (section === 'contracts') return <ContractsWorkspace model={model} focus={focus} pageSectionId={pageSectionId} />;
  if (section === 'raw') return <RawWorkspace model={model} focus={focus} pageSectionId={pageSectionId} />;
  return <ArtifactsWorkspace model={model} focus={focus} pageSectionId={pageSectionId} />;
}

function ArtifactsWorkspace({ model, focus, pageSectionId }: { model: TerminalReadModel; focus?: WorkspaceFocus; pageSectionId?: string }) {
  const { tier } = useShellBreakpoint();
  const compactWorkspace = tier === 's';
  const pageSections = useMemo(() => getWorkspacePageSections('artifacts', model), [model]);
  const [mobileFiltersOpen, setMobileFiltersOpen] = useState<boolean>(() => !compactWorkspace);
  const { artifactRows, artifactPayloads, artifactHandoffs, artifactPrimaryFocus } = model.workspace;
  const view = model.view.workspace.artifacts;
  const [searchParams, setSearchParams] = useSearchParams();
  const toneFilter = focus?.tone || 'all';
  const search = focus?.search || '';
  const searchScope = normalizeSearchScope(focus?.search_scope);
  const focusedId = resolveFocusedArtifactId(artifactRows as unknown as Array<Record<string, unknown>>, focus);
  const focusedRow = artifactRows.find((row) => safeDisplayValue(row.id) === focusedId) || null;
  const focusedRowGroup = focusedRow ? artifactGroup(focusedRow as ArtifactRowState) : null;
  const explicitGroup = normalizeArtifactGroup(focus?.group);
  const activeGroup: ArtifactGroupId = focusedRowGroup && explicitGroup && focusedRowGroup !== explicitGroup
    ? focusedRowGroup
    : explicitGroup || focusedRowGroup || 'research_mainline';

  const groupedRows = useMemo(() => {
    const grouped = Object.fromEntries(ARTIFACT_GROUP_ORDER.map((group) => [group, [] as ArtifactRowState[]])) as Record<ArtifactGroupId, ArtifactRowState[]>;
    artifactRows.forEach((row) => {
      const typedRow = row as ArtifactRowState;
      grouped[artifactGroup(typedRow)].push(typedRow);
    });
    return grouped;
  }, [artifactRows]);

  const primaryRows = useMemo(
    () =>
      Object.fromEntries(
        ARTIFACT_GROUP_ORDER.map((group) => [
          group,
          groupedRows[group].find((row) => artifactLayer(row) === 'canonical') || groupedRows[group][0] || null,
        ]),
      ) as Record<ArtifactGroupId, ArtifactRowState | null>,
    [groupedRows],
  );

  const currentGroupRows = groupedRows[activeGroup] || [];
  const statusCounts = useMemo(
    () =>
      Object.fromEntries(
        ARTIFACT_STATUS_ORDER.map((tone) => [
          tone,
          currentGroupRows.filter((row) => statusTone(row.research_decision || row.status) === tone).length,
        ]),
      ) as Record<(typeof ARTIFACT_STATUS_ORDER)[number], number>,
    [currentGroupRows],
  );

  const filteredRows = useMemo(
    () =>
      currentGroupRows.filter((row) => {
        const rowTone = statusTone(row.research_decision || row.status);
        if (toneFilter !== 'all' && rowTone !== toneFilter) return false;
        if (!search.trim()) return true;
        return artifactSearchText(row, searchScope).includes(search.trim().toLowerCase());
      }),
    [currentGroupRows, toneFilter, search, searchScope],
  );

  const selectedId = useMemo(() => {
    if (focusedId && filteredRows.some((row) => safeDisplayValue(row.id) === focusedId)) return focusedId;
    const primary = primaryRows[activeGroup];
    if (!focusedId && !search.trim() && toneFilter === 'all' && primary && filteredRows.some((row) => safeDisplayValue(row.id) === safeDisplayValue(primary.id))) {
      return safeDisplayValue(primary.id);
    }
    if (filteredRows.length) return safeDisplayValue(filteredRows[0]?.id);
    if (focusedId && focusedRowGroup === activeGroup) return focusedId;
    return primary ? safeDisplayValue(primary.id) : '';
  }, [focusedId, filteredRows, focusedRowGroup, activeGroup, primaryRows, search, toneFilter]);

  useEffect(() => {
    const next = new URLSearchParams(searchParams);
    next.delete('domain');

    if (toneFilter && toneFilter !== 'all') next.set('tone', toneFilter);
    else next.delete('tone');

    if (search.trim()) next.set('search', search.trim());
    else next.delete('search');

    next.set('group', activeGroup);
    next.set('search_scope', searchScope);

    if (selectedId) next.set('artifact', selectedId);
    else next.delete('artifact');

    const primaryFocus = selectedId ? artifactPrimaryFocus[selectedId] : undefined;
    const currentArtifactParam = searchParams.get('artifact') || '';
    const shouldAdoptPrimaryFocus = Boolean(
      selectedId
      && primaryFocus?.panel
      && primaryFocus?.section
      && (
        currentArtifactParam !== selectedId
        || searchParams.get('panel') !== primaryFocus.panel
        || searchParams.get('section') !== primaryFocus.section
        || (primaryFocus?.row || '') !== (searchParams.get('row') || '')
      ),
    );

    if (shouldAdoptPrimaryFocus) {
      if (primaryFocus?.panel) next.set('panel', primaryFocus.panel);
      else next.delete('panel');
      if (primaryFocus?.section) next.set('section', primaryFocus.section);
      else next.delete('section');
      if (primaryFocus?.row) next.set('row', primaryFocus.row);
      else next.delete('row');
    }

    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, { replace: true });
    }
  }, [activeGroup, artifactPrimaryFocus, search, searchParams, searchScope, selectedId, setSearchParams, toneFilter]);

  const selectedRow = artifactRows.find((row) => safeDisplayValue(row.id) === selectedId) || null;
  const selectedPayloadKey = selectedRow?.payload_key ? safeDisplayValue(selectedRow.payload_key) : '';
  const payload = selectedPayloadKey ? artifactPayloads[selectedPayloadKey]?.payload : null;
  const payloadRecord = payload && typeof payload === 'object' && !Array.isArray(payload) ? (payload as Record<string, unknown>) : {};
  const artifactTitle = (row: ArtifactRowState | null) => safeDisplayValue(row?.payload_key || row?.label || row?.id);
  const selectedArtifactPath = selectedRow ? artifactPath(selectedRow as ArtifactRowState, artifactPayloads[selectedPayloadKey]?.path) : '—';
  const selectedRowId = selectedRow?.id ? safeDisplayValue(selectedRow.id) : '';
  const handoffLinks = selectedRowId ? artifactHandoffs[selectedRowId] || [] : [];
  const activeSectionId = pageSectionId
    ? (findWorkspacePageSection('artifacts', model, pageSectionId)?.id || undefined)
    : (getDefaultWorkspacePageSection('artifacts', model)?.id || undefined);

  useEffect(() => {
    setMobileFiltersOpen(!compactWorkspace);
  }, [compactWorkspace]);

  useEffect(() => {
    const target = findWorkspacePageSection('artifacts', model, pageSectionId) || (!pageSectionId ? getDefaultWorkspacePageSection('artifacts', model) : null);
    if (!target) return;
    const handle = window.requestAnimationFrame(() => {
      scrollToWorkspaceSection(target.id);
    });
    return () => window.cancelAnimationFrame(handle);
  }, [model, pageSectionId, pageSections]);

  const fallbackFocusLinks: OperatorAlertLink[] = selectedRowId
    ? [
        {
          id: `${selectedRowId}-artifact-self`,
          label: t('workspace_artifacts_focus_action_artifact'),
          to: buildWorkspaceLink('artifacts', {
            artifact: selectedRowId,
            group: activeGroup,
            tone: toneFilter !== 'all' ? toneFilter : undefined,
            search: search.trim() || undefined,
            search_scope: searchScope,
          }),
        },
        {
          id: `${selectedRowId}-raw-self`,
          label: t('workspace_artifacts_focus_action_raw'),
          to: buildWorkspaceLink('raw', {
            artifact: selectedArtifactPath !== '—' ? selectedArtifactPath : selectedRowId,
            group: activeGroup,
            tone: toneFilter !== 'all' ? toneFilter : undefined,
            search: search.trim() || undefined,
            search_scope: searchScope,
          }),
        },
        ...(artifactPrimaryFocus[selectedRowId]?.panel
          ? [{
              id: `${selectedRowId}-terminal-self`,
              label: t('workspace_artifacts_focus_action_terminal'),
              to: buildTerminalLink(model.surface.effective, {
                artifact: selectedRowId,
                group: activeGroup,
                tone: toneFilter !== 'all' ? toneFilter : undefined,
                search: search.trim() || undefined,
                search_scope: searchScope,
                panel: artifactPrimaryFocus[selectedRowId]?.panel,
                section: artifactPrimaryFocus[selectedRowId]?.section,
                row: artifactPrimaryFocus[selectedRowId]?.row,
              }),
            }]
          : []),
      ]
    : [];
  const focusLinks = [...handoffLinks, ...fallbackFocusLinks].filter(
    (link, index, rows) => rows.findIndex((candidate) => candidate.to === link.to) === index,
  );

  const setArtifactQuery = (artifactId: string | undefined, overrides: Record<string, string | undefined> = {}) => {
    const next = new URLSearchParams(searchParams);
    next.delete('domain');
    next.set('group', overrides.group || activeGroup);
    next.set('search_scope', overrides.search_scope || searchScope);
    const tone = overrides.tone !== undefined ? overrides.tone : toneFilter !== 'all' ? toneFilter : undefined;
    const searchValue = overrides.search !== undefined ? overrides.search : search.trim() || undefined;
    if (tone) next.set('tone', tone);
    else next.delete('tone');
    if (searchValue) next.set('search', searchValue);
    else next.delete('search');
    if (artifactId) next.set('artifact', artifactId);
    else next.delete('artifact');
    setSearchParams(next, { replace: true });
  };

  const handleGroupSelect = (group: ArtifactGroupId) => {
    const nextPrimary = primaryRows[group];
    setArtifactQuery(nextPrimary ? safeDisplayValue(nextPrimary.id) : undefined, {
      group,
      tone: undefined,
    });
  };

  const handleToneToggle = (tone: (typeof ARTIFACT_STATUS_ORDER)[number]) => {
    setArtifactQuery(selectedId || undefined, {
      tone: toneFilter === tone ? undefined : tone,
    });
  };

  const handleSearchChange = (value: string) => {
    setArtifactQuery(selectedId || undefined, {
      search: value,
    });
  };

  const handleSearchScopeChange = (scope: ArtifactSearchScope) => {
    setArtifactQuery(selectedId || undefined, {
      search_scope: scope,
    });
  };

  return (
    <div className={`workspace-grid artifacts-grid ${compactWorkspace ? 'artifacts-grid-compact' : ''}`.trim()}>
      {compactWorkspace ? (
        <WorkspaceQuickNav section="artifacts" focus={focus} pageSections={pageSections} activeSectionId={activeSectionId} />
      ) : null}
      <section
        className={`workspace-left-rail workspace-rhythm-band workspace-rhythm-state ${compactWorkspace ? 'workspace-left-rail-mobile' : ''}`.trim()}
        aria-label="workspace-rhythm-state"
        hidden={compactWorkspace ? !mobileFiltersOpen : undefined}
      >
          <h3>当前状态</h3>
        <div data-search-anchor="workspace-artifacts-map-panel" id="workspace-artifacts-map-panel">
        <PanelCard title={t('workspace_artifacts_map_title')} kicker={t('workspace_artifacts_map_kicker')} meta={t('workspace_artifacts_map_meta')}>
          <div className="stack-list">
            {ARTIFACT_GROUP_ORDER.map((group) => {
              const primary = primaryRows[group];
              return (
                <button
                  key={group}
                  className={`stack-button artifact-group-button ${activeGroup === group ? 'active' : ''}`.trim()}
                  onClick={() => handleGroupSelect(group)}
                >
                  <span className="nav-link-copy">
                    <strong>{artifactGroupLabel(group)}</strong>
                    <small>{primary ? labelFor(artifactTitle(primary)) : t('workspace_artifacts_group_empty')}</small>
                  </span>
                  <span>{safeDisplayValue(groupedRows[group].length)}</span>
                </button>
              );
            })}
          </div>
        </PanelCard>
        </div>

        <div data-search-anchor="workspace-artifacts-status-panel" id="workspace-artifacts-status-panel">
        <PanelCard title={t('workspace_artifacts_status_title')} kicker={t('workspace_artifacts_status_kicker')} meta={`${artifactGroupLabel(activeGroup)} ｜ ${t('workspace_artifacts_status_meta')}`}>
          <div className="stack-list">
            <button className={`stack-button artifact-status-button ${toneFilter === 'all' ? 'active' : ''}`.trim()} onClick={() => setArtifactQuery(selectedId || undefined, { tone: undefined })}>
              <span>{t('workspace_artifacts_all_status')}</span>
              <span>{safeDisplayValue(currentGroupRows.length)}</span>
            </button>
            {ARTIFACT_STATUS_ORDER.map((tone) => (
              <button
                key={tone}
                className={`stack-button artifact-status-button ${toneFilter === tone ? 'active' : ''}`.trim()}
                onClick={() => handleToneToggle(tone)}
              >
                <span>{labelFor(tone)}</span>
                <span>{safeDisplayValue(statusCounts[tone])}</span>
              </button>
            ))}
          </div>
        </PanelCard>
        </div>

        <div data-search-anchor="workspace-artifacts-search-panel" id="workspace-artifacts-search-panel">
        <PanelCard title={t('workspace_artifacts_search_panel_title')} kicker={t('workspace_artifacts_search_panel_kicker')} meta={t('workspace_artifacts_search_panel_meta')}>
          <div className="chip-row artifact-search-scope-row">
            {(['title', 'artifact', 'path'] as ArtifactSearchScope[]).map((scope) => (
              <button
                key={scope}
                className={`chip-button ${searchScope === scope ? 'active' : ''}`.trim()}
                onClick={() => handleSearchScopeChange(scope)}
              >
                {t(`workspace_artifacts_search_scope_${scope}`)}
              </button>
            ))}
          </div>
          <input
            className="search-input"
            value={search}
            onChange={(event) => handleSearchChange(event.target.value)}
            placeholder={t('workspace_artifacts_search_placeholder')}
          />
        </PanelCard>
        </div>
      </section>

      <div className="workspace-rhythm-main">
        {compactWorkspace ? (
          <section className="workspace-rhythm-band workspace-rhythm-state workspace-mobile-state-toggle" aria-label="workspace-mobile-state-toggle">
            <h3>筛选与范围</h3>
            <PanelCard
              title="筛选与范围"
              kicker="mobile / compact"
              meta={mobileFiltersOpen ? '已展开分组、状态与搜索面板' : '先看焦点与证据，按需展开筛选'}
              maximizable={false}
              actions={(
                <ActionButton
                  aria-label={mobileFiltersOpen ? '收起筛选与范围' : '展开筛选与范围'}
                  onClick={() => setMobileFiltersOpen((open) => !open)}
                >
                  {mobileFiltersOpen ? '收起筛选与范围' : '展开筛选与范围'}
                </ActionButton>
              )}
            >
              <KeyValueGrid
                rows={[
                  { key: 'group', label: '当前分组', value: artifactGroupLabel(activeGroup) },
                  { key: 'status', label: '状态过滤', value: toneFilter === 'all' ? '全部状态' : labelFor(toneFilter) },
                  { key: 'scope', label: '搜索范围', value: t(`workspace_artifacts_search_scope_${searchScope}`) },
                  { key: 'search', label: '当前搜索', value: search.trim() || '未设置' },
                ]}
              />
            </PanelCard>
          </section>
        ) : null}
        <section className="workspace-rhythm-band workspace-rhythm-focus" aria-label="workspace-rhythm-focus">
          <h3>当前焦点</h3>
        <div data-search-anchor="workspace-artifacts-focus-panel" id="workspace-artifacts-focus-panel">
        <PanelCard title={t('workspace_artifacts_focus_title')} kicker={t('workspace_artifacts_focus_kicker')} meta={t('workspace_artifacts_focus_meta')}>
          {selectedRow ? (
            <KeyValueGrid
              rows={[
                { ...c('artifact_title', 'workspace_artifacts_focus_field_title'), value: artifactTitle(selectedRow), critical: true },
                { ...c('artifact_group', 'workspace_artifacts_focus_field_group'), value: artifactGroupLabel(activeGroup) },
                { ...c('category', 'category'), value: selectedRow.category || '—' },
                { ...c('status', 'status'), value: selectedRow.research_decision || selectedRow.status || '—', kind: 'badge' as const },
                { ...c('path', 'path'), value: selectedArtifactPath, kind: 'path' as const, showRaw: true },
                ...(selectedPayloadKey && selectedPayloadKey !== selectedRowId
                  ? [{ ...c('payload_key', 'payload_key'), value: selectedPayloadKey }]
                  : []),
              ]}
            />
          ) : (
            <div className="empty-block">{t('workspace_artifacts_focus_empty')}</div>
          )}
        </PanelCard>
        </div>
        </section>

        <section className="workspace-rhythm-band workspace-rhythm-action" aria-label="workspace-rhythm-action">
          <h3>下一步</h3>
        <div data-search-anchor="workspace-artifacts-next-step-panel" id="workspace-artifacts-next-step-panel">
        <PanelCard title="下一步动作" kicker="handoff / action" meta={selectedRow ? artifactTitle(selectedRow) : t('workspace_artifacts_focus_empty')}>
          {selectedRow ? (
            <div className="button-row workspace-handoff-links">
              {focusLinks.map((link) => (
                <Link key={link.id} className={`button ${link.tone === 'negative' ? 'button-primary' : ''}`.trim()} to={link.to}>
                  {link.label}
                </Link>
              ))}
            </div>
          ) : (
            <div className="empty-block">{t('workspace_artifacts_focus_empty')}</div>
          )}
        </PanelCard>
        </div>
        </section>

        <section className="workspace-rhythm-band workspace-rhythm-evidence" aria-label="workspace-rhythm-evidence">
          <h3>证据</h3>
        <div data-search-anchor="workspace-artifacts-target-pool" id="workspace-artifacts-target-pool">
        <PanelCard
          title={t('workspace_artifacts_target_pool_title')}
          kicker={t('workspace_artifacts_target_pool_kicker')}
          meta={`${artifactGroupLabel(activeGroup)} ｜ ${t('workspace_artifacts_filtered_prefix')} ${filteredRows.length} ${t('workspace_artifacts_target_pool_meta_suffix')}`}
        >
          <div className="scroll-region">
            <DrilldownSection
              title={artifactGroupLabel(activeGroup)}
              summary={`${filteredRows.length} ${labelFor('unit_items')} ｜ ${t(`workspace_artifacts_map_${activeGroup}_summary`)}`}
              className={`artifact-layer-section ${artifactGroupClass(activeGroup)}`}
              defaultOpen
            >
              <div className="stack-list">
                {filteredRows.length ? filteredRows.map((row) => renderArtifactButton({
                  row,
                  selected: selectedId === safeDisplayValue(row.id),
                  onSelect: (id) => setArtifactQuery(id),
                  showRawTitle: view.list.titleField.showRaw ?? false,
                  showRawPath: view.list.pathField.showRaw ?? false,
                  titleFor: artifactTitle,
                })) : <div className="empty-block">{t('workspace_artifacts_no_matches')}</div>}
              </div>
            </DrilldownSection>
          </div>
        </PanelCard>
        </div>

        <PanelCard
          title={(
            <ValueText
              value={artifactTitle(selectedRow)}
              showRaw={view.inspector.titleField.showRaw}
            />
          )}
          kicker={t('workspace_artifacts_inspector_kicker')}
          meta={(
            <PathText
              value={selectedRow?.path || t('workspace_artifacts_unselected')}
              showRaw={view.inspector.pathField.showRaw}
            />
          )}
        >
          {selectedRow ? (
            <>
              <div data-search-anchor="workspace-artifacts-inspector-panel" id="workspace-artifacts-inspector-panel" />
              <DrilldownSection title={t('workspace_artifacts_meta_title')} summary={labelFor(safeDisplayValue(selectedRow.status || selectedRow.research_decision))} defaultOpen>
                <KeyValueGrid
                  rows={[
                    ...bindFieldValues(view.metaFields, {
                      category: selectedRow.category || '—',
                      status: selectedRow.status || selectedRow.research_decision || '—',
                      change_class: selectedRow.change_class || '—',
                      generated_at_utc: selectedRow.generated_at_utc || '—',
                      payload_key: selectedRow.payload_key || '—',
                    }),
                    { ...c('path'), value: compactPath(selectedArtifactPath) },
                  ]}
                />
              </DrilldownSection>
              <DrilldownSection title={t('workspace_artifacts_top_keys_title')} summary={Array.isArray(selectedRow.top_level_keys) ? `${selectedRow.top_level_keys.length} ${t('workspace_summary_keys_suffix')}` : t('workspace_summary_none')} defaultOpen>
                <div className="empty-block">
                  {Array.isArray(selectedRow.top_level_keys) && selectedRow.top_level_keys.length ? (
                    selectedRow.top_level_keys.map((key, index) => (
                      <span key={`${key}-${index}`}>
                        {index > 0 ? ' ｜ ' : null}
                        <ValueText value={key} showRaw={view.topLevelKeyField.showRaw} />
                      </span>
                    ))
                  ) : t('workspace_artifacts_no_keys')}
                </div>
              </DrilldownSection>
              {selectedPayloadKey === 'jin10_mcp_snapshot' ? (
                <DrilldownSection title="Jin10 侧边车摘要" summary={safeDisplayValue(payloadRecord.recommended_brief || payloadRecord.takeaway || 'research sidecar')} defaultOpen>
                  {(() => {
                    const jin10 = jin10SnapshotSummary(payloadRecord);
                    return (
                      <>
                        <KeyValueGrid
                          rows={[
                            { ...c('calendar_total', 'summary'), label: '日历事件', value: jin10.calendarTotal, kind: 'number' as const },
                            { ...c('high_importance_count', 'summary'), label: '高重要事件', value: jin10.highImportanceCount, kind: 'number' as const },
                            { ...c('flash_total', 'summary'), label: '最新快讯', value: jin10.flashTotal, kind: 'number' as const },
                            { ...c('quote_watch_total', 'summary'), label: '盯盘品种', value: jin10.quoteWatchTotal, kind: 'number' as const },
                            { ...c('recommended_brief', 'recommended_brief'), value: jin10.recommendedBrief, showRaw: true },
                            { ...c('takeaway', 'takeaway'), value: jin10.takeaway, showRaw: true },
                          ]}
                        />
                        {jin10.highImportanceTitles.length ? <div className="empty-block">高重要日历：{jin10.highImportanceTitles.join(' ｜ ')}</div> : null}
                        {jin10.latestFlashBriefs.length ? <div className="empty-block">最新快讯：{jin10.latestFlashBriefs.join(' ｜ ')}</div> : null}
                        {jin10.quoteWatch.length ? <div className="empty-block">盯盘摘要：{jin10.quoteWatch.join(' ｜ ')}</div> : null}
                      </>
                    );
                  })()}
                </DrilldownSection>
              ) : null}
              {selectedPayloadKey === 'axios_site_snapshot' ? (
                <DrilldownSection title="Axios 侧边车摘要" summary={safeDisplayValue(payloadRecord.recommended_brief || payloadRecord.takeaway || 'research sidecar')} defaultOpen>
                  {(() => {
                    const axios = axiosSnapshotSummary(payloadRecord);
                    return (
                      <>
                        <KeyValueGrid
                          rows={[
                            { ...c('news_total', 'summary'), label: 'Axios 条目', value: axios.newsTotal, kind: 'number' as const },
                            { ...c('local_total', 'summary'), label: '本地新闻', value: axios.localTotal, kind: 'number' as const },
                            { ...c('national_total', 'summary'), label: '全国新闻', value: axios.nationalTotal, kind: 'number' as const },
                            { ...c('recommended_brief', 'recommended_brief'), value: axios.recommendedBrief, showRaw: true },
                            { ...c('takeaway', 'takeaway'), value: axios.takeaway, showRaw: true },
                          ]}
                        />
                        {axios.topTitles.length ? <div className="empty-block">重点标题：{axios.topTitles.join(' ｜ ')}</div> : null}
                        {axios.topKeywords.length ? <div className="empty-block">关键词：{axios.topKeywords.join(' ｜ ')}</div> : null}
                      </>
                    );
                  })()}
                </DrilldownSection>
              ) : null}
              {selectedPayloadKey === 'polymarket_gamma_snapshot' ? (
                <DrilldownSection title="Polymarket 情绪侧边车摘要" summary={safeDisplayValue(payloadRecord.recommended_brief || payloadRecord.takeaway || 'research sidecar')} defaultOpen>
                  {(() => {
                    const polymarket = polymarketSnapshotSummary(payloadRecord);
                    return (
                      <>
                        <KeyValueGrid
                          rows={[
                            { ...c('markets_total', 'summary'), label: '活跃市场', value: polymarket.marketsTotal, kind: 'number' as const },
                            { ...c('bullish_count', 'summary'), label: '偏多合约', value: polymarket.bullishCount, kind: 'number' as const },
                            { ...c('bearish_count', 'summary'), label: '偏空合约', value: polymarket.bearishCount, kind: 'number' as const },
                            { ...c('yes_price_avg', 'summary'), label: '平均 Yes 概率', value: polymarket.yesPriceAvg, showRaw: true },
                            { ...c('recommended_brief', 'recommended_brief'), value: polymarket.recommendedBrief, showRaw: true },
                            { ...c('takeaway', 'takeaway'), value: polymarket.takeaway, showRaw: true },
                          ]}
                        />
                        {polymarket.topTitles.length ? <div className="empty-block">热点问题：{polymarket.topTitles.join(' ｜ ')}</div> : null}
                        {polymarket.topCategories.length ? <div className="empty-block">热门主题：{polymarket.topCategories.join(' ｜ ')}</div> : null}
                      </>
                    );
                  })()}
                </DrilldownSection>
              ) : null}
              {selectedPayloadKey === 'external_intelligence_snapshot' ? (
                <DrilldownSection title="外部情报带摘要" summary={safeDisplayValue(payloadRecord.recommended_brief || payloadRecord.takeaway || 'external intelligence')} defaultOpen>
                  {(() => {
                    const external = externalIntelligenceSummary(payloadRecord);
                    return (
                      <>
                        <KeyValueGrid
                          rows={[
                            { ...c('sources_total', 'summary'), label: '情报源', value: external.sourcesTotal, kind: 'number' as const },
                            { ...c('calendar_total', 'summary'), label: '日历事件', value: external.calendarTotal, kind: 'number' as const },
                            { ...c('high_importance_count', 'summary'), label: '高重要事件', value: external.highImportanceCount, kind: 'number' as const },
                            { ...c('flash_total', 'summary'), label: '最新快讯', value: external.flashTotal, kind: 'number' as const },
                            { ...c('news_total', 'summary'), label: '新闻条目', value: external.newsTotal, kind: 'number' as const },
                            { ...c('local_total', 'summary'), label: '本地新闻', value: external.localTotal, kind: 'number' as const },
                            { ...c('national_total', 'summary'), label: '全国新闻', value: external.nationalTotal, kind: 'number' as const },
                            { ...c('polymarket_markets_total', 'summary'), label: '预测市场', value: external.polymarketMarketsTotal, kind: 'number' as const },
                            { ...c('polymarket_yes_price_avg', 'summary'), label: '平均 Yes 概率', value: external.polymarketYesPriceAvg, showRaw: true },
                            { ...c('polymarket_bullish_count', 'summary'), label: '偏多合约', value: external.polymarketBullishCount, kind: 'number' as const },
                            { ...c('polymarket_bearish_count', 'summary'), label: '偏空合约', value: external.polymarketBearishCount, kind: 'number' as const },
                            { ...c('recommended_brief', 'recommended_brief'), value: external.recommendedBrief, showRaw: true },
                            { ...c('takeaway', 'takeaway'), value: external.takeaway, showRaw: true },
                          ]}
                        />
                        {external.activeSources.length ? <div className="empty-block">激活源：{external.activeSources.join(' ｜ ')}</div> : null}
                        {external.topTitles.length ? <div className="empty-block">重点标题：{external.topTitles.join(' ｜ ')}</div> : null}
                        {external.topKeywords.length ? <div className="empty-block">关键词：{external.topKeywords.join(' ｜ ')}</div> : null}
                        {external.polymarketTopCategories.length ? <div className="empty-block">Polymarket 主题：{external.polymarketTopCategories.join(' ｜ ')}</div> : null}
                        {external.quoteWatch.length ? <div className="empty-block">盯盘摘要：{external.quoteWatch.join(' ｜ ')}</div> : null}
                      </>
                    );
                  })()}
                </DrilldownSection>
              ) : null}
              <DrilldownSection title={t('workspace_artifacts_payload_title')} summary={`${Object.keys(payloadRecord).length} ${t('workspace_summary_payload_fields_suffix')}`} defaultOpen>
                {Object.keys(payloadRecord).length ? (
                  <KeyValueGrid
                    rows={Object.entries(payloadRecord)
                      .slice(0, 16)
                      .map(([key, value]) => ({ ...view.payloadField, key, label: labelFor(key), value }))}
                  />
                ) : (
                  <div className="empty-block">{t('workspace_artifacts_no_payload')}</div>
                )}
              </DrilldownSection>
              <DrilldownSection title={t('workspace_artifacts_raw_title')} summary={t('workspace_artifacts_final_fallback')} defaultOpen={false}>
                <JsonBlock value={payload || selectedRow} />
              </DrilldownSection>
            </>
          ) : (
            <div className="empty-block">{t('workspace_artifacts_no_matches')}</div>
          )}
        </PanelCard>
        </section>
      </div>
    </div>
  );
}

function feedbackMetricTone(metric: 'alignment' | 'blocker' | 'execution' | 'readability', value: number | undefined): string {
  const safe = typeof value === 'number' ? value : 0;
  if (metric === 'alignment') return safe >= 70 ? 'positive' : safe < 45 ? 'negative' : 'warning';
  if (metric === 'blocker') return safe >= 60 ? 'negative' : safe <= 35 ? 'positive' : 'warning';
  if (metric === 'execution') return safe >= 65 ? 'positive' : safe < 45 ? 'negative' : 'warning';
  return safe >= 60 ? 'negative' : safe <= 35 ? 'positive' : 'warning';
}

function renderFeedbackAnchor(anchor: Record<string, unknown>, index: number) {
  const route = safeDisplayValue(anchor.route);
  const artifact = safeDisplayValue(anchor.artifact);
  const component = safeDisplayValue(anchor.component);
  const text = [route !== '—' ? route : '', artifact !== '—' ? artifact : '', component !== '—' ? component : '']
    .filter(Boolean)
    .join(' / ');
  if (!text) return null;
  if (route !== '—') {
    return (
      <Link className="button" key={`${text}-${index}`} to={route}>
        {text}
      </Link>
    );
  }
  return <span className="button" key={`${text}-${index}`}>{text}</span>;
}

function AlignmentWorkspace({ model, pageSectionId }: { model: TerminalReadModel; pageSectionId?: string }) {
  const pageSections = useMemo(() => getWorkspacePageSections('alignment', model), [model]);
  const { tier } = useShellBreakpoint();
  const feedback = model.feedback.projection;
  const isInternal = model.surface.requested === 'internal' && model.surface.effective === 'internal';
  const defaultEventId = preferredFeedbackEventId(feedback);
  const [openEventId, setOpenEventId] = useState(defaultEventId);

  useEffect(() => {
    setOpenEventId(defaultEventId);
  }, [defaultEventId]);

  useEffect(() => {
    const target = findWorkspacePageSection('alignment', model, pageSectionId) || (!pageSectionId ? getDefaultWorkspacePageSection('alignment', model) : null);
    if (!target) return;
    const handle = window.requestAnimationFrame(() => {
      scrollToWorkspaceSection(target.id);
    });
    return () => window.cancelAnimationFrame(handle);
  }, [model, pageSectionId, pageSections]);

  if (!isInternal || !feedback) {
    return (
      <div className="workspace-grid single-column">
        <SectionAnchor id="alignment-summary">
          <PanelCard title="方向对齐投射" kicker="internal_only / locked" meta="仅内部可见">
            <div className="empty-block">仅内部可见</div>
          </PanelCard>
        </SectionAnchor>
      </div>
    );
  }

  const groupedEvents = feedback.events.reduce<Record<string, typeof feedback.events>>((groups, row) => {
    const domain = safeDisplayValue(row.domain || 'global');
    groups[domain] ||= [];
    groups[domain].push(row);
    return groups;
  }, {});

  return (
    <div className="workspace-grid single-column">
      {tier === 's' ? <WorkspaceQuickNav section="alignment" focus={undefined} pageSections={pageSections} activeSectionId={pageSectionId || undefined} /> : null}
      <SectionAnchor id="alignment-summary">
        <PanelCard title="方向对齐投射" kicker="内部反馈投射" meta={safeDisplayValue(feedback.summary.headline || '暂无摘要')}>
          <KeyValueGrid
            rows={[
              { ...c('alignment_score', 'alignment_score'), value: feedback.summary.alignment_score ?? '—', kind: 'number', showRaw: true },
              { ...c('blocker_pressure', 'blocker_pressure'), value: feedback.summary.blocker_pressure ?? '—', kind: 'number', showRaw: true },
              { ...c('execution_clarity', 'execution_clarity'), value: feedback.summary.execution_clarity ?? '—', kind: 'number', showRaw: true },
              { ...c('readability_pressure', 'readability_pressure'), value: feedback.summary.readability_pressure ?? '—', kind: 'number', showRaw: true },
              { ...c('drift_state', 'drift_state'), value: feedback.summary.drift_state || '—', kind: 'badge' },
              { ...c('headline', 'headline'), value: feedback.summary.headline || '—', showRaw: true },
            ]}
          />
        </PanelCard>
      </SectionAnchor>

      <SectionAnchor id="alignment-trends">
        <PanelCard title="趋势投射" kicker="最近 20 个投射点" meta={`${feedback.trends.length} ${t('unit_items')}`}>
          <div className="stack-list">
            {feedback.trends.length ? feedback.trends.map((row, index) => (
              <article className="stack-button" key={`${safeDisplayValue(row.feedback_id)}-${index}`}>
                <strong><ValueText value={safeDisplayValue(row.feedback_id)} showRaw={false} /></strong>
                <small>{safeDisplayValue(row.created_at_utc || '—')}</small>
                <div className="chip-row">
                  <Badge value={feedbackMetricTone('alignment', row.alignment_score)}>{`alignment_score：${safeDisplayValue(row.alignment_score)}`}</Badge>
                  <Badge value={feedbackMetricTone('blocker', row.blocker_pressure)}>{`blocker_pressure：${safeDisplayValue(row.blocker_pressure)}`}</Badge>
                  <Badge value={feedbackMetricTone('execution', row.execution_clarity)}>{`execution_clarity：${safeDisplayValue(row.execution_clarity)}`}</Badge>
                  <Badge value={feedbackMetricTone('readability', row.readability_pressure)}>{`readability_pressure：${safeDisplayValue(row.readability_pressure)}`}</Badge>
                </div>
              </article>
            )) : <div className="empty-block">暂无趋势点</div>}
          </div>
        </PanelCard>
      </SectionAnchor>

      <SectionAnchor id="alignment-events">
        <PanelCard title="反馈事件" kicker="按 domain 分组的 active feedback" meta={`${feedback.events.length} ${t('unit_items')}`}>
          {Object.entries(groupedEvents).length ? (
            <div className="drill-list">
              {Object.entries(groupedEvents).map(([domain, rows]) => (
                <SectionAnchor id={`alignment-domain-${safeDisplayValue(domain)}`} key={domain}>
                  <DrilldownSection title={domain} summary={`${rows.length} ${t('unit_items')}`} defaultOpen>
                    <div className="drill-list">
                      {rows.map((row) => (
                        <AccordionCard
                          id={`alignment-event-${safeDisplayValue(row.feedback_id)}`}
                          key={row.feedback_id}
                          title={safeDisplayValue(row.headline || row.feedback_id)}
                          summary={safeDisplayValue(row.summary || row.recommended_action || '—')}
                          meta={<Badge value={row.status || 'active'}>{safeDisplayValue(row.status || 'active')}</Badge>}
                          open={openEventId === safeDisplayValue(row.feedback_id)}
                          onToggle={() => setOpenEventId((current) => current === safeDisplayValue(row.feedback_id) ? '' : safeDisplayValue(row.feedback_id))}
                        >
                          <KeyValueGrid
                            rows={[
                              { ...c('summary', 'summary'), value: row.summary || '—', showRaw: true },
                              { ...c('recommended_action', 'recommended_action'), value: row.recommended_action || '—', showRaw: true },
                              { ...c('impact_score', 'impact_score'), value: row.impact_score ?? '—', kind: 'number' },
                              { ...c('confidence', 'confidence'), value: row.confidence ?? '—', kind: 'number' },
                              { ...c('source', 'source'), value: row.source || '—' },
                              { ...c('created_at_utc', 'generated_at_utc'), value: row.created_at_utc || '—' },
                            ]}
                          />
                          <div className="button-row workspace-handoff-links">
                            {(row.anchors || []).map((anchor, index) => renderFeedbackAnchor(anchor as unknown as Record<string, unknown>, index))}
                          </div>
                        </AccordionCard>
                      ))}
                    </div>
                  </DrilldownSection>
                </SectionAnchor>
              ))}
            </div>
          ) : (
            <div className="empty-block">暂无反馈事件</div>
          )}
        </PanelCard>
      </SectionAnchor>

      <SectionAnchor id="alignment-actions">
        <PanelCard title="建议动作" kicker="当前动作栈" meta={`${feedback.actions.length} ${t('unit_items')}`}>
          <div className="stack-list">
            {feedback.actions.length ? feedback.actions.map((row, index) => (
              <article className="stack-button" key={`${safeDisplayValue(row.feedback_id)}-${index}`}>
                <strong><ValueText value={safeDisplayValue(row.recommended_action || row.feedback_id)} showRaw={false} /></strong>
                <small>{`impact=${safeDisplayValue(row.impact_score)}`}</small>
                <div className="button-row workspace-handoff-links">
                  {(row.anchors || []).map((anchor, anchorIndex) => renderFeedbackAnchor(anchor as unknown as Record<string, unknown>, anchorIndex))}
                </div>
              </article>
            )) : <div className="empty-block">暂无建议动作</div>}
          </div>
        </PanelCard>
      </SectionAnchor>
    </div>
  );
}

function BacktestsWorkspace({ model, focus, pageSectionId }: { model: TerminalReadModel; focus?: WorkspaceFocus; pageSectionId?: string }) {
  const view = model.view.workspace.backtests;
  const pageSections = useMemo(() => getWorkspacePageSections('backtests', model), [model]);
  const { tier } = useShellBreakpoint();
  const backtestCount = model.labReview.backtests.length;
  const comparisonCount = model.labReview.comparisonRows.length;

  useEffect(() => {
    const target = findWorkspacePageSection('backtests', model, pageSectionId) || (!pageSectionId ? getDefaultWorkspacePageSection('backtests', model) : null);
    if (!target) return;
    const handle = window.requestAnimationFrame(() => {
      scrollToWorkspaceSection(target.id);
    });
    return () => window.cancelAnimationFrame(handle);
  }, [model, pageSectionId, pageSections]);

  return (
    <div className="workspace-grid single-column">
      {tier === 's' ? <WorkspaceQuickNav section="backtests" focus={focus} pageSections={pageSections} activeSectionId={pageSectionId || undefined} /> : null}
      <SectionAnchor id="backtests-overview">
        <PanelCard
          title={t('workspace_backtests_overview_title')}
          kicker="研究摘要 / 验证基线"
          meta="总量与验证窗口基线"
        >
          <KeyValueGrid
            rows={[
              { ...c('backtests_total', 'workspace_backtests_panel_title'), value: backtestCount, kind: 'number' as const },
              { ...c('comparison_total', 'workspace_backtests_comparison_title'), value: comparisonCount, kind: 'number' as const },
              { ...c('change_class', 'change_class'), value: model.meta.change_class || 'RESEARCH_ONLY', kind: 'badge' as const },
            ]}
          />
        </PanelCard>
      </SectionAnchor>
      <SectionAnchor id="backtests-pool">
        <PanelCard
          title={t('workspace_backtests_panel_title')}
          meta={`${backtestCount} ${t('workspace_summary_backtests_suffix')} ｜ 权益 / 收益 / 回撤主池`}
        >
          <DrilldownSection title={t('workspace_backtests_pool_title')} summary={`${backtestCount} ${t('workspace_summary_backtests_suffix')}`} defaultOpen>
            <DenseTable
              columns={view.poolColumns}
              rows={model.labReview.backtests as unknown as Array<Record<string, unknown>>}
            />
          </DrilldownSection>
        </PanelCard>
      </SectionAnchor>
      <SectionAnchor id="backtests-comparison">
        <PanelCard
          title={t('workspace_backtests_comparison_title')}
          meta={`${comparisonCount} ${t('workspace_summary_comparisons_suffix')} ｜ 近期比较 / OOS 切片`}
        >
          <DrilldownSection title={t('workspace_backtests_comparison_title')} summary={`${comparisonCount} ${t('workspace_summary_comparisons_suffix')}`} defaultOpen>
            <DenseTable columns={view.comparisonColumns} rows={model.labReview.comparisonRows} />
          </DrilldownSection>
        </PanelCard>
      </SectionAnchor>
    </div>
  );
}

function ContractsWorkspace({ model, focus, pageSectionId }: { model: TerminalReadModel; focus?: WorkspaceFocus; pageSectionId?: string }) {
  const { tier } = useShellBreakpoint();
  const compactWorkspace = tier === 's';
  const view = model.view.workspace.contracts;
  const [searchParams, setSearchParams] = useSearchParams();
  const surfaceContracts = useMemo(
    () => [...model.workspace.surfaceContracts].sort((left, right) => surfaceOrder(left.id) - surfaceOrder(right.id)),
    [model.workspace.surfaceContracts],
  );
  const publicTopology = useMemo(() => [...model.workspace.publicTopology], [model.workspace.publicTopology]);
  const publicAcceptance = model.workspace.publicAcceptance;
  const interfaceGroups = useMemo(() => buildInterfaceGroups(model.workspace.interfaces), [model.workspace.interfaces]);
  const internalSurface = model.surface.effective === 'internal';
  const pageSections = useMemo(() => getWorkspacePageSections('contracts', model), [model]);
  const defaultAcceptanceCheckId = preferredContractsAccordionId(publicAcceptance.checks, 'workspace_routes_smoke');
  const defaultAcceptanceSubcommandId = preferredContractsAccordionId(publicAcceptance.subcommands, 'workspace_routes_smoke');
  const defaultSourceHeadId = preferredContractsAccordionId(model.workspace.sourceHeads, 'operator_panel');
  const [openAccordion, setOpenAccordion] = useState({
    checks: defaultAcceptanceCheckId,
    subcommands: defaultAcceptanceSubcommandId,
    sourceHeads: defaultSourceHeadId,
  });

  useEffect(() => {
    setOpenAccordion((current) => ({
      checks: current.checks || defaultAcceptanceCheckId,
      subcommands: current.subcommands || defaultAcceptanceSubcommandId,
      sourceHeads: current.sourceHeads || defaultSourceHeadId,
    }));
  }, [defaultAcceptanceCheckId, defaultAcceptanceSubcommandId, defaultSourceHeadId]);

  useEffect(() => {
    const target = findWorkspacePageSection('contracts', model, pageSectionId) || (!pageSectionId ? getDefaultWorkspacePageSection('contracts', model) : null);
    if (!target) return;
    if (target.accordionGroup === 'contracts-acceptance-checks' && target.accordionItemId) {
      setOpenAccordion((current) => ({ ...current, checks: target.accordionItemId || current.checks }));
    }
    if (target.accordionGroup === 'contracts-acceptance-subcommands' && target.accordionItemId) {
      setOpenAccordion((current) => ({ ...current, subcommands: target.accordionItemId || current.subcommands }));
    }
    if (target.accordionGroup === 'contracts-source-heads' && target.accordionItemId) {
      setOpenAccordion((current) => ({ ...current, sourceHeads: target.accordionItemId || current.sourceHeads }));
    }
    const handle = window.requestAnimationFrame(() => {
      scrollToWorkspaceSection(target.id);
    });
    return () => window.cancelAnimationFrame(handle);
  }, [model, pageSectionId, pageSections]);

  return (
    <div className="workspace-grid single-column">
      {compactWorkspace ? <WorkspaceQuickNav section="contracts" focus={focus} pageSections={pageSections} activeSectionId={pageSectionId || undefined} /> : null}
      <SectionAnchor id="contracts-topology">
        <PanelCard title={t('workspace_contracts_topology_title')} kicker={t('workspace_contracts_topology_kicker')} meta={t('workspace_contracts_topology_meta')}>
          {publicTopology.length ? (
            <div className="stack-list">
              {TOPOLOGY_TIER_ORDER.map((tier) => {
                const rows = publicTopology.filter((row) => classifyTopologyTier(row.role) === tier);
                if (!rows.length) return null;
                return (
                  <section className={`topology-band topology-band-${tier}`} key={tier}>
                    <div className="topology-band-head">
                      <strong>{topologySectionLabel(tier)}</strong>
                      <span>{`${rows.length} ${t('workspace_summary_interfaces_suffix')}`}</span>
                    </div>
                    {shouldRenderTopologyFlow(tier) ? (
                      <div className="topology-band-flow">
                        {rows.map((row) => (
                          <div className="topology-flow-step" key={`${row.id}-flow`}>
                            <span className="topology-flow-node"><ValueText value={row.label || row.id} showRaw={false} /></span>
                            <span className="topology-flow-arrow" aria-hidden="true">→</span>
                            <span className="topology-flow-node"><ValueText value={row.fallback || t('workspace_contracts_topology_no_fallback')} showRaw={false} /></span>
                          </div>
                        ))}
                      </div>
                    ) : null}
                    <div className="topology-node-grid">
                      {rows.map((row, index) => (
                        <article className={`topology-node-card topology-node-card-${tier}`} key={row.id}>
                          <div className="topology-node-head">
                            <div className="topology-node-headcopy">
                              <span className="topology-node-index">{`${t('workspace_contracts_topology_node_prefix')} ${index + 1}`}</span>
                              <span className={`topology-node-tier topology-node-tier-${tier}`}>{topologyTierLabel(tier)}</span>
                            </div>
                            <Badge value={row.role}>{labelFor(safeDisplayValue(row.role || '—'))}</Badge>
                          </div>
                          <strong className="topology-node-title"><ValueText value={row.label || row.id} showRaw={false} /></strong>
                          <div className="topology-node-url"><PathText value={row.url || '—'} showRaw /></div>
                          <div className="topology-node-note">{safeDisplayValue(row.notes || row.source || row.fallback)}</div>
                          <KeyValueGrid
                            rows={[
                              { key: 'status', label: t('topology_expected_status'), value: row.expected_status || '—', showRaw: true },
                              { key: 'source', label: t('workspace_contracts_topology_source_label'), value: row.source || '—', showRaw: true },
                            ]}
                          />
                          <div className="topology-node-chain">
                            <span className="topology-node-chain-label">{t('workspace_contracts_topology_chain_label')}</span>
                            {' '}
                            <strong><ValueText value={row.fallback || t('workspace_contracts_topology_no_fallback')} showRaw={false} /></strong>
                          </div>
                        </article>
                      ))}
                    </div>
                  </section>
                );
              })}
            </div>
          ) : (
            <div className="empty-block">{t('workspace_contracts_topology_empty')}</div>
          )}
        </PanelCard>
      </SectionAnchor>

      <PanelCard
        title={t('workspace_contracts_acceptance_title')}
        kicker={t('workspace_contracts_acceptance_kicker')}
        meta={internalSurface ? t('workspace_contracts_acceptance_internal_meta') : t('workspace_contracts_acceptance_public_meta')}
      >
        {publicAcceptance.summary ? (
          <>
            <div
              className="contracts-acceptance-status-strip"
              data-testid="contracts-acceptance-status-strip"
              data-mobile-layout={compactWorkspace ? 'true' : 'false'}
            >
              <div className="metric-strip">
                {publicAcceptance.checks.map((row) => {
                  const rowId = safeDisplayValue(row.id);
                  const active = openAccordion.checks === rowId;
                  const subcommandTarget = publicAcceptance.subcommands.find((item) => safeDisplayValue(item.id) === rowId);
                  const explicitCheckRoute = safeDisplayValue(row.inspector_route);
                  const explicitSubcommandRoute = safeDisplayValue(subcommandTarget?.inspector_route);
                  const checkRoute = explicitCheckRoute !== '—'
                    ? explicitCheckRoute
                    : buildWorkspacePageLink('contracts', {}, `contracts-check-${rowId}`);
                  const subcommandRoute = explicitSubcommandRoute !== '—'
                    ? explicitSubcommandRoute
                    : buildWorkspacePageLink('contracts', {}, `contracts-subcommand-${rowId}`);
                  const researchAuditTarget = (row.research_audit_cases || [])[0];
                  const screenshotActionVisible = rowId === 'topology_smoke' && Boolean(
                    row.root_overview_screenshot_path
                    || row.pages_overview_screenshot_path
                    || row.root_contracts_screenshot_path
                    || row.pages_contracts_screenshot_path,
                  );
                  return (
                    <article
                      key={row.id}
                      className={`metric-tile contracts-acceptance-status-card ${toneClass(statusTone(row.status))} ${active ? 'active' : ''}`.trim()}
                    >
                      <button
                        type="button"
                        className="contracts-acceptance-status-button"
                        onClick={() => {
                          const next = new URLSearchParams(searchParams);
                          next.set('page_section', `contracts-check-${rowId}`);
                          setSearchParams(next, { replace: true });
                          setOpenAccordion((current) => ({ ...current, checks: rowId }));
                          window.requestAnimationFrame(() => {
                            scrollToWorkspaceSection(`contracts-check-${rowId}`);
                          });
                        }}
                      >
                        <span className="metric-label">{row.label}</span>
                        <strong className="metric-value">{labelFor(safeDisplayValue(row.status || '—'))}</strong>
                        <span className="metric-hint">{safeDisplayValue(row.generated_at_utc || row.headline || row.report_path)}</span>
                      </button>
                      {subcommandTarget ? (
                        <div className="button-row contracts-acceptance-status-actions">
                          {screenshotActionVisible ? (
                            <ActionLink to={checkRoute}>
                              {`${row.label} / 查看截图`}
                            </ActionLink>
                          ) : null}
                          {rowId !== 'topology_smoke' && researchAuditTarget?.search_route ? (
                            <ActionLink to={researchAuditTarget.search_route}>
                              {`${row.label} / 查看研究审计`}
                            </ActionLink>
                          ) : null}
                          <ActionLink to={subcommandRoute}>
                            {`${row.label} / 查看子命令`}
                          </ActionLink>
                        </div>
                      ) : null}
                    </article>
                  );
                })}
              </div>
            </div>
            <SectionAnchor id="contracts-acceptance-summary">
              {(() => {
                const aggregatedResearchAuditCases = aggregateResearchAuditCasesFromChecks(publicAcceptance.checks);
                const aggregatedResearchAuditQueries = aggregatedResearchAuditCases
                  .map((row) => safeDisplayValue(row.query))
                  .filter((value) => value && value !== '—')
                  .join(' ｜ ');
                const aggregatedResearchAuditArtifacts = aggregatedResearchAuditCases
                  .map((row) => safeDisplayValue(row.result_artifact))
                  .filter((value) => value && value !== '—')
                  .join(' ｜ ');
                const aggregatedResearchAuditSearchRoute = aggregatedResearchAuditCases
                  .map((row) => safeDisplayValue(row.search_route))
                  .find((value) => value && value !== '—');
                const summaryResearchAuditValues = {
                  research_audit_search_route: aggregatedResearchAuditSearchRoute,
                  research_audit_cases_available: aggregatedResearchAuditCases.length ? true : undefined,
                  research_audit_queries: aggregatedResearchAuditQueries || undefined,
                  research_audit_result_artifacts: aggregatedResearchAuditArtifacts || undefined,
                };
                return (
              <DrilldownSection
                title={t('workspace_contracts_acceptance_summary_title')}
                summary={`${safeDisplayValue(publicAcceptance.summary.status)} ｜ ${safeDisplayValue(publicAcceptance.summary.generated_at_utc)}`}
                meta={<Badge value={publicAcceptance.summary.status} />}
                defaultOpen
              >
                <KeyValueGrid
                  rows={[
                    c('status'),
                    c('change_class', 'change_class'),
                    c('generated_at_utc', 'generated_at_utc'),
                    c('report_path', 'report_path'),
                    c('artifact_path', 'path'),
                    c('topology_status', 'workspace_contracts_acceptance_topology_status'),
                    c('workspace_status', 'workspace_contracts_acceptance_workspace_status'),
                    c('graph_home_status', 'workspace_contracts_acceptance_graph_home_status'),
                    c('topology_generated_at_utc', 'workspace_contracts_acceptance_topology_generated_at'),
                    c('workspace_generated_at_utc', 'workspace_contracts_acceptance_workspace_generated_at'),
                    c('graph_home_generated_at_utc', 'workspace_contracts_acceptance_graph_home_generated_at'),
                    c('frontend_public', 'workspace_contracts_acceptance_frontend_public'),
                    c('root_public_entry', 'workspace_contracts_acceptance_root_header'),
                    c('public_snapshot_fetch_count', 'workspace_contracts_acceptance_public_fetches'),
                    c('internal_snapshot_fetch_count', 'workspace_contracts_acceptance_internal_fetches'),
                    c('orderflow_filter_route', 'workspace_contracts_acceptance_orderflow_route'),
                    c('orderflow_source_available', 'workspace_contracts_acceptance_orderflow_source_available'),
                    c('orderflow_active_artifact', 'workspace_contracts_acceptance_orderflow_active_artifact'),
                    c('orderflow_visible_artifacts', 'workspace_contracts_acceptance_orderflow_visible_artifacts'),
                    c('research_audit_search_route', 'workspace_contracts_acceptance_research_audit_route'),
                    c('research_audit_cases_available', 'workspace_contracts_acceptance_research_audit_cases_available'),
                    c('research_audit_queries', 'workspace_contracts_acceptance_research_audit_queries'),
                    c('research_audit_result_artifacts', 'workspace_contracts_acceptance_research_audit_artifacts'),
                    c('root_contracts_screenshot_path', 'workspace_contracts_acceptance_root_contracts_screenshot'),
                    c('pages_contracts_screenshot_path', 'workspace_contracts_acceptance_pages_contracts_screenshot'),
                  ].map((field) => ({
                    ...field,
                    kind: field.key.includes('path') ? 'path' : undefined,
                    showRaw: field.key.includes('path'),
                    value: (
                      field.key in summaryResearchAuditValues
                        ? summaryResearchAuditValues[field.key as keyof typeof summaryResearchAuditValues]
                        : publicAcceptance.summary?.[field.key as keyof typeof publicAcceptance.summary]
                    ) ?? '—',
                  }))}
                />
                <ArtifactHandoffs
                  links={buildResearchAuditCaseLinks(aggregatedResearchAuditCases, 'contracts-acceptance-summary-audit')}
                  title="穿透层 1.5 / 研究审计直达"
                />
              </DrilldownSection>
                );
              })()}
            </SectionAnchor>

            <SectionAnchor id="contracts-acceptance-checks">
              <DrilldownSection
                title={t('workspace_contracts_acceptance_checks_title')}
                summary={`${publicAcceptance.checks.length} ${t('workspace_summary_interfaces_suffix')}`}
                defaultOpen
              >
                <div className="drill-list">
                  {publicAcceptance.checks.map((row) => {
                    const detailRows = [
                      { ...c('status'), value: row.status || '—' },
                      { ...c('returncode', 'returncode'), value: row.returncode ?? '—', kind: 'number' as const },
                      { ...c('generated_at_utc', 'generated_at_utc'), value: row.generated_at_utc || '—' },
                      { ...c('report_path', 'report_path'), value: row.report_path || '—', kind: 'path' as const, showRaw: true },
                      { ...c('headline', 'workspace_contracts_acceptance_headline'), value: row.headline || '—' },
                      { ...c('frontend_public', 'workspace_contracts_acceptance_frontend_public'), value: row.frontend_public || '—', kind: 'path' as const, showRaw: true },
                      ...(row.tls_fallback_used === undefined
                        ? []
                        : [{ ...c('tls_fallback_used', 'workspace_contracts_acceptance_tls_fallback'), value: row.tls_fallback_used }]),
                      { ...c('root_public_entry', 'workspace_contracts_acceptance_root_header'), value: row.root_public_entry || '—' },
                      ...(row.root_overview_screenshot_path
                        ? [{ ...c('root_overview_screenshot_path', 'workspace_contracts_acceptance_root_overview_screenshot'), value: row.root_overview_screenshot_path, kind: 'path' as const, showRaw: true }]
                        : []),
                      ...(row.pages_overview_screenshot_path
                        ? [{ ...c('pages_overview_screenshot_path', 'workspace_contracts_acceptance_pages_overview_screenshot'), value: row.pages_overview_screenshot_path, kind: 'path' as const, showRaw: true }]
                        : []),
                      ...(row.root_contracts_screenshot_path
                        ? [{ ...c('root_contracts_screenshot_path', 'workspace_contracts_acceptance_root_contracts_screenshot'), value: row.root_contracts_screenshot_path, kind: 'path' as const, showRaw: true }]
                        : []),
                      ...(row.pages_contracts_screenshot_path
                        ? [{ ...c('pages_contracts_screenshot_path', 'workspace_contracts_acceptance_pages_contracts_screenshot'), value: row.pages_contracts_screenshot_path, kind: 'path' as const, showRaw: true }]
                        : []),
                      { ...c('requested_surface', 'workspace_contracts_acceptance_requested_surface'), value: row.requested_surface || '—' },
                      { ...c('effective_surface', 'workspace_contracts_acceptance_effective_surface'), value: row.effective_surface || '—' },
                      { ...c('snapshot_endpoint_observed', 'workspace_contracts_acceptance_snapshot_endpoint'), value: row.snapshot_endpoint_observed || '—', showRaw: true },
                      { ...c('public_snapshot_fetch_count', 'workspace_contracts_acceptance_public_fetches'), value: row.public_snapshot_fetch_count ?? '—', kind: 'number' as const },
                      { ...c('internal_snapshot_fetch_count', 'workspace_contracts_acceptance_internal_fetches'), value: row.internal_snapshot_fetch_count ?? '—', kind: 'number' as const },
                      { ...c('orderflow_filter_route', 'workspace_contracts_acceptance_orderflow_route'), value: row.orderflow_filter_route || '—', kind: 'path' as const, showRaw: true },
                      { ...c('orderflow_source_available', 'workspace_contracts_acceptance_orderflow_source_available'), value: row.orderflow_source_available ?? '—' },
                      { ...c('orderflow_active_artifact', 'workspace_contracts_acceptance_orderflow_active_artifact'), value: row.orderflow_active_artifact || '—' },
                      { ...c('orderflow_visible_artifacts', 'workspace_contracts_acceptance_orderflow_visible_artifacts'), value: row.orderflow_visible_artifacts || '—', showRaw: true },
                      { ...c('research_audit_search_route', 'workspace_contracts_acceptance_research_audit_route'), value: row.research_audit_search_route || '—', kind: 'path' as const, showRaw: true },
                      { ...c('research_audit_cases_available', 'workspace_contracts_acceptance_research_audit_cases_available'), value: row.research_audit_cases_available ?? '—' },
                      { ...c('research_audit_queries', 'workspace_contracts_acceptance_research_audit_queries'), value: row.research_audit_queries || '—', showRaw: true },
                      { ...c('research_audit_result_artifacts', 'workspace_contracts_acceptance_research_audit_artifacts'), value: row.research_audit_result_artifacts || '—', showRaw: true },
                      { ...c('graph_home_resolved_route', 'workspace_contracts_acceptance_graph_home_route'), value: row.graph_home_resolved_route || '—', kind: 'path' as const, showRaw: true },
                      { ...c('graph_home_default_center', 'workspace_contracts_acceptance_graph_home_center'), value: row.graph_home_default_center || '—' },
                      { ...c('graph_home_terminal_link_href', 'workspace_contracts_acceptance_graph_home_terminal_link'), value: row.graph_home_terminal_link_href || '—', kind: 'path' as const, showRaw: true },
                      { ...c('graph_home_workspace_link_href', 'workspace_contracts_acceptance_graph_home_workspace_link'), value: row.graph_home_workspace_link_href || '—', kind: 'path' as const, showRaw: true },
                      { ...c('graph_home_search_link_href', 'workspace_contracts_acceptance_graph_home_search_link'), value: row.graph_home_search_link_href || '—', kind: 'path' as const, showRaw: true },
                      { ...c('graph_home_research_audit_case_ids', 'workspace_contracts_acceptance_graph_home_cases'), value: row.graph_home_research_audit_case_ids || '—', showRaw: true },
                      { ...c('failure_reason', 'failure_reason'), value: row.failure_reason || '—' },
                    ];
                    return (
                      <SectionAnchor id={`contracts-check-${safeDisplayValue(row.id)}`} key={row.id}>
                        <AccordionCard
                          id={`contracts-check-${safeDisplayValue(row.id)}`}
                          title={row.label}
                          summary={safeDisplayValue(row.headline || row.report_path || row.generated_at_utc)}
                          meta={<Badge value={row.status} />}
                          open={openAccordion.checks === safeDisplayValue(row.id)}
                          onToggle={() => setOpenAccordion((current) => ({
                            ...current,
                            checks: current.checks === safeDisplayValue(row.id) ? '' : safeDisplayValue(row.id),
                          }))}
                        >
                          <KeyValueGrid rows={detailRows} />
                          <ArtifactHandoffs
                            links={buildResearchAuditCaseLinks(row.research_audit_cases, `contracts-check-${safeDisplayValue(row.id)}-audit`)}
                            title="研究审计直达"
                          />
                        </AccordionCard>
                      </SectionAnchor>
                    );
                  })}
                </div>
              </DrilldownSection>
            </SectionAnchor>

            <SectionAnchor id="contracts-acceptance-subcommands">
              <DrilldownSection
                title={t('workspace_contracts_acceptance_subcommands_title')}
                summary={`${publicAcceptance.subcommands.length} ${t('workspace_summary_interfaces_suffix')}`}
                defaultOpen
              >
                <div className="drill-list">
                  {publicAcceptance.subcommands.map((row) => (
                    <SectionAnchor id={`contracts-subcommand-${safeDisplayValue(row.id)}`} key={row.id}>
                      <AccordionCard
                        id={`contracts-subcommand-${safeDisplayValue(row.id)}`}
                        title={row.label}
                        summary={safeDisplayValue(row.cmd || row.cwd)}
                        meta={<Badge value={row.returncode === 0 ? 'ok' : 'failed'}>{safeDisplayValue(row.returncode)}</Badge>}
                        open={openAccordion.subcommands === safeDisplayValue(row.id)}
                        onToggle={() => setOpenAccordion((current) => ({
                          ...current,
                          subcommands: current.subcommands === safeDisplayValue(row.id) ? '' : safeDisplayValue(row.id),
                        }))}
                      >
                        <KeyValueGrid
                          rows={[
                            { ...c('returncode', 'returncode'), value: row.returncode ?? '—', kind: 'number' },
                            { ...c('payload_present', 'payload_present'), value: row.payload_present ?? '—' },
                            { ...c('stdout_bytes', 'stdout_bytes'), value: row.stdout_bytes ?? '—', kind: 'number' },
                            { ...c('stderr_bytes', 'stderr_bytes'), value: row.stderr_bytes ?? '—', kind: 'number' },
                            { ...c('cmd', 'workspace_contracts_acceptance_cmd'), value: row.cmd || '—', showRaw: true },
                            { ...c('cwd', 'workspace_contracts_acceptance_cwd'), value: row.cwd || '—', kind: 'path', showRaw: true },
                          ]}
                        />
                        {internalSurface && (row.stdout || row.stderr) ? (
                          <>
                            {row.stdout ? (
                              <DrilldownSection title={t('workspace_contracts_acceptance_stdout')} summary={`${safeDisplayValue(row.stdout_bytes)} ${t('stdout_bytes')}`} defaultOpen>
                                <pre className="json-block">{row.stdout}</pre>
                              </DrilldownSection>
                            ) : null}
                            {row.stderr ? (
                              <DrilldownSection title={t('workspace_contracts_acceptance_stderr')} summary={`${safeDisplayValue(row.stderr_bytes)} ${t('stderr_bytes')}`} defaultOpen>
                                <pre className="json-block">{row.stderr}</pre>
                              </DrilldownSection>
                            ) : null}
                          </>
                        ) : null}
                        {!internalSurface ? null : !row.stdout && !row.stderr ? <div className="empty-block">{t('workspace_contracts_acceptance_no_evidence')}</div> : null}
                      </AccordionCard>
                    </SectionAnchor>
                  ))}
                </div>
              </DrilldownSection>
            </SectionAnchor>
          </>
        ) : (
          <div className="empty-block">{t('workspace_contracts_acceptance_empty')}</div>
        )}
      </PanelCard>

      <SectionAnchor id="contracts-surfaces">
        <div className="panel-split two-up">
          {surfaceContracts.map((row) => (
            <PanelCard
              key={row.id}
              title={`${labelFor(row.label || row.id)} ${t('surface')}`}
              kicker={row.id === 'internal' ? t('workspace_contracts_surface_kicker_internal') : t('workspace_contracts_surface_kicker_public')}
              meta={row.id === 'internal' ? t('workspace_contracts_surface_meta_internal') : t('workspace_contracts_surface_meta_public')}
              className={surfacePanelClass(row.id)}
            >
              <DrilldownSection
                title={row.id === 'internal' ? t('workspace_contracts_assertion_title_internal') : t('workspace_contracts_assertion_title_public')}
                summary={`${labelFor(safeDisplayValue(row.availability))} ｜ ${labelFor(safeDisplayValue(row.redaction_level))}`}
                meta={<Badge value={row.availability as string | undefined} />}
                defaultOpen
              >
                <KeyValueGrid
                  rows={bindFieldValues(view.surfaceAssertionFields, {
                    id: row.label || row.id,
                    availability: row.availability || '—',
                    redaction_level: row.redaction_level || '—',
                    snapshot_path: row.snapshot_path || '—',
                  })}
                />
              </DrilldownSection>

              <DrilldownSection
                title={row.id === 'internal' ? t('workspace_contracts_publish_title_internal') : t('workspace_contracts_publish_title_public')}
                summary={row.id === 'internal' ? t('workspace_contracts_publish_summary_internal') : t('workspace_contracts_publish_summary_public')}
                defaultOpen
              >
                <KeyValueGrid
                  rows={bindFieldValues(view.surfacePublishFields, {
                    route_key: row.route_key || '—',
                    deploy_policy: row.deploy_policy || '—',
                    fallback_snapshot_path: row.fallback_snapshot_path || '—',
                    notes: row.notes || '—',
                  })}
                />
              </DrilldownSection>
            </PanelCard>
          ))}
        </div>
      </SectionAnchor>

      <SectionAnchor id="contracts-interface">
        <PanelCard title={t('workspace_contracts_interface_title')} kicker={t('workspace_contracts_interface_kicker')} meta={t('workspace_contracts_interface_meta')}>
          {interfaceGroups.map((group, index) => (
            <DrilldownSection
              key={group.kind}
              title={`${t('drilldown_layer_prefix')} ${index + 1} / ${group.title}`}
              summary={`${group.rows.length} ${t('workspace_summary_interfaces_suffix')} ｜ ${group.note}`}
              meta={<Badge value={group.kind}>{labelFor(group.kind)}</Badge>}
              defaultOpen={index < 2}
            >
              <DenseTable columns={view.interfaceColumns} rows={group.rows} empty={t('workspace_contracts_interface_empty')} />
            </DrilldownSection>
          ))}
        </PanelCard>
      </SectionAnchor>

      <SectionAnchor id="contracts-source-heads">
        <PanelCard title={t('workspace_contracts_source_heads_title')} kicker={t('workspace_contracts_source_heads_kicker')} meta={t('workspace_contracts_source_heads_meta')}>
          <DrilldownSection title={t('workspace_contracts_source_heads_drill_title')} summary={`${model.workspace.sourceHeads.length} ${t('workspace_contracts_source_heads_summary_suffix')}`} defaultOpen>
            <div className="drill-list">
              {model.workspace.sourceHeads.map((row) => (
                <SectionAnchor id={`contracts-source-head-${safeDisplayValue(row.id)}`} key={String(row.id)}>
                  <AccordionCard
                    id={`contracts-source-head-${safeDisplayValue(row.id)}`}
                    title={<ValueText value={safeDisplayValue(row.label || row.id)} showRaw={view.sourceHeadSummaryFields.label.showRaw} expandable={false} />}
                    summary={<ValueText value={safeDisplayValue(row.summary || row.research_decision || row.status)} showRaw={view.sourceHeadSummaryFields.summary.showRaw} expandable={false} />}
                    meta={<Badge value={row.status as string | undefined} />}
                    open={openAccordion.sourceHeads === safeDisplayValue(row.id)}
                    onToggle={() => setOpenAccordion((current) => ({
                      ...current,
                      sourceHeads: current.sourceHeads === safeDisplayValue(row.id) ? '' : safeDisplayValue(row.id),
                    }))}
                  >
                    <KeyValueGrid
                      rows={bindFieldValues(view.sourceHeadDetailFields, {
                        status: row.status || '—',
                        research_decision: row.research_decision || '—',
                        summary: row.summary || row.recommended_brief || row.takeaway || '—',
                        active_baseline: row.active_baseline || '—',
                        local_candidate: row.local_candidate || '—',
                        transfer_watch: safeDisplayValue(Reflect.get(row, 'transfer_watch')) || '—',
                        return_candidate: safeDisplayValue(Reflect.get(row, 'return_candidate')) || '—',
                        return_watch: safeDisplayValue(Reflect.get(row, 'return_watch')) || '—',
                        generated_at_utc: row.generated_at_utc || '—',
                        path: row.path || '—',
                      })}
                    />
                  </AccordionCard>
                </SectionAnchor>
              ))}
            </div>
          </DrilldownSection>
        </PanelCard>
      </SectionAnchor>

      <SectionAnchor id="contracts-fallback">
        <PanelCard title={t('workspace_contracts_fallback_title')} kicker={t('workspace_contracts_fallback_kicker')} meta={t('workspace_contracts_fallback_meta')}>
          <div className="stack-list">
            {model.workspace.fallbackChain.length ? (
              model.workspace.fallbackChain.map((item, index) => (
                <div className="empty-block" key={`${item}-${index}`}>
                  <strong>#{index + 1}</strong> {compactPath(item)}
                </div>
              ))
            ) : (
              <div className="empty-block">{t('workspace_contracts_no_fallback')}</div>
            )}
          </div>
        </PanelCard>
      </SectionAnchor>
    </div>
  );
}

function surfaceOrder(id: string | undefined): number {
  if (id === 'public') return 0;
  if (id === 'internal') return 1;
  return 9;
}

function surfacePanelClass(id: string | undefined): string {
  if (id === 'public') return 'panel-card-public';
  if (id === 'internal') return 'panel-card-internal';
  return '';
}

function buildInterfaceGroups(rows: TerminalReadModel['workspace']['interfaces']) {
  const kindOrder = ['public', 'local', 'local-static', 'local-private'];
  const notes: Record<string, string> = {
    public: t('workspace_interface_note_public'),
    local: t('workspace_interface_note_local'),
    'local-static': t('workspace_interface_note_local_static'),
    'local-private': t('workspace_interface_note_local_private'),
  };

  return kindOrder.map((kind) => ({
    kind,
    title: labelFor(kind),
    note: notes[kind] || t('workspace_interface_note_undeclared'),
    rows: rows.filter((row) => safeDisplayValue(row.kind) === kind) as unknown as Array<Record<string, unknown>>,
  })).filter((group) => group.rows.length);
}

function RawWorkspace({ model, focus, pageSectionId }: { model: TerminalReadModel; focus?: WorkspaceFocus; pageSectionId?: string }) {
  const { tier } = useShellBreakpoint();
  const focusedArtifact = model.workspace.artifactRows.find((row) => matchesArtifactFocus(row as ArtifactRowState, focus?.artifact)) || null;
  const focusedPayload = focusedArtifact?.payload_key ? model.workspace.artifactPayloads[focusedArtifact.payload_key]?.payload : null;
  const focusedPayloadRecord = focusedPayload && typeof focusedPayload === 'object' && !Array.isArray(focusedPayload)
    ? (focusedPayload as Record<string, unknown>)
    : null;
  const focusedRaw = focusedPayloadRecord && Object.keys(focusedPayloadRecord).length ? focusedPayload : focusedArtifact;
  const handoffLinks = focusedArtifact ? model.workspace.artifactHandoffs[focusedArtifact.id] || [] : [];
  const pageSections = useMemo(() => getWorkspacePageSections('raw', model), [model]);

  useEffect(() => {
    const target = findWorkspacePageSection('raw', model, pageSectionId) || (!pageSectionId ? getDefaultWorkspacePageSection('raw', model) : null);
    if (!target) return;
    const handle = window.requestAnimationFrame(() => {
      scrollToWorkspaceSection(target.id);
    });
    return () => window.cancelAnimationFrame(handle);
  }, [model, pageSectionId, pageSections]);

  return (
    <div className="workspace-grid single-column">
      {tier === 's' ? <WorkspaceQuickNav section="raw" focus={focus} pageSections={pageSections} activeSectionId={pageSectionId || undefined} /> : null}
      <SectionAnchor id="raw-summary">
        <PanelCard title={t('workspace_raw_overview_title')} kicker="研究摘要 / 原始回退" meta="只汇总当前工件键、源路径与 handoff，不复写下层 JSON。">
          <ArtifactHandoffs links={handoffLinks} title={t('workspace_raw_handoff_title')} />
          <KeyValueGrid
            rows={[
              { ...c('artifact', 'artifact'), value: focusedArtifact?.id || '—' },
              { ...c('payload_key', 'payload_key'), value: focusedArtifact?.payload_key || '—' },
              { ...c('path', 'path'), value: focusedArtifact?.path || '—', kind: 'path' as const, showRaw: true },
            ]}
          />
        </PanelCard>
      </SectionAnchor>
      <SectionAnchor id="raw-focus">
        <PanelCard title={t('workspace_raw_panel_title')} meta="聚焦当前工件的源头 JSON，不复写上层摘要。">
          {focusedRaw ? (
            <DrilldownSection title={t('workspace_raw_focus_title')} summary={labelFor(safeDisplayValue(focusedArtifact?.payload_key || focusedArtifact?.label || focusedArtifact?.id))} defaultOpen>
              <JsonBlock value={focusedRaw} />
            </DrilldownSection>
          ) : (
            <div className="empty-block">{t('workspace_artifacts_no_matches')}</div>
          )}
        </PanelCard>
      </SectionAnchor>
      <SectionAnchor id="raw-snapshot">
        <PanelCard title={t('workspace_raw_drill_title')} meta={t('workspace_raw_drill_summary')}>
          <DrilldownSection title={t('workspace_raw_drill_title')} summary={t('workspace_raw_drill_summary')} defaultOpen>
            <JsonBlock value={model.workspace.raw} />
          </DrilldownSection>
        </PanelCard>
      </SectionAnchor>
    </div>
  );
}
