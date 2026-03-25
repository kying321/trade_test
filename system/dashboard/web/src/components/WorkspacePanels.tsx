import type { ReactNode } from 'react';
import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { findWorkspacePageSection, getWorkspacePageSections } from '../navigation/workspace-sections';
import type { OperatorAlertLink, TerminalReadModel, ViewFieldSchema } from '../types/contracts';
import { labelFor } from '../utils/dictionary';
import { buildTerminalLink, buildWorkspaceLink, type SharedFocusState } from '../utils/focus-links';
import { compactPath, safeDisplayValue, statusTone } from '../utils/formatters';
import { AccordionCard, Badge, DenseTable, DrilldownSection, JsonBlock, KeyValueGrid, PanelCard, PathText, ValueText } from './ui-kit';

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
    <button
      key={id}
      className={`stack-button artifact-button artifact-button-${layer} ${selected ? 'active' : ''}`.trim()}
      onClick={() => onSelect(id)}
    >
      <strong>
        <ValueText
          value={titleFor(row)}
          showRaw={showRawTitle}
          expandable={false}
        />
      </strong>
      <span>{labelFor(row.category as string | undefined)}</span>
      <Badge value={row.research_decision || row.status} />
      <PathText value={row.path} showRaw={showRawPath} expandable={false} />
    </button>
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
  return <ArtifactsWorkspace model={model} focus={focus} />;
}

function ArtifactsWorkspace({ model, focus }: { model: TerminalReadModel; focus?: WorkspaceFocus }) {
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
    if (filteredRows.length) return safeDisplayValue(filteredRows[0]?.id);
    if (focusedId && focusedRowGroup === activeGroup) return focusedId;
    const primary = primaryRows[activeGroup];
    return primary ? safeDisplayValue(primary.id) : '';
  }, [focusedId, filteredRows, focusedRowGroup, activeGroup, primaryRows]);

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
  const focusLinks = handoffLinks.length ? handoffLinks : fallbackFocusLinks;

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
    <div className="workspace-grid artifacts-grid">
      <div className="workspace-left-rail">
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

        <PanelCard title={t('workspace_artifacts_focus_title')} kicker={t('workspace_artifacts_focus_kicker')} meta={t('workspace_artifacts_focus_meta')}>
          {selectedRow ? (
            <>
              <KeyValueGrid
                rows={[
                  { ...c('artifact_title', 'workspace_artifacts_focus_field_title'), value: artifactTitle(selectedRow) },
                  { ...c('artifact_group', 'workspace_artifacts_focus_field_group'), value: artifactGroupLabel(activeGroup) },
                  { ...c('category', 'category'), value: selectedRow.category || '—' },
                  { ...c('status', 'status'), value: selectedRow.research_decision || selectedRow.status || '—', kind: 'badge' as const },
                  { ...c('path', 'path'), value: selectedArtifactPath, kind: 'path' as const, showRaw: true },
                  ...(selectedPayloadKey && selectedPayloadKey !== selectedRowId
                    ? [{ ...c('payload_key', 'payload_key'), value: selectedPayloadKey }]
                    : []),
                ]}
              />
              <div className="button-row workspace-handoff-links">
                {focusLinks.map((link) => (
                  <Link key={link.id} className={`button ${link.tone === 'negative' ? 'button-primary' : ''}`.trim()} to={link.to}>
                    {link.label}
                  </Link>
                ))}
              </div>
            </>
          ) : (
            <div className="empty-block">{t('workspace_artifacts_focus_empty')}</div>
          )}
        </PanelCard>
      </div>

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
  const feedback = model.feedback.projection;
  const isInternal = model.surface.requested === 'internal' && model.surface.effective === 'internal';
  const [openEventId, setOpenEventId] = useState(safeDisplayValue(feedback?.events[0]?.feedback_id));

  useEffect(() => {
    setOpenEventId(safeDisplayValue(feedback?.events[0]?.feedback_id));
  }, [feedback?.events]);

  useEffect(() => {
    const target = findWorkspacePageSection('alignment', model, pageSectionId) || (!pageSectionId ? pageSections[0] : null);
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
  const backtestCount = model.labReview.backtests.length;
  const comparisonCount = model.labReview.comparisonRows.length;

  useEffect(() => {
    const target = findWorkspacePageSection('backtests', model, pageSectionId) || (!pageSectionId ? pageSections[0] : null);
    if (!target) return;
    const handle = window.requestAnimationFrame(() => {
      scrollToWorkspaceSection(target.id);
    });
    return () => window.cancelAnimationFrame(handle);
  }, [model, pageSectionId, pageSections]);

  return (
    <div className="workspace-grid single-column">
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
  const view = model.view.workspace.contracts;
  const surfaceContracts = useMemo(
    () => [...model.workspace.surfaceContracts].sort((left, right) => surfaceOrder(left.id) - surfaceOrder(right.id)),
    [model.workspace.surfaceContracts],
  );
  const publicTopology = useMemo(() => [...model.workspace.publicTopology], [model.workspace.publicTopology]);
  const publicAcceptance = model.workspace.publicAcceptance;
  const interfaceGroups = useMemo(() => buildInterfaceGroups(model.workspace.interfaces), [model.workspace.interfaces]);
  const internalSurface = model.surface.effective === 'internal';
  const pageSections = useMemo(() => getWorkspacePageSections('contracts', model), [model]);
  const [openAccordion, setOpenAccordion] = useState({
    checks: safeDisplayValue(publicAcceptance.checks[0]?.id),
    subcommands: safeDisplayValue(publicAcceptance.subcommands[0]?.id),
    sourceHeads: safeDisplayValue(model.workspace.sourceHeads[0]?.id),
  });

  useEffect(() => {
    setOpenAccordion((current) => ({
      checks: current.checks || safeDisplayValue(publicAcceptance.checks[0]?.id),
      subcommands: current.subcommands || safeDisplayValue(publicAcceptance.subcommands[0]?.id),
      sourceHeads: current.sourceHeads || safeDisplayValue(model.workspace.sourceHeads[0]?.id),
    }));
  }, [model.workspace.sourceHeads, publicAcceptance.checks, publicAcceptance.subcommands]);

  useEffect(() => {
    const target = findWorkspacePageSection('contracts', model, pageSectionId) || (!pageSectionId ? pageSections[0] : null);
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
            <SectionAnchor id="contracts-acceptance-summary">
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
                    c('topology_generated_at_utc', 'workspace_contracts_acceptance_topology_generated_at'),
                    c('workspace_generated_at_utc', 'workspace_contracts_acceptance_workspace_generated_at'),
                    c('frontend_public', 'workspace_contracts_acceptance_frontend_public'),
                    c('root_public_entry', 'workspace_contracts_acceptance_root_header'),
                    c('public_snapshot_fetch_count', 'workspace_contracts_acceptance_public_fetches'),
                    c('internal_snapshot_fetch_count', 'workspace_contracts_acceptance_internal_fetches'),
                    c('orderflow_filter_route', 'workspace_contracts_acceptance_orderflow_route'),
                    c('orderflow_active_artifact', 'workspace_contracts_acceptance_orderflow_active_artifact'),
                    c('orderflow_visible_artifacts', 'workspace_contracts_acceptance_orderflow_visible_artifacts'),
                    c('root_contracts_screenshot_path', 'workspace_contracts_acceptance_root_contracts_screenshot'),
                    c('pages_contracts_screenshot_path', 'workspace_contracts_acceptance_pages_contracts_screenshot'),
                  ].map((field) => ({
                    ...field,
                    kind: field.key.includes('path') ? 'path' : undefined,
                    showRaw: field.key.includes('path'),
                    value: publicAcceptance.summary?.[field.key as keyof typeof publicAcceptance.summary] ?? '—',
                  }))}
                />
              </DrilldownSection>
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
                      { ...c('orderflow_active_artifact', 'workspace_contracts_acceptance_orderflow_active_artifact'), value: row.orderflow_active_artifact || '—' },
                      { ...c('orderflow_visible_artifacts', 'workspace_contracts_acceptance_orderflow_visible_artifacts'), value: row.orderflow_visible_artifacts || '—', showRaw: true },
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
  const focusedArtifact = model.workspace.artifactRows.find((row) => matchesArtifactFocus(row as ArtifactRowState, focus?.artifact)) || null;
  const focusedPayload = focusedArtifact?.payload_key ? model.workspace.artifactPayloads[focusedArtifact.payload_key]?.payload : null;
  const focusedPayloadRecord = focusedPayload && typeof focusedPayload === 'object' && !Array.isArray(focusedPayload)
    ? (focusedPayload as Record<string, unknown>)
    : null;
  const focusedRaw = focusedPayloadRecord && Object.keys(focusedPayloadRecord).length ? focusedPayload : focusedArtifact;
  const handoffLinks = focusedArtifact ? model.workspace.artifactHandoffs[focusedArtifact.id] || [] : [];
  const pageSections = useMemo(() => getWorkspacePageSections('raw', model), [model]);

  useEffect(() => {
    const target = findWorkspacePageSection('raw', model, pageSectionId) || (!pageSectionId ? pageSections[0] : null);
    if (!target) return;
    const handle = window.requestAnimationFrame(() => {
      scrollToWorkspaceSection(target.id);
    });
    return () => window.cancelAnimationFrame(handle);
  }, [model, pageSectionId, pageSections]);

  return (
    <div className="workspace-grid single-column">
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
