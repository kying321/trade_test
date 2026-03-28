import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import type { OperatorAlertLink, SurfaceKey, TerminalReadModel, ViewDrilldownSchema } from '../types/contracts';
import { labelFor } from '../utils/dictionary';
import { safeDisplayValue, statusTone } from '../utils/formatters';
import { buildDrilldownRowId, buildTerminalFocusKey, buildTerminalLink, buildWorkspaceLink, type SharedFocusState } from '../utils/focus-links';
import { Badge, DenseTable, DrilldownList, DrilldownSection, FreshnessList, MetricStrip, PanelCard } from './ui-kit';

const t = (key: string) => labelFor(key);

type PanelId = 'orchestration' | 'data-regime' | 'signal-risk' | 'lab-review';
type SectionId =
  | 'freshness'
  | 'guards'
  | 'action-log'
  | 'checklist'
  | 'source-confidence'
  | 'micro-capture'
  | 'regime-summary'
  | 'regime-matrix'
  | 'lane-pressure'
  | 'gate-chain'
  | 'focus-slots'
  | 'control-chain'
  | 'action-stack'
  | 'research-heads'
  | 'manifests'
  | 'backtest-pool'
  | 'comparison-wall'
  | 'stress-summary';

type TerminalFocus = SharedFocusState;

const PANEL_SECTIONS: Array<{ id: PanelId; titleKey: string; sections: Array<{ id: SectionId; titleKey: string }> }> = [
  {
    id: 'orchestration',
    titleKey: 'terminal_panel_orchestration_title',
    sections: [
      { id: 'freshness', titleKey: 'terminal_section_freshness_title' },
      { id: 'guards', titleKey: 'terminal_section_guard_board_title' },
      { id: 'action-log', titleKey: 'terminal_section_action_log_title' },
      { id: 'checklist', titleKey: 'terminal_section_checklist_title' },
    ],
  },
  {
    id: 'data-regime',
    titleKey: 'terminal_panel_data_regime_title',
    sections: [
      { id: 'source-confidence', titleKey: 'terminal_section_source_confidence_title' },
      { id: 'micro-capture', titleKey: 'terminal_section_micro_capture_title' },
      { id: 'regime-summary', titleKey: 'terminal_section_regime_summary_title' },
      { id: 'regime-matrix', titleKey: 'terminal_section_regime_matrix_title' },
    ],
  },
  {
    id: 'signal-risk',
    titleKey: 'terminal_panel_signal_risk_title',
    sections: [
      { id: 'lane-pressure', titleKey: 'terminal_section_lane_pressure_title' },
      { id: 'gate-chain', titleKey: 'terminal_section_gate_chain_title' },
      { id: 'focus-slots', titleKey: 'terminal_section_focus_slots_title' },
      { id: 'control-chain', titleKey: 'terminal_section_control_chain_title' },
      { id: 'action-stack', titleKey: 'terminal_section_action_stack_title' },
    ],
  },
  {
    id: 'lab-review',
    titleKey: 'terminal_panel_lab_review_title',
    sections: [
      { id: 'research-heads', titleKey: 'terminal_section_research_heads_title' },
      { id: 'manifests', titleKey: 'terminal_section_manifests_title' },
      { id: 'backtest-pool', titleKey: 'terminal_section_backtest_pool_title' },
      { id: 'comparison-wall', titleKey: 'terminal_section_comparison_wall_title' },
      { id: 'stress-summary', titleKey: 'terminal_section_stress_summary_title' },
    ],
  },
];

function freshnessSummary(rows: TerminalReadModel['orchestration']['freshness']) {
  const stale = rows.filter((row) => row.state === 'stale').length;
  const warning = rows.filter((row) => row.state === 'warning').length;
  const fresh = rows.filter((row) => row.state === 'fresh').length;
  return `${t('fresh')} ${fresh} / ${t('warning')} ${warning} / ${t('stale')} ${stale}`;
}

function isPanelFocused(focus: TerminalFocus | undefined, panelId: PanelId) {
  return focus?.panel === panelId;
}

function isSectionFocused(focus: TerminalFocus | undefined, panelId: PanelId, sectionId: SectionId) {
  return focus?.panel === panelId && focus?.section === sectionId;
}

function focusedRowId(
  focus: TerminalFocus | undefined,
  panelId: PanelId,
  sectionId: SectionId,
) {
  if (!isSectionFocused(focus, panelId, sectionId)) return undefined;
  return focus?.row || undefined;
}

function resolveFocusedRow(
  focus: TerminalFocus | undefined,
  panelId: PanelId,
  sectionId: SectionId,
  rows: Array<Record<string, unknown>>,
  schema: ViewDrilldownSchema,
) {
  const target = focusedRowId(focus, panelId, sectionId);
  if (!target) return null;
  return rows.find((row) => buildDrilldownRowId(row, schema.rowIdentityKeys) === target) || null;
}

function focusedRowTitle(
  focus: TerminalFocus | undefined,
  panelId: PanelId,
  sectionId: SectionId,
  rows: Array<Record<string, unknown>>,
  schema: ViewDrilldownSchema,
) {
  const row = resolveFocusedRow(focus, panelId, sectionId, rows, schema);
  if (!row) return '';
  return labelFor(safeDisplayValue(row[schema.titleKey]));
}

function sectionState(
  focus: TerminalFocus | undefined,
  panelId: PanelId,
  sectionId: SectionId,
  defaultOpen: boolean,
) {
  const focused = isSectionFocused(focus, panelId, sectionId);
  return {
    sectionKey: `${panelId}-${sectionId}-${focused ? 'focus' : 'base'}`,
    className: focused ? 'drill-section-focused' : '',
    defaultOpen: focused || defaultOpen,
  };
}

function surfacePanelClass(isInternal: boolean, focused: boolean) {
  return `${isInternal ? 'panel-card-internal' : 'panel-card-public'} ${focused ? 'panel-card-focused' : ''}`.trim();
}

function normalizedArtifactRef(value: unknown): string {
  return safeDisplayValue(value).trim().toLowerCase();
}

function artifactRefMatches(row: TerminalReadModel['workspace']['artifactRows'][number], candidate: unknown): boolean {
  const target = normalizedArtifactRef(candidate);
  if (!target || target === '—') return false;
  return [row.id, row.payload_key, row.label, row.path]
    .map((value) => normalizedArtifactRef(value))
    .some((value) => value && value !== '—' && (value === target || value.includes(target) || target.includes(value)));
}

function dedupeLinks(links: OperatorAlertLink[]): OperatorAlertLink[] {
  const seen = new Set<string>();
  return links.filter((link) => {
    const signature = `${link.label}::${link.to}`;
    if (seen.has(signature)) return false;
    seen.add(signature);
    return true;
  });
}

function buildPinnedArtifactWorkspaceHandoffs(model: TerminalReadModel, focus?: TerminalFocus): OperatorAlertLink[] {
  const pinnedArtifact = safeDisplayValue(focus?.artifact);
  if (!pinnedArtifact || pinnedArtifact === '—') return [];

  const artifactRow = model.workspace.artifactRows.find((row) => artifactRefMatches(row, pinnedArtifact)) || null;
  const canonicalArtifact = safeDisplayValue(artifactRow?.id || pinnedArtifact);
  const rawArtifact = safeDisplayValue(artifactRow?.path || pinnedArtifact);
  const links: OperatorAlertLink[] = [];
  const scopedFocus = {
    panel: focus?.panel,
    section: focus?.section,
  };

  if (canonicalArtifact && canonicalArtifact !== '—') {
    links.push({
      id: `pinned-${canonicalArtifact}-artifact`,
      label: `${t('alert_link_view_artifact')} / ${labelFor(safeDisplayValue(artifactRow?.label || canonicalArtifact))}`,
      to: buildWorkspaceLink('artifacts', { artifact: canonicalArtifact, ...scopedFocus }),
    });
  }

  if (rawArtifact && rawArtifact !== '—') {
    links.push({
      id: `pinned-${canonicalArtifact}-raw`,
      label: `${t('alert_link_view_raw')} / ${labelFor(rawArtifact)}`,
      to: buildWorkspaceLink('raw', { artifact: rawArtifact, ...scopedFocus }),
    });
  }

  if (canonicalArtifact && canonicalArtifact !== '—') {
    links.push(...(model.workspace.artifactHandoffs[canonicalArtifact] || []));
  }

  return dedupeLinks(links);
}

function buildCommodityReasoningMetrics(model: TerminalReadModel) {
  const microIds = new Set(['commodity-scenario', 'commodity-chain', 'commodity-range']);
  const repairIds = new Set(['commodity-summary', 'commodity-boundary']);
  return [
    ...model.dataRegime.microCapture.filter((metric) => microIds.has(metric.id)),
    ...model.signalRisk.repairPlan.filter((metric) => repairIds.has(metric.id)),
  ];
}

function resolveTerminalTargetFocus(
  model: TerminalReadModel,
  current: TerminalFocus | undefined,
  next: SharedFocusState,
): SharedFocusState {
  const merged: SharedFocusState = { ...current, ...next };
  const hasExplicitArtifact = Boolean(safeDisplayValue(merged.artifact).trim() && safeDisplayValue(merged.artifact) !== '—');
  if (!hasExplicitArtifact) {
    const primaryArtifact = model.navigation.terminalPrimaryArtifact[buildTerminalFocusKey(merged)];
    if (primaryArtifact) merged.artifact = primaryArtifact;
  }
  return merged;
}

function TerminalFocusRail({
  model,
  surface,
  focus,
}: {
  model: TerminalReadModel;
  surface: SurfaceKey;
  focus?: TerminalFocus;
}) {
  const activePanel = PANEL_SECTIONS.find((panel) => panel.id === focus?.panel) || null;
  return (
    <div className="terminal-focus-nav" aria-label="terminal-focus-nav">
      <div className="button-row terminal-focus-rail">
        {PANEL_SECTIONS.map((panel) => (
          <Link
            key={panel.id}
            className={`chip-button terminal-focus-link ${isPanelFocused(focus, panel.id) ? 'active' : ''}`.trim()}
            to={buildTerminalLink(surface, resolveTerminalTargetFocus(model, focus, { panel: panel.id, section: undefined, row: undefined }))}
          >
            {t(panel.titleKey)}
          </Link>
        ))}
      </div>
      {activePanel ? (
        <div className="button-row terminal-section-rail">
          {activePanel.sections.map((section) => (
            <Link
              key={section.id}
              className={`chip-button terminal-focus-link ${isSectionFocused(focus, activePanel.id, section.id) ? 'active' : ''}`.trim()}
              to={buildTerminalLink(surface, resolveTerminalTargetFocus(model, focus, { panel: activePanel.id, section: section.id, row: undefined }))}
            >
              {t(section.titleKey)}
            </Link>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function TerminalBreadcrumbs({
  model,
  surface,
  focus,
  rowLabel,
}: {
  model: TerminalReadModel;
  surface: SurfaceKey;
  focus?: TerminalFocus;
  rowLabel?: string;
}) {
  const activePanel = PANEL_SECTIONS.find((panel) => panel.id === focus?.panel) || null;
  const activeSection = activePanel?.sections.find((section) => section.id === focus?.section) || null;
  if (!activePanel) return null;

  const backHref = focus?.row && activeSection
    ? buildTerminalLink(surface, resolveTerminalTargetFocus(model, focus, { panel: activePanel.id, section: activeSection.id, row: undefined }))
    : activeSection
      ? buildTerminalLink(surface, resolveTerminalTargetFocus(model, focus, { panel: activePanel.id, section: undefined, row: undefined }))
      : buildTerminalLink(surface, focus);

  return (
    <div className="terminal-breadcrumbs" aria-label="terminal-focus-breadcrumbs">
      <div className="terminal-breadcrumbs-head">
        <span className="terminal-breadcrumbs-label">{t('terminal_focus_trace_title')}</span>
        <Link className="chip-button terminal-breadcrumb-back" to={backHref}>
          {t('terminal_focus_trace_up')}
        </Link>
      </div>
      <div className="terminal-breadcrumbs-trail">
        <Link className="chip-button terminal-breadcrumb-chip" to={buildTerminalLink(surface, resolveTerminalTargetFocus(model, focus, { panel: undefined, section: undefined, row: undefined }))}>
          {t('terminal_focus_trace_root')}
        </Link>
        <span className="terminal-breadcrumb-sep">/</span>
        <Link className="chip-button terminal-breadcrumb-chip" to={buildTerminalLink(surface, resolveTerminalTargetFocus(model, focus, { panel: activePanel.id, section: undefined, row: undefined }))}>
          {t(activePanel.titleKey)}
        </Link>
        {activeSection ? (
          <>
            <span className="terminal-breadcrumb-sep">/</span>
            <Link className="chip-button terminal-breadcrumb-chip" to={buildTerminalLink(surface, resolveTerminalTargetFocus(model, focus, { panel: activePanel.id, section: activeSection.id, row: undefined }))}>
              {t(activeSection.titleKey)}
            </Link>
          </>
        ) : null}
        {focus?.row && rowLabel ? (
          <>
            <span className="terminal-breadcrumb-sep">/</span>
            <span className="terminal-breadcrumb-current">{rowLabel}</span>
          </>
        ) : null}
      </div>
    </div>
  );
}

function TerminalWorkspaceHandoffs({
  links,
}: {
  links: OperatorAlertLink[];
}) {
  if (!links.length) return null;
  return (
    <DrilldownSection title={t('terminal_handoff_title')} summary={`${links.length} ${t('terminal_handoff_summary_suffix')}`} defaultOpen>
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

function PanelSectionLinks({
  model,
  surface,
  panelId,
  focus,
}: {
  model: TerminalReadModel;
  surface: SurfaceKey;
  panelId: PanelId;
  focus?: TerminalFocus;
}) {
  const panel = PANEL_SECTIONS.find((item) => item.id === panelId);
  if (!panel) return null;
  return (
    <div className="button-row terminal-section-links">
      {panel.sections.map((section) => (
        <Link
          key={section.id}
          className={`chip-button terminal-focus-link ${isSectionFocused(focus, panelId, section.id) ? 'active' : ''}`.trim()}
          to={buildTerminalLink(surface, resolveTerminalTargetFocus(model, focus, { panel: panelId, section: section.id, row: undefined }))}
        >
          {t(section.titleKey)}
        </Link>
      ))}
    </div>
  );
}

function TerminalPanelsGrid({
  isInternal,
  children,
}: {
  isInternal: boolean;
  children: ReactNode;
}) {
  return <div className={`terminal-grid ${isInternal ? 'terminal-grid-internal' : 'terminal-grid-public'}`}>{children}</div>;
}

export function TerminalPanels({
  model,
  focus,
}: {
  model: TerminalReadModel;
  focus?: TerminalFocus;
}) {
  const { orchestration, dataRegime, signalRisk, labReview, surface } = model;
  const isInternal = surface.effective === 'internal';
  const view = model.view.terminal;
  const focusSlotsRows = signalRisk.focusSlots as unknown as Array<Record<string, unknown>>;
  const controlChainRows = signalRisk.controlChain as unknown as Array<Record<string, unknown>>;
  const actionQueueRows = signalRisk.actionQueue as unknown as Array<Record<string, unknown>>;
  const commodityReasoningMetrics = buildCommodityReasoningMetrics(model);
  const focusedRowLabel =
    focusedRowTitle(focus, 'signal-risk', 'focus-slots', focusSlotsRows, view.signalRisk.focusSlotsDrilldown)
    || focusedRowTitle(focus, 'signal-risk', 'control-chain', controlChainRows, view.signalRisk.controlChainDrilldown)
    || focusedRowTitle(focus, 'signal-risk', 'action-stack', actionQueueRows, view.signalRisk.actionQueueDrilldown);
  const pinnedArtifactHandoffs = buildPinnedArtifactWorkspaceHandoffs(model, focus);
  const focusedHandoffs = pinnedArtifactHandoffs.length
    ? pinnedArtifactHandoffs
    : model.navigation.terminalHandoffs[buildTerminalFocusKey(focus)]
    || model.navigation.terminalHandoffs[buildTerminalFocusKey({ panel: focus?.panel, section: focus?.section })]
    || [];
  return (
    <section className="terminal-rhythm-layout" aria-label="terminal-rhythm-layout">
      <section className="terminal-rhythm-band terminal-rhythm-state" aria-label="terminal-rhythm-state">
        <h3>当前状态</h3>
        <TerminalFocusRail model={model} surface={surface.effective} focus={focus} />
      </section>
      <section className="terminal-rhythm-band terminal-rhythm-focus" aria-label="terminal-rhythm-focus">
        <h3>当前焦点</h3>
        <TerminalBreadcrumbs model={model} surface={surface.effective} focus={focus} rowLabel={focusedRowLabel} />
      </section>
      <section className="terminal-rhythm-band terminal-rhythm-action" aria-label="terminal-rhythm-action">
        <h3>下一步</h3>
        <TerminalWorkspaceHandoffs links={focusedHandoffs} />
      </section>
      <section className="terminal-rhythm-band terminal-rhythm-evidence" aria-label="terminal-rhythm-evidence">
        <h3>证据</h3>
        <TerminalPanelsGrid isInternal={isInternal}>
        <div data-search-anchor="terminal-orchestration" id="terminal-orchestration">
        <PanelCard
          title={t('terminal_panel_orchestration_title')}
          kicker="运行摘要 / 调度总线"
          meta="把调度心跳、源时效、门禁灯板与人工清单收敛成当前值班入口。"
          actions={<Badge value={surface.effective}>{surface.effectiveLabel}</Badge>}
          className={surfacePanelClass(isInternal, isPanelFocused(focus, 'orchestration'))}
        >
          <PanelSectionLinks model={model} surface={surface.effective} panelId="orchestration" focus={focus} />
          <MetricStrip items={orchestration.topMetrics} showRawValues={view.orchestration.topMetricsStrip.showRawValues} />
          <DrilldownSection
            key={sectionState(focus, 'orchestration', 'freshness', true).sectionKey}
            className={sectionState(focus, 'orchestration', 'freshness', true).className}
            defaultOpen={sectionState(focus, 'orchestration', 'freshness', true).defaultOpen}
            title={t('terminal_section_freshness_title')}
            summary={freshnessSummary(orchestration.freshness)}
            meta={<Badge value={orchestration.freshness.some((row) => row.state === 'stale') ? 'warning' : 'ok'} />}
          >
            <FreshnessList rows={orchestration.freshness} />
          </DrilldownSection>
          <DrilldownSection
            key={sectionState(focus, 'orchestration', 'guards', isInternal).sectionKey}
            className={sectionState(focus, 'orchestration', 'guards', isInternal).className}
            defaultOpen={sectionState(focus, 'orchestration', 'guards', isInternal).defaultOpen}
            title={t('terminal_section_guard_board_title')}
            summary={`${orchestration.guards.length} ${t('terminal_summary_guard_items_suffix')}`}
            meta={<Badge value="healthy">{t('terminal_badge_system_level')}</Badge>}
          >
            <MetricStrip items={orchestration.guards} showRawValues={view.orchestration.guardsStrip.showRawValues} />
          </DrilldownSection>
          <DrilldownSection
            key={sectionState(focus, 'orchestration', 'action-log', isInternal).sectionKey}
            className={sectionState(focus, 'orchestration', 'action-log', isInternal).className}
            defaultOpen={sectionState(focus, 'orchestration', 'action-log', isInternal).defaultOpen}
            title={t('terminal_section_action_log_title')}
            summary={`${orchestration.actionLog.length} ${t('terminal_summary_action_log_suffix')}`}
            meta={<Badge value={orchestration.actionLog[0]?.status}>{t('terminal_badge_latest_status')}</Badge>}
          >
            <DenseTable
              columns={view.orchestration.actionLogColumns}
              rows={orchestration.actionLog as unknown as Array<Record<string, unknown>>}
            />
          </DrilldownSection>
          <DrilldownSection
            key={sectionState(focus, 'orchestration', 'checklist', isInternal).sectionKey}
            className={sectionState(focus, 'orchestration', 'checklist', isInternal).className}
            defaultOpen={sectionState(focus, 'orchestration', 'checklist', isInternal).defaultOpen}
            title={t('terminal_section_checklist_title')}
            summary={`${orchestration.checklist.length} ${t('terminal_summary_checklist_suffix')}`}
            meta={<Badge value={orchestration.checklist[0]?.status}>{t('latest_action')}</Badge>}
          >
            <DenseTable
              columns={view.orchestration.checklistColumns}
              rows={orchestration.checklist as unknown as Array<Record<string, unknown>>}
            />
          </DrilldownSection>
        </PanelCard>
        </div>

        <div data-search-anchor="terminal-data-regime" id="terminal-data-regime">
        <PanelCard
          title={t('terminal_panel_data_regime_title')}
          kicker={t('terminal_panel_data_regime_kicker')}
          meta={t('terminal_panel_data_regime_meta')}
          className={surfacePanelClass(isInternal, isPanelFocused(focus, 'data-regime'))}
        >
          <PanelSectionLinks model={model} surface={surface.effective} panelId="data-regime" focus={focus} />
          <DrilldownSection
            key={sectionState(focus, 'data-regime', 'source-confidence', true).sectionKey}
            className={sectionState(focus, 'data-regime', 'source-confidence', true).className}
            defaultOpen={sectionState(focus, 'data-regime', 'source-confidence', true).defaultOpen}
            title={t('terminal_section_source_confidence_title')}
            summary={`${dataRegime.sourceConfidence.length} ${t('terminal_summary_source_confidence_suffix')}`}
            meta={<Badge value="source_confidence">{t('terminal_badge_source_consistency')}</Badge>}
          >
            <MetricStrip items={dataRegime.sourceConfidence} showRawValues={view.dataRegime.sourceConfidenceStrip.showRawValues} />
          </DrilldownSection>
          <DrilldownSection
            key={sectionState(focus, 'data-regime', 'micro-capture', isInternal).sectionKey}
            className={sectionState(focus, 'data-regime', 'micro-capture', isInternal).className}
            defaultOpen={sectionState(focus, 'data-regime', 'micro-capture', isInternal).defaultOpen}
            title={t('terminal_section_micro_capture_title')}
            summary={`${dataRegime.microCapture.length} ${t('terminal_summary_micro_capture_suffix')}`}
            meta={<Badge value="micro_capture">{t('terminal_badge_micro_layer')}</Badge>}
          >
            <MetricStrip items={dataRegime.microCapture} showRawValues={view.dataRegime.microCaptureStrip.showRawValues} />
          </DrilldownSection>
          <DrilldownSection
            key={sectionState(focus, 'data-regime', 'regime-summary', isInternal).sectionKey}
            className={sectionState(focus, 'data-regime', 'regime-summary', isInternal).className}
            defaultOpen={sectionState(focus, 'data-regime', 'regime-summary', isInternal).defaultOpen}
            title={t('terminal_section_regime_summary_title')}
            summary={`${dataRegime.regimeSummary.length} ${t('terminal_summary_regime_modes_suffix')}`}
          >
            <DenseTable columns={view.dataRegime.regimeSummaryColumns} rows={dataRegime.regimeSummary} />
          </DrilldownSection>
          <DrilldownSection
            key={sectionState(focus, 'data-regime', 'regime-matrix', isInternal).sectionKey}
            className={sectionState(focus, 'data-regime', 'regime-matrix', isInternal).className}
            defaultOpen={sectionState(focus, 'data-regime', 'regime-matrix', isInternal).defaultOpen}
            title={t('terminal_section_regime_matrix_title')}
            summary={`${dataRegime.regimeMatrix.length} ${t('terminal_summary_regime_windows_suffix')}`}
          >
            <DenseTable columns={view.dataRegime.regimeMatrixColumns} rows={dataRegime.regimeMatrix} />
          </DrilldownSection>
        </PanelCard>
        </div>

        <div data-search-anchor="terminal-signal-risk" id="terminal-signal-risk">
        <PanelCard
          title={t('terminal_panel_signal_risk_title')}
          kicker={t('terminal_panel_signal_risk_kicker')}
          meta={t('terminal_panel_signal_risk_meta')}
          className={surfacePanelClass(isInternal, isPanelFocused(focus, 'signal-risk'))}
        >
          <PanelSectionLinks model={model} surface={surface.effective} panelId="signal-risk" focus={focus} />
          <DrilldownSection
            key={sectionState(focus, 'signal-risk', 'lane-pressure', true).sectionKey}
            className={sectionState(focus, 'signal-risk', 'lane-pressure', true).className}
            defaultOpen={sectionState(focus, 'signal-risk', 'lane-pressure', true).defaultOpen}
            title={t('terminal_section_lane_pressure_title')}
            summary={`${signalRisk.laneCards.length} ${t('terminal_summary_lane_pressure_suffix')}`}
            meta={<Badge value={signalRisk.laneCards[0]?.state}>{labelFor(signalRisk.laneCards[0]?.state)}</Badge>}
          >
            <div className="lane-grid">
              {signalRisk.laneCards.map((lane, index) => (
                <article className={`lane-card tone-${statusTone(lane.state)}`} key={`${lane.state}-${index}`}>
                  <div className="lane-topline">
                    <strong>{labelFor(lane.state)}</strong>
                    <Badge value={lane.state} />
                  </div>
                  <div className="lane-value">{safeDisplayValue(lane.count)}</div>
                  <p>{safeDisplayValue(lane.brief)}</p>
                  <span>{safeDisplayValue(lane.head_symbol)} / {safeDisplayValue(lane.head_action)}</span>
                </article>
              ))}
            </div>
          </DrilldownSection>
          <DrilldownSection
            key={sectionState(focus, 'signal-risk', 'gate-chain', isInternal).sectionKey}
            className={sectionState(focus, 'signal-risk', 'gate-chain', isInternal).className}
            defaultOpen={sectionState(focus, 'signal-risk', 'gate-chain', isInternal).defaultOpen}
            title={t('terminal_section_gate_chain_title')}
            summary={`${signalRisk.gateScores.length} ${t('terminal_summary_gate_chain_suffix')}`}
            meta={<Badge value={signalRisk.gateScores[0]?.tone}>{t('terminal_badge_gate_chain')}</Badge>}
          >
            <MetricStrip items={signalRisk.gateScores} showRawValues={view.signalRisk.gateScoresStrip.showRawValues} />
          </DrilldownSection>
          <DrilldownSection
            key={sectionState(focus, 'signal-risk', 'focus-slots', isInternal).sectionKey}
            className={sectionState(focus, 'signal-risk', 'focus-slots', isInternal).className}
            defaultOpen={sectionState(focus, 'signal-risk', 'focus-slots', isInternal).defaultOpen}
            title={t('terminal_section_focus_slots_title')}
            summary={`${signalRisk.focusSlots.length} ${t('terminal_summary_focus_slots_suffix')}`}
            meta={<Badge value={signalRisk.focusSlots[0]?.state}>{t('terminal_badge_focus_slot')}</Badge>}
          >
            <DenseTable
              columns={view.signalRisk.focusSlotsTableColumns}
              rows={signalRisk.focusSlots as unknown as Array<Record<string, unknown>>}
            />
            {isInternal ? (
              <DrilldownList
                rows={focusSlotsRows}
                focusedRowId={focusedRowId(focus, 'signal-risk', 'focus-slots')}
                rowHrefBuilder={(_, rowId) => buildTerminalLink(surface.effective, resolveTerminalTargetFocus(model, focus, { panel: 'signal-risk', section: 'focus-slots', row: rowId }))}
                {...view.signalRisk.focusSlotsDrilldown}
              />
            ) : null}
          </DrilldownSection>
          <DrilldownSection
            key={sectionState(focus, 'signal-risk', 'control-chain', isInternal).sectionKey}
            className={sectionState(focus, 'signal-risk', 'control-chain', isInternal).className}
            defaultOpen={sectionState(focus, 'signal-risk', 'control-chain', isInternal).defaultOpen}
            title={t('terminal_section_control_chain_title')}
            summary={`${signalRisk.controlChain.length} ${t('terminal_summary_control_chain_suffix')}`}
            meta={<Badge value={signalRisk.controlChain[0]?.source_status}>{t('terminal_badge_control_chain')}</Badge>}
          >
            <DenseTable
              columns={view.signalRisk.controlChainTableColumns}
              rows={signalRisk.controlChain as unknown as Array<Record<string, unknown>>}
            />
            {isInternal ? (
              <DrilldownList
                rows={controlChainRows}
                focusedRowId={focusedRowId(focus, 'signal-risk', 'control-chain')}
                rowHrefBuilder={(_, rowId) => buildTerminalLink(surface.effective, resolveTerminalTargetFocus(model, focus, { panel: 'signal-risk', section: 'control-chain', row: rowId }))}
                {...view.signalRisk.controlChainDrilldown}
              />
            ) : null}
          </DrilldownSection>
          <DrilldownSection
            key={sectionState(focus, 'signal-risk', 'action-stack', isInternal).sectionKey}
            className={sectionState(focus, 'signal-risk', 'action-stack', isInternal).className}
            defaultOpen={sectionState(focus, 'signal-risk', 'action-stack', isInternal).defaultOpen}
            title={t('terminal_section_action_stack_title')}
            summary={`${signalRisk.actionQueue.length} ${t('terminal_summary_action_queue_suffix')} / ${signalRisk.optimizationBacklog.length} ${t('terminal_summary_backlog_suffix')}`}
            meta={<Badge value={signalRisk.repairPlan[0]?.value}>{t('terminal_badge_repair_priority')}</Badge>}
          >
            <div className="panel-split two-up">
              <div>
                <DenseTable
                  columns={view.signalRisk.actionQueueTableColumns}
                  rows={signalRisk.actionQueue as unknown as Array<Record<string, unknown>>}
                />
                {isInternal ? (
                  <DrilldownList
                    rows={actionQueueRows}
                    focusedRowId={focusedRowId(focus, 'signal-risk', 'action-stack')}
                    rowHrefBuilder={(_, rowId) => buildTerminalLink(surface.effective, resolveTerminalTargetFocus(model, focus, { panel: 'signal-risk', section: 'action-stack', row: rowId }))}
                    {...view.signalRisk.actionQueueDrilldown}
                  />
                ) : null}
              </div>
              <div>
                <MetricStrip items={signalRisk.repairPlan} showRawValues={view.signalRisk.repairPlanStrip.showRawValues} />
                <DenseTable
                  columns={view.signalRisk.optimizationBacklogColumns}
                  rows={signalRisk.optimizationBacklog}
                />
              </div>
            </div>
          </DrilldownSection>
        </PanelCard>
        </div>

        {commodityReasoningMetrics.length ? (
          <div data-search-anchor="terminal-commodity-reasoning" id="terminal-commodity-reasoning">
            <PanelCard
              title="国内商品推理线"
              kicker="reasoning / visible-priority"
              meta="终端固定摘要：主情景 / 主传导链 / 范围 / 边界 / 失效条件"
              className={surfacePanelClass(isInternal, false)}
            >
              <MetricStrip items={commodityReasoningMetrics} showRawValues />
            </PanelCard>
          </div>
        ) : null}

        <div data-search-anchor="terminal-lab-review" id="terminal-lab-review">
        <PanelCard
          title={t('terminal_panel_lab_review_title')}
          kicker={t('terminal_panel_lab_review_kicker')}
          meta={t('terminal_panel_lab_review_meta')}
          className={surfacePanelClass(isInternal, isPanelFocused(focus, 'lab-review'))}
        >
          <PanelSectionLinks model={model} surface={surface.effective} panelId="lab-review" focus={focus} />
          <DrilldownSection
            key={sectionState(focus, 'lab-review', 'research-heads', true).sectionKey}
            className={sectionState(focus, 'lab-review', 'research-heads', true).className}
            defaultOpen={sectionState(focus, 'lab-review', 'research-heads', true).defaultOpen}
            title={t('terminal_section_research_heads_title')}
            summary={`${labReview.researchHeads.length} ${t('terminal_summary_research_heads_suffix')}`}
          >
            <MetricStrip items={labReview.researchHeads} showRawValues={view.labReview.researchHeadsStrip.showRawValues} />
          </DrilldownSection>
          <DrilldownSection
            key={sectionState(focus, 'lab-review', 'manifests', isInternal).sectionKey}
            className={sectionState(focus, 'lab-review', 'manifests', isInternal).className}
            defaultOpen={sectionState(focus, 'lab-review', 'manifests', isInternal).defaultOpen}
            title={t('terminal_section_manifests_title')}
            summary={`${labReview.manifests.length} ${t('terminal_summary_manifests_suffix')}`}
          >
            <MetricStrip items={labReview.manifests} showRawValues={view.labReview.manifestsStrip.showRawValues} />
          </DrilldownSection>
          <DrilldownSection
            key={sectionState(focus, 'lab-review', 'backtest-pool', isInternal).sectionKey}
            className={sectionState(focus, 'lab-review', 'backtest-pool', isInternal).className}
            defaultOpen={sectionState(focus, 'lab-review', 'backtest-pool', isInternal).defaultOpen}
            title={t('terminal_section_backtest_pool_title')}
            summary={`${labReview.backtests.length} ${t('terminal_summary_backtests_suffix')}`}
          >
            <DenseTable
              columns={view.labReview.backtestPoolColumns}
              rows={labReview.backtests as unknown as Array<Record<string, unknown>>}
            />
          </DrilldownSection>
          <DrilldownSection
            key={sectionState(focus, 'lab-review', 'comparison-wall', isInternal).sectionKey}
            className={sectionState(focus, 'lab-review', 'comparison-wall', isInternal).className}
            defaultOpen={sectionState(focus, 'lab-review', 'comparison-wall', isInternal).defaultOpen}
            title={t('terminal_section_comparison_wall_title')}
            summary={`${labReview.comparisonRows.length} ${t('terminal_summary_comparison_rows_suffix')} / ${labReview.candidateRows.length} ${t('terminal_summary_candidate_rows_suffix')}`}
          >
            <div className="panel-split two-up">
              <DenseTable columns={view.labReview.comparisonColumns} rows={labReview.comparisonRows} />
              <DenseTable columns={view.labReview.candidateColumns} rows={labReview.candidateRows as unknown as Array<Record<string, unknown>>} />
            </div>
          </DrilldownSection>
          <DrilldownSection
            key={sectionState(focus, 'lab-review', 'stress-summary', isInternal).sectionKey}
            className={sectionState(focus, 'lab-review', 'stress-summary', isInternal).className}
            defaultOpen={sectionState(focus, 'lab-review', 'stress-summary', isInternal).defaultOpen}
            title={t('terminal_section_stress_summary_title')}
            summary={`${labReview.stressSummary.length} ${t('terminal_summary_stress_profiles_suffix')}`}
          >
            <DenseTable columns={view.labReview.stressSummaryColumns} rows={labReview.stressSummary} />
          </DrilldownSection>
        </PanelCard>
        </div>
        </TerminalPanelsGrid>
      </section>
    </section>
  );
}
