import type {
  ArtifactPayloadEntry,
  ConversationFeedbackAction,
  ConversationFeedbackAnchor,
  ConversationFeedbackDomainGroup,
  ConversationFeedbackEvent,
  ConversationFeedbackProjection,
  ConversationFeedbackTrendPoint,
  DashboardSnapshot,
  FreshnessRow,
  LoadedSurface,
  MetricItem,
  OperatorAlert,
  OperatorAlertField,
  OperatorAlertLink,
  OperatorAlertSection,
  PublicAcceptanceCheckRow,
  PublicAcceptanceSubcommandRow,
  PublicAcceptanceSummary,
  ResearchCandidateRow,
  SurfaceContract,
  SurfaceView,
  TerminalReadModel,
  TerminalViewSchema,
  ViewColumnSchema,
  ViewDrilldownSchema,
  ViewFieldSchema,
} from '../types/contracts';
import { resolveDisplayIntent } from '../utils/display-policy';
import { ageMinutes, freshnessState, safeDisplayValue, statusTone, toArray, toRecord } from '../utils/formatters';
import { buildDrilldownRowId, buildTerminalFocusKey, buildTerminalLink, buildWorkspaceLink } from '../utils/focus-links';
import { labelFor, surfaceLabel } from '../utils/dictionary';

type Dict = Record<string, unknown>;
type DisplayIntent = ReturnType<typeof resolveDisplayIntent>;
const t = (key: string) => labelFor(key);

function artifactEntry(snapshot: DashboardSnapshot, id: string): ArtifactPayloadEntry | null {
  return snapshot.artifact_payloads?.[id] || null;
}

function artifactPayload(snapshot: DashboardSnapshot, id: string): Dict {
  return toRecord<Dict>(artifactEntry(snapshot, id)?.payload) || {};
}

function eventArtifact(snapshot: DashboardSnapshot, id: string): Dict {
  return toRecord<Dict>(artifactPayload(snapshot, id)) || {};
}

function normalizedString(value: unknown): string | undefined {
  const text = safeDisplayValue(value);
  if (!text || text === '—') return undefined;
  return text;
}

function nested(source: unknown, path: string): unknown {
  let current: unknown = source;
  for (const key of path.split('.')) {
    if (!current || typeof current !== 'object' || Array.isArray(current)) return undefined;
    current = (current as Dict)[key];
  }
  return current;
}

function loadedArtifact(snapshot: DashboardSnapshot, id: string): string | undefined {
  return artifactEntry(snapshot, id)?.path;
}

function displayValue(value: unknown): string {
  return labelFor(safeDisplayValue(value));
}

function buildAssignmentHint(labelKey: string, value: unknown): string {
  return `${t(labelKey)}=${displayValue(value)}`;
}

function joinAssignmentHints(entries: Array<[string, unknown]>, delimiter = ' / '): string {
  return entries.map(([labelKey, value]) => buildAssignmentHint(labelKey, value)).join(delimiter);
}

function toMetric(id: string, labelKey: string, value: unknown, hint?: string, tone?: string): MetricItem {
  return { id, label: t(labelKey), value, hint, tone: tone ? statusTone(tone) : statusTone(value) };
}

function alertField(
  key: string,
  labelKey: string,
  value: unknown,
  options: Partial<Pick<OperatorAlertField, 'kind' | 'showRaw'>> = {},
): OperatorAlertField {
  return {
    key,
    label: t(labelKey),
    value,
    ...options,
  };
}

function alertSection(
  id: string,
  titleKey: string,
  rows: OperatorAlertField[],
  options: Partial<Pick<OperatorAlertSection, 'summary' | 'tone' | 'defaultOpen'>> = {},
): OperatorAlertSection {
  return {
    id,
    title: t(titleKey),
    rows,
    ...options,
  };
}

function resolvedArtifactPath(snapshot: DashboardSnapshot, value: unknown, fallback?: unknown): string | undefined {
  for (const candidate of [value, fallback]) {
    const raw = safeDisplayValue(candidate);
    if (!raw || raw === '—') continue;
    if (raw.includes('/') || raw.includes('\\')) return raw;
    const loaded = loadedArtifact(snapshot, raw);
    if (loaded) return loaded;
  }
  return undefined;
}

function alertSummary(alert: Partial<Pick<OperatorAlert, 'summary' | 'detail'>>): string | undefined {
  return [alert.summary, alert.detail].filter((value) => value && value !== '—').join(' ｜ ') || undefined;
}

function alertLink(id: string, labelKey: string, to: string, tone?: OperatorAlertLink['tone']): OperatorAlertLink {
  return {
    id,
    label: t(labelKey),
    to,
    tone,
  };
}

function artifactAlertLinks(options: {
  idPrefix: string;
  artifact?: unknown;
  rawArtifact?: unknown;
  includeContracts?: boolean;
  terminalFocus?: { surface: SurfaceView['effective']; panel?: string; section?: string; row?: string };
  tone?: OperatorAlertLink['tone'];
}): OperatorAlertLink[] {
  const artifact = safeDisplayValue(options.artifact);
  const rawArtifact = safeDisplayValue(options.rawArtifact || options.artifact);
  const links: OperatorAlertLink[] = [];
  if (artifact !== '—') {
    links.push(alertLink(`${options.idPrefix}-artifact`, 'alert_link_view_artifact', buildWorkspaceLink('artifacts', { artifact }), options.tone));
  }
  if (rawArtifact !== '—') {
    links.push(alertLink(`${options.idPrefix}-raw`, 'alert_link_view_raw', buildWorkspaceLink('raw', { artifact: rawArtifact }), options.tone));
  }
  if (options.terminalFocus?.panel) {
    links.push(
      alertLink(
        `${options.idPrefix}-terminal`,
        'alert_link_view_terminal',
        buildTerminalLink(options.terminalFocus.surface, {
          panel: options.terminalFocus.panel,
          section: options.terminalFocus.section,
          row: options.terminalFocus.row,
        }),
        options.tone,
      ),
    );
  }
  if (options.includeContracts) {
    links.push(alertLink(`${options.idPrefix}-contracts`, 'alert_link_view_contracts', buildWorkspaceLink('contracts'), options.tone));
  }
  return links;
}

function buildFreshnessRow(id: string, labelKey: string, timestamp: string | undefined, artifact?: string, staleAfter = 240, warnAfter = 60): FreshnessRow {
  const minutes = ageMinutes(timestamp);
  return {
    id,
    label: t(labelKey),
    timestamp,
    ageMinutes: minutes,
    state: freshnessState(minutes, staleAfter, warnAfter),
    artifact,
  };
}

function buildSurfaceView(snapshot: DashboardSnapshot, loaded: LoadedSurface): SurfaceView {
  const surfaceContracts = snapshot.surface_contracts || {};
  const requestedContract: SurfaceContract = surfaceContracts[loaded.requestedSurface] || {};
  const effectiveContract: SurfaceContract = surfaceContracts[loaded.effectiveSurface] || {};
  return {
    requested: loaded.requestedSurface,
    effective: loaded.effectiveSurface,
    snapshotPath: loaded.snapshotPath,
    warnings: loaded.warnings,
    requestedLabel: requestedContract.label || surfaceLabel(loaded.requestedSurface),
    effectiveLabel: effectiveContract.label || surfaceLabel(loaded.effectiveSurface),
    redactionLevel: effectiveContract.redaction_level || 'public_summary',
    availability: effectiveContract.availability || 'active',
  };
}

function buildOrchestration(snapshot: DashboardSnapshot, surface: SurfaceView) {
  const operatorPanel = artifactPayload(snapshot, 'operator_panel');
  const scheduler = artifactPayload(snapshot, 'system_scheduler_state');
  const architecture = artifactPayload(snapshot, 'architecture_audit');
  const experience = toRecord<Dict>(snapshot.experience_contract) || {};
  const summary = toRecord<Dict>(operatorPanel.summary) || {};
  const intervalTask = toRecord<Dict>(nested(scheduler, 'interval_tasks.micro_capture')) || {};
  const latestHistory = toRecord<Dict>(toArray<Dict>(scheduler.history)[0]) || {};
  const latestResult = toRecord<Dict>(latestHistory.result) || {};
  const checks = toRecord<Dict>(nested(architecture, 'health.checks')) || {};
  const manifests = toRecord<Dict>(architecture.manifests) || {};

  const topMetrics: MetricItem[] = [
    toMetric('surface', 'orchestration_surface_metric', surface.effectiveLabel, buildAssignmentHint('requested', surface.requestedLabel)),
    toMetric('contract-mode', 'orchestration_contract_mode_metric', experience.mode, t('experience_contract'), String(experience.mode || 'unknown')),
    toMetric('change-class', 'orchestration_change_class_metric', snapshot.meta?.change_class || snapshot.change_class, t('change_class'), String(snapshot.meta?.change_class || snapshot.change_class || 'unknown')),
    toMetric('operator-status', 'orchestration_operator_status_metric', operatorPanel.status, displayValue(summary.operator_head_brief)),
    toMetric('scheduler-status', 'orchestration_scheduler_status_metric', intervalTask.status, displayValue(intervalTask.last_run_ts)),
    toMetric('live-path', 'orchestration_live_path_metric', String(experience.live_path_touched), t('strict_read_only'), String(experience.live_path_touched) === 'true' ? 'negative' : 'positive'),
  ];

  const freshness: FreshnessRow[] = [
    buildFreshnessRow('snapshot-meta', 'freshness_snapshot_generated', snapshot.meta?.generated_at_utc, loadedArtifact(snapshot, 'price_action_breakout_pullback')),
    buildFreshnessRow('operator-panel', 'freshness_operator_panel', typeof operatorPanel.generated_at_utc === 'string' ? operatorPanel.generated_at_utc : undefined, loadedArtifact(snapshot, 'operator_panel')),
    buildFreshnessRow('scheduler-run', 'freshness_scheduler_run', typeof intervalTask.last_run_ts === 'string' ? intervalTask.last_run_ts : undefined, loadedArtifact(snapshot, 'system_scheduler_state'), 120, 30),
    buildFreshnessRow('scheduler-capture', 'freshness_scheduler_capture', typeof latestResult.captured_at_utc === 'string' ? latestResult.captured_at_utc : undefined, loadedArtifact(snapshot, 'system_scheduler_state'), 120, 30),
    buildFreshnessRow('architecture-audit', 'guard_architecture_status', artifactEntry(snapshot, 'architecture_audit')?.summary?.generated_at_utc, loadedArtifact(snapshot, 'architecture_audit'), 1440, 360),
  ];

  const controlChainRows = toArray<Dict>(operatorPanel.control_chain);
  const schedulerHistory = toArray<Dict>(scheduler.history);
  const actionLog = schedulerHistory.slice(0, 6).map((row, index) => ({
    id: `scheduler-${index}`,
    ts: typeof row.ts === 'string' ? row.ts : undefined,
    label: `${safeDisplayValue(row.slot)} / ${safeDisplayValue(row.trigger_time)}`,
    status: safeDisplayValue(row.status),
    detail: safeDisplayValue(toRecord<Dict>(row.result)?.summary || row.result),
    tone: statusTone(row.status),
  }));

  const checklist = toArray<Dict>(operatorPanel.action_checklist)
    .slice(0, surface.effective === 'internal' ? 6 : 4)
    .map((row, index) => ({
      id: `checklist-${index}`,
      ts: typeof row.source_as_of === 'string' ? row.source_as_of : undefined,
      label: `${safeDisplayValue(row.symbol)} / ${safeDisplayValue(row.action)}`,
      status: safeDisplayValue(row.state || row.source_status),
      detail: joinAssignmentHints([
        ['reason', row.reason],
        ['done_when', row.done_when],
      ], ' ｜ '),
      tone: statusTone(row.state || row.source_status),
    }));

  const staleRows = freshness.filter((row) => row.state === 'stale');
  const warningRows = freshness.filter((row) => row.state === 'warning');
  const livePathTouched = String(experience.live_path_touched) === 'true';
  const showRawPaths = surface.effective === 'internal';
  const operatorPanelPath = loadedArtifact(snapshot, 'operator_panel');
  const schedulerPath = loadedArtifact(snapshot, 'system_scheduler_state');
  const snapshotPath = surface.snapshotPath;
  const guardianStatus = safeDisplayValue(summary.openclaw_guardian_clearance_status);
  const guardianTone = statusTone(guardianStatus);
  const guardianBlocked = guardianStatus !== '—' && guardianTone !== 'positive';
  const guardianTargetArtifact = safeDisplayValue(summary.openclaw_guardian_clearance_top_blocker_target_artifact);
  const guardianTargetPath = resolvedArtifactPath(snapshot, guardianTargetArtifact);
  const guardianSummary = [
    summary.openclaw_guardian_clearance_score !== undefined ? buildAssignmentHint('score', summary.openclaw_guardian_clearance_score) : '',
    displayValue(summary.openclaw_guardian_clearance_top_blocker_title || summary.remote_live_gate_brief),
  ].filter(Boolean).join(' ｜ ');
  const guardianDetail = [
    displayValue(summary.openclaw_guardian_clearance_top_blocker_next_action),
    displayValue(summary.openclaw_guardian_clearance_top_blocker_target_artifact),
    displayValue(summary.remote_live_clearing_brief),
  ].filter((value) => value && value !== '—').join(' ｜ ');
  const controlChainBlocker = controlChainRows.find((row) => {
    const blockingReason = safeDisplayValue(row.blocking_reason);
    const tone = statusTone(row.source_status || row.source_decision || row.blocking_reason);
    return blockingReason !== '—' && tone !== 'positive';
  }) || null;
  const controlChainTargetArtifact = safeDisplayValue(controlChainBlocker?.next_target_artifact);
  const controlChainTargetPath = resolvedArtifactPath(snapshot, controlChainTargetArtifact);
  const controlChainSourceArtifact = safeDisplayValue(controlChainBlocker?.source_artifact || controlChainBlocker?.source_artifact_key);
  const controlChainSourcePath = resolvedArtifactPath(snapshot, controlChainBlocker?.source_artifact, controlChainBlocker?.source_artifact_key);
  const latestFailedHistory = schedulerHistory.find((row) => statusTone(row.status) === 'negative') || null;
  const latestFailedResult = toRecord<Dict>(latestFailedHistory?.result) || {};
  const latestFailedAction = latestFailedHistory
    ? {
        id: 'latest-failed-action-row',
        ts: typeof latestFailedHistory.ts === 'string' ? latestFailedHistory.ts : undefined,
        label: `${safeDisplayValue(latestFailedHistory.slot)} / ${safeDisplayValue(latestFailedHistory.trigger_time)}`,
        status: safeDisplayValue(latestFailedHistory.status),
        detail: safeDisplayValue(latestFailedResult.summary || latestFailedHistory.result),
        tone: statusTone(latestFailedHistory.status),
      }
    : null;
  const alerts: OperatorAlert[] = [
    ...(livePathTouched
      ? [{
          id: 'live-path-touched',
          tone: 'negative' as const,
          title: t('alert_live_path_title'),
          summary: `mode=${safeDisplayValue(experience.mode)} ｜ live_path_touched=true`,
          detail: t('alert_live_path_detail'),
          highlights: [
            alertField('mode', 'mode', experience.mode, { showRaw: true }),
            alertField('change_class', 'change_class', snapshot.meta?.change_class || snapshot.change_class, { showRaw: true }),
            alertField('live_path_touched', 'live_path_touched', experience.live_path_touched),
          ],
          sections: [
            alertSection(
              'live-path-source',
              'alert_section_source',
              [
                alertField('experience_contract', 'experience_contract', 'experience_contract', { showRaw: false }),
                alertField('strict_read_only', 'strict_read_only', false),
                alertField('snapshot_path', 'snapshot_path', snapshotPath, { kind: 'path', showRaw: showRawPaths }),
              ],
              {
                summary: t('alert_section_source_summary'),
                tone: 'negative',
                defaultOpen: true,
              },
            ),
          ],
          links: [
            alertLink('live-path-raw', 'alert_link_view_raw', buildWorkspaceLink('raw'), 'negative'),
            alertLink('live-path-terminal', 'alert_link_view_terminal', buildTerminalLink(surface.effective, {
              panel: 'orchestration',
              section: 'guards',
            }), 'warning'),
            alertLink('live-path-contracts', 'alert_link_view_contracts', buildWorkspaceLink('contracts'), 'warning'),
          ],
        }]
      : []),
    ...(guardianBlocked
      ? [{
          id: 'guardian-blocker',
          tone: guardianTone,
          title: t('alert_guardian_blocker_title'),
          summary: guardianSummary || displayValue(summary.remote_live_gate_brief),
          detail: guardianDetail || displayValue(summary.remote_live_clearing_brief),
          highlights: [
            alertField('action', 'action', summary.openclaw_guardian_clearance_top_blocker_next_action, { showRaw: true }),
            alertField('next_target_artifact', 'next_target_artifact', guardianTargetArtifact, { showRaw: true }),
            alertField('source_artifact', 'source_artifact', operatorPanelPath, { kind: 'path', showRaw: showRawPaths }),
          ],
          sections: [
            alertSection(
              'guardian-resolution',
              'alert_section_resolution',
              [
                alertField('status', 'status', summary.openclaw_guardian_clearance_status, { kind: 'badge' }),
                alertField('score', 'score', summary.openclaw_guardian_clearance_score, { kind: 'number' }),
                alertField('blocker_detail', 'blocker_detail', summary.openclaw_guardian_clearance_top_blocker_title, { showRaw: true }),
                alertField('action', 'action', summary.openclaw_guardian_clearance_top_blocker_next_action, { showRaw: true }),
                alertField('next_target_artifact', 'next_target_artifact', guardianTargetArtifact, { showRaw: true }),
                ...(guardianTargetPath
                  ? [alertField('path', 'path', guardianTargetPath, { kind: 'path', showRaw: showRawPaths })]
                  : []),
              ],
              {
                summary: alertSummary({ summary: guardianSummary, detail: guardianDetail }),
                tone: guardianTone,
                defaultOpen: true,
              },
            ),
            alertSection(
              'guardian-source',
              'alert_section_source',
              [
                alertField('guard_remote_live_gate', 'guard_remote_live_gate', summary.remote_live_gate_brief, { showRaw: true }),
                alertField('guard_remote_live_clearing', 'guard_remote_live_clearing', summary.remote_live_clearing_brief, { showRaw: true }),
                alertField('source_artifact', 'source_artifact', operatorPanelPath, { kind: 'path', showRaw: showRawPaths }),
              ],
              {
                summary: t('alert_section_source_summary'),
                tone: guardianTone,
              },
            ),
          ],
          links: artifactAlertLinks({
            idPrefix: 'guardian-blocker',
            artifact: guardianTargetArtifact,
            rawArtifact: operatorPanelPath,
            terminalFocus: {
              surface: surface.effective,
              panel: 'signal-risk',
              section: 'control-chain',
              row: surface.effective === 'internal' ? buildDrilldownRowId({ stage: 'risk' }, ['stage']) : undefined,
            },
            tone: guardianTone,
          }),
        }]
      : []),
    ...(controlChainBlocker
      ? [{
          id: 'control-chain-blocker',
          tone: statusTone(controlChainBlocker.source_status || controlChainBlocker.source_decision || controlChainBlocker.blocking_reason),
          title: t('alert_control_chain_blocker_title'),
          summary: `${displayValue(controlChainBlocker.label || controlChainBlocker.stage)} → ${displayValue(controlChainBlocker.next_target_artifact)}`,
          detail: displayValue(controlChainBlocker.blocking_reason),
          highlights: [
            alertField('blocking_reason', 'blocking_reason', controlChainBlocker.blocking_reason, { showRaw: true }),
            alertField('next_target_artifact', 'next_target_artifact', controlChainTargetArtifact, { showRaw: true }),
            alertField('source_artifact', 'source_artifact', controlChainSourceArtifact || controlChainSourcePath, { showRaw: true }),
          ],
          sections: [
            alertSection(
              'control-chain-resolution',
              'alert_section_resolution',
              [
                alertField('label', 'stage', controlChainBlocker.label || controlChainBlocker.stage, { showRaw: true }),
                alertField('status', 'status', controlChainBlocker.source_status, { kind: 'badge' }),
                alertField('decision', 'decision', controlChainBlocker.source_decision, { showRaw: true }),
                alertField('blocking_reason', 'blocking_reason', controlChainBlocker.blocking_reason, { showRaw: true }),
                alertField('next_target_artifact', 'next_target_artifact', controlChainTargetArtifact, { showRaw: true }),
                ...(controlChainTargetPath
                  ? [alertField('path', 'path', controlChainTargetPath, { kind: 'path', showRaw: showRawPaths })]
                  : []),
              ],
              {
                summary: alertSummary({
                  summary: `${displayValue(controlChainBlocker.label || controlChainBlocker.stage)} → ${displayValue(controlChainBlocker.next_target_artifact)}`,
                  detail: displayValue(controlChainBlocker.blocking_reason),
                }),
                tone: statusTone(controlChainBlocker.source_status || controlChainBlocker.source_decision || controlChainBlocker.blocking_reason),
                defaultOpen: true,
              },
            ),
            alertSection(
              'control-chain-source',
              'alert_section_source',
              [
                alertField('source_brief', 'source_brief', controlChainBlocker.source_brief, { showRaw: true }),
                alertField('source_artifact', 'source_artifact', controlChainSourceArtifact, { showRaw: true }),
                ...(controlChainSourcePath
                  ? [alertField('source_path', 'source_path', controlChainSourcePath, { kind: 'path', showRaw: showRawPaths })]
                  : []),
                alertField('field', 'field', toArray(controlChainBlocker.interface_fields).join(' / '), { showRaw: true }),
              ],
              {
                summary: t('alert_section_source_summary'),
                tone: statusTone(controlChainBlocker.source_status || controlChainBlocker.source_decision || controlChainBlocker.blocking_reason),
              },
            ),
          ],
          links: artifactAlertLinks({
            idPrefix: 'control-chain-blocker',
            artifact: controlChainTargetArtifact,
            rawArtifact: controlChainSourceArtifact || controlChainSourcePath,
            terminalFocus: {
              surface: surface.effective,
              panel: 'signal-risk',
              section: 'control-chain',
              row: surface.effective === 'internal'
                ? buildDrilldownRowId(controlChainBlocker, ['stage'])
                : undefined,
            },
            tone: statusTone(controlChainBlocker.source_status || controlChainBlocker.source_decision || controlChainBlocker.blocking_reason),
          }),
        }]
      : []),
    ...(latestFailedAction
      ? [{
          id: 'latest-failed-action',
          tone: 'negative' as const,
          title: t('alert_latest_failed_action_title'),
          summary: `${displayValue(latestFailedAction.label)} ｜ ${displayValue(latestFailedAction.status)}`,
          detail: displayValue(latestFailedAction.detail),
          highlights: [
            alertField('generated_at_utc', 'timestamp', latestFailedAction.ts),
            alertField('status', 'status', latestFailedAction.status, { kind: 'badge' }),
            alertField('source_artifact', 'source_artifact', schedulerPath, { kind: 'path', showRaw: showRawPaths }),
          ],
          sections: [
            alertSection(
              'failed-action-resolution',
              'alert_section_resolution',
              [
                alertField('latest_action', 'latest_action', latestFailedAction.label, { showRaw: true }),
                alertField('status', 'status', latestFailedAction.status, { kind: 'badge' }),
                alertField('summary', 'summary', latestFailedAction.detail, { showRaw: true }),
                alertField('generated_at_utc', 'timestamp', latestFailedAction.ts),
              ],
              {
                summary: alertSummary({
                  summary: `${displayValue(latestFailedAction.label)} ｜ ${displayValue(latestFailedAction.status)}`,
                  detail: displayValue(latestFailedAction.detail),
                }),
                tone: 'negative',
                defaultOpen: true,
              },
            ),
            alertSection(
              'failed-action-source',
              'alert_section_source',
              [
                alertField('source_artifact', 'source_artifact', schedulerPath, { kind: 'path', showRaw: showRawPaths }),
                alertField('summary', 'summary', latestFailedResult.summary || latestFailedHistory?.result, { showRaw: true }),
              ],
              {
                summary: t('alert_section_source_summary'),
                tone: 'negative',
              },
            ),
          ],
          links: artifactAlertLinks({
            idPrefix: 'latest-failed-action',
            artifact: 'system_scheduler_state',
            rawArtifact: schedulerPath,
            terminalFocus: {
              surface: surface.effective,
              panel: 'orchestration',
              section: 'action-log',
            },
            tone: 'negative',
          }),
        }]
      : []),
    ...(staleRows.length
      ? [{
          id: 'freshness-stale',
          tone: 'negative' as const,
          title: t('alert_freshness_stale_title').replace('{count}', String(staleRows.length)),
          summary: staleRows.slice(0, 3).map((row) => row.label).join(' ｜ '),
          detail: staleRows
            .slice(0, 3)
            .map((row) => `${row.label} / ${row.ageMinutes ?? '—'}m`)
            .join(' ｜ '),
          highlights: [
            alertField('artifact_payload_count', 'alert_source_count', staleRows.length, { kind: 'number' }),
            alertField('title', 'title', staleRows[0]?.label),
            alertField('source_artifact', 'source_artifact', staleRows[0]?.artifact, { kind: 'path', showRaw: showRawPaths }),
          ],
          sections: staleRows.slice(0, 3).map((row, index) =>
            alertSection(
              `freshness-stale-${row.id}`,
              'alert_section_source',
              [
                alertField('label', 'label', row.label),
                alertField('status', 'status', row.state, { kind: 'badge' }),
                alertField('generated_at_utc', 'timestamp', row.timestamp),
                alertField('source_artifact', 'source_artifact', row.artifact, { kind: 'path', showRaw: showRawPaths }),
                alertField('data_interval_minutes', 'alert_age_minutes', row.ageMinutes, { kind: 'number' }),
              ],
              {
                summary: row.label,
                tone: 'negative',
                defaultOpen: index === 0,
              },
            ),
          ),
          links: artifactAlertLinks({
            idPrefix: 'freshness-stale',
            artifact: staleRows[0]?.artifact,
            rawArtifact: staleRows[0]?.artifact,
            terminalFocus: {
              surface: surface.effective,
              panel: 'orchestration',
              section: 'freshness',
            },
            tone: 'negative',
          }),
        }]
      : warningRows.length
        ? [{
            id: 'freshness-warning',
            tone: 'warning' as const,
            title: t('alert_freshness_warning_title').replace('{count}', String(warningRows.length)),
            summary: warningRows.slice(0, 3).map((row) => row.label).join(' ｜ '),
            detail: warningRows
              .slice(0, 3)
              .map((row) => `${row.label} / ${row.ageMinutes ?? '—'}m`)
              .join(' ｜ '),
            highlights: [
              alertField('artifact_payload_count', 'alert_source_count', warningRows.length, { kind: 'number' }),
              alertField('title', 'title', warningRows[0]?.label),
              alertField('source_artifact', 'source_artifact', warningRows[0]?.artifact, { kind: 'path', showRaw: showRawPaths }),
            ],
            sections: warningRows.slice(0, 3).map((row, index) =>
              alertSection(
                `freshness-warning-${row.id}`,
                'alert_section_source',
                [
                  alertField('label', 'label', row.label),
                  alertField('status', 'status', row.state, { kind: 'badge' }),
                  alertField('generated_at_utc', 'timestamp', row.timestamp),
                  alertField('source_artifact', 'source_artifact', row.artifact, { kind: 'path', showRaw: showRawPaths }),
                  alertField('data_interval_minutes', 'alert_age_minutes', row.ageMinutes, { kind: 'number' }),
                ],
                {
                  summary: row.label,
                  tone: 'warning',
                  defaultOpen: index === 0,
                },
              ),
            ),
            links: artifactAlertLinks({
              idPrefix: 'freshness-warning',
              artifact: warningRows[0]?.artifact,
              rawArtifact: warningRows[0]?.artifact,
              terminalFocus: {
                surface: surface.effective,
                panel: 'orchestration',
                section: 'freshness',
              },
              tone: 'warning',
            }),
          }]
        : []),
    ...(surface.requested !== surface.effective || surface.warnings.length
      ? [{
          id: 'surface-fallback',
          tone: 'warning' as const,
          title: t('alert_surface_fallback_title'),
          summary: `${surface.requestedLabel} → ${surface.effectiveLabel}`,
          detail: surface.warnings.join(' ｜ ') || t('alert_surface_fallback_detail'),
          highlights: [
            alertField('requested', 'requested', surface.requestedLabel),
            alertField('surface', 'surface', surface.effectiveLabel),
            alertField('redaction_level', 'redaction_level', surface.redactionLevel),
          ],
          sections: [
            alertSection(
              'surface-fallback-source',
              'alert_section_source',
              [
                alertField('requested', 'requested', surface.requestedLabel),
                alertField('surface', 'surface', surface.effectiveLabel),
                alertField('availability', 'availability', surface.availability),
                alertField('redaction_level', 'redaction_level', surface.redactionLevel),
                alertField('snapshot_path', 'snapshot_path', snapshotPath, { kind: 'path', showRaw: showRawPaths }),
              ],
              {
                summary: `${surface.requestedLabel} → ${surface.effectiveLabel}`,
                tone: 'warning',
                defaultOpen: true,
              },
            ),
          ],
          links: [
            alertLink('surface-fallback-contracts', 'alert_link_view_contracts', buildWorkspaceLink('contracts'), 'warning'),
            alertLink('surface-fallback-raw', 'alert_link_view_raw', buildWorkspaceLink('raw'), 'warning'),
            alertLink('surface-fallback-terminal', 'alert_link_view_terminal', buildTerminalLink(surface.effective), 'warning'),
          ],
        }]
      : []),
  ];

  const guards: MetricItem[] = [
    toMetric('remote-live-gate', 'guard_remote_live_gate', summary.remote_live_gate_brief, t('source_owned_gate_brief')),
    toMetric('remote-live-clearing', 'guard_remote_live_clearing', summary.remote_live_clearing_brief, t('clear vs hold')),
    toMetric('lane-state', 'guard_lane_state', summary.lane_state_brief, t('lane count by state')),
    toMetric('lane-priority', 'guard_lane_priority', summary.lane_priority_order_brief, t('priority_total_collapse')),
    toMetric('arch-status', 'guard_architecture_status', architecture.status, displayValue(nested(architecture, 'health.status'))),
    toMetric('daily-briefing', 'guard_daily_briefing', checks.daily_briefing, t('health.checks.daily_briefing')),
    toMetric('daily-signals', 'guard_daily_signals', checks.daily_signals, t('health.checks.daily_signals')),
    toMetric('daily-positions', 'guard_daily_positions', checks.daily_positions, t('health.checks.daily_positions')),
    toMetric('sqlite', 'guard_sqlite_check', checks.sqlite, t('health.checks.sqlite')),
    toMetric('manifest-review', 'guard_manifest_review', manifests.review_manifest, t('architecture_audit.manifests.review_manifest')),
  ];

  return { alerts, topMetrics, freshness, guards, actionLog, checklist };
}

function buildDataRegime(snapshot: DashboardSnapshot) {
  const scheduler = artifactPayload(snapshot, 'system_scheduler_state');
  const stress = artifactPayload(snapshot, 'mode_stress_matrix');
  const latestHistory = toRecord<Dict>(toArray<Dict>(scheduler.history)[0]) || {};
  const latestResult = toRecord<Dict>(latestHistory.result) || {};
  const latestSummary = toRecord<Dict>(latestResult.summary) || {};
  const rolling = toRecord<Dict>(latestResult.rolling_7d) || {};

  const eventRegime = eventArtifact(snapshot, 'event_regime_snapshot');
  const eventAnalogy = eventArtifact(snapshot, 'event_crisis_analogy');
  const eventAssetShockMap = eventArtifact(snapshot, 'event_asset_shock_map');
  const eventOperatorSummary = eventArtifact(snapshot, 'event_crisis_operator_summary');

  const sourceConfidence: MetricItem[] = [
    toMetric('schema-ok-ratio', 'data_schema_ok_ratio', latestSummary.selected_schema_ok_ratio, t('selected_schema_ok_ratio')),
    toMetric('time-sync-ratio', 'data_time_sync_ok_ratio', latestSummary.selected_time_sync_ok_ratio, t('selected_time_sync_ok_ratio')),
    toMetric('source-sync-ratio', 'data_source_sync_ok_ratio', latestSummary.selected_sync_ok_ratio, t('selected_sync_ok_ratio')),
    toMetric('cross-source-status', 'data_cross_source_status', latestSummary.cross_source_status, t('cross_source_status')),
    toMetric('cross-source-fail-ratio', 'data_cross_source_fail_ratio', latestSummary.cross_source_fail_ratio, buildAssignmentHint('cross_source_fail_ratio_limit', latestSummary.cross_source_fail_ratio_limit)),
    toMetric('symbols-audited', 'data_symbols_audited', latestSummary.cross_source_symbols_audited, buildAssignmentHint('cross_source_symbols_insufficient', latestSummary.cross_source_symbols_insufficient)),
  ];

  const microCapture: MetricItem[] = [
    toMetric('task-enabled', 'data_task_enabled', nested(scheduler, 'interval_tasks.micro_capture.enabled'), t('interval_tasks.micro_capture.enabled')),
    toMetric('interval-minutes', 'data_interval_minutes', nested(scheduler, 'interval_tasks.micro_capture.interval_minutes'), t('interval_tasks.micro_capture.interval_minutes')),
    toMetric('covered-symbols', 'data_covered_symbols', nested(scheduler, 'interval_tasks.micro_capture.symbols'), t('micro_capture.symbols')),
    toMetric('rolling-run-count', 'data_rolling_run_count_7d', rolling.run_count, t('rolling_7d.run_count')),
    toMetric('rolling-pass-ratio', 'data_rolling_pass_ratio_7d', rolling.pass_ratio, t('rolling_7d.pass_ratio')),
    toMetric('rolling-cross-source-fail', 'data_rolling_cross_source_fail_7d', rolling.avg_cross_source_fail_ratio, t('rolling_7d.avg_cross_source_fail_ratio')),
  ];

  const addEventMetric = (metric: MetricItem) => {
    microCapture.push(metric);
  };

  const regimeState = normalizedString(eventRegime.regime_state);
  const headlineDrivers = toArray<unknown>(eventRegime.headline_drivers)
    .map((value) => normalizedString(value))
    .filter((value): value is string => Boolean(value));
  const regimeDriversHint = headlineDrivers.length ? headlineDrivers.join(', ') : undefined;
  const analogyHint = (() => {
    const analogues = toArray<Dict>(eventAnalogy.top_analogues ?? []);
    const ids = analogues
      .map((analogy) => normalizedString(analogy.archetype_id))
      .filter((value): value is string => Boolean(value));
    return ids.length ? `analogues=${ids.join(',')}` : undefined;
  })();

  if (eventRegime.event_severity_score !== undefined) {
    addEventMetric(
      toMetric(
        'event-severity',
        'event_severity_score',
        eventRegime.event_severity_score,
        regimeDriversHint,
        regimeState,
      ),
    );
  }
  if (eventRegime.systemic_risk_score !== undefined) {
    addEventMetric(
      toMetric(
        'event-systemic-risk',
        'systemic_risk_score',
        eventRegime.systemic_risk_score,
        regimeState,
        regimeState,
      ),
    );
  }
  if (eventRegime.regime_state) {
    addEventMetric(
      toMetric(
        'event-regime-state',
        'event_regime_state',
        regimeState,
        analogyHint,
      ),
    );
  }
  const analogyCandidates = toArray<unknown>(eventAnalogy.top_analogues ?? [])
    .map((analogy) => toRecord<Dict>(analogy))
    .filter((row): row is Dict => Boolean(row))
    .map((row) => ({
      row,
      id: normalizedString(row.archetype_id),
    }))
    .filter((candidate): candidate is { row: Dict; id: string } => Boolean(candidate.id));
  const firstAnalogyCandidate = analogyCandidates[0];
  if (firstAnalogyCandidate) {
    const firstAnalogy = firstAnalogyCandidate.row;
    const analogyHintParts: string[] = [];
    if (firstAnalogy.similarity_score !== undefined) {
      analogyHintParts.push(`score=${safeDisplayValue(firstAnalogy.similarity_score)}`);
    }
    const matchAxes = toArray<unknown>(firstAnalogy.match_axes ?? [])
      .map((value) => normalizedString(value))
      .filter((value): value is string => Boolean(value));
    if (matchAxes.length) {
      analogyHintParts.push(`match=${matchAxes.join(',')}`);
    }
    addEventMetric(
      toMetric(
        'event-analogy',
        'event_crisis_analogy',
        normalizedString(firstAnalogy.archetype_id),
        analogyHintParts.join(' / ') || undefined,
      ),
    );
  }
  const assetCandidates = toArray<unknown>(eventAssetShockMap.assets ?? [])
    .map((entry) => toRecord<Dict>(entry))
    .filter((row): row is Dict => Boolean(row))
    .map((row) => {
      return {
        row,
        asset: normalizedString(row.asset),
      };
    })
    .filter((candidate): candidate is { row: Dict; asset: string } => Boolean(candidate.asset));
  const assetNames = assetCandidates.map((candidate) => candidate.asset);
  if (assetNames.length) {
    addEventMetric(
      toMetric(
        'event-shock-map',
        'event_asset_shock_map',
        assetNames.join(' ｜ '),
        `assets=${assetNames.slice(0, 3).join(',')}`,
      ),
    );
  }

  return {
    sourceConfidence,
    microCapture,
    regimeSummary: toArray<Dict>(stress.mode_summary),
    regimeMatrix: toArray<Dict>(stress.matrix).slice(0, 12),
  };
}

function buildSignalRisk(snapshot: DashboardSnapshot, surface: SurfaceView) {
  const operatorPanel = artifactPayload(snapshot, 'operator_panel');
  const summary = toRecord<Dict>(operatorPanel.summary) || {};
  const repairPlan = toRecord<Dict>(operatorPanel.priority_repair_plan) || {};
  const eventOperatorSummary = eventArtifact(snapshot, 'event_crisis_operator_summary');

  const gateScores: MetricItem[] = [
    toMetric('guardian-clearance', 'signal_guardian_clearance', summary.openclaw_guardian_clearance_status, buildAssignmentHint('score', summary.openclaw_guardian_clearance_score)),
    toMetric('canary-gate', 'signal_canary_gate', summary.openclaw_canary_gate_status, displayValue(summary.openclaw_canary_gate_decision)),
    toMetric('promotion-gate', 'signal_promotion_gate', summary.openclaw_promotion_gate_status, displayValue(summary.openclaw_promotion_gate_decision)),
    toMetric('live-boundary', 'signal_live_boundary', summary.openclaw_live_boundary_hold_status, displayValue(summary.openclaw_live_boundary_hold_decision)),
    toMetric('orderflow-policy', 'signal_orderflow_policy', summary.openclaw_orderflow_policy_status, displayValue(summary.openclaw_orderflow_policy_decision)),
    toMetric('execution-ack', 'signal_execution_ack', summary.openclaw_execution_ack_status, displayValue(summary.openclaw_execution_ack_decision)),
    toMetric('quality-score', 'signal_quality_score', summary.openclaw_quality_report_score, joinAssignmentHints([
      ['shadow', summary.openclaw_quality_shadow_learning_score],
      ['execution', summary.openclaw_quality_execution_readiness_score],
    ])),
  ];

  const repairMetrics: MetricItem[] = [
    toMetric('repair-status', 'signal_priority_repair', repairPlan.status, buildAssignmentHint('done_when', repairPlan.done_when)),
    toMetric('repair-brief', 'signal_repair_brief', summary.priority_repair_plan_brief, displayValue(summary.priority_repair_plan_artifact)),
    toMetric('repair-verify', 'signal_repair_verify', summary.priority_repair_verification_brief, displayValue(summary.priority_repair_verification_artifact)),
    toMetric('remote-diagnosis', 'signal_remote_diagnosis', summary.remote_live_diagnosis_brief, displayValue(summary.remote_live_history_brief)),
  ];
  if (eventOperatorSummary.summary || eventOperatorSummary.status) {
    repairMetrics.push(
      toMetric(
        'event-crisis-summary',
        'event_crisis_operator_summary',
        normalizedString(eventOperatorSummary.summary || eventOperatorSummary.status),
        normalizedString(eventOperatorSummary.takeaway),
        normalizedString(eventOperatorSummary.status),
      ),
    );
  }

  return {
    laneCards: toArray(snapshot.artifact_payloads?.operator_panel?.payload && operatorPanel.lane_cards) as TerminalReadModel['signalRisk']['laneCards'],
    focusSlots: toArray(snapshot.artifact_payloads?.operator_panel?.payload && operatorPanel.focus_slots) as TerminalReadModel['signalRisk']['focusSlots'],
    gateScores,
    controlChain: toArray(snapshot.artifact_payloads?.operator_panel?.payload && operatorPanel.control_chain) as TerminalReadModel['signalRisk']['controlChain'],
    actionQueue: toArray(snapshot.artifact_payloads?.operator_panel?.payload && operatorPanel.action_queue).slice(0, surface.effective === 'internal' ? 6 : 4) as TerminalReadModel['signalRisk']['actionQueue'],
    repairPlan: repairMetrics,
    optimizationBacklog: toArray<Dict>(operatorPanel.continuous_optimization_backlog).slice(0, surface.effective === 'internal' ? 6 : 4),
  };
}

function buildLabReview(snapshot: DashboardSnapshot) {
  const recentBacktests = artifactPayload(snapshot, 'recent_strategy_backtests');
  const architecture = artifactPayload(snapshot, 'architecture_audit');
  const stress = artifactPayload(snapshot, 'mode_stress_matrix');
  const sourceHeads = snapshot.source_heads || {};
  const candidateRows: ResearchCandidateRow[] = ['price_action_breakout_pullback', 'price_action_exit_risk', 'cross_section_backtest', 'recent_strategy_backtests'].map((id) => {
    const summary = artifactEntry(snapshot, id)?.summary;
    return {
      id,
      label: labelFor(id),
      status: summary?.status,
      decision: summary?.research_decision,
      changeClass: summary?.change_class,
      path: summary?.path,
      takeaway: summary?.takeaway || summary?.recommended_brief,
    };
  });

  const manifests = toRecord<Dict>(architecture.manifests) || {};
  const researchHeads: MetricItem[] = Object.entries(sourceHeads).slice(0, 6).map(([id, row]) =>
    toMetric(`source-head-${id}`, id, row.summary || row.research_decision || row.status, displayValue(row.path), row.status)
  );

  return {
    researchHeads,
    backtests: snapshot.backtest_artifacts || [],
    comparisonRows: toArray<Dict>(recentBacktests.rows).slice(0, 8),
    manifests: [
      toMetric('manifest-eod', 'lab_manifest_eod', manifests.eod_manifest, t('architecture_audit.manifests.eod_manifest')),
      toMetric('manifest-backtest', 'lab_manifest_backtest', manifests.backtest_manifest, t('architecture_audit.manifests.backtest_manifest')),
      toMetric('manifest-review', 'lab_manifest_review', manifests.review_manifest, t('architecture_audit.manifests.review_manifest')),
      toMetric('stress-best-mode', 'lab_best_mode', stress.best_mode, t('mode_stress_matrix.best_mode')),
      toMetric('temporal-audit', 'lab_temporal_audit_wall', 'unwired', t('lab_temporal_artifact_required'), 'warning'),
    ],
    candidateRows,
    stressSummary: toArray<Dict>(stress.mode_summary),
  };
}

function normalizedFocusValue(value: unknown): string {
  return safeDisplayValue(value).trim().toLowerCase();
}

function artifactRefMatches(row: TerminalReadModel['workspace']['artifactRows'][number], candidate: unknown): boolean {
  const target = normalizedFocusValue(candidate);
  if (!target || target === '—') return false;
  return [row.id, row.payload_key, row.label, row.path]
    .map((value) => normalizedFocusValue(value))
    .some((value) => value && value !== '—' && (value === target || value.includes(target) || target.includes(value)));
}

function terminalHandoffLabel(...parts: Array<string | undefined>): string {
  return [t('alert_link_view_terminal'), ...parts.filter(Boolean)].join(' / ');
}

function terminalHandoffLink(
  id: string,
  surface: SurfaceView,
  focus: Parameters<typeof buildTerminalLink>[1],
  parts: Array<string | undefined>,
  tone: OperatorAlertLink['tone'] = 'neutral',
): OperatorAlertLink {
  return {
    id,
    label: terminalHandoffLabel(...parts),
    to: buildTerminalLink(surface.effective, focus),
    tone,
  };
}

function buildWorkspaceArtifactHandoffs(
  snapshot: DashboardSnapshot,
  surface: SurfaceView,
  artifactRows: Array<TerminalReadModel['workspace']['artifactRows'][number]>,
) {
  const operatorPanel = artifactPayload(snapshot, 'operator_panel');
  const controlChain = toArray<Dict>(operatorPanel.control_chain);
  const focusSlots = toArray<Dict>(operatorPanel.focus_slots);
  const actionQueue = toArray<Dict>(operatorPanel.action_queue);
  const handoffs: Record<string, OperatorAlertLink[]> = {};
  const primaryFocus: Record<string, { panel?: string; section?: string; row?: string }> = {};
  const seen = new Set<string>();

  const push = (
    artifactId: string,
    focus: { panel?: string; section?: string; row?: string },
    link: OperatorAlertLink,
  ) => {
    const signature = `${artifactId}::${link.to}`;
    if (seen.has(signature)) return;
    seen.add(signature);
    handoffs[artifactId] ||= [];
    handoffs[artifactId].push(link);
    if (!primaryFocus[artifactId] && focus.panel && focus.section) {
      primaryFocus[artifactId] = focus;
    }
  };

  artifactRows.forEach((row) => {
    if (artifactRefMatches(row, 'operator_panel')) {
      push(row.id, {
        panel: 'orchestration',
        section: 'freshness',
      }, terminalHandoffLink(`${row.id}-terminal-orch`, surface, {
        artifact: row.id,
        panel: 'orchestration',
        section: 'freshness',
      }, [t('terminal_panel_orchestration_title'), t('terminal_section_freshness_title')]));
    }
    if (artifactRefMatches(row, 'system_scheduler_state')) {
      push(row.id, {
        panel: 'orchestration',
        section: 'action-log',
      }, terminalHandoffLink(`${row.id}-terminal-log`, surface, {
        artifact: row.id,
        panel: 'orchestration',
        section: 'action-log',
      }, [t('terminal_panel_orchestration_title'), t('terminal_section_action_log_title')]));
    }
    if (artifactRefMatches(row, 'architecture_audit')) {
      push(row.id, {
        panel: 'orchestration',
        section: 'guards',
      }, terminalHandoffLink(`${row.id}-terminal-guards`, surface, {
        artifact: row.id,
        panel: 'orchestration',
        section: 'guards',
      }, [t('terminal_panel_orchestration_title'), t('terminal_section_guard_board_title')]));
      push(row.id, {
        panel: 'lab-review',
        section: 'manifests',
      }, terminalHandoffLink(`${row.id}-terminal-manifests`, surface, {
        artifact: row.id,
        panel: 'lab-review',
        section: 'manifests',
      }, [t('terminal_panel_lab_review_title'), t('terminal_section_manifests_title')]));
    }
    if (artifactRefMatches(row, 'mode_stress_matrix')) {
      push(row.id, {
        panel: 'data-regime',
        section: 'regime-matrix',
      }, terminalHandoffLink(`${row.id}-terminal-regime`, surface, {
        artifact: row.id,
        panel: 'data-regime',
        section: 'regime-matrix',
      }, [t('terminal_panel_data_regime_title'), t('terminal_section_regime_matrix_title')]));
      push(row.id, {
        panel: 'lab-review',
        section: 'stress-summary',
      }, terminalHandoffLink(`${row.id}-terminal-stress`, surface, {
        artifact: row.id,
        panel: 'lab-review',
        section: 'stress-summary',
      }, [t('terminal_panel_lab_review_title'), t('terminal_section_stress_summary_title')]));
    }
    if (artifactRefMatches(row, 'recent_strategy_backtests')) {
      push(row.id, {
        panel: 'lab-review',
        section: 'comparison-wall',
      }, terminalHandoffLink(`${row.id}-terminal-compare`, surface, {
        artifact: row.id,
        panel: 'lab-review',
        section: 'comparison-wall',
      }, [t('terminal_panel_lab_review_title'), t('terminal_section_comparison_wall_title')]));
    }
    if (artifactRefMatches(row, 'price_action_breakout_pullback')
      || artifactRefMatches(row, 'price_action_exit_risk')
      || artifactRefMatches(row, 'cross_section_backtest')
      || artifactRefMatches(row, 'intraday_orderflow_blueprint')
      || artifactRefMatches(row, 'intraday_orderflow_research_gate_blocker')) {
      push(row.id, {
        panel: 'lab-review',
        section: 'research-heads',
      }, terminalHandoffLink(`${row.id}-terminal-research`, surface, {
        artifact: row.id,
        panel: 'lab-review',
        section: 'research-heads',
      }, [t('terminal_panel_lab_review_title'), t('terminal_section_research_heads_title')]));
    }

    controlChain.forEach((chain) => {
      if (
        artifactRefMatches(row, chain.next_target_artifact)
        || artifactRefMatches(row, chain.source_artifact_key)
        || artifactRefMatches(row, chain.source_artifact)
      ) {
        const focus = {
          panel: 'signal-risk',
          section: 'control-chain',
          row: surface.effective === 'internal' ? buildDrilldownRowId(chain, ['stage']) : undefined,
        };
        push(row.id, focus, terminalHandoffLink(`${row.id}-control-${safeDisplayValue(chain.stage || chain.label)}`, surface, {
          artifact: row.id,
          ...focus,
        }, [t('terminal_panel_signal_risk_title'), t('terminal_section_control_chain_title'), displayValue(chain.label || chain.stage)]));
      }
    });

    focusSlots.forEach((slot) => {
      if (artifactRefMatches(row, slot.source_artifact)) {
        const focus = {
          panel: 'signal-risk',
          section: 'focus-slots',
          row: surface.effective === 'internal' ? buildDrilldownRowId(slot, ['slot']) : undefined,
        };
        push(row.id, focus, terminalHandoffLink(`${row.id}-slot-${safeDisplayValue(slot.slot)}`, surface, {
          artifact: row.id,
          ...focus,
        }, [t('terminal_panel_signal_risk_title'), t('terminal_section_focus_slots_title'), displayValue(slot.slot)]));
      }
    });

    actionQueue.forEach((queue) => {
      if (artifactRefMatches(row, queue.source_artifact)) {
        const focus = {
          panel: 'signal-risk',
          section: 'action-stack',
          row: surface.effective === 'internal' ? buildDrilldownRowId(queue, ['symbol', 'action']) : undefined,
        };
        push(row.id, focus, terminalHandoffLink(`${row.id}-queue-${safeDisplayValue(queue.symbol)}`, surface, {
          artifact: row.id,
          ...focus,
        }, [t('terminal_panel_signal_risk_title'), t('terminal_section_action_stack_title'), displayValue(queue.symbol)]));
      }
    });
  });

  return { handoffs, primaryFocus };
}

function buildTerminalWorkspaceHandoffs(snapshot: DashboardSnapshot, surface: SurfaceView) {
  const operatorPanel = artifactPayload(snapshot, 'operator_panel');
  const controlChain = toArray<Dict>(operatorPanel.control_chain);
  const focusSlots = toArray<Dict>(operatorPanel.focus_slots);
  const actionQueue = toArray<Dict>(operatorPanel.action_queue);
  const summary = toRecord<Dict>(operatorPanel.summary) || {};
  const handoffs: Record<string, OperatorAlertLink[]> = {};
  const primaryArtifact: Record<string, string> = {};
  const seen = new Set<string>();

  const push = (
    focus: Parameters<typeof buildTerminalFocusKey>[0],
    link: OperatorAlertLink,
    primaryArtifactCandidate?: string,
  ) => {
    const focusKey = buildTerminalFocusKey(focus);
    const signature = `${focusKey}::${link.to}`;
    if (seen.has(signature)) return;
    seen.add(signature);
    handoffs[focusKey] ||= [];
    handoffs[focusKey].push(link);
    if (!primaryArtifact[focusKey] && primaryArtifactCandidate) primaryArtifact[focusKey] = primaryArtifactCandidate;
  };

  const pushWorkspaceLinks = (
    focus: Parameters<typeof buildTerminalFocusKey>[0],
    idPrefix: string,
    artifact?: unknown,
    rawArtifact?: unknown,
    tone: OperatorAlertLink['tone'] = 'neutral',
  ) => {
    const artifactValue = safeDisplayValue(artifact);
    const rawValue = safeDisplayValue(rawArtifact || artifact);
    if (artifactValue && artifactValue !== '—') {
      push(focus, {
        id: `${idPrefix}-artifact`,
        label: `${t('alert_link_view_artifact')} / ${displayValue(artifactValue)}`,
        to: buildWorkspaceLink('artifacts', { artifact: artifactValue, ...focus }),
        tone,
      }, artifactValue);
    }
    if (rawValue && rawValue !== '—') {
      push(focus, {
        id: `${idPrefix}-raw`,
        label: `${t('alert_link_view_raw')} / ${displayValue(rawValue)}`,
        to: buildWorkspaceLink('raw', { artifact: rawValue, ...focus }),
        tone,
      });
    }
  };

  pushWorkspaceLinks({ panel: 'orchestration', section: 'action-log' }, 'orchestration-action-log', 'system_scheduler_state', loadedArtifact(snapshot, 'system_scheduler_state'));
  pushWorkspaceLinks({ panel: 'orchestration', section: 'guards' }, 'orchestration-guards', 'architecture_audit', loadedArtifact(snapshot, 'architecture_audit'));
  pushWorkspaceLinks({ panel: 'orchestration', section: 'freshness' }, 'orchestration-freshness', 'operator_panel', loadedArtifact(snapshot, 'operator_panel'));
  pushWorkspaceLinks({ panel: 'data-regime', section: 'source-confidence' }, 'data-source-confidence', 'system_scheduler_state', loadedArtifact(snapshot, 'system_scheduler_state'));
  pushWorkspaceLinks({ panel: 'data-regime', section: 'micro-capture' }, 'data-micro-capture', 'system_scheduler_state', loadedArtifact(snapshot, 'system_scheduler_state'));
  pushWorkspaceLinks({ panel: 'data-regime', section: 'regime-matrix' }, 'data-regime-matrix', 'mode_stress_matrix', loadedArtifact(snapshot, 'mode_stress_matrix'));
  pushWorkspaceLinks({ panel: 'lab-review', section: 'manifests' }, 'lab-manifests', 'architecture_audit', loadedArtifact(snapshot, 'architecture_audit'));
  pushWorkspaceLinks({ panel: 'lab-review', section: 'comparison-wall' }, 'lab-comparison', 'recent_strategy_backtests', loadedArtifact(snapshot, 'recent_strategy_backtests'));
  pushWorkspaceLinks({ panel: 'lab-review', section: 'stress-summary' }, 'lab-stress', 'mode_stress_matrix', loadedArtifact(snapshot, 'mode_stress_matrix'));
  ['price_action_breakout_pullback', 'price_action_exit_risk', 'cross_section_backtest'].forEach((artifact, index) => {
    pushWorkspaceLinks({ panel: 'lab-review', section: 'research-heads' }, `lab-research-${index}`, artifact, loadedArtifact(snapshot, artifact));
  });
  pushWorkspaceLinks(
    { panel: 'signal-risk', section: 'action-stack' },
    'signal-action-stack-priority',
    summary.priority_repair_plan_artifact,
    summary.priority_repair_plan_artifact,
    'warning',
  );

  controlChain.forEach((row, index) => {
    const focus = {
      panel: 'signal-risk',
      section: 'control-chain',
      row: surface.effective === 'internal' ? buildDrilldownRowId(row, ['stage']) : undefined,
    };
    pushWorkspaceLinks(
      { panel: 'signal-risk', section: 'control-chain' },
      `signal-control-chain-section-${index}`,
      row.next_target_artifact || row.source_artifact_key,
      row.source_artifact || row.source_artifact_key,
      statusTone(row.source_status || row.source_decision),
    );
    pushWorkspaceLinks(
      focus,
      `signal-control-chain-row-${index}`,
      row.next_target_artifact || row.source_artifact_key,
      row.source_artifact || row.source_artifact_key,
      statusTone(row.source_status || row.source_decision),
    );
  });

  focusSlots.forEach((row, index) => {
    const focus = {
      panel: 'signal-risk',
      section: 'focus-slots',
      row: surface.effective === 'internal' ? buildDrilldownRowId(row, ['slot']) : undefined,
    };
    pushWorkspaceLinks({ panel: 'signal-risk', section: 'focus-slots' }, `signal-focus-slots-section-${index}`, row.source_artifact, row.source_artifact, statusTone(row.state));
    pushWorkspaceLinks(focus, `signal-focus-slots-row-${index}`, row.source_artifact, row.source_artifact, statusTone(row.state));
  });

  actionQueue.forEach((row, index) => {
    const focus = {
      panel: 'signal-risk',
      section: 'action-stack',
      row: surface.effective === 'internal' ? buildDrilldownRowId(row, ['symbol', 'action']) : undefined,
    };
    pushWorkspaceLinks({ panel: 'signal-risk', section: 'action-stack' }, `signal-action-stack-section-${index}`, row.source_artifact, row.source_artifact, statusTone(row.state));
    pushWorkspaceLinks(focus, `signal-action-stack-row-${index}`, row.source_artifact, row.source_artifact, statusTone(row.state));
  });

  return { handoffs, primaryArtifact };
}

function buildWorkspace(snapshot: DashboardSnapshot, surface: SurfaceView) {
  const experienceContract = toRecord<Dict>(snapshot.experience_contract) || {};
  const artifactRows = (snapshot.catalog || []).map((row, index) => ({ id: row.id || row.payload_key || `artifact-${index}`, ...row }));
  const workspaceHandoffs = buildWorkspaceArtifactHandoffs(snapshot, surface, artifactRows);
  const defaultFocusRecord = toRecord<Dict>(snapshot.workspace_default_focus) || {};
  return {
    domains: snapshot.domain_summary || [],
    interfaces: snapshot.interface_catalog || [],
    publicTopology: snapshot.public_topology || [],
    publicAcceptance: buildPublicAcceptance(snapshot, surface),
    surfaceContracts: Object.entries(snapshot.surface_contracts || {}).map(([id, row]) => ({ id, ...row })),
    fallbackChain: toArray<string>(experienceContract.fallback_chain),
    sourceHeads: Object.entries(snapshot.source_heads || {}).map(([id, row]) => ({ id, ...row })),
    defaultFocus: {
      artifact: typeof defaultFocusRecord.artifact === 'string' ? defaultFocusRecord.artifact : undefined,
      group: typeof defaultFocusRecord.group === 'string' ? defaultFocusRecord.group : undefined,
      panel: typeof defaultFocusRecord.panel === 'string' ? defaultFocusRecord.panel : undefined,
      section: typeof defaultFocusRecord.section === 'string' ? defaultFocusRecord.section : undefined,
      row: typeof defaultFocusRecord.row === 'string' ? defaultFocusRecord.row : undefined,
      search_scope: typeof defaultFocusRecord.search_scope === 'string' ? defaultFocusRecord.search_scope : undefined,
    },
    artifactRows,
    artifactPrimaryFocus: workspaceHandoffs.primaryFocus,
    artifactHandoffs: workspaceHandoffs.handoffs,
    artifactPayloads: snapshot.artifact_payloads || {},
    raw: snapshot,
  };
}

function buildFeedback(snapshot: DashboardSnapshot): {
  projection: ConversationFeedbackProjection | null;
  domainGroups: ConversationFeedbackDomainGroup[];
} {
  const projectionRecord = toRecord<Dict>(snapshot.conversation_feedback_projection) || {};
  if (!Object.keys(projectionRecord).length) {
    return {
      projection: null,
      domainGroups: [],
    };
  }

  const summary = toRecord<Dict>(projectionRecord.summary) || {};
  const anchors = toArray<Dict>(projectionRecord.anchors).map<ConversationFeedbackAnchor>((row) => ({
    route: safeDisplayValue(row.route) || undefined,
    artifact: safeDisplayValue(row.artifact) || undefined,
    component: safeDisplayValue(row.component) || undefined,
  }));
  const events = toArray<Dict>(projectionRecord.events).map<ConversationFeedbackEvent>((row) => ({
    feedback_id: safeDisplayValue(row.feedback_id),
    created_at_utc: safeDisplayValue(row.created_at_utc) || undefined,
    source: safeDisplayValue(row.source) || undefined,
    domain: safeDisplayValue(row.domain) || undefined,
    headline: safeDisplayValue(row.headline) || undefined,
    summary: safeDisplayValue(row.summary) || undefined,
    recommended_action: safeDisplayValue(row.recommended_action) || undefined,
    alignment_delta: typeof row.alignment_delta === 'number' ? row.alignment_delta : undefined,
    blocker_delta: typeof row.blocker_delta === 'number' ? row.blocker_delta : undefined,
    execution_delta: typeof row.execution_delta === 'number' ? row.execution_delta : undefined,
    readability_delta: typeof row.readability_delta === 'number' ? row.readability_delta : undefined,
    impact_score: typeof row.impact_score === 'number' ? row.impact_score : undefined,
    confidence: typeof row.confidence === 'number' ? row.confidence : undefined,
    status: safeDisplayValue(row.status) || undefined,
    anchors: toArray<Dict>(row.anchors).map((anchor) => ({
      route: safeDisplayValue(anchor.route) || undefined,
      artifact: safeDisplayValue(anchor.artifact) || undefined,
      component: safeDisplayValue(anchor.component) || undefined,
    })),
  }));
  const actions = toArray<Dict>(projectionRecord.actions).map<ConversationFeedbackAction>((row) => ({
    feedback_id: safeDisplayValue(row.feedback_id) || undefined,
    recommended_action: safeDisplayValue(row.recommended_action) || undefined,
    impact_score: typeof row.impact_score === 'number' ? row.impact_score : undefined,
    anchors: toArray<Dict>(row.anchors).map((anchor) => ({
      route: safeDisplayValue(anchor.route) || undefined,
      artifact: safeDisplayValue(anchor.artifact) || undefined,
      component: safeDisplayValue(anchor.component) || undefined,
    })),
  }));
  const trends = toArray<Dict>(projectionRecord.trends).map<ConversationFeedbackTrendPoint>((row) => ({
    feedback_id: safeDisplayValue(row.feedback_id) || undefined,
    created_at_utc: safeDisplayValue(row.created_at_utc) || undefined,
    alignment_score: typeof row.alignment_score === 'number' ? row.alignment_score : undefined,
    blocker_pressure: typeof row.blocker_pressure === 'number' ? row.blocker_pressure : undefined,
    execution_clarity: typeof row.execution_clarity === 'number' ? row.execution_clarity : undefined,
    readability_pressure: typeof row.readability_pressure === 'number' ? row.readability_pressure : undefined,
    drift_state: safeDisplayValue(row.drift_state) || undefined,
  }));

  const domainGroups = Object.values(events.reduce<Record<string, ConversationFeedbackDomainGroup>>((groups, row) => {
    const domain = row.domain || 'global';
    const current = groups[domain] || { domain, count: 0, topHeadline: undefined };
    current.count += 1;
    if (!current.topHeadline) current.topHeadline = row.headline;
    groups[domain] = current;
    return groups;
  }, {})).sort((left, right) => right.count - left.count || left.domain.localeCompare(right.domain));

  return {
    projection: {
      status: safeDisplayValue(projectionRecord.status) || undefined,
      change_class: safeDisplayValue(projectionRecord.change_class) || undefined,
      visibility: safeDisplayValue(projectionRecord.visibility) || undefined,
      generated_at_utc: safeDisplayValue(projectionRecord.generated_at_utc) || undefined,
      summary: {
        alignment_score: typeof summary.alignment_score === 'number' ? summary.alignment_score : undefined,
        blocker_pressure: typeof summary.blocker_pressure === 'number' ? summary.blocker_pressure : undefined,
        execution_clarity: typeof summary.execution_clarity === 'number' ? summary.execution_clarity : undefined,
        readability_pressure: typeof summary.readability_pressure === 'number' ? summary.readability_pressure : undefined,
        drift_state: safeDisplayValue(summary.drift_state) || undefined,
        headline: safeDisplayValue(summary.headline) || undefined,
      },
      events,
      actions,
      trends,
      anchors,
    },
    domainGroups,
  };
}

function buildPublicAcceptance(snapshot: DashboardSnapshot, surface: SurfaceView): {
  summary: PublicAcceptanceSummary | null;
  checks: PublicAcceptanceCheckRow[];
  subcommands: PublicAcceptanceSubcommandRow[];
} {
  const entry = artifactEntry(snapshot, 'dashboard_public_acceptance');
  const payload = toRecord<Dict>(entry?.payload) || {};
  if (!entry && !Object.keys(payload).length) {
    return {
      summary: null,
      checks: [],
      subcommands: [],
    };
  }

  const checks = toRecord<Dict>(payload.checks) || {};
  const topology = toRecord<Dict>(checks.topology_smoke) || {};
  const topologyChecks = toRecord<Dict>(topology.checks) || {};
  const topologyTlsPolicy = toRecord<Dict>(topology.tls_policy) || {};
  const topologyRootSnapshot = toRecord<Dict>(topologyChecks.root_snapshot) || {};
  const topologyRootPublic = toRecord<Dict>(topologyChecks.root_public) || {};
  const topologyRootOverviewBrowser = toRecord<Dict>(topologyChecks.root_overview_browser) || {};
  const topologyPagesOverviewBrowser = toRecord<Dict>(topologyChecks.pages_overview_browser) || {};
  const topologyRootContractsBrowser = toRecord<Dict>(topologyChecks.root_contracts_browser) || {};
  const topologyPagesContractsBrowser = toRecord<Dict>(topologyChecks.pages_contracts_browser) || {};
  const topologyEntrypoints = toRecord<Dict>(topology.entrypoints) || {};
  const workspaceRoutes = toRecord<Dict>(checks.workspace_routes_smoke) || {};
  const workspaceSurface = toRecord<Dict>(workspaceRoutes.surface_assertion) || {};
  const workspaceNetwork = toRecord<Dict>(workspaceRoutes.network_observation) || {};
  const workspaceArtifactsFilter = toRecord<Dict>(workspaceRoutes.artifacts_filter_assertion) || {};
  const subcommands = toRecord<Dict>(payload.subcommands) || {};
  const orderflowVisibleArtifacts = toArray<string>(workspaceArtifactsFilter.visible_artifacts)
    .map((value) => safeDisplayValue(value))
    .filter((value) => value && value !== '—')
    .join(' ｜ ');

  const summary: PublicAcceptanceSummary = {
    artifact_id: 'dashboard_public_acceptance',
    status: safeDisplayValue(payload.status || entry?.summary?.status),
    change_class: safeDisplayValue(payload.change_class || entry?.summary?.change_class),
    generated_at_utc: safeDisplayValue(payload.generated_at_utc || entry?.summary?.generated_at_utc),
    report_path: safeDisplayValue(payload.report_path),
    artifact_path: safeDisplayValue(entry?.path),
    topology_status: safeDisplayValue(topology.status),
    workspace_status: safeDisplayValue(workspaceRoutes.status),
    topology_generated_at_utc: safeDisplayValue(topology.generated_at_utc),
    workspace_generated_at_utc: safeDisplayValue(workspaceRoutes.generated_at_utc),
    frontend_public: safeDisplayValue(topologyRootSnapshot.frontend_public),
    root_public_entry: safeDisplayValue(topologyRootPublic.header_public_entry),
    public_snapshot_fetch_count: typeof workspaceNetwork.public_snapshot_fetch_count === 'number' ? workspaceNetwork.public_snapshot_fetch_count : undefined,
    internal_snapshot_fetch_count: typeof workspaceNetwork.internal_snapshot_fetch_count === 'number' ? workspaceNetwork.internal_snapshot_fetch_count : undefined,
    root_contracts_screenshot_path: safeDisplayValue(topologyRootContractsBrowser.screenshot_path),
    pages_contracts_screenshot_path: safeDisplayValue(topologyPagesContractsBrowser.screenshot_path),
    orderflow_filter_route: safeDisplayValue(workspaceArtifactsFilter.route),
    orderflow_active_artifact: safeDisplayValue(workspaceArtifactsFilter.active_artifact),
    orderflow_visible_artifacts: orderflowVisibleArtifacts || undefined,
  };

  const checkRows: PublicAcceptanceCheckRow[] = [
    {
      id: 'topology_smoke',
      label: t('workspace_contracts_acceptance_topology_label'),
      status: safeDisplayValue(topology.status),
      ok: Boolean(topology.ok),
      generated_at_utc: safeDisplayValue(topology.generated_at_utc),
      report_path: safeDisplayValue(topology.report_path),
      returncode: typeof topology.returncode === 'number' ? topology.returncode : undefined,
      failure_reason: safeDisplayValue(topology.failure_reason),
      headline: [safeDisplayValue(topologyEntrypoints.root_url), safeDisplayValue(topologyEntrypoints.pages_url)].filter((value) => value && value !== '—').join(' ｜ '),
      frontend_public: safeDisplayValue(topologyRootSnapshot.frontend_public),
      tls_fallback_used: typeof topologyTlsPolicy.allow_insecure_tls_fallback === 'boolean' ? topologyTlsPolicy.allow_insecure_tls_fallback : undefined,
      root_public_entry: safeDisplayValue(topologyRootPublic.header_public_entry),
      root_overview_screenshot_path: safeDisplayValue(topologyRootOverviewBrowser.screenshot_path),
      pages_overview_screenshot_path: safeDisplayValue(topologyPagesOverviewBrowser.screenshot_path),
      root_contracts_screenshot_path: safeDisplayValue(topologyRootContractsBrowser.screenshot_path),
      pages_contracts_screenshot_path: safeDisplayValue(topologyPagesContractsBrowser.screenshot_path),
      ...(surface.effective === 'internal' && typeof topology.stdout === 'string' ? { stdout: topology.stdout } : {}),
      ...(surface.effective === 'internal' && typeof topology.stderr === 'string' ? { stderr: topology.stderr } : {}),
    },
    {
      id: 'workspace_routes_smoke',
      label: t('workspace_contracts_acceptance_workspace_label'),
      status: safeDisplayValue(workspaceRoutes.status),
      ok: Boolean(workspaceRoutes.ok),
      generated_at_utc: safeDisplayValue(workspaceRoutes.generated_at_utc),
      report_path: safeDisplayValue(workspaceRoutes.report_path),
      returncode: typeof workspaceRoutes.returncode === 'number' ? workspaceRoutes.returncode : undefined,
      failure_reason: safeDisplayValue(workspaceRoutes.failure_reason),
      headline: safeDisplayValue(workspaceSurface.snapshot_endpoint_observed),
      requested_surface: safeDisplayValue(workspaceSurface.requested_surface),
      effective_surface: safeDisplayValue(workspaceSurface.effective_surface),
      snapshot_endpoint_observed: safeDisplayValue(workspaceSurface.snapshot_endpoint_observed),
      public_snapshot_fetch_count: typeof workspaceNetwork.public_snapshot_fetch_count === 'number' ? workspaceNetwork.public_snapshot_fetch_count : undefined,
      internal_snapshot_fetch_count: typeof workspaceNetwork.internal_snapshot_fetch_count === 'number' ? workspaceNetwork.internal_snapshot_fetch_count : undefined,
      orderflow_filter_route: safeDisplayValue(workspaceArtifactsFilter.route),
      orderflow_active_artifact: safeDisplayValue(workspaceArtifactsFilter.active_artifact),
      orderflow_visible_artifacts: orderflowVisibleArtifacts || undefined,
      ...(surface.effective === 'internal' && typeof workspaceRoutes.stdout === 'string' ? { stdout: workspaceRoutes.stdout } : {}),
      ...(surface.effective === 'internal' && typeof workspaceRoutes.stderr === 'string' ? { stderr: workspaceRoutes.stderr } : {}),
    },
  ].filter((row) => row.status && row.status !== '—');

  const subcommandRows: PublicAcceptanceSubcommandRow[] = Object.entries(subcommands).map(([id, value]) => {
    const row = toRecord<Dict>(value) || {};
    return {
      id,
      label: id === 'topology_smoke' ? t('workspace_contracts_acceptance_topology_command_label') : t('workspace_contracts_acceptance_workspace_command_label'),
      returncode: typeof row.returncode === 'number' ? row.returncode : undefined,
      payload_present: typeof row.payload_present === 'boolean' ? row.payload_present : undefined,
      stdout_bytes: typeof row.stdout_bytes === 'number' ? row.stdout_bytes : undefined,
      stderr_bytes: typeof row.stderr_bytes === 'number' ? row.stderr_bytes : undefined,
      cmd: Array.isArray(row.cmd) ? row.cmd.map((item) => safeDisplayValue(item)).join(' ') : safeDisplayValue(row.cmd),
      cwd: safeDisplayValue(row.cwd),
      ...(surface.effective === 'internal' && typeof row.stdout === 'string' ? { stdout: row.stdout } : {}),
      ...(surface.effective === 'internal' && typeof row.stderr === 'string' ? { stderr: row.stderr } : {}),
    };
  });

  return {
    summary,
    checks: checkRows,
    subcommands: subcommandRows,
  };
}

function allowsMetricStrip(intent: DisplayIntent, section: string): boolean {
  return Boolean(intent.metricStrips[section]);
}

function allowsField(map: DisplayIntent['tables'], section: string, key: string): boolean {
  return Boolean(map[section]?.[key] || map[section]?.['*']);
}

function field(
  key: string,
  labelKey?: string,
  options: Partial<Pick<ViewFieldSchema, 'showRaw' | 'kind'>> = {},
): ViewFieldSchema {
  return {
    key,
    label: t(labelKey || key),
    ...options,
  };
}

function column(
  key: string,
  labelKey?: string,
  options: Partial<Pick<ViewColumnSchema, 'showRaw' | 'kind'>> = {},
): ViewColumnSchema {
  return field(key, labelKey, options);
}

function drilldown(
  titleKey: string,
  statusKey: string | undefined,
  summaryKeys: string[],
  fields: ViewFieldSchema[],
  defaultOpenCount = 1,
  rowIdentityKeys?: string[],
): ViewDrilldownSchema {
  return { titleKey, statusKey, summaryKeys, fields, defaultOpenCount, rowIdentityKeys };
}

function buildViewSchema(intent: DisplayIntent): TerminalViewSchema {
  const showMetric = (section: string) => allowsMetricStrip(intent, section);
  const showTable = (section: string, key: string) => allowsField(intent.tables, section, key);
  const showDrilldown = (section: string, key: string) => allowsField(intent.drilldownFields, section, key);
  const showKeyValue = (section: string, key: string) => allowsField(intent.keyValueFields, section, key);
  const showCompactPath = (section: string, key: string) => allowsField(intent.compactPathFields, section, key);
  const showValueText = (section: string, key: string) => allowsField(intent.valueTextFields, section, key);
  return {
    terminal: {
      orchestration: {
        topMetricsStrip: {
          showRawValues: showMetric('terminal.orchestration.topMetrics'),
        },
        guardsStrip: {
          showRawValues: showMetric('terminal.orchestration.guards'),
        },
        actionLogColumns: [
          column('ts', 'timestamp'),
          column('label', 'slot_trigger'),
          column('status', 'status', { kind: 'badge' }),
          column('detail', 'summary', { showRaw: showTable('terminal.orchestration.actionLog', 'detail') }),
        ],
        checklistColumns: [
          column('ts', 'source_as_of'),
          column('label', 'action', { showRaw: showTable('terminal.orchestration.checklist', 'label') }),
          column('status', 'status', { kind: 'badge' }),
          column('detail', 'checklist_detail', { showRaw: showTable('terminal.orchestration.checklist', 'detail') }),
        ],
      },
      dataRegime: {
        sourceConfidenceStrip: {
          showRawValues: showMetric('terminal.dataRegime.sourceConfidence'),
        },
        microCaptureStrip: {
          showRawValues: showMetric('terminal.dataRegime.microCapture'),
        },
        regimeSummaryColumns: [
          column('mode'),
          column('robustness_score', 'robustness_score', { kind: 'number' }),
          column('avg_annual_return', 'annual_return', { kind: 'percent' }),
          column('worst_drawdown', 'worst_drawdown', { kind: 'percent' }),
          column('avg_profit_factor', 'pf', { kind: 'number' }),
        ],
        regimeMatrixColumns: [
          column('mode'),
          column('window'),
          column('status', 'status', { kind: 'badge' }),
          column('annual_return', 'annual_return', { kind: 'percent' }),
          column('max_drawdown', 'max_drawdown', { kind: 'percent' }),
        ],
      },
      signalRisk: {
        gateScoresStrip: {
          showRawValues: showMetric('terminal.signalRisk.gateScores'),
        },
        focusSlotsTableColumns: [
          column('slot'),
          column('symbol'),
          column('action', 'action', { showRaw: showTable('terminal.signalRisk.focusSlots', 'action') }),
          column('state', 'status', { kind: 'badge' }),
          column('done_when', 'done_when', { showRaw: showTable('terminal.signalRisk.focusSlots', 'done_when') }),
        ],
        focusSlotsDrilldown: drilldown(
          'slot',
          'state',
          ['symbol', 'action', 'reason'],
          [
            field('symbol'),
            field('action', 'action', { showRaw: showDrilldown('terminal.signalRisk.focusSlots', 'action') }),
            field('area'),
            field('target', 'target', { showRaw: showDrilldown('terminal.signalRisk.focusSlots', 'target') }),
            field('reason', 'reason', { showRaw: showDrilldown('terminal.signalRisk.focusSlots', 'reason') }),
            field('blocker_detail', 'blocker_detail', { showRaw: showDrilldown('terminal.signalRisk.focusSlots', 'blocker_detail') }),
            field('done_when', 'done_when', { showRaw: showDrilldown('terminal.signalRisk.focusSlots', 'done_when') }),
            field('priority_score', 'priority_score', { kind: 'number' }),
            field('queue_rank', 'queue_rank', { kind: 'number' }),
            field('source_artifact', 'source_artifact', { showRaw: showDrilldown('terminal.signalRisk.focusSlots', 'source_artifact') }),
            field('source_status'),
            field('source_refresh_action', 'source_refresh_action', { showRaw: showDrilldown('terminal.signalRisk.focusSlots', 'source_refresh_action') }),
          ],
          2,
          ['slot'],
        ),
        controlChainTableColumns: [
          column('label', 'stage'),
          column('source_status', 'status', { kind: 'badge' }),
          column('source_decision', 'decision', { showRaw: showTable('terminal.signalRisk.controlChain', 'source_decision') }),
          column('next_target_artifact', 'next_target_artifact', { showRaw: showTable('terminal.signalRisk.controlChain', 'next_target_artifact') }),
        ],
        controlChainDrilldown: drilldown(
          'label',
          'source_status',
          ['source_decision', 'next_target_artifact'],
          [
            field('mission', 'mission', { showRaw: showDrilldown('terminal.signalRisk.controlChain', 'mission') }),
            field('source_brief', 'source_brief', { showRaw: showDrilldown('terminal.signalRisk.controlChain', 'source_brief') }),
            field('source_decision', 'source_decision', { showRaw: showDrilldown('terminal.signalRisk.controlChain', 'source_decision') }),
            field('source_artifact', 'source_artifact', { showRaw: showDrilldown('terminal.signalRisk.controlChain', 'source_artifact') }),
            field('next_target_artifact', 'next_target_artifact', { showRaw: showDrilldown('terminal.signalRisk.controlChain', 'next_target_artifact') }),
            field('blocking_reason', 'blocking_reason', { showRaw: showDrilldown('terminal.signalRisk.controlChain', 'blocking_reason') }),
            field('interface_fields'),
            field('optimization_brief', 'optimization_brief', { showRaw: showDrilldown('terminal.signalRisk.controlChain', 'optimization_brief') }),
          ],
          2,
          ['stage'],
        ),
        actionQueueTableColumns: [
          column('symbol'),
          column('action', 'action', { showRaw: showTable('terminal.signalRisk.actionQueue', 'action') }),
          column('state', 'status', { kind: 'badge' }),
          column('reason', 'reason', { showRaw: showTable('terminal.signalRisk.actionQueue', 'reason') }),
        ],
        actionQueueDrilldown: drilldown(
          'symbol',
          'state',
          ['action', 'reason'],
          [
            field('action', 'action', { showRaw: showDrilldown('terminal.signalRisk.actionQueue', 'action') }),
            field('reason', 'reason', { showRaw: showDrilldown('terminal.signalRisk.actionQueue', 'reason') }),
            field('target', 'target', { showRaw: showDrilldown('terminal.signalRisk.actionQueue', 'target') }),
            field('done_when', 'done_when', { showRaw: showDrilldown('terminal.signalRisk.actionQueue', 'done_when') }),
            field('priority_score', 'priority_score', { kind: 'number' }),
            field('source_artifact', 'source_artifact', { showRaw: showDrilldown('terminal.signalRisk.actionQueue', 'source_artifact') }),
          ],
          2,
          ['symbol', 'action'],
        ),
        repairPlanStrip: {
          showRawValues: showMetric('terminal.signalRisk.repairPlan'),
        },
        optimizationBacklogColumns: [
          column('stage'),
          column('title', 'title', { showRaw: showTable('terminal.signalRisk.optimizationBacklog', 'title') }),
          column('change_class', 'change_class', { kind: 'badge' }),
          column('why', 'why', { showRaw: showTable('terminal.signalRisk.optimizationBacklog', 'why') }),
        ],
      },
      labReview: {
        researchHeadsStrip: {
          showRawValues: showMetric('terminal.labReview.researchHeads'),
        },
        manifestsStrip: {
          showRawValues: showMetric('terminal.labReview.manifests'),
        },
        backtestPoolColumns: [
          column('label', 'backtest'),
          column('start'),
          column('end'),
          column('total_return', 'total_return', { kind: 'percent' }),
          column('max_drawdown', 'max_drawdown', { kind: 'percent' }),
          column('trades', 'trades', { kind: 'number' }),
        ],
        comparisonColumns: [
          column('strategy_id_canonical'),
          column('market_scope'),
          column('total_return', 'total_return', { kind: 'percent' }),
          column('profit_factor', 'profit_factor', { kind: 'number' }),
          column('recommendation', 'recommendation', { kind: 'badge' }),
        ],
        candidateColumns: [
          column('label', 'artifact'),
          column('status', 'status', { kind: 'badge' }),
          column('decision', 'research_decision', { showRaw: showTable('terminal.labReview.candidateRows', 'decision') }),
          column('takeaway', 'conclusion', { showRaw: showTable('terminal.labReview.candidateRows', 'takeaway') }),
        ],
        stressSummaryColumns: [
          column('mode'),
          column('avg_win_rate', 'win_rate', { kind: 'percent' }),
          column('avg_profit_factor', 'pf', { kind: 'number' }),
          column('total_violations', 'total_violations', { kind: 'number' }),
          column('robustness_score', 'robustness_score', { kind: 'number' }),
        ],
      },
    },
    workspace: {
      artifacts: {
        list: {
          titleField: field('title', 'title', { showRaw: showValueText('workspace.artifacts.list', 'title') }),
          pathField: field('path', 'path', { showRaw: showCompactPath('workspace.artifacts.list', 'path'), kind: 'path' }),
        },
        inspector: {
          titleField: field('title', 'title', { showRaw: showValueText('workspace.artifacts.inspector', 'title') }),
          pathField: field('path', 'path', { showRaw: showCompactPath('workspace.artifacts.inspector', 'path'), kind: 'path' }),
        },
        metaFields: [
          field('category', 'category', { showRaw: showKeyValue('workspace.artifacts.meta', 'category') }),
          field('status', 'status', { showRaw: showKeyValue('workspace.artifacts.meta', 'status') }),
          field('change_class', 'change_class', { showRaw: showKeyValue('workspace.artifacts.meta', 'change_class') }),
          field('generated_at_utc', 'generated_at_utc', { showRaw: showKeyValue('workspace.artifacts.meta', 'generated_at_utc') }),
          field('payload_key', 'payload_key', { showRaw: showKeyValue('workspace.artifacts.meta', 'payload_key') }),
        ],
        topLevelKeyField: field('*', '*', { showRaw: showValueText('workspace.artifacts.topLevelKeys', '*') }),
        payloadField: field('*', '*', { showRaw: showKeyValue('workspace.artifacts.payload', '*') }),
      },
      backtests: {
        poolColumns: [
          column('label', 'backtest'),
          column('equity_curve_sample', 'curve', { kind: 'sparkline' }),
          column('total_return', 'total_return', { kind: 'percent' }),
          column('annual_return', 'annual_return', { kind: 'percent' }),
          column('max_drawdown', 'max_drawdown', { kind: 'percent' }),
          column('profit_factor', 'profit_factor', { kind: 'number' }),
        ],
        comparisonColumns: [
          column('strategy_id_canonical', 'strategy_id_canonical', { showRaw: showTable('workspace.backtests.comparison', 'strategy_id_canonical') }),
          column('market_scope', 'market_scope', { showRaw: showTable('workspace.backtests.comparison', 'market_scope') }),
          column('trade_count', 'trade_count', { kind: 'number' }),
          column('win_rate', 'win_rate', { kind: 'percent' }),
          column('expectancy_r', 'expectancy_r', { kind: 'number' }),
          column('recommendation', 'recommendation', { kind: 'badge' }),
        ],
      },
      contracts: {
        surfaceAssertionFields: [
          field('id', 'surface'),
          field('availability'),
          field('redaction_level'),
          field('snapshot_path'),
        ],
        surfacePublishFields: [
          field('route_key'),
          field('deploy_policy'),
          field('fallback_snapshot_path'),
          field('notes', 'notes', { showRaw: false }),
        ],
        interfaceColumns: [
          column('label', 'interface_entry'),
          column('url', 'entry_route', { kind: 'path' }),
          column('description', 'description', { showRaw: showTable('workspace.contracts.interfaces', 'description') }),
        ],
        sourceHeadSummaryFields: {
          label: field('label', 'label', { showRaw: showValueText('workspace.contracts.sourceHeads.summary', 'label') }),
          summary: field('summary', 'summary', { showRaw: showValueText('workspace.contracts.sourceHeads.summary', 'summary') }),
        },
        sourceHeadMetricFields: {
          metrics: [
            field('label', 'label', { showRaw: showValueText('workspace.contracts.sourceHeads.metrics', 'label') }),
            field('status', 'status', { showRaw: showValueText('workspace.contracts.sourceHeads.metrics', 'status') }),
            field('path', 'path', { showRaw: showValueText('workspace.contracts.sourceHeads.metrics', 'path') }),
          ],
        },
        sourceHeadDetailFields: [
          field('status', 'status', { showRaw: showKeyValue('workspace.contracts.sourceHeads', 'status') }),
          field('research_decision', 'research_decision', { showRaw: showKeyValue('workspace.contracts.sourceHeads', 'research_decision') }),
          field('summary', 'summary', { showRaw: showKeyValue('workspace.contracts.sourceHeads', 'summary') }),
          field('active_baseline'),
          field('local_candidate'),
          field('transfer_watch'),
          field('return_candidate'),
          field('return_watch'),
          field('generated_at_utc'),
          field('path', 'path', { showRaw: showKeyValue('workspace.contracts.sourceHeads', 'path') }),
        ],
      },
    },
  };
}

export function buildTerminalReadModel(loaded: LoadedSurface): TerminalReadModel {
  const { snapshot } = loaded;
  const surface = buildSurfaceView(snapshot, loaded);
  const displayIntent = resolveDisplayIntent(surface.effective);
  const terminalNavigation = buildTerminalWorkspaceHandoffs(snapshot, surface);
  return {
    surface,
    meta: snapshot.meta || {},
    uiRoutes: snapshot.ui_routes || {},
    experienceContract: snapshot.experience_contract || {},
    navigation: {
      terminalPrimaryArtifact: terminalNavigation.primaryArtifact,
      terminalHandoffs: terminalNavigation.handoffs,
    },
    view: buildViewSchema(displayIntent),
    orchestration: buildOrchestration(snapshot, surface),
    dataRegime: buildDataRegime(snapshot),
    signalRisk: buildSignalRisk(snapshot, surface),
    labReview: buildLabReview(snapshot),
    feedback: buildFeedback(snapshot),
    workspace: buildWorkspace(snapshot, surface),
  };
}
