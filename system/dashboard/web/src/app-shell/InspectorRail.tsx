import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { buildOverviewLink } from '../navigation/route-contract';
import { buildSearchLink } from '../search/links';
import type { SurfaceKey, TerminalReadModel } from '../types/contracts';
import { Badge, DrilldownSection, KeyValueGrid } from '../components/ui-kit';
import { buildTerminalFocusKey, buildTerminalLink, buildWorkspaceLink, buildWorkspacePageLink, type SharedFocusState, type WorkspaceSection } from '../utils/focus-links';
import { safeDisplayValue } from '../utils/formatters';
import { labelFor } from '../utils/dictionary';

type InspectorContext = {
  domain: 'overview' | 'terminal' | 'workspace';
  surface?: SurfaceKey;
  section?: WorkspaceSection;
  activeSectionId?: string;
};

type InspectorRailProps = {
  title?: string;
  focus?: SharedFocusState;
  model?: TerminalReadModel | null;
  context?: InspectorContext;
  mode?: 'rail' | 'drawer' | 'inline';
  collapsed?: boolean;
  shellTier?: 'xl' | 'l' | 'm' | 's';
};

type InspectorLink = {
  id: string;
  label: string;
  to: string;
};

type InspectorRows = Array<{
  key: string;
  label: string;
  value: unknown;
  kind?: 'path' | 'badge' | 'number';
  showRaw?: boolean;
}>;

const WORKSPACE_SECTION_LABELS: Record<WorkspaceSection, string> = {
  artifacts: '工件池',
  alignment: '对齐页',
  backtests: '回测池',
  contracts: '契约层',
  raw: '原始层',
};

const ARTIFACT_GROUP_LABELS: Record<string, string> = {
  research_mainline: '研究主线',
  system_anchor: '系统锚点',
  research_exit_risk: '退出与风控',
  research_hold_transfer: '持有迁移',
  research_cross_section: '横截面研究',
  archive: '归档',
};

const SEARCH_SCOPE_LABELS: Record<string, string> = {
  title: '标题',
  artifact: '工件ID',
  path: '路径',
};

function workspaceSectionLabel(section: WorkspaceSection | undefined): string {
  return section ? WORKSPACE_SECTION_LABELS[section] : '工件池';
}

function artifactGroupLabel(group: unknown): string {
  const raw = safeDisplayValue(group);
  return ARTIFACT_GROUP_LABELS[raw] || raw || '—';
}

function searchScopeLabel(scope: unknown): string {
  const raw = safeDisplayValue(scope);
  return SEARCH_SCOPE_LABELS[raw] || raw || '标题';
}

function normalizedInspectorRef(value: unknown): string {
  return safeDisplayValue(value).trim().toLowerCase();
}

function artifactRefMatches(row: NonNullable<TerminalReadModel>['workspace']['artifactRows'][number], candidate: unknown): boolean {
  const target = normalizedInspectorRef(candidate);
  if (!target || target === '—') return false;
  return [row.id, row.payload_key, row.label, row.path]
    .map((value) => normalizedInspectorRef(value))
    .some((value) => value && value !== '—' && (value === target || value.includes(target) || target.includes(value)));
}

function sourceHeadRefMatches(row: NonNullable<TerminalReadModel>['workspace']['sourceHeads'][number], candidate: unknown): boolean {
  const target = normalizedInspectorRef(candidate);
  if (!target || target === '—') return false;
  return [row.id, row.label, row.path, row.locator]
    .map((value) => normalizedInspectorRef(value))
    .some((value) => value && value !== '—' && (value === target || value.includes(target) || target.includes(value)));
}

function resolveArtifactId(model: TerminalReadModel | null | undefined, focus: SharedFocusState | undefined, context: InspectorContext | undefined): string {
  if (focus?.artifact) return safeDisplayValue(focus.artifact);
  if (!model || context?.domain !== 'terminal') return '';
  const focusKey = buildTerminalFocusKey({
    panel: focus?.panel,
    section: focus?.section,
    row: focus?.row,
  });
  const sectionKey = buildTerminalFocusKey({
    panel: focus?.panel,
    section: focus?.section,
  });
  return safeDisplayValue(model.navigation.terminalPrimaryArtifact[focusKey] || model.navigation.terminalPrimaryArtifact[sectionKey]);
}

function dedupeLinks(links: InspectorLink[]): InspectorLink[] {
  const seen = new Set<string>();
  return links.filter((link) => {
    const key = `${link.label}::${link.to}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function optionalInspectorValue(value: unknown): string {
  const raw = safeDisplayValue(value);
  return raw === '—' ? '' : raw;
}

export function InspectorRail({
  title = '对象检查器',
  focus,
  model,
  context = { domain: 'overview', surface: 'public' },
  mode = 'rail',
  collapsed = false,
  shellTier = 'l',
}: InspectorRailProps) {
  if (!model) {
    return (
      <div className="inspector-rail-inner" data-mode={mode} data-shell-tier={shellTier} data-collapsed={collapsed ? 'true' : 'false'}>
        <div className="inspector-title-block">
          <h3>{title}</h3>
          <p>暂无可检查对象，先在主舞台选择工件或终端焦点。</p>
        </div>
      </div>
    );
  }

  const artifactId = resolveArtifactId(model, focus, context);
  const artifactRow = artifactId ? model?.workspace.artifactRows.find((row) => artifactRefMatches(row, artifactId)) : null;
  const canonicalArtifactId = safeDisplayValue(artifactRow?.id || artifactId);
  const sourceHeadCandidates = [
    artifactId,
    artifactRow?.id,
    artifactRow?.payload_key,
    artifactRow?.path,
    artifactRow?.label,
  ];
  const sourceHead = sourceHeadCandidates.length
    ? model?.workspace.sourceHeads.find((row) => sourceHeadCandidates.some((candidate) => sourceHeadRefMatches(row, candidate)))
    : null;
  const snapshotPath = safeDisplayValue(model?.surface.snapshotPath);
  const fallbackChain = model?.workspace.fallbackChain || [];
  const focusKey = buildTerminalFocusKey({
    panel: focus?.panel,
    section: focus?.section,
    row: focus?.row,
  });
  const genericSummaryRows: InspectorRows = [
    { key: 'artifact', label: '工件', value: artifactId || '—' },
    { key: 'panel', label: '面板', value: focus?.panel || '—' },
    { key: 'section', label: '分层', value: focus?.section || '—' },
    { key: 'row', label: '行焦点', value: focus?.row || '—' },
    { key: 'surface', label: '当前视图', value: model?.surface.effectiveLabel || '—' },
    { key: 'change-class', label: '变更级别', value: model?.meta.change_class || 'RESEARCH_ONLY' },
  ];
  const genericEvidenceRows: InspectorRows = [
    { key: 'artifact-path', label: '工件路径', value: artifactRow?.path || '—', kind: 'path' as const, showRaw: true },
    { key: 'source-head-path', label: '源头路径', value: sourceHead?.path || '—', kind: 'path' as const, showRaw: true },
    { key: 'snapshot-path', label: '快照路径', value: snapshotPath, kind: 'path' as const, showRaw: true },
    { key: 'decision', label: '研究决策', value: artifactRow?.research_decision || artifactRow?.status || sourceHead?.status || '—', kind: 'badge' as const },
  ];

  const baseFocus: SharedFocusState = {
    ...focus,
    artifact: canonicalArtifactId || focus?.artifact,
  };

  const drillBackLinks: InspectorLink[] = [
    {
      id: 'back-overview',
      label: '返回总览',
      to: buildOverviewLink(baseFocus),
    },
  ];

  if (context.domain === 'terminal') {
    drillBackLinks.push({
      id: 'back-terminal',
      label: '返回操作终端',
      to: buildTerminalLink(context.surface || 'public', baseFocus),
    });
  }
  if (context.domain === 'workspace') {
    drillBackLinks.push({
      id: 'back-workspace',
      label: '返回研究工作区',
      to: buildWorkspaceLink(context.section || 'artifacts', baseFocus),
    });
  }

  const activeContractsCheckId = context.domain === 'workspace'
    && context.section === 'contracts'
    && context.activeSectionId?.startsWith('contracts-check-')
    ? context.activeSectionId.slice('contracts-check-'.length)
    : undefined;
  const activeContractsSubcommandId = context.domain === 'workspace'
    && context.section === 'contracts'
    && context.activeSectionId?.startsWith('contracts-subcommand-')
    ? context.activeSectionId.slice('contracts-subcommand-'.length)
    : undefined;
  const activeAcceptanceCheck = activeContractsCheckId
    ? model.workspace.publicAcceptance.checks.find((row) => row.id === activeContractsCheckId) || null
    : null;
  const activeAcceptanceSubcommand = activeContractsSubcommandId
    ? model.workspace.publicAcceptance.subcommands.find((row) => row.id === activeContractsSubcommandId) || null
    : null;
  const linkedAcceptanceCheck = activeContractsSubcommandId
    ? model.workspace.publicAcceptance.checks.find((row) => row.id === activeContractsSubcommandId) || null
    : null;
  const primaryResearchAuditCase = activeAcceptanceCheck?.research_audit_cases?.[0]
    || linkedAcceptanceCheck?.research_audit_cases?.[0]
    || null;
  const readonlyLinks: InspectorLink[] = [];
  if (canonicalArtifactId) {
    readonlyLinks.push({
      id: 'view-artifacts',
      label: '查看工件池',
      to: buildWorkspaceLink('artifacts', { artifact: canonicalArtifactId }),
    });
    readonlyLinks.push({
      id: 'view-raw',
      label: '查看原始层',
      to: buildWorkspaceLink('raw', { artifact: canonicalArtifactId }),
    });
  }
  if (canonicalArtifactId) {
    for (const link of model?.workspace.artifactHandoffs[canonicalArtifactId] || []) {
      readonlyLinks.push({
        id: link.id,
        label: link.label,
        to: link.to,
      });
    }
  }
  if (context.domain === 'terminal' && model && !canonicalArtifactId) {
    const sectionKey = buildTerminalFocusKey({
      panel: focus?.panel,
      section: focus?.section,
    });
    for (const link of model.navigation.terminalHandoffs[focusKey] || model.navigation.terminalHandoffs[sectionKey] || []) {
      readonlyLinks.push({
        id: link.id,
        label: link.label,
        to: link.to,
      });
    }
  }

  let focusTitle = '当前焦点';
  let focusSummary = artifactId || focus?.panel || '未选中对象';
  let focusRows = genericSummaryRows;
  let focusMeta: ReactNode = <Badge value={model?.surface.effective || context.surface}>{model?.surface.effectiveLabel || '公开终端'}</Badge>;
  let sourceTitle = '追源与来源';
  let sourceSummary = safeDisplayValue(artifactRow?.label || sourceHead?.label || fallbackChain[0] || snapshotPath);
  let sourceRows = genericEvidenceRows;
  let sourceExtra: ReactNode = fallbackChain.length ? (
    <div className="inspector-fallback-strip">
      <span>回退链</span>
      <strong>{fallbackChain.join(' → ')}</strong>
    </div>
  ) : null;
  let titleDescription = artifactId
    ? '围绕当前焦点显示 source-owned 摘要、回退链与证据跳转。'
    : '未选中对象时显示域摘要与快照来源。';

  if (context.domain === 'terminal') {
    const terminalLocationSummary = [
      focus?.panel ? labelFor(focus.panel) : null,
      focus?.section ? labelFor(focus.section) : null,
      focus?.row ? labelFor(focus.row) : null,
    ].filter(Boolean).join(' / ');
    focusTitle = '终端定位';
    focusSummary = terminalLocationSummary || safeDisplayValue(artifactId || model?.surface.effectiveLabel || '公开终端');
    focusRows = [
      { key: 'panel', label: '当前面板', value: focus?.panel || '未选择' },
      { key: 'section', label: '当前分层', value: focus?.section || '未选择' },
      { key: 'row', label: '当前行焦点', value: focus?.row || '未选择' },
      { key: 'artifact', label: '关联工件', value: artifactId || '未绑定' },
      { key: 'focus-key', label: '穿透键', value: focusKey || '未生成' },
      { key: 'handoffs', label: '可跳转动作', value: dedupeLinks(readonlyLinks).length, kind: 'number' },
    ];
    sourceTitle = artifactRow || sourceHead ? '执行证据' : '下一跳建议';
    sourceSummary = safeDisplayValue(sourceHead?.label || artifactRow?.label || focus?.section || focus?.panel || '等待穿透定位');
    sourceRows = artifactRow || sourceHead
      ? [
          { key: 'artifact-path', label: '工件路径', value: artifactRow?.path || '—', kind: 'path', showRaw: true },
          { key: 'source-head-path', label: '源头路径', value: sourceHead?.path || '—', kind: 'path', showRaw: true },
          { key: 'payload-key', label: 'Payload 键', value: artifactRow?.payload_key || artifactId || '—' },
          { key: 'decision', label: '当前判定', value: artifactRow?.research_decision || artifactRow?.status || sourceHead?.status || '—', kind: 'badge' },
        ]
      : [
          { key: 'panel', label: '下一步面板', value: focus?.panel || '先从左侧选择面板' },
          { key: 'section', label: '下一步分层', value: focus?.section || '再进入分层/段落' },
          { key: 'row', label: '下一步行焦点', value: focus?.row || '必要时再落到行焦点' },
          { key: 'handoffs', label: '证据动作', value: dedupeLinks(readonlyLinks).length, kind: 'number' },
        ];
    sourceExtra = null;
    titleDescription = artifactId
      ? '围绕终端定位显示穿透键、执行证据与只读跳转。'
      : '未选中对象时显示终端定位、下一跳建议与只读跳转。';
  }

  if (context.domain === 'workspace' && context.section !== 'contracts' && context.section !== 'raw') {
    const activeGroup = artifactRow?.artifact_group || focus?.group;
    const researchDecision = artifactRow?.research_decision || artifactRow?.status || sourceHead?.status || '—';
    focusTitle = '研究焦点';
    focusSummary = safeDisplayValue(artifactRow?.label || artifactId || workspaceSectionLabel(context.section));
    focusRows = [
      { key: 'artifact', label: '工件', value: artifactId || '—' },
      { key: 'group', label: '所属分组', value: artifactGroupLabel(activeGroup) },
      { key: 'workspace-section', label: '当前阶段', value: workspaceSectionLabel(context.section) },
      { key: 'panel', label: '研究面板', value: focus?.panel || '—' },
      { key: 'workspace-subsection', label: '当前段落', value: focus?.section || '—' },
      { key: 'search-scope', label: '检索范围', value: searchScopeLabel(focus?.search_scope) },
      { key: 'decision', label: '研究决策', value: researchDecision, kind: 'badge' },
    ];
    focusMeta = <Badge value={researchDecision}>{safeDisplayValue(researchDecision)}</Badge>;
    sourceTitle = '研究追源';
    sourceSummary = safeDisplayValue(artifactRow?.label || sourceHead?.label || artifactId || workspaceSectionLabel(context.section));
    sourceRows = [
      { key: 'artifact-path', label: '工件路径', value: artifactRow?.path || '—', kind: 'path', showRaw: true },
      { key: 'source-head-path', label: '源头路径', value: sourceHead?.path || '—', kind: 'path', showRaw: true },
      { key: 'snapshot-path', label: '快照路径', value: snapshotPath, kind: 'path', showRaw: true },
      { key: 'payload-key', label: 'Payload 键', value: artifactRow?.payload_key || artifactId || '—' },
      { key: 'decision', label: '研究决策', value: researchDecision, kind: 'badge' },
    ];
    sourceExtra = fallbackChain.length ? (
      <div className="inspector-fallback-strip">
        <span>研究回退链</span>
        <strong>{fallbackChain.join(' → ')}</strong>
      </div>
    ) : null;
    titleDescription = artifactId
      ? '围绕研究焦点显示主线语义、追源证据与只读回退。'
      : '未选中对象时显示研究阶段摘要、快照来源与只读回退。';
  }

  if (context.domain === 'workspace' && context.section === 'contracts') {
    const acceptanceSummary = model?.workspace.publicAcceptance.summary;
    const acceptanceCheck = model?.workspace.publicAcceptance.checks.find((row) => row.id === 'workspace-routes-smoke')
      || model?.workspace.publicAcceptance.checks[0];
    const topology = model?.workspace.publicTopology.find((row) => normalizedInspectorRef((row as unknown as Record<string, unknown>).role) === 'public')
      || model?.workspace.publicTopology[0];
    const explicitCheckRoute = optionalInspectorValue(activeAcceptanceCheck?.inspector_route);
    const explicitCheckSearchLink = optionalInspectorValue(activeAcceptanceCheck?.inspector_search_link_href);
    const explicitCheckArtifactLink = optionalInspectorValue(activeAcceptanceCheck?.inspector_artifact_link_href);
    const explicitCheckRawLink = optionalInspectorValue(activeAcceptanceCheck?.inspector_raw_link_href);
    const explicitSubcommandRoute = optionalInspectorValue(activeAcceptanceSubcommand?.inspector_route);
    const explicitSubcommandCheckRoute = optionalInspectorValue(activeAcceptanceSubcommand?.inspector_check_route || linkedAcceptanceCheck?.inspector_route);
    const explicitSubcommandSearchLink = optionalInspectorValue(activeAcceptanceSubcommand?.inspector_search_link_href);
    const explicitSubcommandArtifactLink = optionalInspectorValue(activeAcceptanceSubcommand?.inspector_artifact_link_href);
    const explicitSubcommandRawLink = optionalInspectorValue(activeAcceptanceSubcommand?.inspector_raw_link_href);
    if (activeAcceptanceCheck) {
      const isTopologyCheck = activeAcceptanceCheck.id === 'topology_smoke';
      focusTitle = '当前验收项';
      focusSummary = safeDisplayValue(activeAcceptanceCheck.label || activeContractsCheckId || '未选验收项');
      focusRows = [
        { key: 'check', label: '验收项', value: activeAcceptanceCheck.label || '—' },
        { key: 'status', label: '状态', value: activeAcceptanceCheck.status || '—', kind: 'badge' },
        { key: 'generated', label: '生成时间', value: activeAcceptanceCheck.generated_at_utc || '—' },
        { key: 'report-path', label: '报告路径', value: activeAcceptanceCheck.report_path || '—', kind: 'path', showRaw: true },
      ];
      focusMeta = <Badge value={activeAcceptanceCheck.status || 'unknown'}>{safeDisplayValue(activeAcceptanceCheck.status || '未生成')}</Badge>;
      sourceTitle = '验收证据';
      sourceSummary = safeDisplayValue(activeAcceptanceCheck.headline || activeAcceptanceCheck.report_path || activeAcceptanceCheck.label || '—');
      sourceRows = isTopologyCheck
        ? [
            { key: 'report-path', label: '验收报告', value: activeAcceptanceCheck.report_path || '—', kind: 'path', showRaw: true },
            { key: 'frontend-public', label: '公开前端', value: activeAcceptanceCheck.frontend_public || '—', kind: 'path', showRaw: true },
            { key: 'root-public-entry', label: 'root public entry', value: activeAcceptanceCheck.root_public_entry || '—' },
            { key: 'root-overview-screenshot', label: '根总览截图', value: activeAcceptanceCheck.root_overview_screenshot_path || '—', kind: 'path', showRaw: true },
            { key: 'pages-overview-screenshot', label: 'pages 总览截图', value: activeAcceptanceCheck.pages_overview_screenshot_path || '—', kind: 'path', showRaw: true },
            { key: 'root-contracts-screenshot', label: '根契约截图', value: activeAcceptanceCheck.root_contracts_screenshot_path || '—', kind: 'path', showRaw: true },
            { key: 'pages-contracts-screenshot', label: 'pages 契约截图', value: activeAcceptanceCheck.pages_contracts_screenshot_path || '—', kind: 'path', showRaw: true },
            { key: 'snapshot-endpoint', label: '快照端点', value: activeAcceptanceCheck.snapshot_endpoint_observed || snapshotPath, kind: 'path', showRaw: true },
          ]
        : [
            { key: 'report-path', label: '验收报告', value: activeAcceptanceCheck.report_path || '—', kind: 'path', showRaw: true },
            { key: 'research-audit-route', label: '研究审计检索', value: explicitCheckSearchLink || primaryResearchAuditCase?.search_route || activeAcceptanceCheck.research_audit_search_route || '—', kind: 'path', showRaw: true },
            { key: 'research-audit-artifact-route', label: '研究审计工件', value: explicitCheckArtifactLink || primaryResearchAuditCase?.workspace_route || '—', kind: 'path', showRaw: true },
            { key: 'research-audit-raw-route', label: '研究审计原始层', value: explicitCheckRawLink || (primaryResearchAuditCase?.raw_path ? buildWorkspaceLink('raw', { artifact: primaryResearchAuditCase.raw_path }) : '—'), kind: 'path', showRaw: true },
            { key: 'graph-home-route', label: '图谱主页路由', value: activeAcceptanceCheck.graph_home_resolved_route || '—', kind: 'path', showRaw: true },
            { key: 'graph-home-default-center', label: '图谱主页默认中心', value: activeAcceptanceCheck.graph_home_default_center || '—' },
            { key: 'snapshot-endpoint', label: '快照端点', value: activeAcceptanceCheck.snapshot_endpoint_observed || snapshotPath, kind: 'path', showRaw: true },
          ];
      if (!isTopologyCheck && (explicitCheckSearchLink || primaryResearchAuditCase?.search_route)) {
        readonlyLinks.push({
          id: `contracts-check-audit-search-${activeAcceptanceCheck.id}`,
          label: '查看研究审计检索',
          to: explicitCheckSearchLink || buildSearchLink(primaryResearchAuditCase?.query || '', 'artifact'),
        });
      }
      if (!isTopologyCheck && (explicitCheckArtifactLink || primaryResearchAuditCase?.workspace_route)) {
        readonlyLinks.push({
          id: `contracts-check-audit-artifact-${activeAcceptanceCheck.id}`,
          label: '查看研究审计工件',
          to: explicitCheckArtifactLink || primaryResearchAuditCase?.workspace_route || '',
        });
      }
      if (!isTopologyCheck && (explicitCheckRawLink || primaryResearchAuditCase?.raw_path)) {
        readonlyLinks.push({
          id: `contracts-check-audit-raw-${activeAcceptanceCheck.id}`,
          label: '查看研究审计原始层',
          to: explicitCheckRawLink || buildWorkspaceLink('raw', { artifact: primaryResearchAuditCase?.raw_path }),
        });
      }
      if (activeAcceptanceCheck.graph_home_resolved_route) {
        readonlyLinks.push({
          id: `contracts-check-graph-home-${activeAcceptanceCheck.id}`,
          label: '查看图谱主页',
          to: activeAcceptanceCheck.graph_home_resolved_route || '',
        });
      }
      sourceExtra = fallbackChain.length ? (
        <div className="inspector-fallback-strip">
          <span>契约回退链</span>
          <strong>{fallbackChain.join(' → ')}</strong>
        </div>
      ) : null;
      titleDescription = '当前已锁定到契约验收项，显示报告、研究审计入口与相关路由证据。';
    } else if (activeAcceptanceSubcommand) {
      const isTopologySubcommand = activeAcceptanceSubcommand.id === 'topology_smoke';
      focusTitle = '当前子命令';
      focusSummary = safeDisplayValue(activeAcceptanceSubcommand.label || activeContractsSubcommandId || '未选子命令');
      focusRows = [
        { key: 'subcommand', label: '子命令', value: activeAcceptanceSubcommand.label || '—' },
        { key: 'returncode', label: 'returncode', value: activeAcceptanceSubcommand.returncode ?? '—', kind: 'number' },
        { key: 'payload-present', label: 'payload_present', value: activeAcceptanceSubcommand.payload_present ?? '—' },
        { key: 'stdout-bytes', label: 'stdout_bytes', value: activeAcceptanceSubcommand.stdout_bytes ?? '—', kind: 'number' },
        { key: 'stderr-bytes', label: 'stderr_bytes', value: activeAcceptanceSubcommand.stderr_bytes ?? '—', kind: 'number' },
      ];
      focusMeta = <Badge value={typeof activeAcceptanceSubcommand.returncode === 'number' && activeAcceptanceSubcommand.returncode === 0 ? 'ok' : 'warning'}>
        {safeDisplayValue(activeAcceptanceSubcommand.returncode ?? '—')}
      </Badge>;
      sourceTitle = '子命令证据';
      sourceSummary = safeDisplayValue(activeAcceptanceSubcommand.cmd || activeAcceptanceSubcommand.label || '—');
      sourceRows = isTopologySubcommand
        ? [
            { key: 'cmd', label: 'cmd', value: activeAcceptanceSubcommand.cmd || '—', showRaw: true },
            { key: 'cwd', label: 'cwd', value: activeAcceptanceSubcommand.cwd || '—', kind: 'path', showRaw: true },
            { key: 'report-path', label: '关联验收报告', value: linkedAcceptanceCheck?.report_path || '—', kind: 'path', showRaw: true },
            { key: 'root-overview-screenshot', label: '根总览截图', value: linkedAcceptanceCheck?.root_overview_screenshot_path || '—', kind: 'path', showRaw: true },
            { key: 'pages-overview-screenshot', label: 'pages 总览截图', value: linkedAcceptanceCheck?.pages_overview_screenshot_path || '—', kind: 'path', showRaw: true },
            { key: 'root-contracts-screenshot', label: '根契约截图', value: linkedAcceptanceCheck?.root_contracts_screenshot_path || '—', kind: 'path', showRaw: true },
            { key: 'pages-contracts-screenshot', label: 'pages 契约截图', value: linkedAcceptanceCheck?.pages_contracts_screenshot_path || '—', kind: 'path', showRaw: true },
          ]
        : [
            { key: 'cmd', label: 'cmd', value: activeAcceptanceSubcommand.cmd || '—', showRaw: true },
            { key: 'cwd', label: 'cwd', value: activeAcceptanceSubcommand.cwd || '—', kind: 'path', showRaw: true },
            { key: 'report-path', label: '关联验收报告', value: linkedAcceptanceCheck?.report_path || '—', kind: 'path', showRaw: true },
            { key: 'research-audit-route', label: '研究审计检索', value: explicitSubcommandSearchLink || primaryResearchAuditCase?.search_route || '—', kind: 'path', showRaw: true },
            { key: 'research-audit-artifact-route', label: '研究审计工件', value: explicitSubcommandArtifactLink || primaryResearchAuditCase?.workspace_route || '—', kind: 'path', showRaw: true },
            { key: 'research-audit-raw-route', label: '研究审计原始层', value: explicitSubcommandRawLink || (primaryResearchAuditCase?.raw_path ? buildWorkspaceLink('raw', { artifact: primaryResearchAuditCase.raw_path }) : '—'), kind: 'path', showRaw: true },
            { key: 'graph-home-route', label: '图谱主页路由', value: linkedAcceptanceCheck?.graph_home_resolved_route || '—', kind: 'path', showRaw: true },
          ];
      readonlyLinks.push({
        id: `contracts-subcommand-check-${activeAcceptanceSubcommand.id}`,
        label: `查看${safeDisplayValue(linkedAcceptanceCheck?.label || activeAcceptanceSubcommand.label || '对应验收项')}`,
        to: explicitSubcommandCheckRoute || buildWorkspacePageLink('contracts', baseFocus, `contracts-check-${activeAcceptanceSubcommand.id}`),
      });
      if (!isTopologySubcommand && (explicitSubcommandSearchLink || primaryResearchAuditCase?.search_route)) {
        readonlyLinks.push({
          id: `contracts-subcommand-audit-search-${activeAcceptanceSubcommand.id}`,
          label: '查看研究审计检索',
          to: explicitSubcommandSearchLink || buildSearchLink(primaryResearchAuditCase?.query || '', 'artifact'),
        });
      }
      if (!isTopologySubcommand && (explicitSubcommandArtifactLink || primaryResearchAuditCase?.workspace_route)) {
        readonlyLinks.push({
          id: `contracts-subcommand-audit-artifact-${activeAcceptanceSubcommand.id}`,
          label: '查看研究审计工件',
          to: explicitSubcommandArtifactLink || primaryResearchAuditCase?.workspace_route || '',
        });
      }
      if (!isTopologySubcommand && (explicitSubcommandRawLink || primaryResearchAuditCase?.raw_path)) {
        readonlyLinks.push({
          id: `contracts-subcommand-audit-raw-${activeAcceptanceSubcommand.id}`,
          label: '查看研究审计原始层',
          to: explicitSubcommandRawLink || buildWorkspaceLink('raw', { artifact: primaryResearchAuditCase?.raw_path }),
        });
      }
      sourceExtra = fallbackChain.length ? (
        <div className="inspector-fallback-strip">
          <span>契约回退链</span>
          <strong>{fallbackChain.join(' → ')}</strong>
        </div>
      ) : null;
      titleDescription = '当前已锁定到契约子命令，显示命令参数、回报码与关联验收证据。';
    } else {
      focusTitle = '契约验收摘要';
      focusSummary = safeDisplayValue(acceptanceSummary?.status || '未生成');
      focusRows = [
        { key: 'status', label: '验收状态', value: acceptanceSummary?.status || '—', kind: 'badge' },
        { key: 'generated', label: '生成时间', value: acceptanceSummary?.generated_at_utc || '—' },
        { key: 'topology', label: '拓扑状态', value: acceptanceSummary?.topology_status || '—', kind: 'badge' },
        { key: 'workspace', label: '工作区状态', value: acceptanceSummary?.workspace_status || '—', kind: 'badge' },
        { key: 'change-class', label: '变更级别', value: acceptanceSummary?.change_class || model?.meta.change_class || 'RESEARCH_ONLY' },
      ];
      focusMeta = <Badge value={acceptanceSummary?.status || 'unknown'}>{safeDisplayValue(acceptanceSummary?.status || '未生成')}</Badge>;
      sourceTitle = '公开面链路';
      sourceSummary = safeDisplayValue(acceptanceCheck?.label || topology?.label || snapshotPath);
      sourceRows = [
        { key: 'check', label: '检查项', value: acceptanceCheck?.label || '—' },
        { key: 'report-path', label: '验收报告', value: acceptanceSummary?.report_path || acceptanceCheck?.report_path || '—', kind: 'path', showRaw: true },
        { key: 'public-url', label: '公开入口', value: topology?.url || '—', kind: 'path', showRaw: true },
        { key: 'public-fetch', label: '公开拉取次数', value: acceptanceCheck?.public_snapshot_fetch_count ?? '—', kind: 'number' },
        { key: 'snapshot-endpoint', label: '快照端点', value: acceptanceCheck?.snapshot_endpoint_observed || snapshotPath, kind: 'path', showRaw: true },
      ];
      sourceExtra = fallbackChain.length ? (
        <div className="inspector-fallback-strip">
          <span>契约回退链</span>
          <strong>{fallbackChain.join(' → ')}</strong>
        </div>
      ) : null;
    }
  }

  if (context.domain === 'workspace' && context.section === 'raw') {
    focusTitle = '原始快照摘要';
    focusSummary = safeDisplayValue(artifactRow?.label || artifactId || '原始快照');
    focusRows = [
      { key: 'artifact', label: '工件', value: artifactId || '—' },
      { key: 'artifact-label', label: '工件标题', value: artifactRow?.label || '—' },
      { key: 'raw-view', label: '原始快照视图', value: artifactId ? '聚焦工件原始层 + 全量快照' : '全量快照' },
      { key: 'snapshot', label: '快照路径', value: snapshotPath, kind: 'path', showRaw: true },
      { key: 'status', label: '原始状态', value: safeDisplayValue((model?.workspace.raw as Record<string, unknown> | undefined)?.status || '—'), kind: 'badge' },
    ];
    sourceTitle = '原始证据';
    sourceSummary = safeDisplayValue(snapshotPath);
    sourceRows = [
      { key: 'artifact-path', label: '工件路径', value: artifactRow?.path || '—', kind: 'path', showRaw: true },
      { key: 'source-head-path', label: '源头路径', value: sourceHead?.path || '—', kind: 'path', showRaw: true },
      { key: 'snapshot-path', label: '快照路径', value: snapshotPath, kind: 'path', showRaw: true },
      { key: 'raw-action', label: '原始动作', value: safeDisplayValue((model?.workspace.raw as Record<string, unknown> | undefined)?.action || '—') },
    ];
    sourceExtra = (
      <div className="inspector-fallback-strip">
        <span>原始快照视图</span>
        <strong>{artifactId ? '聚焦工件原始层 + 全量快照' : '全量快照'}</strong>
      </div>
    );
  }

  const dedupedReadonlyLinks = dedupeLinks(readonlyLinks);

  return (
    <div className="inspector-rail-inner" data-mode={mode} data-shell-tier={shellTier} data-collapsed={collapsed ? 'true' : 'false'}>
      <div className="inspector-title-block">
        <h3>{title}</h3>
        <p>{titleDescription}</p>
      </div>

      <DrilldownSection
        title={focusTitle}
        summary={focusSummary}
        defaultOpen
        meta={focusMeta}
      >
        <KeyValueGrid rows={focusRows} />
      </DrilldownSection>

      <DrilldownSection
        title={sourceTitle}
        summary={sourceSummary}
        defaultOpen
      >
        <KeyValueGrid rows={sourceRows} />
        {sourceExtra}
      </DrilldownSection>

      <DrilldownSection
        title="只读回退"
        summary={`${drillBackLinks.length} 个回退入口 / ${dedupeLinks(readonlyLinks).length} 个证据动作`}
        defaultOpen
      >
        <div className="button-row inspector-link-stack">
          {drillBackLinks.map((link) => (
            <Link className="button" key={link.id} to={link.to}>
              {link.label}
            </Link>
          ))}
          {dedupedReadonlyLinks.map((link) => (
            <Link className="button" key={link.id} to={link.to}>
              {link.label}
            </Link>
          ))}
        </div>
      </DrilldownSection>
    </div>
  );
}
