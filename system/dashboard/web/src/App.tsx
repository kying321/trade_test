import { type ReactNode, useCallback, useEffect, useMemo, useState } from 'react';
import { HashRouter, Link, Navigate, Route, Routes, useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Badge, ClampText, DrilldownSection, KeyValueGrid } from './components/ui-kit';
import { useTerminalStore } from './store/use-terminal-store';
import type { SurfaceKey } from './types/contracts';
import type { TerminalReadModel } from './types/contracts';
import { buildTerminalLink, buildWorkspaceLink, buildWorkspacePageLink, type SharedFocusState, type WorkspaceSection } from './utils/focus-links';
import { compactPath, formatDateTime, safeDisplayValue } from './utils/formatters';
import { explainLabel, labelFor, surfaceLabel } from './utils/dictionary';
import { AppShell } from './app-shell/AppShell';
import { ContextHeader, type ContextHeaderSection } from './app-shell/ContextHeader';
import { ContextSidebar } from './app-shell/ContextSidebar';
import { GlobalNoticeLayer } from './app-shell/GlobalNoticeLayer';
import { GlobalTopbar, type ResolvedTheme, type ThemePreference } from './app-shell/GlobalTopbar';
import { InspectorRail } from './app-shell/InspectorRail';
import { useSidebarCollapse } from './hooks/use-sidebar-collapse';
import { useUiTheme } from './hooks/use-ui-theme';
import { PRIMARY_NAV, OPS_SURFACE_NAV, RESEARCH_LEGACY_NAV } from './navigation/nav-config';
import { buildOpsLegacyLink, buildOverviewLink, buildResearchLegacyLink } from './navigation/route-contract';
import { getWorkspacePageSections } from './navigation/workspace-sections';
import { OverviewPage } from './pages/OverviewPage';
import { OpsPage } from './pages/OpsPage';
import { ResearchPage } from './pages/ResearchPage';
import { SearchPage } from './pages/SearchPage';
import { GlobalSearchOverlay } from './search/SearchOverlay';
import { buildSearchLink } from './search/links';
import type { SearchResultEntry, SearchScope } from './search/types';

const POLL_MS = 30_000;
const t = (key: string) => labelFor(key);
const FOCUS_FIELD_LABELS: Array<{ key: keyof SharedFocusState; label: string }> = [
  { key: 'artifact', label: '工件' },
  { key: 'anchor', label: '锚点' },
  { key: 'panel', label: '面板' },
  { key: 'section', label: '分段' },
  { key: 'row', label: '行' },
  { key: 'group', label: '分组' },
  { key: 'view', label: '视图' },
  { key: 'symbol', label: '标的' },
  { key: 'window', label: '窗口' },
  { key: 'domain', label: '域' },
  { key: 'tone', label: '色调' },
  { key: 'search', label: '检索' },
  { key: 'search_scope', label: '检索范围' },
];

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

type HeaderBadgeItem = {
  value?: unknown;
  label: string;
};

function parseFocus(searchParams: URLSearchParams): SharedFocusState {
  return {
    artifact: searchParams.get('artifact') || undefined,
    anchor: searchParams.get('anchor') || undefined,
    domain: searchParams.get('domain') || undefined,
    tone: searchParams.get('tone') || undefined,
    search: searchParams.get('search') || undefined,
    search_scope: searchParams.get('search_scope') || undefined,
    panel: searchParams.get('panel') || undefined,
    section: searchParams.get('section') || undefined,
    row: searchParams.get('row') || undefined,
    group: searchParams.get('group') || undefined,
    view: searchParams.get('view') || undefined,
    symbol: searchParams.get('symbol') || undefined,
    window: searchParams.get('window') || undefined,
  };
}

function parseSearchScope(raw: string | null | undefined): SearchScope {
  if (raw === 'module' || raw === 'route' || raw === 'artifact') return raw;
  return 'all';
}

function resolveWorkspacePageSectionId(
  searchParams: URLSearchParams,
  pageSectionIds: string[],
  focus: SharedFocusState,
): string | undefined {
  const pageSection = searchParams.get('page_section') || undefined;
  if (pageSection && pageSectionIds.includes(pageSection)) return pageSection;
  if (focus.section && pageSectionIds.includes(focus.section)) return focus.section;
  return undefined;
}

function canUseViewportScroll() {
  return typeof navigator === 'undefined' || !/jsdom/i.test(navigator.userAgent);
}

function resolveSearchAnchorId(searchParams: URLSearchParams, focus: SharedFocusState): string | undefined {
  return searchParams.get('anchor') || focus.anchor || undefined;
}

function scrollToSearchAnchor(anchorId: string | undefined) {
  if (!anchorId) return false;
  const target = document.querySelector<HTMLElement>(`[data-search-anchor="${anchorId}"], #${anchorId}`);
  return safeScrollIntoView(target);
}

function scrollToWorkspaceAnchor(sectionId: string) {
  const target = document.querySelector<HTMLElement>(`[data-workspace-section="${sectionId}"]`);
  return safeScrollIntoView(target);
}

function safeScrollIntoView(target: Element | null | undefined) {
  if (!target || typeof (target as HTMLElement).scrollIntoView !== 'function' || !canUseViewportScroll()) return false;
  (target as HTMLElement).scrollIntoView({ block: 'start', behavior: 'auto' });
  return true;
}

function safeResetWindowScroll() {
  if (typeof document !== 'undefined') {
    document.documentElement.scrollTop = 0;
    if (document.body) document.body.scrollTop = 0;
  }
  if (typeof window !== 'undefined' && typeof window.scrollTo === 'function' && canUseViewportScroll()) {
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
  }
}

function useSurfaceRefresh(requested: SurfaceKey, refreshKey?: string, allowFallback = true) {
  const { refresh } = useTerminalStore();
  useEffect(() => {
    void refresh(requested, allowFallback);
  }, [allowFallback, refresh, requested, refreshKey]);

  useEffect(() => {
    const handle = window.setInterval(() => {
      void refresh(requested, allowFallback);
    }, POLL_MS);
    return () => window.clearInterval(handle);
  }, [allowFallback, refresh, requested]);
}

function buildPrimaryNav(focus: SharedFocusState, currentSurface: SurfaceKey) {
  return PRIMARY_NAV.map((item) => {
    if (item.id === 'overview') return { ...item, to: buildOverviewLink(focus) };
    if (item.id === 'ops') return { ...item, to: buildOpsLegacyLink(currentSurface, focus) };
    return { ...item, to: buildResearchLegacyLink('artifacts', focus) };
  });
}

function buildTopbarGlobalSummary(model: TerminalReadModel | null | undefined) {
  return {
    changeClass: labelFor(String(model?.meta.change_class || 'RESEARCH_ONLY')),
    chips: [
      { label: '模式', value: String(model?.experienceContract.mode || 'read_only_snapshot') },
      { label: '快照', value: formatDateTime(model?.meta.generated_at_utc) },
    ],
  };
}

function buildOverviewPageMeta() {
  return {
    topbarChips: [
      { label: '当前页', value: '总览' },
      { label: '职责', value: '入口分流' },
    ],
    sidebarTitle: '一级域路由',
    sidebarSubtitle: '总览入口',
    sidebarDescription: '系统状态 / 入口切换 / 路由总览',
    headerTitle: '总览',
    headerSubtitle: '系统状态 / 入口 / 路由总览',
  };
}

function buildTerminalPageMeta(requested: SurfaceKey) {
  const isInternal = requested === 'internal';
  return {
    topbarChips: [
      { label: '当前页', value: isInternal ? '内部面' : '公开面' },
      { label: '职责', value: isInternal ? '深层证据' : '执行穿透' },
    ],
    sidebarTitle: '终端路由',
    sidebarSubtitle: '面型切换',
    sidebarDescription: isInternal ? '内部只读 / 深层字段 / 执行证据' : '公开摘要 / 执行穿透 / 调度门禁',
    headerTitle: isInternal ? '内部面' : '公开面',
    headerSubtitle: isInternal ? '内部只读 / 深层字段 / 执行证据' : '执行穿透 / 调度与门禁',
    headerBarKicker: isInternal ? '终端 / 深层证据' : '终端 / 运行态',
    headerBarTitle: isInternal ? '内部视图与深层证据' : '视图断言与运行状态',
    headerBarSubtitle: isInternal ? '内部字段 / 执行证据 / 只读边界' : '快照来源 / 刷新状态 / 只读回退',
  };
}

function buildWorkspacePageMeta(section: WorkspaceSection) {
  const pageMeta: Record<WorkspaceSection, {
    topbarChips: Array<{ label: string; value: string }>;
    sidebarTitle: string;
    sidebarSubtitle: string;
    sidebarDescription: string;
    headerTitle: string;
    headerSubtitle: string;
    headerBarKicker: string;
    headerBarTitle: string;
    headerBarSubtitle: string;
  }> = {
    artifacts: {
      topbarChips: [
        { label: '当前页', value: '工件池' },
        { label: '职责', value: '主线筛选' },
      ],
      sidebarTitle: '研究路由',
      sidebarSubtitle: '阶段切换',
      sidebarDescription: '工件 / 回测 / 契约 / 原始',
      headerTitle: '工件池',
      headerSubtitle: '研究地图 / 状态雷达 / 焦点动作',
      headerBarKicker: '工件 / 当前焦点',
      headerBarTitle: '研究地图与当前焦点',
      headerBarSubtitle: '主线分组 / 状态雷达 / 检索范围',
    },
    alignment: {
      topbarChips: [
        { label: '当前页', value: '对齐页' },
        { label: '职责', value: '方向校准' },
      ],
      sidebarTitle: '研究路由',
      sidebarSubtitle: '阶段切换',
      sidebarDescription: '工件 / 对齐 / 回测 / 契约 / 原始',
      headerTitle: '方向对齐投射',
      headerSubtitle: '高价值反馈 / 漂移趋势 / 建议动作',
      headerBarKicker: '对齐 / 内部投射',
      headerBarTitle: '高价值反馈与方向漂移',
      headerBarSubtitle: '内部反馈投射 / 趋势 / 建议动作',
    },
    backtests: {
      topbarChips: [
        { label: '当前页', value: '回测池' },
        { label: '职责', value: '验证比较' },
      ],
      sidebarTitle: '研究路由',
      sidebarSubtitle: '阶段切换',
      sidebarDescription: '工件 / 回测 / 契约 / 原始',
      headerTitle: '回测池',
      headerSubtitle: '时序比较 / 近期比较 / OOS 切片',
      headerBarKicker: '回测 / 验证窗口',
      headerBarTitle: '回测比较与验证窗口',
      headerBarSubtitle: '时序切片 / 近期比较 / OOS 验证',
    },
    contracts: {
      topbarChips: [
        { label: '当前页', value: '契约层' },
        { label: '职责', value: '路由验收' },
      ],
      sidebarTitle: '研究路由',
      sidebarSubtitle: '阶段切换',
      sidebarDescription: '工件 / 回测 / 契约 / 原始',
      headerTitle: '契约层',
      headerSubtitle: '公开入口 / 数据契约 / 路由验收',
      headerBarKicker: '契约 / 验收记录',
      headerBarTitle: '入口拓扑与验收记录',
      headerBarSubtitle: '公开入口 / 子链验收 / 回退链',
    },
    raw: {
      topbarChips: [
        { label: '当前页', value: '原始层' },
        { label: '职责', value: '证据回退' },
      ],
      sidebarTitle: '研究路由',
      sidebarSubtitle: '阶段切换',
      sidebarDescription: '工件 / 回测 / 契约 / 原始',
      headerTitle: '原始层',
      headerSubtitle: '原始快照 / 证据回退 / JSON 穿透',
      headerBarKicker: '原始 / 最终回退',
      headerBarTitle: '原始快照与证据回退',
      headerBarSubtitle: 'JSON 穿透 / 原始字段 / 回退链',
    },
  };
  return pageMeta[section];
}

function artifactGroupLabel(group: string | undefined) {
  if (!group) return '未声明';
  return ARTIFACT_GROUP_LABELS[group] || labelFor(group);
}

function searchScopeLabel(scope: string | undefined) {
  if (!scope) return '标题';
  return SEARCH_SCOPE_LABELS[scope] || labelFor(scope);
}

function buildWorkspaceHeaderBadges(
  model: TerminalReadModel | undefined,
  focus: SharedFocusState,
  section: WorkspaceSection,
): HeaderBadgeItem[] {
  const selectedRow = model ? findWorkspaceFocusRow(model, focus.artifact) : undefined;
  const artifactLabel = selectedRow ? labelFor(selectedRow.label || selectedRow.payload_key || selectedRow.id) : '未选工件';
  const activeGroup = focus.group || selectedRow?.artifact_group;
  const researchDecision = selectedRow?.research_decision || selectedRow?.status || 'neutral';
  return [
    { value: selectedRow?.status || selectedRow?.research_decision || focus.artifact || 'neutral', label: `焦点：${artifactLabel}` },
    { value: researchDecision, label: `结论：${labelFor(String(researchDecision))}` },
    { value: activeGroup, label: `分组：${artifactGroupLabel(activeGroup)}` },
    { value: focus.search_scope || 'title', label: `检索：${searchScopeLabel(focus.search_scope)}` },
    { value: model?.meta.change_class, label: `变更：${labelFor(String(model?.meta.change_class || 'RESEARCH_ONLY'))}` },
  ];
}

function buildOpsSidebarItems(focus: SharedFocusState) {
  return OPS_SURFACE_NAV.map((item) => ({
    ...item,
    to: item.id === 'terminal-internal' ? buildTerminalLink('internal', focus) : buildTerminalLink('public', focus),
  }));
}

function buildResearchSidebarItems(focus: SharedFocusState, model?: TerminalReadModel) {
  return RESEARCH_LEGACY_NAV.map((item) => ({
    ...item,
    to: buildWorkspacePageLink(
      item.section,
      focus,
      model && item.section !== 'artifacts' ? getWorkspacePageSections(item.section, model)[0]?.id : undefined,
    ),
  }));
}

function findWorkspaceFocusRow(model: TerminalReadModel, artifact: string | undefined) {
  const target = safeDisplayValue(artifact).trim().toLowerCase();
  if (target && target !== '—') {
    const matched = model.workspace.artifactRows.find((row) => {
      const candidates = [row.id, row.payload_key, row.label, row.path]
        .map((value) => safeDisplayValue(value).trim().toLowerCase())
        .filter((value) => value && value !== '—');
      return candidates.some((candidate) => candidate === target);
    });
    if (matched) return matched;
  }

  return model.workspace.artifactRows.find((row) => {
    const id = safeDisplayValue(row.id).trim().toLowerCase();
    const payloadKey = safeDisplayValue(row.payload_key).trim().toLowerCase();
    return id === 'price_action_breakout_pullback' || payloadKey === 'price_action_breakout_pullback';
  }) || model.workspace.artifactRows.find((row) => safeDisplayValue(row.artifact_group) === 'research_mainline') || model.workspace.artifactRows[0];
}

function resolveWorkspaceNavigationFocus(focus: SharedFocusState, model?: TerminalReadModel): SharedFocusState {
  if (!model) return focus;

  const resolved = { ...focus };
  const defaultFocus = model.workspace.defaultFocus;
  const hasExplicitArtifact = Boolean(focus.artifact);
  const hasExplicitGroup = Boolean(focus.group);
  const canAdoptDefaultFocus = !hasExplicitArtifact && !hasExplicitGroup;
  const sourceRow = findWorkspaceFocusRow(model, hasExplicitArtifact ? focus.artifact : canAdoptDefaultFocus ? defaultFocus?.artifact : undefined);
  const sourceArtifactId = safeDisplayValue(sourceRow?.id);
  const sourceGroup = safeDisplayValue(sourceRow?.artifact_group);
  const sourcePrimaryFocus = sourceArtifactId !== '—' ? model.workspace.artifactPrimaryFocus[sourceArtifactId] : undefined;

  if (!resolved.search_scope) resolved.search_scope = defaultFocus?.search_scope || 'title';
  if (!resolved.artifact && canAdoptDefaultFocus) resolved.artifact = defaultFocus?.artifact || (sourceArtifactId !== '—' ? sourceArtifactId : undefined);
  if (!resolved.group) resolved.group = canAdoptDefaultFocus ? defaultFocus?.group || (sourceGroup !== '—' ? sourceGroup : undefined) : undefined;
  if (!resolved.panel) resolved.panel = canAdoptDefaultFocus ? defaultFocus?.panel || sourcePrimaryFocus?.panel : sourcePrimaryFocus?.panel;
  if (!resolved.section) resolved.section = canAdoptDefaultFocus ? defaultFocus?.section || sourcePrimaryFocus?.section : sourcePrimaryFocus?.section;
  if (!resolved.row) resolved.row = canAdoptDefaultFocus ? defaultFocus?.row || sourcePrimaryFocus?.row : sourcePrimaryFocus?.row;

  return resolved;
}

function focusWithout(focus: SharedFocusState, keys: Array<keyof SharedFocusState>): SharedFocusState {
  const next: SharedFocusState = { ...focus };
  keys.forEach((key) => {
    delete next[key];
  });
  return next;
}

function buildFocusSection(focus: SharedFocusState): ContextHeaderSection {
  const items = FOCUS_FIELD_LABELS.flatMap(({ key, label }) => {
    const raw = focus[key];
    if (!raw) return [];
    const explained = explainLabel(String(raw));
    return [{
      type: 'fact' as const,
      label,
      value: explained.primary,
      detail: explained.secondary,
      tone: 'neutral' as const,
    }];
  });

  if (!items.length) {
    items.push({
      type: 'fact',
      label: '状态',
      value: '未声明',
      detail: '当前 URL 未携带 artifact / panel / section / row 等穿透参数。',
      tone: 'neutral',
    });
  }

  return {
    id: 'focus',
    title: '穿透焦点',
    items,
  };
}

function buildOverviewHeaderSections(focus: SharedFocusState): ContextHeaderSection[] {
  return [
    {
      id: 'domain',
      title: '当前域',
      items: [
        { type: 'link', label: '总览', value: '一级分流', to: buildOverviewLink(focus), active: true },
        { type: 'link', label: '操作终端', value: '调度 / 信号 / 风控', to: buildOpsLegacyLink('public', focus) },
        { type: 'link', label: '研究工作区', value: '工件 / 回测 / 原始', to: buildResearchLegacyLink('artifacts', focus) },
      ],
    },
    buildFocusSection(focus),
    {
      id: 'backtrack',
      title: '层级回退',
      items: [
        { type: 'link', label: '清空穿透参数', value: '返回总览基线', to: '/overview' },
      ],
    },
  ];
}

function buildSearchHeaderSections(): ContextHeaderSection[] {
  return [
    {
      id: 'search',
      title: '检索范围',
      items: [
        { type: 'fact', label: '覆盖', value: '模块 / 路由 / 工件', detail: '只搜索前端本地索引，不查 docs/code/tests。', tone: 'neutral' },
      ],
    },
  ];
}

function buildTerminalHeaderActions(
  focus: SharedFocusState,
  requested: SurfaceKey,
  effective: SurfaceKey,
): ReactNode {
  const chips: Array<{ key: string; value: string; tone?: string }> = [
    {
      key: 'effective',
      value: `当前视图：${surfaceLabel(effective)}`,
      tone: requested === effective ? 'positive' : 'warning',
    },
  ];

  if (requested !== effective) {
    chips.push({
      key: 'requested',
      value: `请求：${surfaceLabel(requested)}`,
      tone: 'warning',
    });
  }

  if (focus.panel) {
    chips.push({
      key: 'panel',
      value: `面板：${labelFor(focus.panel)}`,
      tone: 'neutral',
    });
  }

  return (
    <div className="context-header-route-badges">
      {chips.map((chip) => (
        <Badge key={chip.key} value={chip.tone}>{chip.value}</Badge>
      ))}
    </div>
  );
}

function buildWorkspaceHeaderActions(
  section: WorkspaceSection,
  pageSections: Array<{ id: string; label: string }>,
  activeSectionId: string | undefined,
): ReactNode {
  const activeSection = activeSectionId ? pageSections.find((item) => item.id === activeSectionId) : undefined;
  const stageLabel = activeSection?.label || buildWorkspacePageMeta(section).topbarChips[0]?.value || labelFor(section);
  return (
    <div className="context-header-route-badges">
      <Badge value="neutral">{`当前阶段：${stageLabel}`}</Badge>
    </div>
  );
}

function HeaderBar({
  kicker,
  title,
  subtitle,
  badges,
}: {
  kicker: string;
  title: string;
  subtitle: string;
  badges?: HeaderBadgeItem[];
}) {
  const { model, loading } = useTerminalStore();
  const effectiveSurface = model?.surface.effective || 'public';
  const operationalMode = labelFor(String(model?.experienceContract.mode || 'read_only_snapshot'));
  const renderedBadges = badges || [
    { value: effectiveSurface, label: surfaceLabel(effectiveSurface) },
    { value: loading ? 'warning' : 'ok', label: loading ? labelFor('refreshing') : labelFor('stable') },
    { value: model?.meta.change_class, label: labelFor(String(model?.meta.change_class || 'RESEARCH_ONLY')) },
    { value: model?.experienceContract.live_path_touched, label: `${t('live_path_touched_badge')}：${labelFor(String(model?.experienceContract.live_path_touched ?? false))}` },
    { value: model?.experienceContract.mode, label: operationalMode },
  ];
  return (
    <header className={`header-bar header-bar-${effectiveSurface}`}>
      <div className="header-bar-copy">
        <p className="panel-kicker">{kicker}</p>
        <h2>{title}</h2>
        <p className="panel-note">{subtitle}</p>
      </div>
      <div className="header-badges">
        {renderedBadges.map((badge, index) => (
          <Badge key={`${badge.label}-${index}`} value={badge.value}>{badge.label}</Badge>
        ))}
      </div>
    </header>
  );
}

function SurfaceNotice() {
  const { model } = useTerminalStore();
  if (!model) return null;
  const fallbackChain = Array.isArray(model.experienceContract.fallback_chain) ? model.experienceContract.fallback_chain.slice(0, 3) : [];
  return (
    <div className={`surface-notice ${model.surface.effective === model.surface.requested ? 'matched' : 'fallback'} surface-notice-${model.surface.effective}`}>
      <div className="surface-notice-head">
        <strong>{t('surface_notice_label')}：</strong>
        {t('surface_notice_requested_label')} {model.surface.requestedLabel} / {t('surface_notice_effective_label')} {model.surface.effectiveLabel}
      </div>
      <div className="header-badges">
        <Badge value={model.surface.availability}>{labelFor(model.surface.availability)}</Badge>
        <Badge value={model.surface.redactionLevel}>{labelFor(model.surface.redactionLevel)}</Badge>
      </div>
      <div className="surface-strata-grid">
        <article className="surface-strata-card">
          <span className="surface-strata-label">{t('surface_visibility_label')}</span>
          <strong>{model.surface.effective === 'internal' ? t('surface_visibility_internal') : t('surface_visibility_public')}</strong>
        </article>
        <article className="surface-strata-card">
          <span className="surface-strata-label">{t('fallback_chain_label')}</span>
          <strong>{fallbackChain.length ? fallbackChain.map((item) => compactPath(String(item))).join(' → ') : t('undeclared')}</strong>
        </article>
        <article className="surface-strata-card">
          <span className="surface-strata-label">{t('redaction_policy_label')}</span>
          <strong>{labelFor(model.surface.redactionLevel)}</strong>
        </article>
      </div>
      {model.surface.warnings.length ? <p>{model.surface.warnings.join('；')}</p> : null}
    </div>
  );
}

function SurfaceUnavailablePanel({
  title,
  detail,
}: {
  title: string;
  detail: string;
}) {
  return (
    <section className="panel-card">
      <p className="panel-kicker">internal_only / unavailable</p>
      <h3>{title}</h3>
      <p className="panel-note">{detail}</p>
    </section>
  );
}

function WorkspaceContextStrip({
  model,
  focus,
  section,
  pageSectionId,
}: {
  model: TerminalReadModel;
  focus: SharedFocusState;
  section: WorkspaceSection;
  pageSectionId?: string;
}) {
  const pageMeta = buildWorkspacePageMeta(section);
  const selectedRow = findWorkspaceFocusRow(model, focus.artifact);
  const selectedId = safeDisplayValue(selectedRow?.id);
  const selectedPayloadKey = safeDisplayValue(selectedRow?.payload_key);
  const selectedLabel = labelFor(String(selectedRow?.label || selectedRow?.payload_key || selectedRow?.id || '未选工件'));
  const selectedPath = safeDisplayValue(selectedRow?.path);
  const activeGroup = focus.group || selectedRow?.artifact_group;
  const researchDecision = selectedRow?.research_decision || selectedRow?.status || 'neutral';
  const currentSectionLabel = pageSectionId ? labelFor(pageSectionId) : pageMeta.headerTitle;
  const handoffs = selectedId !== '—' ? (model.workspace.artifactHandoffs[selectedId] || []) : [];
  const primaryFocus = selectedId !== '—' ? model.workspace.artifactPrimaryFocus[selectedId] : undefined;
  const sectionContextNote = section === 'artifacts'
    ? `检索范围：${searchScopeLabel(focus.search_scope)}`
    : section === 'alignment'
      ? '反馈投射 / 漂移趋势 / 建议动作'
    : section === 'backtests'
      ? '基线总量 / 回测池 / 近期比较'
      : section === 'contracts'
        ? '入口拓扑 / 子链状态 / 回退链'
        : '聚焦工件 / 原始快照 / 最终回退';
  const researchNavActions = [
    ...(section !== 'artifacts'
      ? [{
          id: 'workspace-context-artifacts',
          label: '回工件池',
          to: buildWorkspaceLink('artifacts', focusWithout(focus, ['row'])),
        }]
      : []),
    ...(section !== 'alignment'
      ? [{
          id: 'workspace-context-alignment',
          label: '查看对齐页',
          to: buildWorkspaceLink('alignment', { ...focusWithout(focus, ['row']), view: 'internal' }),
        }]
      : []),
    ...(section !== 'backtests'
      ? [{
          id: 'workspace-context-backtests',
          label: '查看回测池',
          to: buildWorkspaceLink('backtests', focusWithout(focus, ['row'])),
        }]
      : []),
    ...(section !== 'contracts'
      ? [{
          id: 'workspace-context-contracts',
          label: '查看契约层',
          to: buildWorkspaceLink('contracts', focusWithout(focus, ['row'])),
        }]
      : []),
    ...(section !== 'raw'
      ? [{
          id: 'workspace-context-raw',
          label: '查看原始层',
          to: buildWorkspaceLink('raw', focusWithout(focus, ['row'])),
        }]
      : []),
  ];
  const actions = [
    ...handoffs.slice(0, 1),
    ...researchNavActions.map((action) => ({
      ...action,
      tone: 'neutral' as const,
    })),
  ].slice(0, 3);
  const evidenceHint = primaryFocus?.panel
    ? `终端焦点：${labelFor(primaryFocus.panel)}${primaryFocus.section ? ` / ${labelFor(primaryFocus.section)}` : ''}`
    : handoffs.length
      ? `交接动作：${handoffs.length} 条`
      : activeGroup
        ? `分组：${artifactGroupLabel(activeGroup)}`
        : '未声明分组';

  return (
    <section className="workspace-context-strip" aria-label="workspace-context-strip">
      <div className="workspace-context-head">
        <div className="workspace-context-copy">
          <p className="panel-kicker">{pageMeta.headerBarKicker}</p>
          <strong>
            <ClampText raw={pageMeta.headerBarTitle} expandable={false}>
              {pageMeta.headerBarTitle}
            </ClampText>
          </strong>
          <span>
            <ClampText raw={pageMeta.headerBarSubtitle}>
              {pageMeta.headerBarSubtitle}
            </ClampText>
          </span>
        </div>
        <div className="header-badges">
          <Badge value={researchDecision}>结论：{labelFor(String(researchDecision))}</Badge>
          <Badge value={activeGroup}>分组：{artifactGroupLabel(activeGroup)}</Badge>
          <Badge value={model.meta.change_class}>变更：{labelFor(String(model.meta.change_class || 'RESEARCH_ONLY'))}</Badge>
        </div>
      </div>

      <div className="workspace-context-grid">
        <article className="workspace-context-card">
          <span className="workspace-context-label">焦点工件</span>
          <strong><ClampText raw={selectedLabel}>{selectedLabel}</ClampText></strong>
          <small>
            <ClampText raw={selectedPath !== '—' ? selectedPath : undefined}>
              {selectedPath !== '—' ? compactPath(selectedPath) : '未声明路径'}
            </ClampText>
          </small>
        </article>
        <article className="workspace-context-card">
          <span className="workspace-context-label">验证窗口</span>
          <strong><ClampText raw={currentSectionLabel}>{currentSectionLabel}</ClampText></strong>
          <small>{sectionContextNote}</small>
        </article>
        <article className="workspace-context-card">
          <span className="workspace-context-label">证据链</span>
          <strong><ClampText raw={selectedPayloadKey !== '—' ? selectedPayloadKey : selectedId}>{selectedPayloadKey !== '—' ? selectedPayloadKey : selectedId}</ClampText></strong>
          <small>{evidenceHint}</small>
        </article>
      </div>

      {actions.length ? (
        <div className="button-row workspace-context-actions">
          {actions.map((action) => (
            <Link className="button" key={action.id} to={action.to}>
              {action.label}
            </Link>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function OpsContextStrip({
  model,
  focus,
  requested,
  snapshotPath,
}: {
  model: TerminalReadModel;
  focus: SharedFocusState;
  requested: SurfaceKey;
  snapshotPath?: string;
}) {
  const effectiveSurface = model.surface.effective;
  const fallbackChain = Array.isArray(model.experienceContract.fallback_chain) ? model.experienceContract.fallback_chain.slice(0, 3) : [];
  const focusSummary = [
    focus.panel ? labelFor(focus.panel) : null,
    focus.section ? labelFor(focus.section) : null,
    focus.row ? labelFor(focus.row) : null,
  ].filter(Boolean).join(' / ');
  const topAlert = model.orchestration.alerts[0];
  const operatorPanelHref = model.uiRoutes.operator_panel || '/operator_task_visual_panel.html';
  const runtimeSummary = [
    snapshotPath ? compactPath(snapshotPath) : '未声明快照',
    formatDateTime(model.meta.generated_at_utc),
  ].join(' / ');
  const gateSummary = topAlert?.summary || (fallbackChain.length ? fallbackChain.map((item) => compactPath(String(item))).join(' → ') : '无回退链');

  return (
    <section className="ops-context-strip" aria-label="ops-context-strip">
      <div className="ops-context-head">
        <div className="ops-context-copy">
          <p className="panel-kicker">运行上下文</p>
          <strong>
            <ClampText raw={`${surfaceLabel(requested)} / ${surfaceLabel(effectiveSurface)}`} expandable={false}>
              {surfaceLabel(requested)} / {surfaceLabel(effectiveSurface)}
            </ClampText>
          </strong>
          <span>
            <ClampText raw={`${labelFor(String(model.experienceContract.mode || 'read_only_snapshot'))} / ${labelFor(model.surface.redactionLevel)}`}>
              {labelFor(String(model.experienceContract.mode || 'read_only_snapshot'))} / {labelFor(model.surface.redactionLevel)}
            </ClampText>
          </span>
        </div>
        <div className="header-badges">
          <Badge value={requested === effectiveSurface ? 'ok' : 'warning'}>
            视图：{requested === effectiveSurface ? '匹配' : '回退'}
          </Badge>
          <Badge value={model.surface.availability}>可见性：{labelFor(model.surface.availability)}</Badge>
          <Badge value={model.meta.change_class}>变更：{labelFor(String(model.meta.change_class || 'RESEARCH_ONLY'))}</Badge>
        </div>
      </div>

      <div className="ops-context-grid">
        <article className="ops-context-card">
          <span className="ops-context-label">视图断言</span>
          <strong>{surfaceLabel(requested)} → {surfaceLabel(effectiveSurface)}</strong>
          <small>请求视图 / 实际视图 / 公开安全边界</small>
        </article>
        <article className="ops-context-card">
          <span className="ops-context-label">运行态</span>
          <strong><ClampText raw={runtimeSummary}>{runtimeSummary}</ClampText></strong>
          <small>{labelFor(String(model.experienceContract.mode || 'read_only_snapshot'))} / {labelFor(model.surface.availability)}</small>
        </article>
        <article className="ops-context-card">
          <span className="ops-context-label">门禁与告警</span>
          <strong><ClampText raw={topAlert?.title || '无显著告警'}>{topAlert?.title || '无显著告警'}</ClampText></strong>
          <small><ClampText raw={gateSummary}>
            {gateSummary}
          </ClampText></small>
        </article>
      </div>

      {focusSummary ? <p className="ops-context-meta">当前穿透：{focusSummary}</p> : null}

      <div className="button-row ops-context-actions">
        {focus.row ? (
          <Link className="button" to={buildTerminalLink(requested, focusWithout(focus, ['row']))}>清除行焦点</Link>
        ) : null}
        <Link className="button" to={buildOverviewLink(focusWithout(focus, ['panel', 'section', 'row']))}>回总览</Link>
        <Link className="button" to={buildTerminalLink(requested, focusWithout(focus, ['panel', 'section', 'row']))}>回终端主层</Link>
        <a className="button" href={operatorPanelHref} target="_blank" rel="noreferrer">查看运维面板</a>
      </div>
    </section>
  );
}

function SurfaceShell({ children }: { children: ReactNode }) {
  const { model } = useTerminalStore();
  if (!model) return <>{children}</>;
  const effectiveSurface = model.surface.effective;
  const note = effectiveSurface === 'internal'
    ? t('surface_internal_local_only_notice')
    : `${labelFor(model.surface.redactionLevel)} ｜ ${labelFor(String(model.experienceContract.mode || 'read_only_snapshot'))}`;

  return (
    <div className={`surface-shell surface-${effectiveSurface}`}>
      <div className="surface-shell-head">
        <span className={`surface-watermark surface-watermark-${effectiveSurface}`}>
          {effectiveSurface === 'internal' ? t('surface_visibility_internal') : t('surface_visibility_public')}
        </span>
        <div className="surface-shell-note">
          <strong>{surfaceLabel(effectiveSurface)}</strong>
          <span>{note}</span>
        </div>
      </div>
      <div className="surface-shell-stage">
        {children}
      </div>
    </div>
  );
}

function AttentionStack() {
  const { model } = useTerminalStore();
  if (!model?.orchestration.alerts.length) return null;
  return (
    <section className={`attention-stack attention-stack-${model.surface.effective}`} aria-label="operator-alerts">
      {model.orchestration.alerts.map((alert) => (
        <article className={`attention-card attention-card-${model.surface.effective} ${alert.tone}`} key={alert.id}>
          <div className="attention-head">
            <div className="attention-copy">
              <span className="attention-kicker">{t('attention_stack_kicker')}</span>
              <strong>{alert.title}</strong>
              {alert.summary ? <span>{alert.summary}</span> : null}
            </div>
            <Badge value={alert.tone}>{labelFor(alert.tone)}</Badge>
          </div>
          {alert.detail ? <p className="attention-detail">{alert.detail}</p> : null}
          {alert.highlights?.length ? (
            <div className="attention-highlights">
              <KeyValueGrid rows={alert.highlights} />
            </div>
          ) : null}
          {alert.sections?.length ? (
            <div className="attention-drills">
              {alert.sections.map((section, index) => (
                <DrilldownSection
                  key={`${alert.id}-${section.id}`}
                  title={section.title}
                  summary={section.summary}
                  tone={section.tone || alert.tone}
                  defaultOpen={section.defaultOpen ?? index === 0}
                >
                  <KeyValueGrid rows={section.rows} />
                </DrilldownSection>
              ))}
            </div>
          ) : null}
          {alert.links?.length ? (
            <div className="button-row attention-links">
              {alert.links.map((link) => (
                <Link className={`button ${link.tone === 'negative' ? 'button-primary' : ''}`.trim()} key={link.id} to={link.to}>
                  {link.label}
                </Link>
              ))}
            </div>
          ) : null}
        </article>
      ))}
    </section>
  );
}

type AppThemeProps = {
  themePreference: ThemePreference;
  resolvedTheme: ResolvedTheme;
  onThemeChange: (next: ThemePreference) => void;
  sidebarCollapsed: boolean;
  canCollapseSidebar: boolean;
  onToggleSidebarCollapse: () => void;
};

function OverviewRoute({
  themePreference,
  resolvedTheme,
  onThemeChange,
  sidebarCollapsed,
  canCollapseSidebar,
  onToggleSidebarCollapse,
}: AppThemeProps) {
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const { model, loading, error } = useTerminalStore();
  const focus = parseFocus(searchParams);
  const requestedSurface: SurfaceKey = focus.view === 'internal' ? 'internal' : 'public';
  useSurfaceRefresh(requestedSurface, undefined, false);
  const effectiveSurface = model?.surface.effective || requestedSurface;
  const pageMeta = buildOverviewPageMeta();

  useEffect(() => {
    const anchorId = resolveSearchAnchorId(searchParams, focus);
    const handle = window.requestAnimationFrame(() => {
      if (anchorId && scrollToSearchAnchor(anchorId)) return;
      safeResetWindowScroll();
    });
    return () => window.cancelAnimationFrame(handle);
  }, [focus, searchParams]);

  return (
    <AppShell
      sidebarCollapsed={sidebarCollapsed}
      topbar={<GlobalTopbar currentPath={location.pathname} primaryNav={buildPrimaryNav(focus, effectiveSurface)} globalSummary={buildTopbarGlobalSummary(model)} themePreference={themePreference} resolvedTheme={resolvedTheme} onThemeChange={onThemeChange} onOpenSearch={() => window.dispatchEvent(new CustomEvent('fenlie-toggle-search'))} />}
      sidebar={(
        <ContextSidebar
          title={pageMeta.sidebarTitle}
          subtitle={pageMeta.sidebarSubtitle}
          description={pageMeta.sidebarDescription}
          items={PRIMARY_NAV.map((item) => ({ ...item, to: item.id === 'overview' ? buildOverviewLink(focus) : item.id === 'ops' ? buildOpsLegacyLink(requestedSurface, focus) : buildResearchLegacyLink('artifacts', focus) }))}
          utilityLinks={[{ label: '搜索 / Search', to: buildSearchLink() }]}
          collapsed={sidebarCollapsed}
          canCollapse={canCollapseSidebar}
          onToggleCollapse={onToggleSidebarCollapse}
        />
      )}
      header={<ContextHeader title={pageMeta.headerTitle} subtitle={pageMeta.headerSubtitle} breadcrumbs={[{ label: '总览', current: true }]} sections={buildOverviewHeaderSections(focus)} />}
      notice={<GlobalNoticeLayer error={error} warnings={model?.surface.warnings} />}
      inspector={<InspectorRail title="对象检查器" focus={focus} model={model} context={{ domain: 'overview', surface: requestedSurface }} />}
    >
      {loading && !model ? <div className="loading-panel">加载总览中…</div> : null}
      {model ? <OverviewPage model={model} /> : null}
      {!model && !loading && requestedSurface === 'internal' ? (
        <SurfaceUnavailablePanel
          title="内部总览暂不可用"
          detail="内部数据面未就绪；当前总览不会回退到公开快照，请先刷新 internal snapshot 后重试。"
        />
      ) : null}
    </AppShell>
  );
}

function TerminalRoute({
  themePreference,
  resolvedTheme,
  onThemeChange,
  sidebarCollapsed,
  canCollapseSidebar,
  onToggleSidebarCollapse,
}: AppThemeProps) {
  const params = useParams();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const requested = params.surface === 'internal' ? 'internal' : 'public';
  const { model, loading, error, snapshotPath } = useTerminalStore();
  const focus = parseFocus(searchParams);
  const pageMeta = buildTerminalPageMeta(requested);

  useSurfaceRefresh(requested);

  const effectiveSurface = model?.surface.effective || requested;

  useEffect(() => {
    const anchorId = resolveSearchAnchorId(searchParams, focus)
      || (focus.panel && focus.section ? `terminal-${focus.panel}-${focus.section}` : focus.panel ? `terminal-${focus.panel}` : undefined);
    const handle = window.requestAnimationFrame(() => {
      if (anchorId && scrollToSearchAnchor(anchorId)) return;
      if (focus.panel || focus.section) safeResetWindowScroll();
    });
    return () => window.cancelAnimationFrame(handle);
  }, [focus, searchParams]);
  return (
    <AppShell
      sidebarCollapsed={sidebarCollapsed}
      topbar={<GlobalTopbar currentPath={location.pathname} primaryNav={buildPrimaryNav(focus, requested)} globalSummary={buildTopbarGlobalSummary(model)} themePreference={themePreference} resolvedTheme={resolvedTheme} onThemeChange={onThemeChange} onOpenSearch={() => window.dispatchEvent(new CustomEvent('fenlie-toggle-search'))} />}
      sidebar={(
        <ContextSidebar
          title={pageMeta.sidebarTitle}
          subtitle={pageMeta.sidebarSubtitle}
          description={pageMeta.sidebarDescription}
          items={buildOpsSidebarItems(focus)}
          utilityLinks={[{ label: '搜索 / Search', to: buildSearchLink() }]}
          collapsed={sidebarCollapsed}
          canCollapse={canCollapseSidebar}
          onToggleCollapse={onToggleSidebarCollapse}
        />
      )}
      header={<ContextHeader
        compact
        title={pageMeta.headerTitle}
        subtitle={pageMeta.headerSubtitle}
        breadcrumbs={[{ label: '总览' }, { label: '操作终端', current: true }]}
        actions={buildTerminalHeaderActions(focus, requested, effectiveSurface)}
      />}
      notice={<GlobalNoticeLayer error={error} warnings={model?.surface.warnings} />}
      inspector={<InspectorRail title="对象检查器" focus={focus} model={model} context={{ domain: 'terminal', surface: requested }} />}
    >
      {!model && loading ? <div className="loading-panel">{t('loading_terminal')}</div> : null}
      {model ? (
        <SurfaceShell>
          <OpsContextStrip model={model} focus={focus} requested={requested} snapshotPath={snapshotPath} />
          <AttentionStack />
          <OpsPage model={model} focus={focus} surface={requested} />
        </SurfaceShell>
      ) : null}
    </AppShell>
  );
}

function WorkspaceRoute({
  themePreference,
  resolvedTheme,
  onThemeChange,
  sidebarCollapsed,
  canCollapseSidebar,
  onToggleSidebarCollapse,
}: AppThemeProps) {
  const params = useParams();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const section = (params.section || 'artifacts') as WorkspaceSection;
  const { model, loading, error } = useTerminalStore();
  const focus = parseFocus(searchParams);
  const requestedSurface: SurfaceKey = focus.view === 'internal' ? 'internal' : 'public';
  const navigationFocus = resolveWorkspaceNavigationFocus(focus, model || undefined);
  const pageSections = model ? getWorkspacePageSections(section, model) : [];
  const activeSectionId = resolveWorkspacePageSectionId(
    searchParams,
    pageSections.map((item) => item.id),
    focus,
  );
  const anchorId = resolveSearchAnchorId(searchParams, focus);
  const pageMeta = buildWorkspacePageMeta(section);

  useSurfaceRefresh(requestedSurface, section, requestedSurface !== 'internal');

  useEffect(() => {
    const handle = window.requestAnimationFrame(() => {
      if (anchorId && scrollToSearchAnchor(anchorId)) return;
      if (activeSectionId) {
        scrollToWorkspaceAnchor(activeSectionId);
        return;
      }
      const fallbackSectionId = section !== 'artifacts' ? pageSections[0]?.id : undefined;
      if (fallbackSectionId) {
        const target = document.querySelector<HTMLElement>(`[data-workspace-section="${fallbackSectionId}"]`);
        if (safeScrollIntoView(target)) return;
      }
      safeResetWindowScroll();
    });
    return () => window.cancelAnimationFrame(handle);
  }, [activeSectionId, anchorId, location.pathname, pageSections, section]);

  return (
    <AppShell
      sidebarCollapsed={sidebarCollapsed}
      topbar={<GlobalTopbar currentPath={location.pathname} primaryNav={buildPrimaryNav(navigationFocus, requestedSurface)} globalSummary={buildTopbarGlobalSummary(model)} themePreference={themePreference} resolvedTheme={resolvedTheme} onThemeChange={onThemeChange} onOpenSearch={() => window.dispatchEvent(new CustomEvent('fenlie-toggle-search'))} />}
      sidebar={<ContextSidebar
        title={pageMeta.sidebarTitle}
        subtitle={pageMeta.sidebarSubtitle}
        description={pageMeta.sidebarDescription}
        items={buildResearchSidebarItems(navigationFocus, model || undefined)}
        pageSections={pageSections.map((item) => ({
          ...item,
          to: buildWorkspacePageLink(section, navigationFocus, item.id),
        }))}
        activeSectionId={activeSectionId}
        utilityLinks={[{ label: '搜索 / Search', to: buildSearchLink() }]}
        collapsed={sidebarCollapsed}
        canCollapse={canCollapseSidebar}
        onToggleCollapse={onToggleSidebarCollapse}
      />}
      header={<ContextHeader
        compact
        title={pageMeta.headerTitle}
        subtitle={pageMeta.headerSubtitle}
        breadcrumbs={[{ label: '总览' }, { label: '研究工作区', current: true }]}
        actions={buildWorkspaceHeaderActions(section, pageSections, activeSectionId)}
      />}
      notice={<GlobalNoticeLayer error={error} warnings={model?.surface.warnings} />}
      inspector={<InspectorRail title="对象检查器" focus={navigationFocus} model={model} context={{ domain: 'workspace', section, surface: requestedSurface }} />}
    >
      {!model && loading ? <div className="loading-panel">{t('loading_workspace')}</div> : null}
      {model ? (
        <>
          <WorkspaceContextStrip
            model={model}
            focus={navigationFocus}
            section={section}
            pageSectionId={activeSectionId}
          />
          <ResearchPage model={model} focus={navigationFocus} section={section} pageSectionId={activeSectionId} />
        </>
      ) : null}
      {!model && !loading && requestedSurface === 'internal' ? (
        <SurfaceUnavailablePanel
          title={`内部${pageMeta.headerTitle}暂不可用`}
          detail="内部数据面未就绪；当前工作区不会回退到公开快照，请先刷新 internal snapshot 后重试。"
        />
      ) : null}
    </AppShell>
  );
}

function SearchRoute({
  themePreference,
  resolvedTheme,
  onThemeChange,
  sidebarCollapsed,
  canCollapseSidebar,
  onToggleSidebarCollapse,
}: AppThemeProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { model, loading, error } = useTerminalStore();
  const requestedSurface = model?.surface.requested || 'public';
  useSurfaceRefresh(requestedSurface, 'search', requestedSurface !== 'internal');
  const query = searchParams.get('q') || '';
  const scope = parseSearchScope(searchParams.get('scope'));

  const updateSearchParams = (nextQuery: string, nextScope: SearchScope) => {
    const next = new URLSearchParams();
    if (nextQuery.trim()) next.set('q', nextQuery.trim());
    if (nextScope !== 'all') next.set('scope', nextScope);
    setSearchParams(next, { replace: true });
  };

  return (
    <AppShell
      sidebarCollapsed={sidebarCollapsed}
      topbar={<GlobalTopbar currentPath={location.pathname} primaryNav={buildPrimaryNav({}, requestedSurface)} globalSummary={buildTopbarGlobalSummary(model)} themePreference={themePreference} resolvedTheme={resolvedTheme} onThemeChange={onThemeChange} onOpenSearch={() => window.dispatchEvent(new CustomEvent('fenlie-toggle-search'))} />}
      sidebar={(
        <ContextSidebar
          title="全局搜索"
          subtitle="跨页面检索"
          description="模块 / 路由 / 工件统一检索，点击结果后跳到具体页面或阶段。"
          items={PRIMARY_NAV.map((item) => ({ ...item, to: item.id === 'overview' ? buildOverviewLink({}) : item.id === 'ops' ? buildOpsLegacyLink('public', {}) : buildResearchLegacyLink('artifacts', {}) }))}
          utilityLinks={[{ label: '搜索 / Search', to: buildSearchLink(query, scope) }]}
          collapsed={sidebarCollapsed}
          canCollapse={canCollapseSidebar}
          onToggleCollapse={onToggleSidebarCollapse}
        />
      )}
      header={<ContextHeader compact title="全局搜索" subtitle="跨页面、跨模块、跨工件的本地索引搜索" breadcrumbs={[{ label: '总览' }, { label: '搜索', current: true }]} sections={buildSearchHeaderSections()} />}
      notice={<GlobalNoticeLayer error={error} warnings={model?.surface.warnings} />}
      inspector={<InspectorRail title="对象检查器" focus={{}} model={model} context={{ domain: 'overview', surface: requestedSurface }} />}
    >
      {loading && !model ? <div className="loading-panel">加载搜索索引中…</div> : null}
      <SearchPage
        model={model}
        query={query}
        scope={scope}
        onQueryChange={(next) => updateSearchParams(next, scope)}
        onScopeChange={(next) => updateSearchParams(query, next)}
        onPick={(entry) => navigate(entry.destination)}
      />
    </AppShell>
  );
}

function AppRoutes() {
  const location = useLocation();
  const navigate = useNavigate();
  const themeSearchParams = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const themeOverride = themeSearchParams.get('theme');
  const { preference, resolvedTheme, setPreference } = useUiTheme(themeOverride);
  const {
    collapsed: sidebarCollapsed,
    canCollapse: canCollapseSidebar,
    toggleCollapsed: onToggleSidebarCollapse,
  } = useSidebarCollapse();
  const { model } = useTerminalStore();
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchScope, setSearchScope] = useState<SearchScope>('all');

  useEffect(() => {
    const onToggle = () => setSearchOpen((prev) => !prev);
    window.addEventListener('fenlie-toggle-search', onToggle);
    return () => window.removeEventListener('fenlie-toggle-search', onToggle);
  }, []);

  useEffect(() => {
    setSearchOpen(false);
  }, [location.pathname, location.search]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const isModifier = event.metaKey || event.ctrlKey;
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      const isInput = tag === 'input' || tag === 'textarea' || tag === 'select' || Boolean(target?.isContentEditable);
      if (!isModifier || event.key.toLowerCase() !== 'k') return;
      if (isInput && !searchOpen) return;
      event.preventDefault();
      setSearchOpen((prev) => !prev);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [searchOpen]);

  const handleThemeChange = useCallback((next: ThemePreference) => {
    setPreference(next);
    if (!themeOverride) return;
    const nextParams = new URLSearchParams(location.search);
    nextParams.set('theme', next);
    navigate(
      {
        pathname: location.pathname,
        search: `?${nextParams.toString()}`,
      },
      { replace: true },
    );
  }, [location.pathname, location.search, navigate, setPreference, themeOverride]);

  return (
    <>
      <Routes>
        <Route path="/" element={<Navigate to="/overview" replace />} />
        <Route
          path="/overview"
          element={(
            <OverviewRoute
              themePreference={preference}
              resolvedTheme={resolvedTheme}
              onThemeChange={handleThemeChange}
              sidebarCollapsed={sidebarCollapsed}
              canCollapseSidebar={canCollapseSidebar}
              onToggleSidebarCollapse={onToggleSidebarCollapse}
            />
          )}
        />
        <Route
          path="/terminal/:surface"
          element={(
            <TerminalRoute
              themePreference={preference}
              resolvedTheme={resolvedTheme}
              onThemeChange={handleThemeChange}
              sidebarCollapsed={sidebarCollapsed}
              canCollapseSidebar={canCollapseSidebar}
              onToggleSidebarCollapse={onToggleSidebarCollapse}
            />
          )}
        />
        <Route
          path="/search"
          element={(
            <SearchRoute
              themePreference={preference}
              resolvedTheme={resolvedTheme}
              onThemeChange={handleThemeChange}
              sidebarCollapsed={sidebarCollapsed}
              canCollapseSidebar={canCollapseSidebar}
              onToggleSidebarCollapse={onToggleSidebarCollapse}
            />
          )}
        />
        <Route
          path="/workspace/:section"
          element={(
            <WorkspaceRoute
              themePreference={preference}
              resolvedTheme={resolvedTheme}
              onThemeChange={handleThemeChange}
              sidebarCollapsed={sidebarCollapsed}
              canCollapseSidebar={canCollapseSidebar}
              onToggleSidebarCollapse={onToggleSidebarCollapse}
            />
          )}
        />
        <Route path="*" element={<Navigate to="/overview" replace />} />
      </Routes>
      {model ? (
        <GlobalSearchOverlay
          isOpen={searchOpen}
          model={model}
          query={searchQuery}
          scope={searchScope}
          onClose={() => setSearchOpen(false)}
          onQueryChange={setSearchQuery}
          onScopeChange={setSearchScope}
        />
      ) : null}
    </>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <HashRouter>
        <AppRoutes />
      </HashRouter>
    </ErrorBoundary>
  );
}
