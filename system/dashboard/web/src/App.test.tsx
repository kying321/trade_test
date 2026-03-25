import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ResolvedTheme, ThemePreference } from './app-shell/GlobalTopbar';
import App from './App';
import { buildTerminalReadModel } from './adapters/read-model';
import { useTerminalStore } from './store/use-terminal-store';
import type { DashboardSnapshot, LoadedSurface, SurfaceKey } from './types/contracts';

const ACT_WARNING_FRAGMENT = 'not wrapped in act';
const PUBLIC_SNAPSHOT_PATH = '/data/fenlie_dashboard_snapshot.json';
const INTERNAL_SNAPSHOT_PATH = '/data/fenlie_dashboard_internal_snapshot.json';
const DEFAULT_REFRESH = useTerminalStore.getState().refresh;
const DEFAULT_SET_SURFACE = useTerminalStore.getState().setSurface;
let actWarnings: string[] = [];
let unexpectedConsoleErrors: string[] = [];
let consoleErrorSpy: ReturnType<typeof vi.spyOn> | null = null;
let mockedThemePreference: ThemePreference = 'system';
let mockedResolvedTheme: ResolvedTheme = 'dark';
const mockSetThemePreference = vi.fn();

vi.mock('./hooks/use-ui-theme', () => ({
  useUiTheme: () => ({
    preference: mockedThemePreference,
    resolvedTheme: mockedResolvedTheme,
    setPreference: mockSetThemePreference,
  }),
}));

function textOf(selector: string): string {
  return document.querySelector(selector)?.textContent?.replace(/\s+/g, ' ').trim() || '';
}

async function renderApp(): Promise<ReturnType<typeof render>> {
  let view: ReturnType<typeof render> | null = null;
  await act(async () => {
    view = render(<App />);
  });
  if (!view) throw new Error('renderApp failed to mount App');
  return view;
}

async function navigateHash(hash: string) {
  await act(async () => {
    window.location.hash = hash;
    window.dispatchEvent(new HashChangeEvent('hashchange'));
  });
}

async function clickAndFlush(target: HTMLElement) {
  await act(async () => {
    fireEvent.click(target);
  });
}

function mockClampOverflow({ overflows }: { overflows: boolean }) {
  const keys = ['scrollHeight', 'clientHeight', 'scrollWidth', 'clientWidth'] as const;
  const originals = new Map(
    keys.map((key) => [key, Object.getOwnPropertyDescriptor(HTMLElement.prototype, key)]),
  );

  Object.defineProperty(HTMLElement.prototype, 'scrollHeight', {
    configurable: true,
    get(this: HTMLElement) {
      if (this.classList.contains('clamp-copy')) return overflows ? 64 : 32;
      return 32;
    },
  });
  Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
    configurable: true,
    get(this: HTMLElement) {
      if (this.classList.contains('clamp-copy')) return 32;
      return 32;
    },
  });
  Object.defineProperty(HTMLElement.prototype, 'scrollWidth', {
    configurable: true,
    get(this: HTMLElement) {
      if (this.classList.contains('clamp-copy')) return overflows ? 220 : 120;
      return 120;
    },
  });
  Object.defineProperty(HTMLElement.prototype, 'clientWidth', {
    configurable: true,
    get(this: HTMLElement) {
      if (this.classList.contains('clamp-copy')) return 120;
      return 120;
    },
  });

  return () => {
    keys.forEach((key) => {
      const descriptor = originals.get(key);
      if (descriptor) {
        Object.defineProperty(HTMLElement.prototype, key, descriptor);
      } else {
        delete (HTMLElement.prototype as Partial<typeof HTMLElement.prototype>)[key];
      }
    });
  };
}

async function changeAndFlush(target: HTMLElement, value: string) {
  await act(async () => {
    fireEvent.change(target, { target: { value } });
  });
}

const mockSnapshot = {
  status: 'ok',
  change_class: 'RESEARCH_ONLY',
  meta: {
    generated_at_utc: '2026-03-20T10:00:00Z',
    artifact_payload_count: 10,
    catalog_count: 12,
    backtest_artifact_count: 2,
    change_class: 'RESEARCH_ONLY',
  },
  workspace_default_focus: {
    artifact: 'price_action_breakout_pullback',
    group: 'research_mainline',
    panel: 'lab-review',
    section: 'research-heads',
    search_scope: 'title',
  },
  ui_routes: {
    terminal_public: '#/terminal/public',
    terminal_internal: '#/terminal/internal',
    workspace_artifacts: '#/workspace/artifacts',
    operator_panel: '/operator_task_visual_panel.html',
    snapshot_json_internal: '/data/fenlie_dashboard_internal_snapshot.json',
  },
  experience_contract: {
    mode: 'read_only_snapshot',
    live_path_touched: false,
    fallback_chain: ['#/workspace/raw', '/data/fenlie_dashboard_snapshot.json', '/operator_task_visual_panel.html'],
  },
  public_topology: [
    {
      id: 'fenlie_root',
      label: 'Fenlie 主入口',
      role: '主公开入口',
      url: 'https://fuuu.fun',
      expected_status: '200 / root-nav-proxy',
      source: 'Cloudflare Worker route',
      fallback: 'Pages 回退入口',
      notes: '根域现行入口',
    },
    {
      id: 'fenlie_pages',
      label: 'Pages 回退入口',
      role: '公开回退入口',
      url: 'https://fenlie.fuuu.fun',
      expected_status: '200 / pages origin',
      source: 'Cloudflare Pages',
      fallback: 'Fenlie 主入口',
      notes: 'Pages 直接回退',
    },
    {
      id: 'openclaw_legacy',
      label: '旧 OpenClaw',
      role: '旧控制页回退',
      url: 'http://43.153.148.242:3001',
      expected_status: '200 / OpenClaw Control',
      source: '远端静态服务',
      fallback: '无',
      notes: '旧页保留端口',
    },
    {
      id: 'gateway_api',
      label: 'gateway/API',
      role: '执行网关探针',
      url: 'http://43.153.148.242:8787',
      expected_status: '404 / API root',
      source: 'pi-openclaw.service',
      fallback: '无',
      notes: '当前不是前端页',
    },
  ],
  surface_contracts: {
    public: { label: '公开终端', availability: 'active', redaction_level: 'public_summary' },
    internal: { label: '内部终端', availability: 'local_only', redaction_level: 'internal_local_only', deploy_policy: 'strip_from_public_dist' },
  },
  interface_catalog: [
    { id: 'terminal_public', label: '公开终端', kind: 'public', url: '#/terminal/public', description: '主入口' },
    { id: 'snapshot_json_internal', label: '内部快照 JSON', kind: 'local-private', url: '/data/fenlie_dashboard_internal_snapshot.json', description: '本地 only' },
    { id: 'frontend_public', label: '公开看板', kind: 'public', url: 'https://fuuu.fun', description: 'Cloudflare 根域公开入口' },
    { id: 'frontend_dev', label: '本地预览', kind: 'local', url: 'http://127.0.0.1:5173', description: '本地 Vite 预览入口' },
  ],
  source_heads: {
    system_scheduler_state: { label: '调度与心跳', status: 'ok', summary: 'micro_capture ok', path: '/tmp/scheduler.json' },
    price_action_breakout_pullback: { label: 'ETH 主线', status: 'mixed_positive', summary: 'ETHUSDT 15m breakout-pullback', path: '/tmp/eth.json' },
    hold_selection_handoff: {
      label: '持有选择主头',
      status: 'ok',
      summary: 'hold16 基线 + hold8 局部候选 + hold24 收益观察',
      path: 'review/latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json',
      active_baseline: 'hold16_zero',
      local_candidate: 'hold8_zero',
      return_candidate: [],
      return_watch: ['hold24_zero'],
      transfer_watch: ['hold12_zero'],
    },
  },
  domain_summary: [
    { id: 'orchestration', label: 'orchestration', count: 3, payload_count: 3, positive: 2, warning: 1, negative: 0, headline: 'operator panel' },
    { id: 'data-regime', label: 'data-regime', count: 1, payload_count: 1, positive: 1, warning: 0, negative: 0, headline: 'stress matrix' },
  ],
  backtest_artifacts: [
    {
      id: 'bt1',
      label: 'backtest_eth_15m',
      start: '2024-01-01',
      end: '2026-03-19',
      total_return: 0.12,
      annual_return: 0.04,
      max_drawdown: -0.05,
      profit_factor: 1.6,
      trades: 42,
      equity_curve_sample: [{ equity: 1 }, { equity: 1.1 }, { equity: 1.06 }],
    },
  ],
  artifact_payloads: {
    operator_panel: {
      label: 'operator panel',
      path: '/tmp/operator-panel.json',
      summary: { status: 'ok', generated_at_utc: '2026-03-20T10:00:00Z' },
      payload: {
        status: 'ok',
        generated_at_utc: '2026-03-20T10:00:00Z',
        summary: {
          operator_head_brief: 'waiting:ETHUSDT',
          remote_live_gate_brief: 'read_only_surface',
          remote_live_clearing_brief: 'none',
          lane_state_brief: 'waiting=1',
          lane_priority_order_brief: 'waiting@99:1',
          priority_repair_plan_brief: 'shadow_only',
          priority_repair_verification_brief: 'cleared',
          remote_live_diagnosis_brief: 'research_only',
          remote_live_history_brief: '7d:12tr',
          openclaw_guardian_clearance_status: 'blocked',
          openclaw_guardian_clearance_score: 35,
          openclaw_canary_gate_status: 'blocked',
          openclaw_canary_gate_decision: 'deny_canary_until_guardian_clear',
          openclaw_promotion_gate_status: 'blocked',
          openclaw_promotion_gate_decision: 'clear_guardian_review_blockers_before_promotion',
          openclaw_live_boundary_hold_status: 'active',
          openclaw_live_boundary_hold_decision: 'keep_shadow_transport_only',
          openclaw_orderflow_policy_status: 'shadow_policy_blocked',
          openclaw_orderflow_policy_decision: 'reject_until_guardian_clear',
          openclaw_execution_ack_status: 'shadow_no_send_ack_recorded',
          openclaw_execution_ack_decision: 'record_reject_without_transport',
          openclaw_quality_report_score: 44,
          openclaw_quality_shadow_learning_score: 60,
          openclaw_quality_execution_readiness_score: 0,
        },
        lane_cards: [{ state: 'waiting', count: 1, priority_total: 99, brief: 'ETHUSDT', head_symbol: 'ETHUSDT', head_action: 'wait' }],
        focus_slots: [{ slot: 'primary', symbol: 'ETHUSDT', action: 'wait_breakout_pullback', state: 'waiting', done_when: 'edge confirmed' }],
        control_chain: [{ label: '研究', source_status: 'ok', source_decision: 'keep', next_target_artifact: 'price_action_breakout_pullback' }],
        action_queue: [{ symbol: 'ETHUSDT', action: 'wait_breakout_pullback', state: 'waiting', reason: 'setup in progress' }],
        action_checklist: [{ symbol: 'ETHUSDT', action: 'verify oos', state: 'review', reason: 'hold 8 vs 16', done_when: 'forward window done' }],
        priority_repair_plan: { status: 'watch', done_when: 'none' },
        continuous_optimization_backlog: [{ stage: 'research', title: 'validate max_hold_bars', change_class: 'RESEARCH_ONLY', why: 'ETH 15m' }],
      },
    },
    system_scheduler_state: {
      label: 'scheduler',
      path: '/tmp/scheduler.json',
      summary: { status: 'ok' },
      payload: {
        interval_tasks: {
          micro_capture: {
            enabled: true,
            interval_minutes: 30,
            last_run_ts: '2026-03-20T10:00:00Z',
            status: 'ok',
            symbols: ['BTCUSDT', 'ETHUSDT'],
          },
        },
        history: [
          {
            ts: '2026-03-20T10:00:00Z',
            slot: 'micro-capture',
            trigger_time: 'every_30m',
            status: 'ok',
            result: {
              captured_at_utc: '2026-03-20T10:00:00Z',
              summary: {
                selected_schema_ok_ratio: 1,
                selected_time_sync_ok_ratio: 1,
                selected_sync_ok_ratio: 1,
                cross_source_status: 'ok',
                cross_source_fail_ratio: 0,
                cross_source_fail_ratio_limit: 0.5,
                cross_source_symbols_audited: 2,
                cross_source_symbols_insufficient: 0,
              },
              rolling_7d: { run_count: 4, pass_ratio: 0.75, avg_cross_source_fail_ratio: 0.1 },
            },
          },
        ],
      },
    },
    architecture_audit: {
      label: 'architecture audit',
      path: '/tmp/architecture.json',
      summary: { status: 'pass' },
      payload: {
        status: 'healthy',
        health: { status: 'healthy', checks: { daily_briefing: true, daily_signals: true, daily_positions: true, sqlite: true } },
        manifests: { eod_manifest: true, backtest_manifest: true, review_manifest: true },
      },
    },
    mode_stress_matrix: {
      label: 'stress matrix',
      path: '/tmp/stress.json',
      summary: { status: 'ok' },
      payload: {
        best_mode: 'ultra_short',
        mode_summary: [{ mode: 'ultra_short', robustness_score: 81, avg_annual_return: 0.12, worst_drawdown: -0.08, avg_profit_factor: 1.4, avg_win_rate: 0.55, total_violations: 0 }],
        matrix: [{ mode: 'ultra_short', window: '2020_pandemic', status: 'ok', annual_return: 0.08, max_drawdown: -0.05 }],
      },
    },
    recent_strategy_backtests: {
      label: 'recent backtests',
      path: '/tmp/recent.json',
      summary: { status: 'ok', research_decision: 'keep_best' },
      payload: { rows: [{ strategy_id_canonical: 'eth_breakout_pullback', market_scope: 'crypto', total_return: 0.18, profit_factor: 1.8, recommendation: 'keep' }] },
    },
    price_action_breakout_pullback: {
      label: 'eth mainline',
      path: '/tmp/eth.json',
      summary: { status: 'ok', change_class: 'SIM_ONLY', research_decision: 'mixed_positive', takeaway: 'keep ETH 15m breakout-pullback' },
      payload: {},
    },
    price_action_exit_risk: {
      label: 'exit risk',
      path: '/tmp/exit.json',
      summary: { status: 'warning', change_class: 'SIM_ONLY', research_decision: 'forward_compare_8_vs_16', takeaway: 'continue forward window' },
      payload: {},
    },
    cross_section_backtest: {
      label: 'cross section',
      path: '/tmp/cross.json',
      summary: { status: 'warning', change_class: 'SIM_ONLY', research_decision: 'no_edge', takeaway: 'deprioritize cross section' },
      payload: {},
    },
    hold_selection_handoff: {
      label: 'hold selection handoff',
      path: 'review/latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json',
      summary: { status: 'ok', research_decision: 'use_hold_selection_gate_as_canonical_head' },
      payload: {
        active_baseline: 'hold16_zero',
        local_candidate: 'hold8_zero',
        transfer_watch: ['hold12_zero'],
        return_candidate: [],
        return_watch: ['hold24_zero'],
        demoted_candidate: ['hold24_zero', 'pullback_depth_atr_router'],
        source_head_status: 'gate_override_active',
        recommended_brief: 'ETHUSDT:hold_selection_handoff:baseline=hold16_zero,return_watch=hold24_zero',
      },
    },
  },
  catalog: [
    { id: 'operator_panel', payload_key: 'operator_panel', artifact_layer: 'canonical', artifact_group: 'system_anchor', category: 'operator-panel', label: 'operator_panel', status: 'ok' },
    { id: 'price_action_breakout_pullback', payload_key: 'price_action_breakout_pullback', artifact_layer: 'canonical', artifact_group: 'research_mainline', category: 'sim-only', label: 'price_action_breakout_pullback', status: 'ok' },
    { id: 'price_action_exit_risk', payload_key: 'price_action_exit_risk', artifact_layer: 'canonical', artifact_group: 'research_mainline', category: 'sim-only', label: 'price_action_exit_risk', status: 'warning' },
    { id: 'price_action_exit_hold_robustness', payload_key: 'price_action_exit_hold_robustness', artifact_layer: 'canonical', artifact_group: 'research_exit_risk', category: 'sim-only', label: 'price_action_exit_hold_robustness', status: 'ok' },
  ],
};

const workspaceArtifactsSmokeSnapshot = {
  ...mockSnapshot,
  catalog: [
    ...mockSnapshot.catalog,
    {
      id: 'hold_selection_handoff',
      payload_key: 'hold_selection_handoff',
      artifact_layer: 'canonical',
      artifact_group: 'research_hold_transfer',
      category: 'research',
      label: 'hold_selection_handoff',
      status: 'ok',
      path: 'review/latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json',
    },
    {
      id: 'cross_section_backtest',
      payload_key: 'cross_section_backtest',
      artifact_layer: 'canonical',
      artifact_group: 'research_cross_section',
      category: 'backtest',
      label: 'cross_section_backtest',
      status: 'warning',
      path: 'review/20260315T105000Z_crypto_shortline_cross_section_backtest.json',
    },
    {
      id: 'intraday_orderflow_blueprint',
      payload_key: 'intraday_orderflow_blueprint',
      artifact_layer: 'canonical',
      artifact_group: 'research_cross_section',
      category: 'research',
      label: 'intraday_orderflow_blueprint',
      status: 'ok',
      path: 'review/latest_intraday_orderflow_blueprint.json',
    },
    {
      id: 'intraday_orderflow_research_gate_blocker',
      payload_key: 'intraday_orderflow_research_gate_blocker',
      artifact_layer: 'canonical',
      artifact_group: 'research_cross_section',
      category: 'research',
      label: 'intraday_orderflow_research_gate_blocker',
      status: 'warning',
      path: 'review/latest_intraday_orderflow_research_gate_blocker_report.json',
    },
    {
      id: '20260318T163800Z_price_action_breakout_pullback_sim_only',
      artifact_layer: 'archive',
      category: 'sim-only',
      label: '20260318T163800Z_price_action_breakout_pullback_sim_only',
      status: 'ok',
      path: 'review/20260318T163800Z_price_action_breakout_pullback_sim_only.json',
    },
  ],
};

const feedbackProjection = {
  status: 'ok',
  change_class: 'RESEARCH_ONLY',
  visibility: 'internal_only',
  summary: {
    alignment_score: 76,
    blocker_pressure: 24,
    execution_clarity: 69,
    readability_pressure: 33,
    drift_state: 'watch',
    headline: '保持前端信息架构与用户高价值反馈对齐',
  },
  events: [
    {
      feedback_id: 'ui_density_split',
      created_at_utc: '2026-03-20T12:00:00Z',
      source: 'manual',
      domain: 'research',
      headline: '拆分研究工作区左栏四控件',
      summary: '研究地图 / 状态雷达 / 检索定位 / 当前焦点',
      recommended_action: '完成左栏职责拆分',
      impact_score: 91,
      confidence: 0.89,
      status: 'active',
      anchors: [
        {
          route: '/workspace/artifacts',
          artifact: 'price_action_breakout_pullback',
          component: 'WorkspacePanels',
        },
      ],
    },
  ],
  actions: [
    {
      feedback_id: 'ui_density_split',
      recommended_action: '完成左栏职责拆分',
      impact_score: 91,
      anchors: [
        {
          route: '/workspace/artifacts',
          artifact: 'price_action_breakout_pullback',
          component: 'WorkspacePanels',
        },
      ],
    },
  ],
  trends: [
    {
      feedback_id: 'ui_density_split',
      created_at_utc: '2026-03-20T12:00:00Z',
      alignment_score: 76,
      blocker_pressure: 24,
      execution_clarity: 69,
      readability_pressure: 33,
      drift_state: 'watch',
    },
  ],
  anchors: [
    {
      route: '/workspace/artifacts',
      artifact: 'price_action_breakout_pullback',
      component: 'WorkspacePanels',
    },
  ],
};

function buildFeedbackSnapshot(overrides: Partial<DashboardSnapshot> = {}): DashboardSnapshot {
  return {
    ...(mockSnapshot as DashboardSnapshot),
    conversation_feedback_projection: feedbackProjection,
    ...overrides,
  };
}

type SurfaceFixture = {
  snapshot: DashboardSnapshot;
  effectiveSurface?: SurfaceKey;
  snapshotPath?: string;
  warnings?: string[];
};

function buildLoadedSurface(requestedSurface: SurfaceKey, fixture: SurfaceFixture): LoadedSurface {
  const effectiveSurface = fixture.effectiveSurface ?? requestedSurface;
  return {
    snapshot: fixture.snapshot,
    requestedSurface,
    effectiveSurface,
    snapshotPath: fixture.snapshotPath || (effectiveSurface === 'internal' ? INTERNAL_SNAPSHOT_PATH : PUBLIC_SNAPSHOT_PATH),
    warnings: fixture.warnings || [],
  };
}

function primeStore(requestedSurface: SurfaceKey, fixture: SurfaceFixture) {
  const loaded = buildLoadedSurface(requestedSurface, fixture);
  useTerminalStore.setState({
    requestedSurface,
    effectiveSurface: loaded.effectiveSurface,
    snapshotPath: loaded.snapshotPath,
    snapshot: loaded.snapshot,
    model: buildTerminalReadModel(loaded),
    loading: false,
    error: '',
    refresh: useTerminalStore.getState().refresh,
    setSurface: DEFAULT_SET_SURFACE,
  });
}

function installLiveRefresh() {
  useTerminalStore.setState({
    requestedSurface: 'public',
    effectiveSurface: 'public',
    snapshotPath: PUBLIC_SNAPSHOT_PATH,
    snapshot: null,
    model: null,
    loading: false,
    error: '',
    refresh: DEFAULT_REFRESH,
    setSurface: DEFAULT_SET_SURFACE,
  });
}

function installPassiveStore(snapshot: DashboardSnapshot = mockSnapshot as DashboardSnapshot) {
  const loaded = buildLoadedSurface('public', { snapshot });
  useTerminalStore.setState({
    requestedSurface: 'public',
    effectiveSurface: loaded.effectiveSurface,
    snapshotPath: loaded.snapshotPath,
    snapshot: loaded.snapshot,
    model: buildTerminalReadModel(loaded),
    loading: false,
    error: '',
    refresh: vi.fn(async () => undefined),
    setSurface: DEFAULT_SET_SURFACE,
  });
}

function installPassiveStoreForSurface(requestedSurface: SurfaceKey, fixture: SurfaceFixture) {
  const loaded = buildLoadedSurface(requestedSurface, fixture);
  useTerminalStore.setState({
    requestedSurface,
    effectiveSurface: loaded.effectiveSurface,
    snapshotPath: loaded.snapshotPath,
    snapshot: loaded.snapshot,
    model: buildTerminalReadModel(loaded),
    loading: false,
    error: '',
    refresh: vi.fn(async () => undefined),
    setSurface: DEFAULT_SET_SURFACE,
  });
}

async function primeSurface(requestedSurface: SurfaceKey, fixture: SurfaceFixture) {
  await act(async () => {
    primeStore(requestedSurface, fixture);
  });
}

function buildAcceptanceSnapshot() {
  return {
    ...mockSnapshot,
    artifact_payloads: {
      ...mockSnapshot.artifact_payloads,
      dashboard_public_acceptance: {
        label: 'dashboard public acceptance',
        path: 'review/20260320T072734Z_dashboard_public_acceptance.json',
        summary: {
          status: 'ok',
          change_class: 'RESEARCH_ONLY',
          generated_at_utc: '2026-03-20T07:27:34Z',
        },
        payload: {
          status: 'ok',
          change_class: 'RESEARCH_ONLY',
          generated_at_utc: '2026-03-20T07:27:34Z',
          report_path: 'review/20260320T072734Z_dashboard_public_acceptance.json',
          checks: {
            topology_smoke: {
              status: 'ok',
              generated_at_utc: '2026-03-20T07:27:29Z',
              report_path: 'review/20260320T072724Z_dashboard_public_topology_smoke.json',
              returncode: 0,
              tls_policy: {
                allow_insecure_tls_fallback: true,
              },
              entrypoints: {
                root_url: 'https://fuuu.fun',
                pages_url: 'https://fenlie.fuuu.fun',
              },
              checks: {
                root_public: {
                  header_public_entry: 'root-nav-proxy',
                },
                root_snapshot: {
                  frontend_public: 'https://fuuu.fun',
                },
                root_overview_browser: {
                  screenshot_path: 'review/root_overview_browser.png',
                },
                pages_overview_browser: {
                  screenshot_path: 'review/pages_overview_browser.png',
                },
                root_contracts_browser: {
                  screenshot_path: 'review/root_contracts_browser.png',
                },
                pages_contracts_browser: {
                  screenshot_path: 'review/pages_contracts_browser.png',
                },
              },
            },
            workspace_routes_smoke: {
              status: 'ok',
              generated_at_utc: '2026-03-20T07:27:34Z',
              report_path: 'review/20260320T072729Z_dashboard_workspace_routes_browser_smoke.json',
              returncode: 0,
              surface_assertion: {
                requested_surface: 'public',
                effective_surface: 'public',
                snapshot_endpoint_observed: '/data/fenlie_dashboard_snapshot.json',
              },
              network_observation: {
                public_snapshot_fetch_count: 4,
                internal_snapshot_fetch_count: 0,
              },
              artifacts_filter_assertion: {
                route: '#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow',
                group: 'research_cross_section',
                search_scope: 'title',
                search: 'orderflow',
                active_artifact: 'intraday_orderflow_blueprint',
                visible_artifacts: [
                  'intraday_orderflow_blueprint',
                  'intraday_orderflow_research_gate_blocker',
                ],
              },
            },
          },
          subcommands: {
            topology_smoke: {
              returncode: 1,
              payload_present: false,
              stdout_bytes: 19,
              stderr_bytes: 7,
              cmd: ['python3', 'run_dashboard_public_topology_smoke.py'],
              cwd: '/tmp',
              stdout: 'root_public timeout',
              stderr: 'timeout',
            },
            workspace_routes_smoke: {
              returncode: 0,
              payload_present: true,
              stdout_bytes: 120,
              stderr_bytes: 0,
              cmd: ['python3', 'run_dashboard_workspace_artifacts_smoke.py'],
              cwd: '/tmp',
            },
          },
        },
      },
    },
  };
}

describe('Fenlie terminal console', () => {
  beforeEach(() => {
    window.location.hash = '#/terminal/public';
    window.localStorage.clear();
    actWarnings = [];
    unexpectedConsoleErrors = [];
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation((...args) => {
      const message = args.map((arg) => {
        if (typeof arg === 'string') return arg;
        if (arg instanceof Error) return arg.message;
        return String(arg);
      }).join(' ');
      if (message.includes(ACT_WARNING_FRAGMENT)) {
        actWarnings.push(message);
        return;
      }
      unexpectedConsoleErrors.push(message);
    });
    mockedThemePreference = 'system';
    mockedResolvedTheme = 'dark';
    mockSetThemePreference.mockReset();
    document.documentElement.dataset.theme = mockedResolvedTheme;
    vi.stubGlobal('requestAnimationFrame', ((callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    }) as typeof window.requestAnimationFrame);
    vi.stubGlobal('cancelAnimationFrame', vi.fn());
    installPassiveStore();
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      json: async () => mockSnapshot,
      status: 200,
    })));
  });

  afterEach(() => {
    cleanup();
    consoleErrorSpy?.mockRestore();
    expect(unexpectedConsoleErrors).toEqual([]);
    expect(actWarnings).toEqual([]);
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('renders overview as a top-level shell route with primary entry cards', async () => {
    window.location.hash = '#/overview';
    await renderApp();

    expect(await screen.findByRole('heading', { name: '总览' })).toBeTruthy();
    expect(screen.getAllByText('操作终端').length).toBeGreaterThan(0);
    expect(screen.getAllByText('研究工作区').length).toBeGreaterThan(0);
    expect(screen.getAllByText('全局导航').length).toBeGreaterThan(0);
    expect(screen.getAllByText('系统状态 / 入口 / 路由总览').length).toBeGreaterThan(0);
    expect(screen.getByRole('heading', { name: '研究主线摘要' })).toBeTruthy();
    const summaryCard = screen.getByRole('heading', { name: '研究主线摘要' }).closest('section');
    expect(summaryCard?.textContent || '').toContain('hold16_zero');
    expect(summaryCard?.textContent || '').toContain('hold8_zero');
    expect(summaryCard?.textContent || '').toContain('hold12_zero');
    expect(summaryCard?.textContent || '').toContain('hold24_zero');
    const summaryLink = screen.getByRole('link', { name: '查看源头主线' });
    expect(summaryLink.getAttribute('href')).toContain('/workspace/contracts');
    expect(summaryLink.getAttribute('href')).toContain('page_section=contracts-source-heads');
    expect(textOf('.global-topbar-inner')).not.toContain('请求视图');
    expect(textOf('.global-topbar-inner')).not.toContain('实际视图');
    expect(textOf('.global-summary-strip')).toContain('变更级别：仅研究');
  });

  it('renders internal feedback summary on overview when view=internal', async () => {
    installPassiveStoreForSurface('internal', {
      snapshot: buildFeedbackSnapshot(),
      effectiveSurface: 'internal',
      snapshotPath: INTERNAL_SNAPSHOT_PATH,
    });
    window.location.hash = '#/overview?view=internal';

    await renderApp();

    expect(await screen.findByRole('heading', { name: '方向对齐投射' })).toBeTruthy();
    const card = screen.getByRole('heading', { name: '方向对齐投射' }).closest('section');
    expect(card?.textContent || '').toContain('保持前端信息架构与用户高价值反馈对齐');
    expect(card?.textContent || '').toContain('alignment_score');
    const link = screen.getByRole('link', { name: '进入对齐页' });
    expect(link.getAttribute('href')).toContain('/workspace/alignment');
    expect(link.getAttribute('href')).toContain('view=internal');
  });

  it('supports system / dark / light theme tri-state with url override and persisted preference', async () => {
    mockedThemePreference = 'light';
    mockedResolvedTheme = 'light';
    document.documentElement.dataset.theme = 'light';
    window.location.hash = '#/overview';
    await renderApp();

    await screen.findByRole('heading', { name: '总览' });
    expect(document.documentElement.dataset.theme).toBe('light');

    await clickAndFlush(screen.getByRole('button', { name: '夜间主题' }) as HTMLElement);
    expect(mockSetThemePreference).toHaveBeenCalledWith('dark');

    await navigateHash('#/workspace/contracts?theme=light');

    await screen.findByText('接口分层 / 可见边界');
    expect(document.documentElement.dataset.theme).toBe('light');
    expect(screen.getByRole('button', { name: '白天主题' }).getAttribute('aria-pressed')).toBe('true');
  });

  it('renders terminal and workspace routes with the new TS shell', async () => {
    await renderApp();

    expect(screen.getByText('Fenlie / 控制台')).toBeTruthy();
    expect(screen.getAllByText('操作终端').length).toBeGreaterThan(0);
    expect(screen.getAllByText('公开 / 内部 / 执行链').length).toBeGreaterThan(0);
    expect(screen.getAllByText('公开面').length).toBeGreaterThan(0);
    expect(screen.getAllByText('安全摘要').length).toBeGreaterThan(0);
    await screen.findByLabelText('ops-context-strip');
    expect(textOf('.global-summary-strip')).toContain('当前页：公开面');
    expect(textOf('.context-header .panel-kicker')).toContain('公开面');
    expect(textOf('.context-header h2')).toContain('执行穿透 / 调度与门禁');
    expect(textOf('.context-header')).not.toContain('穿透焦点');
    expect(textOf('.context-header')).not.toContain('层级回退');
    expect(document.querySelector('.context-header-controls')).toBeNull();
    expect(textOf('.ops-context-strip')).toContain('运行上下文');
    expect(textOf('.ops-context-strip')).toContain('视图断言');
    expect(screen.getAllByText('当前域').length).toBeGreaterThan(0);
    expect(screen.queryByText('数据面切换')).toBeNull();
    expect(textOf('.global-topbar-inner')).not.toContain('请求视图');
    expect(textOf('.global-topbar-inner')).not.toContain('实际视图');
    expect(textOf('.global-summary-strip')).toContain('变更级别：仅研究');
    expect(textOf('.context-header-route-badges')).toContain('当前视图：公开终端');
    expect(document.querySelector('.top-rail')).toBeNull();
    expect(screen.getByText('运行摘要 / 调度总线')).toBeTruthy();
    expect(screen.getAllByText('总线调度与全局门禁').length).toBeGreaterThan(0);
    expect(screen.getAllByText('信号发生器与风险节流阀').length).toBeGreaterThan(0);
    expect(screen.getAllByText('策略实验室与回测复盘').length).toBeGreaterThan(0);
    expect(screen.getAllByText('查看终端层').length).toBeGreaterThan(0);

    await navigateHash('#/workspace/contracts');

    await waitFor(() => {
      expect(screen.getByLabelText('workspace-context-strip')).toBeTruthy();
    });
    expect(textOf('.global-summary-strip')).toContain('当前页：契约层');
    expect(textOf('.context-header .panel-kicker')).toContain('契约层');
    expect(textOf('.context-header h2')).toContain('公开入口 / 数据契约 / 路由验收');
    expect(textOf('.context-header')).not.toContain('穿透焦点');
    expect(textOf('.context-header')).not.toContain('层级回退');
    expect(document.querySelector('.context-header-controls')).toBeNull();
    expect(textOf('.context-header-route-badges')).toContain('当前阶段：');
    expect(screen.getAllByText('研究工作区').length).toBeGreaterThan(0);
    expect(screen.getAllByText('工件 / 回测 / 契约 / 原始').length).toBeGreaterThan(0);
    expect(screen.getByText('接口分层 / 可见边界')).toBeTruthy();
    expect(screen.getAllByText('源头主线').length).toBeGreaterThan(0);
    expect(screen.getAllByText('接口目录').length).toBeGreaterThan(0);
    expect(screen.queryByText('工作分区')).toBeNull();
    expect(screen.getAllByText('契约层').length).toBeGreaterThan(0);
    expect(screen.getAllByText('公开入口拓扑').length).toBeGreaterThan(0);
    expect(screen.getAllByText('查看原始层').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Fenlie 主入口').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Pages 回退入口').length).toBeGreaterThan(0);
    expect(screen.getAllByText('旧 OpenClaw').length).toBeGreaterThan(0);
    expect(screen.getByText('gateway/API')).toBeTruthy();
    expect(screen.getByText('公开终端 数据面')).toBeTruthy();
    expect(screen.getByText('内部终端 数据面')).toBeTruthy();
    expect(screen.getByText('Cloudflare 根域公开入口')).toBeTruthy();
    expect(screen.getByText('本地预览入口')).toBeTruthy();
    expect(screen.getByLabelText('workspace-context-strip')).toBeTruthy();
    expect(textOf('.workspace-context-strip')).toContain('契约 / 验收记录');
    expect(textOf('.workspace-context-strip')).toContain('入口拓扑与验收记录');
    expect(textOf('.workspace-context-strip')).toContain('焦点工件');
    expect(screen.queryByLabelText('operator-alerts')).toBeNull();
    expect(textOf('.global-topbar')).not.toContain('可视回测与调度台');
  });

  it('creates a dedicated surface frame for public and internal terminal layers', async () => {
    await renderApp();

    await screen.findByLabelText('ops-context-strip');
    expect(document.querySelector('.surface-shell.surface-public')).toBeTruthy();
    expect(textOf('.surface-watermark-public')).toContain('公开只读 / 安全定位');

    await primeSurface('internal', {
      snapshot: mockSnapshot as DashboardSnapshot,
      effectiveSurface: 'internal',
      snapshotPath: INTERNAL_SNAPSHOT_PATH,
    });
    await navigateHash('#/terminal/internal');

    await waitFor(() => {
      expect(document.querySelector('.surface-shell.surface-internal')).toBeTruthy();
    });
    expect(textOf('.surface-watermark-internal')).toContain('本地私有 / 完整路径');
  });

  it('renders public acceptance drilldown on contracts and keeps raw subcommand output hidden on public surface', async () => {
    const acceptanceSnapshot = buildAcceptanceSnapshot();
    installPassiveStore(acceptanceSnapshot as DashboardSnapshot);

    window.location.hash = '#/workspace/contracts';
    await renderApp();

    await screen.findByText('公开面验收');
    expect(screen.getByText('穿透层 1 / 验收总览')).toBeTruthy();
    expect(screen.getByText('穿透层 2 / 子链状态')).toBeTruthy();
    expect(screen.getByText('穿透层 3 / 子命令证据')).toBeTruthy();
    expect(screen.getAllByText('入口拓扑烟测').length).toBeGreaterThan(0);
    expect(screen.getAllByText('工作区五页面烟测').length).toBeGreaterThan(0);
    expect(screen.getByText('root overview 截图')).toBeTruthy();
    expect(screen.getByText('pages overview 截图')).toBeTruthy();
    expect(screen.getAllByText('root contracts 截图').length).toBeGreaterThan(0);
    expect(screen.getAllByText('pages contracts 截图').length).toBeGreaterThan(0);
    expect(screen.getByText('TLS 回退')).toBeTruthy();
    expect(screen.getAllByText('是').length).toBeGreaterThan(0);
    const summaryCard = screen.getByText('穿透层 1 / 验收总览').closest('section');
    expect(summaryCard?.textContent || '').toContain('review/root_contracts_browser.png');
    expect(summaryCard?.textContent || '').toContain('review/pages_contracts_browser.png');
    expect(summaryCard?.textContent || '').toContain('4');
    expect(summaryCard?.textContent || '').toContain('0');
    expect(summaryCard?.textContent || '').toContain('orderflow 过滤路由');
    expect(summaryCard?.textContent || '').toContain('#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow');
    expect(summaryCard?.textContent || '').toContain('orderflow 激活工件');
    expect(summaryCard?.textContent || '').toContain('intraday_orderflow_blueprint');
    expect(summaryCard?.textContent || '').toContain('orderflow 可见工件');
    expect(summaryCard?.textContent || '').toContain('intraday_orderflow_research_gate_blocker');
    expect(screen.queryByText('root_public timeout')).toBeNull();
    expect(screen.queryByText('timeout')).toBeNull();
  });

  it('renders workspace page toc and syncs contracts section deep link into nested accordion state', async () => {
    const acceptanceSnapshot = buildAcceptanceSnapshot();
    installPassiveStore(acceptanceSnapshot as DashboardSnapshot);

    window.location.hash = '#/workspace/contracts?page_section=contracts-subcommand-workspace_routes_smoke';
    await renderApp();

    const toc = await screen.findByRole('navigation', { name: 'page-sections-nav' });
    expect(within(toc).getByText('验收链')).toBeTruthy();
    expect(within(toc).getByText('子命令')).toBeTruthy();
    expect(within(toc).getByText('数据面')).toBeTruthy();
    expect(within(toc).getByText('源头主线')).toBeTruthy();
    await waitFor(() => {
      expect(document.querySelector('[data-accordion-id="contracts-subcommand-workspace_routes_smoke"]')?.getAttribute('data-state')).toBe('open');
    });

    await clickAndFlush(within(toc).getByRole('link', { name: '公开入口拓扑' }) as HTMLElement);
    await waitFor(() => {
      expect(window.location.hash).toContain('page_section=contracts-topology');
    });
  });

  it('keeps contracts source-head accordion triggers free of nested clamp toggles when labels overflow', async () => {
    const restoreOverflow = mockClampOverflow({ overflows: true });

    try {
      installPassiveStore(mockSnapshot as DashboardSnapshot);
      window.location.hash = '#/workspace/contracts?page_section=contracts-source-head-system_scheduler_state';
      await renderApp();

      await waitFor(() => {
        expect(document.querySelector('[data-accordion-id="contracts-source-head-system_scheduler_state"] .accordion-trigger')).toBeTruthy();
      });

      expect(
        document.querySelector('[data-accordion-id="contracts-source-head-system_scheduler_state"] .accordion-trigger .clamp-toggle'),
      ).toBeNull();
    } finally {
      restoreOverflow();
    }
  });

  it('shows page-level toc for backtests and updates section query on drill jump', async () => {
    window.location.hash = '#/workspace/backtests';
    await renderApp();

    const toc = await screen.findByRole('navigation', { name: 'page-sections-nav' });
    await clickAndFlush(within(toc).getByRole('link', { name: '近期比较行' }) as HTMLElement);

    await waitFor(() => {
      expect(window.location.hash).toContain('page_section=backtests-comparison');
    });
  });

  it('renders stage-specific workspace context copy for backtests and contracts', async () => {
    window.location.hash = '#/workspace/backtests';
    await renderApp();

    await screen.findByLabelText('workspace-context-strip');
    expect(screen.getByText('回测 / 验证窗口')).toBeTruthy();
    expect(screen.getByText('回测比较与验证窗口')).toBeTruthy();

    await navigateHash('#/workspace/contracts');

    await screen.findByRole('heading', { name: '公开入口拓扑' });
    await waitFor(() => {
      expect(textOf('.workspace-context-strip')).toContain('契约 / 验收记录');
      expect(textOf('.workspace-context-strip')).toContain('入口拓扑与验收记录');
    });
  });

  it('keeps ops and workspace context strips domain-pure', async () => {
    await renderApp();

    await screen.findByLabelText('ops-context-strip');
    expect(textOf('.ops-context-strip')).toContain('运行上下文');
    expect(textOf('.ops-context-strip')).not.toContain('焦点工件');
    expect(textOf('.ops-context-strip')).not.toContain('查看工件池');

    await navigateHash('#/workspace/contracts');

    await screen.findByRole('heading', { name: '公开入口拓扑' });
    await waitFor(() => {
      expect(textOf('.workspace-context-strip')).toContain('焦点工件');
    });
    expect(textOf('.workspace-context-strip')).not.toContain('查看运维面板');
    expect(textOf('.workspace-context-strip')).not.toContain('视图断言');
  });

  it('keeps first-panel summaries distinct from top context strips', async () => {
    await renderApp();

    await screen.findByLabelText('ops-context-strip');
    expect(screen.getByText('运行摘要 / 调度总线')).toBeTruthy();
    expect(screen.queryByText('面板 1 / 总线调度与全局门禁')).toBeNull();

    await navigateHash('#/workspace/backtests');

    await screen.findByRole('heading', { name: /^回测主池$/ });
    expect(screen.getByText('研究摘要 / 验证基线')).toBeTruthy();

    await navigateHash('#/workspace/raw');

    await screen.findByRole('heading', { name: /^原始快照$/ });
    expect(screen.getByText('研究摘要 / 原始回退')).toBeTruthy();
  });

  it('keeps contracts middle panels role-layered instead of repeating generic workspace copy', async () => {
    window.location.hash = '#/workspace/contracts';
    await renderApp();

    await screen.findByRole('heading', { name: '公开入口拓扑' });

    expect(screen.getByText('验收摘要 / 公开回归')).toBeTruthy();
    expect(screen.getByText('接口分层 / 可见边界')).toBeTruthy();
    expect(screen.getByText('源头摘要 / 主线对照')).toBeTruthy();
    expect(screen.getByText('回退摘要 / 失败退化')).toBeTruthy();

    expect(textOf('.workspace-grid')).toContain('入口拓扑烟测 + workspace 五页面 smoke 汇总');
    expect(textOf('.workspace-grid')).toContain('public / local / local-static / local-private 分层');
    expect(textOf('.workspace-grid')).toContain('只展示 source-owned 主头，不做 UI 二次裁决');
    expect(textOf('.workspace-grid')).toContain('当前 contracts 页的最终退化链，不参与研究判断');

    const acceptanceCard = screen.getByRole('heading', { name: '公开面验收' }).closest('section');
    const interfaceCard = screen.getByRole('heading', { name: '接口目录' }).closest('section');
    const sourceHeadsCard = screen.getByRole('heading', { name: '源头主线' }).closest('section');
    const fallbackCard = screen.getByRole('heading', { name: '回退链' }).closest('section');

    expect(textOf('.workspace-grid')).not.toContain('工作区 / 接口');
    expect(acceptanceCard?.textContent || '').not.toContain('工作区 / 契约 / 验收');
    expect(sourceHeadsCard?.textContent || '').not.toContain('工作区 / 契约');
    expect(fallbackCard?.textContent || '').not.toContain('工作区 / 回退链');
    expect(interfaceCard?.textContent || '').not.toContain('工作区 / 接口');
  });

  it('renders return watch inside contracts source-head drilldown', async () => {
    window.location.hash = '#/workspace/contracts';
    await renderApp();

    const sourceHeadToggle = await screen.findByRole('button', { name: /持有选择主头/i });
    await clickAndFlush(sourceHeadToggle);

    const sourceHeadsCard = screen.getByRole('heading', { name: '源头主线' }).closest('section');
    expect(sourceHeadsCard?.textContent || '').toContain('收益观察');
    expect(sourceHeadsCard?.textContent || '').toContain('hold24_zero');
  });

  it('keeps contracts topology and surface cards distinct from interface and fallback layers', async () => {
    window.location.hash = '#/workspace/contracts';
    await renderApp();

    const topologyCard = screen.getByRole('heading', { name: '公开入口拓扑' }).closest('section');
    const publicSurfaceCard = screen.getByRole('heading', { name: '公开终端 数据面' }).closest('section');
    const internalSurfaceCard = screen.getByRole('heading', { name: '内部终端 数据面' }).closest('section');

    expect(screen.getByText('入口摘要 / 发布拓扑')).toBeTruthy();
    expect(screen.getByText('公开摘要 / 对外发布')).toBeTruthy();
    expect(screen.getByText('内部摘要 / 本地剥离')).toBeTruthy();

    expect(topologyCard?.textContent || '').toContain('根域、Pages、旧 OpenClaw 与 gateway 的对外角色和退化关系');
    expect(topologyCard?.textContent || '').not.toContain('public / local / local-static / local-private 分层');

    expect(publicSurfaceCard?.textContent || '').toContain('只声明公开可见入口、脱敏层级与默认回退。');
    expect(internalSurfaceCard?.textContent || '').toContain('只声明本地可见断言、剥离策略与私有回退。');
    expect(publicSurfaceCard?.textContent || '').not.toContain('工作区 / 契约 / 公开');
    expect(internalSurfaceCard?.textContent || '').not.toContain('工作区 / 契约 / 内部');
  });

  it('keeps public and internal surface drilldowns role-specific instead of sharing generic contract labels', async () => {
    window.location.hash = '#/workspace/contracts';
    await renderApp();

    const publicSurfaceCard = screen.getByRole('heading', { name: '公开终端 数据面' }).closest('section');
    const internalSurfaceCard = screen.getByRole('heading', { name: '内部终端 数据面' }).closest('section');

    expect(publicSurfaceCard?.textContent || '').toContain('穿透层 1 / 公开断言');
    expect(publicSurfaceCard?.textContent || '').toContain('穿透层 2 / 发布入口与默认回退');
    expect(internalSurfaceCard?.textContent || '').toContain('穿透层 1 / 本地断言');
    expect(internalSurfaceCard?.textContent || '').toContain('穿透层 2 / 剥离策略与私有回退');

    expect(publicSurfaceCard?.textContent || '').not.toContain('穿透层 1 / 契约主断言');
    expect(publicSurfaceCard?.textContent || '').not.toContain('穿透层 2 / 发布与回退');
    expect(internalSurfaceCard?.textContent || '').not.toContain('穿透层 1 / 契约主断言');
    expect(internalSurfaceCard?.textContent || '').not.toContain('穿透层 2 / 发布与回退');
  });

  it('renders public topology as node cards with fallback chain semantics instead of generic interface rows', async () => {
    window.location.hash = '#/workspace/contracts';
    await renderApp();

    const topologyCard = screen.getByRole('heading', { name: '公开入口拓扑' }).closest('section');
    expect(topologyCard?.textContent || '').toContain('节点 1');
    expect(topologyCard?.textContent || '').toContain('节点 2');
    expect(topologyCard?.textContent || '').toContain('退化至 Pages 回退入口');
    expect(topologyCard?.textContent || '').toContain('退化至 Fenlie 主入口');
    expect(topologyCard?.textContent || '').toContain('控制源');
    expect(topologyCard?.querySelector('.topology-node-grid')).toBeTruthy();
    expect(topologyCard?.querySelector('.topology-node-card')).toBeTruthy();
    expect(topologyCard?.querySelector('.interface-grid')).toBeNull();
  });

  it('assigns topology nodes into primary / fallback / probe visual tiers', async () => {
    window.location.hash = '#/workspace/contracts';
    await renderApp();

    const topologyCard = screen.getByRole('heading', { name: '公开入口拓扑' }).closest('section');
    expect(topologyCard?.textContent || '').toContain('主入口层');
    expect(topologyCard?.textContent || '').toContain('回退入口层');
    expect(topologyCard?.textContent || '').toContain('探针层');

    expect(topologyCard?.querySelector('.topology-node-card-primary')).toBeTruthy();
    expect(topologyCard?.querySelector('.topology-node-card-fallback')).toBeTruthy();
    expect(topologyCard?.querySelector('.topology-node-card-probe')).toBeTruthy();
  });

  it('groups topology into primary chain, fallback chain, and probe zone sections', async () => {
    window.location.hash = '#/workspace/contracts';
    await renderApp();

    const topologyCard = screen.getByRole('heading', { name: '公开入口拓扑' }).closest('section');
    expect(topologyCard?.textContent || '').toContain('主入口链');
    expect(topologyCard?.textContent || '').toContain('回退链');
    expect(topologyCard?.textContent || '').toContain('探针区');

    expect(topologyCard?.querySelector('.topology-band-primary')).toBeTruthy();
    expect(topologyCard?.querySelector('.topology-band-fallback')).toBeTruthy();
    expect(topologyCard?.querySelector('.topology-band-probe')).toBeTruthy();
  });

  it('shows explicit flow rails for primary and fallback topology bands', async () => {
    window.location.hash = '#/workspace/contracts';
    await renderApp();

    const topologyCard = screen.getByRole('heading', { name: '公开入口拓扑' }).closest('section');
    const primaryBand = topologyCard?.querySelector('.topology-band-primary');
    const fallbackBand = topologyCard?.querySelector('.topology-band-fallback');

    expect(primaryBand?.querySelector('.topology-band-flow')).toBeTruthy();
    expect(fallbackBand?.querySelector('.topology-band-flow')).toBeTruthy();
    expect(primaryBand?.textContent || '').toContain('Fenlie 主入口');
    expect(primaryBand?.textContent || '').toContain('Pages 回退入口');
    expect(fallbackBand?.querySelector('.topology-flow-arrow')).toBeTruthy();
    expect(fallbackBand?.textContent || '').toContain('旧 OpenClaw');
  });

  it('keeps stage kicker singular on backtests and raw long pages', async () => {
    window.location.hash = '#/workspace/backtests';
    await renderApp();

    await screen.findByRole('heading', { name: /^回测主池$/ });
    expect(screen.getAllByText('研究摘要 / 验证基线')).toHaveLength(1);

    await navigateHash('#/workspace/raw');

    await screen.findByRole('heading', { name: /^原始快照$/ });
    expect(screen.getAllByText('研究摘要 / 原始回退')).toHaveLength(1);
  });

  it('keeps workspace context-nav default page_section links and a unique backtests headline', async () => {
    window.location.hash = '#/workspace/artifacts';
    await renderApp();

    const contextNav = await screen.findByRole('navigation', { name: 'context-nav' });
    expect(within(contextNav).getByRole('link', { name: /回测池\s*回测主池/ }).getAttribute('href')).toContain('page_section=backtests-overview');
    expect(within(contextNav).getByRole('link', { name: /契约层\s*公开入口拓扑/ }).getAttribute('href')).toContain('page_section=contracts-topology');
    expect(within(contextNav).getByRole('link', { name: /原始层\s*原始快照/ }).getAttribute('href')).toContain('page_section=raw-summary');

    await clickAndFlush(within(contextNav).getByRole('link', { name: /回测池\s*回测主池/ }) as HTMLElement);

    await screen.findByRole('heading', { name: /^回测主池$/ });
    await waitFor(() => {
      expect(screen.getAllByRole('heading', { name: /^回测主池$/ })).toHaveLength(1);
    });

    await clickAndFlush(within(contextNav).getByRole('link', { name: /原始层\s*原始快照/ }) as HTMLElement);

    await screen.findByRole('heading', { name: /^原始快照$/ });
    await waitFor(() => {
      expect(screen.getAllByRole('heading', { name: /^原始快照$/ })).toHaveLength(1);
    });
  });

  it('supports panel / section / row 三层穿透 URL 焦点与回退路径', async () => {
    window.location.hash = '#/terminal/internal?panel=signal-risk&section=focus-slots&row=primary';
    primeStore('internal', {
      snapshot: mockSnapshot as DashboardSnapshot,
      effectiveSurface: 'internal',
      snapshotPath: INTERNAL_SNAPSHOT_PATH,
    });
    await renderApp();

    await screen.findByText('当前穿透路径');
    expect(screen.getByText('返回上一级')).toBeTruthy();
    expect(screen.getAllByText('当前焦点').length).toBeGreaterThan(0);
    expect(screen.getAllByText('主槽位').length).toBeGreaterThan(0);
    expect(screen.getAllByText('清除行焦点').length).toBeGreaterThan(0);
  });

  it('shows workspace artifact handoff links back into terminal focus', async () => {
    window.location.hash = '#/workspace/artifacts?artifact=price_action_breakout_pullback';
    await renderApp();

    await screen.findByRole('heading', { name: '当前焦点' });
    expect(screen.getAllByText('查看终端层 / 策略实验室与回测复盘 / 穿透层 1 / 研究头部').length).toBeGreaterThan(0);
  });

  it('adopts source-owned terminal focus into workspace url state', async () => {
    window.location.hash = '#/workspace/artifacts?artifact=price_action_breakout_pullback';
    await renderApp();

    await screen.findByLabelText('workspace-context-strip');
    await waitFor(() => {
      expect(window.location.hash).toContain('artifact=price_action_breakout_pullback');
      expect(window.location.hash).toContain('panel=lab-review');
      expect(window.location.hash).toContain('section=research-heads');
      expect(window.location.hash).not.toContain('search=');
    });
  });

  it('prefers exact artifact focus over timestamped fuzzy matches', async () => {
    const duplicateSnapshot = {
      ...mockSnapshot,
      catalog: [
        {
          id: '20260319T103100Z_price_action_breakout_pullback_sim_only.json',
          payload_key: 'price_action_breakout_pullback',
          category: 'sim-only',
          label: '20260319T103100Z_price_action_breakout_pullback_sim_only.json',
          status: 'ok',
          path: 'review/20260319T103100Z_price_action_breakout_pullback_sim_only.json',
        },
        ...mockSnapshot.catalog,
      ],
    };
    installPassiveStore(duplicateSnapshot as DashboardSnapshot);

    window.location.hash = '#/workspace/artifacts?artifact=price_action_breakout_pullback';
    await renderApp();

    await screen.findByLabelText('workspace-context-strip');
    await waitFor(() => {
      expect(window.location.hash).toContain('artifact=price_action_breakout_pullback');
      expect(window.location.hash).not.toContain('20260319T103100Z_price_action_breakout_pullback_sim_only.json');
    });
  });

  it('separates canonical mainlines from archive artifacts in workspace explorer', async () => {
    const layeredSnapshot = {
      ...mockSnapshot,
      catalog: [
        ...mockSnapshot.catalog,
        {
          id: 'hold_selection_handoff',
          payload_key: 'hold_selection_handoff',
          artifact_layer: 'canonical',
          artifact_group: 'research_hold_transfer',
          category: 'research',
          label: 'hold_selection_handoff',
          status: 'ok',
          path: 'review/latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json',
        },
        {
          id: 'cross_section_backtest',
          payload_key: 'cross_section_backtest',
          artifact_layer: 'canonical',
          artifact_group: 'research_cross_section',
          category: 'backtest',
          label: 'cross_section_backtest',
          status: 'warning',
          path: 'review/20260315T105000Z_crypto_shortline_cross_section_backtest.json',
        },
        {
          id: '20260318T163800Z_price_action_breakout_pullback_sim_only',
          artifact_layer: 'archive',
          category: 'sim-only',
          label: '20260318T163800Z_price_action_breakout_pullback_sim_only',
          status: 'ok',
          path: 'review/20260318T163800Z_price_action_breakout_pullback_sim_only.json',
        },
      ],
    };
    installPassiveStore(layeredSnapshot as DashboardSnapshot);

    window.location.hash = '#/workspace/artifacts?artifact=price_action_breakout_pullback';
    await renderApp();

    await screen.findByRole('heading', { name: '研究地图' });
    await waitFor(() => {
      const groupLabels = Array.from(document.querySelectorAll('.artifact-group-button strong'))
        .map((node) => node.textContent?.replace(/\s+/g, ' ').trim() || '');
      expect(groupLabels).toEqual([
        '核心研究主线',
        '系统锚点',
        '退出与风控',
        '持有迁移',
        '横截面与替代族',
        '历史归档层',
      ]);
    });
    await waitFor(() => {
      expect(document.querySelector('.artifact-group-button.active')?.textContent || '').toMatch(/核心研究主线|研究主线/);
      expect(document.querySelector('.artifact-button.active')?.textContent || '').toMatch(/price_action_breakout_pullback|ETH 15m 主线/);
    });
  });

  it('keeps artifact selector buttons free of nested clamp toggles when titles overflow', async () => {
    const restoreOverflow = mockClampOverflow({ overflows: true });

    try {
      window.location.hash = '#/workspace/artifacts?artifact=price_action_breakout_pullback';
      await renderApp();

      await waitFor(() => {
        expect(document.querySelector('.artifact-button')).toBeTruthy();
      });

      expect(document.querySelector('.artifact-button .clamp-toggle')).toBeNull();
    } finally {
      restoreOverflow();
    }
  });

  it('defaults workspace selection to research mainline instead of system anchors', async () => {
    window.location.hash = '#/workspace/artifacts';
    await renderApp();

    await screen.findByLabelText('workspace-context-strip');
    expect(screen.getByRole('heading', { name: '研究地图' })).toBeTruthy();
    expect(screen.getByRole('heading', { name: '状态雷达' })).toBeTruthy();
    expect(screen.getByRole('heading', { name: '检索定位' })).toBeTruthy();
    expect(screen.getByRole('heading', { name: '当前焦点' })).toBeTruthy();
    expect(screen.queryByText('穿透控件')).toBeNull();
    await waitFor(() => {
      expect(document.querySelector('.artifact-group-button.active')?.textContent || '').toMatch(/核心研究主线|研究主线/);
      expect(document.querySelector('.artifact-button.active')?.textContent || '').toMatch(/price_action_breakout_pullback|ETH 15m 主线/);
      expect(window.location.hash).toContain('artifact=price_action_breakout_pullback');
      expect(window.location.hash).toContain('group=research_mainline');
      expect(window.location.hash).toContain('search_scope=title');
    });
  });

  it('hydrates workspace context-nav links from source-owned default focus before manual navigation', async () => {
    window.location.hash = '#/workspace/artifacts';
    await renderApp();

    const contextNav = await screen.findByRole('navigation', { name: 'context-nav' });
    const backtestsHref = within(contextNav).getByRole('link', { name: /回测池\s*回测主池/ }).getAttribute('href') || '';
    const contractsHref = within(contextNav).getByRole('link', { name: /契约层\s*公开入口拓扑/ }).getAttribute('href') || '';
    const rawHref = within(contextNav).getByRole('link', { name: /原始层\s*原始快照/ }).getAttribute('href') || '';

    expect(backtestsHref).toContain('artifact=price_action_breakout_pullback');
    expect(backtestsHref).toContain('group=research_mainline');
    expect(backtestsHref).toContain('panel=lab-review');
    expect(backtestsHref).toContain('section=research-heads');
    expect(backtestsHref).toContain('search_scope=title');
    expect(contractsHref).toContain('artifact=price_action_breakout_pullback');
    expect(rawHref).toContain('artifact=price_action_breakout_pullback');
  });

  it('prefers snapshot workspace_default_focus over frontend heuristic when no artifact is present in url', async () => {
    installPassiveStore({
      ...mockSnapshot,
      workspace_default_focus: {
        artifact: 'price_action_exit_risk',
        group: 'research_mainline',
        panel: 'lab-review',
        section: 'research-heads',
        search_scope: 'title',
      },
    } as DashboardSnapshot);

    window.location.hash = '#/workspace/artifacts';
    await renderApp();

    const contextNav = await screen.findByRole('navigation', { name: 'context-nav' });
    const backtestsHref = within(contextNav).getByRole('link', { name: /回测池\s*回测主池/ }).getAttribute('href') || '';

    await waitFor(() => {
      expect(document.querySelector('.artifact-button.active')?.textContent || '').toMatch(/price_action_exit_risk|退出\/风控/);
      expect(window.location.hash).toContain('artifact=price_action_exit_risk');
      expect(backtestsHref).toContain('artifact=price_action_exit_risk');
      expect(backtestsHref).toContain('group=research_mainline');
      expect(backtestsHref).toContain('panel=lab-review');
      expect(backtestsHref).toContain('section=research-heads');
      expect(backtestsHref).toContain('search_scope=title');
    });
  });

  it('locks the generated public snapshot to mainline-first url focus and four-card left rail state', async () => {
    installPassiveStore(workspaceArtifactsSmokeSnapshot as DashboardSnapshot);

    window.location.hash = '#/workspace/artifacts';
    await renderApp();

    await screen.findByLabelText('workspace-context-strip');
    expect(screen.getByRole('heading', { name: '研究地图' })).toBeTruthy();
    expect(screen.getByRole('heading', { name: '状态雷达' })).toBeTruthy();
    expect(screen.getByRole('heading', { name: '检索定位' })).toBeTruthy();
    expect(screen.getByRole('heading', { name: '当前焦点' })).toBeTruthy();

    await waitFor(() => {
      expect(window.location.hash).toContain('artifact=price_action_breakout_pullback');
      expect(window.location.hash).toContain('group=research_mainline');
      expect(window.location.hash).toContain('search_scope=title');
      expect(window.location.hash).toContain('panel=lab-review');
      expect(window.location.hash).toContain('section=research-heads');
      expect(window.location.hash).not.toContain('artifact=20260319T103100Z_price_action_breakout_pullback_sim_only');
      expect(document.querySelector('.artifact-group-button.active')?.textContent || '').toMatch(/核心研究主线|研究主线/);
      expect(document.querySelector('.artifact-button.active')?.textContent || '').toMatch(/price_action_breakout_pullback|ETH 15m 主线/);
    });
  });

  it('shows terminal focus handoff links back into workspace', async () => {
    window.location.hash = '#/terminal/internal?panel=signal-risk&section=control-chain';
    primeStore('internal', {
      snapshot: mockSnapshot as DashboardSnapshot,
      effectiveSurface: 'internal',
      snapshotPath: INTERNAL_SNAPSHOT_PATH,
    });
    await renderApp();

    await screen.findByText('关联工作台');
    expect(screen.getAllByText('查看工件 / ETH 15m 主线').length).toBeGreaterThan(0);
  });

  it('keeps an explicit terminal artifact pinned across breadcrumbs and workspace handoffs instead of falling back to section defaults', async () => {
    const pinnedArtifactPath = 'review/20260321T115900Z_price_action_breakout_pullback_sim_only.json';
    const conflictingSectionArtifact = '/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260316T193733Z_brooks_structure_review_queue.json';
    const pinnedTerminalSnapshot = {
      ...mockSnapshot,
      artifact_payloads: {
        ...mockSnapshot.artifact_payloads,
        operator_panel: {
          ...mockSnapshot.artifact_payloads.operator_panel,
          payload: {
            ...mockSnapshot.artifact_payloads.operator_panel.payload,
            focus_slots: [
              {
                slot: 'primary',
                symbol: 'SC2603',
                action: 'review_manual_stop_entry',
                state: 'review',
                done_when: 'manual review done',
                source_artifact: conflictingSectionArtifact,
              },
            ],
          },
        },
        price_action_breakout_pullback: {
          ...mockSnapshot.artifact_payloads.price_action_breakout_pullback,
          path: pinnedArtifactPath,
        },
      },
      catalog: mockSnapshot.catalog.map((row) => (
        row.id === 'price_action_breakout_pullback'
          ? { ...row, path: pinnedArtifactPath }
          : row
      )),
    };

    window.location.hash = `#/terminal/public?artifact=${encodeURIComponent(pinnedArtifactPath)}&panel=signal-risk&section=focus-slots&row=primary`;
    installPassiveStore(pinnedTerminalSnapshot as DashboardSnapshot);
    await renderApp();

    await screen.findByText('关联工作台');

    const breadcrumbs = document.querySelector('[aria-label="terminal-focus-breadcrumbs"]');
    expect(breadcrumbs?.querySelector(`a[href*="${encodeURIComponent(pinnedArtifactPath)}"]`)).toBeTruthy();
    expect(breadcrumbs?.querySelector(`a[href*="${encodeURIComponent(conflictingSectionArtifact)}"]`)).toBeNull();

    const handoffLinks = Array.from(document.querySelectorAll('.workspace-handoff-links a')).map((node) => node.getAttribute('href') || '');
    expect(handoffLinks.some((href) => href.includes(`artifact=${encodeURIComponent(pinnedArtifactPath)}`) || href.includes('artifact=price_action_breakout_pullback'))).toBe(true);
    expect(handoffLinks.some((href) => href.includes(encodeURIComponent(conflictingSectionArtifact)))).toBe(false);

    const inspectorLinks = Array.from(document.querySelectorAll('.inspector-link-stack a')).map((node) => node.getAttribute('href') || '');
    expect(inspectorLinks.some((href) => href.includes(encodeURIComponent(conflictingSectionArtifact)))).toBe(false);
  });

  it('refreshes workspace contracts after route change so stale snapshot does not persist', async () => {
    const staleSnapshot = {
      ...mockSnapshot,
      ui_routes: {
        ...mockSnapshot.ui_routes,
        frontend_pages: 'https://fenlie-dashboard-web.pages.dev',
      },
      public_topology: mockSnapshot.public_topology.map((row) => (
        row.id === 'fenlie_pages'
          ? { ...row, url: 'https://fenlie-dashboard-web.pages.dev' }
          : row
      )),
    };
    const freshSnapshot = {
      ...mockSnapshot,
      ui_routes: {
        ...mockSnapshot.ui_routes,
        frontend_pages: 'https://fenlie.fuuu.fun',
      },
      public_topology: mockSnapshot.public_topology.map((row) => (
        row.id === 'fenlie_pages'
          ? { ...row, url: 'https://fenlie.fuuu.fun' }
          : row
      )),
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => staleSnapshot,
      })
      .mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => freshSnapshot,
      });
    vi.stubGlobal('fetch', fetchMock);
    installLiveRefresh();

    window.location.hash = '#/terminal/public';
    await renderApp();

    await screen.findByLabelText('ops-context-strip');
    expect(textOf('.ops-context-strip')).toContain('运行上下文');

    await navigateHash('#/workspace/contracts');

    await screen.findByText('接口分层 / 可见边界');
    await waitFor(() => {
      expect(screen.getAllByText('https://fenlie.fuuu.fun').length).toBeGreaterThan(0);
    });
  });

  it('fetches fresh public snapshot when mounting different workspace sections', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => mockSnapshot,
    }));
    vi.stubGlobal('fetch', fetchMock);
    installLiveRefresh();

    window.location.hash = '#/workspace/artifacts';
    let view: ReturnType<typeof render> = await renderApp();

    await screen.findByRole('heading', { name: '研究地图' });
    const initialCalls = fetchMock.mock.calls.length;
    view.unmount();

    window.location.hash = '#/workspace/backtests?artifact=price_action_breakout_pullback&panel=lab-review&section=research-heads';
    view = await renderApp();
    await screen.findByText('穿透层 1 / 回测池');
    const afterBacktestsCalls = fetchMock.mock.calls.length;
    expect(afterBacktestsCalls).toBeGreaterThan(initialCalls);
    view.unmount();

    window.location.hash = '#/workspace/contracts?artifact=price_action_breakout_pullback&panel=lab-review&section=research-heads';
    view = await renderApp();
    await screen.findByText('接口分层 / 可见边界');
    expect(fetchMock.mock.calls.length).toBeGreaterThan(afterBacktestsCalls);
  });

  it('renders internal alignment workspace page with toc and feedback evidence', async () => {
    installPassiveStoreForSurface('internal', {
      snapshot: buildFeedbackSnapshot(),
      effectiveSurface: 'internal',
      snapshotPath: INTERNAL_SNAPSHOT_PATH,
    });
    window.location.hash = '#/workspace/alignment?view=internal';

    await renderApp();

    expect(await screen.findByRole('heading', { name: '方向对齐投射' })).toBeTruthy();
    expect(screen.getByRole('navigation', { name: 'page-sections-nav' })).toBeTruthy();
    expect(document.body.textContent || '').toContain('拆分研究工作区左栏四控件');
    expect(document.body.textContent || '').toContain('完成左栏职责拆分');
  });

  it('shows locked state on public alignment route without exposing internal feedback details', async () => {
    window.location.hash = '#/workspace/alignment';
    await renderApp();

    expect(await screen.findByRole('heading', { name: '方向对齐投射' })).toBeTruthy();
    expect(document.body.textContent || '').toContain('仅内部可见');
    expect(document.body.textContent || '').not.toContain('拆分研究工作区左栏四控件');
  });

  it('polls public snapshot on workspace routes to keep freshness semantics aligned with terminal', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => mockSnapshot,
    }));
    const intervalHandlers = new Map<number, TimerHandler>();
    const setIntervalSpy = vi.spyOn(window, 'setInterval').mockImplementation((handler, timeout) => {
      if (typeof timeout === 'number') intervalHandlers.set(timeout, handler);
      return 1 as unknown as number;
    });
    const clearIntervalSpy = vi.spyOn(window, 'clearInterval').mockImplementation(() => undefined);
    vi.stubGlobal('fetch', fetchMock);
    installLiveRefresh();

    try {
      window.location.hash = '#/workspace/contracts';
      await renderApp();

      await screen.findByText('接口分层 / 可见边界');
      const initialCalls = fetchMock.mock.calls.length;

      expect(setIntervalSpy).toHaveBeenCalled();
      expect(setIntervalSpy).toHaveBeenCalledWith(expect.any(Function), 30_000);
      const pollHandler = intervalHandlers.get(30_000);
      expect(pollHandler).toBeTypeOf('function');

      await act(async () => {
        await Promise.resolve(typeof pollHandler === 'function' ? pollHandler() : undefined);
      });

      expect(fetchMock.mock.calls.length).toBeGreaterThan(initialCalls);
    } finally {
      setIntervalSpy.mockRestore();
      clearIntervalSpy.mockRestore();
    }
  });

  it('surfaces a top-level alert when internal view falls back to public snapshot', async () => {
    window.location.hash = '#/terminal/internal';
    vi.stubGlobal('fetch', vi.fn(async (input: string | URL) => {
      const url = String(input);
      if (url.includes('fenlie_dashboard_internal_snapshot.json')) {
        return {
          ok: false,
          status: 404,
          json: async () => ({}),
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => mockSnapshot,
      };
    }));
    installLiveRefresh();

    await renderApp();

    await screen.findByText('数据面已发生回退');
    expect(screen.getAllByText('内部终端 → 公开终端').length).toBeGreaterThan(0);
  });

  it('keeps overview internal view in explicit unavailable state instead of falling back to public snapshot', async () => {
    window.location.hash = '#/overview?view=internal';
    const fetchMock = vi.fn(async (input: string | URL) => {
      const url = String(input);
      if (url.includes('fenlie_dashboard_internal_snapshot.json')) {
        return {
          ok: false,
          status: 404,
          json: async () => ({}),
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => mockSnapshot,
      };
    });
    vi.stubGlobal('fetch', fetchMock);
    installLiveRefresh();

    await renderApp();

    await screen.findByText('内部总览暂不可用');
    expect(document.body.textContent || '').toContain('加载失败：/data/fenlie_dashboard_internal_snapshot.json -> 404');
    expect(document.body.textContent || '').not.toContain('数据面已发生回退');
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes('fenlie_dashboard_snapshot.json'))).toBe(false);
  });

  it('keeps workspace internal view in explicit unavailable state instead of falling back to public snapshot', async () => {
    window.location.hash = '#/workspace/alignment?view=internal';
    const fetchMock = vi.fn(async (input: string | URL) => {
      const url = String(input);
      if (url.includes('fenlie_dashboard_internal_snapshot.json')) {
        return {
          ok: false,
          status: 404,
          json: async () => ({}),
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => mockSnapshot,
      };
    });
    vi.stubGlobal('fetch', fetchMock);
    installLiveRefresh();

    await renderApp();

    await screen.findByText('内部方向对齐投射暂不可用');
    expect(document.body.textContent || '').toContain('加载失败：/data/fenlie_dashboard_internal_snapshot.json -> 404');
    expect(document.body.textContent || '').not.toContain('数据面已发生回退');
    expect(document.body.textContent || '').not.toContain('拆分研究工作区左栏四控件');
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes('fenlie_dashboard_snapshot.json'))).toBe(false);
  });

  it('surfaces guardian and control-chain alerts on the terminal surface', async () => {
    const alertSnapshot = {
      ...mockSnapshot,
      artifact_payloads: {
        ...mockSnapshot.artifact_payloads,
        operator_panel: {
          ...mockSnapshot.artifact_payloads.operator_panel,
          payload: {
            ...mockSnapshot.artifact_payloads.operator_panel.payload,
            summary: {
              ...mockSnapshot.artifact_payloads.operator_panel.payload.summary,
              openclaw_guardian_clearance_top_blocker_title: 'ticket_actionability',
              openclaw_guardian_clearance_top_blocker_next_action: 'refresh_signal_source',
              openclaw_guardian_clearance_top_blocker_target_artifact: 'crypto_shortline_pattern_router',
            },
            control_chain: [
              {
                label: '风控',
                source_status: 'blocked',
                source_decision: 'hold',
                next_target_artifact: 'crypto_shortline_pattern_router',
                blocking_reason: 'guardian_ticket_actionability',
              },
            ],
          },
        },
      },
    };
    installPassiveStore(alertSnapshot as DashboardSnapshot);

    await renderApp();

    await screen.findByText('传导链存在阻塞');
    expect(screen.getAllByText('处置路径').length).toBeGreaterThan(0);
    expect(screen.getAllByText('来源断言').length).toBeGreaterThan(0);
    expect(screen.getAllByText('目标工件').length).toBeGreaterThan(0);
    const alertLinks = Array.from(document.querySelectorAll('[aria-label="operator-alerts"] a')).map((node) => node.getAttribute('href'));
    expect(alertLinks).toContain('#/workspace/artifacts?artifact=crypto_shortline_pattern_router');
    expect(alertLinks).toContain('#/workspace/raw?artifact=%2Ftmp%2Foperator-panel.json');
    const alertText = document.querySelector('[aria-label="operator-alerts"]')?.textContent || '';
    expect(alertText).toContain('传导链存在阻塞');
    expect(alertText).toMatch(/ticket_actionability|票据可执行性/);
  });

  it('focuses terminal panel and drill section from url params', async () => {
    window.location.hash = '#/terminal/public?panel=signal-risk&section=control-chain';
    await renderApp();

    await screen.findByLabelText('ops-context-strip');
    const focusedPanel = document.querySelector('.panel-card-focused');
    const focusedDrill = document.querySelector('.drill-section-focused');
    expect(focusedPanel?.textContent || '').toContain('信号发生器与风险节流阀');
    expect(focusedDrill?.textContent || '').toContain('穿透层 4 / 传导控制链');
  });

  it('preselects workspace artifact and raw drilldown from url focus params', async () => {
    window.location.hash = '#/workspace/artifacts?artifact=price_action_exit_risk';
    await renderApp();

    await screen.findByLabelText('workspace-context-strip');
    const activeArtifact = document.querySelector('.artifact-button.active');
    expect(activeArtifact?.textContent || '').toMatch(/price_action_exit_risk|退出\/风控/);

    await navigateHash('#/workspace/raw?artifact=price_action_exit_risk');

    await waitFor(() => {
      expect(screen.getByText('告警定向原始层')).toBeTruthy();
    });
    expect(document.body.textContent || '').toMatch(/price_action_exit_risk|continue forward window/);
  });

  it('syncs workspace explorer state back into the url', async () => {
    window.location.hash = '#/workspace/artifacts';
    await renderApp();

    await screen.findByLabelText('workspace-context-strip');
    const searchInput = document.querySelector('.search-input');
    expect(searchInput).toBeTruthy();
    await changeAndFlush(searchInput as HTMLElement, 'exit');

    await waitFor(() => {
      expect(window.location.hash).toContain('search=exit');
      expect(window.location.hash).toContain('search_scope=title');
      expect(window.location.hash).toContain('group=research_mainline');
    });
  });

  it('corrects conflicting group query to the artifact-owned research group', async () => {
    const groupedSnapshot = {
      ...mockSnapshot,
      catalog: mockSnapshot.catalog.map((row) => (
        row.id === 'price_action_exit_hold_robustness'
          ? { ...row, path: '/tmp/hold-robustness.json' }
          : row
      )),
    };
    installPassiveStore(groupedSnapshot as DashboardSnapshot);

    window.location.hash = '#/workspace/artifacts?artifact=price_action_exit_hold_robustness&group=system_anchor';
    await renderApp();

    await screen.findByLabelText('workspace-context-strip');
    await waitFor(() => {
      expect(window.location.hash).toContain('artifact=price_action_exit_hold_robustness');
      expect(window.location.hash).toContain('group=research_exit_risk');
      expect(document.querySelector('.artifact-group-button.active')?.textContent || '').toMatch(/退出与风控/);
    });
  });

  it('limits artifact search by the selected search scope', async () => {
    const scopedSnapshot = {
      ...workspaceArtifactsSmokeSnapshot,
      catalog: workspaceArtifactsSmokeSnapshot.catalog.map((row) => (
        row.id === 'hold_selection_handoff'
          ? {
              ...row,
              label: 'hold_selection_handoff',
              path: 'review/latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json',
            }
          : row
      )),
    };
    installPassiveStore(scopedSnapshot as DashboardSnapshot);

    window.location.hash = '#/workspace/artifacts?group=research_hold_transfer';
    await renderApp();

    await screen.findByRole('heading', { name: '检索定位' });
    const searchInput = document.querySelector('.search-input');
    expect(searchInput).toBeTruthy();

    await changeAndFlush(searchInput as HTMLElement, 'latest_price_action');
    await waitFor(() => {
      expect(screen.getAllByText('暂无匹配工件').length).toBeGreaterThan(0);
    });

    await clickAndFlush(screen.getByRole('button', { name: '路径' }) as HTMLElement);
    await waitFor(() => {
      expect(window.location.hash).toContain('search_scope=path');
      expect(document.querySelector('.artifact-button.active')?.textContent || '').toMatch(/hold_selection_handoff/);
    });
  });

  it('deep-links orderflow public artifacts inside research cross-section workspace filter', async () => {
    installPassiveStore(workspaceArtifactsSmokeSnapshot as DashboardSnapshot);
    window.location.hash = '#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow';
    await renderApp();

    await screen.findByRole('heading', { name: '研究地图' });
    await waitFor(() => {
      expect(window.location.hash).toContain('group=research_cross_section');
      expect(window.location.hash).toContain('search_scope=title');
      expect(window.location.hash).toContain('search=orderflow');
      expect(window.location.hash).toContain('artifact=intraday_orderflow_blueprint');
    });
    expect(document.querySelector('.artifact-group-button.active')?.textContent || '').toMatch(/横截面与替代族/);
    expect(document.querySelector('.artifact-button.active')?.textContent || '').toMatch(/intraday_orderflow_blueprint/);
    expect(screen.getAllByText('intraday_orderflow_blueprint').length).toBeGreaterThan(0);
    expect(screen.getAllByText('intraday_orderflow_research_gate_blocker').length).toBeGreaterThan(0);
    const focusCard = screen.getByRole('heading', { name: '当前焦点' }).closest('section');
    expect(focusCard?.textContent || '').toContain('查看终端层');
    const terminalLink = Array.from(within(focusCard as HTMLElement).getAllByRole('link'))
      .find((link) => (link.getAttribute('href') || '').includes('#/terminal/public?artifact=intraday_orderflow_blueprint'));
    expect(terminalLink).toBeTruthy();
    if (!terminalLink) {
      throw new Error('expected orderflow focus card to expose terminal handoff link');
    }
    expect(terminalLink.getAttribute('href') || '').toContain('#/terminal/public?artifact=intraday_orderflow_blueprint');
    expect(terminalLink.getAttribute('href') || '').toContain('panel=lab-review');
    expect(terminalLink.getAttribute('href') || '').toContain('section=research-heads');
  });

  it('syncs terminal focus navigation into the url', async () => {
    window.location.hash = '#/terminal/public';
    await renderApp();

    await screen.findByLabelText('ops-context-strip');
    const panelTarget = document.querySelector('[aria-label="terminal-focus-nav"] a[href*="panel=lab-review"]');
    expect(panelTarget).toBeTruthy();
    await clickAndFlush(panelTarget as HTMLElement);

    await waitFor(() => {
      expect(window.location.hash).toContain('panel=lab-review');
    });
  });

  it('preserves shared query focus when switching terminal and workspace nav routes', async () => {
    window.location.hash = '#/terminal/internal?artifact=price_action_breakout_pullback&panel=lab-review&section=research-heads';
    primeStore('internal', {
      snapshot: mockSnapshot as DashboardSnapshot,
      effectiveSurface: 'internal',
      snapshotPath: INTERNAL_SNAPSHOT_PATH,
    });
    await renderApp();

    await screen.findByLabelText('ops-context-strip');
    const workspaceNav = document.querySelector('[aria-label="primary-domains"] a[href*="#/workspace/artifacts"]');
    expect(workspaceNav?.getAttribute('href')).toContain('#/workspace/artifacts?artifact=price_action_breakout_pullback&panel=lab-review&section=research-heads');
    await clickAndFlush(workspaceNav as HTMLElement);

    await waitFor(() => {
      expect(window.location.hash).toContain('#/workspace/artifacts');
      expect(window.location.hash).toContain('artifact=price_action_breakout_pullback');
      expect(window.location.hash).toContain('panel=lab-review');
      expect(window.location.hash).toContain('section=research-heads');
    });
  });
});
