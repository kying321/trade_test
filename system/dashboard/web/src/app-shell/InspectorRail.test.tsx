import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import type { TerminalReadModel } from '../types/contracts';
import { InspectorRail } from './InspectorRail';

function createContractsAcceptanceFixture(): TerminalReadModel['workspace']['publicAcceptance'] {
  return {
    summary: {
      artifact_id: 'dashboard_public_acceptance',
      status: 'ok',
      change_class: 'RESEARCH_ONLY',
      generated_at_utc: '2026-03-20T09:01:54Z',
      report_path: '/system/output/review/dashboard_public_acceptance.json',
      artifact_path: '/system/output/review/dashboard_workspace_routes_browser_smoke.json',
      topology_status: 'ok',
      workspace_status: 'ok',
    },
    checks: [
      {
        id: 'workspace-routes-smoke',
        label: '工作区五页面烟测',
        status: 'ok',
        ok: true,
        report_path: '/system/output/review/dashboard_workspace_routes_browser_smoke.json',
        requested_surface: 'public',
        effective_surface: 'public',
        snapshot_endpoint_observed: '/data/fenlie_dashboard_snapshot.json',
        public_snapshot_fetch_count: 4,
        internal_snapshot_fetch_count: 0,
        inspector_route: '#/workspace/contracts?page_section=contracts-check-workspace_routes_smoke',
        inspector_search_link_href: '#/search?q=explicit_workspace_routes_check&scope=artifact',
        inspector_artifact_link_href: '#/workspace/artifacts?artifact=audit:explicit:workspace_routes_check:trade_journal',
        inspector_raw_link_href: '#/workspace/raw?artifact=%2Ftmp%2Fexplicit_workspace_routes_check.csv',
      },
      {
        id: 'graph_home_smoke',
        label: '图谱主页烟测',
        status: 'ok',
        ok: true,
        report_path: '/system/output/review/dashboard_graph_home_browser_smoke.json',
        graph_home_resolved_route: '#/graph-home',
        graph_home_default_center: '交易中枢',
        inspector_route: '#/workspace/contracts?page_section=contracts-check-graph_home_smoke',
        inspector_search_link_href: '#/search?q=explicit_graph_home_check&scope=artifact',
        inspector_artifact_link_href: '#/workspace/artifacts?artifact=audit:explicit:graph_home_check:trade_journal',
        inspector_raw_link_href: '#/workspace/raw?artifact=%2Ftmp%2Fexplicit_graph_home_check.csv',
        research_audit_cases: [
          {
            case_id: 'optimizer_trial_trade_journal',
            query: 'trial_001_ultra_short_trade_journal',
            search_route: '#/search?q=trial_001_ultra_short_trade_journal&scope=artifact',
            workspace_route: '#/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal',
            raw_path: 'system/output/research/20260330_133956/ultra_short/trial_001_ultra_short_trade_journal.csv',
            result_artifact: 'audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal',
          },
        ],
      },
      {
        id: 'topology_smoke',
        label: '入口拓扑烟测',
        status: 'ok',
        ok: true,
        report_path: '/system/output/review/dashboard_public_topology_smoke.json',
        frontend_public: 'https://fuuu.fun',
        root_public_entry: 'root-nav-proxy',
        root_overview_screenshot_path: 'review/root_overview_browser.png',
        pages_overview_screenshot_path: 'review/pages_overview_browser.png',
        root_contracts_screenshot_path: 'review/root_contracts_browser.png',
        pages_contracts_screenshot_path: 'review/pages_contracts_browser.png',
        inspector_route: '#/workspace/contracts?page_section=contracts-check-topology_smoke',
      },
    ],
    subcommands: [
      {
        id: 'graph_home_smoke',
        label: '图谱主页子命令',
        returncode: 0,
        payload_present: true,
        stdout_bytes: 96,
        stderr_bytes: 0,
        cmd: 'python3 run_dashboard_workspace_artifacts_smoke.py --mode graph_home',
        cwd: '/tmp',
        inspector_route: '#/workspace/contracts?page_section=contracts-subcommand-graph_home_smoke',
        inspector_search_link_href: '#/search?q=explicit_graph_home_subcommand&scope=artifact',
        inspector_artifact_link_href: '#/workspace/artifacts?artifact=audit:explicit:graph_home_subcommand:trade_journal',
        inspector_raw_link_href: '#/workspace/raw?artifact=%2Ftmp%2Fexplicit_graph_home_subcommand.csv',
        inspector_check_route: '#/workspace/contracts?page_section=contracts-check-graph_home_smoke',
      },
      {
        id: 'workspace-routes-smoke',
        label: '工作区路由子命令',
        returncode: 0,
        payload_present: true,
        stdout_bytes: 120,
        stderr_bytes: 0,
        cmd: 'python3 run_dashboard_workspace_artifacts_smoke.py',
        cwd: '/tmp',
        inspector_route: '#/workspace/contracts?page_section=contracts-subcommand-workspace_routes_smoke',
        inspector_search_link_href: '#/search?q=explicit_workspace_routes_subcommand&scope=artifact',
        inspector_artifact_link_href: '#/workspace/artifacts?artifact=audit:explicit:workspace_routes_subcommand:trade_journal',
        inspector_raw_link_href: '#/workspace/raw?artifact=%2Ftmp%2Fexplicit_workspace_routes_subcommand.csv',
        inspector_check_route: '#/workspace/contracts?page_section=contracts-check-workspace_routes_smoke',
      },
      {
        id: 'topology_smoke',
        label: '入口拓扑子命令',
        returncode: 0,
        payload_present: true,
        stdout_bytes: 32,
        stderr_bytes: 0,
        cmd: 'python3 run_dashboard_public_topology_smoke.py',
        cwd: '/tmp',
        inspector_route: '#/workspace/contracts?page_section=contracts-subcommand-topology_smoke',
        inspector_check_route: '#/workspace/contracts?page_section=contracts-check-topology_smoke',
      },
    ],
  };
}

function createModel(): TerminalReadModel {
  return {
    surface: {
      requested: 'public',
      effective: 'public',
      snapshotPath: '/data/fenlie_dashboard_snapshot.json',
      warnings: [],
      requestedLabel: '公开终端',
      effectiveLabel: '公开终端',
      redactionLevel: 'public_summary',
      availability: 'active',
    },
    meta: {},
    uiRoutes: {},
    experienceContract: {},
    navigation: {
      terminalPrimaryArtifact: {
        'lab-review|research-heads|': 'price_action_breakout_pullback',
      },
      terminalHandoffs: {},
    },
    view: {} as TerminalReadModel['view'],
    orchestration: {
      alerts: [],
      topMetrics: [],
      freshness: [],
      guards: [],
      actionLog: [],
      checklist: [],
    },
    dataRegime: {
      sourceConfidence: [],
      microCapture: [],
      regimeSummary: [],
      regimeMatrix: [],
    },
    signalRisk: {
      laneCards: [],
      focusSlots: [],
      gateScores: [],
      controlChain: [],
      actionQueue: [],
      repairPlan: [],
      optimizationBacklog: [],
    },
    labReview: {
      researchHeads: [],
      backtests: [],
      comparisonRows: [],
      manifests: [],
      candidateRows: [],
      stressSummary: [],
    },
    feedback: {
      projection: null,
      domainGroups: [],
    },
    workspace: {
      domains: [],
      interfaces: [],
      publicTopology: [
        {
          id: 'root-public',
          label: 'Root Public',
          role: 'public',
          url: 'https://fuuu.fun',
          expected_status: '200',
        },
      ],
      publicAcceptance: createContractsAcceptanceFixture(),
      surfaceContracts: [],
      fallbackChain: ['/data/fenlie_dashboard_snapshot.json'],
      sourceHeads: [
        {
          id: 'price_action_breakout_pullback',
          label: 'ETH 15m 主线',
          summary: 'ETH 15m breakout-pullback',
          path: '/system/output/review/source_head_eth.json',
          status: 'selected',
        },
      ],
      artifactRows: [
        {
          id: 'price_action_breakout_pullback',
          label: 'ETH 15m 主线',
          payload_key: 'price_action_breakout_pullback',
          path: '/system/output/review/price_action_breakout_pullback.json',
          category: 'review',
          artifact_group: 'research_mainline',
          research_decision: 'selected',
          status: 'ok',
        },
      ],
      artifactPrimaryFocus: {
        price_action_breakout_pullback: {
          panel: 'lab-review',
          section: 'research-heads',
        },
      },
      artifactHandoffs: {
        price_action_breakout_pullback: [
          {
            id: 'handoff-raw',
            label: '查看原始层',
            to: '/workspace/raw?artifact=price_action_breakout_pullback',
          },
        ],
      },
      artifactPayloads: {},
      raw: {
        action: 'dashboard_snapshot',
        status: 'ok',
      },
    },
  };
}

describe('InspectorRail', () => {
  it('renders source-owned focus summary evidence paths and drill-back links for workspace focus', () => {
    render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          focus={{
            artifact: 'price_action_breakout_pullback',
            panel: 'lab-review',
            section: 'research-heads',
            row: 'ETHUSDT',
          }}
          model={createModel()}
          context={{
            domain: 'workspace',
            section: 'artifacts',
            surface: 'public',
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('研究焦点')).toBeTruthy();
    expect(screen.getByText('研究追源')).toBeTruthy();
    expect(screen.getByText('只读回退')).toBeTruthy();
    expect(screen.getByText('所属分组')).toBeTruthy();
    expect(screen.getByText('检索范围')).toBeTruthy();
    expect(screen.getAllByTitle('/system/output/review/price_action_breakout_pullback.json').length).toBeGreaterThan(0);
    expect(screen.getAllByTitle('/system/output/review/source_head_eth.json').length).toBeGreaterThan(0);
    expect(screen.getAllByTitle('/data/fenlie_dashboard_snapshot.json').length).toBeGreaterThan(0);

    const overviewLink = screen.getByRole('link', { name: '返回总览' });
    expect(overviewLink.getAttribute('href')).toBe('/overview?artifact=price_action_breakout_pullback&panel=lab-review&section=research-heads&row=ETHUSDT');

    const workspaceLink = screen.getByRole('link', { name: '返回研究工作区' });
    expect(workspaceLink.getAttribute('href')).toBe('/workspace/artifacts?artifact=price_action_breakout_pullback&panel=lab-review&section=research-heads&row=ETHUSDT');

    expect(screen.getByRole('link', { name: '查看原始层' }).getAttribute('href')).toBe('/workspace/raw?artifact=price_action_breakout_pullback');
  });

  it('renders terminal-specific inspector vocabulary with positioning and next-step guidance instead of repeating top-strip view fields', () => {
    render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          focus={{
            panel: 'signal-risk',
            section: 'focus-slots',
            row: 'primary',
          }}
          model={createModel()}
          context={{
            domain: 'terminal',
            surface: 'public',
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('终端定位')).toBeTruthy();
    expect(screen.getByText('下一跳建议')).toBeTruthy();
    expect(screen.getByText('穿透键')).toBeTruthy();
    expect(screen.getByText('证据动作')).toBeTruthy();
    expect(screen.queryByText('请求视图')).toBeNull();
    expect(screen.queryByText('快照可见性')).toBeNull();
  });

  it('renders contracts-specific inspector template for acceptance and topology context', () => {
    render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          model={createModel()}
          context={{
            domain: 'workspace',
            section: 'contracts',
            surface: 'public',
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('契约验收摘要')).toBeTruthy();
    expect(screen.getByText('公开面链路')).toBeTruthy();
    expect(screen.getAllByText('工作区五页面烟测').length).toBeGreaterThan(0);
    expect(screen.getAllByTitle('/system/output/review/dashboard_public_acceptance.json').length).toBeGreaterThan(0);
  });

  it('keeps contracts summary focus source-owned by id instead of depending on array order', () => {
    const model = createModel();
    model.workspace.publicAcceptance.checks = [
      ...(model.workspace.publicAcceptance.checks.filter((row) => row.id === 'topology_smoke')),
      ...(model.workspace.publicAcceptance.checks.filter((row) => row.id === 'graph_home_smoke')),
      ...(model.workspace.publicAcceptance.checks.filter((row) => row.id === 'workspace-routes-smoke')),
    ];
    model.workspace.publicTopology = [
      {
        id: 'fallback-probe',
        label: 'Fallback Probe',
        role: 'fallback',
        url: 'https://fallback.example.com',
        expected_status: '200',
      },
      ...model.workspace.publicTopology,
    ];

    render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          model={model}
          context={{
            domain: 'workspace',
            section: 'contracts',
            surface: 'public',
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('契约验收摘要')).toBeTruthy();
    expect(screen.getByText('公开面链路')).toBeTruthy();
    expect(screen.getAllByText('工作区五页面烟测').length).toBeGreaterThan(0);
    expect(screen.getAllByTitle('https://fuuu.fun').length).toBeGreaterThan(0);
    expect(screen.queryByTitle('https://fallback.example.com')).toBeNull();
  });

  it('renders contracts acceptance focus with research-audit search artifact and raw links', () => {
    render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          model={createModel()}
          context={{
            domain: 'workspace',
            section: 'contracts',
            surface: 'public',
            activeSectionId: 'contracts-check-graph_home_smoke',
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('当前验收项')).toBeTruthy();
    expect(screen.getByText('验收证据')).toBeTruthy();
    expect(screen.getByRole('link', { name: '查看研究审计检索' }).getAttribute('href')).toBe('/search?q=explicit_graph_home_check&scope=artifact');
    expect(screen.getByRole('link', { name: '查看研究审计工件' }).getAttribute('href')).toBe('/workspace/artifacts?artifact=audit:explicit:graph_home_check:trade_journal');
    expect(screen.getByRole('link', { name: '查看研究审计原始层' }).getAttribute('href')).toBe('/workspace/raw?artifact=%2Ftmp%2Fexplicit_graph_home_check.csv');
  });

  it('renders contracts subcommand focus with linked research-audit artifact and raw links', () => {
    render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          model={createModel()}
          context={{
            domain: 'workspace',
            section: 'contracts',
            surface: 'public',
            activeSectionId: 'contracts-subcommand-graph_home_smoke',
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('当前子命令')).toBeTruthy();
    expect(screen.getByText('子命令证据')).toBeTruthy();
    expect(screen.getByRole('link', { name: '查看研究审计检索' }).getAttribute('href')).toBe('/search?q=explicit_graph_home_subcommand&scope=artifact');
    expect(screen.getByRole('link', { name: '查看研究审计工件' }).getAttribute('href')).toBe('/workspace/artifacts?artifact=audit:explicit:graph_home_subcommand:trade_journal');
    expect(screen.getByRole('link', { name: '查看研究审计原始层' }).getAttribute('href')).toBe('/workspace/raw?artifact=%2Ftmp%2Fexplicit_graph_home_subcommand.csv');
  });

  it('renders workspace-routes contracts focus from row-level explicit inspector map', () => {
    render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          model={createModel()}
          context={{
            domain: 'workspace',
            section: 'contracts',
            surface: 'public',
            activeSectionId: 'contracts-check-workspace-routes-smoke',
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('当前验收项')).toBeTruthy();
    expect(screen.getByRole('link', { name: '查看研究审计检索' }).getAttribute('href')).toBe('/search?q=explicit_workspace_routes_check&scope=artifact');
    expect(screen.getByRole('link', { name: '查看研究审计工件' }).getAttribute('href')).toBe('/workspace/artifacts?artifact=audit:explicit:workspace_routes_check:trade_journal');
    expect(screen.getByRole('link', { name: '查看研究审计原始层' }).getAttribute('href')).toBe('/workspace/raw?artifact=%2Ftmp%2Fexplicit_workspace_routes_check.csv');
  });

  it('renders workspace-routes subcommand focus from row-level explicit inspector map', () => {
    render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          model={createModel()}
          context={{
            domain: 'workspace',
            section: 'contracts',
            surface: 'public',
            activeSectionId: 'contracts-subcommand-workspace-routes-smoke',
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('当前子命令')).toBeTruthy();
    expect(screen.getByRole('link', { name: '查看研究审计检索' }).getAttribute('href')).toBe('/search?q=explicit_workspace_routes_subcommand&scope=artifact');
    expect(screen.getByRole('link', { name: '查看研究审计工件' }).getAttribute('href')).toBe('/workspace/artifacts?artifact=audit:explicit:workspace_routes_subcommand:trade_journal');
    expect(screen.getByRole('link', { name: '查看研究审计原始层' }).getAttribute('href')).toBe('/workspace/raw?artifact=%2Ftmp%2Fexplicit_workspace_routes_subcommand.csv');
  });

  it('does not borrow shared research-audit cases for workspace-routes check when row-level source-owned links are absent', () => {
    const model = createModel();
    const workspaceRoutesCheck = model.workspace.publicAcceptance.checks.find((row) => row.id === 'workspace-routes-smoke');
    if (workspaceRoutesCheck) {
      workspaceRoutesCheck.inspector_search_link_href = undefined;
      workspaceRoutesCheck.inspector_artifact_link_href = undefined;
      workspaceRoutesCheck.inspector_raw_link_href = undefined;
      workspaceRoutesCheck.research_audit_cases = [];
    }

    render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          model={model}
          context={{
            domain: 'workspace',
            section: 'contracts',
            surface: 'public',
            activeSectionId: 'contracts-check-workspace-routes-smoke',
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('当前验收项')).toBeTruthy();
    expect(screen.queryByRole('link', { name: '查看研究审计检索' })).toBeNull();
    expect(screen.queryByRole('link', { name: '查看研究审计工件' })).toBeNull();
    expect(screen.queryByRole('link', { name: '查看研究审计原始层' })).toBeNull();
  });

  it('does not borrow shared research-audit cases for workspace-routes subcommand when row-level source-owned links are absent', () => {
    const model = createModel();
    const workspaceRoutesSubcommand = model.workspace.publicAcceptance.subcommands.find((row) => row.id === 'workspace-routes-smoke');
    if (workspaceRoutesSubcommand) {
      workspaceRoutesSubcommand.inspector_search_link_href = undefined;
      workspaceRoutesSubcommand.inspector_artifact_link_href = undefined;
      workspaceRoutesSubcommand.inspector_raw_link_href = undefined;
    }

    render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          model={model}
          context={{
            domain: 'workspace',
            section: 'contracts',
            surface: 'public',
            activeSectionId: 'contracts-subcommand-workspace-routes-smoke',
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('当前子命令')).toBeTruthy();
    expect(screen.queryByRole('link', { name: '查看研究审计检索' })).toBeNull();
    expect(screen.queryByRole('link', { name: '查看研究审计工件' })).toBeNull();
    expect(screen.queryByRole('link', { name: '查看研究审计原始层' })).toBeNull();
  });

  it('keeps topology contracts focus free of unrelated research-audit links', () => {
    render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          model={createModel()}
          context={{
            domain: 'workspace',
            section: 'contracts',
            surface: 'public',
            activeSectionId: 'contracts-check-topology_smoke',
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('当前验收项')).toBeTruthy();
    expect(screen.queryByRole('link', { name: '查看研究审计检索' })).toBeNull();
    expect(screen.queryByRole('link', { name: '查看研究审计工件' })).toBeNull();
    expect(screen.queryByRole('link', { name: '查看研究审计原始层' })).toBeNull();
  });

  it('keeps topology contracts subcommand focus free of unrelated research-audit links', () => {
    render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          model={createModel()}
          context={{
            domain: 'workspace',
            section: 'contracts',
            surface: 'public',
            activeSectionId: 'contracts-subcommand-topology_smoke',
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('当前子命令')).toBeTruthy();
    expect(screen.queryByRole('link', { name: '查看研究审计检索' })).toBeNull();
    expect(screen.queryByRole('link', { name: '查看研究审计工件' })).toBeNull();
    expect(screen.queryByRole('link', { name: '查看研究审计原始层' })).toBeNull();
  });

  it('does not fall back to summary legacy inspector links for active contracts check when row-level source-owned links are absent', () => {
    const model = createModel();
    const graphHomeCheck = model.workspace.publicAcceptance.checks.find((row) => row.id === 'graph_home_smoke');
    if (graphHomeCheck) {
      graphHomeCheck.inspector_search_link_href = undefined;
      graphHomeCheck.inspector_artifact_link_href = undefined;
      graphHomeCheck.inspector_raw_link_href = undefined;
      graphHomeCheck.research_audit_cases = [];
    }
    Object.assign((model.workspace.publicAcceptance.summary || {}) as any, {
      contracts_acceptance_check_search_link_href: '#/search?q=legacy_summary_check&scope=artifact',
      contracts_acceptance_check_artifact_link_href: '#/workspace/artifacts?artifact=audit:legacy_summary_check',
      contracts_acceptance_check_raw_link_href: '#/workspace/raw?artifact=%2Ftmp%2Flegacy_summary_check.csv',
    });

    render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          model={model}
          context={{
            domain: 'workspace',
            section: 'contracts',
            surface: 'public',
            activeSectionId: 'contracts-check-graph_home_smoke',
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('当前验收项')).toBeTruthy();
    expect(screen.queryByRole('link', { name: '查看研究审计检索' })).toBeNull();
    expect(screen.queryByRole('link', { name: '查看研究审计工件' })).toBeNull();
    expect(screen.queryByRole('link', { name: '查看研究审计原始层' })).toBeNull();
  });

  it('does not fall back to summary legacy inspector links for active contracts subcommand when row-level source-owned links are absent', () => {
    const model = createModel();
    const graphHomeCheck = model.workspace.publicAcceptance.checks.find((row) => row.id === 'graph_home_smoke');
    if (graphHomeCheck) {
      graphHomeCheck.research_audit_cases = [];
    }
    const graphHomeSubcommand = model.workspace.publicAcceptance.subcommands.find((row) => row.id === 'graph_home_smoke');
    if (graphHomeSubcommand) {
      graphHomeSubcommand.inspector_search_link_href = undefined;
      graphHomeSubcommand.inspector_artifact_link_href = undefined;
      graphHomeSubcommand.inspector_raw_link_href = undefined;
    }
    Object.assign((model.workspace.publicAcceptance.summary || {}) as any, {
      contracts_acceptance_subcommand_search_link_href: '#/search?q=legacy_summary_subcommand&scope=artifact',
      contracts_acceptance_subcommand_artifact_link_href: '#/workspace/artifacts?artifact=audit:legacy_summary_subcommand',
      contracts_acceptance_subcommand_raw_link_href: '#/workspace/raw?artifact=%2Ftmp%2Flegacy_summary_subcommand.csv',
      contracts_acceptance_check_route: '#/workspace/contracts?page_section=contracts-check-graph_home_smoke',
    });

    render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          model={model}
          context={{
            domain: 'workspace',
            section: 'contracts',
            surface: 'public',
            activeSectionId: 'contracts-subcommand-graph_home_smoke',
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('当前子命令')).toBeTruthy();
    expect(screen.queryByRole('link', { name: '查看研究审计检索' })).toBeNull();
    expect(screen.queryByRole('link', { name: '查看研究审计工件' })).toBeNull();
    expect(screen.queryByRole('link', { name: '查看研究审计原始层' })).toBeNull();
  });

  it('renders raw-specific inspector template for raw snapshot context', () => {
    render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          focus={{
            artifact: 'price_action_breakout_pullback',
          }}
          model={createModel()}
          context={{
            domain: 'workspace',
            section: 'raw',
            surface: 'public',
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('原始快照摘要')).toBeTruthy();
    expect(screen.getByText('原始证据')).toBeTruthy();
    expect(screen.getAllByText('原始快照视图').length).toBeGreaterThan(0);
    expect(screen.getAllByTitle('/data/fenlie_dashboard_snapshot.json').length).toBeGreaterThan(0);
  });

  it('maps terminal artifact path focus back to workspace artifact evidence when focus carries a path instead of catalog id', () => {
    const model = createModel();
    model.navigation.terminalPrimaryArtifact['signal-risk|focus-slots|primary'] = '/system/output/review/price_action_breakout_pullback.json';
    model.navigation.terminalHandoffs['signal-risk|focus-slots|'] = [
      {
        id: 'stale-artifact-handoff',
        label: '查看工件 / /tmp/conflict.json',
        to: '/workspace/artifacts?artifact=%2Ftmp%2Fconflict.json&panel=signal-risk&section=focus-slots',
      },
      {
        id: 'stale-raw-handoff',
        label: '查看原始层 / /tmp/conflict.json',
        to: '/workspace/raw?artifact=%2Ftmp%2Fconflict.json&panel=signal-risk&section=focus-slots',
      },
    ];

    render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          focus={{
            panel: 'signal-risk',
            section: 'focus-slots',
            row: 'primary',
          }}
          model={model}
          context={{
            domain: 'terminal',
            surface: 'public',
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('执行证据')).toBeTruthy();
    expect(screen.queryByText('下一跳建议')).toBeNull();
    expect(screen.getAllByTitle('/system/output/review/price_action_breakout_pullback.json').length).toBeGreaterThan(0);
    expect(screen.getByRole('link', { name: '返回总览' }).getAttribute('href')).toBe('/overview?artifact=price_action_breakout_pullback&panel=signal-risk&section=focus-slots&row=primary');
    expect(screen.getByRole('link', { name: '查看工件池' }).getAttribute('href')).toBe('/workspace/artifacts?artifact=price_action_breakout_pullback');
    expect(screen.queryByRole('link', { name: /conflict\\.json/ })).toBeNull();
  });

  it('renders explicit empty state and mode metadata when model is absent', () => {
    const { container } = render(
      <MemoryRouter>
        <InspectorRail
          title="对象检查器"
          model={null}
          mode="inline"
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('暂无可检查对象，先在主舞台选择工件或终端焦点。')).toBeTruthy();
    expect(container.querySelector('.inspector-rail-inner')?.getAttribute('data-mode')).toBe('inline');
  });
});
