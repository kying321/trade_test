import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import type { TerminalReadModel } from '../types/contracts';
import { InspectorRail } from './InspectorRail';

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
      publicAcceptance: {
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
          },
        ],
        subcommands: [],
      },
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
});
