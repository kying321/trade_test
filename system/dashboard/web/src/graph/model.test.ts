import { describe, expect, it } from 'vitest';
import type { TerminalReadModel } from '../types/contracts';
import { buildGraphFocusState, buildGraphHomeModel } from './model';

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
    meta: {
      generated_at_utc: '2026-03-28T11:00:00Z',
      change_class: 'RESEARCH_ONLY',
    },
    uiRoutes: {
      terminal_public: '#/terminal/public',
      terminal_internal: '#/terminal/internal',
      workspace_artifacts: '#/workspace/artifacts',
    },
    experienceContract: {
      mode: 'read_only_snapshot',
    },
    navigation: {
      terminalPrimaryArtifact: {},
      terminalHandoffs: {},
    },
    view: {} as TerminalReadModel['view'],
    orchestration: {
      alerts: [],
      topMetrics: [],
      freshness: [],
      guards: [
        {
          id: 'remote_live_gate',
          label: '远端实盘门禁',
          value: 'block',
          tone: 'warning',
        },
      ],
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
      actionQueue: [
        {
          slot: 'slot-1',
          symbol: 'ETHUSDT',
          action: '收窄风险',
          reason: '门禁漂移',
          state: 'warning',
          source_artifact: 'price_action_breakout_pullback',
        },
      ],
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
      projection: {
        status: 'active',
        summary: {
          headline: '保持前端信息架构与用户意图对齐',
        },
        events: [],
        actions: [
          {
            feedback_id: 'ui_density_split',
            recommended_action: '拆分研究工作区左栏',
            impact_score: 84,
            anchors: [
              {
                route: '/workspace/artifacts',
                artifact: 'price_action_breakout_pullback',
                component: 'WorkspacePanels',
              },
            ],
          },
        ],
        trends: [],
        anchors: [
          {
            route: '/workspace/artifacts',
            artifact: 'price_action_breakout_pullback',
            component: 'WorkspacePanels',
          },
        ],
      },
      domainGroups: [],
    },
    workspace: {
      domains: [
        { id: 'orchestration', label: '调度域', count: 3, headline: 'operator panel' },
        { id: 'commodity', label: '国内商品', count: 2, headline: 'policy_relief_chain' },
      ],
      interfaces: [],
      publicTopology: [],
      publicAcceptance: {
        summary: null,
        checks: [
          {
            id: 'graph_home_smoke',
            label: '图谱主页烟测',
            status: 'ok',
            inspector_route: '#/workspace/contracts?page_section=contracts-check-graph_home_smoke',
          },
          {
            id: 'workspace_routes_smoke',
            label: '工作区五页面烟测',
            status: 'ok',
            inspector_route: '#/workspace/contracts?page_section=contracts-check-workspace_routes_smoke',
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
          summary: 'ETH breakout-pullback',
          path: '/review/eth.json',
          status: 'ok',
        },
        {
          id: 'hold_selection_handoff',
          label: '持仓迁移',
          summary: '当前主线持仓迁移摘要',
          path: '/review/hold_selection_handoff.json',
          status: 'warning',
        },
      ],
      defaultFocus: {
        artifact: 'price_action_breakout_pullback',
        group: 'research_mainline',
        panel: 'lab-review',
        section: 'research-heads',
        search_scope: 'title',
      },
      artifactRows: [
        {
          id: 'price_action_breakout_pullback',
          label: 'ETH 15m 主线',
          artifact_group: 'research_mainline',
          category: 'research',
          research_decision: 'selected',
          status: 'ok',
          path: '/review/eth.json',
        },
      ],
      artifactPrimaryFocus: {},
      artifactHandoffs: {},
      artifactPayloads: {},
      raw: {},
    },
  };
}

describe('graph home model', () => {
  it('builds trade hub, pipeline stages, and route/artifact nodes from read model', () => {
    const graph = buildGraphHomeModel(createModel(), []);

    expect(graph.nodes.some((node) => node.id === 'trade-hub' && node.kind === 'strategy')).toBe(true);
    expect(graph.nodes.some((node) => node.id === 'pipeline-market-input' && node.kind === 'pipeline_stage')).toBe(true);
    expect(graph.nodes.some((node) => node.id === 'route-terminal-public' && node.kind === 'route')).toBe(true);
    expect(graph.nodes.some((node) => node.id === 'artifact-price_action_breakout_pullback' && node.kind === 'artifact')).toBe(true);
    expect(graph.edges.some((edge) => edge.source === 'trade-hub' && edge.target === 'pipeline-research-judgment')).toBe(true);
  });

  it('builds focused deep-link destinations for stage, route, and artifact nodes', () => {
    const graph = buildGraphHomeModel(createModel(), []);
    const artifactDestination = graph.nodes.find((node) => node.id === 'artifact-price_action_breakout_pullback')?.destination;
    const artifactParams = new URLSearchParams(artifactDestination?.split('?')[1] || '');

    expect(graph.nodes.find((node) => node.id === 'pipeline-market-input')?.destination).toBe('/terminal/public?anchor=terminal-data-regime&panel=data-regime');
    expect(graph.nodes.find((node) => node.id === 'pipeline-feedback')?.destination).toBe('/workspace/alignment?page_section=alignment-actions');
    expect(graph.nodes.find((node) => node.id === 'route-workspace-contracts')?.destination).toBe('/workspace/contracts?page_section=contracts-source-heads');
    expect(artifactDestination?.startsWith('/workspace/artifacts?')).toBe(true);
    expect(artifactParams.get('artifact')).toBe('price_action_breakout_pullback');
    expect(artifactParams.get('anchor')).toBe('workspace-artifacts-focus-panel');
    expect(artifactParams.get('search_scope')).toBe('title');
    expect(artifactParams.get('panel')).toBe('lab-review');
    expect(artifactParams.get('section')).toBe('research-heads');
    expect(artifactParams.get('group')).toBe('research_mainline');
  });

  it('derives source-head nodes from source-owned contracts heads even when they are not artifact rows', () => {
    const graph = buildGraphHomeModel(createModel(), []);
    const sourceHeadNode = graph.nodes.find((node) => node.id === 'source-head-hold_selection_handoff');

    expect(sourceHeadNode).toBeTruthy();
    expect(sourceHeadNode?.kind).toBe('artifact');
    expect(sourceHeadNode?.label).toBe('持仓迁移');
    expect(sourceHeadNode?.destination).toBe('/workspace/contracts?page_section=contracts-source-head-hold_selection_handoff');
    expect(graph.edges.some((edge) => edge.source === 'pipeline-research-judgment' && edge.target === 'source-head-hold_selection_handoff')).toBe(true);
  });

  it('derives contracts acceptance route nodes from source-owned inspector routes', () => {
    const graph = buildGraphHomeModel(createModel(), []);
    const acceptanceNode = graph.nodes.find((node) => node.id === 'acceptance-check-graph_home_smoke');

    expect(acceptanceNode).toBeTruthy();
    expect(acceptanceNode?.kind).toBe('route');
    expect(acceptanceNode?.label).toBe('图谱主页烟测');
    expect(acceptanceNode?.status).toBe('ok');
    expect(acceptanceNode?.destination).toBe('/workspace/contracts?page_section=contracts-check-graph_home_smoke');
    expect(graph.edges.some((edge) => edge.source === 'route-workspace-contracts' && edge.target === 'acceptance-check-graph_home_smoke')).toBe(true);
  });

  it('derives execution guard and action-queue nodes from source-owned terminal state', () => {
    const graph = buildGraphHomeModel(createModel(), []);
    const guardsRoute = graph.nodes.find((node) => node.id === 'route-terminal-guards');
    const guardNode = graph.nodes.find((node) => node.id === 'guard-remote-live-gate');
    const actionStackRoute = graph.nodes.find((node) => node.id === 'route-terminal-action-stack');
    const queueNode = graph.nodes.find((node) => node.id === 'queue-ethusdt');

    expect(guardsRoute?.destination).toBe('/terminal/public?panel=orchestration&section=guards');
    expect(guardNode?.label).toBe('远端实盘门禁');
    expect(graph.edges.some((edge) => edge.source === 'route-terminal-guards' && edge.target === 'guard-remote-live-gate')).toBe(true);
    expect(actionStackRoute?.destination).toBe('/terminal/public?panel=signal-risk&section=action-stack');
    expect(queueNode?.label).toContain('ETHUSDT');
    expect(queueNode?.subtitle).toContain('收窄风险');
    expect(queueNode?.destination).toBe('/terminal/public?artifact=price_action_breakout_pullback&panel=signal-risk&section=action-stack&row=ETHUSDT%3A%3A%E6%94%B6%E7%AA%84%E9%A3%8E%E9%99%A9&symbol=ETHUSDT');
    expect(graph.edges.some((edge) => edge.source === 'route-terminal-action-stack' && edge.target === 'queue-ethusdt')).toBe(true);
  });

  it('derives feedback action nodes from source-owned feedback projection', () => {
    const graph = buildGraphHomeModel(createModel(), []);
    const alignmentRoute = graph.nodes.find((node) => node.id === 'route-workspace-alignment');
    const feedbackNode = graph.nodes.find((node) => node.id === 'feedback-action-ui-density-split');

    expect(alignmentRoute?.destination).toBe('/workspace/alignment?page_section=alignment-actions');
    expect(feedbackNode?.label).toBe('拆分研究工作区左栏');
    expect(feedbackNode?.status).toBe('active');
    expect(feedbackNode?.destination).toBe('/workspace/alignment?artifact=price_action_breakout_pullback&anchor=alignment-event-ui_density_split&page_section=alignment-events');
    expect(graph.edges.some((edge) => edge.source === 'pipeline-feedback' && edge.target === 'feedback-action-ui-density-split')).toBe(true);
    expect(graph.edges.some((edge) => edge.source === 'route-workspace-alignment' && edge.target === 'feedback-action-ui-density-split')).toBe(true);
  });

  it('builds a focused impact chain without leaking into sibling branches', () => {
    const graph = buildGraphHomeModel(createModel(), []);
    const focus = buildGraphFocusState(graph, 'queue-ethusdt');

    expect(focus.activeNodeIds).toEqual([
      'trade-hub',
      'pipeline-execution-risk',
      'route-terminal-action-stack',
      'queue-ethusdt',
    ]);
    expect(focus.activeEdgeIds).toEqual([
      'trade-hub->pipeline-execution-risk',
      'pipeline-execution-risk->route-terminal-action-stack',
      'route-terminal-action-stack->queue-ethusdt',
    ]);
    expect(focus.activeNodeIds).not.toContain('route-terminal-public');
    expect(focus.activeNodeIds).not.toContain('route-search');
  });
});
