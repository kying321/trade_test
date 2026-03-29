import { describe, expect, it } from 'vitest';
import type { TerminalReadModel } from '../types/contracts';
import { buildGraphHomeModel } from './model';

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
      domains: [
        { id: 'orchestration', label: '调度域', count: 3, headline: 'operator panel' },
        { id: 'commodity', label: '国内商品', count: 2, headline: 'policy_relief_chain' },
      ],
      interfaces: [],
      publicTopology: [],
      publicAcceptance: {
        summary: null,
        checks: [],
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
});
