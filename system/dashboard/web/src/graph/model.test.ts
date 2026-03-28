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
});
