import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { TerminalReadModel } from '../types/contracts';
import { GraphHomePage } from './GraphHomePage';

vi.mock('../graph/GraphHomeRenderer', () => ({
  GraphHomeRenderer: ({
    nodes,
    onSelect,
  }: {
    nodes: Array<{ id: string; label: string }>;
    onSelect: (nodeId: string) => void;
  }) => (
    <div>
      {nodes.slice(0, 4).map((node) => (
        <button key={node.id} type="button" onClick={() => onSelect(node.id)}>
          {node.label}
        </button>
      ))}
    </div>
  ),
}));

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
        { id: 'commodity', label: '国内商品', count: 2, headline: 'policy_relief_chain' },
      ],
      interfaces: [],
      publicTopology: [],
      publicAcceptance: { summary: null, checks: [], subcommands: [] },
      surfaceContracts: [],
      fallbackChain: [],
      sourceHeads: [
        { id: 'price_action_breakout_pullback', label: 'ETH 15m 主线', summary: 'ETH breakout-pullback', path: '/review/eth.json', status: 'ok' },
      ],
      defaultFocus: {
        artifact: 'price_action_breakout_pullback',
        group: 'research_mainline',
        panel: 'lab-review',
        section: 'research-heads',
        search_scope: 'title',
      },
      artifactRows: [
        { id: 'price_action_breakout_pullback', label: 'ETH 15m 主线', path: '/review/eth.json', category: 'research', artifact_group: 'research_mainline', research_decision: 'selected', status: 'ok' },
      ],
      artifactPrimaryFocus: {},
      artifactHandoffs: {},
      artifactPayloads: {},
      raw: {},
    },
  };
}

describe('GraphHomePage', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it('recenters detail on node selection and persists default pipeline locally', async () => {
    render(<GraphHomePage model={createModel()} />);

    expect(screen.getByRole('heading', { name: '图谱化主页' })).toBeTruthy();
    expect(screen.getByRole('heading', { name: '自定义管道' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: '市场输入' }));
    expect(screen.getByRole('heading', { name: '市场输入' })).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: '加入自定义管道' }));
    await waitFor(() => {
      expect(window.localStorage.getItem('graph_home_pipelines_v1')).toContain('pipeline-market-input');
    });
  });
});
