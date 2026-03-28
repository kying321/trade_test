import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { HashRouter } from 'react-router-dom';
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
    render(
      <HashRouter>
        <GraphHomePage model={createModel()} />
      </HashRouter>,
    );

    expect(screen.getByRole('heading', { name: '图谱化主页' })).toBeTruthy();
    expect(screen.getByRole('heading', { name: '自定义管道' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: '市场输入' }));
    expect(screen.getByRole('heading', { name: '市场输入' })).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: '加入自定义管道' }));
    await waitFor(() => {
      expect(window.localStorage.getItem('graph_home_pipelines_v1')).toContain('pipeline-market-input');
    });
  });

  it('shows localized filters, explicit quick-entry links, and a recenter action for the current node', () => {
    render(
      <HashRouter>
        <GraphHomePage model={createModel()} />
      </HashRouter>,
    );

    expect(screen.getByRole('button', { name: '管道阶段' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '市场输入域' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '子页面' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '研究工件' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '自定义管道' })).toBeTruthy();
    expect(screen.getByRole('link', { name: '去操作终端' }).getAttribute('href')).toBe('#/terminal/public');
    expect(screen.getByRole('link', { name: '去研究工作区' }).getAttribute('href')).toBe('#/workspace/artifacts');
    expect(screen.getByRole('link', { name: '打开全局搜索' }).getAttribute('href')).toBe('#/search');
    expect(screen.getByText('为什么现在看它')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: '市场输入' }));
    expect(screen.getByRole('heading', { name: '市场输入' })).toBeTruthy();
    expect(screen.getByText('它是一等阶段节点')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: '回到交易中枢' }));
    expect(screen.getByRole('heading', { name: '交易中枢' })).toBeTruthy();
  });

  it('surfaces precise deep links for selected stage and artifact nodes instead of coarse page jumps', () => {
    render(
      <HashRouter>
        <GraphHomePage model={createModel()} />
      </HashRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: '市场输入' }));
    expect(screen.getByRole('link', { name: '进入子网页' }).getAttribute('href')).toBe('#/terminal/public?anchor=terminal-data-regime&panel=data-regime');

    fireEvent.change(screen.getByPlaceholderText('过滤节点：标签 / 副标题'), { target: { value: 'ETH' } });
    fireEvent.click(screen.getByRole('button', { name: 'ETH 15m 主线' }));
    const artifactHref = screen.getByRole('link', { name: '进入子网页' }).getAttribute('href') || '';
    const artifactParams = new URLSearchParams(artifactHref.split('?')[1] || '');
    expect(artifactHref.startsWith('#/workspace/artifacts?')).toBe(true);
    expect(artifactParams.get('artifact')).toBe('price_action_breakout_pullback');
    expect(artifactParams.get('anchor')).toBe('workspace-artifacts-focus-panel');
    expect(artifactParams.get('search_scope')).toBe('title');
    expect(artifactParams.get('panel')).toBe('lab-review');
    expect(artifactParams.get('section')).toBe('research-heads');
    expect(artifactParams.get('group')).toBe('research_mainline');
  });
});
