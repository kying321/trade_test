import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { TerminalReadModel } from '../types/contracts';
import type { GraphNode } from '../graph/types';
import { GraphHomePage, buildMissingSourceRepairHint, buildMissingSourceRepairLinks } from './GraphHomePage';

vi.mock('../graph/GraphHomeRenderer', () => ({
  GraphHomeRenderer: ({
    nodes,
    onSelect,
  }: {
    nodes: Array<{ id: string; label: string }>;
    onSelect: (nodeId: string) => void;
  }) => (
    <div>
      {nodes.map((node) => (
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
      terminal_public: '/terminal/public',
      workspace_artifacts: '/workspace/artifacts',
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
          hint: '等待远端实盘门禁解除',
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
        { id: 'commodity', label: '国内商品', count: 2, headline: 'policy_relief_chain' },
      ],
      interfaces: [],
      publicTopology: [],
      publicAcceptance: {
        summary: {
          artifact_id: 'dashboard_public_acceptance',
        },
        checks: [],
        subcommands: [],
      },
      researchAuditCases: [
        {
          case_id: 'optimizer_trial_trade_journal',
          query: 'trial_001_ultra_short_trade_journal',
          search_route: '/search?q=trial_001_ultra_short_trade_journal&scope=artifact',
          workspace_route: '/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal',
          raw_path: 'system/output/research/20260330_133956/ultra_short/trial_001_ultra_short_trade_journal.csv',
          result_artifact: 'audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal',
        },
      ],
      surfaceContracts: [],
      fallbackChain: [],
      sourceHeads: [
        { id: 'price_action_breakout_pullback', label: 'ETH 15m 主线', summary: 'ETH breakout-pullback', path: '/review/eth.json', status: 'ok' },
        { id: 'hold_selection_handoff', label: '持仓迁移', summary: '当前主线持仓迁移摘要', path: '/review/hold_selection_handoff.json', status: 'warning' },
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

  it('keeps graph home in loading state when model is not ready', () => {
    render(
      <MemoryRouter>
        <GraphHomePage model={null} />
      </MemoryRouter>,
    );

    expect(screen.getByRole('heading', { name: '图谱化主页' })).toBeTruthy();
    expect(screen.getByText('图谱主页加载中…')).toBeTruthy();
  });

  it('recenters detail on node selection and persists default pipeline locally', async () => {
    render(
      <MemoryRouter>
        <GraphHomePage model={createModel()} />
      </MemoryRouter>,
    );

    expect(screen.getByRole('heading', { name: '图谱化主页' })).toBeTruthy();
    expect(screen.getByRole('heading', { name: '自定义管道' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: '市场输入' }));
    expect(screen.getByRole('heading', { name: '市场输入' })).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: '加入自定义管道' }));
    await waitFor(() => {
      expect(window.localStorage.getItem('graph_home_pipelines_v1')).toContain('pipeline-market-input');
    });
    const pipelineItem = screen.getByTestId('graph-pipeline-item-pipeline-market-input');
    expect(within(pipelineItem).getByText('它是一等阶段节点')).toBeTruthy();
    expect(within(pipelineItem).getByText(/这一层用来承接中枢的一级工作带/)).toBeTruthy();
  });

  it('shows localized filters, explicit quick-entry links, and a recenter action for the current node', () => {
    render(
      <MemoryRouter>
        <GraphHomePage model={createModel()} />
      </MemoryRouter>,
    );

    expect(screen.getByRole('button', { name: '管道阶段' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '市场输入域' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '子页面' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '研究工件' })).toBeTruthy();
    expect(screen.getAllByRole('button', { name: '自定义管道' }).length).toBeGreaterThan(0);
    expect(screen.getByRole('link', { name: '去操作终端' }).getAttribute('href')).toBe('/ops/risk');
    expect(screen.getByRole('link', { name: '去研究工作区' }).getAttribute('href')).toBe('/workspace/artifacts');
    expect(screen.getByRole('link', { name: '去 CPA 管理' }).getAttribute('href')).toBe('/cpa');
    expect(screen.getByRole('link', { name: '打开全局搜索' }).getAttribute('href')).toBe('/search');
    expect(screen.getByText('为什么现在看它')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: '市场输入' }));
    expect(screen.getByRole('heading', { name: '市场输入' })).toBeTruthy();
    expect(screen.getByText('它是一等阶段节点')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: '回到交易中枢' }));
    expect(screen.getByRole('heading', { name: '交易中枢' })).toBeTruthy();
  });

  it('surfaces precise deep links for selected stage and artifact nodes instead of coarse page jumps', () => {
    render(
      <MemoryRouter>
        <GraphHomePage model={createModel()} />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: '市场输入' }));
    expect(screen.getByRole('link', { name: '进入子网页' }).getAttribute('href')).toBe('/ops/risk?anchor=terminal-data-regime&panel=data-regime');

    fireEvent.change(screen.getByPlaceholderText('过滤节点：标签 / 副标题'), { target: { value: 'ETH' } });
    fireEvent.click(screen.getByRole('button', { name: 'ETH 15m 主线' }));
    const artifactHref = screen.getByRole('link', { name: '进入子网页' }).getAttribute('href') || '';
    const artifactParams = new URLSearchParams(artifactHref.split('?')[1] || '');
    expect(artifactHref.startsWith('/workspace/artifacts?')).toBe(true);
    expect(artifactParams.get('artifact')).toBe('price_action_breakout_pullback');
    expect(artifactParams.get('anchor')).toBe('workspace-artifacts-focus-panel');
    expect(artifactParams.get('search_scope')).toBe('title');
    expect(artifactParams.get('panel')).toBe('lab-review');
    expect(artifactParams.get('section')).toBe('research-heads');
    expect(artifactParams.get('group')).toBe('research_mainline');
  });

  it('shows research-audit deep links inside the detail panel for feedback and search nodes', () => {
    render(
      <MemoryRouter>
        <GraphHomePage model={createModel()} />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: '复盘反馈' }));
    expect(screen.getByRole('heading', { name: '复盘反馈' })).toBeTruthy();
    expect(screen.getByRole('link', { name: '检索 / trial_001_ultra_short_trade_journal' }).getAttribute('href')).toBe('/search?q=trial_001_ultra_short_trade_journal&scope=artifact');
    expect(screen.getByRole('link', { name: '工件 / audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal' }).getAttribute('href')).toBe('/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal');
    expect(screen.getByRole('link', { name: /原始层/ }).getAttribute('href')).toBe('/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv');

    fireEvent.click(screen.getByRole('button', { name: '搜索' }));
    expect(screen.getByRole('heading', { name: '搜索' })).toBeTruthy();
    expect(screen.getByRole('link', { name: '检索 / trial_001_ultra_short_trade_journal' }).getAttribute('href')).toBe('/search?q=trial_001_ultra_short_trade_journal&scope=artifact');
  });

  it('does not fall back to acceptance summary research-audit cases when workspace source-owned cases are absent', () => {
    const model = createModel();
    model.workspace.researchAuditCases = [];
    Object.assign((model.workspace.publicAcceptance.summary ||= { artifact_id: 'dashboard_public_acceptance' }) as any, {
      research_audit_cases: [
        {
          case_id: 'summary_only_case',
          query: 'summary_only_case',
          search_route: '/search?q=summary_only_case&scope=artifact',
          workspace_route: '/workspace/artifacts?artifact=audit:summary_only_case',
          raw_path: 'system/output/research/summary_only_case.csv',
          result_artifact: 'audit:summary_only_case',
        },
      ],
    });

    render(
      <MemoryRouter>
        <GraphHomePage model={model} />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: '复盘反馈' }));
    expect(screen.getByRole('heading', { name: '复盘反馈' })).toBeTruthy();
    expect(screen.queryByText('研究审计直达：1 条')).toBeNull();
    expect(screen.queryByRole('link', { name: '检索 / summary_only_case' })).toBeNull();
    expect(screen.queryByRole('link', { name: '工件 / audit:summary_only_case' })).toBeNull();
  });

  it('shows source-owned detail evidence for action-queue and feedback nodes', () => {
    render(
      <MemoryRouter>
        <GraphHomePage model={createModel()} />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: 'ETHUSDT' }));
    expect(screen.getByRole('heading', { name: 'ETHUSDT' })).toBeTruthy();
    expect(screen.getByText('信源契约')).toBeTruthy();
    expect(screen.getByText('signalRisk.actionQueue')).toBeTruthy();
    expect(screen.getByText('源头工件')).toBeTruthy();
    expect(screen.getAllByTitle('price_action_breakout_pullback').length).toBeGreaterThan(0);
    expect(screen.getByText('触发原因')).toBeTruthy();
    expect(screen.getAllByText(/门禁漂移/).length).toBeGreaterThan(0);
    expect(screen.getByText('为什么影响当前中心')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: '拆分研究工作区左栏' }));
    expect(screen.getByRole('heading', { name: '拆分研究工作区左栏' })).toBeTruthy();
    expect(screen.getByText('反馈ID')).toBeTruthy();
    expect(screen.getByText('ui_density_split')).toBeTruthy();
    expect(screen.getByText('锚点工件')).toBeTruthy();
    expect(screen.getAllByTitle('price_action_breakout_pullback').length).toBeGreaterThan(0);
    expect(screen.getByText('下一跳')).toBeTruthy();
    expect(screen.getAllByTitle('/workspace/alignment?artifact=price_action_breakout_pullback&anchor=alignment-event-ui_density_split&page_section=alignment-events').length).toBeGreaterThan(0);
  });

  it('shows source-head contract path in the detail panel for source-owned contract nodes', () => {
    render(
      <MemoryRouter>
        <GraphHomePage model={createModel()} />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: '持仓迁移' }));
    expect(screen.getByRole('heading', { name: '持仓迁移' })).toBeTruthy();
    expect(screen.getByText('信源契约')).toBeTruthy();
    expect(screen.getByText('workspace.sourceHeads')).toBeTruthy();
    expect(screen.getByText('源头路径')).toBeTruthy();
    expect(screen.getAllByTitle('/review/hold_selection_handoff.json').length).toBeGreaterThan(0);
    expect(screen.getByText('下一跳')).toBeTruthy();
    expect(screen.getAllByTitle('/workspace/contracts?page_section=contracts-source-head-hold_selection_handoff').length).toBeGreaterThan(0);
  });

  it('uses row-level and anchor-level deep links for queue and feedback nodes instead of coarse section jumps', () => {
    render(
      <MemoryRouter>
        <GraphHomePage model={createModel()} />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: 'ETHUSDT' }));
    expect(screen.getByRole('link', { name: '进入子网页' }).getAttribute('href')).toBe(
      '/ops/risk?artifact=price_action_breakout_pullback&panel=signal-risk&section=action-stack&row=ETHUSDT%3A%3A%E6%94%B6%E7%AA%84%E9%A3%8E%E9%99%A9&symbol=ETHUSDT',
    );

    fireEvent.click(screen.getByRole('button', { name: '拆分研究工作区左栏' }));
    expect(screen.getByRole('link', { name: '进入子网页' }).getAttribute('href')).toBe(
      '/workspace/alignment?artifact=price_action_breakout_pullback&anchor=alignment-event-ui_density_split&page_section=alignment-events',
    );
  });

  it('shows source-owned contract tag and status inside the custom pipeline list', async () => {
    render(
      <MemoryRouter>
        <GraphHomePage model={createModel()} />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: 'ETHUSDT' }));
    fireEvent.click(screen.getByRole('button', { name: '加入自定义管道' }));

    await waitFor(() => {
      expect(screen.getByTestId('graph-pipeline-item-queue-ethusdt')).toBeTruthy();
    });
    const pipelineItem = screen.getByTestId('graph-pipeline-item-queue-ethusdt');
    expect(within(pipelineItem).getByText('ETHUSDT')).toBeTruthy();
    expect(within(pipelineItem).getByText('signalRisk.actionQueue')).toBeTruthy();
    expect(within(pipelineItem).getByText('观察')).toBeTruthy();
  });

  it('surfaces current impact-chain counts for the selected node', () => {
    render(
      <MemoryRouter>
        <GraphHomePage model={createModel()} />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: 'ETHUSDT' }));
    expect(screen.getByText(/当前影响链：/)).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: '拆分研究工作区左栏' }));
    expect(screen.getByText(/当前影响链：/)).toBeTruthy();
  });

  it('toggles between full graph and locked impact-chain view', () => {
    render(
      <MemoryRouter>
        <GraphHomePage model={createModel()} />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: 'ETHUSDT' }));
    expect(screen.getByRole('button', { name: '锁定当前链' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '搜索' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '国内商品' })).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: '锁定当前链' }));
    expect(screen.getByRole('button', { name: '返回全图' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: '搜索' })).toBeNull();
    expect(screen.queryByRole('button', { name: '国内商品' })).toBeNull();
    expect(screen.getByRole('button', { name: '动作队列' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '执行与风控' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '交易中枢' })).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: '返回全图' }));
    expect(screen.getByRole('button', { name: '锁定当前链' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '搜索' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '国内商品' })).toBeTruthy();
  });

  it('warns when the current impact chain contains nodes without source-owned detail', () => {
    render(
      <MemoryRouter>
        <GraphHomePage model={createModel()} />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: '国内商品' }));
    const warningBlock = screen.getByTestId('graph-home-missing-source');
    expect(within(warningBlock).getByText('缺失信源节点：1')).toBeTruthy();
    expect(within(warningBlock).getByText('国内商品 / 市场输入域')).toBeTruthy();
  });

  it('offers repair links for nodes missing source-owned detail', () => {
    render(
      <MemoryRouter>
        <GraphHomePage model={createModel()} />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: '国内商品' }));
    const warningBlock = screen.getByTestId('graph-home-missing-source');
    expect(within(warningBlock).getByText('国内商品 / 市场输入域')).toBeTruthy();
    expect(within(warningBlock).getByText('先核对契约层，再搜索节点来源')).toBeTruthy();
    expect(within(warningBlock).getByRole('link', { name: '国内商品 / 查看契约层' }).getAttribute('href')).toBe('/workspace/contracts?page_section=contracts-source-heads');
    expect(within(warningBlock).getByRole('link', { name: '国内商品 / 搜索节点' }).getAttribute('href')).toBe('/search?q=%E5%9B%BD%E5%86%85%E5%95%86%E5%93%81');
  });

  it('builds kind-specific missing-source repair plans', () => {
    const marketNode = { id: 'market-cn', kind: 'market', label: '国内商品', importance: 5, upstream: [], downstream: [] } as GraphNode;
    const routeNode = { id: 'route-search', kind: 'route', label: '搜索', destination: '/search?q=test', importance: 5, upstream: [], downstream: [] } as GraphNode;
    const artifactNode = { id: 'artifact-eth', kind: 'artifact', label: 'ETH 主线', destination: '/workspace/artifacts?artifact=eth', importance: 5, upstream: [], downstream: [] } as GraphNode;

    expect(buildMissingSourceRepairHint(marketNode)).toBe('先核对契约层，再搜索节点来源');
    expect(buildMissingSourceRepairHint(routeNode)).toBe('先进入页面确认来源，再回查检索');
    expect(buildMissingSourceRepairHint(artifactNode)).toBe('先查看工件，再核对检索结果');

    expect(buildMissingSourceRepairLinks(marketNode)).toEqual([
      { label: '国内商品 / 查看契约层', to: '/workspace/contracts?page_section=contracts-source-heads' },
      { label: '国内商品 / 搜索节点', to: '/search?q=%E5%9B%BD%E5%86%85%E5%95%86%E5%93%81' },
    ]);
    expect(buildMissingSourceRepairLinks(routeNode)).toEqual([
      { label: '搜索 / 进入页面', to: '/search?q=test' },
      { label: '搜索 / 搜索节点', to: '/search?q=%E6%90%9C%E7%B4%A2' },
    ]);
    expect(buildMissingSourceRepairLinks(artifactNode)).toEqual([
      { label: 'ETH 主线 / 查看工件', to: '/workspace/artifacts?artifact=eth' },
      { label: 'ETH 主线 / 搜索节点', to: '/search?q=ETH+%E4%B8%BB%E7%BA%BF' },
    ]);
  });

  it('shows a graph legend for current chain warning and dimmed nodes', () => {
    render(
      <MemoryRouter>
        <GraphHomePage model={createModel()} />
      </MemoryRouter>,
    );

    const legend = screen.getByTestId('graph-home-legend');
    expect(within(legend).getByText('当前链高亮')).toBeTruthy();
    expect(within(legend).getByText('缺失信源')).toBeTruthy();
    expect(within(legend).getByText('非当前链降噪')).toBeTruthy();
  });

  it('toggles legend layers interactively', () => {
    render(
      <MemoryRouter>
        <GraphHomePage model={createModel()} />
      </MemoryRouter>,
    );

    const legend = screen.getByTestId('graph-home-legend');
    const focusToggle = within(legend).getByRole('button', { name: '当前链高亮' });
    const warningToggle = within(legend).getByRole('button', { name: '缺失信源' });
    const dimToggle = within(legend).getByRole('button', { name: '非当前链降噪' });

    expect(focusToggle.getAttribute('data-active')).toBe('true');
    expect(warningToggle.getAttribute('data-active')).toBe('true');
    expect(dimToggle.getAttribute('data-active')).toBe('true');

    fireEvent.click(focusToggle);
    fireEvent.click(warningToggle);
    fireEvent.click(dimToggle);

    expect(focusToggle.getAttribute('data-active')).toBe('false');
    expect(warningToggle.getAttribute('data-active')).toBe('false');
    expect(dimToggle.getAttribute('data-active')).toBe('false');
  });

  it('restores default legend layers after manual toggles', () => {
    render(
      <MemoryRouter>
        <GraphHomePage model={createModel()} />
      </MemoryRouter>,
    );

    const legend = screen.getByTestId('graph-home-legend');
    const focusToggle = within(legend).getByRole('button', { name: '当前链高亮' });
    const warningToggle = within(legend).getByRole('button', { name: '缺失信源' });
    const dimToggle = within(legend).getByRole('button', { name: '非当前链降噪' });

    fireEvent.click(focusToggle);
    fireEvent.click(warningToggle);
    fireEvent.click(dimToggle);
    expect(focusToggle.getAttribute('data-active')).toBe('false');
    expect(warningToggle.getAttribute('data-active')).toBe('false');
    expect(dimToggle.getAttribute('data-active')).toBe('false');

    fireEvent.click(screen.getByRole('button', { name: '恢复默认图层' }));
    expect(focusToggle.getAttribute('data-active')).toBe('true');
    expect(warningToggle.getAttribute('data-active')).toBe('true');
    expect(dimToggle.getAttribute('data-active')).toBe('true');
  });
});
