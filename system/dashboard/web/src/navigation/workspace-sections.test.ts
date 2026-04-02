import { describe, expect, it } from 'vitest';
import type { TerminalReadModel } from '../types/contracts';
import { getDefaultWorkspacePageSection, getWorkspacePageSections } from './workspace-sections';

function createModel(): TerminalReadModel {
  return {
    surface: {} as TerminalReadModel['surface'],
    meta: {},
    uiRoutes: {},
    experienceContract: {},
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
      domainGroups: [
        { domain: 'macro', count: 2, topHeadline: 'macro' },
      ],
    },
    workspace: {
      domains: [],
      interfaces: [],
      publicTopology: [],
      publicAcceptance: {
        summary: null,
        checks: [
          { id: 'graph_home_smoke', label: '图谱主页烟测' },
          { id: 'workspace_routes_smoke', label: '工作区五页面烟测' },
        ],
        subcommands: [
          { id: 'graph_home_smoke', label: '图谱主页子命令' },
          { id: 'workspace_routes_smoke', label: '工作区路由子命令' },
        ],
      },
      researchAuditCases: [],
      surfaceContracts: [],
      fallbackChain: [],
      sourceHeads: [
        { id: 'system_scheduler_state', label: '调度与心跳', path: '/tmp/scheduler.json' },
        { id: 'operator_panel', label: '操作面板', path: '/tmp/operator_panel.json' },
      ],
      defaultFocus: undefined,
      artifactRows: [
        { id: 'price_action_breakout_pullback', label: 'ETH 主线' },
      ],
      artifactPrimaryFocus: {},
      artifactHandoffs: {},
      artifactPayloads: {},
      raw: {},
    },
  };
}

describe('getDefaultWorkspacePageSection', () => {
  it('returns explicit default section ids instead of depending on list order', () => {
    const model = createModel();

    expect(getDefaultWorkspacePageSection('alignment', model)?.id).toBe('alignment-summary');
    expect(getDefaultWorkspacePageSection('backtests', model)?.id).toBe('backtests-overview');
    expect(getDefaultWorkspacePageSection('contracts', model)?.id).toBe('contracts-topology');
    expect(getDefaultWorkspacePageSection('raw', model)?.id).toBe('raw-summary');
  });

  it('falls back to the first section only when the explicit default is missing', () => {
    const model = createModel();
    model.workspace.artifactRows = [];

    expect(getDefaultWorkspacePageSection('raw', model)?.id).toBe('raw-summary');

    model.feedback.domainGroups = [];
    expect(getDefaultWorkspacePageSection('alignment', model)?.id).toBe('alignment-summary');
  });

  it('keeps operator_panel in contracts source-head sections even when it appears after the first three rows', () => {
    const model = createModel();
    model.workspace.sourceHeads = [
      { id: 'system_scheduler_state', label: '调度与心跳', path: '/tmp/scheduler.json' },
      { id: 'price_action_breakout_pullback', label: 'ETH 主线', path: '/tmp/eth.json' },
      { id: 'hold_selection_handoff', label: '持有选择主头', path: '/tmp/hold.json' },
      { id: 'operator_panel', label: '操作面板', path: '/tmp/operator_panel.json' },
    ];

    const sections = getWorkspacePageSections('contracts', model);
    expect(sections.some((item) => item.id === 'contracts-source-head-operator_panel')).toBe(true);
  });

  it('keeps raw focus label source-owned instead of depending on the first artifact row order', () => {
    const model = createModel();
    model.workspace.defaultFocus = {
      artifact: 'price_action_breakout_pullback',
      group: 'research_mainline',
      search_scope: 'title',
    };
    model.workspace.artifactRows = [
      { id: 'operator_panel', payload_key: 'operator_panel', label: 'operator panel row' },
      { id: 'price_action_breakout_pullback', payload_key: 'price_action_breakout_pullback', label: 'ETH 主线' },
    ];

    const sections = getWorkspacePageSections('raw', model);
    expect(sections.find((item) => item.id === 'raw-focus')?.label).toBe('price_action_breakout_pullback');
  });
});
