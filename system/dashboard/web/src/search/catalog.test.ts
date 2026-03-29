import { describe, expect, it } from 'vitest';
import { buildTerminalReadModel } from '../adapters/read-model';
import type { DashboardSnapshot, LoadedSurface } from '../types/contracts';
import { buildSearchCatalog, searchCatalog } from './catalog';

function buildLoaded(snapshot: DashboardSnapshot): LoadedSurface {
  return {
    snapshot,
    requestedSurface: 'public',
    effectiveSurface: 'public',
    snapshotPath: '/data/fenlie_dashboard_snapshot.json',
    warnings: [],
  };
}

const snapshot: DashboardSnapshot = {
  status: 'ok',
  change_class: 'RESEARCH_ONLY',
  meta: { change_class: 'RESEARCH_ONLY' },
  ui_routes: {
    terminal_public: '#/terminal/public',
    workspace_artifacts: '#/workspace/artifacts',
  },
  experience_contract: { mode: 'read_only_snapshot', live_path_touched: false, fallback_chain: [] },
  catalog: [
    {
      id: 'hold_selection_handoff',
      payload_key: 'hold_selection_handoff',
      artifact_group: 'research_hold_transfer',
      category: 'research',
      label: 'hold_selection_handoff',
      status: 'ok',
      path: 'review/latest_hold_selection.json',
    },
  ],
  interface_catalog: [],
  public_topology: [],
  domain_summary: [],
  source_heads: {
    hold_selection_handoff: {
      label: 'hold selection handoff',
      status: 'ok',
      summary: 'hold16 baseline',
      path: 'review/latest_hold_selection.json',
    },
  },
  surface_contracts: {},
  backtest_artifacts: [],
  artifact_payloads: {
    commodity_reasoning_summary: {
      label: 'commodity reasoning summary',
      path: '/tmp/commodity-summary.json',
      summary: { status: 'ok' },
      payload: {
        primary_scenario_brief: 'policy_relief_watch',
        primary_chain_brief: 'policy_relief_chain',
        contracts_in_focus: ['BU2606'],
      },
    },
  },
  workspace_default_focus: {
    artifact: 'hold_selection_handoff',
    group: 'research_hold_transfer',
    search_scope: 'title',
  },
};

describe('search catalog', () => {
  it('搜索国内商品推理线时返回总览与终端模块', () => {
    const model = buildTerminalReadModel(buildLoaded(snapshot));
    const results = searchCatalog(buildSearchCatalog(model), '国内商品推理线', 'all');
    expect(results.some((row) => row.destination.includes('/overview'))).toBe(true);
    expect(results.some((row) => row.destination.includes('/terminal/public'))).toBe(true);
  });

  it('搜索契约层时命中 workspace contracts 路由', () => {
    const model = buildTerminalReadModel(buildLoaded(snapshot));
    const results = searchCatalog(buildSearchCatalog(model), '契约层', 'all');
    expect(results.some((row) => row.destination.includes('/workspace/contracts'))).toBe(true);
  });

  it('搜索工件 id 时命中 artifact 结果', () => {
    const model = buildTerminalReadModel(buildLoaded(snapshot));
    const results = searchCatalog(buildSearchCatalog(model), 'hold_selection_handoff', 'artifact');
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].category).toBe('artifact');
    expect(results[0].destination).toContain('/workspace/artifacts');
  });
});
