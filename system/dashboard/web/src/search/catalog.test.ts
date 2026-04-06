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
    terminal_public: '/ops/risk',
    workspace_artifacts: '/workspace/artifacts',
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
    recent_strategy_backtests: {
      label: 'recent backtests',
      path: '/tmp/recent.json',
      summary: { status: 'ok', research_decision: 'keep_best' },
      payload: {
        rows: [],
        mode_summaries: [
          {
            mode: 'ultra_short',
            trial_artifacts: [
              {
                trial: 1,
                mode: 'ultra_short',
                score: 0.42,
                trade_journal_path: '/tmp/research/ultra_short/trial_001_ultra_short_trade_journal.csv',
                holding_exposure_path: '/tmp/research/ultra_short/trial_001_ultra_short_holding_daily_symbol_exposure.csv',
              },
            ],
          },
        ],
      },
    },
    strategy_lab_summary: {
      label: 'strategy lab summary',
      path: '/tmp/strategy_lab/summary.json',
      summary: { status: 'ok', research_decision: 'selected' },
      payload: {
        best_candidate: {
          name: 'balanced_flow_04',
          trade_journal_path: '/tmp/strategy_lab/best_trade_journal.csv',
          holding_exposure_path: '/tmp/strategy_lab/best_holding_daily_symbol_exposure.csv',
        },
        candidates: [
          {
            name: 'balanced_flow_04',
            trade_journal_path: '/tmp/strategy_lab/candidate_01_balanced_flow_04_trade_journal.csv',
            holding_exposure_path: '/tmp/strategy_lab/candidate_01_balanced_flow_04_holding_daily_symbol_exposure.csv',
          },
        ],
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
    expect(results.some((row) => row.destination.includes('/ops/overview'))).toBe(true);
    expect(results.some((row) => row.destination.includes('/ops/risk'))).toBe(true);
  });

  it('搜索 Jin10 研究侧边车时返回总览模块与工件结果', () => {
    const enrichedSnapshot: DashboardSnapshot = {
      ...snapshot,
      catalog: [
        ...(snapshot.catalog || []),
        {
          id: 'jin10_mcp_snapshot',
          payload_key: 'jin10_mcp_snapshot',
          artifact_group: 'system_anchor',
          category: 'research',
          label: 'jin10_mcp_snapshot',
          status: 'ok',
          path: 'review/latest_jin10_mcp_snapshot.json',
        },
      ],
      artifact_payloads: {
        ...(snapshot.artifact_payloads || {}),
        jin10_mcp_snapshot: {
          label: 'jin10 mcp snapshot',
          path: 'review/latest_jin10_mcp_snapshot.json',
          summary: {
            status: 'ok',
            recommended_brief: 'calendar=244 | flash=20 | quotes=2',
            takeaway: '美国至4月3日当周EIA原油库存(万桶)',
          },
          payload: {
            summary: {
              high_importance_titles: ['美国至4月3日当周EIA原油库存(万桶)'],
              latest_flash_briefs: ['中国品牌车在曼谷国际车展预订量首次超越日系车'],
              quote_watch: [{ code: 'XAUUSD', name: '现货黄金' }],
            },
          },
        },
      },
    };
    const model = buildTerminalReadModel(buildLoaded(enrichedSnapshot));
    const results = searchCatalog(buildSearchCatalog(model), 'Jin10 研究侧边车', 'all');
    expect(results.some((row) => row.destination.includes('/ops/overview'))).toBe(true);
    expect(results.some((row) => row.destination.includes('/workspace/artifacts'))).toBe(true);
  });

  it('搜索 Axios 研究侧边车时返回总览模块与工件结果', () => {
    const enrichedSnapshot: DashboardSnapshot = {
      ...snapshot,
      catalog: [
        ...(snapshot.catalog || []),
        {
          id: 'axios_site_snapshot',
          payload_key: 'axios_site_snapshot',
          artifact_group: 'system_anchor',
          category: 'research',
          label: 'axios_site_snapshot',
          status: 'ok',
          path: 'review/latest_axios_site_snapshot.json',
        },
      ],
      artifact_payloads: {
        ...(snapshot.artifact_payloads || {}),
        axios_site_snapshot: {
          label: 'axios site snapshot',
          path: 'review/latest_axios_site_snapshot.json',
          summary: {
            status: 'ok',
            recommended_brief: 'axios news=10 | local=8 | national=2',
            takeaway: "Gilbert's Heritage Park to open first phase in September",
          },
          payload: {
            summary: {
              top_titles: [
                "Gilbert's Heritage Park to open first phase in September",
                'Anthropic cuts third party usage',
              ],
              top_keywords: ['Arizona', 'Axios'],
            },
          },
        },
      },
    };
    const model = buildTerminalReadModel(buildLoaded(enrichedSnapshot));
    const results = searchCatalog(buildSearchCatalog(model), 'Axios 研究侧边车', 'all');
    expect(results.some((row) => row.destination.includes('/ops/overview'))).toBe(true);
    expect(results.some((row) => row.destination.includes('/workspace/artifacts'))).toBe(true);
  });

  it('搜索 Polymarket 情绪侧边车时返回总览模块与工件结果', () => {
    const enrichedSnapshot: DashboardSnapshot = {
      ...snapshot,
      catalog: [
        ...(snapshot.catalog || []),
        {
          id: 'polymarket_gamma_snapshot',
          payload_key: 'polymarket_gamma_snapshot',
          artifact_group: 'system_anchor',
          category: 'research',
          label: 'polymarket_gamma_snapshot',
          status: 'ok',
          path: 'review/latest_polymarket_gamma_snapshot.json',
        },
      ],
      artifact_payloads: {
        ...(snapshot.artifact_payloads || {}),
        polymarket_gamma_snapshot: {
          label: 'polymarket gamma snapshot',
          path: 'review/latest_polymarket_gamma_snapshot.json',
          summary: {
            status: 'ok',
            recommended_brief: 'poly markets=12 | yes_avg=56.3% | bull=3 | bear=2',
            takeaway: 'Will BTC hit $120k in April?',
          },
          payload: {
            summary: {
              top_titles: [
                'Will BTC hit $120k in April?',
                'Will the Fed cut before June?',
              ],
              top_categories: ['Crypto', 'Economy'],
            },
          },
        },
      },
    };
    const model = buildTerminalReadModel(buildLoaded(enrichedSnapshot));
    const results = searchCatalog(buildSearchCatalog(model), 'Polymarket 情绪侧边车', 'all');
    expect(results.some((row) => row.destination.includes('/ops/overview'))).toBe(true);
    expect(results.some((row) => row.destination.includes('/workspace/artifacts'))).toBe(true);
  });

  it('搜索外部情报带时返回总览模块与工件结果', () => {
    const enrichedSnapshot: DashboardSnapshot = {
      ...snapshot,
      catalog: [
        ...(snapshot.catalog || []),
        {
          id: 'external_intelligence_snapshot',
          payload_key: 'external_intelligence_snapshot',
          artifact_group: 'system_anchor',
          category: 'research',
          label: 'external_intelligence_snapshot',
          status: 'ok',
          path: 'review/latest_external_intelligence_snapshot.json',
        },
      ],
      artifact_payloads: {
        ...(snapshot.artifact_payloads || {}),
        external_intelligence_snapshot: {
          label: 'external intelligence snapshot',
          path: 'review/latest_external_intelligence_snapshot.json',
          summary: {
            status: 'ok',
            recommended_brief: 'sources=2 | calendar=244 | flash=20 | quotes=2 | news=10',
            takeaway: "美国至4月3日当周EIA原油库存(万桶) ｜ Gilbert's Heritage Park to open first phase in September",
          },
          payload: {
            summary: {
              top_titles: [
                "Gilbert's Heritage Park to open first phase in September",
                'Anthropic cuts third party usage',
              ],
              top_keywords: ['Arizona', 'Axios'],
              quote_watch: ['现货黄金', 'WTI原油'],
            },
          },
        },
      },
    };
    const model = buildTerminalReadModel(buildLoaded(enrichedSnapshot));
    const results = searchCatalog(buildSearchCatalog(model), '外部情报带', 'all');
    expect(results.some((row) => row.destination.includes('/ops/overview'))).toBe(true);
    expect(results.some((row) => row.destination.includes('/workspace/artifacts'))).toBe(true);
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

  it('搜索 trial/candidate 审计工件时命中派生 artifact 结果', () => {
    const model = buildTerminalReadModel(buildLoaded(snapshot));
    const trial = searchCatalog(buildSearchCatalog(model), 'trial_001_ultra_short_trade_journal', 'artifact');
    const candidate = searchCatalog(buildSearchCatalog(model), 'balanced_flow_04', 'artifact');
    expect(trial.some((row) => row.id.includes('trial_001'))).toBe(true);
    expect(candidate.some((row) => row.id.includes('balanced_flow_04'))).toBe(true);
  });
});
