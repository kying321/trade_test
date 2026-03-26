import { describe, expect, it } from 'vitest';
import type { LoadedSurface } from '../types/contracts';
import { buildTerminalReadModel } from './read-model';

function findLinkByPath(links: Array<{ to: string }> | undefined, pathname: string) {
  return (links || []).find((link) => new URL(link.to, 'https://fenlie.local').pathname === pathname);
}

function expectLinkToMatch(
  link: { to: string } | undefined,
  pathname: string,
  expectedQuery: Record<string, string>,
) {
  expect(link).toBeTruthy();
  const url = new URL(link?.to || '', 'https://fenlie.local');
  expect(url.pathname).toBe(pathname);
  Object.entries(expectedQuery).forEach(([key, value]) => {
    expect(url.searchParams.get(key)).toBe(value);
  });
}

const loaded: LoadedSurface = {
  requestedSurface: 'internal',
  effectiveSurface: 'internal',
  snapshotPath: '/data/fenlie_dashboard_internal_snapshot.json',
  warnings: [],
  snapshot: {
    change_class: 'RESEARCH_ONLY',
    meta: {
      generated_at_utc: '2026-03-20T00:00:00Z',
      change_class: 'RESEARCH_ONLY',
    },
    experience_contract: {
      mode: 'research_only',
      live_path_touched: false,
    },
    surface_contracts: {
      internal: {},
      public: {},
    },
    artifact_payloads: {
      operator_panel: {
        path: '/tmp/operator_panel.json',
        payload: {
          generated_at_utc: '2026-03-20T00:05:00Z',
          summary: {
            operator_head_brief: 'review_queue_now',
            remote_live_gate_brief: 'blocked_by_remote_live_gate',
            remote_live_clearing_brief: 'clearing_required',
            lane_state_brief: 'review=2',
            lane_priority_order_brief: 'review@99:2',
            openclaw_guardian_clearance_status: 'blocked',
            openclaw_guardian_clearance_score: 0.62,
            openclaw_canary_gate_status: 'blocked',
            openclaw_canary_gate_decision: 'deny_canary_until_guardian_clear',
            openclaw_promotion_gate_status: 'waiting',
            openclaw_promotion_gate_decision: 'clear_guardian_review_blockers_before_promotion',
            openclaw_live_boundary_hold_status: 'active',
            openclaw_live_boundary_hold_decision: 'hold_remote_idle_until_ticket_ready',
            openclaw_orderflow_policy_status: 'watch',
            openclaw_orderflow_policy_decision: 'selected_watch_only_orderflow_slice',
            openclaw_execution_ack_status: 'waiting',
            openclaw_execution_ack_decision: 'shadow_no_send_ack_recorded',
            openclaw_quality_report_score: 0.71,
            openclaw_quality_shadow_learning_score: 0.82,
            openclaw_quality_execution_readiness_score: 0.31,
            priority_repair_plan_brief: 'repair_queue_now',
            priority_repair_plan_artifact: '/tmp/priority_repair_plan.json',
            priority_repair_verification_brief: 'review_queue_next',
            priority_repair_verification_artifact: '/tmp/priority_repair_verify.json',
            remote_live_diagnosis_brief: 'current_head_outside_remote_live_scope',
            remote_live_history_brief: 'profitability_confirmed_but_auto_live_blocked',
          },
          action_checklist: [
            {
              symbol: 'ETHUSDT',
              action: 'review_manual_stop_entry',
              state: 'pending',
              reason: 'mode_drift',
              done_when: 'clear_ops_live_gate_condition',
              source_as_of: '2026-03-20T00:06:00Z',
            },
          ],
          lane_cards: [],
          focus_slots: [],
          control_chain: [],
          action_queue: [],
          continuous_optimization_backlog: [],
          priority_repair_plan: {
            status: 'pending',
            done_when: 'clear_ops_live_gate_condition',
          },
        },
      },
      system_scheduler_state: {
        path: '/tmp/system_scheduler_state.json',
        payload: {
          interval_tasks: {
            micro_capture: {
              status: 'ok',
              last_run_ts: '2026-03-20T00:10:00Z',
              enabled: true,
              interval_minutes: 30,
              symbols: ['ETHUSDT', 'SOLUSDT'],
            },
          },
          history: [
            {
              ts: '2026-03-20T00:10:00Z',
              slot: 'micro_capture',
              trigger_time: '2026-03-20T00:10:00Z',
              status: 'ok',
              result: {
                captured_at_utc: '2026-03-20T00:10:30Z',
                summary: {
                  selected_schema_ok_ratio: 0.95,
                  selected_time_sync_ok_ratio: 1,
                  selected_sync_ok_ratio: 0.9,
                  cross_source_status: 'pass',
                  cross_source_fail_ratio: 0.08,
                  cross_source_fail_ratio_limit: 0.2,
                  cross_source_symbols_audited: 12,
                  cross_source_symbols_insufficient: 1,
                },
                rolling_7d: {
                  run_count: 14,
                  pass_ratio: 0.93,
                  avg_cross_source_fail_ratio: 0.07,
                },
              },
            },
          ],
        },
      },
      architecture_audit: {
        path: '/tmp/architecture_audit.json',
        payload: {
          status: 'pass',
          health: {
            status: 'healthy',
            checks: {
              daily_briefing: 'pass',
              daily_signals: 'pass',
              daily_positions: 'warning',
              sqlite: 'pass',
            },
          },
          manifests: {
            eod_manifest: 'ok',
            backtest_manifest: 'ok',
            review_manifest: 'ok',
          },
        },
        summary: {
          generated_at_utc: '2026-03-20T00:11:00Z',
        },
      },
      mode_stress_matrix: {
        payload: {
          mode_summary: [],
          matrix: [],
          best_mode: 'balanced',
        },
      },
      recent_strategy_backtests: {
        payload: {
          rows: [],
        },
      },
      price_action_breakout_pullback: {
        path: '/tmp/price_action_breakout_pullback.json',
        summary: {
          status: 'positive',
          research_decision: 'keep_best',
          change_class: 'SIM_ONLY',
          path: '/tmp/price_action_breakout_pullback.json',
          takeaway: 'forward_compare_8_vs_16',
        },
      },
      price_action_exit_risk: {
        summary: {
          status: 'mixed',
          research_decision: 'forward_compare_8_vs_16',
          change_class: 'SIM_ONLY',
          path: '/tmp/price_action_exit_risk.json',
        },
      },
      cross_section_backtest: {
        summary: {
          status: 'negative',
          research_decision: 'no_edge',
          change_class: 'SIM_ONLY',
          path: '/tmp/cross_section_backtest.json',
        },
      },
    },
    source_heads: {
      price_action_breakout_pullback: {
        status: 'positive',
        summary: 'keep_best',
        path: '/tmp/price_action_breakout_pullback.json',
      },
    },
    backtest_artifacts: [],
    catalog: [],
    domain_summary: [],
    interface_catalog: [],
    ui_routes: {},
    conversation_feedback_projection: {
      status: 'ok',
      change_class: 'RESEARCH_ONLY',
      visibility: 'internal_only',
      summary: {
        alignment_score: 74,
        blocker_pressure: 22,
        execution_clarity: 68,
        readability_pressure: 31,
        drift_state: 'watch',
        headline: '保持前端信息架构与用户意图对齐',
      },
      events: [
        {
          feedback_id: 'ui_density_split',
          created_at_utc: '2026-03-20T00:20:00Z',
          source: 'manual',
          domain: 'research',
          headline: '拆分左栏四控件',
          summary: '研究地图 / 状态雷达 / 检索定位 / 当前焦点',
          recommended_action: '完成左栏职责拆分',
          impact_score: 91,
          confidence: 0.89,
          status: 'active',
          anchors: [
            {
              route: '/workspace/artifacts',
              artifact: 'price_action_breakout_pullback',
              component: 'WorkspacePanels',
            },
          ],
        },
      ],
      actions: [
        {
          feedback_id: 'ui_density_split',
          recommended_action: '完成左栏职责拆分',
          impact_score: 91,
          anchors: [
            {
              route: '/workspace/artifacts',
              artifact: 'price_action_breakout_pullback',
              component: 'WorkspacePanels',
            },
          ],
        },
      ],
      trends: [
        {
          feedback_id: 'ui_density_split',
          created_at_utc: '2026-03-20T00:20:00Z',
          alignment_score: 74,
          blocker_pressure: 22,
          execution_clarity: 68,
          readability_pressure: 31,
          drift_state: 'watch',
        },
      ],
      anchors: [
        {
          route: '/workspace/artifacts',
          artifact: 'price_action_breakout_pullback',
          component: 'WorkspacePanels',
        },
      ],
    },
  },
};

describe('buildTerminalReadModel', () => {
  it('uses shared contract labels in adapter-built metrics and details', () => {
    const model = buildTerminalReadModel(loaded);

    expect(model.orchestration.topMetrics[0]).toMatchObject({
      label: '数据面',
      hint: '请求=内部终端',
    });
    expect(model.orchestration.freshness[0].label).toBe('快照生成');
    expect(model.orchestration.guards[0].label).toBe('远端实盘门禁');
    expect(model.orchestration.checklist[0].detail).toBe('原因=模式漂移 ｜ 完成条件=清除运维实盘门禁条件');
    expect(model.labReview.manifests[0].label).toBe('日终清单');
    expect(model.labReview.manifests[4].hint).toBe('需单独 temporal artifact 补齐');
    expect(model.view.terminal.orchestration.topMetricsStrip.showRawValues).toBe(true);
    expect(model.view.terminal.signalRisk.repairPlanStrip.showRawValues).toBe(true);
    expect(model.view.terminal.signalRisk.focusSlotsTableColumns.find((column) => column.key === 'action')?.showRaw).toBe(true);
    expect(model.view.terminal.orchestration.actionLogColumns[2]).toMatchObject({
      key: 'status',
      kind: 'badge',
      label: '状态',
    });
    expect(model.view.terminal.signalRisk.focusSlotsDrilldown.fields[1]).toMatchObject({
      key: 'action',
      showRaw: true,
      label: '动作',
    });
    expect(model.view.terminal.signalRisk.focusSlotsDrilldown.fields.find((field) => field.key === 'source_artifact')?.showRaw).toBe(true);
    expect(model.view.terminal.labReview.candidateColumns.find((column) => column.key === 'decision')?.showRaw).toBe(true);
    expect(model.view.workspace.backtests.poolColumns[1]).toMatchObject({
      key: 'equity_curve_sample',
      kind: 'sparkline',
      label: '曲线',
    });
    expect(model.view.workspace.contracts.interfaceColumns[1]).toMatchObject({
      key: 'url',
      kind: 'path',
      label: '入口',
    });
    expect(model.view.workspace.artifacts.list.titleField.showRaw).toBe(true);
    expect(model.view.workspace.artifacts.inspector.pathField.showRaw).toBe(true);
    expect(model.view.workspace.artifacts.topLevelKeyField.showRaw).toBe(true);
    expect(model.view.workspace.artifacts.payloadField.showRaw).toBe(true);
    expect(model.view.workspace.contracts.sourceHeadSummaryFields.label.showRaw).toBe(true);
    expect(model.view.workspace.contracts.sourceHeadDetailFields.find((field) => field.key === 'path')?.showRaw).toBe(true);
  });

  it('maps event crisis artifacts into terminal data-regime and signal-risk metrics', () => {
    const eventSnapshot = {
      ...loaded.snapshot,
      artifact_payloads: {
        ...loaded.snapshot.artifact_payloads,
        event_regime_snapshot: {
          payload: {
            event_severity_score: 0.82,
            systemic_risk_score: 0.61,
            regime_state: 'sector_stress',
            headline_drivers: ['credit_liquidity_stress'],
          },
        },
        event_crisis_analogy: {
          payload: {
            top_analogues: [
              {
                archetype_id: 'gfc_2008',
                similarity_score: 0.77,
                match_axes: ['credit', 'contagion'],
                mismatch_axes: ['policy'],
              },
            ],
          },
        },
        event_asset_shock_map: {
          payload: {
            assets: [
              {
                asset: 'BTC',
                class: 'crypto',
                shock_direction_bias: 'down',
              },
            ],
          },
        },
        event_crisis_operator_summary: {
          payload: {
            status: 'watch',
            summary: 'event crisis watch',
            takeaway: 'monitor private credit flow',
          },
        },
      },
    };
    const model = buildTerminalReadModel({
      ...loaded,
      snapshot: eventSnapshot,
    });

    expect(model.dataRegime.microCapture.some((metric) => metric.id === 'event-severity')).toBe(true);
    expect(model.dataRegime.microCapture.some((metric) => metric.id === 'event-regime-state')).toBe(true);
    expect(model.dataRegime.microCapture.some((metric) => metric.id === 'event-analogy')).toBe(true);
    expect(model.dataRegime.microCapture.some((metric) => metric.id === 'event-shock-map')).toBe(true);
    expect(model.signalRisk.repairPlan.some((metric) => metric.id === 'event-crisis-summary')).toBe(true);
  });

  it('builds a dedicated feedback slice without polluting artifact rows', () => {
    const model = buildTerminalReadModel(loaded);

    expect(model.feedback.projection?.summary.headline).toBe('保持前端信息架构与用户意图对齐');
    expect(model.feedback.projection?.events[0]?.feedback_id).toBe('ui_density_split');
    expect(model.feedback.domainGroups[0]).toMatchObject({
      domain: 'research',
      count: 1,
    });
    expect(model.workspace.artifactRows.find((row) => row.id === 'ui_density_split')).toBeUndefined();
  });

  it('resolves public surface view schema to hidden-by-default raw policies', () => {
    const model = buildTerminalReadModel({
      ...loaded,
      requestedSurface: 'public',
      effectiveSurface: 'public',
    });

    expect(model.view.terminal.orchestration.topMetricsStrip.showRawValues).toBe(false);
    expect(model.view.terminal.signalRisk.repairPlanStrip.showRawValues).toBe(false);
    expect(model.view.terminal.signalRisk.focusSlotsTableColumns.find((column) => column.key === 'action')?.showRaw).toBe(false);
    expect(model.view.terminal.signalRisk.focusSlotsDrilldown.fields[1]?.showRaw).toBe(false);
    expect(model.view.terminal.signalRisk.focusSlotsDrilldown.fields.find((field) => field.key === 'source_artifact')?.showRaw).toBe(false);
    expect(model.view.terminal.labReview.candidateColumns.find((column) => column.key === 'decision')?.showRaw).toBe(false);
    expect(model.view.workspace.backtests.comparisonColumns[0]?.showRaw).toBe(false);
    expect(model.view.workspace.artifacts.list.titleField.showRaw).toBe(false);
    expect(model.view.workspace.artifacts.topLevelKeyField.showRaw).toBe(false);
    expect(model.view.workspace.artifacts.payloadField.showRaw).toBe(false);
    expect(model.view.workspace.contracts.sourceHeadSummaryFields.label.showRaw).toBe(false);
    expect(model.view.workspace.contracts.sourceHeadDetailFields.find((field) => field.key === 'path')?.showRaw).toBe(false);
  });

  it('promotes fallback, freshness drift, and live-path markers into top-level alerts', () => {
    const artifactPayloads = loaded.snapshot.artifact_payloads!;
    const operatorPanel = artifactPayloads.operator_panel as { payload?: Record<string, unknown> };
    const operatorPayload = operatorPanel.payload as Record<string, unknown>;
    const operatorSummary = operatorPayload.summary as Record<string, unknown>;
    const schedulerState = artifactPayloads.system_scheduler_state as { payload?: Record<string, unknown> };
    const schedulerPayload = schedulerState.payload as Record<string, unknown>;
    const model = buildTerminalReadModel({
      ...loaded,
      requestedSurface: 'internal',
      effectiveSurface: 'public',
      warnings: ['内部终端仅在本地/私有环境保留，已回退到公开快照'],
      snapshot: {
        ...loaded.snapshot,
        meta: {
          ...loaded.snapshot.meta,
          generated_at_utc: '2020-01-01T00:00:00Z',
        },
        experience_contract: {
          ...loaded.snapshot.experience_contract,
          live_path_touched: true,
        },
        artifact_payloads: {
          ...artifactPayloads,
          operator_panel: {
            ...operatorPanel,
            payload: {
              ...operatorPayload,
              summary: {
                ...operatorSummary,
                openclaw_guardian_clearance_top_blocker_title: 'ticket_actionability',
                openclaw_guardian_clearance_top_blocker_next_action: 'refresh_signal_source',
                openclaw_guardian_clearance_top_blocker_target_artifact: 'crypto_shortline_pattern_router',
              },
              control_chain: [
                {
                  label: '风控',
                  source_status: 'blocked',
                  source_decision: 'hold',
                  next_target_artifact: 'crypto_shortline_pattern_router',
                  blocking_reason: 'guardian_ticket_actionability',
                },
              ],
            },
          },
          system_scheduler_state: {
            ...schedulerState,
            payload: {
              ...schedulerPayload,
              history: [
                {
                  ts: '2026-03-20T00:10:00Z',
                  slot: 'micro_capture',
                  trigger_time: '2026-03-20T00:10:00Z',
                  status: 'failed',
                  result: {
                    summary: 'cross_source_fail_ratio_breach',
                  },
                },
              ],
            },
          },
        },
      },
    });

    expect(model.orchestration.alerts.map((alert) => alert.id)).toEqual([
      'live-path-touched',
      'guardian-blocker',
      'control-chain-blocker',
      'latest-failed-action',
      'freshness-stale',
      'surface-fallback',
    ]);
    expect(model.orchestration.alerts[0]).toMatchObject({
      tone: 'negative',
      title: '检测到 live path 标记',
    });
    expect(model.orchestration.alerts[1]).toMatchObject({
      id: 'guardian-blocker',
      title: '守门阻断待清障',
    });
    expect(model.orchestration.alerts[1]?.summary).toContain('票据可执行性');
    expect(model.orchestration.alerts[1]?.links?.[0]).toMatchObject({
      label: '查看工件',
      to: '/workspace/artifacts?artifact=crypto_shortline_pattern_router',
    });
    expect(model.orchestration.alerts[1]?.links?.some((link) => link.label === '查看终端层')).toBe(true);
    expect(model.orchestration.alerts[1]?.highlights?.[0]).toMatchObject({
      label: '动作',
    });
    expect(model.orchestration.alerts[1]?.sections?.[0]).toMatchObject({
      title: '处置路径',
    });
    expect(model.orchestration.alerts[2]).toMatchObject({
      id: 'control-chain-blocker',
      title: '传导链存在阻塞',
    });
    expect(model.orchestration.alerts[2]?.links?.find((link) => link.label === '查看终端层')?.to).toBe('/terminal/public?panel=signal-risk&section=control-chain');
    expect(model.orchestration.alerts[2]?.sections?.[1]?.rows.find((row) => row.key === 'source_artifact')?.label).toBe('来源工件');
    expect(model.orchestration.alerts[3]).toMatchObject({
      id: 'latest-failed-action',
      title: '最新失败日志需要处理',
    });
    expect(model.orchestration.alerts[3]?.highlights?.[2]).toMatchObject({
      label: '来源工件',
    });
    expect(model.orchestration.alerts[5]).toMatchObject({
      tone: 'warning',
      title: '数据面已发生回退',
      summary: '内部终端 → 公开终端',
    });
    expect(model.orchestration.alerts[5]?.links?.[0]?.to).toBe('/workspace/contracts');
    expect(model.orchestration.alerts[5]?.links?.find((link) => link.label === '查看终端层')?.to).toBe('/terminal/public');
    expect(model.orchestration.alerts[4]?.title).toContain('过期源');
  });

  it('builds source-owned workspace handoffs back into terminal focus routes', () => {
    const model = buildTerminalReadModel({
      ...loaded,
      snapshot: {
        ...loaded.snapshot,
        catalog: [
          { id: 'price_action_breakout_pullback', payload_key: 'price_action_breakout_pullback', category: 'sim-only', label: 'price_action_breakout_pullback', status: 'ok' },
          { id: 'crypto_shortline_pattern_router', payload_key: 'crypto_shortline_pattern_router', category: 'review', label: 'crypto_shortline_pattern_router', status: 'warning' },
        ],
        artifact_payloads: {
          ...loaded.snapshot.artifact_payloads,
          crypto_shortline_pattern_router: {
            path: '/tmp/crypto_shortline_pattern_router.json',
            payload: {},
            summary: { status: 'warning' },
          },
          operator_panel: {
            ...(loaded.snapshot.artifact_payloads?.operator_panel || {}),
            payload: {
              ...((loaded.snapshot.artifact_payloads?.operator_panel as { payload?: Record<string, unknown> })?.payload || {}),
              control_chain: [
                {
                  stage: 'risk',
                  label: '风控',
                  source_status: 'blocked',
                  source_decision: 'hold',
                  next_target_artifact: 'crypto_shortline_pattern_router',
                },
              ],
            },
          },
        },
      },
    });

    expectLinkToMatch(
      findLinkByPath(model.workspace.artifactHandoffs.price_action_breakout_pullback, '/terminal/internal'),
      '/terminal/internal',
      {
        artifact: 'price_action_breakout_pullback',
        panel: 'lab-review',
        section: 'research-heads',
      },
    );
    expectLinkToMatch(
      findLinkByPath(model.workspace.artifactHandoffs.crypto_shortline_pattern_router, '/terminal/internal'),
      '/terminal/internal',
      {
        artifact: 'crypto_shortline_pattern_router',
        panel: 'signal-risk',
        section: 'control-chain',
        row: 'risk',
      },
    );
  });

  it('builds terminal handoffs for orderflow research artifacts in research cross-section', () => {
    const model = buildTerminalReadModel({
      ...loaded,
      snapshot: {
        ...loaded.snapshot,
        catalog: [
          {
            id: 'intraday_orderflow_blueprint',
            payload_key: 'intraday_orderflow_blueprint',
            artifact_group: 'research_cross_section',
            category: 'research',
            label: 'intraday_orderflow_blueprint',
            status: 'ok',
          },
          {
            id: 'intraday_orderflow_research_gate_blocker',
            payload_key: 'intraday_orderflow_research_gate_blocker',
            artifact_group: 'research_cross_section',
            category: 'research',
            label: 'intraday_orderflow_research_gate_blocker',
            status: 'warning',
          },
        ],
        artifact_payloads: {
          ...loaded.snapshot.artifact_payloads,
          intraday_orderflow_blueprint: {
            path: '/tmp/latest_intraday_orderflow_blueprint.json',
            payload: {},
            summary: { status: 'ok' },
          },
          intraday_orderflow_research_gate_blocker: {
            path: '/tmp/latest_intraday_orderflow_research_gate_blocker_report.json',
            payload: {},
            summary: { status: 'warning' },
          },
        },
      },
    });

    expectLinkToMatch(
      findLinkByPath(model.workspace.artifactHandoffs.intraday_orderflow_blueprint, '/terminal/internal'),
      '/terminal/internal',
      {
        artifact: 'intraday_orderflow_blueprint',
        panel: 'lab-review',
        section: 'research-heads',
      },
    );
    expectLinkToMatch(
      findLinkByPath(model.workspace.artifactHandoffs.intraday_orderflow_research_gate_blocker, '/terminal/internal'),
      '/terminal/internal',
      {
        artifact: 'intraday_orderflow_research_gate_blocker',
        panel: 'lab-review',
        section: 'research-heads',
      },
    );
  });

  it('builds terminal focus handoffs back into workspace artifact/raw routes', () => {
    const model = buildTerminalReadModel({
      ...loaded,
      snapshot: {
        ...loaded.snapshot,
        artifact_payloads: {
          ...loaded.snapshot.artifact_payloads,
          operator_panel: {
            ...(loaded.snapshot.artifact_payloads?.operator_panel || {}),
            payload: {
              ...((loaded.snapshot.artifact_payloads?.operator_panel as { payload?: Record<string, unknown> })?.payload || {}),
              control_chain: [
                {
                  stage: 'risk',
                  label: '风控',
                  source_status: 'blocked',
                  source_decision: 'hold',
                  source_artifact: '/tmp/crypto_shortline_pattern_router.json',
                  next_target_artifact: 'crypto_shortline_pattern_router',
                },
              ],
            },
            summary: { status: 'warning' },
          },
          crypto_shortline_pattern_router: {
            path: '/tmp/crypto_shortline_pattern_router.json',
            payload: {},
            summary: { status: 'warning' },
          },
        },
      },
    });

    const handoffs = model.navigation.terminalHandoffs['signal-risk|control-chain|risk'];
    expectLinkToMatch(
      findLinkByPath(handoffs, '/workspace/artifacts'),
      '/workspace/artifacts',
      {
        artifact: 'crypto_shortline_pattern_router',
        panel: 'signal-risk',
        section: 'control-chain',
        row: 'risk',
      },
    );
    expectLinkToMatch(
      findLinkByPath(handoffs, '/workspace/raw'),
      '/workspace/raw',
      {
        artifact: '/tmp/crypto_shortline_pattern_router.json',
        panel: 'signal-risk',
        section: 'control-chain',
        row: 'risk',
      },
    );
  });

  it('maps public acceptance into workspace contracts and keeps raw evidence internal-only', () => {
    const acceptanceLoaded: LoadedSurface = {
      ...loaded,
      snapshot: {
        ...loaded.snapshot,
        artifact_payloads: {
          ...loaded.snapshot.artifact_payloads,
          dashboard_public_acceptance: {
            path: '/tmp/dashboard_public_acceptance.json',
            summary: {
              status: 'ok',
              change_class: 'RESEARCH_ONLY',
              generated_at_utc: '2026-03-20T07:27:34Z',
            },
            payload: {
              status: 'ok',
              change_class: 'RESEARCH_ONLY',
              generated_at_utc: '2026-03-20T07:27:34Z',
              report_path: '/tmp/dashboard_public_acceptance.json',
              checks: {
                topology_smoke: {
                  status: 'ok',
                  generated_at_utc: '2026-03-20T07:27:29Z',
                  report_path: '/tmp/dashboard_public_topology_smoke.json',
                  returncode: 0,
                  tls_policy: {
                    allow_insecure_tls_fallback: true,
                  },
                  entrypoints: {
                    root_url: 'https://fuuu.fun',
                    pages_url: 'https://fenlie.fuuu.fun',
                  },
                  checks: {
                    root_public: {
                      header_public_entry: 'root-nav-proxy',
                    },
                    root_snapshot: {
                      frontend_public: 'https://fuuu.fun',
                    },
                    root_overview_browser: {
                      screenshot_path: '/tmp/root_overview_browser.png',
                    },
                    pages_overview_browser: {
                      screenshot_path: '/tmp/pages_overview_browser.png',
                    },
                    root_contracts_browser: {
                      screenshot_path: '/tmp/root_contracts_browser.png',
                    },
                    pages_contracts_browser: {
                      screenshot_path: '/tmp/pages_contracts_browser.png',
                    },
                  },
                },
                workspace_routes_smoke: {
                  status: 'ok',
                  generated_at_utc: '2026-03-20T07:27:34Z',
                  report_path: '/tmp/dashboard_workspace_routes_browser_smoke.json',
                  returncode: 0,
                  surface_assertion: {
                    requested_surface: 'public',
                    effective_surface: 'public',
                    snapshot_endpoint_observed: '/data/fenlie_dashboard_snapshot.json',
                  },
                  network_observation: {
                    public_snapshot_fetch_count: 4,
                    internal_snapshot_fetch_count: 0,
                  },
                  artifacts_filter_assertion: {
                    route: '#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow',
                    group: 'research_cross_section',
                    search_scope: 'title',
                    search: 'orderflow',
                    active_artifact: 'intraday_orderflow_blueprint',
                    visible_artifacts: [
                      'intraday_orderflow_blueprint',
                      'intraday_orderflow_research_gate_blocker',
                    ],
                  },
                },
              },
              subcommands: {
                topology_smoke: {
                  returncode: 1,
                  payload_present: false,
                  stdout_bytes: 19,
                  stderr_bytes: 7,
                  cmd: ['python3', 'run_dashboard_public_topology_smoke.py'],
                  cwd: '/tmp',
                  stdout: 'root_public timeout',
                  stderr: 'timeout',
                },
                workspace_routes_smoke: {
                  returncode: 0,
                  payload_present: true,
                  stdout_bytes: 120,
                  stderr_bytes: 0,
                  cmd: ['python3', 'run_dashboard_workspace_artifacts_smoke.py'],
                  cwd: '/tmp',
                },
              },
            },
          },
        },
      },
    };

    const internalModel = buildTerminalReadModel(acceptanceLoaded);
    const internalAcceptance = (internalModel.workspace as any).publicAcceptance;
    expect(internalAcceptance.summary.status).toBe('ok');
    expect(internalAcceptance.summary.topology_generated_at_utc).toBe('2026-03-20T07:27:29Z');
    expect(internalAcceptance.summary.workspace_generated_at_utc).toBe('2026-03-20T07:27:34Z');
    expect(internalAcceptance.summary.root_contracts_screenshot_path).toBe('/tmp/root_contracts_browser.png');
    expect(internalAcceptance.summary.pages_contracts_screenshot_path).toBe('/tmp/pages_contracts_browser.png');
    expect(internalAcceptance.summary.public_snapshot_fetch_count).toBe(4);
    expect(internalAcceptance.summary.internal_snapshot_fetch_count).toBe(0);
    expect(internalAcceptance.summary.orderflow_filter_route).toBe('#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow');
    expect(internalAcceptance.summary.orderflow_active_artifact).toBe('intraday_orderflow_blueprint');
    expect(internalAcceptance.summary.orderflow_visible_artifacts).toBe('intraday_orderflow_blueprint ｜ intraday_orderflow_research_gate_blocker');
    expect(internalAcceptance.checks[0].frontend_public).toBe('https://fuuu.fun');
    expect(internalAcceptance.checks[0].tls_fallback_used).toBe(true);
    expect(internalAcceptance.checks[0].root_overview_screenshot_path).toBe('/tmp/root_overview_browser.png');
    expect(internalAcceptance.checks[0].pages_overview_screenshot_path).toBe('/tmp/pages_overview_browser.png');
    expect(internalAcceptance.checks[1].public_snapshot_fetch_count).toBe(4);
    expect(internalAcceptance.checks[1].orderflow_filter_route).toBe('#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow');
    expect(internalAcceptance.checks[1].orderflow_active_artifact).toBe('intraday_orderflow_blueprint');
    expect(internalAcceptance.checks[1].orderflow_visible_artifacts).toBe('intraday_orderflow_blueprint ｜ intraday_orderflow_research_gate_blocker');
    expect(internalAcceptance.subcommands[0].stdout).toBe('root_public timeout');

    const publicModel = buildTerminalReadModel({
      ...acceptanceLoaded,
      requestedSurface: 'public',
      effectiveSurface: 'public',
      snapshotPath: '/data/fenlie_dashboard_snapshot.json',
    });
    const publicAcceptance = (publicModel.workspace as any).publicAcceptance;
    expect(publicAcceptance.subcommands[0].stdout).toBeUndefined();
    expect(publicAcceptance.subcommands[0].stderr).toBeUndefined();
  });
});
