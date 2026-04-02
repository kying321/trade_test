from __future__ import annotations

import datetime as dt
import contextlib
import importlib.util
import json
import sys
from pathlib import Path
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_dashboard_workspace_artifacts_smoke.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dashboard_workspace_artifacts_smoke_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_workspace_routes_smoke_spec_covers_all_workspace_sections(tmp_path: Path) -> None:
    mod = load_module()
    route_assertions = [
        *mod.PUBLIC_WORKSPACE_ROUTE_ASSERTIONS[:1],
        *mod.PUBLIC_WORKSPACE_ROUTE_ASSERTIONS[1:],
    ]
    route_assertions[0] = {
        **route_assertions[0],
        "markers": [
            "系统状态 / 入口 / 路由总览",
            "研究主线摘要",
            "查看源头主线",
            "契约验收",
            "hold16_zero",
            "国内商品推理线",
            "policy_relief_watch",
            "policy_relief_chain",
            "BU2606",
        ],
    }
    route_assertions[1] = {
        **route_assertions[1],
        "expected_default_artifact": "operator_panel",
        "expected_focus_panel": "orchestration",
        "expected_focus_section": "freshness",
    }
    screenshot_path = tmp_path / "workspace-routes-smoke.png"
    result_path = tmp_path / "workspace-routes-smoke.json"

    spec = mod.build_workspace_routes_smoke_spec(
        base_url="http://127.0.0.1:4173/",
        screenshot_path=screenshot_path,
        result_path=result_path,
        route_assertions=route_assertions,
    )

    assert "/ops/overview" in spec
    assert "系统状态 / 入口 / 路由总览" in spec
    assert "研究主线摘要" in spec
    assert "查看源头主线" in spec
    assert "契约验收" in spec
    assert "hold16_zero" in spec
    assert "国内商品推理线" in spec
    assert "policy_relief_watch" in spec
    assert "policy_relief_chain" in spec
    assert "BU2606" in spec
    assert "/workspace/artifacts" in spec
    assert "const defaultWorkspaceArtifact = String(workspaceStartRoute.expected_default_artifact || 'price_action_breakout_pullback');" in spec
    assert "const defaultWorkspacePanel = String(workspaceStartRoute.expected_focus_panel || 'lab-review');" in spec
    assert "const defaultWorkspaceSection = String(workspaceStartRoute.expected_focus_section || 'research-heads');" in spec
    assert "await page.waitForFunction((artifactId) => `${window.location.pathname}${window.location.search}`.includes(`artifact=${artifactId}`), defaultWorkspaceArtifact);" in spec
    assert "await page.waitForFunction((panelId) => `${window.location.pathname}${window.location.search}`.includes(`panel=${panelId}`), defaultWorkspacePanel);" in spec
    assert "await page.waitForFunction((sectionId) => `${window.location.pathname}${window.location.search}`.includes(`section=${sectionId}`), defaultWorkspaceSection);" in spec
    assert "await expect(activeArtifactValue).toHaveAttribute('title', defaultWorkspaceArtifact);" in spec
    assert "工件池" in spec
    assert "/workspace/alignment" in spec
    assert "对齐页" in spec
    assert "仅内部可见" in spec
    assert "回测池" in spec
    assert "回测主池" in spec
    assert "原始层" in spec
    assert "原始快照" in spec
    assert "契约层" in spec
    assert "公开入口拓扑" in spec
    assert "公开面验收" in spec
    assert "穿透层 1 / 验收总览" in spec
    assert "接口目录" in spec
    assert "源头主线" in spec
    assert "回退链" in spec
    assert "研究地图" in spec
    assert "/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow" in spec
    assert "intraday_orderflow_blueprint" in spec
    assert "intraday_orderflow_research_gate_blocker" in spec
    assert "source_available: orderflowArtifacts.length > 0" in spec
    assert "active_artifact: orderflowActiveArtifact" in spec
    assert "/workspace/artifacts?artifact=price_action_exit_risk_break_even_review_conclusion" in spec
    assert "await expectStableMarker(page, exitRiskReviewActiveArtifact);" in spec
    assert "source_available: exitRiskReviewArtifacts.length > 0" in spec
    assert "active_artifact: exitRiskReviewActiveArtifact" in spec
    assert "const exitRiskReviewSectionState = await page.evaluate((sectionHint) => {" in spec
    assert "const normalizedSectionHint = String(sectionHint || '').replace(/\\s+/g, '').toLowerCase();" in spec
    assert "const sections = Array.from(document.querySelectorAll('.artifact-layer-section'));" in spec
    assert "const summaryText = String(node.querySelector('summary')?.textContent || '').replace(/\\s+/g, '').toLowerCase();" in spec
    assert "target.open = true;" in spec
    assert "if (target instanceof HTMLDetailsElement) {" in spec
    assert "expect(exitRiskReviewSectionState.opened).toBeTruthy();" in spec
    assert "exitRiskReviewVisibleArtifacts = await page.evaluate(({ sectionHint, artifactIds }) => {" in spec
    assert "const rawTitles = Array.from(target.querySelectorAll('.value-text[title]'))" in spec
    assert "expect(exitRiskReviewVisibleArtifacts).toEqual(exitRiskReviewArtifacts);" in spec
    assert "price_action_exit_risk_break_even_guarded_review" in spec
    assert "price_action_exit_risk_break_even_review_packet" in spec
    assert "price_action_exit_risk_break_even_review_conclusion" in spec
    assert "price_action_exit_risk_break_even_primary_anchor_review" in spec
    assert "price_action_exit_risk_hold_selection_aligned_break_even_review_lane" in spec
    assert "支撑证据" in spec
    assert "expectStableMarker" in spec
    assert "toLowerCase().includes(text.toLowerCase())" in spec
    assert "await expect(page.locator('body')).toContainText(marker);" in spec
    assert "context-nav" in spec
    assert "async function clickContextNav(page, label)" in spec
    assert "const expandToggle = page.getByRole('button', { name: '展开侧边导航' });" in spec
    assert "await page.getByRole('button', { name: '收起侧边导航' }).waitFor();" in spec
    assert "await page.goto('http://127.0.0.1:4173/' + route.route, { waitUntil: 'networkidle' });" in spec
    assert "internalSnapshotRequests.length" in spec
    assert "document.documentElement.dataset.theme" in spec
    assert "contracts-acceptance-subcommands" in spec
    assert "contracts-check-graph_home_smoke" in spec
    assert "contracts-subcommand-graph_home_smoke" in spec
    assert "contractsAcceptanceInspectorAssertion" in spec
    assert "[data-testid=\"contracts-acceptance-status-strip\"]" in spec
    assert "查看研究审计检索" in spec
    assert "查看研究审计工件" in spec
    assert "查看研究审计原始层" in spec
    assert "let pageSectionActiveLabel = '';" in spec
    assert "let pageSectionAccordionState = '';" in spec
    assert "await page.waitForFunction(() => `${window.location.pathname}${window.location.search}`.includes('page_section=contracts-acceptance-subcommands'));" in spec
    assert "active_label: pageSectionActiveLabel" in spec
    assert "accordion_state: pageSectionAccordionState" in spec
    assert "/workspace/contracts?page_section=contracts-source-head-operator_panel" in spec
    assert "data-accordion-id=\"contracts-source-head-operator_panel\"" in spec
    assert "状态" in spec
    assert "研究结论" in spec
    assert "生成时间" in spec
    assert "路径" in spec
    assert "/workspace/contracts?page_section=contracts-fallback" in spec
    assert "/data/fenlie_dashboard_snapshot.json" in spec
    assert "/operator_task_visual_panel.html" in spec
    assert "await expect(activeExitRiskReviewArtifact).toHaveAttribute('title', exitRiskReviewActiveArtifact);" in spec
    assert str(result_path) in spec


def test_load_public_workspace_route_assertions_uses_source_owned_active_baseline(tmp_path: Path) -> None:
    mod = load_module()
    dist_dir = tmp_path / "dist"
    data_dir = dist_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "fenlie_dashboard_snapshot.json").write_text(
        json.dumps(
            {
                "artifact_payloads": {
                    "hold_selection_handoff": {
                        "payload": {
                            "active_baseline": "hold16_zero",
                        }
                    },
                    "commodity_reasoning_summary": {
                        "payload": {
                            "primary_scenario_brief": "supply_chain_tightening",
                            "primary_chain_brief": "feedstock_cost_push_chain",
                            "contracts_in_focus": ["BU2606"],
                        }
                    }
                },
                "workspace_default_focus": {
                    "artifact": "operator_panel",
                    "group": "system_anchor",
                    "panel": "orchestration",
                    "section": "freshness",
                    "search_scope": "title",
                },
                "catalog": [
                    {
                        "id": "operator_panel",
                        "payload_key": "operator_panel",
                        "label": "操作面板",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    route_assertions = mod.load_public_workspace_route_assertions(dist_dir=dist_dir)

    assert route_assertions[0]["markers"] == [
        "系统状态 / 入口 / 路由总览",
        "研究主线摘要",
        "查看源头主线",
        "契约验收",
        "hold16_zero",
        "国内商品推理线",
        "supply_chain_tightening",
        "feedstock_cost_push_chain",
        "BU2606",
    ]
    assert route_assertions[1]["expected_default_artifact"] == "operator_panel"
    assert route_assertions[1]["expected_focus_panel"] == "orchestration"
    assert route_assertions[1]["expected_focus_section"] == "freshness"
    assert route_assertions[4]["markers"] == ["告警定向原始层", "操作面板"]


def test_load_public_workspace_route_assertions_derives_research_audit_search_cases(tmp_path: Path) -> None:
    mod = load_module()
    dist_dir = tmp_path / "dist"
    data_dir = dist_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "fenlie_dashboard_snapshot.json").write_text(
        json.dumps(
            {
                "artifact_payloads": {
                    "recent_strategy_backtests": {
                        "summary": {
                            "status": "ok",
                            "research_decision": "keep_best",
                        },
                        "payload": {
                            "mode_summaries": [
                                {
                                    "mode": "ultra_short",
                                    "trial_artifacts": [
                                        {
                                            "trial": 1,
                                            "mode": "ultra_short",
                                            "trade_journal_path": "/tmp/research/ultra_short/trial_001_ultra_short_trade_journal.csv",
                                            "holding_exposure_path": "/tmp/research/ultra_short/trial_001_ultra_short_holding_daily_symbol_exposure.csv",
                                        }
                                    ],
                                }
                            ]
                        },
                    },
                    "strategy_lab_summary": {
                        "summary": {
                            "status": "ok",
                            "research_decision": "selected",
                        },
                        "payload": {
                            "best_candidate": {
                                "name": "report_momentum_02",
                                "trade_journal_path": "/tmp/strategy_lab/best_report_momentum_02_trade_journal.csv",
                            },
                            "candidates": [
                                {
                                    "name": "report_momentum_02",
                                    "trade_journal_path": "/tmp/strategy_lab/candidate_02_report_momentum_02_trade_journal.csv",
                                }
                            ],
                        },
                    },
                },
                "workspace_default_focus": {
                    "artifact": "operator_panel",
                    "group": "system_anchor",
                    "panel": "orchestration",
                    "section": "freshness",
                    "search_scope": "title",
                },
                "catalog": [
                    {
                        "id": "operator_panel",
                        "payload_key": "operator_panel",
                        "label": "操作面板",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    route_assertions = mod.load_public_workspace_route_assertions(dist_dir=dist_dir)

    assert route_assertions[1]["research_audit_search_cases"] == [
        {
            "case_id": "optimizer_trial_trade_journal",
            "scope": "artifact",
            "query": "trial_001_ultra_short_trade_journal",
            "search_route": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
            "result_artifact": "audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
            "result_label": "ultra_short / trial_001 / trade_journal",
            "workspace_route": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
            "raw_path": "/tmp/research/ultra_short/trial_001_ultra_short_trade_journal.csv",
        },
        {
            "case_id": "strategy_lab_candidate_trade_journal",
            "scope": "artifact",
            "query": "candidate_02_report_momentum_02_trade_journal",
            "search_route": "/search?q=candidate_02_report_momentum_02_trade_journal&scope=artifact",
            "result_artifact": "audit:strategy_lab_summary:report_momentum_02:trade_journal",
            "result_label": "report_momentum_02 / trade_journal",
            "workspace_route": "/workspace/artifacts?artifact=audit:strategy_lab_summary:report_momentum_02:trade_journal",
            "raw_path": "/tmp/strategy_lab/candidate_02_report_momentum_02_trade_journal.csv",
        },
    ]


def test_load_commodity_visibility_route_assertions_reads_public_snapshot_markers(tmp_path: Path) -> None:
    mod = load_module()
    dist_dir = tmp_path / "dist"
    data_dir = dist_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "fenlie_dashboard_snapshot.json").write_text(
        json.dumps(
            {
                "artifact_payloads": {
                    "commodity_reasoning_summary": {
                        "payload": {
                            "primary_scenario_brief": "policy_relief_watch",
                            "primary_chain_brief": "policy_relief_chain",
                            "contracts_in_focus": ["BU2606"],
                        }
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    route_assertions = mod.load_commodity_visibility_route_assertions(dist_dir=dist_dir)

    assert route_assertions == [
        {
            "route": "/ops/overview",
            "nav_label": "总览",
            "headline": "总览",
            "markers": ["国内商品推理线", "policy_relief_watch", "policy_relief_chain", "BU2606"],
        },
        {
            "route": "/ops/risk",
            "nav_label": "操作终端",
            "headline": "Observe / Diagnose / Act",
            "markers": ["国内商品推理线", "policy_relief_watch", "policy_relief_chain", "BU2606"],
        },
    ]


def test_build_commodity_visibility_smoke_spec_covers_overview_and_terminal_public(tmp_path: Path) -> None:
    mod = load_module()
    screenshot_path = tmp_path / "commodity-visibility-smoke.png"
    result_path = tmp_path / "commodity-visibility-smoke.json"

    spec = mod.build_commodity_visibility_smoke_spec(
        base_url="http://127.0.0.1:4173/",
        screenshot_path=screenshot_path,
        result_path=result_path,
        route_assertions=[
            {
                "route": "/ops/overview",
                "nav_label": "总览",
                "headline": "总览",
                "markers": ["国内商品推理线", "policy_relief_watch", "policy_relief_chain", "BU2606"],
            },
                {
                    "route": "/ops/risk",
                    "nav_label": "操作终端",
                    "headline": "Observe / Diagnose / Act",
                    "markers": ["国内商品推理线", "policy_relief_watch", "policy_relief_chain", "BU2606"],
                },
            ],
    )

    assert "/ops/overview" in spec
    assert "/ops/risk" in spec
    assert "国内商品推理线" in spec
    assert "policy_relief_watch" in spec
    assert "policy_relief_chain" in spec
    assert "BU2606" in spec
    assert "commodity visibility smoke" in spec
    assert "visited_routes" in spec
    assert "await expect(page.locator('body')).toContainText(marker);" in spec
    assert "requested_surface: 'public'" in spec
    assert str(result_path) in spec


def test_build_workspace_routes_smoke_spec_covers_research_audit_search_and_raw_deeplinks(tmp_path: Path) -> None:
    mod = load_module()
    screenshot_path = tmp_path / "workspace-routes-smoke.png"
    result_path = tmp_path / "workspace-routes-smoke.json"
    route_assertions = [
        *mod.PUBLIC_WORKSPACE_ROUTE_ASSERTIONS[:1],
        {
            **mod.PUBLIC_WORKSPACE_ROUTE_ASSERTIONS[1],
            "research_audit_search_cases": [
                {
                    "case_id": "optimizer_trial_trade_journal",
                    "scope": "artifact",
                    "query": "trial_001_ultra_short_trade_journal",
                    "search_route": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                    "result_artifact": "audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                    "result_label": "ultra_short / trial_001 / trade_journal",
                    "workspace_route": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                    "raw_path": "/tmp/research/ultra_short/trial_001_ultra_short_trade_journal.csv",
                },
                {
                    "case_id": "strategy_lab_candidate_trade_journal",
                    "scope": "artifact",
                    "query": "candidate_02_report_momentum_02_trade_journal",
                    "search_route": "/search?q=candidate_02_report_momentum_02_trade_journal&scope=artifact",
                    "result_artifact": "audit:strategy_lab_summary:report_momentum_02:trade_journal",
                    "result_label": "report_momentum_02 / trade_journal",
                    "workspace_route": "/workspace/artifacts?artifact=audit:strategy_lab_summary:report_momentum_02:trade_journal",
                    "raw_path": "/tmp/strategy_lab/candidate_02_report_momentum_02_trade_journal.csv",
                },
            ],
        },
        *mod.PUBLIC_WORKSPACE_ROUTE_ASSERTIONS[2:],
    ]

    spec = mod.build_workspace_routes_smoke_spec(
        base_url="http://127.0.0.1:4173/",
        screenshot_path=screenshot_path,
        result_path=result_path,
        route_assertions=route_assertions,
    )

    assert "const researchAuditCases = Array.isArray(workspaceStartRoute.research_audit_search_cases)" in spec
    assert "research_audit_search_assertion" in spec
    assert "/search?q=trial_001_ultra_short_trade_journal&scope=artifact" in spec
    assert "audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal" in spec
    assert "candidate_02_report_momentum_02_trade_journal" in spec
    assert "audit:strategy_lab_summary:report_momentum_02:trade_journal" in spec
    assert "await expect(page.locator('.global-search-input')).toHaveValue(auditQuery);" in spec
    assert "await expectStableMarker(page, auditRawPath);" in spec
    assert "const searchResultButton = page.getByRole('button', { name: new RegExp(escapeRegExp(auditRawPath)) }).first();" in spec
    assert "await expectStableMarker(page, auditResultArtifact);" in spec
    assert "const rawLink = page.locator(`a[href*=\"${encodeURIComponent(auditRawPath)}\"]`).first();" in spec
    assert "await rawLink.click();" in spec
    assert "search_route: auditCase.search_route" in spec
    assert "workspace_route: auditCase.workspace_route" in spec
    assert "raw_path: auditCase.raw_path" in spec


def test_build_graph_home_smoke_spec_covers_default_route_fallback_and_quick_links(tmp_path: Path) -> None:
    mod = load_module()
    screenshot_path = tmp_path / "graph-home-smoke.png"
    result_path = tmp_path / "graph-home-smoke.json"

    spec = mod.build_graph_home_smoke_spec(
        base_url="http://127.0.0.1:4173/",
        screenshot_path=screenshot_path,
        result_path=result_path,
        route_assertions=[
            {
                "route": "/graph-home",
                "nav_label": "图谱主页",
                "headline": "图谱化主页",
                "markers": ["默认管道", "推荐下一跳", "回到交易中枢", "去操作终端", "去研究工作区", "打开全局搜索"],
                "research_audit_search_cases": [
                    {
                        "case_id": "optimizer_trial_trade_journal",
                        "query": "trial_001_ultra_short_trade_journal",
                        "search_route": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                        "result_artifact": "audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                        "workspace_route": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                        "raw_path": "system/output/research/20260330_133956/ultra_short/trial_001_ultra_short_trade_journal.csv",
                    }
                ],
            }
        ],
    )

    assert "const DEFAULT_ROUTE = '/';" in spec
    assert "const FALLBACK_ROUTE = '/unknown-route';" in spec
    assert "/graph-home" in spec
    assert "图谱化主页" in spec
    assert "默认管道" in spec
    assert "推荐下一跳" in spec
    assert "去操作终端" in spec
    assert "去研究工作区" in spec
    assert "打开全局搜索" in spec
    assert "graph home smoke" in spec
    assert "graph_home_assertion" in spec
    assert "const resolvedRoute = await page.evaluate(() => `${window.location.pathname}${window.location.search}`);" in spec
    assert "const graphCanvas = page.locator('canvas').first();" in spec
    assert "candidateOffsets" in spec
    assert "canvas_selection_assertion" in spec
    assert "terminal_link_href" in spec
    assert "workspace_link_href" in spec
    assert "search_link_href" in spec
    assert "researchAuditCases" in spec
    assert "research_audit_link_assertion" in spec
    assert "research_audit_link_assertions" in spec
    assert "researchAuditLinkAssertions.push" in spec
    assert "async function waitForRoute(page, expectedPathname, expectedParams = {})" in spec
    assert "selected_heading: String(defaultCenter || '').trim().replace(/^中心：/, '')" in spec
    assert "检索 / ${auditCase.query}" in spec
    assert "工件 / ${auditCase.result_artifact}" in spec
    assert "原始层 / ${auditCase.raw_path}" in spec
    assert "const auditResultArtifact = String(auditCase.result_artifact || '');" in spec
    assert "await waitForRoute(page, '/workspace/artifacts', { artifact: auditResultArtifact });" in spec
    assert str(result_path) in spec


def test_build_artifact_payload_reports_graph_home_surface(tmp_path: Path) -> None:
    mod = load_module()
    report_path = tmp_path / "report.json"
    screenshot_path = tmp_path / "graph-home-smoke.png"

    payload = mod.build_artifact_payload(
        workspace=tmp_path,
        report_path=report_path,
        screenshot_path=screenshot_path,
        build_result={"returncode": 0},
        server_ready_seconds=0.11,
        smoke_result={
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "playwright_result": {
                    "requested_surface": "public",
                    "effective_surface": "public",
                    "visited_routes": [
                        {"route": "/graph-home", "headline": "图谱化主页"},
                        {"route": "/ops/risk", "headline": "Observe / Diagnose / Act"},
                        {"route": "/workspace/artifacts", "headline": "工件目标池"},
                        {"route": "/search", "headline": "全局关键词搜索"},
                    ],
                "snapshot_requests": [
                    "http://127.0.0.1:4173/data/fenlie_dashboard_snapshot.json?ts=1",
                ],
                "internal_snapshot_requests": [],
                "graph_home_assertion": {
                    "default_route": "/",
                    "fallback_route": "/unknown-route",
                    "resolved_route": "/graph-home",
                    "default_center": "交易中枢",
                    "terminal_link_href": "/ops/risk",
                    "workspace_link_href": "/workspace/artifacts",
                    "search_link_href": "/search",
                    "canvas_selection_assertion": {
                        "selected_heading": "执行与风控",
                        "selected_center": "执行与风控",
                        "recenter_heading": "交易中枢",
                    },
                    "research_audit_link_assertions": [
                        {
                            "selected_heading": "交易中枢",
                            "case_id": "optimizer_trial_trade_journal",
                            "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                        },
                        {
                            "selected_heading": "交易中枢",
                            "case_id": "strategy_lab_candidate_trade_journal",
                            "search_link_href": "/search?q=candidate_01_trend_convex_01_trade_journal&scope=artifact",
                            "artifact_link_href": "/workspace/artifacts?artifact=audit:strategy_lab_summary:trend_convex_01:trade_journal",
                            "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2Fstrategy_lab_20260330_134050%2Fcandidate_01_trend_convex_01_trade_journal.csv",
                        },
                    ],
                },
            },
        },
        base_url="http://127.0.0.1:4173/",
        mode="graph_home",
        expected_route_markers=[
            {
                "route": "/graph-home",
                "nav_label": "图谱主页",
                "headline": "图谱化主页",
                "markers": ["默认管道", "推荐下一跳", "回到交易中枢", "去操作终端", "去研究工作区", "打开全局搜索"],
            }
        ],
    )

    assert payload["action"] == "dashboard_graph_home_browser_smoke"
    assert payload["ok"] is True
    assert payload["routes"] == [
        {"route": "/graph-home", "headline": "图谱化主页"},
        {"route": "/ops/risk", "headline": "Observe / Diagnose / Act"},
        {"route": "/workspace/artifacts", "headline": "工件目标池"},
        {"route": "/search", "headline": "全局关键词搜索"},
    ]
    assert payload["graph_home_assertion"] == {
        "default_route": "/",
        "fallback_route": "/unknown-route",
        "resolved_route": "/graph-home",
        "default_center": "交易中枢",
        "terminal_link_href": "/ops/risk",
        "workspace_link_href": "/workspace/artifacts",
        "search_link_href": "/search",
        "canvas_selection_assertion": {
            "selected_heading": "执行与风控",
            "selected_center": "执行与风控",
            "recenter_heading": "交易中枢",
        },
        "research_audit_link_assertions": [
            {
                "selected_heading": "交易中枢",
                "case_id": "optimizer_trial_trade_journal",
                "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
            },
            {
                "selected_heading": "交易中枢",
                "case_id": "strategy_lab_candidate_trade_journal",
                "search_link_href": "/search?q=candidate_01_trend_convex_01_trade_journal&scope=artifact",
                "artifact_link_href": "/workspace/artifacts?artifact=audit:strategy_lab_summary:trend_convex_01:trade_journal",
                "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2Fstrategy_lab_20260330_134050%2Fcandidate_01_trend_convex_01_trade_journal.csv",
            },
        ],
    }


def test_build_graph_home_narrow_smoke_spec_covers_small_shell_tier_and_sidebar_contract(tmp_path: Path) -> None:
    mod = load_module()
    screenshot_path = tmp_path / "graph-home-narrow-smoke.png"
    result_path = tmp_path / "graph-home-narrow-smoke.json"

    spec = mod.build_graph_home_narrow_smoke_spec(
        base_url="http://127.0.0.1:4173/",
        screenshot_path=screenshot_path,
        result_path=result_path,
        route_assertions=[
            {
                "route": "/graph-home",
                "nav_label": "图谱主页",
                "headline": "图谱化主页",
                "markers": ["默认管道", "推荐下一跳", "去操作终端", "去研究工作区", "打开全局搜索"],
            }
        ],
    )

    assert "graph home narrow smoke" in spec
    assert "const DEFAULT_ROUTE = '/';" in spec
    assert "await page.waitForFunction(() => window.location.pathname.includes('/graph-home'));" in spec
    assert "const resolvedRoute = await page.evaluate(() => window.location.pathname);" in spec
    assert "viewport: { width: 390, height: 844 }" in spec
    assert "data-shell-tier" in spec
    assert "sidebar_toggle_visible" in spec
    assert "sidebar_utility_link_href" in spec
    assert "搜索 / Search" in spec
    assert str(result_path) in spec


def test_build_artifact_payload_reports_graph_home_narrow_surface(tmp_path: Path) -> None:
    mod = load_module()
    report_path = tmp_path / "report.json"
    screenshot_path = tmp_path / "graph-home-narrow-smoke.png"

    payload = mod.build_artifact_payload(
        workspace=tmp_path,
        report_path=report_path,
        screenshot_path=screenshot_path,
        build_result={"returncode": 0},
        server_ready_seconds=0.12,
        smoke_result={
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "playwright_result": {
                "requested_surface": "public",
                "effective_surface": "public",
                "visited_routes": [
                    {"route": "/graph-home", "headline": "图谱化主页"},
                ],
                "snapshot_requests": [
                    "http://127.0.0.1:4173/data/fenlie_dashboard_snapshot.json?ts=1",
                ],
                "internal_snapshot_requests": [],
                "graph_home_assertion": {
                    "default_route": "/",
                    "resolved_route": "/graph-home",
                    "shell_tier": "s",
                    "sidebar_toggle_visible": False,
                    "sidebar_utility_link_href": "/search",
                },
            },
        },
        base_url="http://127.0.0.1:4173/",
        mode="graph_home_narrow",
        expected_route_markers=[
            {
                "route": "/graph-home",
                "nav_label": "图谱主页",
                "headline": "图谱化主页",
                "markers": ["默认管道", "推荐下一跳", "去操作终端", "去研究工作区", "打开全局搜索"],
            }
        ],
    )

    assert payload["action"] == "dashboard_graph_home_narrow_browser_smoke"
    assert payload["ok"] is True
    assert payload["graph_home_assertion"] == {
        "default_route": "/",
        "resolved_route": "/graph-home",
        "shell_tier": "s",
        "sidebar_toggle_visible": False,
        "sidebar_utility_link_href": "/search",
    }


def test_build_graph_home_pipeline_smoke_spec_covers_drag_reorder_and_refresh_persistence(tmp_path: Path) -> None:
    mod = load_module()
    screenshot_path = tmp_path / "graph-home-pipeline-smoke.png"
    result_path = tmp_path / "graph-home-pipeline-smoke.json"

    spec = mod.build_graph_home_pipeline_smoke_spec(
        base_url="http://127.0.0.1:4173/",
        screenshot_path=screenshot_path,
        result_path=result_path,
        route_assertions=[
            {
                "route": "/graph-home",
                "nav_label": "图谱主页",
                "headline": "图谱化主页",
                "markers": ["加入自定义管道", "创建默认管道"],
            }
        ],
    )

    assert "graph home pipeline smoke" in spec
    assert "const DEFAULT_ROUTE = '/';" in spec
    assert "await page.waitForFunction(() => window.location.pathname.includes('/graph-home'));" in spec
    assert "const resolvedRoute = await page.evaluate(() => window.location.pathname);" in spec
    assert "graph_home_pipelines_v1" in spec
    assert "dragTo" in spec
    assert "pipeline_persistence_assertion" in spec
    assert "await page.reload({ waitUntil: 'networkidle' });" in spec
    assert "加入自定义管道" in spec
    assert str(result_path) in spec


def test_build_artifact_payload_reports_graph_home_pipeline_surface(tmp_path: Path) -> None:
    mod = load_module()
    report_path = tmp_path / "report.json"
    screenshot_path = tmp_path / "graph-home-pipeline-smoke.png"

    payload = mod.build_artifact_payload(
        workspace=tmp_path,
        report_path=report_path,
        screenshot_path=screenshot_path,
        build_result={"returncode": 0},
        server_ready_seconds=0.14,
        smoke_result={
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "playwright_result": {
                "requested_surface": "public",
                "effective_surface": "public",
                "visited_routes": [
                    {"route": "/graph-home", "headline": "图谱化主页"},
                ],
                "snapshot_requests": [
                    "http://127.0.0.1:4173/data/fenlie_dashboard_snapshot.json?ts=1",
                ],
                "internal_snapshot_requests": [],
                "graph_home_assertion": {
                    "default_route": "/",
                    "resolved_route": "/graph-home",
                    "pipeline_persistence_assertion": {
                        "selected_heading": "执行与风控",
                        "initial_order": ["交易中枢", "执行与风控"],
                        "reordered_order": ["执行与风控", "交易中枢"],
                        "persisted_order": ["执行与风控", "交易中枢"],
                    },
                },
            },
        },
        base_url="http://127.0.0.1:4173/",
        mode="graph_home_pipeline",
        expected_route_markers=[
            {
                "route": "/graph-home",
                "nav_label": "图谱主页",
                "headline": "图谱化主页",
                "markers": ["加入自定义管道", "创建默认管道"],
            }
        ],
    )

    assert payload["action"] == "dashboard_graph_home_pipeline_browser_smoke"
    assert payload["ok"] is True
    assert payload["graph_home_assertion"] == {
        "default_route": "/",
        "resolved_route": "/graph-home",
        "pipeline_persistence_assertion": {
            "selected_heading": "执行与风控",
            "initial_order": ["交易中枢", "执行与风控"],
            "reordered_order": ["执行与风控", "交易中枢"],
            "persisted_order": ["执行与风控", "交易中枢"],
        },
    }


def test_build_artifact_payload_reports_commodity_visibility_surface(tmp_path: Path) -> None:
    mod = load_module()
    report_path = tmp_path / "report.json"
    screenshot_path = tmp_path / "commodity-visibility-smoke.png"

    payload = mod.build_artifact_payload(
        workspace=tmp_path,
        report_path=report_path,
        screenshot_path=screenshot_path,
        build_result={"returncode": 0},
        server_ready_seconds=0.14,
        smoke_result={
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "playwright_result": {
                "requested_surface": "public",
                "effective_surface": "public",
                "visited_routes": [
                    {"route": "/ops/overview", "headline": "总览"},
                    {"route": "/ops/risk", "headline": "Observe / Diagnose / Act"},
                ],
                "snapshot_requests": [
                    "http://127.0.0.1:4173/data/fenlie_dashboard_snapshot.json?ts=1",
                ],
                "internal_snapshot_requests": [],
            },
        },
        base_url="http://127.0.0.1:4173/",
        mode="commodity_visibility",
        expected_route_markers=[
            {
                "route": "/ops/overview",
                "nav_label": "总览",
                "headline": "总览",
                "markers": ["国内商品推理线", "policy_relief_watch", "policy_relief_chain", "BU2606"],
            },
                {
                    "route": "/ops/risk",
                    "nav_label": "操作终端",
                    "headline": "Observe / Diagnose / Act",
                    "markers": ["国内商品推理线", "policy_relief_watch", "policy_relief_chain", "BU2606"],
                },
            ],
    )

    assert payload["action"] == "dashboard_commodity_visibility_browser_smoke"
    assert payload["ok"] is True
    assert payload["surface_assertion"]["requested_surface"] == "public"
    assert payload["surface_assertion"]["effective_surface"] == "public"
    assert payload["routes"] == [
        {"route": "/ops/overview", "headline": "总览"},
        {"route": "/ops/risk", "headline": "Observe / Diagnose / Act"},
    ]
    assert payload["expected_route_markers"] == [
        {
            "route": "/ops/overview",
            "nav_label": "总览",
            "headline": "总览",
            "markers": ["国内商品推理线", "policy_relief_watch", "policy_relief_chain", "BU2606"],
        },
        {
            "route": "/ops/risk",
            "nav_label": "操作终端",
            "headline": "Observe / Diagnose / Act",
            "markers": ["国内商品推理线", "policy_relief_watch", "policy_relief_chain", "BU2606"],
        },
    ]


def test_build_artifact_payload_reports_workspace_route_matrix(tmp_path: Path) -> None:
    mod = load_module()
    report_path = tmp_path / "report.json"
    screenshot_path = tmp_path / "workspace-routes-smoke.png"

    payload = mod.build_artifact_payload(
        workspace=tmp_path,
        report_path=report_path,
        screenshot_path=screenshot_path,
        build_result={"returncode": 0},
        server_ready_seconds=0.42,
        smoke_result={
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "playwright_result": {
                "requested_surface": "public",
                "effective_surface": "public",
                "visited_routes": [
                    {"route": "/ops/overview", "headline": "总览"},
                    {"route": "/workspace/artifacts", "headline": "工件目标池"},
                    {"route": "/workspace/alignment", "headline": "方向对齐投射"},
                    {"route": "/workspace/backtests", "headline": "回测主池"},
                    {"route": "/workspace/raw", "headline": "原始快照"},
                    {"route": "/workspace/contracts", "headline": "公开入口拓扑"},
                ],
                "snapshot_requests": [
                    "http://127.0.0.1:4173/data/fenlie_dashboard_snapshot.json?ts=1",
                    "http://127.0.0.1:4173/data/fenlie_dashboard_snapshot.json?ts=2",
                    "http://127.0.0.1:4173/data/fenlie_dashboard_snapshot.json?ts=3",
                    "http://127.0.0.1:4173/data/fenlie_dashboard_snapshot.json?ts=4",
                    "http://127.0.0.1:4173/data/fenlie_dashboard_snapshot.json?ts=5",
                ],
                "internal_snapshot_requests": [],
                "theme_assertion": {
                    "route": "/workspace/contracts?theme=light",
                    "requested_theme": "light",
                    "resolved_theme": "light",
                },
                "page_section_assertion": {
                    "route": "/workspace/contracts?page_section=contracts-acceptance-subcommands",
                    "page_section": "contracts-acceptance-subcommands",
                    "active_label": "子命令证据",
                    "accordion_state": "",
                },
                "contracts_source_head_assertion": {
                    "route": "/workspace/contracts?page_section=contracts-source-head-operator_panel",
                    "page_section": "contracts-source-head-operator_panel",
                    "source_head_id": "operator_panel",
                    "accordion_state": "open",
                    "visible_markers": [
                        "状态",
                        "研究结论",
                        "生成时间",
                        "路径",
                    ],
                },
                "contracts_source_gap_assertion": {
                    "route": "/workspace/contracts?page_section=contracts-fallback",
                    "page_section": "contracts-fallback",
                    "visible_markers": [
                        "/workspace/raw",
                        "/data/fenlie_dashboard_snapshot.json",
                        "/operator_task_visual_panel.html",
                    ],
                },
                "artifacts_filter_assertion": {
                    "route": "/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
                    "group": "research_cross_section",
                    "search_scope": "title",
                    "search": "orderflow",
                    "source_available": True,
                    "active_artifact": "intraday_orderflow_blueprint",
                    "visible_artifacts": [
                        "intraday_orderflow_blueprint",
                        "intraday_orderflow_research_gate_blocker",
                    ],
                },
                "artifacts_exit_risk_review_assertion": {
                    "route": "/workspace/artifacts?artifact=price_action_exit_risk_break_even_review_conclusion",
                    "group": "research_exit_risk",
                    "search_scope": "title",
                    "search": "",
                    "source_available": True,
                    "section_label": "支撑证据",
                    "active_artifact": "price_action_exit_risk_break_even_review_conclusion",
                    "visible_artifacts": [
                        "price_action_exit_risk_break_even_guarded_review",
                        "price_action_exit_risk_break_even_review_packet",
                        "price_action_exit_risk_break_even_review_conclusion",
                        "price_action_exit_risk_break_even_primary_anchor_review",
                        "price_action_exit_risk_hold_selection_aligned_break_even_review_lane",
                    ],
                    "visible_markers": [
                        "price_action_exit_risk_break_even_guarded_review",
                        "price_action_exit_risk_break_even_review_packet",
                        "price_action_exit_risk_break_even_review_conclusion",
                        "price_action_exit_risk_break_even_primary_anchor_review",
                        "price_action_exit_risk_hold_selection_aligned_break_even_review_lane",
                    ],
                },
                "research_audit_search_assertion": {
                    "route": "/search",
                    "cases_available": True,
                    "cases": [
                        {
                            "case_id": "optimizer_trial_trade_journal",
                            "scope": "artifact",
                            "query": "trial_001_ultra_short_trade_journal",
                            "search_route": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "result_artifact": "audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "workspace_route": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_path": "/tmp/research/ultra_short/trial_001_ultra_short_trade_journal.csv",
                        },
                        {
                            "case_id": "strategy_lab_candidate_trade_journal",
                            "scope": "artifact",
                            "query": "candidate_02_report_momentum_02_trade_journal",
                            "search_route": "/search?q=candidate_02_report_momentum_02_trade_journal&scope=artifact",
                            "result_artifact": "audit:strategy_lab_summary:report_momentum_02:trade_journal",
                            "workspace_route": "/workspace/artifacts?artifact=audit:strategy_lab_summary:report_momentum_02:trade_journal",
                            "raw_path": "/tmp/strategy_lab/candidate_02_report_momentum_02_trade_journal.csv",
                        },
                    ],
                },
                "contracts_acceptance_inspector_assertion": {
                    "checks_by_id": {
                        "workspace_routes_smoke": {
                            "route": "/workspace/contracts?page_section=contracts-check-workspace_routes_smoke",
                            "page_section": "contracts-check-workspace_routes_smoke",
                            "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_link_href": "/workspace/raw?artifact=%2Ftmp%2Fresearch%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                        },
                        "graph_home_smoke": {
                            "route": "/workspace/contracts?page_section=contracts-check-graph_home_smoke",
                            "page_section": "contracts-check-graph_home_smoke",
                            "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_link_href": "/workspace/raw?artifact=%2Ftmp%2Fresearch%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                        }
                    },
                    "subcommands_by_id": {
                        "workspace_routes_smoke": {
                            "route": "/workspace/contracts?page_section=contracts-subcommand-workspace_routes_smoke",
                            "page_section": "contracts-subcommand-workspace_routes_smoke",
                            "check_route": "/workspace/contracts?page_section=contracts-check-workspace_routes_smoke",
                            "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_link_href": "/workspace/raw?artifact=%2Ftmp%2Fresearch%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                        },
                        "graph_home_smoke": {
                            "route": "/workspace/contracts?page_section=contracts-subcommand-graph_home_smoke",
                            "page_section": "contracts-subcommand-graph_home_smoke",
                            "check_route": "/workspace/contracts?page_section=contracts-check-graph_home_smoke",
                            "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_link_href": "/workspace/raw?artifact=%2Ftmp%2Fresearch%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                        }
                    },
                },
            },
        },
        base_url="http://127.0.0.1:4173/",
    )

    assert payload["action"] == "dashboard_workspace_routes_browser_smoke"
    assert payload["ok"] is True
    assert payload["change_class"] == "RESEARCH_ONLY"
    assert [row["route"] for row in payload["routes"]] == [
        "/ops/overview",
        "/workspace/artifacts",
        "/workspace/alignment",
        "/workspace/backtests",
        "/workspace/raw",
        "/workspace/contracts",
    ]
    assert payload["network_observation"]["public_snapshot_fetch_count"] == 5
    assert payload["network_observation"]["internal_snapshot_fetch_count"] == 0
    assert payload["surface_assertion"]["requested_surface"] == "public"
    assert payload["surface_assertion"]["effective_surface"] == "public"
    assert payload["theme_assertion"] == {
        "route": "/workspace/contracts?theme=light",
        "requested_theme": "light",
        "resolved_theme": "light",
    }
    assert payload["page_section_assertion"] == {
        "applicable": True,
        "route": "/workspace/contracts?page_section=contracts-acceptance-subcommands",
        "page_section": "contracts-acceptance-subcommands",
        "active_label": "子命令证据",
        "accordion_state": "",
    }
    assert payload["contracts_source_head_assertion"] == {
        "applicable": True,
        "route": "/workspace/contracts?page_section=contracts-source-head-operator_panel",
        "page_section": "contracts-source-head-operator_panel",
        "source_head_id": "operator_panel",
        "accordion_state": "open",
        "visible_markers": [
            "状态",
            "研究结论",
            "生成时间",
            "路径",
        ],
    }
    assert payload["contracts_source_gap_assertion"] == {
        "applicable": True,
        "route": "/workspace/contracts?page_section=contracts-fallback",
        "page_section": "contracts-fallback",
        "visible_markers": [
            "/workspace/raw",
            "/data/fenlie_dashboard_snapshot.json",
            "/operator_task_visual_panel.html",
        ],
    }
    assert payload["artifacts_filter_assertion"] == {
        "applicable": True,
        "route": "/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
        "group": "research_cross_section",
        "search_scope": "title",
        "search": "orderflow",
        "source_available": True,
        "active_artifact": "intraday_orderflow_blueprint",
        "visible_artifacts": [
            "intraday_orderflow_blueprint",
            "intraday_orderflow_research_gate_blocker",
        ],
    }
    assert payload["artifacts_exit_risk_review_assertion"] == {
        "applicable": True,
        "route": "/workspace/artifacts?artifact=price_action_exit_risk_break_even_review_conclusion",
        "group": "research_exit_risk",
        "search_scope": "title",
        "search": "",
        "source_available": True,
        "section_label": "支撑证据",
        "active_artifact": "price_action_exit_risk_break_even_review_conclusion",
        "visible_artifacts": [
            "price_action_exit_risk_break_even_guarded_review",
            "price_action_exit_risk_break_even_review_packet",
            "price_action_exit_risk_break_even_review_conclusion",
            "price_action_exit_risk_break_even_primary_anchor_review",
            "price_action_exit_risk_hold_selection_aligned_break_even_review_lane",
        ],
        "visible_markers": [
            "price_action_exit_risk_break_even_guarded_review",
            "price_action_exit_risk_break_even_review_packet",
            "price_action_exit_risk_break_even_review_conclusion",
            "price_action_exit_risk_break_even_primary_anchor_review",
            "price_action_exit_risk_hold_selection_aligned_break_even_review_lane",
        ],
    }
    assert payload["research_audit_search_assertion"] == {
        "applicable": True,
        "route": "/search",
        "cases_available": True,
        "cases": [
            {
                "case_id": "optimizer_trial_trade_journal",
                "scope": "artifact",
                "query": "trial_001_ultra_short_trade_journal",
                "search_route": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                "result_artifact": "audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                "workspace_route": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                "raw_path": "/tmp/research/ultra_short/trial_001_ultra_short_trade_journal.csv",
            },
            {
                "case_id": "strategy_lab_candidate_trade_journal",
                "scope": "artifact",
                "query": "candidate_02_report_momentum_02_trade_journal",
                "search_route": "/search?q=candidate_02_report_momentum_02_trade_journal&scope=artifact",
                "result_artifact": "audit:strategy_lab_summary:report_momentum_02:trade_journal",
                "workspace_route": "/workspace/artifacts?artifact=audit:strategy_lab_summary:report_momentum_02:trade_journal",
                "raw_path": "/tmp/strategy_lab/candidate_02_report_momentum_02_trade_journal.csv",
            },
        ],
    }
    assert payload["contracts_acceptance_inspector_assertion"] == {
        "applicable": True,
        "checks_by_id": {
            "topology_smoke": {
                "route": "/workspace/contracts?page_section=contracts-check-topology_smoke",
                "page_section": "contracts-check-topology_smoke",
                "search_link_href": "",
                "artifact_link_href": "",
                "raw_link_href": "",
            },
            "workspace_routes_smoke": {
                "route": "/workspace/contracts?page_section=contracts-check-workspace_routes_smoke",
                "page_section": "contracts-check-workspace_routes_smoke",
                "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                "raw_link_href": "/workspace/raw?artifact=%2Ftmp%2Fresearch%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
            },
            "graph_home_smoke": {
                "route": "/workspace/contracts?page_section=contracts-check-graph_home_smoke",
                "page_section": "contracts-check-graph_home_smoke",
                "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                "raw_link_href": "/workspace/raw?artifact=%2Ftmp%2Fresearch%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
            }
        },
        "subcommands_by_id": {
            "topology_smoke": {
                "route": "/workspace/contracts?page_section=contracts-subcommand-topology_smoke",
                "page_section": "contracts-subcommand-topology_smoke",
                "check_route": "/workspace/contracts?page_section=contracts-check-topology_smoke",
                "search_link_href": "",
                "artifact_link_href": "",
                "raw_link_href": "",
            },
            "workspace_routes_smoke": {
                "route": "/workspace/contracts?page_section=contracts-subcommand-workspace_routes_smoke",
                "page_section": "contracts-subcommand-workspace_routes_smoke",
                "check_route": "/workspace/contracts?page_section=contracts-check-workspace_routes_smoke",
                "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                "raw_link_href": "/workspace/raw?artifact=%2Ftmp%2Fresearch%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
            },
            "graph_home_smoke": {
                "route": "/workspace/contracts?page_section=contracts-subcommand-graph_home_smoke",
                "page_section": "contracts-subcommand-graph_home_smoke",
                "check_route": "/workspace/contracts?page_section=contracts-check-graph_home_smoke",
                "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                "raw_link_href": "/workspace/raw?artifact=%2Ftmp%2Fresearch%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
            }
        },
    }


def test_build_internal_alignment_smoke_spec_uses_internal_snapshot_projection_markers(tmp_path: Path) -> None:
    mod = load_module()
    screenshot_path = tmp_path / "internal-alignment-smoke.png"
    result_path = tmp_path / "internal-alignment-smoke.json"

    spec = mod.build_internal_alignment_smoke_spec(
        base_url="http://127.0.0.1:4173/",
        screenshot_path=screenshot_path,
        result_path=result_path,
        projection_summary_headline="将高价值对话反馈投射到内部对齐页",
        top_event_headline="拆分研究工作区左栏四控件",
        top_action="先刷新 internal snapshot 并验证对齐页，再收紧 internal fallback 语义。",
    )

    assert "/workspace/alignment?view=internal&page_section=alignment-summary" in spec
    assert "方向对齐投射" in spec
    assert "将高价值对话反馈投射到内部对齐页" in spec
    assert "拆分研究工作区左栏四控件" in spec
    assert "先刷新 internal snapshot 并验证对齐页，再收紧 internal fallback 语义。" in spec
    assert "internalSnapshotRequests.length" in spec
    assert "publicSnapshotRequests.length" in spec
    assert "expect(publicSnapshotRequests.length).toBe(0)" in spec
    assert "requested_surface: 'internal'" in spec
    assert "effective_surface: 'internal'" in spec
    assert str(result_path) in spec


def test_load_internal_terminal_focus_expectations_reads_focus_slot_source_row(tmp_path: Path) -> None:
    mod = load_module()
    dist_dir = tmp_path / "dist"
    data_dir = dist_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "fenlie_dashboard_internal_snapshot.json").write_text(
        json.dumps(
            {
                "artifact_payloads": {
                    "operator_panel": {
                        "payload": {
                            "focus_slots": [
                                {
                                    "slot": "primary",
                                    "symbol": "XAUUSD",
                                    "action": "wait_for_paper_execution_close_evidence",
                                    "reason": "paper_execution_close_evidence_pending",
                                }
                            ]
                        }
                    },
                    "commodity_reasoning_summary": {
                        "payload": {
                            "primary_scenario_brief": "supply_chain_tightening",
                            "primary_chain_brief": "feedstock_cost_push_chain",
                            "contracts_in_focus": ["BU2606"],
                        }
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    expectations = mod.load_internal_terminal_focus_expectations(dist_dir=dist_dir)

    assert expectations["focus_row_id"] == "primary"
    assert expectations["focus_row_label"] == "主槽位"
    assert expectations["route_assertions"] == [
        {
            "route": "/terminal/internal?panel=signal-risk&section=focus-slots",
            "nav_label": "信号发生器与风险节流阀",
            "headline": "信号发生器与风险节流阀",
            "markers": [
                "信号发生器与风险节流阀",
                "穿透层 3 / 焦点槽位",
                "主槽位",
                "supply_chain_tightening",
                "feedstock_cost_push_chain",
                "BU2606",
            ],
        }
    ]


def test_build_internal_terminal_focus_smoke_spec_asserts_drilldown_focus_link_outside_summary(tmp_path: Path) -> None:
    mod = load_module()
    screenshot_path = tmp_path / "internal-terminal-focus-smoke.png"
    result_path = tmp_path / "internal-terminal-focus-smoke.json"

    spec = mod.build_internal_terminal_focus_smoke_spec(
        base_url="http://127.0.0.1:4173/",
        screenshot_path=screenshot_path,
        result_path=result_path,
        focus_row_id="primary",
        focus_row_label="主槽位",
        visible_markers=[
            "信号发生器与风险节流阀",
            "穿透层 3 / 焦点槽位",
            "主槽位",
            "国内商品推理线",
            "supply_chain_tightening",
            "feedstock_cost_push_chain",
            "BU2606",
        ],
    )

    assert "/terminal/internal?panel=signal-risk&section=focus-slots" in spec
    assert "信号发生器与风险节流阀" in spec
    assert "穿透层 3 / 焦点槽位" in spec
    assert "主槽位" in spec
    assert "国内商品推理线" in spec
    assert "supply_chain_tightening" in spec
    assert "feedstock_cost_push_chain" in spec
    assert "BU2606" in spec
    assert "summary.drill-card-summary .drill-card-link" in spec
    assert ".drill-card-summary .drill-card-link" in spec
    assert "expect(summaryLinkCount).toBeGreaterThanOrEqual(1)" in spec
    assert "定位此项" in spec
    assert "当前焦点" in spec
    assert "row=primary" in spec
    assert "for (const marker of VISIBLE_MARKERS)" in spec
    assert "expect(publicSnapshotRequests.length).toBe(0)" in spec
    assert "requested_surface: 'internal'" in spec
    assert "terminal_drilldown_assertion" in spec
    assert str(result_path) in spec


def test_build_artifact_payload_reports_internal_alignment_surface(tmp_path: Path) -> None:
    mod = load_module()
    report_path = tmp_path / "report.json"
    screenshot_path = tmp_path / "internal-alignment-smoke.png"

    payload = mod.build_artifact_payload(
        workspace=tmp_path,
        report_path=report_path,
        screenshot_path=screenshot_path,
        build_result={"returncode": 0},
        server_ready_seconds=0.24,
        smoke_result={
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "playwright_result": {
                "requested_surface": "internal",
                "effective_surface": "internal",
                "visited_routes": [
                    {
                        "route": "/workspace/alignment?view=internal&page_section=alignment-summary",
                        "headline": "方向对齐投射",
                    }
                ],
                "snapshot_requests": [],
                "internal_snapshot_requests": [
                    "http://127.0.0.1:4173/data/fenlie_dashboard_internal_snapshot.json?ts=1",
                ],
                "projection_assertion": {
                    "headline": "将高价值对话反馈投射到内部对齐页",
                    "top_event_headline": "拆分研究工作区左栏四控件",
                    "top_action": "先刷新 internal snapshot 并验证对齐页，再收紧 internal fallback 语义。",
                },
            },
        },
        base_url="http://127.0.0.1:4173/",
        mode="internal_alignment",
        expected_route_markers=[
            {
                "route": "/workspace/alignment?view=internal&page_section=alignment-summary",
                "nav_label": "对齐页",
                "headline": "方向对齐投射",
                "markers": [
                    "将高价值对话反馈投射到内部对齐页",
                    "拆分研究工作区左栏四控件",
                ],
            }
        ],
    )

    assert payload["action"] == "dashboard_internal_alignment_browser_smoke"
    assert payload["ok"] is True
    assert payload["surface_assertion"]["requested_surface"] == "internal"
    assert payload["surface_assertion"]["effective_surface"] == "internal"
    assert payload["surface_assertion"]["snapshot_endpoint_observed"] == "/data/fenlie_dashboard_internal_snapshot.json"
    assert payload["network_observation"]["public_snapshot_fetch_count"] == 0
    assert payload["network_observation"]["internal_snapshot_fetch_count"] == 1
    assert payload["projection_assertion"] == {
        "headline": "将高价值对话反馈投射到内部对齐页",
        "top_event_headline": "拆分研究工作区左栏四控件",
        "top_action": "先刷新 internal snapshot 并验证对齐页，再收紧 internal fallback 语义。",
    }
    assert payload["page_section_assertion"] == {
        "applicable": False,
        "route": "",
        "page_section": "",
        "active_label": "",
        "accordion_state": "",
    }
    assert payload["contracts_source_head_assertion"] == {
        "applicable": False,
        "route": "",
        "page_section": "",
        "source_head_id": "",
        "accordion_state": "",
        "visible_markers": [],
    }
    assert payload["contracts_source_gap_assertion"] == {
        "applicable": False,
        "route": "",
        "page_section": "",
        "visible_markers": [],
    }
    assert payload["artifacts_filter_assertion"] == {
        "applicable": False,
        "route": "",
        "group": "",
        "search_scope": "",
        "search": "",
        "source_available": False,
        "active_artifact": "",
        "visible_artifacts": [],
    }
    assert payload["artifacts_exit_risk_review_assertion"] == {
        "applicable": False,
        "route": "",
        "group": "",
        "search_scope": "",
        "search": "",
        "source_available": False,
        "section_label": "",
        "active_artifact": "",
        "visible_artifacts": [],
        "visible_markers": [],
    }
    assert payload["expected_route_markers"] == [
        {
            "route": "/workspace/alignment?view=internal&page_section=alignment-summary",
            "nav_label": "对齐页",
            "headline": "方向对齐投射",
            "markers": [
                "将高价值对话反馈投射到内部对齐页",
                "拆分研究工作区左栏四控件",
            ],
        }
    ]


def test_build_artifact_payload_reports_internal_terminal_focus_surface(tmp_path: Path) -> None:
    mod = load_module()
    report_path = tmp_path / "report.json"
    screenshot_path = tmp_path / "internal-terminal-focus-smoke.png"

    payload = mod.build_artifact_payload(
        workspace=tmp_path,
        report_path=report_path,
        screenshot_path=screenshot_path,
        build_result={"returncode": 0},
        server_ready_seconds=0.19,
        smoke_result={
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "playwright_result": {
                "requested_surface": "internal",
                "effective_surface": "internal",
                "visited_routes": [
                    {
                        "route": "/terminal/internal?panel=signal-risk&section=focus-slots",
                        "headline": "信号发生器与风险节流阀",
                    }
                ],
                "snapshot_requests": [],
                "internal_snapshot_requests": [
                    "http://127.0.0.1:4173/data/fenlie_dashboard_internal_snapshot.json?ts=1",
                ],
                "terminal_drilldown_assertion": {
                    "route": "/terminal/internal?panel=signal-risk&section=focus-slots",
                    "panel": "signal-risk",
                    "section": "focus-slots",
                    "focus_row_id": "primary",
                    "focus_row_label": "主槽位",
                    "summary_link_count": 1,
                    "summary_label_before_click": "定位此项",
                    "summary_label_after_click": "当前焦点",
                    "summary_link_href": "/terminal/internal?panel=signal-risk&section=focus-slots&row=primary",
                },
            },
        },
        base_url="http://127.0.0.1:4173/",
        mode="internal_terminal_focus",
        expected_route_markers=[
            {
                "route": "/terminal/internal?panel=signal-risk&section=focus-slots",
                "nav_label": "信号发生器与风险节流阀",
                "headline": "信号发生器与风险节流阀",
                "markers": [
                    "信号发生器与风险节流阀",
                    "穿透层 3 / 焦点槽位",
                    "主槽位",
                    "supply_chain_tightening",
                    "feedstock_cost_push_chain",
                    "BU2606",
                ],
            }
        ],
    )

    assert payload["action"] == "dashboard_internal_terminal_focus_browser_smoke"
    assert payload["ok"] is True
    assert payload["surface_assertion"]["requested_surface"] == "internal"
    assert payload["surface_assertion"]["effective_surface"] == "internal"
    assert payload["surface_assertion"]["snapshot_endpoint_observed"] == "/data/fenlie_dashboard_internal_snapshot.json"
    assert payload["network_observation"]["public_snapshot_fetch_count"] == 0
    assert payload["network_observation"]["internal_snapshot_fetch_count"] == 1
    assert payload["expected_focus_panel"] == "signal-risk"
    assert payload["expected_focus_section"] == "focus-slots"
    assert payload["page_section_assertion"] == {
        "applicable": False,
        "route": "",
        "page_section": "",
        "active_label": "",
        "accordion_state": "",
    }
    assert payload["contracts_source_head_assertion"] == {
        "applicable": False,
        "route": "",
        "page_section": "",
        "source_head_id": "",
        "accordion_state": "",
        "visible_markers": [],
    }
    assert payload["contracts_source_gap_assertion"] == {
        "applicable": False,
        "route": "",
        "page_section": "",
        "visible_markers": [],
    }
    assert payload["artifacts_filter_assertion"] == {
        "applicable": False,
        "route": "",
        "group": "",
        "search_scope": "",
        "search": "",
        "source_available": False,
        "active_artifact": "",
        "visible_artifacts": [],
    }
    assert payload["artifacts_exit_risk_review_assertion"] == {
        "applicable": False,
        "route": "",
        "group": "",
        "search_scope": "",
        "search": "",
        "source_available": False,
        "section_label": "",
        "active_artifact": "",
        "visible_artifacts": [],
        "visible_markers": [],
    }
    assert payload["terminal_drilldown_assertion"] == {
        "route": "/terminal/internal?panel=signal-risk&section=focus-slots",
        "panel": "signal-risk",
        "section": "focus-slots",
        "focus_row_id": "primary",
        "focus_row_label": "主槽位",
        "summary_link_count": 1,
        "summary_label_before_click": "定位此项",
        "summary_label_after_click": "当前焦点",
        "summary_link_href": "/terminal/internal?panel=signal-risk&section=focus-slots&row=primary",
    }
    assert payload["expected_route_markers"] == [
        {
            "route": "/terminal/internal?panel=signal-risk&section=focus-slots",
            "nav_label": "信号发生器与风险节流阀",
            "headline": "信号发生器与风险节流阀",
            "markers": [
                "信号发生器与风险节流阀",
                "穿透层 3 / 焦点槽位",
                "主槽位",
                "supply_chain_tightening",
                "feedstock_cost_push_chain",
                "BU2606",
            ],
        }
    ]


def test_build_artifact_payload_reports_internal_alignment_manual_probe_surface(tmp_path: Path) -> None:
    mod = load_module()
    report_path = tmp_path / "report.json"
    screenshot_path = tmp_path / "internal-alignment-manual-probe-smoke.png"

    payload = mod.build_artifact_payload(
        workspace=tmp_path,
        report_path=report_path,
        screenshot_path=screenshot_path,
        build_result={"returncode": 0},
        server_ready_seconds=0.21,
        smoke_result={
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "playwright_result": {
                "requested_surface": "internal",
                "effective_surface": "internal",
                "visited_routes": [
                    {
                        "route": "/workspace/alignment?view=internal&page_section=alignment-summary",
                        "headline": "方向对齐投射",
                    }
                ],
                "snapshot_requests": [],
                "internal_snapshot_requests": [
                    "http://127.0.0.1:4173/data/fenlie_dashboard_internal_snapshot.json?ts=1",
                ],
                "projection_assertion": {
                    "headline": "Manual probe headline",
                    "top_event_headline": "Manual probe headline",
                    "top_action": "Manual probe action",
                },
            },
        },
        base_url="http://127.0.0.1:4173/",
        mode="internal_alignment_manual_probe",
        expected_route_markers=[
            {
                "route": "/workspace/alignment?view=internal&page_section=alignment-summary",
                "nav_label": "对齐页",
                "headline": "方向对齐投射",
                "markers": [
                    "Manual probe headline",
                    "Manual probe action",
                ],
            }
        ],
    )

    assert payload["action"] == "dashboard_internal_alignment_manual_probe_browser_smoke"
    assert payload["surface_assertion"]["requested_surface"] == "internal"
    assert payload["surface_assertion"]["effective_surface"] == "internal"
    assert payload["network_observation"]["public_snapshot_fetch_count"] == 0
    assert payload["network_observation"]["internal_snapshot_fetch_count"] == 1
    assert payload["projection_assertion"] == {
        "headline": "Manual probe headline",
        "top_event_headline": "Manual probe headline",
        "top_action": "Manual probe action",
    }


def test_build_artifact_payload_keeps_failure_assertion_for_manual_probe_failures(tmp_path: Path) -> None:
    mod = load_module()
    report_path = tmp_path / "report.json"
    screenshot_path = tmp_path / "internal-alignment-manual-probe-smoke.png"

    payload = mod.build_artifact_payload(
        workspace=tmp_path,
        report_path=report_path,
        screenshot_path=screenshot_path,
        build_result={"returncode": 0},
        server_ready_seconds=0.0,
        smoke_result={
            "returncode": 1,
            "stdout": "",
            "stderr": "refresh_after_manual_probe_restore_failed",
            "playwright_result": {},
        },
        base_url="http://127.0.0.1:4173/",
        mode="internal_alignment_manual_probe",
        expected_route_markers=[
            {
                "route": "/workspace/alignment?view=internal&page_section=alignment-summary",
                "nav_label": "对齐页",
                "headline": "方向对齐投射",
                "markers": [],
            }
        ],
        failure_assertion={
            "failure_stage": "refresh_after_manual_probe_restore_failed",
            "failure_detail": "refresh_after_manual_probe_restore_failed",
            "probe_feedback_id": mod.MANUAL_PROBE_FEEDBACK_ID,
            "manual_probe_state": {
                "manual_file_exists": False,
                "manual_row_count": 0,
                "manual_probe_present": False,
            },
        },
        force_failed=True,
    )

    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["failure_assertion"] == {
        "failure_stage": "refresh_after_manual_probe_restore_failed",
        "failure_detail": "refresh_after_manual_probe_restore_failed",
        "probe_feedback_id": mod.MANUAL_PROBE_FEEDBACK_ID,
        "manual_probe_state": {
            "manual_file_exists": False,
            "manual_row_count": 0,
            "manual_probe_present": False,
        },
    }


def test_temporary_manual_probe_restores_manual_jsonl_after_context(tmp_path: Path, monkeypatch) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    manual_path = review_dir / "conversation_feedback_events_internal.jsonl"
    original_rows = [
        {
            "feedback_id": "existing_manual",
            "headline": "existing manual",
            "summary": "existing summary",
            "recommended_action": "existing action",
        }
    ]
    manual_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in original_rows) + "\n",
        encoding="utf-8",
    )

    calls: list[tuple[str, list[str], str]] = []

    def fake_ensure_success(*, name: str, cmd: list[str], cwd: Path):
        calls.append((name, cmd, str(cwd)))
        return {"name": name, "returncode": 0}

    monkeypatch.setattr(mod, "ensure_success", fake_ensure_success)
    monkeypatch.setattr(mod, "current_python_executable", lambda: "/tmp/fake-python")

    runtime_now = dt.datetime(2026, 3, 22, 7, 0, tzinfo=dt.timezone.utc)
    with mod.temporary_manual_probe(
        workspace=workspace,
        system_root=system_root,
        review_dir=review_dir,
        runtime_now=runtime_now,
    ) as probe_event:
        rows = [json.loads(line) for line in manual_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert [row["feedback_id"] for row in rows] == ["existing_manual", probe_event["feedback_id"]]
        assert probe_event["source"] == "manual"
        assert probe_event["created_at_utc"] == "2026-03-22T07:00:00Z"

    restored_rows = [json.loads(line) for line in manual_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["feedback_id"] for row in restored_rows] == ["existing_manual"]
    assert [name for name, _, _ in calls] == [
        "refresh_after_manual_probe_seed",
        "refresh_after_manual_probe_restore",
    ]
    assert calls[0][1][0] == "/tmp/fake-python"
    assert calls[1][1][0] == "/tmp/fake-python"


def test_http_server_uses_current_python_executable(tmp_path: Path, monkeypatch) -> None:
    mod = load_module()
    seen: dict[str, object] = {}

    class DummyProc:
        def terminate(self) -> None:
            seen["terminated"] = True

        def wait(self, timeout: float | None = None) -> None:
            seen["wait_timeout"] = timeout

    def fake_popen(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["cwd"] = str(kwargs.get("cwd"))
        return DummyProc()

    monkeypatch.setattr(mod, "current_python_executable", lambda: "/tmp/http-python")
    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

    with mod.http_server(dist_dir=tmp_path, host="127.0.0.1", port=4317):
        pass

    assert seen["cmd"] == [
        "/tmp/http-python",
        "-m",
        "http.server",
        "4317",
        "--bind",
        "127.0.0.1",
        "--directory",
        str(tmp_path),
    ]
    assert seen["cwd"] == str(tmp_path)
    assert seen["terminated"] is True
    assert seen["wait_timeout"] == 5


def test_temporary_manual_probe_restores_manual_jsonl_when_seed_refresh_fails(tmp_path: Path, monkeypatch) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    manual_path = review_dir / "conversation_feedback_events_internal.jsonl"
    original_text = json.dumps(
        {
            "feedback_id": "existing_manual",
            "headline": "existing manual",
            "summary": "existing summary",
            "recommended_action": "existing action",
        },
        ensure_ascii=False,
    ) + "\n"
    manual_path.write_text(original_text, encoding="utf-8")

    def fake_ensure_success(*, name: str, cmd: list[str], cwd: Path):
        raise RuntimeError(f"{name}_failed")

    monkeypatch.setattr(mod, "ensure_success", fake_ensure_success)

    runtime_now = dt.datetime(2026, 3, 22, 7, 5, tzinfo=dt.timezone.utc)
    try:
        with mod.temporary_manual_probe(
            workspace=workspace,
            system_root=system_root,
            review_dir=review_dir,
            runtime_now=runtime_now,
        ):
            raise AssertionError("should_not_enter_context")
    except RuntimeError as exc:
        assert str(exc) == "refresh_after_manual_probe_seed_failed"

    assert manual_path.read_text(encoding="utf-8") == original_text


def test_main_writes_failure_artifact_when_manual_probe_restore_fails(tmp_path: Path, monkeypatch) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    web_root = system_root / "dashboard" / "web"
    dist_dir = web_root / "dist"
    review_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "index.html").write_text("<!doctype html><title>ok</title>", encoding="utf-8")

    fixed_now = dt.datetime(2026, 3, 22, 7, 10, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(mod, "now_utc", lambda: fixed_now)
    monkeypatch.setattr(mod, "choose_port", lambda host: 4173)
    monkeypatch.setattr(mod, "wait_http_ready", lambda url, timeout_seconds: 0.12)
    monkeypatch.setattr(mod, "http_server", lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(
        mod,
        "load_internal_alignment_expectations",
        lambda dist_dir: {
            "headline": "Manual probe headline",
            "top_event_headline": "Manual probe headline",
            "top_action": "Manual probe action",
            "route_assertions": [
                {
                    "route": "/workspace/alignment?view=internal&page_section=alignment-summary",
                    "nav_label": "对齐页",
                    "headline": "方向对齐投射",
                    "markers": ["Manual probe headline", "Manual probe action"],
                }
            ],
        },
    )
    monkeypatch.setattr(
        mod,
        "run_playwright_smoke",
        lambda **kwargs: {
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "playwright_result": {
                "requested_surface": "internal",
                "effective_surface": "internal",
                "visited_routes": [
                    {
                        "route": "/workspace/alignment?view=internal&page_section=alignment-summary",
                        "headline": "方向对齐投射",
                    }
                ],
                "snapshot_requests": [],
                "internal_snapshot_requests": [
                    "http://127.0.0.1:4173/data/fenlie_dashboard_internal_snapshot.json?ts=1",
                ],
                "projection_assertion": {
                    "headline": "Manual probe headline",
                    "top_event_headline": "Manual probe headline",
                    "top_action": "Manual probe action",
                },
            },
        },
    )

    @contextlib.contextmanager
    def broken_probe(**kwargs):
        yield {
            "feedback_id": mod.MANUAL_PROBE_FEEDBACK_ID,
            "created_at_utc": "2026-03-22T07:10:00Z",
            "source": "manual",
        }
        raise RuntimeError("refresh_after_manual_probe_restore_failed")

    monkeypatch.setattr(mod, "temporary_manual_probe", broken_probe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dashboard_workspace_artifacts_smoke.py",
            "--workspace",
            str(workspace),
            "--skip-build",
            "--mode",
            "internal_alignment_manual_probe",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1

    report_path = review_dir / "20260322T071000Z_dashboard_internal_alignment_manual_probe_browser_smoke.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["action"] == "dashboard_internal_alignment_manual_probe_browser_smoke"
    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["failure_assertion"]["failure_stage"] == "refresh_after_manual_probe_restore_failed"
    assert payload["failure_assertion"]["manual_probe_state"] == {
        "manual_file_exists": False,
        "manual_row_count": 0,
        "manual_probe_present": False,
    }


def test_main_writes_failure_artifact_when_build_fails_before_smoke(tmp_path: Path, monkeypatch) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    web_root = system_root / "dashboard" / "web"
    review_dir.mkdir(parents=True, exist_ok=True)
    web_root.mkdir(parents=True, exist_ok=True)

    fixed_now = dt.datetime(2026, 3, 22, 7, 20, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(mod, "now_utc", lambda: fixed_now)

    def fake_ensure_success(*, name: str, cmd: list[str], cwd: Path):
        raise RuntimeError("dashboard_build_failed: mocked build failure")

    monkeypatch.setattr(mod, "ensure_success", fake_ensure_success)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dashboard_workspace_artifacts_smoke.py",
            "--workspace",
            str(workspace),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1

    report_path = review_dir / "20260322T072000Z_dashboard_workspace_routes_browser_smoke.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["action"] == "dashboard_workspace_routes_browser_smoke"
    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["failure_assertion"] == {
        "failure_stage": "dashboard_build_failed",
        "failure_detail": "dashboard_build_failed: mocked build failure",
    }


def test_main_writes_failure_artifact_when_choose_port_fails(tmp_path: Path, monkeypatch) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    web_root = system_root / "dashboard" / "web"
    dist_dir = web_root / "dist"
    review_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "index.html").write_text("<!doctype html><title>ok</title>", encoding="utf-8")

    fixed_now = dt.datetime(2026, 3, 22, 7, 25, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(mod, "now_utc", lambda: fixed_now)
    monkeypatch.setattr(mod, "choose_port", lambda host: (_ for _ in ()).throw(RuntimeError("choose_port_failed: mocked port allocation failure")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dashboard_workspace_artifacts_smoke.py",
            "--workspace",
            str(workspace),
            "--skip-build",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1

    report_path = review_dir / "20260322T072500Z_dashboard_workspace_routes_browser_smoke.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["action"] == "dashboard_workspace_routes_browser_smoke"
    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["failure_assertion"] == {
        "failure_stage": "choose_port_failed",
        "failure_detail": "choose_port_failed: mocked port allocation failure",
    }
