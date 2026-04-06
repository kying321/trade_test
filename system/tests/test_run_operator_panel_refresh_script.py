from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_operator_panel_refresh.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_operator_panel_refresh_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_json_rejects_non_mapping_payload(monkeypatch) -> None:
    mod = load_module()

    class DummyProc:
        returncode = 0
        stdout = '["not-a-mapping"]\n'
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: DummyProc())

    with pytest.raises(RuntimeError, match="demo_invalid_payload"):
        mod.run_json(name="demo", cmd=["python3", "fake.py"])


def test_event_summary_has_geostrategy_fields_rejects_stale_payload(tmp_path: Path) -> None:
    mod = load_module()
    stale = tmp_path / "latest_event_crisis_operator_summary.json"
    stale.write_text(
        json.dumps(
            {
                "event_crisis_primary_theater_brief": None,
                "event_crisis_dominant_chain_brief": "",
                "event_crisis_safety_margin_brief": None,
                "event_crisis_hard_boundary_brief": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert mod.event_summary_has_geostrategy_fields(stale) is False

    stale.write_text(
        json.dumps(
            {
                "event_crisis_primary_theater_brief": "usd_liquidity_and_sanctions",
                "event_crisis_dominant_chain_brief": "credit_intermediary_chain",
                "event_crisis_safety_margin_brief": "system_margin=0.42",
                "event_crisis_hard_boundary_brief": "new_risk_hard_block",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert mod.event_summary_has_geostrategy_fields(stale) is True


def test_main_refreshes_panel_and_snapshot_into_public_and_dist(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    dashboard_root = system_root / "dashboard" / "web"
    public_dir = dashboard_root / "public"
    dist_dir = dashboard_root / "dist"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    seen_cmds: dict[str, list[str]] = {}

    def fake_run_json(*, name: str, cmd: list[str]) -> dict:
        seen_cmds[name] = list(cmd)
        public_dir.mkdir(parents=True, exist_ok=True)
        (public_dir / "data").mkdir(parents=True, exist_ok=True)

        if name == "run_event_crisis_pipeline":
            (review_dir / "latest_event_crisis_operator_summary.json").write_text(
                json.dumps(
                    {
                        "event_crisis_primary_theater_brief": "usd_liquidity_and_sanctions",
                        "event_crisis_dominant_chain_brief": "credit_intermediary_chain",
                        "event_crisis_safety_margin_brief": "system_margin=0.42",
                        "event_crisis_hard_boundary_brief": "new_risk_hard_block",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return {"artifacts": {"operator_summary": str(review_dir / "latest_event_crisis_operator_summary.json")}}
        if name == "run_external_intelligence_refresh":
            return {
                "status": "partial",
                "artifact_json": str(review_dir / "latest_external_intelligence_refresh.json"),
                "external_intelligence_path": str(review_dir / "latest_external_intelligence_snapshot.json"),
                "recommended_brief": "sources=1 | calendar=0 | flash=0 | quotes=0 | news=10",
                "takeaway": "NBA takes bids on European league, eyes 2027",
                "snapshot_skipped": True,
            }
        if name == "build_operator_task_visual_panel":
            (public_dir / "operator_task_visual_panel.html").write_text(
                "<html><body>panel</body></html>\n",
                encoding="utf-8",
            )
            (public_dir / "operator_task_visual_panel_data.json").write_text(
                json.dumps({"summary": {"operator_head_brief": "ETH"}}) + "\n",
                encoding="utf-8",
            )
            return {
                "artifact": str(review_dir / "20260321T084000Z_operator_task_visual_panel.json"),
                "html": str(review_dir / "20260321T084000Z_operator_task_visual_panel.html"),
            "summary": {
                "operator_head_brief": "ETHUSDT:wait_for_pullback",
                "review_head_brief": "review:hold16_anchor",
                "repair_head_brief": "repair:none",
                "remote_live_gate_brief": "remote_live:not_applicable",
                "lane_state_brief": "lanes:stable",
                "lane_priority_order_brief": "ETHUSDT>BNBUSDT",
                "action_queue_brief": "queue:empty",
                "crypto_refresh_reuse_brief": "crypto_refresh:reuse_ok",
                "remote_live_history_brief": "remote_history:n/a",
                "brooks_refresh_brief": "brooks:ok",
                "event_crisis_primary_theater_brief": "usd_liquidity_and_sanctions",
                "event_crisis_dominant_chain_brief": "credit_intermediary_chain",
                "event_crisis_safety_margin_brief": "system_margin=0.42",
                "event_crisis_hard_boundary_brief": "new_risk_hard_block",
                "commodity_reasoning_primary_scenario_brief": "supply_chain_tightening",
                "commodity_reasoning_primary_chain_brief": "feedstock_cost_push_chain",
                "commodity_reasoning_range_scope_brief": "contract_focused",
                "commodity_reasoning_boundary_strength_brief": "tight",
                "commodity_reasoning_invalidator_brief": "basis_weak",
            },
            }
        if name == "build_conversation_feedback_projection_internal":
            return {
                "artifact": str(review_dir / "20260321T084000Z_conversation_feedback_projection_internal.json"),
                "latest_artifact": str(review_dir / "latest_conversation_feedback_projection_internal.json"),
                "status": "ok",
            }
        if name == "build_cpa_control_plane_snapshot":
            (public_dir / "data" / "cpa_control_plane_snapshot.json").write_text(
                json.dumps({"summary": {"historical_success_total": 20}}) + "\n",
                encoding="utf-8",
            )
            return {
                "artifact_json": str(review_dir / "latest_cpa_control_plane_snapshot.json"),
                "public_path": str(public_dir / "data" / "cpa_control_plane_snapshot.json"),
                "status": "ok",
            }
        if name == "build_dashboard_frontend_snapshot":
            (public_dir / "data" / "fenlie_dashboard_snapshot.json").write_text(
                json.dumps({"surface": "public"}) + "\n",
                encoding="utf-8",
            )
            (public_dir / "data" / "fenlie_dashboard_internal_snapshot.json").write_text(
                json.dumps({"surface": "internal"}) + "\n",
                encoding="utf-8",
            )
            return {
                "outputs": [
                    str(public_dir / "data" / "fenlie_dashboard_snapshot.json"),
                    str(public_dir / "data" / "fenlie_dashboard_internal_snapshot.json"),
                ]
            }
        raise AssertionError(name)

    monkeypatch.setattr(mod, "run_json", fake_run_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_operator_panel_refresh.py",
            "--workspace",
            str(workspace),
            "--now",
            "2026-03-21T08:40:00Z",
        ],
    )

    mod.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["mode"] == "operator_panel_refresh"
    assert payload["operator_head_brief"] == "ETHUSDT:wait_for_pullback"
    assert payload["review_head_brief"] == "review:hold16_anchor"
    assert payload["lane_priority_order_brief"] == "ETHUSDT>BNBUSDT"
    assert payload["event_crisis_primary_theater_brief"] == "usd_liquidity_and_sanctions"
    assert payload["event_crisis_dominant_chain_brief"] == "credit_intermediary_chain"
    assert payload["event_crisis_safety_margin_brief"] == "system_margin=0.42"
    assert payload["event_crisis_hard_boundary_brief"] == "new_risk_hard_block"
    assert payload["external_intelligence_status"] == "partial"
    assert payload["external_intelligence_refresh_artifact"] == str(review_dir / "latest_external_intelligence_refresh.json")
    assert payload["external_intelligence_snapshot_artifact"] == str(review_dir / "latest_external_intelligence_snapshot.json")
    assert payload["external_intelligence_recommended_brief"] == "sources=1 | calendar=0 | flash=0 | quotes=0 | news=10"
    assert payload["external_intelligence_takeaway"] == "NBA takes bids on European league, eyes 2027"
    assert payload["cpa_control_plane_status"] == "ok"
    assert payload["cpa_control_plane_artifact"] == str(review_dir / "latest_cpa_control_plane_snapshot.json")
    assert payload["cpa_control_plane_public_path"] == str(public_dir / "data" / "cpa_control_plane_snapshot.json")
    assert payload["commodity_reasoning_primary_scenario_brief"] == "supply_chain_tightening"
    assert payload["commodity_reasoning_primary_chain_brief"] == "feedstock_cost_push_chain"
    assert payload["commodity_reasoning_range_scope_brief"] == "contract_focused"
    assert payload["commodity_reasoning_boundary_strength_brief"] == "tight"
    assert payload["commodity_reasoning_invalidator_brief"] == "basis_weak"
    for removed_key in (
        "event_crisis_regime_brief",
        "event_crisis_top_analogue_brief",
        "event_crisis_watch_assets_brief",
        "event_crisis_guard_brief",
    ):
        assert removed_key not in payload
    assert payload["snapshot_outputs"] == [
        str(public_dir / "data" / "fenlie_dashboard_snapshot.json"),
        str(public_dir / "data" / "fenlie_dashboard_internal_snapshot.json"),
    ]
    assert Path(payload["panel_dist_html"]).read_text(encoding="utf-8") == "<html><body>panel</body></html>\n"
    assert json.loads(Path(payload["panel_dist_json"]).read_text(encoding="utf-8")) == {
        "summary": {"operator_head_brief": "ETH"}
    }
    assert json.loads(Path(payload["snapshot_dist"]).read_text(encoding="utf-8")) == {"surface": "public"}
    assert json.loads(Path(payload["snapshot_internal_dist"]).read_text(encoding="utf-8")) == {
        "surface": "internal"
    }
    assert all(row["copied"] is True for row in payload["sync_results"])
    assert seen_cmds["build_operator_task_visual_panel"] == [
        "python3",
        str(system_root / "scripts" / "build_operator_task_visual_panel.py"),
        "--review-dir",
        str(review_dir),
        "--dashboard-dist",
        str(public_dir),
        "--now",
        "2026-03-21T08:40:00Z",
    ]
    assert seen_cmds["run_external_intelligence_refresh"] == [
        "python3",
        str(system_root / "scripts" / "run_external_intelligence_refresh.py"),
        "--workspace",
        str(workspace),
        "--public-dir",
        str(public_dir),
        "--now",
        "2026-03-21T08:40:00Z",
        "--skip-dashboard-snapshot",
    ]
    assert seen_cmds["build_dashboard_frontend_snapshot"] == [
        "python3",
        str(system_root / "scripts" / "build_dashboard_frontend_snapshot.py"),
        "--workspace",
        str(workspace),
        "--public-dir",
        str(public_dir),
    ]
    assert seen_cmds["build_conversation_feedback_projection_internal"] == [
        "python3",
        str(system_root / "scripts" / "build_conversation_feedback_projection_internal.py"),
        "--review-dir",
        str(review_dir),
        "--now",
        "2026-03-21T08:40:00Z",
    ]
    assert seen_cmds["build_cpa_control_plane_snapshot"] == [
        "python3",
        str(system_root / "scripts" / "build_cpa_control_plane_snapshot.py"),
        "--workspace",
        str(workspace),
        "--public-dir",
        str(public_dir),
    ]
    assert seen_cmds["run_event_crisis_pipeline"] == [
        "python3",
        str(system_root / "scripts" / "run_event_crisis_pipeline.py"),
        "--mode",
        "snapshot",
        "--output-root",
        str(system_root / "output"),
        "--now",
        "2026-03-21T08:40:00Z",
    ]


def test_build_summary_does_not_report_degraded_operator_head_when_full_source_panel_exists(tmp_path: Path) -> None:
    mod = load_module()
    payload = mod.build_summary(
        workspace=tmp_path,
        public_dir=tmp_path / "public",
        dist_dir=tmp_path / "dist",
        external_refresh_payload={
            "status": "partial",
            "artifact_json": str(tmp_path / "latest_external_intelligence_refresh.json"),
            "external_intelligence_path": str(tmp_path / "latest_external_intelligence_snapshot.json"),
            "recommended_brief": "sources=1 | calendar=0 | flash=0 | quotes=0 | news=10",
            "takeaway": "NBA takes bids on European league, eyes 2027",
        },
        cpa_control_payload={
            "status": "ok",
            "artifact_json": str(tmp_path / "latest_cpa_control_plane_snapshot.json"),
            "public_path": str(tmp_path / "public" / "data" / "cpa_control_plane_snapshot.json"),
        },
        panel_payload={
            "summary": {
                "operator_head_brief": "review:brooks_structure:SC2603:review_manual_stop_entry:92",
                "review_head_brief": "review:brooks_structure:SC2603:review_manual_stop_entry:92",
                "repair_head_brief": "inactive:-",
                "remote_live_gate_brief": "current_head_outside_remote_live_scope:brooks_structure:SC2603:scope_unknown",
                "lane_state_brief": "waiting=0 | review=1 | watch=0 | blocked=0 | repair=0",
                "lane_priority_order_brief": "review@92:1 > waiting@0:0 > watch@0:0 > blocked@0:0 > repair@0:0",
                "action_queue_brief": "1:brooks_structure:SC2603:review_manual_stop_entry",
                "brooks_refresh_brief": "ready:SC2603:second_entry_trend_continuation:manual_structure_review_now",
                "event_crisis_primary_theater_brief": "usd_liquidity_and_sanctions",
                "event_crisis_dominant_chain_brief": "usd_liquidity_chain",
                "event_crisis_safety_margin_brief": "system_margin=0.627433",
                "event_crisis_hard_boundary_brief": "none",
            }
        },
        snapshot_payload={"outputs": []},
        feedback_payload={},
        sync_results=[],
    )
    assert payload["operator_head_brief"] == "review:brooks_structure:SC2603:review_manual_stop_entry:92"
    assert not payload["operator_head_brief"].startswith("degraded:")
    assert payload["external_intelligence_status"] == "partial"


def test_main_degrades_when_external_intelligence_refresh_fails(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    dashboard_root = system_root / "dashboard" / "web"
    public_dir = dashboard_root / "public"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    def fake_run_json(*, name: str, cmd: list[str]) -> dict:
        public_dir.mkdir(parents=True, exist_ok=True)
        (public_dir / "data").mkdir(parents=True, exist_ok=True)
        if name == "run_event_crisis_pipeline":
            (review_dir / "latest_event_crisis_operator_summary.json").write_text(
                json.dumps(
                    {
                        "event_crisis_primary_theater_brief": "usd_liquidity_and_sanctions",
                        "event_crisis_dominant_chain_brief": "credit_intermediary_chain",
                        "event_crisis_safety_margin_brief": "system_margin=0.42",
                        "event_crisis_hard_boundary_brief": "new_risk_hard_block",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return {"artifacts": {"operator_summary": str(review_dir / "latest_event_crisis_operator_summary.json")}}
        if name == "run_external_intelligence_refresh":
            raise RuntimeError("run_external_intelligence_refresh_failed: axios timeout")
        if name == "build_operator_task_visual_panel":
            (public_dir / "operator_task_visual_panel.html").write_text("<html><body>panel</body></html>\n", encoding="utf-8")
            (public_dir / "operator_task_visual_panel_data.json").write_text(
                json.dumps({"summary": {"operator_head_brief": "ETH"}}) + "\n",
                encoding="utf-8",
            )
            return {
                "artifact": str(review_dir / "20260321T084000Z_operator_task_visual_panel.json"),
                "html": str(review_dir / "20260321T084000Z_operator_task_visual_panel.html"),
                "summary": {"operator_head_brief": "ETHUSDT:wait_for_pullback"},
            }
        if name == "build_conversation_feedback_projection_internal":
            return {"artifact": str(review_dir / "feedback.json"), "latest_artifact": str(review_dir / "latest_feedback.json")}
        if name == "build_cpa_control_plane_snapshot":
            (public_dir / "data" / "cpa_control_plane_snapshot.json").write_text(
                json.dumps({"summary": {"historical_success_total": 20}}) + "\n",
                encoding="utf-8",
            )
            return {
                "status": "ok",
                "artifact_json": str(review_dir / "latest_cpa_control_plane_snapshot.json"),
                "public_path": str(public_dir / "data" / "cpa_control_plane_snapshot.json"),
            }
        if name == "build_dashboard_frontend_snapshot":
            (public_dir / "data" / "fenlie_dashboard_snapshot.json").write_text(json.dumps({"surface": "public"}) + "\n", encoding="utf-8")
            (public_dir / "data" / "fenlie_dashboard_internal_snapshot.json").write_text(json.dumps({"surface": "internal"}) + "\n", encoding="utf-8")
            return {"outputs": [str(public_dir / "data" / "fenlie_dashboard_snapshot.json")]}
        raise AssertionError(name)

    monkeypatch.setattr(mod, "run_json", fake_run_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_operator_panel_refresh.py",
            "--workspace",
            str(workspace),
            "--now",
            "2026-03-21T08:40:00Z",
        ],
    )

    mod.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["external_intelligence_status"] == "degraded_request_failed"
    assert payload["external_intelligence_takeaway"] == "axios timeout"
    assert payload["cpa_control_plane_status"] == "ok"
    assert payload["snapshot_outputs"] == [str(public_dir / "data" / "fenlie_dashboard_snapshot.json")]
