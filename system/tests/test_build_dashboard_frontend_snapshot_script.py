from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(
    "/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_dashboard_frontend_snapshot.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("build_dashboard_frontend_snapshot_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_public_snapshot(tmp_path: Path) -> dict:
    mod = load_module()
    workspace = tmp_path / "workspace"
    review_dir = workspace / "system" / "output" / "review"
    artifacts_dir = workspace / "system" / "output" / "artifacts"
    public_dir = workspace / "system" / "dashboard" / "web" / "public"
    review_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    public_dir.mkdir(parents=True)
    (public_dir / "data").mkdir(parents=True)

    breakout_path = review_dir / "20260319T103100Z_price_action_breakout_pullback_sim_only.json"
    write_json(
        breakout_path,
        {
            "action": "build_price_action_breakout_pullback_sim_only",
            "ok": True,
            "status": "ok",
            "change_class": "SIM_ONLY",
            "generated_at_utc": "2026-03-19T10:31:00Z",
            "recommended_brief": "ETHUSDT keep",
        },
    )
    anchor_path = review_dir / "2026-02-22_architecture_audit.json"
    write_json(
        anchor_path,
        {
            "status": "pass",
            "generated_at_utc": "2026-02-22T00:00:00Z",
        },
    )
    write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_hold_robustness_sim_only.json",
        {
            "action": "build_price_action_breakout_pullback_exit_hold_robustness_sim_only",
            "ok": True,
            "status": "ok",
            "change_class": "SIM_ONLY",
            "generated_at_utc": "2026-03-19T08:11:48Z",
        },
    )
    write_json(
        review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
        {
            "status": "ok",
            "change_class": "SIM_ONLY",
            "generated_at_utc": "2026-03-19T08:30:00Z",
            "research_decision": "use_hold_selection_gate_as_canonical_head",
            "active_baseline": "hold16_zero",
            "local_candidate": "hold8_zero",
            "transfer_watch": ["hold12_zero"],
            "return_candidate": ["hold24_zero"],
            "demoted_candidate": ["hold24_zero", "pullback_depth_atr_router"],
        },
    )
    write_json(
        review_dir / "20260315T105000Z_crypto_shortline_cross_section_backtest.json",
        {
            "status": "warning",
            "change_class": "SIM_ONLY",
            "generated_at_utc": "2026-03-15T10:50:00Z",
            "research_decision": "no_edge",
        },
    )
    event_regime_path = review_dir / "latest_event_regime_snapshot.json"
    write_json(
        event_regime_path,
        {
            "event_severity_score": 0.78,
            "systemic_risk_score": 0.63,
            "regime_state": "sector_stress",
            "top_risk_assets": ["BTC", "ETH", "GOLD"],
            "headline_drivers": ["credit_liquidity_stress"],
        },
    )
    event_analogy_path = review_dir / "latest_event_crisis_analogy.json"
    write_json(
        event_analogy_path,
        {
            "top_analogues": [
                {
                    "archetype_id": "gfc_2008",
                    "similarity_score": 0.82,
                    "match_axes": ["contagion", "credit"],
                    "mismatch_axes": ["policy"],
                },
            ],
        },
    )
    event_shock_map_path = review_dir / "latest_event_asset_shock_map.json"
    write_json(
        event_shock_map_path,
        {
            "assets": [
                {
                    "asset": "BTC",
                    "class": "crypto",
                    "shock_direction_bias": "down",
                    "expected_volatility_rank": "high",
                    "contagion_sensitivity": "strong",
                    "risk_1d": "elevated",
                    "risk_3d": "elevated",
                    "risk_7d": "elevated",
                }
            ],
        },
    )
    event_operator_summary_path = review_dir / "latest_event_crisis_operator_summary.json"
    write_json(
        event_operator_summary_path,
        {
            "status": "watch",
            "change_class": "RESEARCH_ONLY",
            "summary": "event crisis watch",
            "takeaway": "monitor private credit flow",
        },
    )
    event_regime_path = review_dir / "latest_event_regime_snapshot.json"
    write_json(
        event_regime_path,
        {
            "event_severity_score": 0.78,
            "systemic_risk_score": 0.63,
            "regime_state": "sector_stress",
            "top_risk_assets": ["BTC", "ETH", "GOLD"],
            "headline_drivers": ["credit_liquidity_stress"],
        },
    )
    event_analogy_path = review_dir / "latest_event_crisis_analogy.json"
    write_json(
        event_analogy_path,
        {
            "top_analogues": [
                {
                    "archetype_id": "gfc_2008",
                    "similarity_score": 0.82,
                    "match_axes": ["contagion", "credit"],
                    "mismatch_axes": ["policy"],
                },
            ],
        },
    )
    event_shock_map_path = review_dir / "latest_event_asset_shock_map.json"
    write_json(
        event_shock_map_path,
        {
            "assets": [
                {
                    "asset": "BTC",
                    "class": "crypto",
                    "shock_direction_bias": "down",
                    "expected_volatility_rank": "high",
                    "contagion_sensitivity": "strong",
                    "risk_1d": "elevated",
                    "risk_3d": "elevated",
                    "risk_7d": "elevated",
                }
            ],
        },
    )
    event_operator_summary_path = review_dir / "latest_event_crisis_operator_summary.json"
    write_json(
        event_operator_summary_path,
        {
            "status": "watch",
            "change_class": "RESEARCH_ONLY",
            "summary": "event crisis watch",
            "takeaway": "monitor private credit flow",
        },
    )

    mod.build_surface_snapshot(
        surface=mod.SurfaceSpec(
            key="public",
            output_name="fenlie_dashboard_snapshot.json",
            expose_absolute_paths=False,
            redaction_level="public_summary",
        ),
        workspace=workspace,
        public_dir=public_dir,
        review_dir=review_dir,
        artifacts_dir=artifacts_dir,
        selected_paths={
            "price_action_breakout_pullback": (breakout_path, "sim-only", "research_mainline"),
            "architecture_audit": (anchor_path, "orchestration", "system_anchor"),
            "price_action_exit_hold_robustness": (review_dir / "latest_price_action_breakout_pullback_exit_hold_robustness_sim_only.json", "sim-only", "research_exit_risk"),
            "hold_selection_handoff": (review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json", "research", "research_hold_transfer"),
        "cross_section_backtest": (review_dir / "20260315T105000Z_crypto_shortline_cross_section_backtest.json", "backtest", "research_cross_section"),
        "event_regime_snapshot": (event_regime_path, "research", "event_insight"),
        "event_crisis_analogy": (event_analogy_path, "research", "event_insight"),
        "event_asset_shock_map": (event_shock_map_path, "research", "event_insight"),
        "event_crisis_operator_summary": (event_operator_summary_path, "research", "event_insight"),
        },
        route_contract={"ui_routes": {}, "surface_contracts": {}, "experience_contract": {}},
        source_head_contract={
            "source_heads": [
                {
                    "id": "hold_selection_handoff",
                    "label": "持有选择主头",
                }
            ]
        },
        max_catalog=20,
        max_backtests=0,
        max_equity_points=20,
    )

    snapshot_path = public_dir / "data" / "fenlie_dashboard_snapshot.json"
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def build_dual_snapshots(tmp_path: Path, *, feedback_projection: dict | None = None) -> tuple[dict, dict]:
    mod = load_module()
    workspace = tmp_path / "workspace"
    review_dir = workspace / "system" / "output" / "review"
    artifacts_dir = workspace / "system" / "output" / "artifacts"
    public_dir = workspace / "system" / "dashboard" / "web" / "public"
    review_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    public_dir.mkdir(parents=True)
    (public_dir / "data").mkdir(parents=True)

    breakout_path = review_dir / "20260319T103100Z_price_action_breakout_pullback_sim_only.json"
    write_json(
        breakout_path,
        {
            "action": "build_price_action_breakout_pullback_sim_only",
            "ok": True,
            "status": "ok",
            "change_class": "SIM_ONLY",
            "generated_at_utc": "2026-03-19T10:31:00Z",
            "recommended_brief": "ETHUSDT keep",
        },
    )
    if feedback_projection is not None:
        write_json(
            review_dir / "latest_conversation_feedback_projection_internal.json",
            feedback_projection,
        )

    selected_paths = {
        "price_action_breakout_pullback": (breakout_path, "sim-only", "research_mainline"),
    }
    route_contract = {"ui_routes": {}, "surface_contracts": {}, "experience_contract": {}}
    source_head_contract = {"source_heads": []}
    for surface in (
        mod.SurfaceSpec(
            key="public",
            output_name="fenlie_dashboard_snapshot.json",
            expose_absolute_paths=False,
            redaction_level="public_summary",
        ),
        mod.SurfaceSpec(
            key="internal",
            output_name="fenlie_dashboard_internal_snapshot.json",
            expose_absolute_paths=True,
            redaction_level="full_internal",
        ),
    ):
        mod.build_surface_snapshot(
            surface=surface,
            workspace=workspace,
            public_dir=public_dir,
            review_dir=review_dir,
            artifacts_dir=artifacts_dir,
            selected_paths=selected_paths,
            route_contract=route_contract,
            source_head_contract=source_head_contract,
            max_catalog=20,
            max_backtests=0,
            max_equity_points=20,
        )
    public_snapshot = json.loads((public_dir / "data" / "fenlie_dashboard_snapshot.json").read_text(encoding="utf-8"))
    internal_snapshot = json.loads((public_dir / "data" / "fenlie_dashboard_internal_snapshot.json").read_text(encoding="utf-8"))
    return public_snapshot, internal_snapshot


def test_selected_artifact_uses_alias_as_catalog_id(tmp_path: Path) -> None:
    snapshot = build_public_snapshot(tmp_path)

    breakout_row = next(row for row in snapshot["catalog"] if row.get("payload_key") == "price_action_breakout_pullback")
    assert breakout_row["id"] == "price_action_breakout_pullback"
    assert breakout_row["path"] == "review/20260319T103100Z_price_action_breakout_pullback_sim_only.json"
    assert breakout_row["artifact_layer"] == "canonical"
    assert breakout_row["artifact_group"] == "research_mainline"

    anchor_row = next(row for row in snapshot["catalog"] if row.get("payload_key") == "architecture_audit")
    assert anchor_row["artifact_group"] == "system_anchor"

    exit_risk_row = next(row for row in snapshot["catalog"] if row.get("payload_key") == "price_action_exit_hold_robustness")
    assert exit_risk_row["artifact_group"] == "research_exit_risk"

    hold_transfer_row = next(row for row in snapshot["catalog"] if row.get("payload_key") == "hold_selection_handoff")
    assert hold_transfer_row["artifact_group"] == "research_hold_transfer"

    cross_section_row = next(row for row in snapshot["catalog"] if row.get("payload_key") == "cross_section_backtest")
    assert cross_section_row["artifact_group"] == "research_cross_section"


def test_event_crisis_artifacts_exposed(tmp_path: Path) -> None:
    snapshot = build_public_snapshot(tmp_path)

    payloads = snapshot["artifact_payloads"]
    assert "event_regime_snapshot" in payloads
    assert "event_crisis_analogy" in payloads
    assert "event_asset_shock_map" in payloads
    assert "event_crisis_operator_summary" in payloads


def test_public_catalog_dedupes_selected_artifact_against_same_review_file(tmp_path: Path) -> None:
    snapshot = build_public_snapshot(tmp_path)

    matching_rows = [
        row
        for row in snapshot["catalog"]
        if row.get("path") == "review/20260319T103100Z_price_action_breakout_pullback_sim_only.json"
    ]
    assert len(matching_rows) == 1
    assert matching_rows[0]["id"] == "price_action_breakout_pullback"

    hold_robustness_rows = [
        row
        for row in snapshot["catalog"]
        if row.get("path") == "review/latest_price_action_breakout_pullback_exit_hold_robustness_sim_only.json"
    ]
    assert len(hold_robustness_rows) == 1
    assert hold_robustness_rows[0]["id"] == "price_action_exit_hold_robustness"
    assert hold_robustness_rows[0]["artifact_layer"] == "canonical"
    assert hold_robustness_rows[0]["artifact_group"] == "research_exit_risk"


def test_snapshot_emits_source_owned_workspace_default_focus(tmp_path: Path) -> None:
    snapshot = build_public_snapshot(tmp_path)

    assert snapshot["workspace_default_focus"] == {
        "artifact": "price_action_breakout_pullback",
        "group": "research_mainline",
        "search_scope": "title",
        "panel": "lab-review",
        "section": "research-heads",
    }


def test_snapshot_source_heads_include_return_candidate_from_handoff(tmp_path: Path) -> None:
    snapshot = build_public_snapshot(tmp_path)

    source_head = snapshot["source_heads"]["hold_selection_handoff"]
    assert source_head["active_baseline"] == "hold16_zero"
    assert source_head["local_candidate"] == "hold8_zero"
    assert source_head["transfer_watch"] == ["hold12_zero"]
    assert source_head["return_candidate"] == ["hold24_zero"]


def test_snapshot_preserves_public_topology_contract(tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    review_dir = workspace / "system" / "output" / "review"
    artifacts_dir = workspace / "system" / "output" / "artifacts"
    public_dir = workspace / "system" / "dashboard" / "web" / "public"
    review_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    public_dir.mkdir(parents=True)
    (public_dir / "data").mkdir(parents=True)

    breakout_path = review_dir / "20260319T103100Z_price_action_breakout_pullback_sim_only.json"
    write_json(
        breakout_path,
        {
            "status": "ok",
            "generated_at_utc": "2026-03-19T10:31:00Z",
        },
    )

    mod.build_surface_snapshot(
        surface=mod.SurfaceSpec(
            key="public",
            output_name="fenlie_dashboard_snapshot.json",
            expose_absolute_paths=False,
            redaction_level="public_summary",
        ),
        workspace=workspace,
        public_dir=public_dir,
        review_dir=review_dir,
        artifacts_dir=artifacts_dir,
        selected_paths={
            "price_action_breakout_pullback": (breakout_path, "sim-only", "research_mainline"),
        },
        route_contract={
            "ui_routes": {
                "frontend_public": "https://fuuu.fun",
                "frontend_pages": "https://fenlie.fuuu.fun",
            },
            "surface_contracts": {},
            "experience_contract": {},
            "public_topology": [
                {
                    "id": "fenlie_root",
                    "label": "Fenlie 主入口",
                    "url_key": "frontend_public",
                    "expected_status": "200 / root-nav-proxy",
                },
                {
                    "id": "fenlie_pages",
                    "label": "Pages 回退入口",
                    "url_key": "frontend_pages",
                    "expected_status": "200 / pages origin",
                },
            ],
        },
        source_head_contract={"source_heads": []},
        max_catalog=20,
        max_backtests=0,
        max_equity_points=20,
    )

    snapshot_path = public_dir / "data" / "fenlie_dashboard_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["public_topology"][0]["id"] == "fenlie_root"
    assert snapshot["public_topology"][0]["url"] == "https://fuuu.fun"


def test_internal_snapshot_includes_feedback_projection_only_for_internal_surface(tmp_path: Path) -> None:
    public_snapshot, internal_snapshot = build_dual_snapshots(
        tmp_path,
        feedback_projection={
            "status": "ok",
            "change_class": "RESEARCH_ONLY",
            "visibility": "internal_only",
            "summary": {
                "headline": "keep dashboard aligned with operator intent",
                "alignment_score": 72,
                "blocker_pressure": 24,
                "execution_clarity": 68,
                "readability_pressure": 31,
                "drift_state": "watch",
            },
            "events": [
                {
                    "feedback_id": "ui_density_split",
                    "headline": "split left rail into four controls",
                    "summary": "research map / radar / search / focus",
                    "impact_score": 92,
                    "status": "active",
                }
            ],
            "actions": [
                {
                    "feedback_id": "ui_density_split",
                    "recommended_action": "split left rail structure",
                }
            ],
            "trends": [],
            "anchors": [],
        },
    )

    assert "conversation_feedback_projection" not in public_snapshot
    assert internal_snapshot["conversation_feedback_projection"]["visibility"] == "internal_only"
    assert internal_snapshot["conversation_feedback_projection"]["summary"]["headline"] == "keep dashboard aligned with operator intent"


def test_internal_snapshot_sets_feedback_projection_null_when_missing(tmp_path: Path) -> None:
    public_snapshot, internal_snapshot = build_dual_snapshots(tmp_path)

    assert "conversation_feedback_projection" not in public_snapshot
    assert "conversation_feedback_projection" in internal_snapshot
    assert internal_snapshot["conversation_feedback_projection"] is None


def test_repo_route_contract_uses_custom_pages_fallback_domain() -> None:
    route_contract_path = Path(
        "/Users/jokenrobot/Downloads/Folders/fenlie/system/config/dashboard_snapshot_ui_routes.json"
    )
    route_contract = json.loads(route_contract_path.read_text(encoding="utf-8"))
    ui_routes = route_contract["ui_routes"]

    assert ui_routes["frontend_public"] == "https://fuuu.fun"
    assert ui_routes["frontend_pages"] == "https://fenlie.fuuu.fun"


def test_artifact_selection_includes_public_acceptance_review() -> None:
    artifact_contract_path = Path(
        "/Users/jokenrobot/Downloads/Folders/fenlie/system/config/dashboard_snapshot_artifact_selection.json"
    )
    artifact_contract = json.loads(artifact_contract_path.read_text(encoding="utf-8"))
    item = next(
        row for row in artifact_contract["artifact_selection"] if row.get("id") == "dashboard_public_acceptance"
    )

    assert item["path_mode"] == "latest_review_suffix"
    assert item["value"] == "dashboard_public_acceptance.json"


def test_artifact_selection_includes_orderflow_blueprint_and_gate_blocker() -> None:
    artifact_contract_path = Path(
        "/Users/jokenrobot/Downloads/Folders/fenlie/system/config/dashboard_snapshot_artifact_selection.json"
    )
    artifact_contract = json.loads(artifact_contract_path.read_text(encoding="utf-8"))
    by_id = {row.get("id"): row for row in artifact_contract["artifact_selection"]}

    blueprint = by_id["intraday_orderflow_blueprint"]
    blocker = by_id["intraday_orderflow_research_gate_blocker"]

    assert blueprint["artifact_group"] == "research_cross_section"
    assert blueprint["category"] == "research"
    assert blueprint["path_mode"] == "review_path"
    assert blueprint["value"] == "latest_intraday_orderflow_blueprint.json"

    assert blocker["artifact_group"] == "research_cross_section"
    assert blocker["category"] == "research"
    assert blocker["path_mode"] == "review_path"
    assert blocker["value"] == "latest_intraday_orderflow_research_gate_blocker_report.json"
