from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "refresh_commodity_paper_execution_state.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("commodity_refresh_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step_now_is_monotonic() -> None:
    mod = _load_module()
    base = mod.parse_now("2026-03-11T08:40:00Z")
    assert mod.step_now(base, 0).isoformat() == "2026-03-11T08:40:00+00:00"
    assert mod.step_now(base, 3).isoformat() == "2026-03-11T08:40:03+00:00"
    assert mod.step_now(base, 5) > mod.step_now(base, 4)


def test_write_hot_brief_snapshot_persists_refresh_owned_copy(tmp_path: Path) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)
    source_path = review_dir / "20260316T090525Z_hot_universe_operator_brief.json"
    source_text = json.dumps(
        {"status": "ok", "artifact": str(source_path), "operator_status": "ok"},
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    source_path.write_text(source_text, encoding="utf-8")

    snapshot_path = mod.write_hot_brief_snapshot(
        review_dir,
        stamp="20260316T090525Z",
        brief_payload={"artifact": str(source_path), "operator_status": "ok"},
    )

    assert snapshot_path.name == "20260316T090525Z_commodity_paper_execution_refresh_hot_brief_snapshot.json"
    assert snapshot_path.read_text(encoding="utf-8") == source_text


def test_main_rebuilds_ticket_to_queue_chain_before_bridge_refresh(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    context_path = review_dir / "NEXT_WINDOW_CONTEXT_LATEST.md"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    calls: list[tuple[str, list[str]]] = []

    def fake_run_json_step(*, step_name: str, cmd: list[str]) -> dict:
        calls.append((step_name, list(cmd)))
        as_of = cmd[cmd.index("--now") + 1] if "--now" in cmd else "2026-03-16T09:05:20Z"
        artifact_name = {
            "ticket_lane_refresh": "commodity_paper_ticket_lane.json",
            "ticket_book_refresh": "commodity_paper_ticket_book.json",
            "execution_preview_refresh": "commodity_paper_execution_preview.json",
            "execution_artifact_refresh": "commodity_paper_execution_artifact.json",
            "execution_queue_refresh": "commodity_paper_execution_queue.json",
            "bridge_refresh": "commodity_paper_execution_bridge.json",
            "review_refresh": "commodity_paper_execution_review.json",
            "retro_refresh": "commodity_paper_execution_retro.json",
            "gap_refresh": "commodity_paper_execution_gap_report.json",
            "brief_refresh": "hot_universe_operator_brief.json",
        }[step_name]
        payload = {
            "ok": True,
            "status": "ok",
            "as_of": as_of,
            "artifact": str(review_dir / artifact_name),
        }
        if step_name == "ticket_lane_refresh":
            payload.update(
                {
                    "ticket_status": "paper-ready",
                    "next_ticket_batch": "asphalt_cn",
                    "next_ticket_symbols": ["BU2606"],
                }
            )
        elif step_name == "ticket_book_refresh":
            payload.update(
                {
                    "ticket_book_status": "paper-ready",
                    "next_ticket_id": "commodity-paper-ticket:asphalt_cn:BU2606",
                    "next_ticket_symbol": "BU2606",
                }
            )
        elif step_name == "execution_preview_refresh":
            payload.update(
                {
                    "execution_preview_status": "paper-execution-ready",
                    "next_execution_batch": "asphalt_cn",
                    "next_execution_symbols": ["BU2606"],
                    "next_execution_ticket_ids": ["commodity-paper-ticket:asphalt_cn:BU2606"],
                }
            )
        elif step_name == "execution_artifact_refresh":
            payload.update(
                {
                    "execution_artifact_status": "paper-execution-artifact-ready",
                    "execution_batch": "asphalt_cn",
                    "execution_symbols": ["BU2606"],
                }
            )
        elif step_name == "execution_queue_refresh":
            payload.update(
                {
                    "execution_queue_status": "paper-execution-queued",
                    "execution_batch": "asphalt_cn",
                    "execution_symbols": ["BU2606"],
                    "next_execution_symbol": "BU2606",
                }
            )
        elif step_name == "bridge_refresh":
            payload.update({"bridge_status": "bridge_empty"})
        elif step_name == "review_refresh":
            payload.update({"execution_review_status": "paper-execution-review-empty"})
        elif step_name == "retro_refresh":
            payload.update({"execution_retro_status": "paper-execution-retro-empty"})
        elif step_name == "gap_refresh":
            payload.update({"gap_status": "blocking_gap_active"})
        elif step_name == "brief_refresh":
            payload.update(
                {
                    "operator_status": "commodity-paper-execution-gap-blocked",
                    "operator_stack_brief": "commodity:gap | crypto:-",
                }
            )
        return payload

    monkeypatch.setattr(mod, "run_json_step", fake_run_json_step)

    rc = mod.main(
        [
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--context-path",
            str(context_path),
            "--now",
            "2026-03-16T09:05:20Z",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["ticket_lane_artifact"] == str(review_dir / "commodity_paper_ticket_lane.json")
    assert payload["ticket_book_artifact"] == str(review_dir / "commodity_paper_ticket_book.json")
    assert payload["execution_preview_artifact"] == str(review_dir / "commodity_paper_execution_preview.json")
    assert payload["execution_artifact"] == str(review_dir / "commodity_paper_execution_artifact.json")
    assert payload["execution_queue_artifact"] == str(review_dir / "commodity_paper_execution_queue.json")
    assert payload["bridge_artifact"] == str(review_dir / "commodity_paper_execution_bridge.json")
    assert payload["review_artifact"] == str(review_dir / "commodity_paper_execution_review.json")
    assert payload["retro_artifact"] == str(review_dir / "commodity_paper_execution_retro.json")
    assert payload["gap_artifact"] == str(review_dir / "commodity_paper_execution_gap_report.json")
    assert payload["brief_source_artifact"] == str(review_dir / "hot_universe_operator_brief.json")
    assert [row["name"] for row in payload["steps"]] == [
        "ticket_lane_refresh",
        "ticket_book_refresh",
        "execution_preview_refresh",
        "execution_artifact_refresh",
        "execution_queue_refresh",
        "bridge_refresh",
        "review_refresh",
        "retro_refresh",
        "gap_refresh",
        "brief_refresh",
    ]
    assert calls[0][0] == "ticket_lane_refresh"


def test_main_rebuilds_open_positions_from_bridge_payload(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    context_path = review_dir / "NEXT_WINDOW_CONTEXT_LATEST.md"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    calls: list[str] = []

    def fake_run_json_step(*, step_name: str, cmd: list[str]) -> dict:
        calls.append(step_name)
        as_of = cmd[cmd.index("--now") + 1] if "--now" in cmd else "2026-03-16T09:05:20Z"
        payload = {"ok": True, "status": "ok", "as_of": as_of, "artifact": str(review_dir / f"{step_name}.json")}
        if step_name == "ticket_lane_refresh":
            payload.update({"ticket_status": "paper-ready"})
        elif step_name == "ticket_book_refresh":
            payload.update({"ticket_book_status": "paper-ready"})
        elif step_name == "execution_preview_refresh":
            payload.update({"execution_preview_status": "paper-execution-ready"})
        elif step_name == "execution_artifact_refresh":
            payload.update({"execution_artifact_status": "paper-execution-artifact-ready"})
        elif step_name == "execution_queue_refresh":
            payload.update(
                {
                    "execution_queue_status": "paper-execution-queued",
                    "execution_batch": "asphalt_cn",
                    "execution_symbols": ["BU2606"],
                    "next_execution_symbol": "BU2606",
                }
            )
        elif step_name == "bridge_refresh":
            payload.update(
                {
                    "bridge_status": "bridge_noop_already_bridged",
                    "bridge_items": [
                        {
                            "execution_id": "commodity-paper-execution:asphalt_cn:BU2606",
                            "source_ticket_id": "commodity-paper-ticket:asphalt_cn:BU2606",
                            "symbol": "BU2606",
                            "signal_source_side": "LONG",
                            "size_pct": 9.064,
                            "risk_pct": 0.668,
                            "entry_price": 4532.0,
                            "stop_price": 4198.0,
                            "target_price": 5200.0,
                            "quote_usdt": 9064.0,
                            "signal_date": "2026-03-27",
                            "regime_gate": "paper_only",
                            "execution_price_normalization_mode": "paper_proxy_reference",
                            "paper_proxy_price_normalized": False,
                            "signal_price_reference_kind": "contract_native_daily",
                            "signal_price_reference_source": "akshare.futures_zh_daily_sina:BU0",
                            "signal_price_reference_provider": "akshare.futures_zh_daily_sina",
                            "signal_price_reference_symbol": "BU0",
                            "bridge_idempotency_key": "demo-key",
                            "bridge_status": "already_bridged",
                            "already_present": True,
                        }
                    ],
                }
            )
        elif step_name == "review_refresh":
            payload.update({"execution_review_status": "paper-execution-close-evidence-pending"})
        elif step_name == "retro_refresh":
            payload.update({"execution_retro_status": "paper-execution-close-evidence-pending"})
        elif step_name == "gap_refresh":
            payload.update({"gap_status": "gap_clear"})
        elif step_name == "brief_refresh":
            payload.update({"operator_status": "commodity-paper-execution-close-evidence-pending", "operator_stack_brief": "commodity:close-evidence:BU2606 | crypto:-"})
        return payload

    monkeypatch.setattr(mod, "run_json_step", fake_run_json_step)

    rc = mod.main(
        [
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--context-path",
            str(context_path),
            "--now",
            "2026-03-16T09:05:20Z",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    positions_path = output_root / "artifacts" / "paper_positions_open.json"
    assert positions_path.exists()
    positions_payload = json.loads(positions_path.read_text(encoding="utf-8"))
    assert positions_payload["positions"][0]["symbol"] == "BU2606"
    assert positions_payload["positions"][0]["source_execution_id"] == "commodity-paper-execution:asphalt_cn:BU2606"
    assert "bridge_refresh" in calls
    assert payload["status"] == "ok"


def test_rebuild_open_positions_skips_closed_execution_ids(tmp_path: Path) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    db_path = output_root / "artifacts" / "lie_engine.db"
    positions_path_existing = output_root / "artifacts" / "paper_positions_open.json"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    positions_path_existing.write_text(
        json.dumps(
            {
                "as_of": "2026-03-28T00:00:00Z",
                "positions": [
                    {
                        "open_date": "2026-03-27",
                        "symbol": "BU2606",
                        "side": "LONG",
                        "status": "OPEN",
                        "source_execution_id": "commodity-paper-execution:asphalt_cn:BU2606",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE executed_plans (
              date TEXT,
              symbol TEXT,
              status TEXT,
              bridge_execution_id TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO executed_plans (date, symbol, status, bridge_execution_id) VALUES (?, ?, ?, ?)",
            ("2026-03-28", "BU2606", "CLOSED", None),
        )
        conn.commit()

    positions_path = mod.rebuild_open_positions_from_bridge_payload(
        output_root=output_root,
        bridge_payload={
            "as_of": "2026-03-28T00:00:00Z",
            "bridge_items": [
                {
                    "execution_id": "commodity-paper-execution:asphalt_cn:BU2606",
                    "source_ticket_id": "commodity-paper-ticket:asphalt_cn:BU2606",
                    "symbol": "BU2606",
                    "signal_source_side": "LONG",
                    "size_pct": 9.064,
                    "risk_pct": 0.668,
                    "entry_price": 4532.0,
                    "stop_price": 4198.0,
                    "target_price": 5200.0,
                    "quote_usdt": 9064.0,
                    "signal_date": "2026-03-27",
                    "regime_gate": "paper_only",
                    "execution_price_normalization_mode": "paper_proxy_reference",
                    "paper_proxy_price_normalized": False,
                    "signal_price_reference_kind": "contract_native_daily",
                    "signal_price_reference_source": "akshare.futures_zh_daily_sina:BU0",
                    "signal_price_reference_provider": "akshare.futures_zh_daily_sina",
                    "signal_price_reference_symbol": "BU0",
                    "bridge_idempotency_key": "demo-key",
                    "bridge_status": "already_bridged",
                    "already_present": True,
                }
            ],
        },
    )

    payload = json.loads(positions_path.read_text(encoding="utf-8"))
    assert payload["positions"] == []
