from __future__ import annotations

import importlib.util
import json
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
