from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_signal_source_refresh_readiness.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_signal_source_refresh_readiness_marks_no_newer_candidate(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    (output_root / "daily").mkdir(parents=True, exist_ok=True)
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260315T100020Z_crypto_signal_source_freshness.json",
        {
            "freshness_status": "route_signal_row_stale",
            "freshness_brief": "route_signal_row_stale:SOLUSDT:2026-02-05:age_days=38:recent5_review",
            "route_signal_date": "2026-02-05",
            "route_signal_age_days": 38,
            "ticket_signal_source_artifact_date": "2026-03-06",
            "ticket_signal_source_artifact_age_days": 9,
            "freshness_ok": False,
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_remote_ticket_actionability_state.json",
        {
            "ticket_signal_source_artifact_date": "2026-03-06",
            "ticket_signal_source_age_days": 9,
            "route_signal_age_days": 38,
        },
    )
    (review_dir / "20260306_strategy_recent5_signals.json").write_text("{}", encoding="utf-8")
    (output_root / "daily" / "2026-03-06_signals.json").write_text("[]", encoding="utf-8")

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-15T10:05:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "no_newer_crypto_signal_candidate_available"
    assert payload["readiness_decision"] == "generate_fresh_crypto_signal_source_before_rebuild_tickets"
    assert payload["refresh_needed"] is True
    assert payload["newer_candidate_available"] is False
    assert payload["latest_candidate_artifact_date"] == "2026-03-06"


def test_build_crypto_signal_source_refresh_readiness_marks_newer_candidate_available(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    (output_root / "daily").mkdir(parents=True, exist_ok=True)
    _write_json(review_dir / "20260315T100000Z_remote_intent_queue.json", {"preferred_route_symbol": "SOLUSDT"})
    _write_json(review_dir / "20260315T100010Z_crypto_route_operator_brief.json", {"review_priority_head_symbol": "SOLUSDT"})
    _write_json(
        review_dir / "20260315T100020Z_crypto_signal_source_freshness.json",
        {
            "freshness_status": "route_signal_row_stale",
            "ticket_signal_source_artifact_date": "2026-03-06",
            "freshness_ok": False,
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_remote_ticket_actionability_state.json",
        {
            "ticket_signal_source_artifact_date": "2026-03-06",
        },
    )
    (output_root / "daily" / "2026-03-15_signals.json").write_text("[]", encoding="utf-8")

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-15T10:05:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "newer_signal_candidate_available"
    assert payload["readiness_decision"] == "rebuild_tickets_with_newer_signal_candidate"
    assert payload["newer_candidate_available"] is True


def test_build_crypto_signal_source_refresh_readiness_uses_shortline_source_as_latest_candidate(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    _write_json(review_dir / "20260315T100000Z_remote_intent_queue.json", {"preferred_route_symbol": "SOLUSDT"})
    _write_json(review_dir / "20260315T100010Z_crypto_route_operator_brief.json", {"review_priority_head_symbol": "SOLUSDT"})
    _write_json(
        review_dir / "20260315T100020Z_crypto_signal_source_freshness.json",
        {
            "freshness_status": "route_signal_row_stale",
            "ticket_signal_source_artifact_date": "2026-03-06",
            "freshness_ok": False,
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_remote_ticket_actionability_state.json",
        {
            "ticket_signal_source_artifact_date": "2026-03-06",
        },
    )
    _write_json(review_dir / "20260315T110600Z_crypto_shortline_signal_source.json", {"signals": {}})

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-15T10:05:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["latest_review_shortline_signal_source_date"] == "2026-03-15"
    assert payload["latest_candidate_artifact_date"] == "2026-03-15"
    assert payload["readiness_status"] == "newer_signal_candidate_available"
    assert payload["readiness_decision"] == "rebuild_tickets_with_newer_signal_candidate"


def test_build_crypto_signal_source_refresh_readiness_prefers_freshness_source_fields(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    _write_json(review_dir / "20260315T100000Z_remote_intent_queue.json", {"preferred_route_symbol": "SOLUSDT"})
    _write_json(review_dir / "20260315T100010Z_crypto_route_operator_brief.json", {"review_priority_head_symbol": "SOLUSDT"})
    _write_json(
        review_dir / "20260315T100020Z_crypto_signal_source_freshness.json",
        {
            "freshness_status": "route_signal_row_fresh",
            "freshness_brief": "route_signal_row_fresh:SOLUSDT:2026-03-15:age_days=0:crypto_shortline_signal_source",
            "route_signal_date": "2026-03-15",
            "route_signal_age_days": 0,
            "ticket_signal_source_artifact_date": "2026-03-15",
            "ticket_signal_source_artifact_age_days": 0,
            "freshness_ok": True,
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_remote_ticket_actionability_state.json",
        {
            "ticket_signal_source_artifact_date": "2026-03-06",
            "ticket_signal_source_age_days": 9,
            "route_signal_age_days": 38,
        },
    )
    _write_json(review_dir / "20260315T110600Z_crypto_shortline_signal_source.json", {"signals": {}})

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-15T10:05:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "signal_source_refresh_not_required"
    assert payload["ticket_signal_source_artifact_date"] == "2026-03-15"
    assert payload["ticket_signal_source_age_days"] == 0
    assert payload["route_signal_date"] == "2026-03-15"
    assert payload["route_signal_age_days"] == 0
