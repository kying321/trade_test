from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_shadow_clock_evidence.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_shadow_clock_evidence_marks_shadow_learning_allowed_when_timestamp_chain_is_present(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T111005Z_openclaw_orderflow_executor_heartbeat.json",
        {
            "generated_at_utc": "2026-03-15T11:10:05Z",
            "executor_status": "shadow_guarded_idle",
            "intent_symbol": "SOLUSDT",
        },
    )
    _write_json(
        review_dir / "20260315T111010Z_openclaw_orderflow_executor_state.json",
        {
            "generated_at_utc": "2026-03-15T11:10:10Z",
            "executor_status": "shadow_guarded_executor_ready",
        },
    )
    _write_json(
        review_dir / "20260315T102000Z_remote_execution_journal.json",
        {
            "generated_at_utc": "2026-03-15T10:20:00Z",
            "intent_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "last_entry": {"recorded_at_utc": "2026-03-15T10:20:00Z"},
        },
    )
    _write_json(
        review_dir / "20260315T095500Z_remote_execution_identity_state.json",
        {"ready_check_scope_market": "portfolio_margin_um"},
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:12:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["evidence_status"] == "shadow_clock_evidence_present"
    assert payload["shadow_learning_allowed"] is True
    assert payload["timestamp_chain_ok"] is True
    assert payload["route_symbol"] == "SOLUSDT"
    assert payload["remote_market"] == "portfolio_margin_um"
    assert Path(str(payload["artifact"])).name == "20260315T111200Z_remote_shadow_clock_evidence.json"
