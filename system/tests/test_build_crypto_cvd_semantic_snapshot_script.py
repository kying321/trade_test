from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_cvd_semantic_snapshot.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_cvd_semantic_snapshot_reads_micro_capture(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "micro_capture"
    source_path = artifact_dir / "20260310T094850Z_micro_capture.json"
    _write_json(
        source_path,
        {
            "as_of": "2026-03-10",
            "status": "degraded",
            "pass": False,
            "symbols_requested": 3,
            "symbols_selected": 3,
            "cvd_semantics": {
                "context_counts": {"absorption": 1, "reversal": 1, "failed_auction": 1},
                "trust_counts": {"single_exchange_low": 3},
                "veto_hint_counts": {"low_sample_or_gap_risk": 3},
            },
            "selected_micro": [
                {
                    "symbol": "BTCUSDT",
                    "cvd_context_mode": "absorption",
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "time_sync_ok": False,
                    "schema_ok": True,
                    "sync_ok": True,
                    "trade_count": 95,
                    "evidence_score": 0.61,
                    "cvd_context_note": "aggressive_flow_without_clean_price_result|time_sync_risk",
                },
                {
                    "symbol": "ETHUSDT",
                    "cvd_context_mode": "reversal",
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "time_sync_ok": False,
                    "schema_ok": True,
                    "sync_ok": True,
                    "trade_count": 88,
                    "evidence_score": 0.58,
                    "cvd_context_note": "delta_and_price_move_disagree|time_sync_risk",
                },
                {
                    "symbol": "SOLUSDT",
                    "cvd_context_mode": "failed_auction",
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "time_sync_ok": False,
                    "schema_ok": True,
                    "sync_ok": True,
                    "trade_count": 71,
                    "evidence_score": 0.55,
                    "cvd_context_note": "range_expanded_but_close_failed_to_hold|time_sync_risk",
                },
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--now",
            "2026-03-10T18:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["trust_counts"]["single_exchange_low"] == 3
    assert payload["watch_only_symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert payload["trend_confirmation_watch"] == []
    assert payload["reversal_absorption_watch"] == []
    assert payload["artifact_label"] == "crypto-cvd-semantic-snapshot:degraded"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["markdown"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_build_crypto_cvd_semantic_snapshot_reports_missing_selected_micro(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    artifact_dir = tmp_path / "artifacts" / "micro_capture"
    source_path = artifact_dir / "20260310T094850Z_micro_capture.json"
    _write_json(
        source_path,
        {
            "as_of": "2026-03-10",
            "status": "degraded",
            "pass": False,
            "symbols_requested": 0,
            "symbols_selected": 0,
            "cvd_semantics": {},
            "selected_micro": [],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--now",
            "2026-03-10T18:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["status"] == "selected_micro_missing"
    assert payload["symbols"] == []
