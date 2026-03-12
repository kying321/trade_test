from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_cvd_queue_handoff.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_cvd_queue_handoff_marks_watch_only_batches(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    queue_path = review_dir / "20260310T174352Z_crypto_cvd_queue_profile.json"
    semantic_path = review_dir / "20260310T095206Z_crypto_cvd_semantic_snapshot.json"
    _write_json(
        queue_path,
        {
            "action": "derive_crypto_cvd_queue_profile",
            "status": "derived",
            "crypto_cvd_queue_profile": {
                "priority_batches": ["crypto_majors", "crypto_hot"],
                "overall_takeaway": "Use CVD-lite as a queue filter.",
                "batch_profiles": [
                    {
                        "batch": "crypto_majors",
                        "status_label": "pilot_only",
                        "queue_mode": "reversal_absorption_watch",
                        "trust_requirement": "basket_consensus",
                        "dominant_regime": "震荡",
                        "leader_symbols": ["ETHUSDT"],
                        "preferred_contexts": ["reversal", "absorption", "failed_auction"],
                        "veto_biases": ["trend_chase"],
                        "cvd_eligible_symbols": ["BTCUSDT", "ETHUSDT"],
                    },
                    {
                        "batch": "crypto_hot",
                        "status_label": "pilot_only",
                        "queue_mode": "trend_confirmation",
                        "trust_requirement": "dual_leader_alignment",
                        "dominant_regime": "下跌趋势",
                        "leader_symbols": ["ZECUSDT"],
                        "preferred_contexts": ["continuation", "failed_auction"],
                        "veto_biases": ["effort_result_divergence"],
                        "cvd_eligible_symbols": ["BTCUSDT", "SOLUSDT"],
                    },
                ],
            },
        },
    )
    _write_json(
        semantic_path,
        {
            "action": "build_crypto_cvd_semantic_snapshot",
            "ok": True,
            "status": "ok",
            "source_status": "degraded",
            "takeaway": "All current CVD-lite observations are downgraded to watch-only.",
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "classification": "watch_only",
                    "cvd_context_mode": "absorption",
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "active_reasons": ["time_sync_risk"],
                },
                {
                    "symbol": "ETHUSDT",
                    "classification": "watch_only",
                    "cvd_context_mode": "reversal",
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "active_reasons": ["time_sync_risk"],
                },
                {
                    "symbol": "SOLUSDT",
                    "classification": "watch_only",
                    "cvd_context_mode": "failed_auction",
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "active_reasons": ["time_sync_risk"],
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
    assert payload["operator_status"] == "queue-watch-only"
    assert payload["ready_now_batches"] == []
    assert payload["watch_only_batches"] == ["crypto_majors", "crypto_hot"]
    assert payload["focus_stack_brief"] == "queue_priority -> live_semantics -> watch_only_gate"
    assert payload["next_focus_batch"] == "crypto_hot"
    assert payload["next_focus_action"] == "defer_until_micro_recovers"
    assert payload["queue_stack_brief"] == "crypto_hot -> crypto_majors"
    assert payload["deferred_batches"] == ["crypto_hot", "crypto_majors"]
    assert len(payload["runtime_queue"]) == 2
    assert payload["runtime_queue"][0]["batch"] == "crypto_hot"
    assert payload["runtime_queue"][0]["queue_rank"] == 1
    assert payload["runtime_queue"][0]["queue_action"] == "defer_until_micro_recovers"
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["markdown"])).exists()
    assert Path(str(payload["checksum"])).exists()


def test_build_crypto_cvd_queue_handoff_reports_missing_queue_profile(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    semantic_path = review_dir / "20260310T095206Z_crypto_cvd_semantic_snapshot.json"
    _write_json(
        semantic_path,
        {
            "action": "build_crypto_cvd_semantic_snapshot",
            "ok": True,
            "status": "ok",
            "source_status": "degraded",
            "symbols": [],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
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
    assert payload["status"] == "queue_profile_missing"
