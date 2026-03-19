from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_profile_location_watch.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, text_payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text_payload, encoding="utf-8")


def test_build_crypto_shortline_profile_location_watch_marks_lvn_and_key_level_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T062000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T062001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T062002Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {"location_priority": ["HVN", "POC"]},
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "route_state": "watch",
                    "execution_state": "Bias_Only",
                    "location_tag": "LVN",
                    "missing_gates": [
                        "profile_location=LVN",
                        "cvd_key_level_context",
                        "mss",
                    ],
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing profile_location=LVN, cvd_key_level_context, mss.",
                    "profile_proxy": {
                        "current_bin_volume": 10.0,
                        "poc_bin_volume": 100.0,
                    },
                    "micro_signals": {
                        "key_level_confirmed": True,
                        "attack_side": "buyers",
                        "attack_presence": "buyers_attacking",
                    },
                }
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
            "2026-03-16T06:22:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "profile_location_lvn_key_level_ready_rotation_far"
    assert (
        payload["watch_decision"]
        == "monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate"
    )
    assert payload["blocker_target_artifact"] == "crypto_shortline_profile_location_watch"
    assert payload["blocker_title"] == "Track LVN-to-HVN/POC rotation before shortline setup promotion"
    assert payload["profile_location_missing"] is True
    assert payload["key_level_context_missing"] is True
    assert payload["key_level_context_effective_status"] == "key_level_ready_waiting_profile_alignment"
    assert payload["profile_alignment_state"] == "misaligned:LVN->HVN,POC"
    assert payload["rotation_proximity_state"] == "far"
    assert payload["profile_rotation_alignment_band"] == "far"
    assert payload["profile_rotation_next_milestone"] == "toward_hvn_poc"
    assert payload["location_tag"] == "LVN"
    assert payload["preferred_location_tags"] == ["HVN", "POC"]
    assert payload["profile_volume_ratio"] == 0.1
    assert "profile_location=LVN" in payload["blocker_detail"]
    assert "cvd_key_level_context" in payload["blocker_detail"]


def test_build_crypto_shortline_profile_location_watch_marks_final_rotation_when_approaching(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T074000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T074001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T074002Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {"location_priority": ["HVN", "POC"]},
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "route_state": "watch",
                    "execution_state": "Bias_Only",
                    "location_tag": "LVN",
                    "missing_gates": [
                        "profile_location=LVN",
                        "cvd_key_level_context",
                        "mss",
                    ],
                    "profile_proxy": {
                        "current_bin_volume": 38.0,
                        "poc_bin_volume": 100.0,
                    },
                    "micro_signals": {
                        "key_level_confirmed": True,
                        "attack_side": "buyers",
                        "attack_presence": "buyers_attacking",
                    },
                }
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
            "2026-03-16T07:42:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "profile_location_lvn_key_level_ready_rotation_approaching"
    assert (
        payload["watch_decision"]
        == "monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate"
    )
    assert payload["blocker_title"] == "Track final LVN-to-HVN/POC rotation before shortline setup promotion"
    assert payload["rotation_proximity_state"] == "approaching"
    assert payload["profile_rotation_alignment_band"] == "approaching"
    assert payload["profile_rotation_next_milestone"] == "into_final_band"
    assert payload["profile_volume_ratio"] == 0.38
    assert payload["profile_rotation_confidence"] == 0.53
    assert payload["active_rotation_targets"] == ["HVN", "POC"]


def test_build_crypto_shortline_profile_location_watch_marks_final_band_when_near_poc_bin(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T074500Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T074501Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T074502Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {"location_priority": ["HVN", "POC"]},
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "route_state": "watch",
                    "execution_state": "Bias_Only",
                    "location_tag": "LVN",
                    "missing_gates": [
                        "profile_location=LVN",
                        "cvd_key_level_context",
                        "mss",
                    ],
                    "profile_proxy": {
                        "current_bin_volume": 62.0,
                        "poc_bin_volume": 100.0,
                    },
                    "micro_signals": {
                        "key_level_confirmed": True,
                        "attack_side": "buyers",
                        "attack_presence": "buyers_attacking",
                    },
                }
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
            "2026-03-16T07:45:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "profile_location_lvn_key_level_ready_rotation_final_band"
    assert (
        payload["watch_decision"]
        == "monitor_last_profile_rotation_into_hvn_poc_then_recheck_execution_gate"
    )
    assert payload["rotation_proximity_state"] == "final_band"
    assert payload["profile_rotation_alignment_band"] == "final"
    assert payload["profile_rotation_next_milestone"] == "last_band"
    assert payload["active_rotation_targets"] == ["HVN", "POC"]
    assert payload["profile_rotation_confidence"] == 0.77


def test_build_crypto_shortline_profile_location_watch_uses_live_bars_for_target_distance(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    bars_path = review_dir / "20260316T074503Z_crypto_shortline_live_bars_snapshot_bars.csv"
    _write_json(
        review_dir / "20260316T074500Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T074501Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T074502Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {"location_priority": ["HVN", "POC"]},
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "route_state": "watch",
                    "execution_state": "Bias_Only",
                    "location_tag": "LVN",
                    "missing_gates": [
                        "profile_location=LVN",
                        "cvd_key_level_context",
                        "mss",
                    ],
                    "profile_proxy": {
                        "current_bin_volume": 20.0,
                        "poc_bin_volume": 100.0,
                        "bins": 6,
                    },
                    "micro_signals": {
                        "key_level_confirmed": True,
                        "attack_side": "buyers",
                        "attack_presence": "buyers_attacking",
                    },
                }
            ],
        },
    )
    _write_csv(
        bars_path,
        "\n".join(
            [
                "symbol,ts,open,high,low,close,volume",
                "SOLUSDT,2026-03-16T00:00:00Z,90,90,90,90,0",
                "SOLUSDT,2026-03-16T01:00:00Z,91,91,91,91,0",
                "SOLUSDT,2026-03-16T02:00:00Z,92,92,92,92,0",
                "SOLUSDT,2026-03-16T03:00:00Z,93,93,93,93,100",
                "SOLUSDT,2026-03-16T04:00:00Z,94,94,94,94,0",
                "SOLUSDT,2026-03-16T05:00:00Z,95,95,95,95,5",
                "",
            ]
        ),
    )
    _write_json(
        review_dir / "20260316T074503Z_crypto_shortline_live_bars_snapshot.json",
        {
            "bars_artifact": str(bars_path),
            "route_symbol": "SOLUSDT",
            "snapshot_status": "bars_live_snapshot_ready",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T07:45:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["rotation_proximity_state"] == "approaching"
    assert payload["profile_rotation_alignment_band"] == "approaching"
    assert payload["profile_rotation_next_milestone"] == "into_final_band"
    assert payload["profile_rotation_target_tag"] == "HVN"
    assert payload["profile_rotation_target_bin_distance"] == 2
    assert payload["profile_rotation_target_distance_bps"] == 219.298246


def test_build_crypto_shortline_profile_location_watch_ignores_future_stamped_execution_gate(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T052000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T052001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    current_gate_path = review_dir / "20260316T052002Z_crypto_shortline_execution_gate.json"
    _write_json(
        current_gate_path,
        {
            "shortline_policy": {"location_priority": ["HVN", "POC"]},
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "route_state": "watch",
                    "execution_state": "Bias_Only",
                    "location_tag": "LVN",
                    "missing_gates": ["profile_location=LVN", "cvd_key_level_context", "mss"],
                    "blocker_detail": "current_gate",
                    "profile_proxy": {"current_bin_volume": 10.0, "poc_bin_volume": 100.0},
                    "micro_signals": {
                        "key_level_confirmed": True,
                        "attack_side": "buyers",
                        "attack_presence": "buyers_attacking",
                    },
                }
            ],
        },
    )
    future_gate_path = review_dir / "20260316T075702Z_crypto_shortline_execution_gate.json"
    _write_json(
        future_gate_path,
        {
            "shortline_policy": {"location_priority": ["HVN", "POC"]},
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "route_state": "watch",
                    "execution_state": "Bias_Only",
                    "location_tag": "MID",
                    "missing_gates": ["profile_location=MID"],
                    "blocker_detail": "future_gate",
                    "profile_proxy": {"current_bin_volume": 99.0, "poc_bin_volume": 100.0},
                    "micro_signals": {"key_level_confirmed": False},
                }
            ],
        },
    )
    os.utime(future_gate_path, None)

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T05:22:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["artifacts"]["crypto_shortline_execution_gate"] == str(current_gate_path)
    assert payload["location_tag"] == "LVN"
    assert payload["watch_status"] == "profile_location_lvn_key_level_ready_rotation_far"
    assert "current_gate" in payload["blocker_detail"]
    assert "future_gate" not in payload["blocker_detail"]
