#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_NON_OVERLAP_TRAIN_DAYS = [20, 25, 30, 35, 40, 45, 50, 55, 60]
DEFAULT_OVERLAP_TRAIN_DAYS = [35, 40]
DEFAULT_CAPACITY_CANDIDATE_TRAIN_DAYS = [20, 25, 30, 35, 40, 45, 50, 55, 60]


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    text_value = str(raw or "").strip()
    if not text_value:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_stamp(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def text(value: Any) -> str:
    return str(value or "").strip()


def parse_int_list(raw: str, default: list[int]) -> list[int]:
    parsed = [int(chunk.strip()) for chunk in text(raw).split(",") if chunk.strip()]
    values = parsed or list(default)
    if not values:
        raise ValueError("empty_train_day_list")
    if any(day <= 0 for day in values):
        raise ValueError("train_days_must_be_positive")
    return sorted(dict.fromkeys(values))


def resolve_system_root(workspace: Path) -> Path:
    if (workspace / "system").exists():
        return workspace / "system"
    if workspace.name == "system":
        return workspace
    raise FileNotFoundError(f"cannot_resolve_system_root:{workspace}")


def sort_key(path: Path) -> tuple[str, float, str]:
    return (path.name, path.stat().st_mtime, path.name)


def latest_review_artifact(review_dir: Path, pattern: str, error_code: str) -> Path:
    candidates = [path for path in review_dir.glob(pattern) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(error_code)
    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


def preferred_review_artifact(review_dir: Path, *, latest_name: str, pattern: str, error_code: str) -> Path:
    latest_alias = review_dir / latest_name
    if latest_alias.is_file():
        return latest_alias
    return latest_review_artifact(review_dir, pattern, error_code)


def preferred_intraday_dataset(review_dir: Path, *, error_code: str) -> Path:
    return preferred_review_artifact(
        review_dir,
        latest_name="latest_public_intraday_crypto_bars_dataset.csv",
        pattern="*_public_intraday_crypto_bars_dataset.csv",
        error_code=error_code,
    )


def require_path(path: Path, code: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(code)
    return path


def current_python_executable() -> str:
    return sys.executable or "python3"


def run_json(*, name: str, cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or f"returncode={proc.returncode}"
        raise RuntimeError(f"{name}_failed: {detail}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name}_invalid_json: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name}_invalid_payload")
    return payload


def payload_path(payload: dict[str, Any], *, name: str, key: str = "json_path") -> str:
    value = text(payload.get(key))
    if not value:
        raise RuntimeError(f"{name}_missing_{key}")
    return value


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid_json_mapping:{path}")
    return payload


def offset_stamp(base_time: dt.datetime, seconds: int) -> str:
    return fmt_stamp(base_time + dt.timedelta(seconds=int(seconds)))


def dataset_symbol_window(path: Path, symbol: str) -> dict[str, Any] | None:
    import pandas as pd

    frame = pd.read_csv(path, usecols=["ts", "symbol"])
    if frame.empty:
        return None
    work = frame[frame["symbol"] == symbol].copy()
    if work.empty:
        return None
    work["ts"] = pd.to_datetime(work["ts"], utc=True)
    return {
        "path": str(path),
        "row_count": int(len(work)),
        "coverage_start": work["ts"].min(),
        "coverage_end": work["ts"].max(),
    }


def choose_long_dataset(review_dir: Path, *, derivation_dataset_path: Path, symbol: str) -> Path:
    derivation_window = dataset_symbol_window(derivation_dataset_path, symbol)
    if not derivation_window:
        raise FileNotFoundError("derivation_dataset_missing_symbol_for_hold_transfer")

    candidates: list[dict[str, Any]] = []
    for path in review_dir.glob("*_public_intraday_crypto_bars_dataset.csv"):
        resolved = path.expanduser().resolve()
        if resolved == derivation_dataset_path:
            continue
        window = dataset_symbol_window(resolved, symbol)
        if not window:
            continue
        if (
            int(window["row_count"]) > int(derivation_window["row_count"])
            and window["coverage_start"] < derivation_window["coverage_start"]
            and window["coverage_end"] >= derivation_window["coverage_end"]
        ):
            candidates.append({"path": resolved, **window})

    if not candidates:
        raise FileNotFoundError("missing_long_dataset_for_hold_transfer")

    chosen = max(
        candidates,
        key=lambda row: (
            int(row["row_count"]),
            -int(row["coverage_start"].value),
            int(row["coverage_end"].value),
            Path(text(row["path"])).name,
        ),
    )
    return Path(text(chosen["path"])).expanduser().resolve()


def resolve_derivation_dataset_path(
    *,
    explicit_dataset_path: str,
    base_payload: dict[str, Any],
    review_dir: Path,
) -> Path:
    if text(explicit_dataset_path):
        return require_path(
            Path(explicit_dataset_path).expanduser().resolve(),
            "missing_derivation_dataset_path",
        )

    base_dataset_path = text(base_payload.get("dataset_path"))
    if base_dataset_path:
        candidate = Path(base_dataset_path).expanduser().resolve()
        if candidate.exists():
            return candidate

    return preferred_intraday_dataset(
        review_dir,
        error_code="no_public_intraday_crypto_bars_dataset_found",
    )


def resolve_long_dataset_path(
    *,
    explicit_long_dataset_path: str,
    review_dir: Path,
    derivation_dataset_path: Path,
    symbol: str,
) -> Path:
    if text(explicit_long_dataset_path):
        return require_path(
            Path(explicit_long_dataset_path).expanduser().resolve(),
            "missing_long_dataset_path",
        )

    family_transfer_head = review_dir / "latest_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only.json"
    if family_transfer_head.exists():
        payload = load_json_mapping(family_transfer_head)
        long_dataset_path = text(payload.get("long_dataset_path"))
        if long_dataset_path:
            candidate = Path(long_dataset_path).expanduser().resolve()
            if candidate.exists():
                return candidate

    return choose_long_dataset(
        review_dir,
        derivation_dataset_path=derivation_dataset_path,
        symbol=symbol,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the full SIM_ONLY ETH hold-side upstream chain and emit canonical hold handoff + stop condition from a single entrypoint."
    )
    parser.add_argument("--workspace", default=".", help="Fenlie workspace root or system directory.")
    parser.add_argument("--review-dir", default="", help="Optional explicit review directory override.")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--dataset-path", default="", help="Optional explicit derivation dataset override.")
    parser.add_argument("--long-dataset-path", default="", help="Optional explicit long-history dataset override.")
    parser.add_argument("--base-artifact-path", default="")
    parser.add_argument("--hold-robustness-path", default="")
    parser.add_argument("--rider-triage-path", default="")
    parser.add_argument("--non-overlap-train-days", default="20,25,30,35,40,45,50,55,60")
    parser.add_argument("--overlap-train-days", default="35,40")
    parser.add_argument("--capacity-candidate-train-days", default="20,25,30,35,40,45,50,55,60")
    parser.add_argument("--validation-days", type=int, default=10)
    parser.add_argument("--non-overlap-step-days", type=int, default=10)
    parser.add_argument("--overlap-step-days", type=int, default=5)
    parser.add_argument("--now", help="Explicit UTC timestamp used to derive the shared builder stamp.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    system_root = resolve_system_root(workspace)
    review_dir = Path(args.review_dir).expanduser().resolve() if text(args.review_dir) else system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    symbol = text(args.symbol).upper()
    runtime_now = parse_now(args.now)
    stamp = fmt_stamp(runtime_now)
    non_overlap_train_days = parse_int_list(args.non_overlap_train_days, DEFAULT_NON_OVERLAP_TRAIN_DAYS)
    overlap_train_days = parse_int_list(args.overlap_train_days, DEFAULT_OVERLAP_TRAIN_DAYS)
    capacity_candidate_train_days = parse_int_list(
        args.capacity_candidate_train_days,
        DEFAULT_CAPACITY_CANDIDATE_TRAIN_DAYS,
    )

    base_artifact_path = (
        Path(args.base_artifact_path).expanduser().resolve()
        if text(args.base_artifact_path)
        else latest_review_artifact(
            review_dir,
            "*_price_action_breakout_pullback_sim_only.json",
            "no_price_action_breakout_pullback_sim_only_artifact_found",
        )
    )
    base_payload = load_json_mapping(base_artifact_path)
    derivation_dataset_path = resolve_derivation_dataset_path(
        explicit_dataset_path=args.dataset_path,
        base_payload=base_payload,
        review_dir=review_dir,
    )
    long_dataset_path = resolve_long_dataset_path(
        explicit_long_dataset_path=args.long_dataset_path,
        review_dir=review_dir,
        derivation_dataset_path=derivation_dataset_path,
        symbol=symbol,
    )
    hold_robustness_path = (
        Path(args.hold_robustness_path).expanduser().resolve()
        if text(args.hold_robustness_path)
        else latest_review_artifact(
            review_dir,
            "latest_price_action_breakout_pullback_exit_hold_robustness_sim_only.json",
            "missing_exit_hold_robustness_latest",
        )
    )
    rider_triage_path = (
        Path(args.rider_triage_path).expanduser().resolve()
        if text(args.rider_triage_path)
        else latest_review_artifact(
            review_dir,
            "latest_price_action_breakout_pullback_exit_rider_triage_sim_only.json",
            "missing_exit_rider_triage_latest",
        )
    )

    hold_family_triage_payload = run_json(
        name="build_hold_family_triage",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_hold_family_triage_sim_only.py"),
            "--dataset-path",
            str(derivation_dataset_path),
            "--base-artifact-path",
            str(base_artifact_path),
            "--review-dir",
            str(review_dir),
            "--symbol",
            symbol,
            "--stamp",
            offset_stamp(runtime_now, 0),
        ],
    )
    hold_family_triage_path = Path(
        payload_path(hold_family_triage_payload, name="build_hold_family_triage")
    ).expanduser().resolve()

    frontier_report_payload = run_json(
        name="build_hold_frontier_report",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_hold_frontier_report_sim_only.py"),
            "--review-dir",
            str(review_dir),
            "--base-artifact-path",
            str(base_artifact_path),
            "--hold-robustness-path",
            str(hold_robustness_path),
            "--hold-family-triage-path",
            str(hold_family_triage_path),
            "--rider-triage-path",
            str(rider_triage_path),
            "--stamp",
            offset_stamp(runtime_now, 1),
        ],
    )
    frontier_report_path = Path(
        payload_path(frontier_report_payload, name="build_hold_frontier_report")
    ).expanduser().resolve()

    frontier_cost_payload = run_json(
        name="build_hold_frontier_cost_sensitivity",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_hold_frontier_cost_sensitivity_sim_only.py"),
            "--dataset-path",
            str(derivation_dataset_path),
            "--base-artifact-path",
            str(base_artifact_path),
            "--review-dir",
            str(review_dir),
            "--symbol",
            symbol,
            "--stamp",
            offset_stamp(runtime_now, 2),
        ],
    )
    frontier_cost_path = Path(
        payload_path(frontier_cost_payload, name="build_hold_frontier_cost_sensitivity")
    ).expanduser().resolve()

    router_hypothesis_payload = run_json(
        name="build_hold_router_hypothesis",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_hold_router_hypothesis_sim_only.py"),
            "--dataset-path",
            str(derivation_dataset_path),
            "--base-artifact-path",
            str(base_artifact_path),
            "--review-dir",
            str(review_dir),
            "--symbol",
            symbol,
            "--stamp",
            offset_stamp(runtime_now, 3),
        ],
    )
    router_hypothesis_path = Path(
        payload_path(router_hypothesis_payload, name="build_hold_router_hypothesis")
    ).expanduser().resolve()

    router_transfer_payload = run_json(
        name="build_hold_router_transfer",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only.py"),
            "--long-dataset-path",
            str(long_dataset_path),
            "--derivation-dataset-path",
            str(derivation_dataset_path),
            "--base-artifact-path",
            str(base_artifact_path),
            "--router-artifact-path",
            str(router_hypothesis_path),
            "--review-dir",
            str(review_dir),
            "--symbol",
            symbol,
            "--stamp",
            offset_stamp(runtime_now, 4),
        ],
    )
    router_transfer_path = Path(
        payload_path(router_transfer_payload, name="build_hold_router_transfer")
    ).expanduser().resolve()

    family_transfer_payload = run_json(
        name="build_hold_family_transfer",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only.py"),
            "--long-dataset-path",
            str(long_dataset_path),
            "--derivation-dataset-path",
            str(derivation_dataset_path),
            "--base-artifact-path",
            str(base_artifact_path),
            "--current-hold-family-artifact-path",
            str(hold_family_triage_path),
            "--review-dir",
            str(review_dir),
            "--symbol",
            symbol,
            "--stamp",
            offset_stamp(runtime_now, 5),
        ],
    )
    family_transfer_path = Path(
        payload_path(family_transfer_payload, name="build_hold_family_transfer")
    ).expanduser().resolve()

    non_overlap_compare_paths: list[str] = []
    for index, train_days in enumerate(non_overlap_train_days):
        payload = run_json(
            name="build_exit_hold_forward_compare",
            cmd=[
                current_python_executable(),
                str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.py"),
                "--dataset-path",
                str(derivation_dataset_path),
                "--base-artifact-path",
                str(base_artifact_path),
                "--symbol",
                symbol,
                "--review-dir",
                str(review_dir),
                "--stamp",
                offset_stamp(runtime_now, 6 + index),
                "--train-days",
                str(train_days),
                "--validation-days",
                str(args.validation_days),
                "--step-days",
                str(args.non_overlap_step_days),
            ],
        )
        non_overlap_compare_paths.append(
            payload_path(payload, name="build_exit_hold_forward_compare")
        )

    window_consensus_payload = run_json(
        name="build_exit_hold_window_consensus",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_hold_window_consensus_sim_only.py"),
            *[item for path in non_overlap_compare_paths for item in ("--compare-path", path)],
            "--review-dir",
            str(review_dir),
            "--stamp",
            offset_stamp(runtime_now, 6 + len(non_overlap_train_days)),
        ],
    )
    window_consensus_path = Path(
        payload_path(window_consensus_payload, name="build_exit_hold_window_consensus")
    ).expanduser().resolve()

    overlap_compare_paths: list[str] = []
    overlap_start_offset = 7 + len(non_overlap_train_days)
    for index, train_days in enumerate(overlap_train_days):
        payload = run_json(
            name="build_exit_hold_forward_compare",
            cmd=[
                current_python_executable(),
                str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.py"),
                "--dataset-path",
                str(derivation_dataset_path),
                "--base-artifact-path",
                str(base_artifact_path),
                "--symbol",
                symbol,
                "--review-dir",
                str(review_dir),
                "--stamp",
                offset_stamp(runtime_now, overlap_start_offset + index),
                "--train-days",
                str(train_days),
                "--validation-days",
                str(args.validation_days),
                "--step-days",
                str(args.overlap_step_days),
            ],
        )
        overlap_compare_paths.append(
            payload_path(payload, name="build_exit_hold_forward_compare")
        )

    overlap_sidecar_offset = overlap_start_offset + len(overlap_train_days)
    overlap_sidecar_payload = run_json(
        name="build_exit_hold_overlap_sidecar",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_hold_overlap_sidecar_sim_only.py"),
            *[item for path in overlap_compare_paths for item in ("--compare-path", path)],
            "--review-dir",
            str(review_dir),
            "--stamp",
            offset_stamp(runtime_now, overlap_sidecar_offset),
        ],
    )
    overlap_sidecar_path = Path(
        payload_path(overlap_sidecar_payload, name="build_exit_hold_overlap_sidecar")
    ).expanduser().resolve()

    capacity_offset = overlap_sidecar_offset + 1
    forward_capacity_payload = run_json(
        name="build_exit_hold_forward_window_capacity",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.py"),
            "--dataset-path",
            str(derivation_dataset_path),
            "--symbol",
            symbol,
            "--review-dir",
            str(review_dir),
            "--stamp",
            offset_stamp(runtime_now, capacity_offset),
            "--validation-days",
            str(args.validation_days),
            "--step-days",
            str(args.non_overlap_step_days),
            "--candidate-train-days",
            ",".join(str(day) for day in capacity_candidate_train_days),
        ],
    )
    forward_capacity_path = Path(
        payload_path(forward_capacity_payload, name="build_exit_hold_forward_window_capacity")
    ).expanduser().resolve()

    gate_offset = capacity_offset + 1
    gate_blocker_payload = run_json(
        name="build_hold_selection_gate_blocker",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only.py"),
            "--frontier-report-path",
            str(frontier_report_path),
            "--frontier-cost-path",
            str(frontier_cost_path),
            "--router-hypothesis-path",
            str(router_hypothesis_path),
            "--router-transfer-path",
            str(router_transfer_path),
            "--family-transfer-path",
            str(family_transfer_path),
            "--window-consensus-path",
            str(window_consensus_path),
            "--forward-capacity-path",
            str(forward_capacity_path),
            "--overlap-sidecar-path",
            str(overlap_sidecar_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            offset_stamp(runtime_now, gate_offset),
        ],
    )
    gate_blocker_path = Path(
        payload_path(gate_blocker_payload, name="build_hold_selection_gate_blocker")
    ).expanduser().resolve()

    handoff_offset = gate_offset + 1
    handoff_payload = run_json(
        name="build_hold_selection_handoff",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_hold_selection_handoff_sim_only.py"),
            "--gate-blocker-path",
            str(gate_blocker_path),
            "--frontier-report-path",
            str(frontier_report_path),
            "--family-transfer-path",
            str(family_transfer_path),
            "--router-transfer-path",
            str(router_transfer_path),
            "--forward-capacity-path",
            str(forward_capacity_path),
            "--overlap-sidecar-path",
            str(overlap_sidecar_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            offset_stamp(runtime_now, handoff_offset),
        ],
    )
    handoff_path = Path(
        payload_path(handoff_payload, name="build_hold_selection_handoff")
    ).expanduser().resolve()
    handoff_artifact = load_json_mapping(handoff_path) if handoff_path.exists() else dict(handoff_payload)

    stop_offset = handoff_offset + 1
    stop_condition_payload = run_json(
        name="build_exit_hold_forward_stop",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.py"),
            "--forward-capacity-path",
            str(forward_capacity_path),
            "--overlap-sidecar-path",
            str(overlap_sidecar_path),
            "--handoff-path",
            str(handoff_path),
            "--window-consensus-path",
            str(window_consensus_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            offset_stamp(runtime_now, stop_offset),
        ],
    )
    stop_condition_path = Path(
        payload_path(stop_condition_payload, name="build_exit_hold_forward_stop")
    ).expanduser().resolve()
    stop_condition_artifact = (
        load_json_mapping(stop_condition_path) if stop_condition_path.exists() else dict(stop_condition_payload)
    )

    audit_offset = stop_offset + 1
    source_gap_audit_payload = run_json(
        name="build_hold_upstream_source_gap_audit",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_hold_upstream_source_gap_audit_sim_only.py"),
            "--workspace",
            str(workspace),
            "--review-dir",
            str(review_dir),
            "--stamp",
            offset_stamp(runtime_now, audit_offset),
        ],
    )
    source_gap_audit_path = Path(
        payload_path(source_gap_audit_payload, name="build_hold_upstream_source_gap_audit")
    ).expanduser().resolve()
    source_gap_audit_artifact = (
        load_json_mapping(source_gap_audit_path) if source_gap_audit_path.exists() else dict(source_gap_audit_payload)
    )

    print(
        json.dumps(
            {
                "ok": True,
                "mode": "hold_upstream_refresh_sim_only",
                "change_class": "SIM_ONLY",
                "stamp": stamp,
                "workspace": str(workspace),
                "review_dir": str(review_dir),
                "symbol": symbol,
                "derivation_dataset_path": str(derivation_dataset_path),
                "long_dataset_path": str(long_dataset_path),
                "base_artifact_path": str(base_artifact_path),
                "hold_robustness_path": str(hold_robustness_path),
                "rider_triage_path": str(rider_triage_path),
                "non_overlap_train_days": non_overlap_train_days,
                "overlap_train_days": overlap_train_days,
                "capacity_candidate_train_days": capacity_candidate_train_days,
                "hold_family_triage_path": str(hold_family_triage_path),
                "frontier_report_path": str(frontier_report_path),
                "frontier_cost_path": str(frontier_cost_path),
                "router_hypothesis_path": str(router_hypothesis_path),
                "router_transfer_path": str(router_transfer_path),
                "family_transfer_path": str(family_transfer_path),
                "non_overlap_compare_paths": non_overlap_compare_paths,
                "window_consensus_path": str(window_consensus_path),
                "overlap_compare_paths": overlap_compare_paths,
                "overlap_sidecar_path": str(overlap_sidecar_path),
                "forward_capacity_path": str(forward_capacity_path),
                "gate_blocker_path": str(gate_blocker_path),
                "handoff_path": str(handoff_path),
                "stop_condition_path": str(stop_condition_path),
                "source_gap_audit_path": str(source_gap_audit_path),
                "handoff_latest_json_path": text(handoff_payload.get("latest_json_path")),
                "stop_condition_latest_json_path": text(stop_condition_payload.get("latest_json_path")),
                "source_gap_audit_latest_json_path": text(source_gap_audit_payload.get("latest_json_path")),
                "research_decision": text(handoff_payload.get("research_decision") or handoff_artifact.get("research_decision")),
                "source_head_status": text(handoff_payload.get("source_head_status") or handoff_artifact.get("source_head_status")),
                "active_baseline": text(handoff_artifact.get("active_baseline")),
                "local_candidate": text(handoff_artifact.get("local_candidate")),
                "transfer_watch": list(handoff_artifact.get("transfer_watch") or []),
                "stop_condition_research_decision": text(
                    stop_condition_payload.get("research_decision") or stop_condition_artifact.get("research_decision")
                ),
                "source_gap_audit_research_decision": text(
                    source_gap_audit_payload.get("research_decision") or source_gap_audit_artifact.get("research_decision")
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
