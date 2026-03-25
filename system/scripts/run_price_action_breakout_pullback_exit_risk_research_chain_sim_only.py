#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_FORWARD_TRAIN_DAYS = [30, 40, 45, 50, 55, 60]
DEFAULT_BREAK_EVEN_TRAIN_DAYS = [30, 40, 45, 50, 55, 60]
DEFAULT_BREAK_EVEN_STEP_DAYS = 5
ALIGNED_BREAK_EVEN_REVIEW_DIRNAME = "hold_selection_aligned_break_even_review"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    text = str(raw or "").strip()
    if not text:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_stamp(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def fmt_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_exit_params(payload: Any) -> dict[str, Any]:
    data = dict(payload or {})
    return {
        "max_hold_bars": int(data.get("max_hold_bars") or 0),
        "break_even_trigger_r": float(data.get("break_even_trigger_r") or 0.0),
        "trailing_stop_atr": float(data.get("trailing_stop_atr") or 0.0),
        "cooldown_after_losses": int(data.get("cooldown_after_losses") or 0),
        "cooldown_bars": int(data.get("cooldown_bars") or 0),
    }


def scaled_decimal(value: float, *, scale: int, width: int = 0) -> str:
    scaled = int(round(float(value or 0.0) * scale))
    if scaled == 0:
        return "0"
    return f"{scaled:0{width}d}" if width else str(scaled)


def format_exit_risk_anchor_slug(params: dict[str, Any]) -> str:
    max_hold_bars = int(params.get("max_hold_bars") or 0)
    trailing = scaled_decimal(float(params.get("trailing_stop_atr") or 0.0), scale=10)
    break_even = float(params.get("break_even_trigger_r") or 0.0)
    break_even_slug = "no_be" if break_even <= 0 else f"be{scaled_decimal(break_even, scale=100, width=3)}"
    return f"hold{max_hold_bars}_trail{trailing}_{break_even_slug}"


def extract_hold_bars(value: Any) -> int:
    normalized = text(value)
    if not normalized.startswith("hold"):
        return 0
    digits = []
    for ch in normalized[4:]:
        if not ch.isdigit():
            break
        digits.append(ch)
    return int("".join(digits)) if digits else 0


def artifact_or_payload(path_text: str, fallback_payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(path_text).expanduser().resolve() if text(path_text) else None
    if path is not None and path.exists():
        return load_json_mapping(path)
    return dict(fallback_payload)


def resolve_hold_selection_active_hold_bars(hold_selection_handoff: dict[str, Any]) -> int:
    direct = int(hold_selection_handoff.get("active_baseline_hold_bars") or 0)
    if direct > 0:
        return direct
    return extract_hold_bars(hold_selection_handoff.get("active_baseline"))


def select_hold_aligned_exit_params(
    *,
    hold_selection_handoff: dict[str, Any],
    forward_blocker_artifact: dict[str, Any],
    exit_risk_artifact: dict[str, Any],
) -> dict[str, Any]:
    target_hold_bars = resolve_hold_selection_active_hold_bars(hold_selection_handoff)
    if target_hold_bars <= 0:
        return {}

    challenge_pair = dict(forward_blocker_artifact.get("challenge_pair") or {})
    candidates = [
        normalize_exit_params(challenge_pair.get("challenger_exit_params")),
        normalize_exit_params(challenge_pair.get("baseline_exit_params")),
        normalize_exit_params(exit_risk_artifact.get("selected_exit_params")),
        normalize_exit_params(exit_risk_artifact.get("validation_leader_exit_params")),
    ]
    for candidate in candidates:
        if int(candidate.get("max_hold_bars") or 0) != target_hold_bars:
            continue
        aligned = dict(candidate)
        aligned["break_even_trigger_r"] = 0.0
        return aligned
    return {}


def canonical_anchor_consumer_refresh_required(
    *,
    canonical_handoff_artifact: dict[str, Any],
    forward_blocker_artifact: dict[str, Any],
    break_even_sidecar_artifact: dict[str, Any],
) -> bool:
    if text(canonical_handoff_artifact.get("research_decision")) != "use_exit_risk_handoff_as_canonical_anchor":
        return False
    if text(canonical_handoff_artifact.get("source_head_status")) != "challenger_anchor_active":
        return False

    canonical_anchor = text(canonical_handoff_artifact.get("active_baseline"))
    if not canonical_anchor:
        return False

    sidecar_active_baseline = text(break_even_sidecar_artifact.get("active_baseline"))
    sidecar_drift = bool(sidecar_active_baseline) and sidecar_active_baseline != canonical_anchor

    challenge_pair = dict(forward_blocker_artifact.get("challenge_pair") or {})
    blocker_baseline_params = normalize_exit_params(challenge_pair.get("baseline_exit_params"))
    blocker_baseline_slug = (
        format_exit_risk_anchor_slug(blocker_baseline_params)
        if int(blocker_baseline_params.get("max_hold_bars") or 0) > 0
        else ""
    )
    blocker_drift = bool(blocker_baseline_slug) and blocker_baseline_slug != canonical_anchor
    return sidecar_drift or blocker_drift


def select_canonical_anchor_exit_params(
    *,
    canonical_handoff_artifact: dict[str, Any],
    forward_blocker_artifact: dict[str, Any],
    exit_risk_artifact: dict[str, Any],
) -> dict[str, Any]:
    canonical_anchor = text(canonical_handoff_artifact.get("active_baseline"))
    if not canonical_anchor:
        return {}

    challenge_pair = dict(forward_blocker_artifact.get("challenge_pair") or {})
    candidates = [
        normalize_exit_params(challenge_pair.get("challenger_exit_params")),
        normalize_exit_params(challenge_pair.get("baseline_exit_params")),
        normalize_exit_params(exit_risk_artifact.get("selected_exit_params")),
        normalize_exit_params(exit_risk_artifact.get("validation_leader_exit_params")),
    ]
    for candidate in candidates:
        if int(candidate.get("max_hold_bars") or 0) <= 0:
            continue
        if format_exit_risk_anchor_slug(candidate) == canonical_anchor:
            return dict(candidate)

    canonical_hold_bars = extract_hold_bars(canonical_anchor)
    for candidate in candidates:
        if int(candidate.get("max_hold_bars") or 0) == canonical_hold_bars:
            return dict(candidate)
    return {}


def write_canonical_anchor_seed(
    *,
    workspace: Path,
    review_dir: Path,
    symbol: str,
    runtime_now: dt.datetime,
    offset: int,
    dataset_path: Path,
    base_artifact_path: Path,
    exit_risk_path: Path,
    canonical_handoff_json_path: Path,
    canonical_handoff_artifact: dict[str, Any],
    canonical_exit_params: dict[str, Any],
) -> dict[str, Any]:
    anchor_slug = format_exit_risk_anchor_slug(canonical_exit_params)
    anchor_seed_stamp = offset_stamp(runtime_now, offset)
    anchor_seed_path = (
        review_dir / f"{anchor_seed_stamp}_price_action_breakout_pullback_exit_risk_canonical_anchor_seed_sim_only.json"
    )
    anchor_seed_latest_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_canonical_anchor_seed_sim_only.json"
    anchor_seed_payload = {
        "action": "build_price_action_breakout_pullback_exit_risk_canonical_anchor_seed_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(runtime_now + dt.timedelta(seconds=offset)),
        "symbol": symbol,
        "family": "price_action_breakout_pullback",
        "workspace": str(workspace),
        "dataset_path": str(dataset_path),
        "base_artifact_path": str(base_artifact_path),
        "exit_risk_path": str(exit_risk_path),
        "canonical_handoff_path": str(canonical_handoff_json_path),
        "research_decision": "canonical_anchor_seed_ready_for_consumer_refresh",
        "source_head_status": "source_owned_consumer_refresh_seed",
        "active_baseline": anchor_slug,
        "active_baseline_hold_bars": int(canonical_exit_params.get("max_hold_bars") or 0),
        "active_baseline_exit_params": dict(canonical_exit_params),
        "selected_exit_params": dict(canonical_exit_params),
        "validation_leader_exit_params": dict(canonical_exit_params),
        "superseded_anchor": text(canonical_handoff_artifact.get("superseded_anchor")),
        "watch_candidate": text(canonical_handoff_artifact.get("watch_candidate")),
        "next_research_priority": text(canonical_handoff_artifact.get("next_research_priority"))
        or "refresh_canonical_exit_risk_consumers_after_challenger_promotion",
    }
    write_json(anchor_seed_path, anchor_seed_payload)
    write_json(anchor_seed_latest_path, anchor_seed_payload)
    return {
        "json_path": str(anchor_seed_path),
        "latest_json_path": str(anchor_seed_latest_path),
        "active_baseline": anchor_slug,
    }


def derive_aligned_lane_research_decision(
    *,
    canonical_handoff_artifact: dict[str, Any],
    primary_anchor_review_artifact: dict[str, Any],
    review_conclusion_artifact: dict[str, Any],
) -> str:
    canonical_conflict_active = (
        text(canonical_handoff_artifact.get("research_decision"))
        == "exit_risk_handoff_blocked_upstream_hold_selection_anchor_conflict"
        or text(canonical_handoff_artifact.get("source_head_status")) == "upstream_hold_selection_conflict"
    )
    primary_decision = text(primary_anchor_review_artifact.get("research_decision"))
    review_conclusion_decision = text(review_conclusion_artifact.get("research_decision"))
    if (
        canonical_conflict_active
        and primary_decision == "break_even_primary_anchor_review_complete_keep_baseline_anchor"
        and review_conclusion_decision == "break_even_review_conclusion_ready_keep_baseline_anchor_review_only"
    ):
        return "hold_selection_aligned_break_even_review_lane_ready_but_canonical_handoff_conflict_remains"
    if primary_decision:
        return (
            "hold_selection_aligned_break_even_review_lane_"
            + primary_decision.replace("break_even_primary_anchor_review_", "")
        )
    return "hold_selection_aligned_break_even_review_lane_inconclusive"


def build_hold_selection_aligned_break_even_review_lane(
    *,
    workspace: Path,
    system_root: Path,
    review_dir: Path,
    symbol: str,
    runtime_now: dt.datetime,
    start_offset: int,
    dataset_path: Path,
    base_artifact_path: Path,
    exit_risk_path: Path,
    exit_risk_artifact: dict[str, Any],
    canonical_handoff_json_path: Path,
    canonical_handoff_artifact: dict[str, Any],
    forward_blocker_path: str,
    forward_blocker_artifact: dict[str, Any],
    forward_consensus_path: str,
    break_even_train_days: list[int],
    validation_days: int,
    break_even_step_days: int,
    hold_selection_handoff_path: Path,
) -> dict[str, Any]:
    conflict_active = (
        text(canonical_handoff_artifact.get("research_decision"))
        == "exit_risk_handoff_blocked_upstream_hold_selection_anchor_conflict"
        or text(canonical_handoff_artifact.get("source_head_status")) == "upstream_hold_selection_conflict"
    )
    if not conflict_active:
        return {"enabled": False, "skip_reason": "canonical_handoff_not_in_upstream_hold_selection_conflict", "next_offset": start_offset - 1}

    hold_selection_handoff = (
        load_json_mapping(hold_selection_handoff_path)
        if hold_selection_handoff_path.is_file()
        else {}
    )
    if not hold_selection_handoff:
        return {"enabled": False, "skip_reason": "hold_selection_handoff_missing", "next_offset": start_offset - 1}

    aligned_exit_params = select_hold_aligned_exit_params(
        hold_selection_handoff=hold_selection_handoff,
        forward_blocker_artifact=forward_blocker_artifact,
        exit_risk_artifact=exit_risk_artifact,
    )
    if not int(aligned_exit_params.get("max_hold_bars") or 0):
        return {"enabled": False, "skip_reason": "no_hold_selection_aligned_exit_params_available", "next_offset": start_offset - 1}

    hold_selection_active_baseline = text(hold_selection_handoff.get("active_baseline"))
    hold_selection_active_hold_bars = resolve_hold_selection_active_hold_bars(hold_selection_handoff)
    lane_dir = review_dir / ALIGNED_BREAK_EVEN_REVIEW_DIRNAME
    lane_dir.mkdir(parents=True, exist_ok=True)
    offset = int(start_offset)

    anchor_slug = format_exit_risk_anchor_slug(aligned_exit_params)
    anchor_seed_stamp = offset_stamp(runtime_now, offset)
    anchor_seed_path = lane_dir / f"{anchor_seed_stamp}_price_action_breakout_pullback_exit_risk_hold_selection_aligned_anchor_seed_sim_only.json"
    anchor_seed_latest_path = lane_dir / "latest_price_action_breakout_pullback_exit_risk_hold_selection_aligned_anchor_seed_sim_only.json"
    anchor_seed_payload = {
        "action": "build_price_action_breakout_pullback_exit_risk_hold_selection_aligned_anchor_seed_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(runtime_now + dt.timedelta(seconds=offset)),
        "symbol": symbol,
        "family": "price_action_breakout_pullback",
        "workspace": str(workspace),
        "dataset_path": str(dataset_path),
        "base_artifact_path": str(base_artifact_path),
        "exit_risk_path": str(exit_risk_path),
        "canonical_handoff_path": str(canonical_handoff_json_path),
        "hold_selection_handoff_path": str(hold_selection_handoff_path),
        "research_decision": "hold_selection_aligned_break_even_anchor_seed_ready",
        "source_head_status": "review_only_seed",
        "review_lane_mode": "hold_selection_aligned_same_hold_break_even_review_only",
        "active_baseline": anchor_slug,
        "active_baseline_hold_bars": int(aligned_exit_params.get("max_hold_bars") or 0),
        "active_baseline_exit_params": aligned_exit_params,
        "selected_exit_params": aligned_exit_params,
        "validation_leader_exit_params": aligned_exit_params,
        "hold_selection_active_baseline": hold_selection_active_baseline,
        "hold_selection_active_hold_bars": hold_selection_active_hold_bars,
        "canonical_conflict_anchor": text(canonical_handoff_artifact.get("active_baseline")),
        "canonical_conflict_watch_candidate": text(canonical_handoff_artifact.get("watch_candidate")),
        "next_research_priority": "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail",
    }
    write_json(anchor_seed_path, anchor_seed_payload)
    write_json(anchor_seed_latest_path, anchor_seed_payload)

    offset += 1
    aligned_break_even_compare_paths: list[str] = []
    for train_days in break_even_train_days:
        step_stamp = offset_stamp(runtime_now, offset)
        compare_payload = run_json(
            name="build_exit_risk_break_even_forward_compare_aligned",
            cmd=[
                current_python_executable(),
                str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_forward_compare_sim_only.py"),
                "--dataset-path",
                str(dataset_path),
                "--base-artifact-path",
                str(base_artifact_path),
                "--exit-risk-path",
                str(anchor_seed_path),
                "--symbol",
                symbol,
                "--review-dir",
                str(lane_dir),
                "--stamp",
                step_stamp,
                "--train-days",
                str(train_days),
                "--validation-days",
                str(validation_days),
                "--step-days",
                str(break_even_step_days),
            ],
        )
        aligned_break_even_compare_paths.append(
            payload_path(
                compare_payload,
                name="build_exit_risk_break_even_forward_compare_aligned",
            )
        )
        offset += 1

    sidecar_stamp = offset_stamp(runtime_now, offset)
    aligned_sidecar_payload = run_json(
        name="build_exit_risk_break_even_sidecar_aligned",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.py"),
            "--review-dir",
            str(lane_dir),
            "--stamp",
            sidecar_stamp,
            *[
                item
                for compare_path in aligned_break_even_compare_paths
                for item in ("--compare-path", compare_path)
            ],
        ],
    )
    aligned_sidecar_path = payload_path(
        aligned_sidecar_payload,
        name="build_exit_risk_break_even_sidecar_aligned",
    )
    aligned_sidecar_artifact = artifact_or_payload(
        aligned_sidecar_path,
        aligned_sidecar_payload,
    )

    offset += 1
    aligned_handoff_stamp = offset_stamp(runtime_now, offset)
    aligned_handoff_path = lane_dir / f"{aligned_handoff_stamp}_price_action_breakout_pullback_exit_risk_hold_selection_aligned_handoff_sim_only.json"
    aligned_handoff_latest_path = lane_dir / "latest_price_action_breakout_pullback_exit_risk_hold_selection_aligned_handoff_sim_only.json"
    aligned_handoff_payload = {
        "action": "build_price_action_breakout_pullback_exit_risk_hold_selection_aligned_handoff_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(runtime_now + dt.timedelta(seconds=offset)),
        "symbol": symbol,
        "family": "price_action_breakout_pullback",
        "canonical_source_head": str(aligned_handoff_latest_path),
        "review_lane_mode": "hold_selection_aligned_same_hold_break_even_review_only",
        "review_lane_scope": "same_hold_same_trailing_break_even_delta_only",
        "canonical_scope": "review_only_lane",
        "research_decision": "use_exit_risk_handoff_as_canonical_anchor",
        "source_head_status": "baseline_anchor_active",
        "active_baseline": anchor_slug,
        "active_baseline_hold_bars": int(aligned_exit_params.get("max_hold_bars") or 0),
        "active_baseline_exit_params": aligned_exit_params,
        "watch_candidate": text(aligned_sidecar_artifact.get("watch_candidate")),
        "hold_selection_handoff_path": str(hold_selection_handoff_path),
        "hold_selection_active_baseline": hold_selection_active_baseline,
        "hold_selection_active_hold_bars": hold_selection_active_hold_bars,
        "upstream_hold_alignment_state": "aligned_review_only_lane",
        "blocked_now": [
            "promote_hold24_conflict_anchor_as_canonical_mainline_exit_risk_source_head",
        ],
        "allowed_now": [
            "run_break_even_candidate_guarded_review_packet",
            "keep_hold_selection_baseline_as_upstream_mainline_gate",
            "treat_hold_selection_aligned_break_even_candidate_as_review_only_sidecar",
        ],
        "next_research_priority": (
            text(aligned_sidecar_artifact.get("next_research_priority"))
            or "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail"
        ),
        "consumer_rule": (
            "这是一条 hold-selection 对齐的 review-only lane，不替代 "
            "`latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json` 的 canonical mainline；"
            "只用于 same-hold break-even 评审证据链。"
        ),
        "canonical_handoff_path": str(canonical_handoff_json_path),
        "canonical_handoff_research_decision": text(canonical_handoff_artifact.get("research_decision")),
        "canonical_handoff_source_head_status": text(canonical_handoff_artifact.get("source_head_status")),
    }
    write_json(aligned_handoff_path, aligned_handoff_payload)
    write_json(aligned_handoff_latest_path, aligned_handoff_payload)

    offset += 1
    guarded_review_stamp = offset_stamp(runtime_now, offset)
    aligned_guarded_review_payload = run_json(
        name="build_exit_risk_guarded_review_aligned",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_guarded_review_sim_only.py"),
            "--handoff-path",
            str(aligned_handoff_path),
            "--forward-blocker-path",
            str(forward_blocker_path),
            "--break-even-sidecar-path",
            aligned_sidecar_path,
            "--review-dir",
            str(lane_dir),
            "--stamp",
            guarded_review_stamp,
        ],
    )
    aligned_guarded_review_path = payload_path(
        aligned_guarded_review_payload,
        name="build_exit_risk_guarded_review_aligned",
    )
    aligned_guarded_review_artifact = artifact_or_payload(
        aligned_guarded_review_path,
        aligned_guarded_review_payload,
    )

    offset += 1
    review_packet_stamp = offset_stamp(runtime_now, offset)
    aligned_review_packet_payload = run_json(
        name="build_exit_risk_review_packet_aligned",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_review_packet_sim_only.py"),
            "--guarded-review-path",
            aligned_guarded_review_path,
            "--handoff-path",
            str(aligned_handoff_path),
            "--break-even-sidecar-path",
            aligned_sidecar_path,
            "--review-dir",
            str(lane_dir),
            "--stamp",
            review_packet_stamp,
        ],
    )
    aligned_review_packet_path = payload_path(
        aligned_review_packet_payload,
        name="build_exit_risk_review_packet_aligned",
    )
    aligned_review_packet_artifact = artifact_or_payload(
        aligned_review_packet_path,
        aligned_review_packet_payload,
    )

    offset += 1
    review_conclusion_stamp = offset_stamp(runtime_now, offset)
    aligned_review_conclusion_payload = run_json(
        name="build_exit_risk_review_conclusion_aligned",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_review_conclusion_sim_only.py"),
            "--review-packet-path",
            aligned_review_packet_path,
            "--guarded-review-path",
            aligned_guarded_review_path,
            "--handoff-path",
            str(aligned_handoff_path),
            "--review-dir",
            str(lane_dir),
            "--stamp",
            review_conclusion_stamp,
        ],
    )
    aligned_review_conclusion_path = payload_path(
        aligned_review_conclusion_payload,
        name="build_exit_risk_review_conclusion_aligned",
    )
    aligned_review_conclusion_artifact = artifact_or_payload(
        aligned_review_conclusion_path,
        aligned_review_conclusion_payload,
    )

    offset += 1
    primary_anchor_review_stamp = offset_stamp(runtime_now, offset)
    aligned_primary_anchor_review_payload = run_json(
        name="build_exit_risk_primary_anchor_review_aligned",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_primary_anchor_review_sim_only.py"),
            "--review-conclusion-path",
            aligned_review_conclusion_path,
            "--review-packet-path",
            aligned_review_packet_path,
            "--handoff-path",
            str(aligned_handoff_path),
            "--forward-consensus-path",
            str(forward_consensus_path),
            "--review-dir",
            str(lane_dir),
            "--stamp",
            primary_anchor_review_stamp,
        ],
    )
    aligned_primary_anchor_review_path = payload_path(
        aligned_primary_anchor_review_payload,
        name="build_exit_risk_primary_anchor_review_aligned",
    )
    aligned_primary_anchor_review_artifact = artifact_or_payload(
        aligned_primary_anchor_review_path,
        aligned_primary_anchor_review_payload,
    )

    offset += 1
    summary_stamp = offset_stamp(runtime_now, offset)
    summary_path = review_dir / f"{summary_stamp}_price_action_breakout_pullback_exit_risk_hold_selection_aligned_break_even_review_lane_sim_only.json"
    summary_latest_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_hold_selection_aligned_break_even_review_lane_sim_only.json"
    summary_payload = {
        "action": "build_price_action_breakout_pullback_exit_risk_hold_selection_aligned_break_even_review_lane_sim_only",
        "ok": True,
        "status": "ok",
        "enabled": True,
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(runtime_now + dt.timedelta(seconds=offset)),
        "symbol": symbol,
        "family": "price_action_breakout_pullback",
        "review_dir": str(lane_dir),
        "anchor_seed_path": str(anchor_seed_path),
        "anchor_seed_latest_json_path": str(anchor_seed_latest_path),
        "handoff_path": str(aligned_handoff_path),
        "handoff_latest_json_path": str(aligned_handoff_latest_path),
        "break_even_compare_paths": aligned_break_even_compare_paths,
        "break_even_sidecar_path": aligned_sidecar_path,
        "break_even_sidecar_latest_json_path": text(aligned_sidecar_payload.get("latest_json_path")),
        "break_even_sidecar_research_decision": text(
            aligned_sidecar_payload.get("research_decision")
            or aligned_sidecar_artifact.get("research_decision")
        ),
        "guarded_review_path": aligned_guarded_review_path,
        "guarded_review_research_decision": text(
            aligned_guarded_review_payload.get("research_decision")
            or aligned_guarded_review_artifact.get("research_decision")
        ),
        "review_packet_path": aligned_review_packet_path,
        "review_packet_research_decision": text(
            aligned_review_packet_payload.get("research_decision")
            or aligned_review_packet_artifact.get("research_decision")
        ),
        "review_conclusion_path": aligned_review_conclusion_path,
        "review_conclusion_research_decision": text(
            aligned_review_conclusion_payload.get("research_decision")
            or aligned_review_conclusion_artifact.get("research_decision")
        ),
        "review_conclusion_arbitration_state": text(
            aligned_review_conclusion_payload.get("arbitration_state")
            or aligned_review_conclusion_artifact.get("arbitration_state")
        ),
        "primary_anchor_review_path": aligned_primary_anchor_review_path,
        "primary_anchor_review_research_decision": text(
            aligned_primary_anchor_review_payload.get("research_decision")
            or aligned_primary_anchor_review_artifact.get("research_decision")
        ),
        "primary_anchor_review_state": text(
            aligned_primary_anchor_review_payload.get("review_state")
            or aligned_primary_anchor_review_artifact.get("review_state")
        ),
        "active_baseline": anchor_slug,
        "preferred_watch_candidate": text(aligned_sidecar_artifact.get("watch_candidate")),
        "canonical_handoff_path": str(canonical_handoff_json_path),
        "canonical_handoff_research_decision": text(canonical_handoff_artifact.get("research_decision")),
        "canonical_handoff_source_head_status": text(canonical_handoff_artifact.get("source_head_status")),
        "hold_selection_handoff_path": str(hold_selection_handoff_path),
        "hold_selection_active_baseline": hold_selection_active_baseline,
        "hold_selection_active_hold_bars": hold_selection_active_hold_bars,
        "research_decision": derive_aligned_lane_research_decision(
            canonical_handoff_artifact=canonical_handoff_artifact,
            primary_anchor_review_artifact=aligned_primary_anchor_review_artifact,
            review_conclusion_artifact=aligned_review_conclusion_artifact,
        ),
        "next_research_priority": (
            text(aligned_primary_anchor_review_artifact.get("next_research_priority"))
            or text(aligned_review_conclusion_artifact.get("next_research_priority"))
            or "integrate_hold_selection_aligned_break_even_review_path_without_repromoting_conflict_anchor"
        ),
        "consumer_rule": (
            "当 canonical exit/risk handoff 被 upstream hold selection conflict 阻断时，"
            "若要读取 same-hold break-even review 证据，必须先读这条 aligned review lane summary，"
            "再穿透到 lane 内部 guarded review / review packet / review conclusion / primary anchor review。"
        ),
    }
    write_json(summary_path, summary_payload)
    write_json(summary_latest_path, summary_payload)
    summary_payload["json_path"] = str(summary_path)
    summary_payload["latest_json_path"] = str(summary_latest_path)
    summary_payload["next_offset"] = offset
    return summary_payload


def parse_int_list(raw: str, default: list[int]) -> list[int]:
    parsed = [int(chunk.strip()) for chunk in text(raw).split(",") if chunk.strip()]
    values = parsed or list(default)
    if not values:
        raise ValueError("empty_train_day_list")
    if any(day <= 0 for day in values):
        raise ValueError("train_days_must_be_positive")
    unique_values = sorted(set(values))
    return unique_values


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


def resolve_dataset_path(
    *,
    explicit_dataset_path: str,
    exit_risk_payload: dict[str, Any],
    base_payload: dict[str, Any],
    review_dir: Path,
) -> Path:
    if text(explicit_dataset_path):
        return require_path(
            Path(explicit_dataset_path).expanduser().resolve(),
            "missing_dataset_path",
        )

    for payload in (exit_risk_payload, base_payload):
        dataset_text = text(payload.get("dataset_path"))
        if not dataset_text:
            continue
        candidate = Path(dataset_text).expanduser().resolve()
        if candidate.exists():
            return candidate

    return preferred_intraday_dataset(
        review_dir,
        error_code="no_public_intraday_crypto_bars_dataset_found",
    )


def offset_stamp(base_time: dt.datetime, seconds: int) -> str:
    return fmt_stamp(base_time + dt.timedelta(seconds=int(seconds)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full SIM_ONLY ETH exit/risk research chain and emit the canonical handoff as the single source-owned result."
    )
    parser.add_argument("--workspace", default=".", help="Fenlie workspace root or system directory.")
    parser.add_argument("--review-dir", default="", help="Optional explicit review directory override.")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--base-artifact-path", default="")
    parser.add_argument("--exit-risk-path", default="")
    parser.add_argument("--seed-blocker-path", default="")
    parser.add_argument("--hold-forward-stop-path", default="")
    parser.add_argument("--forward-train-days", default="30,40,45,50,55,60")
    parser.add_argument("--break-even-train-days", default="30,40,45,50,55,60")
    parser.add_argument("--break-even-step-days", type=int, default=DEFAULT_BREAK_EVEN_STEP_DAYS)
    parser.add_argument("--validation-days", type=int, default=10)
    parser.add_argument("--step-days", type=int, default=10)
    parser.add_argument("--now", help="Explicit UTC timestamp used to derive the shared builder stamp.")
    parser.add_argument("--refresh-panel", action="store_true", help="Optionally refresh the operator panel after the handoff is rebuilt.")
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
    forward_train_days = parse_int_list(args.forward_train_days, DEFAULT_FORWARD_TRAIN_DAYS)
    break_even_train_days = parse_int_list(args.break_even_train_days, DEFAULT_BREAK_EVEN_TRAIN_DAYS)

    base_artifact_path = (
        Path(args.base_artifact_path).expanduser().resolve()
        if text(args.base_artifact_path)
        else preferred_review_artifact(
            review_dir,
            latest_name="latest_price_action_breakout_pullback_sim_only.json",
            pattern="*_price_action_breakout_pullback_sim_only.json",
            error_code="no_price_action_breakout_pullback_sim_only_artifact_found",
        )
    )
    exit_risk_path = (
        Path(args.exit_risk_path).expanduser().resolve()
        if text(args.exit_risk_path)
        else preferred_review_artifact(
            review_dir,
            latest_name="latest_price_action_breakout_pullback_exit_risk_sim_only.json",
            pattern="*_price_action_breakout_pullback_exit_risk_sim_only.json",
            error_code="no_exit_risk_sim_only_artifact_found",
        )
    )
    base_artifact = load_json_mapping(base_artifact_path)
    exit_risk_artifact = load_json_mapping(exit_risk_path)
    dataset_path = resolve_dataset_path(
        explicit_dataset_path=args.dataset_path,
        exit_risk_payload=exit_risk_artifact,
        base_payload=base_artifact,
        review_dir=review_dir,
    )
    explicit_seed_blocker_path = (
        Path(args.seed_blocker_path).expanduser().resolve() if text(args.seed_blocker_path) else None
    )
    explicit_hold_forward_stop_path = (
        Path(args.hold_forward_stop_path).expanduser().resolve() if text(args.hold_forward_stop_path) else None
    )

    stop_refresh_payload: dict[str, Any] | None = None
    hold_upstream_refresh_payload: dict[str, Any] | None = None
    hold_upstream_handoff_path = ""
    hold_upstream_handoff_latest_json_path = ""
    if explicit_hold_forward_stop_path is not None:
        hold_forward_stop_path = explicit_hold_forward_stop_path
        hold_forward_stop_latest_json_path = ""
    else:
        upstream_refresh_offset = 0
        python_exec = current_python_executable()
        upstream_cmd = [
            python_exec,
            str(system_root / "scripts" / "run_price_action_breakout_pullback_hold_upstream_refresh_sim_only.py"),
            "--workspace",
            str(workspace),
            "--review-dir",
            str(review_dir),
            "--symbol",
            symbol,
            "--base-artifact-path",
            str(base_artifact_path),
            "--now",
            fmt_utc(runtime_now + dt.timedelta(seconds=upstream_refresh_offset)),
        ]
        hold_upstream_refresh_payload = run_json(
            name="run_hold_upstream_refresh",
            cmd=upstream_cmd,
        )
        hold_forward_stop_path = Path(
            payload_path(
                hold_upstream_refresh_payload,
                name="run_hold_upstream_refresh",
                key="stop_condition_path",
            )
        ).expanduser().resolve()
        hold_forward_stop_latest_json_path = text(
            hold_upstream_refresh_payload.get("stop_condition_latest_json_path")
        )
        hold_upstream_handoff_path = text(hold_upstream_refresh_payload.get("handoff_path"))
        hold_upstream_handoff_latest_json_path = text(
            hold_upstream_refresh_payload.get("handoff_latest_json_path")
        )
    if explicit_hold_forward_stop_path is None and hold_upstream_refresh_payload is None:
        stop_refresh_offset = 0
        stop_refresh_payload = run_json(
            name="run_exit_hold_forward_stop_refresh",
            cmd=[
                current_python_executable(),
                str(system_root / "scripts" / "run_price_action_breakout_pullback_exit_hold_forward_stop_refresh_sim_only.py"),
                "--workspace",
                str(workspace),
                "--review-dir",
                str(review_dir),
                "--now",
                fmt_utc(runtime_now + dt.timedelta(seconds=stop_refresh_offset)),
            ],
        )
        hold_forward_stop_path = Path(
            payload_path(stop_refresh_payload, name="run_exit_hold_forward_stop_refresh")
        ).expanduser().resolve()
        hold_forward_stop_latest_json_path = text(stop_refresh_payload.get("latest_json_path"))

    if explicit_seed_blocker_path is not None:
        seed_blocker_path = explicit_seed_blocker_path
    else:
        seed_blocker_offset = 1 if (hold_upstream_refresh_payload is not None or stop_refresh_payload is not None) else 0
        seed_blocker_stamp = offset_stamp(runtime_now, seed_blocker_offset)
        seed_blocker_payload = run_json(
            name="build_exit_risk_forward_blocker",
            cmd=[
                current_python_executable(),
                str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.py"),
                "--exit-risk-path",
                str(exit_risk_path),
                "--hold-forward-stop-path",
                str(hold_forward_stop_path),
                "--review-dir",
                str(review_dir),
                "--stamp",
                seed_blocker_stamp,
            ],
        )
        seed_blocker_path = Path(
            payload_path(seed_blocker_payload, name="build_exit_risk_forward_blocker")
        ).expanduser().resolve()

    forward_compare_paths: list[str] = []
    forward_compare_start_offset = 2 if (hold_upstream_refresh_payload is not None or stop_refresh_payload is not None) else 1
    for index, train_days in enumerate(forward_train_days):
        step_stamp = offset_stamp(runtime_now, forward_compare_start_offset + index)
        payload = run_json(
            name="build_exit_risk_forward_compare",
            cmd=[
                current_python_executable(),
                str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_forward_compare_sim_only.py"),
                "--dataset-path",
                str(dataset_path),
                "--base-artifact-path",
                str(base_artifact_path),
                "--challenge-pair-path",
                str(seed_blocker_path),
                "--symbol",
                symbol,
                "--review-dir",
                str(review_dir),
                "--stamp",
                step_stamp,
                "--train-days",
                str(train_days),
                "--validation-days",
                str(args.validation_days),
                "--step-days",
                str(args.step_days),
            ],
        )
        forward_compare_paths.append(payload_path(payload, name="build_exit_risk_forward_compare"))

    consensus_offset = forward_compare_start_offset + len(forward_train_days)
    forward_consensus_stamp = offset_stamp(runtime_now, consensus_offset)
    forward_consensus_payload = run_json(
        name="build_exit_risk_forward_consensus",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.py"),
            "--review-dir",
            str(review_dir),
            "--stamp",
            forward_consensus_stamp,
            *[item for compare_path in forward_compare_paths for item in ("--compare-path", compare_path)],
        ],
    )
    forward_consensus_path = payload_path(forward_consensus_payload, name="build_exit_risk_forward_consensus")

    break_even_compare_paths: list[str] = []
    break_even_start_offset = consensus_offset + 1
    for index, train_days in enumerate(break_even_train_days):
        step_stamp = offset_stamp(runtime_now, break_even_start_offset + index)
        payload = run_json(
            name="build_exit_risk_break_even_forward_compare",
            cmd=[
                current_python_executable(),
                str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_forward_compare_sim_only.py"),
                "--dataset-path",
                str(dataset_path),
                "--base-artifact-path",
                str(base_artifact_path),
                "--exit-risk-path",
                str(exit_risk_path),
                "--symbol",
                symbol,
                "--review-dir",
                str(review_dir),
                "--stamp",
                step_stamp,
                "--train-days",
                str(train_days),
                "--validation-days",
                str(args.validation_days),
                "--step-days",
                str(args.break_even_step_days),
            ],
        )
        break_even_compare_paths.append(payload_path(payload, name="build_exit_risk_break_even_forward_compare"))

    break_even_sidecar_offset = break_even_start_offset + len(break_even_train_days)
    break_even_sidecar_stamp = offset_stamp(runtime_now, break_even_sidecar_offset)
    break_even_sidecar_payload = run_json(
        name="build_exit_risk_break_even_sidecar",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.py"),
            "--review-dir",
            str(review_dir),
            "--stamp",
            break_even_sidecar_stamp,
            *[item for compare_path in break_even_compare_paths for item in ("--compare-path", compare_path)],
        ],
    )
    break_even_sidecar_path = payload_path(break_even_sidecar_payload, name="build_exit_risk_break_even_sidecar")

    tail_capacity_offset = break_even_sidecar_offset + 1
    tail_capacity_stamp = offset_stamp(runtime_now, tail_capacity_offset)
    tail_capacity_payload = run_json(
        name="build_exit_risk_forward_tail_capacity",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.py"),
            "--dataset-path",
            str(dataset_path),
            "--symbol",
            symbol,
            "--review-dir",
            str(review_dir),
            "--stamp",
            tail_capacity_stamp,
            "--validation-days",
            str(args.validation_days),
            "--step-days",
            str(args.step_days),
            "--candidate-train-days",
            ",".join(str(day) for day in forward_train_days),
        ],
    )
    tail_capacity_path = payload_path(tail_capacity_payload, name="build_exit_risk_forward_tail_capacity")

    blocker_offset = tail_capacity_offset + 1
    forward_blocker_stamp = offset_stamp(runtime_now, blocker_offset)
    forward_blocker_payload = run_json(
        name="build_exit_risk_forward_blocker",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.py"),
            "--exit-risk-path",
            str(exit_risk_path),
            "--hold-forward-stop-path",
            str(hold_forward_stop_path),
            "--forward-consensus-path",
            forward_consensus_path,
            "--break-even-sidecar-path",
            break_even_sidecar_path,
            "--tail-capacity-path",
            tail_capacity_path,
            "--review-dir",
            str(review_dir),
            "--stamp",
            forward_blocker_stamp,
        ],
    )
    forward_blocker_path = payload_path(forward_blocker_payload, name="build_exit_risk_forward_blocker")

    handoff_offset = blocker_offset + 1
    handoff_stamp = offset_stamp(runtime_now, handoff_offset)
    handoff_payload = run_json(
        name="build_exit_risk_handoff",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_handoff_sim_only.py"),
            "--exit-risk-path",
            str(exit_risk_path),
            "--forward-blocker-path",
            forward_blocker_path,
            "--forward-consensus-path",
            forward_consensus_path,
            "--break-even-sidecar-path",
            break_even_sidecar_path,
            "--tail-capacity-path",
            tail_capacity_path,
            "--review-dir",
            str(review_dir),
            "--stamp",
            handoff_stamp,
        ],
    )
    handoff_json_path = Path(payload_path(handoff_payload, name="build_exit_risk_handoff")).expanduser().resolve()
    handoff_artifact = load_json_mapping(handoff_json_path) if handoff_json_path.exists() else dict(handoff_payload)
    break_even_sidecar_artifact = artifact_or_payload(
        break_even_sidecar_path,
        break_even_sidecar_payload,
    )
    current_break_even_compare_paths = list(break_even_compare_paths)
    current_break_even_sidecar_payload = break_even_sidecar_payload
    current_break_even_sidecar_path = break_even_sidecar_path
    current_break_even_sidecar_artifact = break_even_sidecar_artifact
    current_forward_blocker_payload = forward_blocker_payload
    current_forward_blocker_path = forward_blocker_path
    current_forward_blocker_artifact = artifact_or_payload(
        forward_blocker_path,
        forward_blocker_payload,
    )
    current_handoff_payload = handoff_payload
    current_handoff_json_path = handoff_json_path
    current_handoff_artifact = handoff_artifact
    current_handoff_offset = handoff_offset
    current_exit_risk_consumer_path = exit_risk_path
    canonical_anchor_seed_payload: dict[str, Any] | None = None

    if canonical_anchor_consumer_refresh_required(
        canonical_handoff_artifact=current_handoff_artifact,
        forward_blocker_artifact=current_forward_blocker_artifact,
        break_even_sidecar_artifact=current_break_even_sidecar_artifact,
    ):
        canonical_exit_params = select_canonical_anchor_exit_params(
            canonical_handoff_artifact=current_handoff_artifact,
            forward_blocker_artifact=current_forward_blocker_artifact,
            exit_risk_artifact=exit_risk_artifact,
        )
        if int(canonical_exit_params.get("max_hold_bars") or 0) > 0:
            seed_offset = current_handoff_offset + 1
            canonical_anchor_seed_payload = write_canonical_anchor_seed(
                workspace=workspace,
                review_dir=review_dir,
                symbol=symbol,
                runtime_now=runtime_now,
                offset=seed_offset,
                dataset_path=dataset_path,
                base_artifact_path=base_artifact_path,
                exit_risk_path=exit_risk_path,
                canonical_handoff_json_path=current_handoff_json_path,
                canonical_handoff_artifact=current_handoff_artifact,
                canonical_exit_params=canonical_exit_params,
            )
            current_exit_risk_consumer_path = Path(
                payload_path(canonical_anchor_seed_payload, name="write_canonical_anchor_seed")
            ).expanduser().resolve()

            refreshed_break_even_compare_paths: list[str] = []
            refresh_break_even_start_offset = seed_offset + 1
            for index, train_days in enumerate(break_even_train_days):
                step_stamp = offset_stamp(runtime_now, refresh_break_even_start_offset + index)
                payload = run_json(
                    name="build_exit_risk_break_even_forward_compare",
                    cmd=[
                        current_python_executable(),
                        str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_forward_compare_sim_only.py"),
                        "--dataset-path",
                        str(dataset_path),
                        "--base-artifact-path",
                        str(base_artifact_path),
                        "--exit-risk-path",
                        str(current_exit_risk_consumer_path),
                        "--symbol",
                        symbol,
                        "--review-dir",
                        str(review_dir),
                        "--stamp",
                        step_stamp,
                        "--train-days",
                        str(train_days),
                        "--validation-days",
                        str(args.validation_days),
                        "--step-days",
                        str(args.break_even_step_days),
                    ],
                )
                refreshed_break_even_compare_paths.append(
                    payload_path(payload, name="build_exit_risk_break_even_forward_compare")
                )

            current_break_even_compare_paths = refreshed_break_even_compare_paths
            refresh_sidecar_offset = refresh_break_even_start_offset + len(break_even_train_days)
            refresh_sidecar_stamp = offset_stamp(runtime_now, refresh_sidecar_offset)
            current_break_even_sidecar_payload = run_json(
                name="build_exit_risk_break_even_sidecar",
                cmd=[
                    current_python_executable(),
                    str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.py"),
                    "--review-dir",
                    str(review_dir),
                    "--stamp",
                    refresh_sidecar_stamp,
                    *[
                        item
                        for compare_path in current_break_even_compare_paths
                        for item in ("--compare-path", compare_path)
                    ],
                ],
            )
            current_break_even_sidecar_path = payload_path(
                current_break_even_sidecar_payload,
                name="build_exit_risk_break_even_sidecar",
            )
            current_break_even_sidecar_artifact = artifact_or_payload(
                current_break_even_sidecar_path,
                current_break_even_sidecar_payload,
            )

            refresh_blocker_offset = refresh_sidecar_offset + 1
            refresh_blocker_stamp = offset_stamp(runtime_now, refresh_blocker_offset)
            current_forward_blocker_payload = run_json(
                name="build_exit_risk_forward_blocker",
                cmd=[
                    current_python_executable(),
                    str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.py"),
                    "--exit-risk-path",
                    str(current_exit_risk_consumer_path),
                    "--hold-forward-stop-path",
                    str(hold_forward_stop_path),
                    "--forward-consensus-path",
                    forward_consensus_path,
                    "--break-even-sidecar-path",
                    current_break_even_sidecar_path,
                    "--tail-capacity-path",
                    tail_capacity_path,
                    "--review-dir",
                    str(review_dir),
                    "--stamp",
                    refresh_blocker_stamp,
                ],
            )
            current_forward_blocker_path = payload_path(
                current_forward_blocker_payload,
                name="build_exit_risk_forward_blocker",
            )
            current_forward_blocker_artifact = artifact_or_payload(
                current_forward_blocker_path,
                current_forward_blocker_payload,
            )

            refresh_handoff_offset = refresh_blocker_offset + 1
            refresh_handoff_stamp = offset_stamp(runtime_now, refresh_handoff_offset)
            current_handoff_payload = run_json(
                name="build_exit_risk_handoff",
                cmd=[
                    current_python_executable(),
                    str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_handoff_sim_only.py"),
                    "--exit-risk-path",
                    str(current_exit_risk_consumer_path),
                    "--forward-blocker-path",
                    current_forward_blocker_path,
                    "--forward-consensus-path",
                    forward_consensus_path,
                    "--break-even-sidecar-path",
                    current_break_even_sidecar_path,
                    "--tail-capacity-path",
                    tail_capacity_path,
                    "--review-dir",
                    str(review_dir),
                    "--stamp",
                    refresh_handoff_stamp,
                ],
            )
            current_handoff_json_path = Path(
                payload_path(current_handoff_payload, name="build_exit_risk_handoff")
            ).expanduser().resolve()
            current_handoff_artifact = (
                load_json_mapping(current_handoff_json_path)
                if current_handoff_json_path.exists()
                else dict(current_handoff_payload)
            )
            current_handoff_offset = refresh_handoff_offset

    break_even_compare_paths = current_break_even_compare_paths
    break_even_sidecar_payload = current_break_even_sidecar_payload
    break_even_sidecar_path = current_break_even_sidecar_path
    forward_blocker_payload = current_forward_blocker_payload
    forward_blocker_path = current_forward_blocker_path
    handoff_payload = current_handoff_payload
    handoff_json_path = current_handoff_json_path
    handoff_artifact = current_handoff_artifact
    handoff_offset = current_handoff_offset

    guarded_review_offset = handoff_offset + 1
    guarded_review_stamp = offset_stamp(runtime_now, guarded_review_offset)
    guarded_review_payload = run_json(
        name="build_exit_risk_guarded_review",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_guarded_review_sim_only.py"),
            "--handoff-path",
            str(handoff_json_path),
            "--forward-blocker-path",
            forward_blocker_path,
            "--break-even-sidecar-path",
            break_even_sidecar_path,
            "--review-dir",
            str(review_dir),
            "--stamp",
            guarded_review_stamp,
        ],
    )
    guarded_review_path = payload_path(guarded_review_payload, name="build_exit_risk_guarded_review")

    review_packet_offset = guarded_review_offset + 1
    review_packet_stamp = offset_stamp(runtime_now, review_packet_offset)
    review_packet_payload = run_json(
        name="build_exit_risk_review_packet",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_review_packet_sim_only.py"),
            "--guarded-review-path",
            guarded_review_path,
            "--handoff-path",
            str(handoff_json_path),
            "--break-even-sidecar-path",
            break_even_sidecar_path,
            "--review-dir",
            str(review_dir),
            "--stamp",
            review_packet_stamp,
        ],
    )
    review_packet_path = payload_path(review_packet_payload, name="build_exit_risk_review_packet")

    review_conclusion_offset = review_packet_offset + 1
    review_conclusion_stamp = offset_stamp(runtime_now, review_conclusion_offset)
    review_conclusion_payload = run_json(
        name="build_exit_risk_review_conclusion",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_review_conclusion_sim_only.py"),
            "--review-packet-path",
            review_packet_path,
            "--guarded-review-path",
            guarded_review_path,
            "--handoff-path",
            str(handoff_json_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            review_conclusion_stamp,
        ],
    )
    review_conclusion_path = payload_path(review_conclusion_payload, name="build_exit_risk_review_conclusion")
    review_conclusion_json_path = Path(review_conclusion_path).expanduser().resolve()
    review_conclusion_artifact = (
        load_json_mapping(review_conclusion_json_path) if review_conclusion_json_path.exists() else dict(review_conclusion_payload)
    )

    primary_anchor_review_offset = review_conclusion_offset + 1
    primary_anchor_review_stamp = offset_stamp(runtime_now, primary_anchor_review_offset)
    primary_anchor_review_payload = run_json(
        name="build_exit_risk_primary_anchor_review",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_break_even_primary_anchor_review_sim_only.py"),
            "--review-conclusion-path",
            review_conclusion_path,
            "--review-packet-path",
            review_packet_path,
            "--handoff-path",
            str(handoff_json_path),
            "--forward-consensus-path",
            forward_consensus_path,
            "--review-dir",
            str(review_dir),
            "--stamp",
            primary_anchor_review_stamp,
        ],
    )
    primary_anchor_review_path = payload_path(primary_anchor_review_payload, name="build_exit_risk_primary_anchor_review")
    primary_anchor_review_json_path = Path(primary_anchor_review_path).expanduser().resolve()
    primary_anchor_review_artifact = (
        load_json_mapping(primary_anchor_review_json_path)
        if primary_anchor_review_json_path.exists()
        else dict(primary_anchor_review_payload)
    )

    hold_selection_handoff_path = review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json"
    aligned_break_even_review_lane = build_hold_selection_aligned_break_even_review_lane(
        workspace=workspace,
        system_root=system_root,
        review_dir=review_dir,
        symbol=symbol,
        runtime_now=runtime_now,
        start_offset=primary_anchor_review_offset + 1,
        dataset_path=dataset_path,
        base_artifact_path=base_artifact_path,
        exit_risk_path=exit_risk_path,
        exit_risk_artifact=exit_risk_artifact,
        canonical_handoff_json_path=handoff_json_path,
        canonical_handoff_artifact=handoff_artifact,
        forward_blocker_path=forward_blocker_path,
        forward_blocker_artifact=artifact_or_payload(forward_blocker_path, forward_blocker_payload),
        forward_consensus_path=forward_consensus_path,
        break_even_train_days=break_even_train_days,
        validation_days=int(args.validation_days),
        break_even_step_days=int(args.break_even_step_days),
        hold_selection_handoff_path=hold_selection_handoff_path,
    )

    refreshed_handoff_payload = handoff_payload
    refreshed_handoff_json_path = handoff_json_path
    refreshed_handoff_artifact = handoff_artifact
    refreshed_handoff_stamp = handoff_stamp
    refreshed_handoff_offset = int(aligned_break_even_review_lane.get("next_offset") or primary_anchor_review_offset)
    if bool(aligned_break_even_review_lane.get("enabled")):
        refreshed_handoff_offset += 1
        refreshed_handoff_stamp = offset_stamp(runtime_now, refreshed_handoff_offset)
        refreshed_handoff_payload = run_json(
            name="build_exit_risk_handoff",
            cmd=[
                current_python_executable(),
                str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_handoff_sim_only.py"),
                "--exit-risk-path",
                str(exit_risk_path),
                "--forward-blocker-path",
                forward_blocker_path,
                "--forward-consensus-path",
                forward_consensus_path,
                "--break-even-sidecar-path",
                break_even_sidecar_path,
                "--tail-capacity-path",
                tail_capacity_path,
                "--review-dir",
                str(review_dir),
                "--stamp",
                refreshed_handoff_stamp,
            ],
        )
        refreshed_handoff_json_path = Path(
            payload_path(refreshed_handoff_payload, name="build_exit_risk_handoff")
        ).expanduser().resolve()
        refreshed_handoff_artifact = (
            load_json_mapping(refreshed_handoff_json_path)
            if refreshed_handoff_json_path.exists()
            else dict(refreshed_handoff_payload)
        )

    source_gap_audit_offset = refreshed_handoff_offset + 1
    source_gap_audit_stamp = offset_stamp(runtime_now, source_gap_audit_offset)
    source_gap_audit_payload = run_json(
        name="build_exit_risk_source_gap_audit",
        cmd=[
            current_python_executable(),
            str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_source_gap_audit_sim_only.py"),
            "--workspace",
            str(workspace),
            "--review-dir",
            str(review_dir),
            "--stamp",
            source_gap_audit_stamp,
        ],
    )
    source_gap_audit_json_path = Path(
        payload_path(source_gap_audit_payload, name="build_exit_risk_source_gap_audit")
    ).expanduser().resolve()
    source_gap_audit_artifact = (
        load_json_mapping(source_gap_audit_json_path) if source_gap_audit_json_path.exists() else dict(source_gap_audit_payload)
    )

    panel_refresh_payload: dict[str, Any] | None = None
    if args.refresh_panel:
        panel_refresh_offset = source_gap_audit_offset + 1
        panel_cmd = [
            current_python_executable(),
            str(system_root / "scripts" / "run_operator_panel_refresh.py"),
            "--workspace",
            str(workspace),
        ]
        panel_cmd.extend(["--now", fmt_utc(runtime_now + dt.timedelta(seconds=panel_refresh_offset))])
        panel_refresh_payload = run_json(name="run_operator_panel_refresh", cmd=panel_cmd)

    print(
        json.dumps(
            {
                "ok": True,
                "mode": "exit_risk_research_chain_sim_only",
                "change_class": "SIM_ONLY",
                "stamp": stamp,
                "handoff_stamp": refreshed_handoff_stamp,
                "workspace": str(workspace),
                "review_dir": str(review_dir),
                "symbol": symbol,
                "dataset_path": str(dataset_path),
                "base_artifact_path": str(base_artifact_path),
                "exit_risk_path": str(exit_risk_path),
                "seed_blocker_path": str(seed_blocker_path),
                "hold_forward_stop_path": str(hold_forward_stop_path),
                "hold_forward_stop_latest_json_path": hold_forward_stop_latest_json_path,
                "hold_upstream_handoff_path": hold_upstream_handoff_path,
                "hold_upstream_handoff_latest_json_path": hold_upstream_handoff_latest_json_path,
                "forward_compare_train_days": forward_train_days,
                "break_even_train_days": break_even_train_days,
                "validation_days": int(args.validation_days),
                "step_days": int(args.step_days),
                "break_even_step_days": int(args.break_even_step_days),
                "forward_compare_paths": forward_compare_paths,
                "forward_consensus_path": forward_consensus_path,
                "forward_consensus_latest_json_path": text(forward_consensus_payload.get("latest_json_path")),
                "forward_blocker_path": forward_blocker_path,
                "forward_blocker_latest_json_path": text(forward_blocker_payload.get("latest_json_path")),
                "break_even_compare_paths": break_even_compare_paths,
                "break_even_sidecar_path": break_even_sidecar_path,
                "break_even_sidecar_latest_json_path": text(break_even_sidecar_payload.get("latest_json_path")),
                "tail_capacity_path": tail_capacity_path,
                "tail_capacity_latest_json_path": text(tail_capacity_payload.get("latest_json_path")),
                "guarded_review_path": guarded_review_path,
                "guarded_review_latest_json_path": text(guarded_review_payload.get("latest_json_path")),
                "guarded_review_research_decision": text(guarded_review_payload.get("research_decision")),
                "review_packet_path": review_packet_path,
                "review_packet_latest_json_path": text(review_packet_payload.get("latest_json_path")),
                "review_packet_research_decision": text(review_packet_payload.get("research_decision")),
                "review_conclusion_path": review_conclusion_path,
                "review_conclusion_latest_json_path": text(review_conclusion_payload.get("latest_json_path")),
                "review_conclusion_research_decision": text(
                    review_conclusion_payload.get("research_decision")
                    or review_conclusion_artifact.get("research_decision")
                ),
                "review_conclusion_arbitration_state": text(
                    review_conclusion_payload.get("arbitration_state")
                    or review_conclusion_artifact.get("arbitration_state")
                ),
                "primary_anchor_review_path": primary_anchor_review_path,
                "primary_anchor_review_latest_json_path": text(primary_anchor_review_payload.get("latest_json_path")),
                "primary_anchor_review_research_decision": text(
                    primary_anchor_review_payload.get("research_decision")
                    or primary_anchor_review_artifact.get("research_decision")
                ),
                "primary_anchor_review_state": text(
                    primary_anchor_review_payload.get("review_state")
                    or primary_anchor_review_artifact.get("review_state")
                ),
                "source_gap_audit_path": str(source_gap_audit_json_path),
                "source_gap_audit_latest_json_path": text(source_gap_audit_payload.get("latest_json_path")),
                "source_gap_audit_research_decision": text(
                    source_gap_audit_payload.get("research_decision")
                    or source_gap_audit_artifact.get("research_decision")
                ),
                "source_gap_audit_finding_count": int(
                    source_gap_audit_payload.get("finding_count")
                    or source_gap_audit_artifact.get("finding_count")
                    or 0
                ),
                "json_path": str(refreshed_handoff_json_path),
                "latest_json_path": text(refreshed_handoff_payload.get("latest_json_path")),
                "research_decision": text(
                    refreshed_handoff_payload.get("research_decision") or refreshed_handoff_artifact.get("research_decision")
                ),
                "source_head_status": text(
                    refreshed_handoff_payload.get("source_head_status") or refreshed_handoff_artifact.get("source_head_status")
                ),
                "active_baseline": text(refreshed_handoff_artifact.get("active_baseline")),
                "superseded_anchor": text(refreshed_handoff_artifact.get("superseded_anchor")),
                "transfer_watch": list(refreshed_handoff_artifact.get("transfer_watch") or []),
                "baseline_windows": int(refreshed_handoff_artifact.get("baseline_windows") or 0),
                "tie_windows": int(refreshed_handoff_artifact.get("tie_windows") or 0),
                "challenger_windows": int(refreshed_handoff_artifact.get("challenger_windows") or 0),
                "hold_selection_aligned_break_even_review_lane": {
                    key: value
                    for key, value in aligned_break_even_review_lane.items()
                    if key != "next_offset"
                },
                "hold_upstream_refresh": hold_upstream_refresh_payload,
                "stop_refresh": stop_refresh_payload,
                "panel_refresh": panel_refresh_payload,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
