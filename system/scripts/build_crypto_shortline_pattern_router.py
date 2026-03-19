#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5


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


def fmt_utc(value: dt.datetime | None) -> str:
    effective = value or now_utc()
    return effective.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def join_unique(parts: list[Any], *, sep: str = " | ") -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for part in parts:
        value = text(part)
        if not value or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return sep.join(ordered)


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def parsed_artifact_stamp(path: Path) -> dt.datetime | None:
    stamp = artifact_stamp(path)
    if not stamp:
        return None
    try:
        return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def artifact_sort_key(
    path: Path,
    reference_now: dt.datetime | None = None,
) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    effective_now = reference_now or now_utc()
    future_cutoff = effective_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    return (0 if is_future else 1, artifact_stamp(path), path.stat().st_mtime, path.name)


def find_latest(
    review_dir: Path,
    pattern: str,
    reference_now: dt.datetime | None = None,
) -> Path | None:
    files = sorted(
        review_dir.glob(pattern),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    if not files:
        return None
    future_cutoff = (reference_now or now_utc()) + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    for path in files:
        stamp_dt = parsed_artifact_stamp(path)
        if stamp_dt is None or stamp_dt <= future_cutoff:
            return path
    return files[0]


def payload_scope_symbol(payload: dict[str, Any]) -> str:
    for key in ("route_symbol", "symbol", "focus_symbol"):
        value = text(payload.get(key)).upper()
        if value:
            return value
    return ""


def payload_matches_symbol(payload: dict[str, Any], symbol: str) -> bool:
    normalized = text(symbol).upper()
    if not normalized:
        return True
    scoped = payload_scope_symbol(payload)
    return not scoped or scoped == normalized


def find_latest_scoped(
    review_dir: Path,
    pattern: str,
    symbol: str,
    reference_now: dt.datetime | None = None,
) -> Path | None:
    files = sorted(
        review_dir.glob(pattern),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    if not files:
        return None
    future_cutoff = (reference_now or now_utc()) + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    for path in files:
        stamp_dt = parsed_artifact_stamp(path)
        if stamp_dt is not None and stamp_dt > future_cutoff:
            continue
        try:
            payload = load_json_mapping(path)
        except Exception:
            continue
        if payload_matches_symbol(payload, symbol):
            return path
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prune_artifacts(
    review_dir: Path,
    *,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, float(ttl_hours)))
    protected = {path.name for path in current_paths}
    candidates: list[Path] = []
    for pattern in (
        "*_crypto_shortline_pattern_router.json",
        "*_crypto_shortline_pattern_router.md",
        "*_crypto_shortline_pattern_router_checksum.json",
    ):
        candidates.extend(review_dir.glob(pattern))

    existing: list[tuple[float, Path]] = []
    for path in candidates:
        try:
            existing.append((path.stat().st_mtime, path))
        except OSError:
            continue

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for _, path in sorted(existing, key=lambda item: item[0], reverse=True):
        if path.name in protected:
            survivors.append(path)
            continue
        try:
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            path.unlink(missing_ok=True)
            pruned_age.append(str(path))
        else:
            survivors.append(path)

    pruned_keep: list[str] = []
    for path in survivors[max(1, int(keep)) :]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Shortline Pattern Router",
            "",
            f"- brief: `{text(payload.get('pattern_brief'))}`",
            f"- family: `{text(payload.get('pattern_family'))}`",
            f"- stage: `{text(payload.get('pattern_stage'))}`",
            f"- status: `{text(payload.get('pattern_status'))}`",
            f"- decision: `{text(payload.get('pattern_decision'))}`",
            f"- structural_ready: `{payload.get('structural_ready')}`",
            f"- confidence_score: `{payload.get('pattern_confidence_score')}`",
            f"- profile_rotation_alignment_band: `{text(payload.get('profile_rotation_alignment_band')) or '-'}`",
            f"- profile_rotation_target_bin_distance: `{payload.get('profile_rotation_target_bin_distance')}`",
            f"- profile_rotation_target_distance_bps: `{payload.get('profile_rotation_target_distance_bps')}`",
            f"- underlying_target_artifact: `{text(payload.get('underlying_target_artifact')) or '-'}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def value_rotation_alignment_band(
    *, payload: dict[str, Any] | None = None, status: str, target_bin_distance: Any = None
) -> str:
    source_band = text(as_dict(payload).get("profile_rotation_alignment_band"))
    if source_band:
        return source_band
    status_text = text(status)
    if status_text.endswith("_final_band") or status_text.endswith("_final"):
        return "final"
    if status_text.endswith("_approaching"):
        return "approaching"
    try:
        distance_value = int(target_bin_distance)
    except (TypeError, ValueError):
        distance_value = None
    if distance_value is not None:
        if distance_value <= 1:
            return "final"
        if distance_value == 2:
            return "approaching"
    return "far"


def build_payload(
    *,
    reference_now: dt.datetime,
    route_symbol_override: str,
    route_focus_symbol: str,
    intent_payload: dict[str, Any],
    operator_payload: dict[str, Any],
    profile_payload: dict[str, Any],
    mss_payload: dict[str, Any],
    cvd_payload: dict[str, Any],
    retest_payload: dict[str, Any],
    setup_transition_payload: dict[str, Any],
    liquidity_event_payload: dict[str, Any],
    signal_quality_payload: dict[str, Any],
    sizing_payload: dict[str, Any],
) -> dict[str, Any]:
    route_symbol = (
        text(route_symbol_override)
        or text(intent_payload.get("preferred_route_symbol"))
        or text(operator_payload.get("review_priority_head_symbol"))
        or text(operator_payload.get("next_focus_symbol"))
        or "-"
    ).upper()
    route_action = text(intent_payload.get("preferred_route_action")) or text(
        operator_payload.get("next_focus_action")
    )
    remote_market = text(intent_payload.get("remote_market")) or "portfolio_margin_um"

    profile_status = text(profile_payload.get("watch_status"))
    profile_decision = text(profile_payload.get("watch_decision"))
    mss_status = text(mss_payload.get("watch_status"))
    mss_decision = text(mss_payload.get("watch_decision"))
    cvd_status = text(cvd_payload.get("watch_status"))
    cvd_decision = text(cvd_payload.get("watch_decision"))
    retest_status = text(retest_payload.get("watch_status"))
    retest_decision = text(retest_payload.get("watch_decision"))
    setup_status = text(setup_transition_payload.get("transition_status"))
    setup_decision = text(setup_transition_payload.get("transition_decision"))
    setup_primary_gate = text(setup_transition_payload.get("primary_missing_gate"))
    liquidity_event_status = text(liquidity_event_payload.get("trigger_status"))
    signal_quality_status = text(signal_quality_payload.get("watch_status"))
    sizing_status = text(sizing_payload.get("watch_status"))

    pattern_family = "pattern_watch_only"
    pattern_stage = "unqualified"
    pattern_status = "pattern_watch_only_unqualified"
    pattern_decision = "collect_more_structure_before_routing_pattern"
    blocker_title = "Collect more structure before shortline pattern promotion"
    underlying_target_artifact = "crypto_shortline_execution_gate"
    underlying_status = ""
    structural_ready = False
    done_when = (
        f"{route_symbol} resolves the active structural blocker and the selected shortline pattern is ready "
        "for execution-gate reassessment"
    )
    profile_rotation_target_bin_distance = profile_payload.get("profile_rotation_target_bin_distance")
    profile_rotation_target_distance_bps = profile_payload.get("profile_rotation_target_distance_bps")
    profile_rotation_alignment_band = ""

    if profile_status.startswith("profile_location_lvn_key_level_ready_rotation_"):
        pattern_family = "value_rotation_scalp"
        pattern_stage = "profile_alignment"
        structural_ready = False
        underlying_target_artifact = (
            text(profile_payload.get("next_action_target_artifact"))
            or text(profile_payload.get("blocker_target_artifact"))
            or "crypto_shortline_profile_location_watch"
        )
        underlying_status = profile_status
        profile_rotation_alignment_band = value_rotation_alignment_band(
            payload=profile_payload,
            status=profile_status,
            target_bin_distance=profile_rotation_target_bin_distance,
        )
        if profile_rotation_alignment_band == "final":
            pattern_status = "value_rotation_scalp_wait_profile_alignment_final"
            pattern_decision = "monitor_final_value_rotation_into_hvn_poc_then_recheck_execution_gate"
            blocker_title = "Track final value-rotation alignment before shortline scalp promotion"
        elif profile_rotation_alignment_band == "approaching":
            pattern_status = "value_rotation_scalp_wait_profile_alignment_approaching"
            pattern_decision = "monitor_value_rotation_into_final_band_then_recheck_execution_gate"
            blocker_title = "Track approaching value-rotation alignment before shortline scalp promotion"
        else:
            pattern_status = "value_rotation_scalp_wait_profile_alignment_far"
            pattern_decision = "monitor_value_rotation_toward_hvn_poc_then_recheck_execution_gate"
            blocker_title = "Track value-rotation alignment before shortline scalp promotion"
        done_when = (
            f"{route_symbol} rotates from LVN toward HVN/POC, then the shortline execution gate can reassess "
            "whether the value-rotation scalp is executable"
        )
    elif mss_status.startswith("value_rotation_scalp_mss_precondition_profile_alignment_"):
        pattern_family = "value_rotation_scalp"
        pattern_stage = "profile_alignment"
        structural_ready = False
        underlying_target_artifact = (
            text(mss_payload.get("next_action_target_artifact"))
            or text(mss_payload.get("blocker_target_artifact"))
            or "crypto_shortline_profile_location_watch"
        )
        underlying_status = mss_status
        profile_rotation_alignment_band = value_rotation_alignment_band(
            payload=mss_payload,
            status=mss_status,
            target_bin_distance=mss_payload.get("profile_rotation_target_bin_distance"),
        )
        if profile_rotation_alignment_band == "final":
            pattern_status = "value_rotation_scalp_wait_profile_alignment_final"
            pattern_decision = "monitor_final_value_rotation_into_hvn_poc_then_recheck_execution_gate"
            blocker_title = "Track final value-rotation alignment before shortline scalp promotion"
        elif profile_rotation_alignment_band == "approaching":
            pattern_status = "value_rotation_scalp_wait_profile_alignment_approaching"
            pattern_decision = "monitor_value_rotation_into_final_band_then_recheck_execution_gate"
            blocker_title = "Track approaching value-rotation alignment before shortline scalp promotion"
        else:
            pattern_status = "value_rotation_scalp_wait_profile_alignment_far"
            pattern_decision = "monitor_value_rotation_toward_hvn_poc_then_recheck_execution_gate"
            blocker_title = "Track value-rotation alignment before shortline scalp promotion"
        done_when = (
            f"{route_symbol} rotates from LVN toward HVN/POC so MSS can be reassessed for the "
            "value-rotation scalp"
        )
    elif mss_status.startswith("mss_waiting_after_liquidity_sweep"):
        pattern_family = "sweep_reversal"
        pattern_stage = "mss_confirmation"
        structural_ready = False
        pattern_status = "sweep_reversal_wait_mss"
        pattern_decision = "wait_for_sweep_reversal_mss_then_recheck_execution_gate"
        blocker_title = "Track sweep-reversal MSS before shortline scalp promotion"
        underlying_target_artifact = (
            text(mss_payload.get("next_action_target_artifact"))
            or text(mss_payload.get("blocker_target_artifact"))
            or "crypto_shortline_mss_watch"
        )
        underlying_status = mss_status
        done_when = (
            f"{route_symbol} confirms MSS/CHOCH after the detected liquidity sweep, then the shortline execution "
            "gate can reassess the sweep-reversal scalp"
        )
    elif retest_status.startswith("value_rotation_scalp_retest_precondition_profile_alignment_"):
        pattern_family = "value_rotation_scalp"
        pattern_stage = "profile_alignment"
        structural_ready = False
        underlying_target_artifact = (
            text(retest_payload.get("next_action_target_artifact"))
            or text(retest_payload.get("blocker_target_artifact"))
            or "crypto_shortline_profile_location_watch"
        )
        underlying_status = retest_status
        profile_rotation_alignment_band = value_rotation_alignment_band(
            payload=retest_payload,
            status=retest_status,
            target_bin_distance=retest_payload.get("profile_rotation_target_bin_distance"),
        )
        if profile_rotation_alignment_band == "final":
            pattern_status = "value_rotation_scalp_wait_profile_alignment_final"
            pattern_decision = "monitor_final_value_rotation_into_hvn_poc_then_recheck_execution_gate"
            blocker_title = "Track final value-rotation alignment before shortline scalp promotion"
        elif profile_rotation_alignment_band == "approaching":
            pattern_status = "value_rotation_scalp_wait_profile_alignment_approaching"
            pattern_decision = "monitor_value_rotation_into_final_band_then_recheck_execution_gate"
            blocker_title = "Track approaching value-rotation alignment before shortline scalp promotion"
        else:
            pattern_status = "value_rotation_scalp_wait_profile_alignment_far"
            pattern_decision = "monitor_value_rotation_toward_hvn_poc_then_recheck_execution_gate"
            blocker_title = "Track value-rotation alignment before shortline scalp promotion"
        done_when = (
            f"{route_symbol} rotates from LVN toward HVN/POC so retest readiness can be reassessed "
            "for the value-rotation scalp"
        )
    elif retest_status == "value_rotation_scalp_retest_precondition_mss_pending":
        pattern_family = "value_rotation_scalp"
        pattern_stage = "mss_confirmation"
        structural_ready = False
        pattern_status = "value_rotation_scalp_wait_mss"
        pattern_decision = "wait_for_value_rotation_mss_then_recheck_execution_gate"
        blocker_title = "Track value-rotation MSS before shortline scalp promotion"
        underlying_target_artifact = (
            text(retest_payload.get("next_action_target_artifact"))
            or text(retest_payload.get("blocker_target_artifact"))
            or "crypto_shortline_mss_watch"
        )
        underlying_status = retest_status
        done_when = (
            f"{route_symbol} confirms MSS/CHOCH after value rotation so the shortline execution gate can "
            "reassess the scalp"
        )
    elif retest_status == "value_rotation_scalp_retest_precondition_cvd_pending":
        pattern_family = "value_rotation_scalp"
        pattern_stage = "cvd_confirmation"
        structural_ready = False
        pattern_status = "value_rotation_scalp_wait_cvd_confirmation"
        pattern_decision = "wait_for_value_rotation_cvd_confirmation_then_recheck_execution_gate"
        blocker_title = "Track value-rotation CVD confirmation before shortline scalp promotion"
        underlying_target_artifact = (
            text(retest_payload.get("next_action_target_artifact"))
            or text(retest_payload.get("blocker_target_artifact"))
            or "crypto_shortline_cvd_confirmation_watch"
        )
        underlying_status = retest_status
        done_when = (
            f"{route_symbol} confirms local-window-valid CVD direction after value rotation MSS so the shortline "
            "execution gate can reassess the scalp"
        )
    elif retest_status == "value_rotation_scalp_wait_retest":
        pattern_family = "value_rotation_scalp"
        pattern_stage = "retest_confirmation"
        structural_ready = False
        pattern_status = "value_rotation_scalp_wait_retest"
        pattern_decision = "wait_for_value_rotation_retest_then_recheck_execution_gate"
        blocker_title = "Track value-rotation retest before shortline scalp promotion"
        underlying_target_artifact = (
            text(retest_payload.get("next_action_target_artifact"))
            or text(retest_payload.get("blocker_target_artifact"))
            or "crypto_shortline_retest_watch"
        )
        underlying_status = retest_status
        done_when = (
            f"{route_symbol} completes the required FVG/OB/Breaker retest after value rotation confirmation, "
            "then the shortline execution gate can reassess the scalp"
        )
    elif retest_status.startswith("imbalance_continuation_wait_retest"):
        pattern_family = "imbalance_continuation"
        pattern_stage = "imbalance_retest"
        structural_ready = False
        if retest_status.endswith("_final"):
            pattern_status = "imbalance_continuation_wait_retest_final"
            pattern_decision = "monitor_final_imbalance_retest_then_recheck_execution_gate"
            blocker_title = "Track final imbalance retest before shortline continuation promotion"
        elif retest_status.endswith("_approaching"):
            pattern_status = "imbalance_continuation_wait_retest_approaching"
            pattern_decision = "monitor_imbalance_retest_into_final_band_then_recheck_execution_gate"
            blocker_title = "Track approaching imbalance retest before shortline continuation promotion"
        elif retest_status.endswith("_far"):
            pattern_status = "imbalance_continuation_wait_retest_far"
            pattern_decision = "monitor_imbalance_retest_band_then_recheck_execution_gate"
            blocker_title = "Track imbalance retest band before shortline continuation promotion"
        else:
            pattern_status = "imbalance_continuation_wait_retest"
            pattern_decision = "wait_for_imbalance_retest_then_recheck_execution_gate"
            blocker_title = "Track imbalance retest before shortline continuation promotion"
        underlying_target_artifact = (
            text(retest_payload.get("next_action_target_artifact"))
            or text(retest_payload.get("blocker_target_artifact"))
            or "crypto_shortline_retest_watch"
        )
        underlying_status = retest_status
        done_when = (
            f"{route_symbol} completes the required FVG/OB/Breaker retest, then the shortline execution gate can "
            "reassess continuation quality"
        )
    elif setup_primary_gate == "fvg_ob_breaker_retest" or "fvg_ob_breaker_retest" in setup_status:
        pattern_family = "imbalance_continuation"
        pattern_stage = "imbalance_retest"
        structural_ready = False
        pattern_status = "imbalance_continuation_wait_retest"
        pattern_decision = "wait_for_imbalance_retest_then_recheck_execution_gate"
        blocker_title = "Track imbalance retest before shortline continuation promotion"
        underlying_target_artifact = (
            text(setup_transition_payload.get("next_action_target_artifact"))
            or text(setup_transition_payload.get("blocker_target_artifact"))
            or "crypto_shortline_setup_transition_watch"
        )
        underlying_status = setup_status
        done_when = (
            f"{route_symbol} completes the required FVG/OB/Breaker retest, then the shortline execution gate can "
            "reassess continuation quality"
        )
    elif cvd_status.startswith("value_rotation_scalp_cvd_precondition_profile_alignment_"):
        pattern_family = "value_rotation_scalp"
        pattern_stage = "profile_alignment"
        structural_ready = False
        underlying_target_artifact = (
            text(cvd_payload.get("next_action_target_artifact"))
            or text(cvd_payload.get("blocker_target_artifact"))
            or "crypto_shortline_profile_location_watch"
        )
        underlying_status = cvd_status
        profile_rotation_alignment_band = value_rotation_alignment_band(
            payload=cvd_payload,
            status=cvd_status,
            target_bin_distance=cvd_payload.get("profile_rotation_target_bin_distance"),
        )
        if profile_rotation_alignment_band == "final":
            pattern_status = "value_rotation_scalp_wait_profile_alignment_final"
            pattern_decision = "monitor_final_value_rotation_into_hvn_poc_then_recheck_execution_gate"
            blocker_title = "Track final value-rotation alignment before shortline scalp promotion"
        elif profile_rotation_alignment_band == "approaching":
            pattern_status = "value_rotation_scalp_wait_profile_alignment_approaching"
            pattern_decision = "monitor_value_rotation_into_final_band_then_recheck_execution_gate"
            blocker_title = "Track approaching value-rotation alignment before shortline scalp promotion"
        else:
            pattern_status = "value_rotation_scalp_wait_profile_alignment_far"
            pattern_decision = "monitor_value_rotation_toward_hvn_poc_then_recheck_execution_gate"
            blocker_title = "Track value-rotation alignment before shortline scalp promotion"
        done_when = (
            f"{route_symbol} rotates from LVN toward HVN/POC so CVD confirmation can be reassessed "
            "for the value-rotation scalp"
        )
    elif cvd_status == "value_rotation_scalp_cvd_precondition_mss_pending":
        pattern_family = "value_rotation_scalp"
        pattern_stage = "mss_confirmation"
        structural_ready = False
        pattern_status = "value_rotation_scalp_wait_mss"
        pattern_decision = "wait_for_value_rotation_mss_then_recheck_execution_gate"
        blocker_title = "Track value-rotation MSS before shortline scalp promotion"
        underlying_target_artifact = (
            text(cvd_payload.get("next_action_target_artifact"))
            or text(cvd_payload.get("blocker_target_artifact"))
            or "crypto_shortline_mss_watch"
        )
        underlying_status = cvd_status
        done_when = (
            f"{route_symbol} confirms MSS/CHOCH after value rotation so the shortline execution gate can "
            "reassess the scalp"
        )
    elif cvd_status == "value_rotation_scalp_wait_cvd_confirmation":
        pattern_family = "value_rotation_scalp"
        pattern_stage = "cvd_confirmation"
        structural_ready = False
        pattern_status = "value_rotation_scalp_wait_cvd_confirmation"
        pattern_decision = "wait_for_value_rotation_cvd_confirmation_then_recheck_execution_gate"
        blocker_title = "Track value-rotation CVD confirmation before shortline scalp promotion"
        underlying_target_artifact = (
            text(cvd_payload.get("next_action_target_artifact"))
            or text(cvd_payload.get("blocker_target_artifact"))
            or "crypto_shortline_cvd_confirmation_watch"
        )
        underlying_status = cvd_status
        done_when = (
            f"{route_symbol} confirms local-window-valid CVD direction after value rotation MSS, then "
            "the shortline execution gate can reassess the scalp"
        )
    elif cvd_status and cvd_status != "cvd_confirmation_precondition_mss_pending":
        pattern_family = "sweep_reversal"
        pattern_stage = "cvd_confirmation"
        structural_ready = False
        pattern_status = "sweep_reversal_wait_cvd_confirmation"
        pattern_decision = "wait_for_sweep_reversal_cvd_confirmation_then_recheck_execution_gate"
        blocker_title = "Track sweep-reversal CVD confirmation before shortline scalp promotion"
        underlying_target_artifact = (
            text(cvd_payload.get("next_action_target_artifact"))
            or text(cvd_payload.get("blocker_target_artifact"))
            or "crypto_shortline_cvd_confirmation_watch"
        )
        underlying_status = cvd_status
        done_when = (
            f"{route_symbol} confirms CVD in the same direction as the active sweep-reversal candidate, then "
            "the shortline execution gate can reassess the setup"
        )
    elif signal_quality_status or sizing_status:
        pattern_family = "pattern_candidate_unqualified"
        pattern_stage = "quality_or_size"
        structural_ready = False
        pattern_status = "pattern_candidate_unqualified_quality_or_size"
        if signal_quality_status:
            pattern_decision = "improve_shortline_signal_quality_then_recheck_execution_gate"
            underlying_target_artifact = (
                text(signal_quality_payload.get("next_action_target_artifact"))
                or text(signal_quality_payload.get("blocker_target_artifact"))
                or "crypto_shortline_signal_quality_watch"
            )
            underlying_status = signal_quality_status
            blocker_title = "Improve shortline pattern quality before promotion"
        else:
            pattern_decision = "raise_effective_shortline_size_then_recheck_execution_gate"
            underlying_target_artifact = (
                text(sizing_payload.get("next_action_target_artifact"))
                or text(sizing_payload.get("blocker_target_artifact"))
                or "crypto_shortline_sizing_watch"
            )
            underlying_status = sizing_status
            blocker_title = "Raise shortline pattern size before promotion"
        done_when = (
            f"{route_symbol} clears quality and size constraints so the routed shortline pattern can be evaluated "
            "on post-cost expectancy"
        )

    pattern_confidence_score = 35
    if liquidity_event_status:
        pattern_confidence_score += 15
    if "key_level_ready" in profile_status:
        pattern_confidence_score += 20
    if pattern_family == "sweep_reversal":
        pattern_confidence_score += 10
    if signal_quality_status:
        pattern_confidence_score -= 15
    if sizing_status:
        pattern_confidence_score -= 15
    pattern_confidence_score = max(0, min(100, pattern_confidence_score))

    pattern_brief = ":".join(
        [pattern_status, route_symbol or "-", pattern_decision, remote_market or "-", pattern_family]
    )
    blocker_detail = join_unique(
        [
            (
                f"profile_location_watch={text(profile_payload.get('watch_brief'))}:{profile_decision}"
                if text(profile_payload.get("watch_brief")) or profile_decision
                else ""
            ),
            (
                f"profile_rotation_target={text(profile_payload.get('profile_rotation_target_tag'))}:"
                f"bins={text(profile_payload.get('profile_rotation_target_bin_distance'))}:"
                f"bps={text(profile_payload.get('profile_rotation_target_distance_bps'))}"
                if text(profile_payload.get("profile_rotation_target_tag"))
                or profile_payload.get("profile_rotation_target_bin_distance") is not None
                else ""
            ),
            (
                f"mss_watch={text(mss_payload.get('watch_brief'))}:{mss_decision}"
                if text(mss_payload.get("watch_brief")) or mss_decision
                else ""
            ),
            (
                f"cvd_confirmation_watch={text(cvd_payload.get('watch_brief'))}:{cvd_decision}"
                if text(cvd_payload.get("watch_brief")) or cvd_decision
                else ""
            ),
            (
                f"retest_watch={text(retest_payload.get('watch_brief'))}:{retest_decision}"
                if text(retest_payload.get("watch_brief")) or retest_decision
                else ""
            ),
            (
                f"setup_transition_watch={text(setup_transition_payload.get('transition_brief'))}:{setup_decision}"
                if text(setup_transition_payload.get("transition_brief")) or setup_decision
                else ""
            ),
            (
                f"liquidity_event_trigger={text(liquidity_event_payload.get('trigger_brief'))}:{text(liquidity_event_payload.get('trigger_decision'))}"
                if text(liquidity_event_payload.get("trigger_brief"))
                or text(liquidity_event_payload.get("trigger_decision"))
                else ""
            ),
            (
                f"signal_quality_watch={text(signal_quality_payload.get('watch_brief'))}:{text(signal_quality_payload.get('watch_decision'))}"
                if text(signal_quality_payload.get("watch_brief"))
                or text(signal_quality_payload.get("watch_decision"))
                else ""
            ),
            (
                f"sizing_watch={text(sizing_payload.get('watch_brief'))}:{text(sizing_payload.get('watch_decision'))}"
                if text(sizing_payload.get("watch_brief"))
                or text(sizing_payload.get("watch_decision"))
                else ""
            ),
        ]
    )
    return {
        "action": "build_crypto_shortline_pattern_router",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": route_symbol,
        "route_focus_symbol": text(route_focus_symbol).upper(),
        "route_action": route_action,
        "remote_market": remote_market,
        "pattern_family": pattern_family,
        "pattern_stage": pattern_stage,
        "pattern_status": pattern_status,
        "pattern_brief": pattern_brief,
        "pattern_decision": pattern_decision,
        "pattern_confidence_score": pattern_confidence_score,
        "profile_rotation_alignment_band": profile_rotation_alignment_band,
        "profile_rotation_target_bin_distance": profile_rotation_target_bin_distance,
        "profile_rotation_target_distance_bps": profile_rotation_target_distance_bps,
        "structural_ready": structural_ready,
        "blocker_title": blocker_title,
        "blocker_target_artifact": "crypto_shortline_pattern_router",
        "underlying_target_artifact": underlying_target_artifact,
        "underlying_status": underlying_status,
        "next_action": pattern_decision,
        "next_action_target_artifact": underlying_target_artifact,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-owned crypto shortline pattern router artifact."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    parser.add_argument("--symbol", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)

    intent_path = find_latest(review_dir, "*_remote_intent_queue.json", reference_now)
    operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json", reference_now)
    if operator_path is None:
        raise SystemExit("missing_required_artifacts:crypto_route_operator_brief")

    intent_payload = (
        load_json_mapping(intent_path) if intent_path is not None and intent_path.exists() else {}
    )
    operator_payload = load_json_mapping(operator_path)
    route_focus_symbol = (
        text(intent_payload.get("preferred_route_symbol"))
        or text(operator_payload.get("review_priority_head_symbol"))
        or text(operator_payload.get("next_focus_symbol"))
        or "-"
    ).upper()
    selected_symbol = text(args.symbol).upper() or route_focus_symbol

    profile_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_profile_location_watch.json", selected_symbol, reference_now
    )
    mss_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_mss_watch.json", selected_symbol, reference_now
    )
    cvd_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_cvd_confirmation_watch.json", selected_symbol, reference_now
    )
    retest_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_retest_watch.json", selected_symbol, reference_now
    )
    setup_transition_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_setup_transition_watch.json", selected_symbol, reference_now
    )
    liquidity_event_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_liquidity_event_trigger.json", selected_symbol, reference_now
    )
    signal_quality_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_signal_quality_watch.json", selected_symbol, reference_now
    )
    sizing_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_sizing_watch.json", selected_symbol, reference_now
    )

    payload = build_payload(
        reference_now=reference_now,
        route_symbol_override=selected_symbol,
        route_focus_symbol=route_focus_symbol,
        intent_payload=intent_payload,
        operator_payload=operator_payload,
        profile_payload=load_json_mapping(profile_path)
        if profile_path is not None and profile_path.exists()
        else {},
        mss_payload=load_json_mapping(mss_path) if mss_path is not None and mss_path.exists() else {},
        cvd_payload=load_json_mapping(cvd_path) if cvd_path is not None and cvd_path.exists() else {},
        retest_payload=load_json_mapping(retest_path)
        if retest_path is not None and retest_path.exists()
        else {},
        setup_transition_payload=load_json_mapping(setup_transition_path)
        if setup_transition_path is not None and setup_transition_path.exists()
        else {},
        liquidity_event_payload=load_json_mapping(liquidity_event_path)
        if liquidity_event_path is not None and liquidity_event_path.exists()
        else {},
        signal_quality_payload=load_json_mapping(signal_quality_path)
        if signal_quality_path is not None and signal_quality_path.exists()
        else {},
        sizing_payload=load_json_mapping(sizing_path)
        if sizing_path is not None and sizing_path.exists()
        else {},
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_pattern_router.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_pattern_router.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_pattern_router_checksum.json"

    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_paths=[artifact, markdown, checksum],
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )
    payload.update(
        {
            "artifact": str(artifact),
            "markdown": str(markdown),
            "checksum": str(checksum),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
            "artifacts": {
                "remote_intent_queue": str(intent_path) if intent_path else "",
                "crypto_route_operator_brief": str(operator_path),
                "crypto_shortline_profile_location_watch": str(profile_path)
                if profile_path
                else "",
                "crypto_shortline_mss_watch": str(mss_path) if mss_path else "",
                "crypto_shortline_cvd_confirmation_watch": str(cvd_path) if cvd_path else "",
                "crypto_shortline_retest_watch": str(retest_path) if retest_path else "",
                "crypto_shortline_setup_transition_watch": str(setup_transition_path)
                if setup_transition_path
                else "",
                "crypto_shortline_liquidity_event_trigger": str(liquidity_event_path)
                if liquidity_event_path
                else "",
                "crypto_shortline_signal_quality_watch": str(signal_quality_path)
                if signal_quality_path
                else "",
                "crypto_shortline_sizing_watch": str(sizing_path) if sizing_path else "",
            },
        }
    )
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
