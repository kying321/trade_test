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


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def dedupe_text(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = text(raw)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def join_unique(parts: list[Any], *, sep: str = " | ") -> str:
    return sep.join(dedupe_text(parts))


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
    return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def artifact_sort_key(
    path: Path, reference_now: dt.datetime | None = None
) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    effective_now = reference_now or now_utc()
    future_cutoff = effective_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    return (0 if is_future else 1, artifact_stamp(path), path.stat().st_mtime, path.name)


def find_latest(
    review_dir: Path, pattern: str, reference_now: dt.datetime | None = None
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
        "*_crypto_shortline_retest_watch.json",
        "*_crypto_shortline_retest_watch.md",
        "*_crypto_shortline_retest_watch_checksum.json",
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


def route_symbol(intent_payload: dict[str, Any], operator_payload: dict[str, Any]) -> str:
    return (
        text(intent_payload.get("preferred_route_symbol"))
        or text(operator_payload.get("review_priority_head_symbol"))
        or text(operator_payload.get("next_focus_symbol"))
    ).upper()


def route_action(intent_payload: dict[str, Any], operator_payload: dict[str, Any]) -> str:
    return text(intent_payload.get("preferred_route_action")) or text(
        operator_payload.get("next_focus_action")
    )


def resolve_symbol(
    *,
    explicit_symbol: str,
    intent_payload: dict[str, Any],
    operator_payload: dict[str, Any],
) -> str:
    return text(explicit_symbol).upper() or route_symbol(intent_payload, operator_payload)


def find_gate_row(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(payload.get("symbols")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Shortline Retest Watch",
            "",
            f"- brief: `{text(payload.get('watch_brief'))}`",
            f"- decision: `{text(payload.get('watch_decision'))}`",
            f"- pattern_family: `{text(payload.get('pattern_family')) or '-'}`",
            f"- retest_stage: `{text(payload.get('retest_stage')) or '-'}`",
            f"- execution_state: `{text(payload.get('execution_state'))}`",
            f"- retest_signal_side: `{text(payload.get('retest_signal_side')) or '-'}`",
            f"- profile_watch_status: `{text(payload.get('profile_watch_status')) or '-'}`",
            f"- mss_watch_status: `{text(payload.get('mss_watch_status')) or '-'}`",
            f"- cvd_watch_status: `{text(payload.get('cvd_confirmation_watch_status')) or '-'}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def profile_alignment_band(profile_watch_status: str, profile_watch_payload: dict[str, Any]) -> str:
    source_band = text(profile_watch_payload.get("profile_rotation_alignment_band"))
    if source_band:
        return source_band
    status = text(profile_watch_status)
    if status.endswith("_final_band"):
        return "final"
    if status.endswith("_approaching"):
        return "approaching"
    target_bin_distance = profile_watch_payload.get("profile_rotation_target_bin_distance")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-owned crypto shortline retest watch artifact."
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
    gate_path = find_latest(review_dir, "*_crypto_shortline_execution_gate.json", reference_now)
    if operator_path is None or gate_path is None:
        missing = [
            name
            for name, path in (
                ("crypto_route_operator_brief", operator_path),
                ("crypto_shortline_execution_gate", gate_path),
            )
            if path is None
        ]
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    intent_payload = (
        load_json_mapping(intent_path) if intent_path is not None and intent_path.exists() else {}
    )
    operator_payload = load_json_mapping(operator_path)
    gate_payload = load_json_mapping(gate_path)
    route_focus_symbol = route_symbol(intent_payload, operator_payload)
    symbol = resolve_symbol(
        explicit_symbol=args.symbol,
        intent_payload=intent_payload,
        operator_payload=operator_payload,
    )
    profile_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_profile_location_watch.json", symbol, reference_now
    )
    mss_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_mss_watch.json", symbol, reference_now
    )
    cvd_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_cvd_confirmation_watch.json", symbol, reference_now
    )
    profile_watch_payload = (
        load_json_mapping(profile_watch_path)
        if profile_watch_path is not None and profile_watch_path.exists()
        else {}
    )
    mss_watch_payload = (
        load_json_mapping(mss_watch_path)
        if mss_watch_path is not None and mss_watch_path.exists()
        else {}
    )
    cvd_watch_payload = (
        load_json_mapping(cvd_watch_path)
        if cvd_watch_path is not None and cvd_watch_path.exists()
        else {}
    )

    action = route_action(intent_payload, operator_payload)
    remote_market = text(intent_payload.get("remote_market")) or "portfolio_margin_um"
    gate_row = find_gate_row(gate_payload, symbol)
    structure = as_dict(gate_row.get("structure_signals"))
    missing_gates = dedupe_text(as_list(gate_row.get("missing_gates")))
    execution_state = text(gate_row.get("execution_state")) or "Bias_Only"
    route_state = text(gate_row.get("route_state"))

    fvg_long = bool(structure.get("fvg_long", False))
    fvg_short = bool(structure.get("fvg_short", False))
    retest_present = fvg_long or fvg_short
    retest_signal_side = "long" if fvg_long else "short" if fvg_short else ""
    retest_missing = "fvg_ob_breaker_retest" in missing_gates

    profile_watch_status = text(profile_watch_payload.get("watch_status"))
    profile_watch_brief = text(profile_watch_payload.get("watch_brief"))
    mss_watch_status = text(mss_watch_payload.get("watch_status"))
    mss_watch_brief = text(mss_watch_payload.get("watch_brief"))
    cvd_watch_status = text(cvd_watch_payload.get("watch_status"))
    cvd_watch_brief = text(cvd_watch_payload.get("watch_brief"))

    blocker_target_artifact = "crypto_shortline_retest_watch"
    next_action_target_artifact = "crypto_shortline_retest_watch"
    pattern_family = "pattern_watch_only"
    retest_stage = "unqualified"

    if execution_state == "Setup_Ready" or not retest_missing:
        watch_status = "fvg_ob_breaker_retest_cleared"
        watch_decision = "review_next_shortline_stage"
        blocker_title = "FVG/OB retest cleared for next-stage shortline review"
        pattern_family = "pattern_ready"
        retest_stage = "cleared"
        done_when = f"{symbol} loses retest readiness or leaves Setup_Ready"
    elif profile_watch_status.startswith("profile_location_lvn_key_level_ready_rotation_"):
        pattern_family = "value_rotation_scalp"
        retest_stage = "profile_alignment_precondition"
        next_action_target_artifact = (
            text(profile_watch_payload.get("next_action_target_artifact"))
            or text(profile_watch_payload.get("blocker_target_artifact"))
            or "crypto_shortline_profile_location_watch"
        )
        alignment_band = profile_alignment_band(profile_watch_status, profile_watch_payload)
        if alignment_band == "final":
            watch_status = "value_rotation_scalp_retest_precondition_profile_alignment_final"
            watch_decision = "monitor_final_value_rotation_into_hvn_poc_then_recheck_retest"
            blocker_title = "Track final value-rotation alignment before retest confirmation for shortline scalp"
        elif alignment_band == "approaching":
            watch_status = "value_rotation_scalp_retest_precondition_profile_alignment_approaching"
            watch_decision = "monitor_value_rotation_into_final_band_then_recheck_retest"
            blocker_title = "Track approaching value-rotation alignment before retest confirmation for shortline scalp"
        else:
            watch_status = "value_rotation_scalp_retest_precondition_profile_alignment_far"
            watch_decision = "monitor_value_rotation_toward_hvn_poc_then_recheck_retest"
            blocker_title = "Track value-rotation alignment before retest confirmation for shortline scalp"
        done_when = (
            f"{symbol} rotates from LVN toward HVN/POC so the FVG/OB/Breaker retest can be reassessed "
            "for the value-rotation scalp"
        )
    elif mss_watch_status.startswith("value_rotation_scalp_mss_precondition_profile_alignment_"):
        pattern_family = "value_rotation_scalp"
        retest_stage = "profile_alignment_precondition"
        next_action_target_artifact = (
            text(mss_watch_payload.get("next_action_target_artifact"))
            or text(mss_watch_payload.get("blocker_target_artifact"))
            or "crypto_shortline_profile_location_watch"
        )
        if mss_watch_status.endswith("_final"):
            watch_status = "value_rotation_scalp_retest_precondition_profile_alignment_final"
            watch_decision = "monitor_final_value_rotation_into_hvn_poc_then_recheck_retest"
            blocker_title = "Track final value-rotation alignment before retest confirmation for shortline scalp"
        elif mss_watch_status.endswith("_approaching"):
            watch_status = "value_rotation_scalp_retest_precondition_profile_alignment_approaching"
            watch_decision = "monitor_value_rotation_into_final_band_then_recheck_retest"
            blocker_title = "Track approaching value-rotation alignment before retest confirmation for shortline scalp"
        else:
            watch_status = "value_rotation_scalp_retest_precondition_profile_alignment_far"
            watch_decision = "monitor_value_rotation_toward_hvn_poc_then_recheck_retest"
            blocker_title = "Track value-rotation alignment before retest confirmation for shortline scalp"
        done_when = (
            f"{symbol} rotates from LVN toward HVN/POC so the retest stage can be reassessed "
            "for the value-rotation scalp"
        )
    elif cvd_watch_status.startswith("value_rotation_scalp_cvd_precondition_profile_alignment_"):
        pattern_family = "value_rotation_scalp"
        retest_stage = "profile_alignment_precondition"
        next_action_target_artifact = (
            text(cvd_watch_payload.get("next_action_target_artifact"))
            or text(cvd_watch_payload.get("blocker_target_artifact"))
            or "crypto_shortline_profile_location_watch"
        )
        if cvd_watch_status.endswith("_final"):
            watch_status = "value_rotation_scalp_retest_precondition_profile_alignment_final"
            watch_decision = "monitor_final_value_rotation_into_hvn_poc_then_recheck_retest"
            blocker_title = "Track final value-rotation alignment before retest confirmation for shortline scalp"
        elif cvd_watch_status.endswith("_approaching"):
            watch_status = "value_rotation_scalp_retest_precondition_profile_alignment_approaching"
            watch_decision = "monitor_value_rotation_into_final_band_then_recheck_retest"
            blocker_title = "Track approaching value-rotation alignment before retest confirmation for shortline scalp"
        else:
            watch_status = "value_rotation_scalp_retest_precondition_profile_alignment_far"
            watch_decision = "monitor_value_rotation_toward_hvn_poc_then_recheck_retest"
            blocker_title = "Track value-rotation alignment before retest confirmation for shortline scalp"
        done_when = (
            f"{symbol} rotates from LVN toward HVN/POC so the retest stage can be reassessed "
            "for the value-rotation scalp"
        )
    elif cvd_watch_status == "value_rotation_scalp_cvd_precondition_mss_pending":
        watch_status = "value_rotation_scalp_retest_precondition_mss_pending"
        watch_decision = "wait_for_value_rotation_mss_then_recheck_retest"
        blocker_title = "Track value-rotation MSS before retest confirmation for shortline scalp"
        pattern_family = "value_rotation_scalp"
        retest_stage = "mss_precondition"
        next_action_target_artifact = (
            text(cvd_watch_payload.get("next_action_target_artifact"))
            or text(cvd_watch_payload.get("blocker_target_artifact"))
            or "crypto_shortline_mss_watch"
        )
        done_when = (
            f"{symbol} confirms MSS/CHOCH after value rotation so the retest stage can be reassessed "
            "for the scalp"
        )
    elif cvd_watch_status == "value_rotation_scalp_wait_cvd_confirmation":
        watch_status = "value_rotation_scalp_retest_precondition_cvd_pending"
        watch_decision = "wait_for_value_rotation_cvd_confirmation_then_recheck_retest"
        blocker_title = "Track value-rotation CVD confirmation before retest for shortline scalp"
        pattern_family = "value_rotation_scalp"
        retest_stage = "cvd_precondition"
        next_action_target_artifact = (
            text(cvd_watch_payload.get("next_action_target_artifact"))
            or text(cvd_watch_payload.get("blocker_target_artifact"))
            or "crypto_shortline_cvd_confirmation_watch"
        )
        done_when = (
            f"{symbol} confirms local-window-valid CVD direction after value rotation MSS so the retest stage "
            "can be reassessed"
        )
    elif text(cvd_watch_payload.get("pattern_family")) == "value_rotation_scalp":
        watch_status = "value_rotation_scalp_wait_retest"
        watch_decision = "wait_for_value_rotation_retest_then_recheck_execution_gate"
        blocker_title = "Track value-rotation retest before shortline scalp promotion"
        pattern_family = "value_rotation_scalp"
        retest_stage = "retest_confirmation"
        done_when = (
            f"{symbol} completes the required FVG/OB/Breaker retest after value rotation confirmation, "
            "then the shortline execution gate can reassess the scalp"
        )
    else:
        pattern_family = "imbalance_continuation"
        retest_stage = "retest_confirmation"
        continuation_alignment_band = (
            profile_alignment_band(profile_watch_status, profile_watch_payload)
            if "_rotation_" in profile_watch_status
            else ""
        )
        if continuation_alignment_band:
            next_action_target_artifact = (
                text(profile_watch_payload.get("next_action_target_artifact"))
                or text(profile_watch_payload.get("blocker_target_artifact"))
                or "crypto_shortline_profile_location_watch"
            )
        if continuation_alignment_band == "final":
            watch_status = "imbalance_continuation_wait_retest_final"
            watch_decision = "monitor_final_imbalance_retest_then_recheck_execution_gate"
            blocker_title = "Track final imbalance retest before shortline continuation promotion"
            done_when = (
                f"{symbol} completes the final FVG/OB/Breaker retest band and confirms the continuation retest, "
                "then the shortline execution gate can reassess continuation quality"
            )
        elif continuation_alignment_band == "approaching":
            watch_status = "imbalance_continuation_wait_retest_approaching"
            watch_decision = "monitor_imbalance_retest_into_final_band_then_recheck_execution_gate"
            blocker_title = "Track approaching imbalance retest before shortline continuation promotion"
            done_when = (
                f"{symbol} rotates into the final retest band and completes the required FVG/OB/Breaker retest, "
                "then the shortline execution gate can reassess continuation quality"
            )
        elif continuation_alignment_band == "far":
            watch_status = "imbalance_continuation_wait_retest_far"
            watch_decision = "monitor_imbalance_retest_band_then_recheck_execution_gate"
            blocker_title = "Track imbalance retest band before shortline continuation promotion"
            done_when = (
                f"{symbol} rotates into the imbalance retest band and completes the required FVG/OB/Breaker retest, "
                "then the shortline execution gate can reassess continuation quality"
            )
        else:
            watch_status = "imbalance_continuation_wait_retest"
            watch_decision = "wait_for_imbalance_retest_then_recheck_execution_gate"
            blocker_title = "Track imbalance retest before shortline continuation promotion"
            done_when = (
                f"{symbol} completes the required FVG/OB/Breaker retest, then the shortline execution gate can "
                "reassess continuation quality"
            )

    watch_brief = ":".join([watch_status, symbol or "-", watch_decision, remote_market or "-"])
    blocker_detail = join_unique(
        [
            f"execution_state={execution_state}",
            f"route_state={route_state or action or '-'}",
            "fvg_ob_breaker_retest" if retest_missing else "",
            f"fvg_long={str(fvg_long).lower()}",
            f"fvg_short={str(fvg_short).lower()}",
            f"retest_signal_side={retest_signal_side}" if retest_signal_side else "",
            profile_watch_brief,
            mss_watch_brief,
            cvd_watch_brief,
            text(gate_row.get("blocker_detail")),
        ]
    )

    payload = {
        "status": "ok",
        "as_of": fmt_utc(reference_now),
        "route_symbol": symbol,
        "route_focus_symbol": route_focus_symbol,
        "route_action": action,
        "remote_market": remote_market,
        "watch_status": watch_status,
        "watch_brief": watch_brief,
        "watch_decision": watch_decision,
        "blocker_title": blocker_title,
        "blocker_target_artifact": blocker_target_artifact,
        "blocker_detail": blocker_detail,
        "next_action": watch_decision,
        "next_action_target_artifact": next_action_target_artifact,
        "done_when": done_when,
        "pattern_family": pattern_family,
        "retest_stage": retest_stage,
        "execution_state": execution_state,
        "missing_gates": missing_gates,
        "retest_missing": retest_missing,
        "retest_present": retest_present,
        "retest_signal_side": retest_signal_side,
        "fvg_long": fvg_long,
        "fvg_short": fvg_short,
        "profile_watch_status": profile_watch_status,
        "profile_watch_brief": profile_watch_brief,
        "profile_alignment_band": (
            profile_alignment_band(profile_watch_status, profile_watch_payload)
            if "_rotation_" in profile_watch_status
            else ""
        ),
        "profile_rotation_next_milestone": str(
            profile_watch_payload.get("profile_rotation_next_milestone") or ""
        ),
        "profile_rotation_target_bin_distance": profile_watch_payload.get(
            "profile_rotation_target_bin_distance"
        ),
        "profile_rotation_target_distance_bps": profile_watch_payload.get(
            "profile_rotation_target_distance_bps"
        ),
        "mss_watch_status": mss_watch_status,
        "mss_watch_brief": mss_watch_brief,
        "cvd_confirmation_watch_status": cvd_watch_status,
        "cvd_confirmation_watch_brief": cvd_watch_brief,
        "source_artifacts": {
            "remote_intent_queue": str(intent_path) if intent_path else "",
            "crypto_route_operator_brief": str(operator_path),
            "crypto_shortline_execution_gate": str(gate_path),
            "crypto_shortline_profile_location_watch": str(profile_watch_path)
            if profile_watch_path
            else "",
            "crypto_shortline_mss_watch": str(mss_watch_path) if mss_watch_path else "",
            "crypto_shortline_cvd_confirmation_watch": str(cvd_watch_path)
            if cvd_watch_path
            else "",
        },
    }

    stamp = reference_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_crypto_shortline_retest_watch.json"
    md_path = review_dir / f"{stamp}_crypto_shortline_retest_watch.md"
    checksum_path = review_dir / f"{stamp}_crypto_shortline_retest_watch_checksum.json"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload) + "\n", encoding="utf-8")
    checksum_payload = {
        "artifact": str(json_path),
        "sha256": sha256_file(json_path),
        "generated_at": fmt_utc(reference_now),
    }
    checksum_path.write_text(
        json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_paths=[json_path, md_path, checksum_path],
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )
    payload.update(
        {
            "artifact": str(json_path),
            "markdown": str(md_path),
            "checksum": str(checksum_path),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
