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

from build_crypto_shortline_execution_gate import (
    find_latest_bars_file,
    read_bars_file,
    structure_signals,
)


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
SWEEP_PROXIMITY_ARMED_BPS = 75.0
SWEEP_PROXIMITY_NEAR_BPS = 200.0


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


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = list(review_dir.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda item: (artifact_stamp(item), item.stat().st_mtime, item.name))


def find_previous(review_dir: Path, pattern: str, *, exclude: Path | None = None) -> Path | None:
    files = [path for path in review_dir.glob(pattern) if exclude is None or path != exclude]
    if not files:
        return None
    ranked = sorted(
        files,
        key=lambda item: (artifact_stamp(item), item.stat().st_mtime, item.name),
        reverse=True,
    )
    return ranked[0] if ranked else None


def load_live_bars_snapshot(
    review_dir: Path,
    *,
    route_symbol: str,
) -> tuple[Path | None, dict[str, Any]]:
    path = find_latest(review_dir, "*_crypto_shortline_live_bars_snapshot.json")
    if path is None or not path.exists():
        return None, {}
    payload = load_json_mapping(path)
    if text(payload.get("route_symbol")).upper() != route_symbol.upper():
        return None, {}
    return path, payload


def load_live_orderflow_snapshot(
    review_dir: Path,
    *,
    route_symbol: str,
) -> tuple[Path | None, dict[str, Any]]:
    path = find_latest(review_dir, "*_crypto_shortline_live_orderflow_snapshot.json")
    if path is None or not path.exists():
        return None, {}
    payload = load_json_mapping(path)
    if text(payload.get("route_symbol")).upper() != route_symbol.upper():
        return None, {}
    return path, payload


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
        "*_crypto_shortline_liquidity_event_trigger.json",
        "*_crypto_shortline_liquidity_event_trigger.md",
        "*_crypto_shortline_liquidity_event_trigger_checksum.json",
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


def find_gate_row(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(payload.get("symbols")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def sweep_flags(row: dict[str, Any]) -> tuple[bool, bool]:
    structure = as_dict(row.get("structure_signals"))
    return bool(structure.get("sweep_long", False)), bool(structure.get("sweep_short", False))


def live_sweep_flags(
    *,
    output_root: Path,
    route_symbol: str,
    explicit_bars_file: str,
) -> dict[str, Any]:
    bars_path = (
        Path(explicit_bars_file).expanduser().resolve()
        if text(explicit_bars_file)
        else find_latest_bars_file(output_root, [route_symbol])
    )
    if bars_path is None or not bars_path.exists():
        return {
            "available": False,
            "source": "",
            "artifact": "",
            "sweep_long": False,
            "sweep_short": False,
        }

    try:
        frame = read_bars_file(bars_path)
    except Exception:
        return {
            "available": False,
            "source": "",
            "artifact": "",
            "sweep_long": False,
            "sweep_short": False,
        }

    if "symbol" not in frame.columns:
        return {
            "available": False,
            "source": "",
            "artifact": str(bars_path),
            "sweep_long": False,
            "sweep_short": False,
        }
    symbol_frame = frame[frame["symbol"].astype(str).str.upper() == route_symbol.upper()].copy()
    if symbol_frame.empty:
        return {
            "available": False,
            "source": "",
            "artifact": str(bars_path),
            "sweep_long": False,
            "sweep_short": False,
        }

    try:
        structure = structure_signals(symbol_frame.sort_values("ts").reset_index(drop=True))
    except Exception:
        return {
            "available": False,
            "source": "",
            "artifact": str(bars_path),
            "sweep_long": False,
            "sweep_short": False,
        }

    return {
        "available": True,
        "source": "bars_live_snapshot",
        "artifact": str(bars_path),
        "sweep_long": bool(structure.get("sweep_long", False)),
        "sweep_short": bool(structure.get("sweep_short", False)),
    }


def orderflow_pressure_flags(snapshot_payload: dict[str, Any]) -> dict[str, Any]:
    if not snapshot_payload:
        return {
            "available": False,
            "pressure_present": False,
            "pressure_eligible": False,
            "pressure_side": "",
            "strength": 0.0,
            "quality_ok": False,
            "time_sync_ok": False,
        }
    queue_imbalance = float(snapshot_payload.get("queue_imbalance") or 0.0)
    ofi_norm = float(snapshot_payload.get("ofi_norm") or 0.0)
    micro_alignment = float(snapshot_payload.get("micro_alignment") or 0.0)
    cvd_delta_ratio = float(snapshot_payload.get("cvd_delta_ratio") or 0.0)
    attack_side = text(snapshot_payload.get("cvd_attack_side")).lower()
    attack_presence = text(snapshot_payload.get("cvd_attack_presence")).lower()
    snapshot_status = text(snapshot_payload.get("snapshot_status"))
    quality_ok = bool(snapshot_payload.get("micro_quality_ok", False))
    time_sync_ok = bool(snapshot_payload.get("time_sync_ok", False))
    pressure_eligible = quality_ok and time_sync_ok and snapshot_status == "live_orderflow_snapshot_ready"
    strength = max(abs(queue_imbalance), abs(ofi_norm), abs(micro_alignment), abs(cvd_delta_ratio))
    if attack_side in {"buyers", "sellers"}:
        pressure_side = "long" if attack_side == "buyers" else "short"
    elif micro_alignment > 0.0 or cvd_delta_ratio > 0.0:
        pressure_side = "long"
    elif micro_alignment < 0.0 or cvd_delta_ratio < 0.0:
        pressure_side = "short"
    else:
        pressure_side = ""
    pressure_present = bool(
        strength >= 0.10
        and pressure_side
        and attack_presence not in {"", "missing_micro_capture", "no_clear_attack"}
        and pressure_eligible
    )
    return {
        "available": True,
        "pressure_present": pressure_present,
        "pressure_eligible": pressure_eligible,
        "pressure_side": pressure_side,
        "strength": strength,
        "quality_ok": quality_ok,
        "time_sync_ok": time_sync_ok,
        "snapshot_status": snapshot_status,
        "snapshot_brief": text(snapshot_payload.get("snapshot_brief")),
        "snapshot_decision": text(snapshot_payload.get("snapshot_decision")),
        "queue_imbalance": queue_imbalance,
        "ofi_norm": ofi_norm,
        "micro_alignment": micro_alignment,
        "cvd_delta_ratio": cvd_delta_ratio,
        "cvd_veto_hint": text(snapshot_payload.get("cvd_veto_hint")),
        "cvd_locality_status": text(snapshot_payload.get("cvd_locality_status")),
        "cvd_attack_presence": attack_presence,
        "cvd_attack_side": attack_side,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Shortline Liquidity Event Trigger",
            "",
            f"- brief: `{text(payload.get('trigger_brief'))}`",
            f"- decision: `{text(payload.get('trigger_decision'))}`",
            f"- current_sweep_long: `{payload.get('current_sweep_long')}`",
            f"- current_sweep_short: `{payload.get('current_sweep_short')}`",
            f"- orderflow_pressure_present: `{payload.get('orderflow_pressure_present')}`",
            f"- orderflow_pressure_side: `{payload.get('orderflow_pressure_side')}`",
            f"- orderflow_pressure_strength: `{payload.get('orderflow_pressure_strength')}`",
            f"- sweep_proximity_state: `{text(payload.get('sweep_proximity_state'))}`",
            f"- active_sweep_distance_bps: `{payload.get('active_sweep_distance_bps')}`",
            f"- previous_sweep_long: `{payload.get('previous_sweep_long')}`",
            f"- previous_sweep_short: `{payload.get('previous_sweep_short')}`",
            f"- sweep_state_changed: `{payload.get('sweep_state_changed')}`",
            f"- newly_observed: `{payload.get('sweep_event_newly_observed')}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-owned liquidity-event trigger for crypto shortline promotion."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--bars-file", default="")
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)

    intent_path = find_latest(review_dir, "*_remote_intent_queue.json")
    operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json")
    current_gate_path = find_latest(review_dir, "*_crypto_shortline_execution_gate.json")
    material_change_path = find_latest(review_dir, "*_crypto_shortline_material_change_trigger.json")

    if operator_path is None or current_gate_path is None:
        missing = [
            name
            for name, path in (
                ("crypto_route_operator_brief", operator_path),
                ("crypto_shortline_execution_gate", current_gate_path),
            )
            if path is None
        ]
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    previous_gate_path = find_previous(
        review_dir,
        "*_crypto_shortline_execution_gate.json",
        exclude=current_gate_path,
    )
    previous_trigger_path = find_latest(review_dir, "*_crypto_shortline_liquidity_event_trigger.json")

    intent_payload = (
        load_json_mapping(intent_path) if intent_path is not None and intent_path.exists() else {}
    )
    operator_payload = load_json_mapping(operator_path)
    current_gate_payload = load_json_mapping(current_gate_path)
    previous_gate_payload = (
        load_json_mapping(previous_gate_path)
        if previous_gate_path is not None and previous_gate_path.exists()
        else {}
    )
    previous_trigger_payload = (
        load_json_mapping(previous_trigger_path)
        if previous_trigger_path is not None and previous_trigger_path.exists()
        else {}
    )
    material_change_payload = (
        load_json_mapping(material_change_path)
        if material_change_path is not None and material_change_path.exists()
        else {}
    )

    route_symbol = (
        text(intent_payload.get("preferred_route_symbol"))
        or text(operator_payload.get("review_priority_head_symbol"))
        or text(operator_payload.get("next_focus_symbol"))
    ).upper()
    route_action = text(intent_payload.get("preferred_route_action")) or text(
        operator_payload.get("next_focus_action")
    )
    remote_market = text(intent_payload.get("remote_market")) or "portfolio_margin_um"

    current_row = find_gate_row(current_gate_payload, route_symbol)
    previous_row = find_gate_row(previous_gate_payload, route_symbol)
    live_bars_snapshot_path, live_bars_snapshot_payload = load_live_bars_snapshot(
        review_dir,
        route_symbol=route_symbol,
    )
    live_orderflow_snapshot_path, live_orderflow_snapshot_payload = load_live_orderflow_snapshot(
        review_dir,
        route_symbol=route_symbol,
    )
    live_sweep_payload = live_sweep_flags(
        output_root=output_root,
        route_symbol=route_symbol,
        explicit_bars_file=args.bars_file,
    )
    orderflow_pressure = orderflow_pressure_flags(live_orderflow_snapshot_payload)
    gate_current_sweep_long, gate_current_sweep_short = sweep_flags(current_row)
    snapshot_structure = as_dict(live_bars_snapshot_payload.get("structure_signals"))
    if snapshot_structure:
        current_sweep_long = bool(snapshot_structure.get("sweep_long", False))
        current_sweep_short = bool(snapshot_structure.get("sweep_short", False))
        current_signal_source = "crypto_shortline_live_bars_snapshot"
        current_signal_source_artifact = str(live_bars_snapshot_path)
    elif orderflow_pressure.get("available", False):
        current_sweep_long = False
        current_sweep_short = False
        current_signal_source = "crypto_shortline_live_orderflow_snapshot"
        current_signal_source_artifact = str(live_orderflow_snapshot_path)
    elif live_sweep_payload.get("available", False):
        current_sweep_long = bool(live_sweep_payload.get("sweep_long", False))
        current_sweep_short = bool(live_sweep_payload.get("sweep_short", False))
        current_signal_source = text(live_sweep_payload.get("source"))
        current_signal_source_artifact = text(live_sweep_payload.get("artifact"))
    else:
        current_sweep_long = gate_current_sweep_long
        current_sweep_short = gate_current_sweep_short
        current_signal_source = "execution_gate_snapshot"
        current_signal_source_artifact = str(current_gate_path)
    if previous_trigger_payload:
        previous_sweep_long = bool(previous_trigger_payload.get("current_sweep_long", False))
        previous_sweep_short = bool(previous_trigger_payload.get("current_sweep_short", False))
        previous_signal_source = "previous_liquidity_event_trigger"
        previous_signal_source_artifact = str(previous_trigger_path)
    else:
        previous_sweep_long, previous_sweep_short = sweep_flags(previous_row)
        previous_signal_source = "execution_gate_snapshot"
        previous_signal_source_artifact = str(previous_gate_path) if previous_gate_path else ""
    current_has_sweep = current_sweep_long or current_sweep_short
    previous_has_sweep = previous_sweep_long or previous_sweep_short
    sweep_state_changed = (
        current_sweep_long != previous_sweep_long
        or current_sweep_short != previous_sweep_short
    )
    sweep_event_newly_observed = current_has_sweep and not previous_has_sweep
    sweep_event_lost = previous_has_sweep and not current_has_sweep
    current_route_state = text(current_row.get("route_state"))
    previous_route_state = text(previous_row.get("route_state"))
    route_state_changed = current_route_state != previous_route_state
    current_missing_gates = dedupe_text(as_list(current_row.get("missing_gates")))
    liquidity_sweep_missing = not current_has_sweep
    previous_trigger_status = text(previous_trigger_payload.get("trigger_status"))
    previous_pressure_present = bool(previous_trigger_payload.get("orderflow_pressure_present", False))
    previous_pressure_side = text(previous_trigger_payload.get("orderflow_pressure_side"))
    previous_pressure_count = int(previous_trigger_payload.get("pressure_persistence_count") or 0)
    pressure_persistence_state = "none"
    pressure_persistence_count = 0
    live_bars_snapshot_status = text(live_bars_snapshot_payload.get("snapshot_status"))
    live_bars_snapshot_brief = text(live_bars_snapshot_payload.get("snapshot_brief"))
    live_bars_snapshot_decision = text(live_bars_snapshot_payload.get("snapshot_decision"))
    live_bars_snapshot_fresh = bool(live_bars_snapshot_payload.get("latest_bar_fresh", False))
    live_bars_latest_bar_age_days = live_bars_snapshot_payload.get("latest_bar_age_days")
    active_sweep_side = ""
    active_sweep_distance_bps: float | None = None
    sweep_proximity_state = "unknown"
    sweep_long_reference = float(live_bars_snapshot_payload.get("sweep_long_reference") or 0.0)
    sweep_short_reference = float(live_bars_snapshot_payload.get("sweep_short_reference") or 0.0)
    if text(orderflow_pressure.get("pressure_side")) == "long":
        active_sweep_side = "long"
        raw_distance = live_bars_snapshot_payload.get("distance_low_to_sweep_long_bps")
        active_sweep_distance_bps = float(raw_distance) if raw_distance is not None else None
    elif text(orderflow_pressure.get("pressure_side")) == "short":
        active_sweep_side = "short"
        raw_distance = live_bars_snapshot_payload.get("distance_high_to_sweep_short_bps")
        active_sweep_distance_bps = float(raw_distance) if raw_distance is not None else None
    if current_has_sweep:
        sweep_proximity_state = "triggered"
    elif active_sweep_distance_bps is None:
        sweep_proximity_state = "unknown"
    elif active_sweep_distance_bps <= SWEEP_PROXIMITY_ARMED_BPS:
        sweep_proximity_state = "armed"
    elif active_sweep_distance_bps <= SWEEP_PROXIMITY_NEAR_BPS:
        sweep_proximity_state = "near"
    else:
        sweep_proximity_state = "far"

    blocker_target_artifact = "crypto_shortline_liquidity_event_trigger"
    next_action_target_artifact = blocker_target_artifact
    if sweep_event_newly_observed:
        trigger_status = "new_liquidity_sweep_event_detected"
        trigger_decision = "refresh_shortline_execution_gate_after_liquidity_event"
        blocker_title = "Refresh shortline execution gate after new liquidity sweep event"
        next_action_target_artifact = "crypto_shortline_execution_gate"
        done_when = (
            f"{route_symbol} refreshes the shortline execution gate and advances to the next stage"
        )
    elif current_has_sweep:
        trigger_status = "liquidity_sweep_event_persisting"
        trigger_decision = "refresh_shortline_execution_gate_for_post_sweep_stage"
        blocker_title = "Liquidity sweep persists; refresh shortline execution gate for next stage"
        next_action_target_artifact = "crypto_shortline_execution_gate"
        done_when = (
            f"{route_symbol} refreshes the shortline execution gate and either advances to MSS or loses the sweep"
        )
    elif live_bars_snapshot_payload and live_bars_snapshot_status != "bars_live_snapshot_ready":
        trigger_status = "liquidity_sweep_bars_snapshot_stale"
        if orderflow_pressure.get("pressure_present", False):
            trigger_status = "liquidity_sweep_pressure_bars_snapshot_stale"
        trigger_decision = "refresh_live_bars_snapshot_then_recheck_execution_gate"
        blocker_title = text(live_bars_snapshot_payload.get("blocker_title")) or (
            "Refresh live bars snapshot before shortline event detection"
        )
        blocker_target_artifact = (
            text(live_bars_snapshot_payload.get("blocker_target_artifact"))
            or "crypto_shortline_live_bars_snapshot"
        )
        next_action_target_artifact = (
            text(live_bars_snapshot_payload.get("next_action_target_artifact"))
            or blocker_target_artifact
        )
        done_when = text(live_bars_snapshot_payload.get("done_when")) or (
            f"{route_symbol} latest bar age becomes fresh enough for sweep event detection"
        )
    elif orderflow_pressure.get("pressure_present", False):
        same_pressure_side = text(orderflow_pressure.get("pressure_side")) == previous_pressure_side
        if sweep_proximity_state == "far":
            if previous_pressure_present and same_pressure_side and previous_has_sweep is False:
                pressure_persistence_state = "persisting"
                pressure_persistence_count = max(2, previous_pressure_count + 1)
                trigger_status = "liquidity_sweep_pressure_persisting_far_from_trigger"
            else:
                pressure_persistence_state = "new"
                pressure_persistence_count = 1
                trigger_status = "liquidity_sweep_pending_pressure_far_from_trigger"
            trigger_decision = "wait_for_price_to_approach_liquidity_sweep_band_then_recheck_execution_gate"
            blocker_title = "Wait for price to approach the liquidity sweep band while orderflow pressure persists"
            done_when = (
                f"{route_symbol} moves close enough to the liquidity sweep band, records a sweep event, or the current orderflow pressure dissipates"
            )
        elif sweep_proximity_state in {"near", "armed"}:
            if previous_pressure_present and same_pressure_side and previous_has_sweep is False:
                pressure_persistence_state = "persisting"
                pressure_persistence_count = max(2, previous_pressure_count + 1)
                trigger_status = "liquidity_sweep_pressure_near_trigger"
            else:
                pressure_persistence_state = "new"
                pressure_persistence_count = 1
                trigger_status = "liquidity_sweep_pending_pressure_near_trigger"
            trigger_decision = "monitor_near_sweep_pressure_then_recheck_execution_gate"
            blocker_title = "Track near-sweep pressure for liquidity sweep confirmation"
            done_when = (
                f"{route_symbol} records a fresh liquidity sweep event, leaves the near-sweep band, or the current orderflow pressure dissipates"
            )
        elif previous_pressure_present and same_pressure_side and previous_has_sweep is False:
            pressure_persistence_state = "persisting"
            pressure_persistence_count = max(2, previous_pressure_count + 1)
            trigger_status = "liquidity_sweep_pressure_persisting"
            trigger_decision = "monitor_persistent_orderflow_pressure_for_liquidity_sweep"
            blocker_title = "Track persistent orderflow pressure for liquidity sweep confirmation"
            done_when = (
                f"{route_symbol} records a fresh liquidity sweep event or the persistent orderflow pressure decays"
            )
        else:
            pressure_persistence_state = "new"
            pressure_persistence_count = 1
            trigger_status = "liquidity_sweep_pending_orderflow_pressure"
            trigger_decision = "wait_for_liquidity_sweep_event_then_recheck_execution_gate"
            blocker_title = "Track liquidity sweep confirmation while orderflow pressure builds"
            done_when = (
                f"{route_symbol} records a fresh liquidity sweep event or the current orderflow pressure dissipates"
            )
    elif sweep_event_lost:
        trigger_status = "liquidity_sweep_event_lost_wait_new_event"
        trigger_decision = "wait_for_liquidity_sweep_event_then_recheck_execution_gate"
        blocker_title = "Track a fresh liquidity sweep event before shortline setup promotion"
        done_when = (
            f"{route_symbol} prints a fresh liquidity sweep event and the next gate refresh confirms the updated stage"
        )
    else:
        trigger_status = "no_liquidity_sweep_event_observed"
        trigger_decision = "wait_for_liquidity_sweep_event_then_recheck_execution_gate"
        blocker_title = "Track new liquidity sweep event before shortline setup promotion"
        done_when = (
            f"{route_symbol} records a new liquidity sweep event and the next gate refresh confirms the updated stage"
        )

    trigger_brief = ":".join([trigger_status, route_symbol or "-", trigger_decision, remote_market or "-"])
    blocker_detail = join_unique(
        [
            f"execution_state={text(current_row.get('execution_state')) or '-'}",
            f"route_state={current_route_state or '-'}",
            f"previous_route_state={previous_route_state or '-'}",
            f"liquidity_sweep_missing={str(liquidity_sweep_missing).lower()}",
            f"current_signal_source={current_signal_source or '-'}",
            f"previous_signal_source={previous_signal_source or '-'}",
            f"live_bars_snapshot_status={live_bars_snapshot_status or '-'}",
            f"live_bars_snapshot_fresh={str(bool(live_bars_snapshot_fresh)).lower()}",
            (
                f"live_bars_latest_bar_age_days={live_bars_latest_bar_age_days}"
                if live_bars_latest_bar_age_days is not None
                else ""
            ),
            f"live_bars_snapshot_brief={live_bars_snapshot_brief or '-'}",
            f"live_bars_snapshot_decision={live_bars_snapshot_decision or '-'}",
            f"current_sweep_long={str(current_sweep_long).lower()}",
            f"current_sweep_short={str(current_sweep_short).lower()}",
            f"previous_sweep_long={str(previous_sweep_long).lower()}",
            f"previous_sweep_short={str(previous_sweep_short).lower()}",
            f"sweep_state_changed={str(sweep_state_changed).lower()}",
            f"sweep_event_newly_observed={str(sweep_event_newly_observed).lower()}",
            f"route_state_changed={str(route_state_changed).lower()}",
            f"orderflow_pressure_present={str(bool(orderflow_pressure.get('pressure_present', False))).lower()}",
            f"orderflow_pressure_eligible={str(bool(orderflow_pressure.get('pressure_eligible', False))).lower()}",
            f"orderflow_pressure_side={text(orderflow_pressure.get('pressure_side')) or '-'}",
            f"orderflow_pressure_strength={float(orderflow_pressure.get('strength') or 0.0):g}",
            f"orderflow_pressure_quality_ok={str(bool(orderflow_pressure.get('quality_ok', False))).lower()}",
            f"orderflow_pressure_time_sync_ok={str(bool(orderflow_pressure.get('time_sync_ok', False))).lower()}",
            f"active_sweep_side={active_sweep_side or '-'}",
            (
                f"active_sweep_distance_bps={active_sweep_distance_bps:g}"
                if active_sweep_distance_bps is not None
                else ""
            ),
            f"sweep_proximity_state={sweep_proximity_state}",
            (
                f"sweep_long_reference={sweep_long_reference:g}"
                if sweep_long_reference
                else ""
            ),
            (
                f"sweep_short_reference={sweep_short_reference:g}"
                if sweep_short_reference
                else ""
            ),
            f"pressure_persistence_state={pressure_persistence_state}",
            f"pressure_persistence_count={pressure_persistence_count}",
            f"previous_trigger_status={previous_trigger_status or '-'}",
            f"orderflow_snapshot_status={text(orderflow_pressure.get('snapshot_status')) or '-'}",
            f"orderflow_snapshot_brief={text(orderflow_pressure.get('snapshot_brief')) or '-'}",
            f"orderflow_snapshot_decision={text(orderflow_pressure.get('snapshot_decision')) or '-'}",
            f"orderflow_queue_imbalance={float(orderflow_pressure.get('queue_imbalance') or 0.0):g}",
            f"orderflow_ofi_norm={float(orderflow_pressure.get('ofi_norm') or 0.0):g}",
            f"orderflow_micro_alignment={float(orderflow_pressure.get('micro_alignment') or 0.0):g}",
            f"orderflow_cvd_delta_ratio={float(orderflow_pressure.get('cvd_delta_ratio') or 0.0):g}",
            f"orderflow_cvd_veto_hint={text(orderflow_pressure.get('cvd_veto_hint'))}" if text(orderflow_pressure.get('cvd_veto_hint')) else "",
            f"orderflow_cvd_locality_status={text(orderflow_pressure.get('cvd_locality_status'))}" if text(orderflow_pressure.get('cvd_locality_status')) else "",
            f"orderflow_cvd_attack_presence={text(orderflow_pressure.get('cvd_attack_presence'))}" if text(orderflow_pressure.get('cvd_attack_presence')) else "",
            (
                f"material_change_trigger={text(material_change_payload.get('trigger_brief'))}:"
                f"{text(material_change_payload.get('trigger_decision'))}"
                if material_change_payload
                else ""
            ),
            text(current_row.get("blocker_detail")),
        ]
    )

    payload = {
        "action": "build_crypto_shortline_liquidity_event_trigger",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": route_symbol,
        "route_action": route_action,
        "remote_market": remote_market,
        "trigger_status": trigger_status,
        "trigger_brief": trigger_brief,
        "trigger_decision": trigger_decision,
        "blocker_title": blocker_title,
        "blocker_target_artifact": blocker_target_artifact,
        "blocker_detail": blocker_detail,
        "next_action": trigger_decision,
        "next_action_target_artifact": next_action_target_artifact,
        "done_when": done_when,
        "liquidity_sweep_missing": liquidity_sweep_missing,
        "current_sweep_long": current_sweep_long,
        "current_sweep_short": current_sweep_short,
        "previous_sweep_long": previous_sweep_long,
        "previous_sweep_short": previous_sweep_short,
        "current_has_sweep": current_has_sweep,
        "previous_has_sweep": previous_has_sweep,
        "sweep_state_changed": sweep_state_changed,
        "sweep_event_newly_observed": sweep_event_newly_observed,
        "sweep_event_lost": sweep_event_lost,
        "route_state_changed": route_state_changed,
        "current_route_state": current_route_state,
        "previous_route_state": previous_route_state,
        "current_signal_source": current_signal_source,
        "current_signal_source_artifact": current_signal_source_artifact,
        "orderflow_pressure_present": bool(orderflow_pressure.get("pressure_present", False)),
        "orderflow_pressure_eligible": bool(orderflow_pressure.get("pressure_eligible", False)),
        "orderflow_pressure_side": text(orderflow_pressure.get("pressure_side")),
        "orderflow_pressure_strength": float(orderflow_pressure.get("strength") or 0.0),
        "orderflow_pressure_quality_ok": bool(orderflow_pressure.get("quality_ok", False)),
        "orderflow_pressure_time_sync_ok": bool(orderflow_pressure.get("time_sync_ok", False)),
        "active_sweep_side": active_sweep_side,
        "active_sweep_distance_bps": active_sweep_distance_bps,
        "sweep_proximity_state": sweep_proximity_state,
        "sweep_long_reference": sweep_long_reference or None,
        "sweep_short_reference": sweep_short_reference or None,
        "pressure_persistence_state": pressure_persistence_state,
        "pressure_persistence_count": pressure_persistence_count,
        "previous_trigger_status": previous_trigger_status,
        "orderflow_snapshot_status": text(orderflow_pressure.get("snapshot_status")),
        "orderflow_snapshot_brief": text(orderflow_pressure.get("snapshot_brief")),
        "orderflow_snapshot_decision": text(orderflow_pressure.get("snapshot_decision")),
        "previous_signal_source": previous_signal_source,
        "previous_signal_source_artifact": previous_signal_source_artifact,
        "live_bars_snapshot_status": live_bars_snapshot_status,
        "live_bars_snapshot_brief": live_bars_snapshot_brief,
        "live_bars_snapshot_decision": live_bars_snapshot_decision,
        "live_bars_snapshot_fresh": live_bars_snapshot_fresh,
        "live_bars_latest_bar_age_days": live_bars_latest_bar_age_days,
        "artifacts": {
            "remote_intent_queue": str(intent_path) if intent_path else "",
            "crypto_route_operator_brief": str(operator_path),
            "crypto_shortline_live_bars_snapshot": str(live_bars_snapshot_path)
            if live_bars_snapshot_path
            else "",
            "crypto_shortline_live_orderflow_snapshot": str(live_orderflow_snapshot_path)
            if live_orderflow_snapshot_path
            else "",
            "bars_live_snapshot": text(live_sweep_payload.get("artifact")),
            "current_crypto_shortline_execution_gate": str(current_gate_path),
            "previous_crypto_shortline_execution_gate": str(previous_gate_path)
            if previous_gate_path
            else "",
            "previous_crypto_shortline_liquidity_event_trigger": str(previous_trigger_path)
            if previous_trigger_path
            else "",
            "crypto_shortline_material_change_trigger": str(material_change_path)
            if material_change_path
            else "",
        },
    }

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_liquidity_event_trigger.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_liquidity_event_trigger.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_liquidity_event_trigger_checksum.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
                "generated_at_utc": fmt_utc(reference_now),
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
        keep=args.artifact_keep,
        ttl_hours=args.artifact_ttl_hours,
    )
    payload.update(
        {
            "artifact": str(artifact),
            "markdown": str(markdown),
            "checksum": str(checksum),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
