#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
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
DEFAULT_LOCATION_PRIORITY = ("HVN", "POC")
FAR_ROTATION_RATIO = 0.35
APPROACHING_ROTATION_RATIO = 0.5


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


def safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


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


def artifact_sort_key(path: Path, reference_now: dt.datetime | None = None) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    effective_now = reference_now or now_utc()
    future_cutoff = effective_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    return (0 if is_future else 1, artifact_stamp(path), path.stat().st_mtime, path.name)


def find_latest(review_dir: Path, pattern: str, reference_now: dt.datetime | None = None) -> Path | None:
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


def load_symbol_bars_from_csv(path: Path, symbol: str) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_symbol = text(row.get("symbol")).upper()
            if row_symbol != symbol.upper():
                continue
            close_value = safe_float(row.get("close"))
            volume_value = safe_float(row.get("volume"))
            if close_value is None or volume_value is None:
                continue
            rows.append((close_value, max(0.0, volume_value)))
    return rows


def build_profile_rotation_metrics(
    bars_rows: list[tuple[float, float]],
    *,
    preferred_location_tags: list[str],
    bins: int,
) -> dict[str, Any]:
    if not bars_rows:
        return {}
    closes = [row[0] for row in bars_rows]
    volumes = [row[1] for row in bars_rows]
    lo = min(closes)
    hi = max(closes)
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) <= 1e-9:
        return {}

    bin_count = max(6, min(24, int(bins)))
    node_volumes = [0.0 for _ in range(bin_count)]
    current_idx = 0
    for idx, (close_px, vol) in enumerate(zip(closes, volumes, strict=False)):
        node_idx = min(
            max(int(((float(close_px) - lo) / max(1e-9, hi - lo)) * bin_count), 0),
            bin_count - 1,
        )
        node_volumes[node_idx] += float(vol)
        if idx == len(closes) - 1:
            current_idx = node_idx

    poc_idx = int(max(range(len(node_volumes)), key=lambda item: node_volumes[item]))
    positive = sorted(v for v in node_volumes if v > 0.0)
    if not positive:
        return {}
    hvn_threshold = positive[max(0, int(0.75 * (len(positive) - 1)))]
    hvn_candidates = [
        idx for idx, volume in enumerate(node_volumes) if volume >= hvn_threshold and idx != current_idx
    ]
    if not hvn_candidates and poc_idx != current_idx:
        hvn_candidates = [poc_idx]

    edges = [lo + ((hi - lo) * idx / bin_count) for idx in range(bin_count + 1)]

    def center_price(index: int) -> float:
        return float((edges[index] + edges[index + 1]) / 2.0)

    def distance_bps(index: int) -> float:
        current_close = float(closes[-1])
        if abs(current_close) <= 1e-9:
            return 0.0
        return round(abs(center_price(index) - current_close) / abs(current_close) * 10000.0, 6)

    target_rows: list[dict[str, Any]] = []
    if "POC" in preferred_location_tags:
        target_rows.append(
            {
                "tag": "POC",
                "bin_index": poc_idx,
                "bin_distance": abs(current_idx - poc_idx),
                "distance_bps": distance_bps(poc_idx),
            }
        )
    if "HVN" in preferred_location_tags and hvn_candidates:
        nearest_hvn = min(
            hvn_candidates,
            key=lambda idx: (abs(current_idx - idx), distance_bps(idx), -node_volumes[idx]),
        )
        target_rows.append(
            {
                "tag": "HVN",
                "bin_index": nearest_hvn,
                "bin_distance": abs(current_idx - nearest_hvn),
                "distance_bps": distance_bps(nearest_hvn),
            }
        )

    if not target_rows:
        return {}

    nearest_target = min(
        target_rows,
        key=lambda row: (int(row["bin_distance"]), float(row["distance_bps"]), str(row["tag"])),
    )
    return {
        "target_rows": target_rows,
        "current_bin_index": current_idx,
        "poc_bin_index": poc_idx,
        "nearest_target_tag": str(nearest_target["tag"]),
        "nearest_target_bin_index": int(nearest_target["bin_index"]),
        "nearest_target_bin_distance": int(nearest_target["bin_distance"]),
        "nearest_target_distance_bps": float(nearest_target["distance_bps"]),
    }


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
        "*_crypto_shortline_profile_location_watch.json",
        "*_crypto_shortline_profile_location_watch.md",
        "*_crypto_shortline_profile_location_watch_checksum.json",
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


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Shortline Profile Location Watch",
            "",
            f"- brief: `{text(payload.get('watch_brief'))}`",
            f"- decision: `{text(payload.get('watch_decision'))}`",
            f"- location_tag: `{text(payload.get('location_tag'))}`",
            f"- profile_alignment_state: `{text(payload.get('profile_alignment_state')) or '-'}`",
            f"- preferred_location_tags: `{', '.join(as_list(payload.get('preferred_location_tags'))) or '-'}`",
            f"- profile_location_missing: `{payload.get('profile_location_missing')}`",
            f"- key_level_context_missing: `{payload.get('key_level_context_missing')}`",
            f"- key_level_context_effective_status: `{text(payload.get('key_level_context_effective_status')) or '-'}`",
            f"- rotation_proximity_state: `{text(payload.get('rotation_proximity_state')) or '-'}`",
            f"- profile_rotation_alignment_band: `{text(payload.get('profile_rotation_alignment_band')) or '-'}`",
            f"- profile_rotation_next_milestone: `{text(payload.get('profile_rotation_next_milestone')) or '-'}`",
            f"- profile_rotation_confidence: `{payload.get('profile_rotation_confidence')}`",
            f"- active_rotation_targets: `{', '.join(as_list(payload.get('active_rotation_targets'))) or '-'}`",
            f"- profile_rotation_target_tag: `{text(payload.get('profile_rotation_target_tag')) or '-'}`",
            f"- profile_rotation_target_bin_distance: `{payload.get('profile_rotation_target_bin_distance')}`",
            f"- profile_rotation_target_distance_bps: `{payload.get('profile_rotation_target_distance_bps')}`",
            f"- key_level_confirmed: `{payload.get('key_level_confirmed')}`",
            f"- attack_presence: `{text(payload.get('attack_presence')) or '-'}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-owned crypto shortline profile-location watch artifact."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def rotation_band_and_milestone(rotation_proximity_state: str) -> tuple[str, str]:
    state = text(rotation_proximity_state)
    if state == "far":
        return ("far", "toward_hvn_poc")
    if state == "approaching":
        return ("approaching", "into_final_band")
    if state == "final_band":
        return ("final", "last_band")
    if state == "aligned":
        return ("aligned", "aligned")
    return ("", "")


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)

    intent_path = find_latest(review_dir, "*_remote_intent_queue.json", reference_now)
    operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json", reference_now)
    gate_path = find_latest(review_dir, "*_crypto_shortline_execution_gate.json", reference_now)
    live_bars_snapshot_path = find_latest(
        review_dir,
        "*_crypto_shortline_live_bars_snapshot.json",
        reference_now,
    )

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
    live_bars_snapshot_payload = (
        load_json_mapping(live_bars_snapshot_path)
        if live_bars_snapshot_path is not None and live_bars_snapshot_path.exists()
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

    gate_row = find_gate_row(gate_payload, route_symbol)
    shortline_policy = as_dict(gate_payload.get("shortline_policy"))
    profile_proxy = as_dict(gate_row.get("profile_proxy"))
    micro_signals = as_dict(gate_row.get("micro_signals"))
    missing_gates = dedupe_text(as_list(gate_row.get("missing_gates")))
    preferred_location_tags = dedupe_text(
        as_list(shortline_policy.get("location_priority"))
    ) or list(DEFAULT_LOCATION_PRIORITY)
    location_tag = text(gate_row.get("location_tag")) or text(profile_proxy.get("location_tag")) or "MID"
    execution_state = text(gate_row.get("execution_state")) or "Bias_Only"
    route_state = text(gate_row.get("route_state"))
    profile_location_missing_code = next(
        (item for item in missing_gates if item.startswith("profile_location=")),
        "",
    )
    profile_location_missing = bool(profile_location_missing_code)
    key_level_context_missing = "cvd_key_level_context" in missing_gates
    key_level_confirmed = bool(micro_signals.get("key_level_confirmed", False))
    attack_side = text(micro_signals.get("attack_side"))
    attack_presence = text(micro_signals.get("attack_presence"))
    current_bin_volume = float(profile_proxy.get("current_bin_volume") or 0.0)
    poc_bin_volume = float(profile_proxy.get("poc_bin_volume") or 0.0)
    profile_volume_ratio = (
        round(current_bin_volume / poc_bin_volume, 6) if poc_bin_volume > 0 else None
    )
    live_bars_artifact = Path(
        text(live_bars_snapshot_payload.get("bars_artifact"))
    ).expanduser() if text(live_bars_snapshot_payload.get("bars_artifact")) else None
    profile_rotation_metrics = {}
    if live_bars_artifact and live_bars_artifact.exists():
        profile_rotation_metrics = build_profile_rotation_metrics(
            load_symbol_bars_from_csv(live_bars_artifact, route_symbol),
            preferred_location_tags=preferred_location_tags,
            bins=int(profile_proxy.get("bins") or 12),
        )
    profile_rotation_target_tag = text(profile_rotation_metrics.get("nearest_target_tag"))
    profile_rotation_target_bin_distance = (
        int(profile_rotation_metrics["nearest_target_bin_distance"])
        if "nearest_target_bin_distance" in profile_rotation_metrics
        else None
    )
    profile_rotation_target_distance_bps = (
        round(float(profile_rotation_metrics["nearest_target_distance_bps"]), 6)
        if "nearest_target_distance_bps" in profile_rotation_metrics
        else None
    )
    active_rotation_targets = preferred_location_tags if profile_location_missing else [location_tag]
    if profile_location_missing and profile_volume_ratio is not None:
        rotation_score = max(0.0, min(1.0, profile_volume_ratio))
        if key_level_confirmed:
            rotation_score = min(1.0, rotation_score + 0.1)
        if attack_presence:
            rotation_score = min(1.0, rotation_score + 0.05)
        profile_rotation_confidence = round(rotation_score, 6)
    elif not profile_location_missing:
        profile_rotation_confidence = 1.0
    else:
        profile_rotation_confidence = None

    blocker_target_artifact = "crypto_shortline_profile_location_watch"
    next_action_target_artifact = "crypto_shortline_profile_location_watch"
    if profile_location_missing:
        profile_alignment_state = (
            f"misaligned:{location_tag}->{','.join(preferred_location_tags)}"
        )
    else:
        profile_alignment_state = f"aligned:{location_tag}"
    if profile_location_missing and profile_volume_ratio is not None:
        if profile_rotation_target_bin_distance is not None:
            if profile_rotation_target_bin_distance >= 3:
                rotation_proximity_state = "far"
            elif profile_rotation_target_bin_distance == 2:
                rotation_proximity_state = "approaching"
            else:
                rotation_proximity_state = "final_band"
        elif profile_volume_ratio < FAR_ROTATION_RATIO:
            rotation_proximity_state = "far"
        elif profile_volume_ratio < APPROACHING_ROTATION_RATIO:
            rotation_proximity_state = "approaching"
        else:
            rotation_proximity_state = "final_band"
    elif profile_location_missing:
        rotation_proximity_state = "unknown"
    else:
        rotation_proximity_state = "aligned"
    profile_rotation_alignment_band, profile_rotation_next_milestone = (
        rotation_band_and_milestone(rotation_proximity_state)
    )
    if profile_location_missing and key_level_confirmed:
        key_level_context_effective_status = "key_level_ready_waiting_profile_alignment"
    elif key_level_context_missing:
        key_level_context_effective_status = "key_level_context_missing"
    else:
        key_level_context_effective_status = "key_level_context_ready"

    if execution_state == "Setup_Ready" and not profile_location_missing and not key_level_context_missing:
        watch_status = "profile_location_aligned"
        watch_decision = "review_guarded_canary_promotion"
        blocker_title = "Profile-location aligned for next-stage shortline review"
        done_when = (
            f"{route_symbol} loses preferred profile location, loses key-level context, or leaves Setup_Ready"
        )
    elif profile_location_missing:
        if key_level_confirmed:
            if rotation_proximity_state == "far":
                watch_status = f"profile_location_{location_tag.lower()}_key_level_ready_rotation_far"
                watch_decision = "monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate"
                blocker_title = "Track LVN-to-HVN/POC rotation before shortline setup promotion"
            elif rotation_proximity_state == "approaching":
                watch_status = f"profile_location_{location_tag.lower()}_key_level_ready_rotation_approaching"
                watch_decision = (
                    "monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate"
                )
                blocker_title = "Track final LVN-to-HVN/POC rotation before shortline setup promotion"
            elif rotation_proximity_state == "final_band":
                watch_status = (
                    f"profile_location_{location_tag.lower()}_key_level_ready_rotation_final_band"
                )
                watch_decision = (
                    "monitor_last_profile_rotation_into_hvn_poc_then_recheck_execution_gate"
                )
                blocker_title = (
                    "Track last LVN-to-HVN/POC rotation band before shortline setup promotion"
                )
            else:
                watch_status = f"profile_location_{location_tag.lower()}_key_level_ready_waiting_alignment"
                watch_decision = "wait_for_profile_location_alignment_then_recheck_execution_gate"
                blocker_title = "Track profile-location alignment before shortline setup promotion"
        elif key_level_context_missing:
            watch_status = f"profile_location_{location_tag.lower()}_key_level_context_blocked"
            watch_decision = "wait_for_profile_location_alignment_then_recheck_execution_gate"
            blocker_title = "Track profile-location alignment before shortline setup promotion"
        else:
            watch_status = f"profile_location_{location_tag.lower()}_blocked"
            watch_decision = "wait_for_profile_location_alignment_then_recheck_execution_gate"
            blocker_title = "Track profile-location alignment before shortline setup promotion"
        done_when = (
            f"{route_symbol} rotates into {','.join(preferred_location_tags)} and clears any key-level context blocker, "
            "then the shortline execution gate refresh confirms the next stage"
        )
    elif key_level_context_missing:
        watch_status = "profile_location_ready_key_level_context_missing"
        watch_decision = "wait_for_cvd_key_level_context_then_recheck_execution_gate"
        blocker_title = "Track CVD key-level context before shortline setup promotion"
        done_when = (
            f"{route_symbol} keeps preferred profile location and clears cvd_key_level_context, "
            "then the shortline execution gate refresh confirms the next stage"
        )
    else:
        watch_status = "profile_location_watch_unclear"
        watch_decision = "refresh_shortline_execution_gate_after_profile_context_change"
        blocker_title = "Refresh shortline execution gate after profile-context change"
        done_when = (
            f"{route_symbol} refreshes profile context and the shortline execution gate exposes the next blocking stage"
        )

    watch_brief = ":".join([watch_status, route_symbol or "-", watch_decision, remote_market or "-"])
    blocker_detail = join_unique(
        [
            f"execution_state={execution_state}",
            f"route_state={route_state or '-'}",
            f"location_tag={location_tag}",
            f"preferred_location_tags={','.join(preferred_location_tags)}",
            profile_location_missing_code,
            "cvd_key_level_context" if key_level_context_missing else "",
            f"key_level_confirmed={str(key_level_confirmed).lower()}",
            f"attack_side={attack_side}" if attack_side else "",
            f"attack_presence={attack_presence}" if attack_presence else "",
            (
                f"profile_rotation_alignment_band={profile_rotation_alignment_band}"
                if profile_rotation_alignment_band
                else ""
            ),
            (
                f"profile_rotation_next_milestone={profile_rotation_next_milestone}"
                if profile_rotation_next_milestone
                else ""
            ),
            (
                f"profile_rotation_confidence={profile_rotation_confidence}"
                if profile_rotation_confidence is not None
                else ""
            ),
            (
                f"active_rotation_targets={','.join(active_rotation_targets)}"
                if active_rotation_targets
                else ""
            ),
            (
                f"profile_rotation_target_tag={profile_rotation_target_tag}"
                if profile_rotation_target_tag
                else ""
            ),
            (
                f"profile_rotation_target_bin_distance={profile_rotation_target_bin_distance}"
                if profile_rotation_target_bin_distance is not None
                else ""
            ),
            (
                f"profile_rotation_target_distance_bps={profile_rotation_target_distance_bps}"
                if profile_rotation_target_distance_bps is not None
                else ""
            ),
            (
                f"profile_volume_ratio={profile_volume_ratio}"
                if profile_volume_ratio is not None
                else ""
            ),
            text(gate_row.get("blocker_detail")),
        ]
    )

    payload = {
        "action": "build_crypto_shortline_profile_location_watch",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": route_symbol,
        "route_action": route_action,
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
        "execution_state": execution_state,
        "route_state": route_state,
        "location_tag": location_tag,
        "preferred_location_tags": preferred_location_tags,
        "profile_alignment_state": profile_alignment_state,
        "rotation_proximity_state": rotation_proximity_state,
        "profile_rotation_alignment_band": profile_rotation_alignment_band,
        "profile_rotation_next_milestone": profile_rotation_next_milestone,
        "profile_rotation_confidence": profile_rotation_confidence,
        "active_rotation_targets": active_rotation_targets,
        "profile_rotation_target_tag": profile_rotation_target_tag,
        "profile_rotation_target_bin_distance": profile_rotation_target_bin_distance,
        "profile_rotation_target_distance_bps": profile_rotation_target_distance_bps,
        "profile_location_missing": profile_location_missing,
        "profile_location_missing_code": profile_location_missing_code,
        "key_level_context_missing": key_level_context_missing,
        "key_level_context_effective_status": key_level_context_effective_status,
        "key_level_confirmed": key_level_confirmed,
        "attack_side": attack_side,
        "attack_presence": attack_presence,
        "current_bin_volume": current_bin_volume,
        "poc_bin_volume": poc_bin_volume,
        "profile_volume_ratio": profile_volume_ratio,
        "artifacts": {
            "remote_intent_queue": str(intent_path) if intent_path else "",
            "crypto_route_operator_brief": str(operator_path),
            "crypto_shortline_execution_gate": str(gate_path),
            "crypto_shortline_live_bars_snapshot": str(live_bars_snapshot_path)
            if live_bars_snapshot_path
            else "",
        },
    }

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_profile_location_watch.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_profile_location_watch.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_profile_location_watch_checksum.json"

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

    payload["artifact"] = str(artifact)
    payload["markdown"] = str(markdown)
    payload["checksum"] = str(checksum)
    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_paths=[artifact, markdown, checksum],
        keep=args.artifact_keep,
        ttl_hours=args.artifact_ttl_hours,
    )
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
