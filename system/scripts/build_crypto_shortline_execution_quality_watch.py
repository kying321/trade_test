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


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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
        "*_crypto_shortline_execution_quality_watch.json",
        "*_crypto_shortline_execution_quality_watch.md",
        "*_crypto_shortline_execution_quality_watch_checksum.json",
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


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Shortline Execution Quality Watch",
            "",
            f"- brief: `{text(payload.get('watch_brief'))}`",
            f"- decision: `{text(payload.get('watch_decision'))}`",
            f"- pattern_family: `{text(payload.get('pattern_family')) or '-'}`",
            f"- pattern_stage: `{text(payload.get('pattern_stage')) or '-'}`",
            f"- execution_quality_score: `{payload.get('execution_quality_score')}`",
            f"- micro_quality_ok: `{payload.get('micro_quality_ok')}`",
            f"- trust_ok: `{payload.get('trust_ok')}`",
            f"- time_sync_ok: `{payload.get('time_sync_ok')}`",
            f"- queue_imbalance: `{payload.get('queue_imbalance')}`",
            f"- ofi_norm: `{payload.get('ofi_norm')}`",
            f"- micro_alignment: `{payload.get('micro_alignment')}`",
            f"- cvd_delta_ratio: `{payload.get('cvd_delta_ratio')}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def bounded_component(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return max(0.0, min(abs(float(value)) / cap, 1.0))


def compute_execution_quality_score(
    *,
    snapshot_status: str,
    micro_quality_ok: bool,
    trust_ok: bool,
    time_sync_ok: bool,
    queue_imbalance: float,
    ofi_norm: float,
    micro_alignment: float,
    cvd_delta_ratio: float,
    trade_count: int,
    evidence_score: float,
    cvd_veto_hint: str,
) -> float:
    score = 0.0
    if snapshot_status == "live_orderflow_snapshot_ready":
        score += 25.0
    if micro_quality_ok:
        score += 15.0
    if trust_ok:
        score += 15.0
    if time_sync_ok:
        score += 10.0
    score += 10.0 * bounded_component(queue_imbalance, 0.25)
    score += 10.0 * bounded_component(ofi_norm, 0.25)
    score += 7.5 * bounded_component(micro_alignment, 0.25)
    score += 7.5 * bounded_component(cvd_delta_ratio, 0.25)
    score += 5.0 * bounded_component(float(trade_count), 200.0)
    score += 5.0 * max(0.0, min(evidence_score, 1.0))
    if cvd_veto_hint:
        score -= 10.0
    return max(0.0, min(round(score, 3), 100.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-owned crypto shortline execution-quality watch artifact."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)

    intent_path = find_latest(review_dir, "*_remote_intent_queue.json", reference_now)
    operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json", reference_now)
    snapshot_path = find_latest(
        review_dir, "*_crypto_shortline_live_orderflow_snapshot.json", reference_now
    )
    pattern_router_path = find_latest(
        review_dir, "*_crypto_shortline_pattern_router.json", reference_now
    )
    if operator_path is None:
        raise SystemExit("missing_required_artifacts:crypto_route_operator_brief")

    intent_payload = (
        load_json_mapping(intent_path) if intent_path is not None and intent_path.exists() else {}
    )
    operator_payload = load_json_mapping(operator_path)
    snapshot_payload = (
        load_json_mapping(snapshot_path)
        if snapshot_path is not None and snapshot_path.exists()
        else {}
    )
    pattern_router_payload = (
        load_json_mapping(pattern_router_path)
        if pattern_router_path is not None and pattern_router_path.exists()
        else {}
    )

    symbol = route_symbol(intent_payload, operator_payload)
    action = route_action(intent_payload, operator_payload)
    remote_market = text(intent_payload.get("remote_market")) or "portfolio_margin_um"

    pattern_family = text(pattern_router_payload.get("pattern_family"))
    pattern_stage = text(pattern_router_payload.get("pattern_stage"))
    pattern_status = text(pattern_router_payload.get("pattern_status"))
    snapshot_status = text(snapshot_payload.get("snapshot_status"))
    snapshot_brief = text(snapshot_payload.get("snapshot_brief"))
    snapshot_decision = text(snapshot_payload.get("snapshot_decision"))
    micro_quality_ok = bool(snapshot_payload.get("micro_quality_ok", False))
    trust_ok = bool(snapshot_payload.get("trust_ok", False))
    time_sync_ok = bool(snapshot_payload.get("time_sync_ok", False))
    queue_imbalance = to_float(snapshot_payload.get("queue_imbalance"))
    ofi_norm = to_float(snapshot_payload.get("ofi_norm"))
    micro_alignment = to_float(snapshot_payload.get("micro_alignment"))
    cvd_delta_ratio = to_float(snapshot_payload.get("cvd_delta_ratio"))
    cvd_veto_hint = text(snapshot_payload.get("cvd_veto_hint"))
    trade_count = int(snapshot_payload.get("trade_count") or 0)
    evidence_score = to_float(snapshot_payload.get("evidence_score"))
    execution_quality_score = compute_execution_quality_score(
        snapshot_status=snapshot_status,
        micro_quality_ok=micro_quality_ok,
        trust_ok=trust_ok,
        time_sync_ok=time_sync_ok,
        queue_imbalance=queue_imbalance,
        ofi_norm=ofi_norm,
        micro_alignment=micro_alignment,
        cvd_delta_ratio=cvd_delta_ratio,
        trade_count=trade_count,
        evidence_score=evidence_score,
        cvd_veto_hint=cvd_veto_hint,
    )

    value_rotation_active = pattern_family == "value_rotation_scalp"

    if not symbol:
        watch_status = "execution_quality_route_symbol_missing"
        watch_decision = "resolve_route_symbol_before_execution_quality_check"
        blocker_title = "Resolve route symbol before execution-quality review"
        done_when = "the route symbol is available for shortline execution-quality review"
    elif snapshot_status in {"", "live_orderflow_snapshot_missing"}:
        watch_status = "execution_quality_snapshot_missing"
        watch_decision = "capture_execution_quality_snapshot_then_recheck_execution_gate"
        blocker_title = "Capture execution-quality snapshot before shortline setup promotion"
        done_when = f"{symbol} has a live orderflow snapshot artifact for execution-quality review"
    elif (
        snapshot_status != "live_orderflow_snapshot_ready"
        or not micro_quality_ok
        or not trust_ok
        or not time_sync_ok
        or execution_quality_score < 55.0
    ):
        if value_rotation_active:
            watch_status = "value_rotation_scalp_execution_quality_degraded"
            watch_decision = "repair_value_rotation_execution_quality_then_recheck_execution_gate"
            blocker_title = "Repair value-rotation execution quality before shortline scalp promotion"
            done_when = (
                f"{symbol} keeps live orderflow snapshot ready with trust_ok, time_sync_ok, and "
                "cost-sensitive execution quality for the value-rotation scalp"
            )
        else:
            watch_status = "execution_quality_degraded"
            watch_decision = "repair_execution_quality_then_recheck_execution_gate"
            blocker_title = "Repair shortline execution quality before setup promotion"
            done_when = (
                f"{symbol} keeps live orderflow snapshot ready with trustworthy microstructure quality"
            )
    else:
        if value_rotation_active:
            watch_status = "value_rotation_scalp_execution_quality_clear"
            watch_decision = "recheck_execution_gate_after_value_rotation_execution_quality_clear"
            blocker_title = "Value-rotation execution quality clear for shortline scalp promotion"
            done_when = (
                f"{symbol} reintroduces execution-quality degradation while the value-rotation scalp remains active"
            )
        else:
            watch_status = "execution_quality_watch_clear"
            watch_decision = "recheck_execution_gate_after_execution_quality_clear"
            blocker_title = "Execution quality clear for shortline setup promotion"
            done_when = f"{symbol} reintroduces execution-quality degradation or leaves Setup_Ready"

    watch_brief = ":".join([watch_status, symbol or "-", watch_decision, remote_market or "-"])
    blocker_detail = join_unique(
        [
            snapshot_brief,
            f"pattern_router={pattern_status}:{pattern_family}:{pattern_stage}"
            if pattern_status or pattern_family or pattern_stage
            else "",
            f"execution_quality_score={execution_quality_score:g}",
            f"micro_quality_ok={str(micro_quality_ok).lower()}",
            f"trust_ok={str(trust_ok).lower()}",
            f"time_sync_ok={str(time_sync_ok).lower()}",
            f"queue_imbalance={queue_imbalance:g}",
            f"ofi_norm={ofi_norm:g}",
            f"micro_alignment={micro_alignment:g}",
            f"cvd_delta_ratio={cvd_delta_ratio:g}",
            f"trade_count={trade_count}",
            f"evidence_score={evidence_score:g}",
            f"cvd_veto_hint={cvd_veto_hint}" if cvd_veto_hint else "",
            f"snapshot_decision={snapshot_decision}" if snapshot_decision else "",
        ]
    )

    payload = {
        "action": "build_crypto_shortline_execution_quality_watch",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": symbol,
        "route_action": action,
        "remote_market": remote_market,
        "watch_status": watch_status,
        "watch_brief": watch_brief,
        "watch_decision": watch_decision,
        "blocker_title": blocker_title,
        "blocker_target_artifact": "crypto_shortline_execution_quality_watch",
        "blocker_detail": blocker_detail,
        "next_action": watch_decision,
        "next_action_target_artifact": "crypto_shortline_execution_quality_watch",
        "done_when": done_when,
        "pattern_family": pattern_family,
        "pattern_stage": pattern_stage,
        "pattern_status": pattern_status,
        "execution_quality_score": execution_quality_score,
        "snapshot_status": snapshot_status,
        "snapshot_brief": snapshot_brief,
        "snapshot_decision": snapshot_decision,
        "micro_quality_ok": micro_quality_ok,
        "trust_ok": trust_ok,
        "time_sync_ok": time_sync_ok,
        "queue_imbalance": queue_imbalance,
        "ofi_norm": ofi_norm,
        "micro_alignment": micro_alignment,
        "cvd_delta_ratio": cvd_delta_ratio,
        "trade_count": trade_count,
        "evidence_score": evidence_score,
        "cvd_veto_hint": cvd_veto_hint,
        "artifacts": {
            "remote_intent_queue": str(intent_path) if intent_path else "",
            "crypto_route_operator_brief": str(operator_path),
            "crypto_shortline_live_orderflow_snapshot": str(snapshot_path)
            if snapshot_path
            else "",
            "crypto_shortline_pattern_router": str(pattern_router_path)
            if pattern_router_path
            else "",
        },
    }

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_execution_quality_watch.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_execution_quality_watch.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_execution_quality_watch_checksum.json"
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
