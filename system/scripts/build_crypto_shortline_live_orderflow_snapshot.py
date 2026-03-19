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

from build_crypto_shortline_execution_gate import load_shortline_policy, micro_signals


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
DEFAULT_ARTIFACT_DIR = DEFAULT_OUTPUT_ROOT / "artifacts" / "micro_capture"
DEFAULT_CONFIG_PATH = SYSTEM_ROOT / "config.yaml"
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


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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


def artifact_sort_key(path: Path, reference_now: dt.datetime) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    future_cutoff = reference_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    return (0 if is_future else 1, artifact_stamp(path), path.stat().st_mtime, path.name)


def find_latest(review_dir: Path, pattern: str, reference_now: dt.datetime) -> Path | None:
    files = list(review_dir.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda item: artifact_sort_key(item, reference_now))


def find_latest_micro_capture(artifact_dir: Path, reference_now: dt.datetime) -> Path | None:
    candidates = sorted(
        artifact_dir.glob("*_micro_capture.json"),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    for path in candidates:
        stamp_dt = parsed_artifact_stamp(path)
        if stamp_dt is None:
            continue
        if stamp_dt <= reference_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES):
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
        "*_crypto_shortline_live_orderflow_snapshot.json",
        "*_crypto_shortline_live_orderflow_snapshot.md",
        "*_crypto_shortline_live_orderflow_snapshot_checksum.json",
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


def find_symbol_micro_row(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(payload.get("selected_micro")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Shortline Live Orderflow Snapshot",
            "",
            f"- brief: `{text(payload.get('snapshot_brief'))}`",
            f"- status: `{text(payload.get('snapshot_status'))}`",
            f"- decision: `{text(payload.get('snapshot_decision'))}`",
            f"- route_symbol: `{text(payload.get('route_symbol'))}`",
            f"- queue_imbalance: `{payload.get('queue_imbalance')}`",
            f"- ofi_norm: `{payload.get('ofi_norm')}`",
            f"- micro_alignment: `{payload.get('micro_alignment')}`",
            f"- cvd_delta_ratio: `{payload.get('cvd_delta_ratio')}`",
            f"- raw_time_sync_ok: `{payload.get('raw_time_sync_ok')}`",
            f"- time_sync_ok: `{payload.get('time_sync_ok')}`",
            f"- time_sync_resolution_source: `{text(payload.get('time_sync_resolution_source'))}`",
            f"- time_sync_verification_brief: `{text(payload.get('time_sync_verification_brief'))}`",
            f"- micro_quality_ok: `{payload.get('micro_quality_ok')}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-owned live orderflow snapshot for crypto shortline quality gating."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--micro-capture-file", default="")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)

    intent_path = find_latest(review_dir, "*_remote_intent_queue.json", reference_now)
    operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json", reference_now)
    time_sync_verification_path = find_latest(
        review_dir, "*_system_time_sync_repair_verification_report.json", reference_now
    )
    if operator_path is None:
        raise SystemExit("missing_required_artifacts:crypto_route_operator_brief")

    intent_payload = (
        load_json_mapping(intent_path) if intent_path is not None and intent_path.exists() else {}
    )
    operator_payload = load_json_mapping(operator_path)
    time_sync_verification_payload = (
        load_json_mapping(time_sync_verification_path)
        if time_sync_verification_path is not None and time_sync_verification_path.exists()
        else {}
    )
    symbol = route_symbol(intent_payload, operator_payload)
    action = route_action(intent_payload, operator_payload)
    remote_market = text(intent_payload.get("remote_market")) or "portfolio_margin_um"

    micro_path = (
        Path(args.micro_capture_file).expanduser().resolve()
        if text(args.micro_capture_file)
        else find_latest_micro_capture(artifact_dir, reference_now)
    )

    snapshot_status = "live_orderflow_snapshot_missing"
    snapshot_decision = "collect_live_orderflow_snapshot_before_shortline_signal_quality_check"
    blocker_title = "Collect live orderflow snapshot before shortline signal quality review"
    blocker_detail = ""
    done_when = "a route symbol and matching live orderflow snapshot are available"
    micro_capture_status = ""
    micro_capture_summary = ""
    selected_micro_count = 0
    symbol_present = False
    schema_ok = False
    gap_ok = False
    raw_time_sync_ok = False
    time_sync_ok = False
    time_sync_resolution_source = "micro_capture"
    micro_quality_ok = False
    trust_ok = False
    cvd_ready = False
    cvd_long = False
    cvd_short = False
    cvd_context_mode = ""
    cvd_veto_hint = ""
    cvd_locality_status = ""
    cvd_attack_presence = ""
    cvd_attack_side = ""
    queue_imbalance = 0.0
    ofi_norm = 0.0
    micro_alignment = 0.0
    cvd_delta_ratio = 0.0
    trade_count = 0
    evidence_score = 0.0
    raw_row: dict[str, Any] = {}
    time_sync_verification_brief = text(time_sync_verification_payload.get("verification_brief"))
    time_sync_verification_cleared = bool(time_sync_verification_payload.get("cleared", False))
    verification_stamp = (
        parsed_artifact_stamp(time_sync_verification_path)
        if time_sync_verification_path is not None
        else None
    )
    micro_stamp = parsed_artifact_stamp(micro_path) if micro_path is not None else None

    if not symbol:
        snapshot_status = "route_symbol_missing_for_live_orderflow_snapshot"
        snapshot_decision = "resolve_route_symbol_before_shortline_signal_quality_check"
        blocker_title = "Resolve route symbol before live orderflow snapshot review"
        blocker_detail = "route_symbol_missing"
        done_when = "the route symbol is available from remote_intent_queue or crypto_route_operator_brief"
    elif micro_path is None or not micro_path.exists():
        blocker_detail = f"micro_capture_artifact_missing:{symbol}"
    else:
        micro_payload = load_json_mapping(micro_path)
        micro_capture_status = text(micro_payload.get("status"))
        micro_capture_summary = text(micro_payload.get("summary"))
        selected_rows = as_list(micro_payload.get("selected_micro"))
        selected_micro_count = len(selected_rows)
        if not selected_rows:
            snapshot_status = "live_orderflow_snapshot_selected_micro_missing"
            snapshot_decision = "refresh_micro_capture_before_shortline_signal_quality_check"
            blocker_title = "Refresh micro capture before shortline signal quality review"
            blocker_detail = join_unique(
                [
                    f"selected_micro_missing:{symbol}",
                    micro_capture_status,
                    micro_capture_summary,
                ]
            )
            done_when = f"{symbol} is present in selected_micro for the latest micro_capture artifact"
        else:
            raw_row = find_symbol_micro_row(micro_payload, symbol)
            if not raw_row:
                snapshot_status = "route_symbol_missing_in_live_orderflow_snapshot"
                snapshot_decision = "capture_route_symbol_orderflow_before_shortline_signal_quality_check"
                blocker_title = "Capture route symbol orderflow before shortline signal quality review"
                blocker_detail = join_unique(
                    [
                        f"route_symbol_missing_in_selected_micro:{symbol}",
                        micro_capture_status,
                        micro_capture_summary,
                    ]
                )
                done_when = f"{symbol} appears in selected_micro for the latest micro_capture artifact"
            else:
                symbol_present = True
                schema_ok = bool(raw_row.get("schema_ok", False))
                gap_ok = bool(raw_row.get("gap_ok", False))
                raw_time_sync_ok = bool(raw_row.get("time_sync_ok", False))
                time_sync_ok = raw_time_sync_ok
                if raw_time_sync_ok:
                    time_sync_resolution_source = "micro_capture"
                elif (
                    time_sync_verification_cleared
                    and verification_stamp is not None
                    and (micro_stamp is None or verification_stamp >= micro_stamp)
                ):
                    time_sync_ok = True
                    time_sync_resolution_source = "repair_verification_override"
                elif time_sync_verification_cleared:
                    time_sync_resolution_source = "micro_capture_false_verification_stale"
                else:
                    time_sync_resolution_source = "micro_capture"
                queue_imbalance = to_float(raw_row.get("queue_imbalance"))
                ofi_norm = to_float(raw_row.get("ofi_norm"))
                micro_alignment = to_float(raw_row.get("micro_alignment"))
                cvd_delta_ratio = to_float(raw_row.get("cvd_delta_ratio"))
                trade_count = int(raw_row.get("trade_count") or 0)
                evidence_score = to_float(raw_row.get("evidence_score"))

                shortline = load_shortline_policy(config_path)
                effective_row = dict(raw_row)
                effective_row["time_sync_ok"] = time_sync_ok
                micro = micro_signals(effective_row, shortline=shortline)
                micro_quality_ok = bool(micro.get("quality_ok", False))
                trust_ok = bool(micro.get("trust_ok", False))
                cvd_ready = bool(micro.get("cvd_ready", False))
                cvd_long = bool(micro.get("cvd_long", False))
                cvd_short = bool(micro.get("cvd_short", False))
                cvd_context_mode = text(micro.get("context")) or text(raw_row.get("cvd_context_mode"))
                cvd_veto_hint = text(micro.get("veto_hint")) or text(raw_row.get("cvd_veto_hint"))
                cvd_locality_status = text(micro.get("cvd_locality_status")) or text(
                    raw_row.get("cvd_locality_status")
                )
                cvd_attack_presence = text(micro.get("attack_presence")) or text(
                    raw_row.get("cvd_attack_presence")
                )
                cvd_attack_side = text(micro.get("attack_side"))

                if schema_ok and gap_ok and time_sync_ok:
                    snapshot_status = "live_orderflow_snapshot_ready"
                    snapshot_decision = "use_live_orderflow_snapshot_for_shortline_signal_quality"
                    blocker_title = "Use live orderflow snapshot for shortline signal quality review"
                    done_when = (
                        f"{symbol} loses live orderflow coverage or the latest micro_capture no longer passes schema/gap/time_sync"
                    )
                else:
                    snapshot_status = "live_orderflow_snapshot_degraded"
                    snapshot_decision = "repair_live_orderflow_snapshot_quality_then_recheck_signal_quality"
                    blocker_title = "Repair live orderflow snapshot quality before shortline signal quality review"
                    done_when = (
                        f"{symbol} regains schema_ok, gap_ok, and time_sync_ok in the latest micro_capture snapshot"
                    )

                blocker_detail = join_unique(
                    [
                        f"schema_ok={str(schema_ok).lower()}",
                        f"gap_ok={str(gap_ok).lower()}",
                        f"raw_time_sync_ok={str(raw_time_sync_ok).lower()}",
                        f"time_sync_ok={str(time_sync_ok).lower()}",
                        (
                            f"time_sync_resolution_source={time_sync_resolution_source}"
                            if time_sync_resolution_source
                            else ""
                        ),
                        (
                            f"time_sync_verification={time_sync_verification_brief}"
                            if time_sync_verification_brief
                            else ""
                        ),
                        f"micro_quality_ok={str(micro_quality_ok).lower()}",
                        f"trust_ok={str(trust_ok).lower()}",
                        f"cvd_ready={str(cvd_ready).lower()}",
                        f"cvd_context_mode={cvd_context_mode}",
                        f"cvd_veto_hint={cvd_veto_hint}" if cvd_veto_hint else "",
                        f"cvd_locality_status={cvd_locality_status}" if cvd_locality_status else "",
                        f"cvd_attack_presence={cvd_attack_presence}" if cvd_attack_presence else "",
                        f"cvd_attack_side={cvd_attack_side}" if cvd_attack_side else "",
                        f"queue_imbalance={queue_imbalance:g}",
                        f"ofi_norm={ofi_norm:g}",
                        f"micro_alignment={micro_alignment:g}",
                        f"cvd_delta_ratio={cvd_delta_ratio:g}",
                        f"trade_count={trade_count}",
                        f"evidence_score={evidence_score:g}",
                        micro_capture_status,
                        micro_capture_summary,
                    ]
                )

    snapshot_brief = ":".join([snapshot_status, symbol or "-", snapshot_decision, remote_market or "-"])
    payload = {
        "action": "build_crypto_shortline_live_orderflow_snapshot",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": symbol,
        "route_action": action,
        "remote_market": remote_market,
        "snapshot_status": snapshot_status,
        "snapshot_brief": snapshot_brief,
        "snapshot_decision": snapshot_decision,
        "signal_source": "crypto_shortline_live_orderflow_snapshot",
        "blocker_title": blocker_title,
        "blocker_target_artifact": "crypto_shortline_live_orderflow_snapshot",
        "blocker_detail": blocker_detail,
        "next_action": snapshot_decision,
        "next_action_target_artifact": "crypto_shortline_live_orderflow_snapshot",
        "done_when": done_when,
        "micro_capture_status": micro_capture_status,
        "micro_capture_summary": micro_capture_summary,
        "selected_micro_count": selected_micro_count,
        "symbol_present": symbol_present,
        "schema_ok": schema_ok,
        "gap_ok": gap_ok,
        "raw_time_sync_ok": raw_time_sync_ok,
        "time_sync_ok": time_sync_ok,
        "time_sync_resolution_source": time_sync_resolution_source,
        "time_sync_verification_brief": time_sync_verification_brief,
        "micro_quality_ok": micro_quality_ok,
        "trust_ok": trust_ok,
        "cvd_ready": cvd_ready,
        "cvd_long": cvd_long,
        "cvd_short": cvd_short,
        "cvd_context_mode": cvd_context_mode,
        "cvd_veto_hint": cvd_veto_hint,
        "cvd_locality_status": cvd_locality_status,
        "cvd_attack_presence": cvd_attack_presence,
        "cvd_attack_side": cvd_attack_side,
        "queue_imbalance": queue_imbalance,
        "ofi_norm": ofi_norm,
        "micro_alignment": micro_alignment,
        "cvd_delta_ratio": cvd_delta_ratio,
        "trade_count": trade_count,
        "evidence_score": evidence_score,
        "row": raw_row,
        "artifacts": {
            "remote_intent_queue": str(intent_path) if intent_path else "",
            "crypto_route_operator_brief": str(operator_path),
            "micro_capture": str(micro_path) if micro_path else "",
            "system_time_sync_repair_verification_report": str(time_sync_verification_path)
            if time_sync_verification_path
            else "",
        },
    }

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_live_orderflow_snapshot.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_live_orderflow_snapshot.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_live_orderflow_snapshot_checksum.json"
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
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
