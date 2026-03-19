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


def find_latest_scoped(review_dir: Path, pattern: str, symbol: str) -> Path | None:
    files = sorted(
        review_dir.glob(pattern),
        key=lambda item: (artifact_stamp(item), item.stat().st_mtime, item.name),
        reverse=True,
    )
    for path in files:
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
        "*_crypto_shortline_liquidity_sweep_watch.json",
        "*_crypto_shortline_liquidity_sweep_watch.md",
        "*_crypto_shortline_liquidity_sweep_watch_checksum.json",
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
            "# Crypto Shortline Liquidity Sweep Watch",
            "",
            f"- brief: `{text(payload.get('watch_brief'))}`",
            f"- decision: `{text(payload.get('watch_decision'))}`",
            f"- liquidity_sweep_missing: `{payload.get('liquidity_sweep_missing')}`",
            f"- liquidity_sweep_long: `{payload.get('liquidity_sweep_long')}`",
            f"- liquidity_sweep_short: `{payload.get('liquidity_sweep_short')}`",
            f"- mss_long: `{payload.get('mss_long')}`",
            f"- mss_short: `{payload.get('mss_short')}`",
            f"- price_reference_blocked: `{payload.get('price_reference_blocked')}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-owned crypto shortline liquidity-sweep watch artifact."
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

    intent_path = find_latest(review_dir, "*_remote_intent_queue.json")
    operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json")
    gate_path = find_latest(review_dir, "*_crypto_shortline_execution_gate.json")
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
    route_focus_symbol = (
        text(intent_payload.get("preferred_route_symbol"))
        or text(operator_payload.get("review_priority_head_symbol"))
        or text(operator_payload.get("next_focus_symbol"))
    ).upper()
    route_symbol = text(args.symbol).upper() or route_focus_symbol
    route_action = text(intent_payload.get("preferred_route_action")) or text(
        operator_payload.get("next_focus_action")
    )
    remote_market = text(intent_payload.get("remote_market")) or "portfolio_margin_um"
    material_change_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_material_change_trigger.json", route_symbol
    )
    diagnosis_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_ticket_constraint_diagnosis.json", route_symbol
    )
    liquidity_event_trigger_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_liquidity_event_trigger.json", route_symbol
    )
    price_reference_watch_path = find_latest_scoped(
        review_dir, "*_crypto_shortline_price_reference_watch.json", route_symbol
    )
    material_change_payload = (
        load_json_mapping(material_change_path)
        if material_change_path is not None and material_change_path.exists()
        else {}
    )
    diagnosis_payload = (
        load_json_mapping(diagnosis_path)
        if diagnosis_path is not None and diagnosis_path.exists()
        else {}
    )
    liquidity_event_trigger_payload = (
        load_json_mapping(liquidity_event_trigger_path)
        if liquidity_event_trigger_path is not None and liquidity_event_trigger_path.exists()
        else {}
    )
    price_reference_watch_payload = (
        load_json_mapping(price_reference_watch_path)
        if price_reference_watch_path is not None and price_reference_watch_path.exists()
        else {}
    )

    gate_row = find_gate_row(gate_payload, route_symbol)
    structure_signals = as_dict(gate_row.get("structure_signals"))
    missing_gates = dedupe_text(as_list(gate_row.get("missing_gates")))
    raw_ticket_row_reasons = dedupe_text(as_list(diagnosis_payload.get("ticket_row_reasons")))
    execution_state = text(gate_row.get("execution_state")) or "Bias_Only"
    route_state = text(gate_row.get("route_state"))
    liquidity_sweep_long = bool(structure_signals.get("sweep_long", False))
    liquidity_sweep_short = bool(structure_signals.get("sweep_short", False))
    mss_long = bool(structure_signals.get("mss_long", False))
    mss_short = bool(structure_signals.get("mss_short", False))
    liquidity_sweep_missing = "liquidity_sweep" in missing_gates or not (
        liquidity_sweep_long or liquidity_sweep_short
    )
    blocker_target_artifact = "crypto_shortline_liquidity_sweep_watch"
    next_action_target_artifact = "crypto_shortline_liquidity_sweep_watch"
    liquidity_event_trigger_status = text(liquidity_event_trigger_payload.get("trigger_status"))
    liquidity_event_trigger_decision = text(liquidity_event_trigger_payload.get("trigger_decision"))
    price_reference_watch_status = text(price_reference_watch_payload.get("watch_status"))
    price_reference_watch_decision = text(price_reference_watch_payload.get("watch_decision"))
    if price_reference_watch_payload:
        if "price_reference_blocked" in price_reference_watch_payload:
            price_reference_blocked = bool(
                price_reference_watch_payload.get("price_reference_blocked", False)
            )
        else:
            price_reference_blocked = price_reference_watch_status not in {"", "price_reference_ready"}
    else:
        price_reference_blocked = "proxy_price_reference_only" in raw_ticket_row_reasons
    ticket_row_reasons = list(raw_ticket_row_reasons)
    if not price_reference_blocked:
        ticket_row_reasons = [
            reason for reason in ticket_row_reasons if reason != "proxy_price_reference_only"
        ]
    if execution_state == "Setup_Ready" and not liquidity_sweep_missing:
        watch_status = "liquidity_sweep_cleared_setup_ready"
        watch_decision = "review_guarded_canary_promotion"
        blocker_title = "Liquidity sweep cleared for guarded canary review"
        done_when = (
            f"{route_symbol} loses setup quality, loses executable price reference, or leaves Setup_Ready"
        )
        if price_reference_blocked and price_reference_watch_status:
            watch_status = price_reference_watch_status
            watch_decision = (
                price_reference_watch_decision
                or "repair_price_reference_then_recheck_execution_gate"
            )
            blocker_title = text(price_reference_watch_payload.get("blocker_title")) or (
                "Build executable price reference before shortline setup promotion"
            )
            blocker_target_artifact = (
                text(price_reference_watch_payload.get("blocker_target_artifact"))
                or "crypto_shortline_price_reference_watch"
            )
            next_action_target_artifact = (
                text(price_reference_watch_payload.get("next_action_target_artifact"))
                or blocker_target_artifact
            )
            done_when = text(price_reference_watch_payload.get("done_when")) or done_when
    elif liquidity_sweep_missing:
        watch_status = "liquidity_sweep_waiting"
        watch_decision = "wait_for_liquidity_sweep_then_recheck_execution_gate"
        blocker_title = "Track liquidity sweep before shortline setup promotion"
        done_when = (
            f"{route_symbol} records a liquidity sweep with executable price reference, "
            "then the shortline execution gate refresh confirms the next stage"
        )
        if liquidity_event_trigger_status:
            watch_status = liquidity_event_trigger_status
            watch_decision = liquidity_event_trigger_decision or watch_decision
            blocker_title = (
                text(liquidity_event_trigger_payload.get("blocker_title")) or blocker_title
            )
            blocker_target_artifact = (
                text(liquidity_event_trigger_payload.get("blocker_target_artifact"))
                or "crypto_shortline_liquidity_event_trigger"
            )
            next_action_target_artifact = (
                text(liquidity_event_trigger_payload.get("next_action_target_artifact"))
                or blocker_target_artifact
            )
            done_when = text(liquidity_event_trigger_payload.get("done_when")) or done_when
        if price_reference_blocked:
            watch_status = f"{watch_status}_proxy_price_blocked"
    else:
        watch_status = "liquidity_sweep_cleared_wait_next_stage"
        watch_decision = "refresh_shortline_execution_gate_for_next_stage"
        blocker_title = "Liquidity sweep cleared; wait for the next shortline stage"
        done_when = (
            f"{route_symbol} clears the next shortline stage and keeps an executable price reference"
        )
        if price_reference_blocked:
            watch_status = f"{watch_status}_proxy_price_blocked"

    watch_brief = ":".join([watch_status, route_symbol or "-", watch_decision, remote_market or "-"])
    blocker_detail = join_unique(
        [
            f"execution_state={execution_state}",
            f"route_state={route_state or '-'}",
            f"liquidity_sweep_missing={str(liquidity_sweep_missing).lower()}",
            f"liquidity_sweep_long={str(liquidity_sweep_long).lower()}",
            f"liquidity_sweep_short={str(liquidity_sweep_short).lower()}",
            f"mss_long={str(mss_long).lower()}",
            f"mss_short={str(mss_short).lower()}",
            f"price_reference_blocked={str(price_reference_blocked).lower()}",
            (
                f"ticket_row_reasons={','.join(ticket_row_reasons)}"
                if ticket_row_reasons
                else ""
            ),
            text(gate_row.get("blocker_detail")),
            (
                f"material_change_trigger={text(material_change_payload.get('trigger_brief'))}:"
                f"{text(material_change_payload.get('trigger_decision'))}"
                if material_change_payload
                else ""
            ),
            (
                f"liquidity_event_trigger={text(liquidity_event_trigger_payload.get('trigger_brief'))}:"
                f"{text(liquidity_event_trigger_payload.get('trigger_decision'))}"
                if liquidity_event_trigger_payload
                else ""
            ),
        ]
    )

    payload = {
        "action": "build_crypto_shortline_liquidity_sweep_watch",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": route_symbol,
        "route_focus_symbol": route_focus_symbol,
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
        "shortline_execution_state": execution_state,
        "shortline_route_state": route_state,
        "missing_gates": missing_gates,
        "liquidity_sweep_missing": liquidity_sweep_missing,
        "liquidity_sweep_long": liquidity_sweep_long,
        "liquidity_sweep_short": liquidity_sweep_short,
        "mss_long": mss_long,
        "mss_short": mss_short,
        "price_reference_blocked": price_reference_blocked,
        "price_reference_watch_status": price_reference_watch_status,
        "price_reference_watch_decision": price_reference_watch_decision,
        "ticket_row_reasons": ticket_row_reasons,
        "material_change_trigger_status": text(material_change_payload.get("trigger_status")),
        "material_change_trigger_decision": text(material_change_payload.get("trigger_decision")),
        "liquidity_event_trigger_status": liquidity_event_trigger_status,
        "liquidity_event_trigger_decision": liquidity_event_trigger_decision,
        "artifacts": {
            "remote_intent_queue": str(intent_path) if intent_path else "",
            "crypto_route_operator_brief": str(operator_path),
            "crypto_shortline_execution_gate": str(gate_path),
            "crypto_shortline_material_change_trigger": str(material_change_path)
            if material_change_path
            else "",
            "crypto_shortline_liquidity_event_trigger": str(liquidity_event_trigger_path)
            if liquidity_event_trigger_path
            else "",
            "crypto_shortline_ticket_constraint_diagnosis": str(diagnosis_path)
            if diagnosis_path
            else "",
            "crypto_shortline_price_reference_watch": str(price_reference_watch_path)
            if price_reference_watch_path
            else "",
        },
    }

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_liquidity_sweep_watch.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_liquidity_sweep_watch.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_liquidity_sweep_watch_checksum.json"
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
