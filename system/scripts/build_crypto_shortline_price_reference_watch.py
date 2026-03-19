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


def select_ticket_surface_path(review_dir: Path, route_symbol: str) -> Path | None:
    candidates = sorted(
        review_dir.glob("*_signal_to_order_tickets.json"),
        key=lambda item: (artifact_stamp(item), item.stat().st_mtime, item.name),
        reverse=True,
    )
    if not candidates:
        return None
    normalized_route = text(route_symbol).upper()
    for path in candidates:
        try:
            payload = load_json_mapping(path)
        except Exception:
            continue
        for row in as_list(payload.get("tickets")):
            item = as_dict(row)
            if text(item.get("symbol")).upper() == normalized_route:
                return path
    return candidates[0]


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
        "*_crypto_shortline_price_reference_watch.json",
        "*_crypto_shortline_price_reference_watch.md",
        "*_crypto_shortline_price_reference_watch_checksum.json",
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


def find_route_row(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(payload.get("tickets")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def find_signal_row(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    rows = as_list(as_dict(payload.get("signals")).get(symbol))
    for row in rows:
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return as_dict(rows[0]) if rows else {}


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Shortline Price Reference Watch",
            "",
            f"- brief: `{text(payload.get('watch_brief'))}`",
            f"- decision: `{text(payload.get('watch_decision'))}`",
            f"- price_reference_kind: `{text(payload.get('price_reference_kind'))}`",
            f"- price_reference_source: `{text(payload.get('price_reference_source'))}`",
            f"- execution_price_ready: `{payload.get('execution_price_ready')}`",
            f"- missing_level_fields: `{','.join(as_list(payload.get('missing_level_fields'))) or '-'}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-owned crypto shortline price-reference watch artifact."
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

    intent_path = find_latest(review_dir, "*_remote_intent_queue.json")
    operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json")
    signal_source_path = find_latest(review_dir, "*_crypto_shortline_signal_source.json")
    diagnosis_path = find_latest(review_dir, "*_crypto_shortline_ticket_constraint_diagnosis.json")
    if operator_path is None:
        raise SystemExit("missing_required_artifacts:crypto_route_operator_brief")

    intent_payload = (
        load_json_mapping(intent_path) if intent_path is not None and intent_path.exists() else {}
    )
    operator_payload = load_json_mapping(operator_path)
    symbol = route_symbol(intent_payload, operator_payload)
    tickets_path = select_ticket_surface_path(review_dir, symbol)
    if tickets_path is None:
        raise SystemExit("missing_required_artifacts:signal_to_order_tickets")

    tickets_payload = load_json_mapping(tickets_path)
    signal_source_payload = (
        load_json_mapping(signal_source_path)
        if signal_source_path is not None and signal_source_path.exists()
        else {}
    )
    diagnosis_payload = (
        load_json_mapping(diagnosis_path)
        if diagnosis_path is not None and diagnosis_path.exists()
        else {}
    )

    action = route_action(intent_payload, operator_payload)
    remote_market = text(intent_payload.get("remote_market")) or "portfolio_margin_um"
    route_row = find_route_row(tickets_payload, symbol)
    route_signal = as_dict(route_row.get("signal"))
    signal_row = find_signal_row(signal_source_payload, symbol) or route_signal
    route_row_reasons = dedupe_text(as_list(route_row.get("reasons")))

    levels = as_dict(route_row.get("levels"))
    entry_price = float(levels.get("entry_price") or signal_row.get("entry_price") or 0.0)
    stop_price = float(levels.get("stop_price") or signal_row.get("stop_price") or 0.0)
    target_price = float(levels.get("target_price") or signal_row.get("target_price") or 0.0)
    price_reference_kind = text(signal_row.get("price_reference_kind")) or text(
        route_signal.get("price_reference_kind")
    )
    price_reference_source = text(signal_row.get("price_reference_source")) or text(
        route_signal.get("price_reference_source")
    )
    execution_price_ready = bool(
        signal_row.get("execution_price_ready", route_signal.get("execution_price_ready", False))
    )
    missing_level_fields = [
        name
        for name, value in (
            ("entry_price", entry_price),
            ("stop_price", stop_price),
            ("target_price", target_price),
        )
        if float(value or 0.0) <= 0.0
    ]
    price_reference_blocked = "proxy_price_reference_only" in route_row_reasons or (
        not execution_price_ready or bool(missing_level_fields)
    )
    diagnosis_brief = text(diagnosis_payload.get("diagnosis_brief"))
    diagnosis_decision = text(diagnosis_payload.get("diagnosis_decision"))
    price_template_missing = price_reference_kind == "shortline_missing_price_template"
    price_template_proxy = price_reference_kind == "shortline_template_proxy"

    if not price_reference_blocked:
        watch_status = "price_reference_ready"
        watch_decision = "recheck_shortline_execution_gate_after_price_reference_ready"
        blocker_title = "Executable price reference ready for shortline setup promotion"
    elif price_template_missing:
        watch_status = "price_reference_missing_template_proxy_only"
        watch_decision = "build_price_template_then_recheck_execution_gate"
        blocker_title = "Build executable price reference before shortline setup promotion"
    elif price_template_proxy:
        watch_status = "price_reference_template_proxy_only"
        watch_decision = "replace_proxy_price_reference_then_recheck_execution_gate"
        blocker_title = "Replace proxy price reference before shortline setup promotion"
    else:
        watch_status = "price_reference_not_executable"
        watch_decision = "repair_price_reference_then_recheck_execution_gate"
        blocker_title = "Repair executable price reference before shortline setup promotion"

    watch_brief = ":".join([watch_status, symbol or "-", watch_decision, remote_market or "-"])
    done_when = (
        f"{symbol} keeps execution_price_ready=true, entry/stop/target stay non-zero, "
        "and the route ticket drops proxy_price_reference_only"
    )
    blocker_detail = join_unique(
        [
            f"price_reference_kind={price_reference_kind or '-'}",
            f"price_reference_source={price_reference_source or '-'}",
            f"execution_price_ready={str(execution_price_ready).lower()}",
            (
                f"missing_level_fields={','.join(missing_level_fields)}"
                if missing_level_fields
                else "missing_level_fields=-"
            ),
            f"entry_price={entry_price:g}",
            f"stop_price={stop_price:g}",
            f"target_price={target_price:g}",
            (
                f"ticket_row_reasons={','.join(route_row_reasons)}"
                if route_row_reasons
                else ""
            ),
            diagnosis_brief,
            diagnosis_decision,
        ]
    )

    payload = {
        "action": "build_crypto_shortline_price_reference_watch",
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
        "blocker_target_artifact": "crypto_shortline_price_reference_watch",
        "blocker_detail": blocker_detail,
        "next_action": watch_decision,
        "next_action_target_artifact": "crypto_shortline_price_reference_watch",
        "done_when": done_when,
        "price_reference_blocked": price_reference_blocked,
        "price_reference_kind": price_reference_kind,
        "price_reference_source": price_reference_source,
        "execution_price_ready": execution_price_ready,
        "missing_level_fields": missing_level_fields,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "ticket_row_reasons": route_row_reasons,
        "artifacts": {
            "remote_intent_queue": str(intent_path) if intent_path else "",
            "crypto_route_operator_brief": str(operator_path),
            "signal_to_order_tickets": str(tickets_path),
            "crypto_shortline_signal_source": str(signal_source_path) if signal_source_path else "",
            "crypto_shortline_ticket_constraint_diagnosis": str(diagnosis_path)
            if diagnosis_path
            else "",
        },
    }

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_price_reference_watch.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_price_reference_watch.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_price_reference_watch_checksum.json"
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
