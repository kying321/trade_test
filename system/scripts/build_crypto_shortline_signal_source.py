#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


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


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


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
        "*_crypto_shortline_signal_source.json",
        "*_crypto_shortline_signal_source.md",
        "*_crypto_shortline_signal_source_checksum.json",
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


def normalize_symbols(raw: str) -> tuple[str, ...]:
    parts = [text(part).upper() for part in str(raw).split(",")]
    normalized = [part for part in parts if part]
    return tuple(normalized) if normalized else DEFAULT_SYMBOLS


def iter_template_candidates(review_dir: Path, output_root: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for pattern in ("*_crypto_shortline_signal_source.json", "*_strategy_recent5_signals.json"):
        for path in sorted(review_dir.glob(pattern), reverse=True):
            key = str(path)
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            candidates.append(path)
    for directory in (output_root / "daily", output_root / "state" / "output" / "daily"):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*_signals.json"), reverse=True):
            key = str(path)
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            candidates.append(path)
    return candidates


def load_template_rows(review_dir: Path, output_root: Path) -> dict[str, dict[str, Any]]:
    rows_by_symbol: dict[str, dict[str, Any]] = {}
    for path in iter_template_candidates(review_dir, output_root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("signals"), dict):
            for symbol, rows in payload.get("signals", {}).items():
                if text(symbol).upper() in rows_by_symbol:
                    continue
                if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                    row = dict(rows[0])
                    row["_template_path"] = str(path)
                    rows_by_symbol[text(symbol).upper()] = row
        elif isinstance(payload, list):
            for raw in payload:
                if not isinstance(raw, dict):
                    continue
                symbol = text(raw.get("symbol")).upper()
                if not symbol or symbol in rows_by_symbol:
                    continue
                row = dict(raw)
                row["_template_path"] = str(path)
                rows_by_symbol[symbol] = row
    return rows_by_symbol


def find_gate_row(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(payload.get("symbols")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def derive_side(gate_row: dict[str, Any], template_row: dict[str, Any]) -> str:
    micro = as_dict(gate_row.get("micro_signals"))
    attack_side = text(micro.get("attack_side")).lower()
    if attack_side == "buyers":
        return "LONG"
    if attack_side == "sellers":
        return "SHORT"
    alignment = safe_float(micro.get("micro_alignment"), 0.0)
    if alignment > 0:
        return "LONG"
    if alignment < 0:
        return "SHORT"
    template_side = text(template_row.get("side")).upper()
    return template_side if template_side in {"LONG", "SHORT"} else "LONG"


def derive_confidence(gate_row: dict[str, Any]) -> float:
    micro = as_dict(gate_row.get("micro_signals"))
    execution_state = text(gate_row.get("execution_state"))
    missing_gates = dedupe_text(as_list(gate_row.get("missing_gates")))
    alignment = abs(safe_float(micro.get("micro_alignment"), 0.0))
    evidence_score = safe_float(micro.get("evidence_score"), 0.0)
    trade_count = safe_float(micro.get("trade_count"), 0.0)
    confidence = max(12.0, min(55.0, alignment * 100.0))
    confidence += min(5.0, evidence_score * 2.0)
    if trade_count >= 500:
        confidence += 1.0
    if execution_state == "Setup_Ready" and not missing_gates:
        confidence = max(confidence, 70.0)
    return round(min(95.0, confidence), 6)


def derive_convexity(gate_row: dict[str, Any], template_row: dict[str, Any]) -> float:
    execution_state = text(gate_row.get("execution_state"))
    missing_gates = dedupe_text(as_list(gate_row.get("missing_gates")))
    if execution_state == "Setup_Ready" and not missing_gates:
        return round(max(3.2, safe_float(template_row.get("convexity_ratio"), 3.2)), 6)
    evidence_score = safe_float(as_dict(gate_row.get("micro_signals")).get("evidence_score"), 1.0)
    template_conv = safe_float(template_row.get("convexity_ratio"), 0.0)
    return round(min(2.5, max(1.0, evidence_score, min(template_conv, 2.5))), 6)


def reuse_price_template(template_row: dict[str, Any], side: str) -> tuple[float, float, float, str]:
    template_side = text(template_row.get("side")).upper()
    if template_side != side:
        return 0.0, 0.0, 0.0, ""
    entry_price = safe_float(template_row.get("entry_price"), 0.0)
    stop_price = safe_float(template_row.get("stop_price"), 0.0)
    target_price = safe_float(template_row.get("target_price"), 0.0)
    if entry_price <= 0.0 or stop_price <= 0.0 or target_price <= 0.0:
        return 0.0, 0.0, 0.0, ""
    return entry_price, stop_price, target_price, text(template_row.get("_template_path"))


def reuse_live_price_template(
    template_payload: dict[str, Any],
    *,
    symbol: str,
    side: str,
) -> tuple[float, float, float, str]:
    templates = as_dict(template_payload.get("templates"))
    for raw_symbol, raw_side_map in templates.items():
        if text(raw_symbol).upper() != symbol.upper():
            continue
        side_map = as_dict(raw_side_map)
        for raw_side, raw_template in side_map.items():
            if text(raw_side).upper() != side.upper():
                continue
            template = as_dict(raw_template)
            if not bool(template.get("template_ready", False)):
                return 0.0, 0.0, 0.0, ""
            entry_price = safe_float(template.get("entry_price"), 0.0)
            stop_price = safe_float(template.get("stop_price"), 0.0)
            target_price = safe_float(template.get("target_price"), 0.0)
            if entry_price <= 0.0 or stop_price <= 0.0 or target_price <= 0.0:
                return 0.0, 0.0, 0.0, ""
            return (
                entry_price,
                stop_price,
                target_price,
                text(template_payload.get("artifact")) or text(template_payload.get("price_reference_source")),
            )
    if text(template_payload.get("route_symbol")).upper() != symbol.upper():
        return 0.0, 0.0, 0.0, ""
    if text(template_payload.get("template_side")).upper() != side.upper():
        return 0.0, 0.0, 0.0, ""
    if not bool(template_payload.get("template_ready", False)):
        return 0.0, 0.0, 0.0, ""
    entry_price = safe_float(template_payload.get("entry_price"), 0.0)
    stop_price = safe_float(template_payload.get("stop_price"), 0.0)
    target_price = safe_float(template_payload.get("target_price"), 0.0)
    if entry_price <= 0.0 or stop_price <= 0.0 or target_price <= 0.0:
        return 0.0, 0.0, 0.0, ""
    return (
        entry_price,
        stop_price,
        target_price,
        text(template_payload.get("artifact")) or text(template_payload.get("price_reference_source")),
    )


def build_signal_row(
    *,
    symbol: str,
    gate_row: dict[str, Any],
    template_row: dict[str, Any],
    live_template_payload: dict[str, Any],
    as_of: dt.date,
) -> dict[str, Any]:
    side = derive_side(gate_row, template_row)
    entry_price, stop_price, target_price, template_path = reuse_live_price_template(
        live_template_payload, symbol=symbol, side=side
    )
    execution_price_ready = bool(template_path)
    if template_path:
        price_reference_kind = "shortline_live_bars_template"
    else:
        (
            entry_price,
            stop_price,
            target_price,
            template_path,
        ) = reuse_price_template(template_row, side)
        price_reference_kind = (
            "shortline_template_proxy" if template_path else "shortline_missing_price_template"
        )
    execution_state = text(gate_row.get("execution_state")) or "Bias_Only"
    route_action = text(gate_row.get("route_action")) or "-"
    route_state = text(gate_row.get("route_state")) or "-"
    missing_gates = dedupe_text(as_list(gate_row.get("missing_gates")))
    micro = as_dict(gate_row.get("micro_signals"))
    notes = " | ".join(
        part
        for part in [
            f"generated_from=crypto_shortline_execution_gate:{execution_state}",
            f"route={route_state}:{route_action}",
            text(gate_row.get("blocker_detail")),
            (
                f"micro_context={text(micro.get('context'))};"
                f"attack_side={text(micro.get('attack_side'))};"
                f"veto_hint={text(micro.get('veto_hint'))}"
            ),
            (
                f"price_template={Path(template_path).name}"
                if template_path
                else "price_template=missing_or_side_mismatch"
            ),
        ]
        if part
    )
    factor_flags = dedupe_text(
        [
            "generated_from_crypto_shortline_execution_gate",
            f"execution_state:{execution_state}",
            f"route_state:{route_state}",
            f"route_action:{route_action}",
            (
                "price_reference:live_bars_template"
                if execution_price_ready
                else "price_reference:proxy_only"
            ),
            *(f"missing_gate:{item}" for item in missing_gates),
        ]
    )
    price_reference_source = template_path or "crypto_shortline_execution_gate"
    return {
        "date": as_of.isoformat(),
        "symbol": symbol,
        "side": side,
        "regime": "短线观察",
        "execution_state": execution_state,
        "route_action": route_action,
        "route_state": route_state,
        "missing_gates": missing_gates,
        "pattern_hint_brief": text(gate_row.get("pattern_hint_brief")),
        "confidence": derive_confidence(gate_row),
        "convexity_ratio": derive_convexity(gate_row, template_row),
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "factor_flags": factor_flags,
        "notes": notes,
        "combo_id": "crypto_shortline_execution_gate",
        "combo_id_canonical": "crypto_shortline_execution_gate",
        "confirmation_indicator": "crypto_shortline_gate_proxy",
        "price_reference_kind": price_reference_kind,
        "price_reference_source": price_reference_source,
        "execution_price_ready": execution_price_ready,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Crypto Shortline Signal Source",
        "",
        f"- generated_at_utc: `{payload.get('generated_at_utc') or ''}`",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_execution_gate_artifact: `{payload.get('source_execution_gate_artifact') or ''}`",
        f"- source_route_brief_artifact: `{payload.get('source_route_brief_artifact') or ''}`",
        "",
        "## Signals",
    ]
    signals = payload.get("signals", {}) if isinstance(payload.get("signals"), dict) else {}
    for symbol, rows in signals.items():
        row = rows[0] if isinstance(rows, list) and rows else {}
        lines.append(
            f"- `{symbol}` side=`{row.get('side') or ''}` conf=`{row.get('confidence')}` conv=`{row.get('convexity_ratio')}` exec_ready=`{row.get('execution_price_ready')}` price_ref=`{row.get('price_reference_source') or ''}`"
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fresh crypto shortline signal source artifact.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=5)
    args = parser.parse_args()

    review_dir = Path(args.review_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    runtime_now = parse_now(args.now)
    as_of = runtime_now.date()

    execution_gate_path = find_latest(review_dir, "*_crypto_shortline_execution_gate.json")
    route_brief_path = find_latest(review_dir, "*_crypto_route_operator_brief.json")
    cvd_queue_path = find_latest(review_dir, "*_crypto_cvd_queue_handoff.json")
    price_template_snapshot_path = find_latest(
        review_dir, "*_crypto_shortline_price_template_snapshot.json"
    )
    if execution_gate_path is None or route_brief_path is None:
        raise SystemExit("missing_crypto_shortline_source_inputs")

    execution_gate_payload = load_json_mapping(execution_gate_path)
    route_brief_payload = load_json_mapping(route_brief_path)
    cvd_queue_payload = load_json_mapping(cvd_queue_path) if cvd_queue_path and cvd_queue_path.exists() else {}
    price_template_snapshot_payload = (
        load_json_mapping(price_template_snapshot_path)
        if price_template_snapshot_path and price_template_snapshot_path.exists()
        else {}
    )
    template_rows = load_template_rows(review_dir, output_root)
    symbols = normalize_symbols(args.symbols)

    signals: dict[str, list[dict[str, Any]]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        gate_row = find_gate_row(execution_gate_payload, symbol)
        template_row = template_rows.get(symbol, {})
        signal_row = build_signal_row(
            symbol=symbol,
            gate_row=gate_row,
            template_row=template_row,
            live_template_payload=price_template_snapshot_payload,
            as_of=as_of,
        )
        signals[symbol] = [signal_row]
        diagnostics[symbol] = {
            "execution_state": text(gate_row.get("execution_state")),
            "route_action": text(gate_row.get("route_action")),
            "route_state": text(gate_row.get("route_state")),
            "missing_gates": dedupe_text(as_list(gate_row.get("missing_gates"))),
            "template_source": text(template_row.get("_template_path")),
            "live_template_source": text(price_template_snapshot_payload.get("artifact")),
            "price_reference_kind": text(signal_row.get("price_reference_kind")),
            "execution_price_ready": bool(signal_row.get("execution_price_ready", False)),
        }

    payload = {
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(runtime_now),
        "as_of": as_of.isoformat(),
        "source_execution_gate_artifact": str(execution_gate_path),
        "source_route_brief_artifact": str(route_brief_path),
        "source_cvd_queue_artifact": str(cvd_queue_path) if cvd_queue_path else "",
        "source_price_template_snapshot_artifact": str(price_template_snapshot_path)
        if price_template_snapshot_path
        else "",
        "source_kind": "crypto_shortline_signal_source",
        "selection_reason": "current_shortline_gate_projection",
        "symbols": list(symbols),
        "signals": signals,
        "symbol_diagnostics": diagnostics,
        "summary": {
            "signal_row_count": len(symbols),
            "price_template_reused_count": sum(
                1
                for rows in signals.values()
                for row in rows
                if text(row.get("price_reference_kind")) == "shortline_template_proxy"
            ),
            "live_price_template_count": sum(
                1
                for rows in signals.values()
                for row in rows
                if text(row.get("price_reference_kind")) == "shortline_live_bars_template"
            ),
            "proxy_price_only_count": sum(
                1
                for rows in signals.values()
                for row in rows
                if not bool(row.get("execution_price_ready", True))
            ),
        },
        "route_focus_symbol": text(route_brief_payload.get("review_priority_head_symbol"))
        or text(route_brief_payload.get("next_focus_symbol")),
        "route_focus_action": text(route_brief_payload.get("next_focus_action")),
        "queue_status": text(cvd_queue_payload.get("queue_status")),
        "queue_next_focus_action": text(cvd_queue_payload.get("next_focus_action")),
    }

    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_crypto_shortline_signal_source.json"
    md_path = review_dir / f"{stamp}_crypto_shortline_signal_source.md"
    checksum_path = review_dir / f"{stamp}_crypto_shortline_signal_source_checksum.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_paths=[json_path, md_path, checksum_path],
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )
    payload["governance"] = {
        "artifact_keep": max(1, int(args.artifact_keep)),
        "artifact_ttl_hours": max(1.0, float(args.artifact_ttl_hours)),
        "pruned_keep": pruned_keep,
        "pruned_age": pruned_age,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_payload = {
        "generated_at_utc": fmt_utc(runtime_now),
        "files": [
            {"path": str(json_path), "sha256": sha256_file(json_path)},
            {"path": str(md_path), "sha256": sha256_file(md_path)},
        ],
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(json_path),
                "status": "ok",
                "brief": f"crypto_shortline_signal_source:{payload.get('route_focus_symbol') or '-'}:{payload.get('route_focus_action') or '-'}",
                "as_of": payload.get("as_of"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
