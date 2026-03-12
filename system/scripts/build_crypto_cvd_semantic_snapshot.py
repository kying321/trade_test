#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
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
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
DEFAULT_MICRO_CAPTURE_DIR = DEFAULT_OUTPUT_ROOT / "artifacts" / "micro_capture"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_mapping(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def find_latest_micro_capture(artifact_dir: Path) -> Path | None:
    files = sorted(
        artifact_dir.glob("*_micro_capture.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def classify_symbol(row: dict[str, Any]) -> tuple[str, list[str]]:
    context = str(row.get("cvd_context_mode", "unclear")).strip() or "unclear"
    trust = str(row.get("cvd_trust_tier_hint", "unavailable")).strip() or "unavailable"
    veto = str(row.get("cvd_veto_hint", "")).strip()
    note = str(row.get("cvd_context_note", "")).strip()
    reasons: list[str] = []
    if not bool(row.get("time_sync_ok", True)):
        reasons.append("time_sync_risk")
    if trust in {"single_exchange_low", "unavailable"}:
        reasons.append("trust_low")
    if veto:
        reasons.append(veto)
    if "time_sync_risk" in note and "time_sync_risk" not in reasons:
        reasons.append("time_sync_risk")

    if reasons:
        return "watch_only", reasons
    if context == "continuation":
        return "trend_confirmation_watch", reasons
    if context in {"reversal", "absorption", "failed_auction"}:
        return "reversal_absorption_watch", reasons
    return "unclear", reasons


def build_symbol_rows(selected_micro: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    rows: list[dict[str, Any]] = []
    buckets: dict[str, list[str]] = {
        "trend_confirmation_watch": [],
        "reversal_absorption_watch": [],
        "watch_only": [],
        "unclear": [],
    }
    for row in selected_micro:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        classification, reasons = classify_symbol(row)
        buckets.setdefault(classification, []).append(symbol)
        rows.append(
            {
                "symbol": symbol,
                "classification": classification,
                "cvd_context_mode": str(row.get("cvd_context_mode", "unclear")),
                "cvd_trust_tier_hint": str(row.get("cvd_trust_tier_hint", "unavailable")),
                "cvd_veto_hint": str(row.get("cvd_veto_hint", "")),
                "time_sync_ok": bool(row.get("time_sync_ok", False)),
                "schema_ok": bool(row.get("schema_ok", False)),
                "sync_ok": bool(row.get("sync_ok", False)),
                "trade_count": int(row.get("trade_count", 0) or 0),
                "evidence_score": float(row.get("evidence_score", 0.0) or 0.0),
                "cvd_context_note": str(row.get("cvd_context_note", "")),
                "active_reasons": reasons,
            }
        )
    return rows, buckets


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Crypto CVD Semantic Snapshot",
        "",
        f"- source artifact: `{payload.get('source_artifact') or ''}`",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- status: `{payload.get('source_status') or ''}`",
        f"- pass: `{payload.get('source_pass')}`",
        f"- takeaway: {payload.get('takeaway') or ''}",
        "",
        "## Semantic Summary",
        f"- context counts: `{json.dumps(payload.get('context_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- trust counts: `{json.dumps(payload.get('trust_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- veto counts: `{json.dumps(payload.get('veto_hint_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Queue Classification",
        f"- trend-confirmation-watch: `{', '.join(payload.get('trend_confirmation_watch', [])) or '-'}`",
        f"- reversal-absorption-watch: `{', '.join(payload.get('reversal_absorption_watch', [])) or '-'}`",
        f"- watch-only: `{', '.join(payload.get('watch_only_symbols', [])) or '-'}`",
        f"- unclear: `{', '.join(payload.get('unclear_symbols', [])) or '-'}`",
        "",
        "## Symbols",
    ]
    for row in payload.get("symbols", []):
        if not isinstance(row, dict):
            continue
        lines.extend(
            [
                f"- `{row.get('symbol')}`",
                f"  - class: `{row.get('classification')}`",
                f"  - context: `{row.get('cvd_context_mode')}`",
                f"  - trust: `{row.get('cvd_trust_tier_hint')}`",
                f"  - veto: `{row.get('cvd_veto_hint') or '-'}`",
                f"  - time_sync_ok: `{row.get('time_sync_ok')}`",
                f"  - trade_count: `{row.get('trade_count')}`",
                f"  - evidence_score: `{row.get('evidence_score')}`",
                f"  - reasons: `{', '.join(row.get('active_reasons', [])) or '-'}`",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def prune_artifacts(
    review_dir: Path,
    *,
    current_artifact: Path,
    current_checksum: Path,
    current_markdown: Path,
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    review_dir.mkdir(parents=True, exist_ok=True)
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, ttl_hours))
    protected = {current_artifact.name, current_checksum.name, current_markdown.name}
    candidates: list[Path] = []
    for pattern in (
        "*_crypto_cvd_semantic_snapshot.json",
        "*_crypto_cvd_semantic_snapshot_checksum.json",
        "*_crypto_cvd_semantic_snapshot.md",
    ):
        candidates.extend(review_dir.glob(pattern))

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
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
    json_like = [p for p in survivors if p.suffix in {".json", ".md"}]
    for path in json_like[keep:]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a semantic snapshot from the latest micro capture artifact.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_MICRO_CAPTURE_DIR))
    parser.add_argument("--micro-capture-file", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=16)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    now_ts = dt.datetime.fromisoformat(args.now) if str(args.now).strip() else now_utc()

    source_path = (
        Path(args.micro_capture_file).expanduser().resolve()
        if str(args.micro_capture_file).strip()
        else find_latest_micro_capture(artifact_dir)
    )
    source_payload = load_json_mapping(source_path)

    status = "ok"
    ok = True
    selected_micro = []
    if source_payload is None:
        status = "micro_capture_missing"
        ok = False
    else:
        selected_micro = source_payload.get("selected_micro", [])
        if not isinstance(selected_micro, list) or not selected_micro:
            status = "selected_micro_missing"
            ok = False

    symbol_rows, buckets = build_symbol_rows(selected_micro if isinstance(selected_micro, list) else [])
    semantics = source_payload.get("cvd_semantics", {}) if isinstance(source_payload, dict) else {}
    context_counts = semantics.get("context_counts", {}) if isinstance(semantics, dict) else {}
    trust_counts = semantics.get("trust_counts", {}) if isinstance(semantics, dict) else {}
    veto_hint_counts = semantics.get("veto_hint_counts", {}) if isinstance(semantics, dict) else {}

    watch_only_symbols = buckets.get("watch_only", [])
    if not ok:
        takeaway = "No usable micro-capture snapshot was available."
    elif len(watch_only_symbols) == len(symbol_rows):
        takeaway = "All current CVD-lite observations are downgraded to watch-only; keep them as review filters until micro quality recovers."
    elif buckets.get("trend_confirmation_watch"):
        takeaway = "Use current CVD-lite as trend confirmation for the symbols listed under trend-confirmation-watch."
    elif buckets.get("reversal_absorption_watch"):
        takeaway = "Current CVD-lite is more useful as reversal/absorption watch than continuation confirmation."
    else:
        takeaway = "Current CVD-lite snapshot is mixed and should remain advisory."

    out: dict[str, Any] = {
        "action": "build_crypto_cvd_semantic_snapshot",
        "ok": ok,
        "status": status,
        "generated_at": fmt_utc(now_ts),
        "source_artifact": None if source_path is None else str(source_path),
        "as_of": source_payload.get("as_of") if isinstance(source_payload, dict) else None,
        "source_status": source_payload.get("status") if isinstance(source_payload, dict) else None,
        "source_pass": bool(source_payload.get("pass", False)) if isinstance(source_payload, dict) else False,
        "symbols_requested": int(source_payload.get("symbols_requested", 0) or 0) if isinstance(source_payload, dict) else 0,
        "symbols_selected": int(source_payload.get("symbols_selected", 0) or 0) if isinstance(source_payload, dict) else 0,
        "context_counts": context_counts if isinstance(context_counts, dict) else {},
        "trust_counts": trust_counts if isinstance(trust_counts, dict) else {},
        "veto_hint_counts": veto_hint_counts if isinstance(veto_hint_counts, dict) else {},
        "focus_stack_brief": "trend_confirmation -> reversal_absorption -> watch_only",
        "trend_confirmation_watch": buckets.get("trend_confirmation_watch", []),
        "reversal_absorption_watch": buckets.get("reversal_absorption_watch", []),
        "watch_only_symbols": watch_only_symbols,
        "unclear_symbols": buckets.get("unclear", []),
        "symbols": symbol_rows,
        "takeaway": takeaway,
        "artifact_status_label": "cvd-semantic-snapshot-ok" if ok else status,
        "artifact_label": None,
        "artifact_tags": [],
        "artifact": None,
        "markdown": None,
        "checksum": None,
        "pruned_keep": [],
        "pruned_age": [],
    }

    suffix = str(out.get("source_status") or status)
    out["artifact_label"] = f"crypto-cvd-semantic-snapshot:{suffix}"
    out["artifact_tags"] = ["crypto-cvd", "micro-capture", str(suffix)]

    review_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_ts.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_crypto_cvd_semantic_snapshot.json"
    markdown_path = review_dir / f"{stamp}_crypto_cvd_semantic_snapshot.md"
    checksum_path = review_dir / f"{stamp}_crypto_cvd_semantic_snapshot_checksum.json"
    out["artifact"] = str(artifact_path)
    out["markdown"] = str(markdown_path)
    out["checksum"] = str(checksum_path)

    artifact_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(out), encoding="utf-8")
    checksum_path.write_text(
        json.dumps(
            {
                "generated_at": fmt_utc(now_ts),
                "artifact_ttl_hours": max(1.0, float(args.artifact_ttl_hours)),
                "files": [
                    {
                        "path": str(artifact_path),
                        "sha256": sha256_file(artifact_path),
                        "size_bytes": int(artifact_path.stat().st_size),
                    },
                    {
                        "path": str(markdown_path),
                        "sha256": sha256_file(markdown_path),
                        "size_bytes": int(markdown_path.stat().st_size),
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_artifact=artifact_path,
        current_checksum=checksum_path,
        current_markdown=markdown_path,
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )
    out["pruned_keep"] = pruned_keep
    out["pruned_age"] = pruned_age
    artifact_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
