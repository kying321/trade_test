#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sqlite3
from typing import Any

from lie_engine.engine import LieEngine
from lie_engine.research.real_data import load_real_data_bundle


def resolve_system_root(workspace: Path) -> Path:
    if (workspace / "system").exists():
        return workspace / "system"
    if workspace.name == "system":
        return workspace
    raise FileNotFoundError(f"cannot resolve system root from {workspace}")


def parse_date(raw: str) -> date:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("date_is_required")
    return date.fromisoformat(text)


def unique_symbols(*groups: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for raw in group:
            sym = str(raw or "").strip()
            if not sym or sym in seen:
                continue
            seen.add(sym)
            out.append(sym)
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cleanup_legacy_macro_sources(sqlite_path: Path) -> dict[str, Any]:
    legacy_sources = [
        "binance.macro_proxy",
        "binance_spot_public",
        "bybit_spot_public",
        "open_source_primary",
        "open_source_secondary",
    ]
    if not sqlite_path.exists():
        return {"rows_removed": 0, "removed_sources": []}
    conn = sqlite3.connect(sqlite_path)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='macro' LIMIT 1"
        ).fetchone()
        if not exists:
            return {"rows_removed": 0, "removed_sources": []}
        placeholders = ",".join("?" for _ in legacy_sources)
        rows = conn.execute(
            f"SELECT source, COUNT(*) FROM macro WHERE source IN ({placeholders}) GROUP BY source",
            legacy_sources,
        ).fetchall()
        removed_sources = sorted(str(row[0]) for row in rows if row and row[0])
        rows_removed = int(sum(int(row[1]) for row in rows))
        if rows_removed > 0:
            conn.execute(f"DELETE FROM macro WHERE source IN ({placeholders})", legacy_sources)
            conn.commit()
        return {"rows_removed": rows_removed, "removed_sources": removed_sources}
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Fenlie public internet data across SQLite/artifacts/research_cache.")
    parser.add_argument("--workspace", default=".", help="Fenlie workspace root or system directory.")
    parser.add_argument("--date", required=True, help="Refresh date in YYYY-MM-DD.")
    parser.add_argument("--lookback-days", type=int, default=420, help="Trailing lookback for research cache bars.")
    parser.add_argument("--workers", type=int, default=6, help="Workers for research cache refresh.")
    parser.add_argument("--report-symbol-cap", type=int, default=5, help="Equity report/news symbol cap for research bundle.")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    system_root = resolve_system_root(workspace)
    as_of = parse_date(args.date)

    eng = LieEngine(config_path=system_root / "config.yaml")
    requested_symbols = eng._core_symbols()
    persist_symbols = eng._persisted_ingestion_symbols(requested_symbols)
    bars, ingest = eng._run_ingestion(as_of, requested_symbols)

    aux_symbols = eng._persisted_aux_symbols()
    research_symbols = unique_symbols(requested_symbols, aux_symbols)
    cache_dir = eng.ctx.output_dir / "artifacts" / "research_cache"
    bundle = load_real_data_bundle(
        core_symbols=research_symbols,
        start=as_of - timedelta(days=max(1, int(args.lookback_days))),
        end=as_of,
        max_symbols=max(1, len(research_symbols)),
        report_symbol_cap=max(1, int(args.report_symbol_cap)),
        workers=max(1, int(args.workers)),
        cache_dir=cache_dir,
        cache_ttl_hours=0.0,
        strict_cutoff=as_of,
        review_days=0,
        include_post_review=False,
    )
    macro_cleanup = cleanup_legacy_macro_sources(eng.ctx.sqlite_path)

    feature_path = eng.ctx.output_dir / "artifacts" / "feature" / f"{as_of.isoformat()}_bars_feature.parquet"
    artifact_path = eng.ctx.output_dir / "artifacts" / "public_data_refresh" / f"{as_of.isoformat()}_public_data_refresh.json"
    payload = {
        "ok": True,
        "mode": "public_data_refresh",
        "workspace": str(workspace),
        "system_root": str(system_root),
        "as_of": as_of.isoformat(),
        "requested_symbols": requested_symbols,
        "persist_symbols": persist_symbols,
        "research_cache_universe": list(getattr(bundle, "universe", []) or []),
        "sqlite_path": str(eng.ctx.sqlite_path),
        "feature_path": str(feature_path),
        "feature_exists": feature_path.exists(),
        "research_cache_meta": str((getattr(bundle, "fetch_stats", {}) or {}).get("cache_path", "")),
        "research_cache_hit": bool((getattr(bundle, "fetch_stats", {}) or {}).get("cache_hit", False)),
        "requested_model_rows": int(len(bars)),
        "persisted_normalized_rows": int(len(getattr(ingest, "normalized_bars", []))),
        "persisted_macro_rows": int(len(getattr(ingest, "macro", []))),
        "persisted_news_rows": int(len(getattr(ingest, "news", []))),
        "persisted_sentiment_keys": int(len(getattr(ingest, "sentiment", {}) or {})),
        "macro_cleanup": macro_cleanup,
        "research_cache_bars_rows": int(len(getattr(bundle, "bars", []))),
        "research_cache_news_records": int(getattr(bundle, "news_records", 0) or 0),
        "research_cache_report_records": int(getattr(bundle, "report_records", 0) or 0),
    }
    write_json(artifact_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
