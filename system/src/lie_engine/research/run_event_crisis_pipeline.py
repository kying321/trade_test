from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Mapping, Sequence

from lie_engine.research.event_crisis_pipeline import (
    build_event_asset_shock_map,
    build_event_crisis_analogy,
    build_event_regime_snapshot,
)
from lie_engine.research.event_crisis_sources import (
    DEFAULT_PRIORITY_ASSETS,
    normalize_market_inputs,
    normalize_public_event_rows,
)

def _parse_now(raw: str | None) -> dt.datetime:
    if not raw:
        return dt.datetime.now(dt.timezone.utc)
    text = raw.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)

def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def run_pipeline(
    *,
    output_root: Path,
    mode: str,
    event_rows: Sequence[Mapping[str, object]] | None = None,
    market_inputs: Mapping[str, float] | None = None,
    generated_at: dt.datetime | None = None,
) -> Mapping[str, Path]:
    now = generated_at or dt.datetime.now(dt.timezone.utc)
    normalized_events = normalize_public_event_rows(event_rows or [], default_ts=now)
    normalized_markets = normalize_market_inputs(market_inputs)

    intake_payload = {
        "generated_at_utc": now.isoformat().replace("+00:00", "Z"),
        "mode": mode,
        "event_count": len(normalized_events),
        "events": normalized_events,
    }

    regime_snapshot = build_event_regime_snapshot(
        event_rows=normalized_events, market_inputs=normalized_markets
    )
    analogy_payload = build_event_crisis_analogy(
        event_rows=normalized_events, market_inputs=normalized_markets
    )
    asset_shock_map = build_event_asset_shock_map(
        event_rows=normalized_events, market_inputs=normalized_markets
    )
    operator_summary = {
        "generated_at_utc": intake_payload["generated_at_utc"],
        "mode": mode,
        "regime_state": regime_snapshot.get("regime_state"),
        "event_severity_score": regime_snapshot.get("event_severity_score"),
        "systemic_risk_score": regime_snapshot.get("systemic_risk_score"),
        "top_analogues": analogy_payload.get("top_analogues"),
        "priority_assets": DEFAULT_PRIORITY_ASSETS,
    }

    review_dir = output_root / "review"
    intake_path = review_dir / "latest_event_intake.json"
    regime_path = review_dir / "latest_event_regime_snapshot.json"
    analogy_path = review_dir / "latest_event_crisis_analogy.json"
    asset_map_path = review_dir / "latest_event_asset_shock_map.json"
    operator_summary_path = review_dir / "latest_event_crisis_operator_summary.json"

    _write_json(intake_path, intake_payload)
    _write_json(regime_path, regime_snapshot)
    _write_json(analogy_path, analogy_payload)
    _write_json(asset_map_path, asset_shock_map)
    _write_json(operator_summary_path, operator_summary)

    return {
        "intake": intake_path,
        "regime": regime_path,
        "analogy": analogy_path,
        "asset_map": asset_map_path,
        "operator_summary": operator_summary_path,
    }

def load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("artifact must be a JSON object")
    return payload

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the event crisis pipeline and write latest review artifacts.")
    parser.add_argument("--mode", choices=["snapshot", "hourly", "eod_summary"], required=True)
    parser.add_argument("--output-root", default="output", help="Root directory to write artifacts")
    parser.add_argument("--event-rows-file", help="JSON file with public event rows")
    parser.add_argument("--market-inputs-file", help="JSON file with market input scores")
    parser.add_argument("--now", help="ISO8601 UTC timestamp for generated_at")
    args = parser.parse_args()

    event_rows: list[Mapping[str, object]] = []
    if args.event_rows_file:
        event_rows = list(load_json(Path(args.event_rows_file)).get("events") or [])  # type: ignore[assignment]

    market_inputs: Mapping[str, float] | None = None
    if args.market_inputs_file:
        market_inputs = {**(load_json(Path(args.market_inputs_file)))}  # type: ignore[assignment]

    generated_at = _parse_now(args.now)
    artifacts = run_pipeline(
        output_root=Path(args.output_root),
        mode=args.mode,
        event_rows=event_rows,
        market_inputs=market_inputs,
        generated_at=generated_at,
    )

    print(json.dumps({"artifacts": {k: str(v) for k, v in artifacts.items()}}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
