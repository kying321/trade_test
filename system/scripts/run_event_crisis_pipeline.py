from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from lie_engine.research.event_crisis_pipeline import (
    build_event_asset_shock_map,
    build_event_crisis_analogy,
    build_event_live_guard_overlay,
    build_event_regime_snapshot,
)
from lie_engine.research.event_crisis_sources import (
    BOOTSTRAP_MARKET_INPUTS,
    DEFAULT_PRIORITY_ASSETS,
    normalize_market_inputs,
    normalize_public_event_rows,
)
from lie_engine.research.event_game_state import build_event_game_state_snapshot
from lie_engine.research.event_transmission import build_event_transmission_chain_map
from lie_engine.research.event_safety_margin import build_event_safety_margin_snapshot


ARTIFACT_ORDER = [
    "latest_event_intake.json",
    "latest_event_game_state_snapshot.json",
    "latest_event_transmission_chain_map.json",
    "latest_event_regime_snapshot.json",
    "latest_event_crisis_analogy.json",
    "latest_event_asset_shock_map.json",
    "latest_event_safety_margin_snapshot.json",
    "event_live_guard_overlay.json",
    "latest_event_crisis_operator_summary.json",
]


def _ensure_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _parse_now(raw: str | None) -> dt.datetime:
    if not raw:
        return _ensure_utc(dt.datetime.now(dt.timezone.utc))
    text = raw.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = dt.datetime.fromisoformat(text)
    return _ensure_utc(parsed)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hard_boundary_brief(payload: Mapping[str, object]) -> str:
    hard_boundaries = payload.get("hard_boundaries")
    if not isinstance(hard_boundaries, Mapping):
        return "none"
    active = [key for key, value in hard_boundaries.items() if bool(value)]
    return active[0] if active else "none"


def run_pipeline(
    *,
    output_root: Path,
    mode: str,
    event_rows: Sequence[Mapping[str, object]] | None = None,
    market_inputs: Mapping[str, float] | None = None,
    generated_at: dt.datetime | None = None,
) -> Mapping[str, Path]:
    now = _ensure_utc(generated_at or dt.datetime.now(dt.timezone.utc))
    normalized_events = normalize_public_event_rows(event_rows or [], default_ts=now)
    bootstrap_mode = len(normalized_events) == 0 and not market_inputs
    normalized_markets = normalize_market_inputs(
        market_inputs,
        defaults=BOOTSTRAP_MARKET_INPUTS if bootstrap_mode else None,
    )

    intake_payload = {
        "generated_at_utc": now.isoformat().replace("+00:00", "Z"),
        "mode": mode,
        "event_count": len(normalized_events),
        "events": normalized_events,
    }

    game_state_snapshot = build_event_game_state_snapshot(
        event_rows=normalized_events, market_inputs=normalized_markets
    )
    transmission_chain_map = build_event_transmission_chain_map(
        game_state_snapshot=game_state_snapshot
    )
    regime_snapshot = build_event_regime_snapshot(
        event_rows=normalized_events,
        market_inputs=normalized_markets,
        game_state_snapshot=game_state_snapshot,
        transmission_chain_map=transmission_chain_map,
    )
    analogy_payload = build_event_crisis_analogy(
        event_rows=normalized_events, market_inputs=normalized_markets
    )
    asset_shock_map = build_event_asset_shock_map(
        event_rows=normalized_events,
        market_inputs=normalized_markets,
        transmission_chain_map=transmission_chain_map,
    )
    safety_margin_snapshot = build_event_safety_margin_snapshot(
        game_state_snapshot=game_state_snapshot,
        transmission_chain_map=transmission_chain_map,
        regime_snapshot=regime_snapshot,
    )
    overlay_payload = build_event_live_guard_overlay(
        regime_snapshot=regime_snapshot,
        safety_margin_snapshot=safety_margin_snapshot,
        transmission_chain_map=transmission_chain_map,
        generated_at=now,
    )
    operator_summary = {
        "generated_at_utc": intake_payload["generated_at_utc"],
        "mode": mode,
        "status": regime_snapshot.get("regime_state") or "watch",
        "summary": regime_snapshot.get("regime_state") or "watch",
        "takeaway": dominant_chain if (dominant_chain := transmission_chain_map.get("dominant_chain")) else regime_snapshot.get("regime_state"),
        "regime_state": regime_snapshot.get("regime_state"),
        "event_severity_score": regime_snapshot.get("event_severity_score"),
        "systemic_risk_score": regime_snapshot.get("systemic_risk_score"),
        "top_analogues": analogy_payload.get("top_analogues"),
        "priority_assets": DEFAULT_PRIORITY_ASSETS,
        "event_crisis_primary_theater_brief": game_state_snapshot.get("primary_theater")
        or transmission_chain_map.get("primary_theater"),
        "event_crisis_dominant_chain_brief": transmission_chain_map.get("dominant_chain"),
        "event_crisis_safety_margin_brief": (
            f"system_margin={safety_margin_snapshot.get('system_margin_score')}"
            if safety_margin_snapshot.get("system_margin_score") is not None
            else ""
        ),
        "event_crisis_hard_boundary_brief": _hard_boundary_brief(safety_margin_snapshot),
    }

    review_dir = output_root / "review"
    intake_path = review_dir / "latest_event_intake.json"
    game_state_path = review_dir / "latest_event_game_state_snapshot.json"
    transmission_path = review_dir / "latest_event_transmission_chain_map.json"
    regime_path = review_dir / "latest_event_regime_snapshot.json"
    analogy_path = review_dir / "latest_event_crisis_analogy.json"
    asset_map_path = review_dir / "latest_event_asset_shock_map.json"
    safety_margin_path = review_dir / "latest_event_safety_margin_snapshot.json"
    operator_summary_path = review_dir / "latest_event_crisis_operator_summary.json"
    overlay_path = output_root / "state" / "event_live_guard_overlay.json"

    _write_json(intake_path, intake_payload)
    _write_json(game_state_path, game_state_snapshot)
    _write_json(transmission_path, transmission_chain_map)
    _write_json(regime_path, regime_snapshot)
    _write_json(analogy_path, analogy_payload)
    _write_json(asset_map_path, asset_shock_map)
    _write_json(safety_margin_path, safety_margin_snapshot)
    _write_json(overlay_path, overlay_payload)
    _write_json(operator_summary_path, operator_summary)

    return {
        "intake": intake_path,
        "game_state": game_state_path,
        "transmission": transmission_path,
        "regime": regime_path,
        "analogy": analogy_path,
        "asset_map": asset_map_path,
        "safety_margin": safety_margin_path,
        "overlay": overlay_path,
        "operator_summary": operator_summary_path,
    }


def load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("artifact must be a JSON object")
    return payload


def load_event_rows_from_file(path: Path) -> list[Mapping[str, object]]:
    payload = load_json(path)
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("events must be a list")
    return [row for row in events if isinstance(row, Mapping)]


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
        event_rows = load_event_rows_from_file(Path(args.event_rows_file))

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
