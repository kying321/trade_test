#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
BROOKS_SCRIPT_PATH = SYSTEM_ROOT / "scripts" / "backtest_brooks_price_action_all_market.py"
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    text = str(raw or "").strip()
    if not text:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def parsed_artifact_stamp(path: Path) -> dt.datetime | None:
    stamp = artifact_stamp(path)
    if not stamp:
        return None
    return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def artifact_sort_key(path: Path) -> tuple[str, float, str]:
    return (artifact_stamp(path), path.stat().st_mtime, path.name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prune_review_artifacts(
    review_dir: Path,
    *,
    stem: str,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
    now_dt: dt.datetime | None = None,
) -> tuple[list[str], list[str]]:
    effective_now = now_dt or now_utc()
    cutoff = effective_now - dt.timedelta(hours=max(1.0, float(ttl_hours)))
    protected = {path.name for path in current_paths}
    candidates = sorted(review_dir.glob(f"*_{stem}*"), key=lambda item: item.stat().st_mtime, reverse=True)

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for path in candidates:
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


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid_json_mapping:{path}")
    return payload


def latest_artifact(review_dir: Path, suffix: str) -> Path:
    candidates = list(review_dir.glob(f"*_{suffix}.json"))
    if not candidates:
        raise FileNotFoundError(f"no_{suffix}_artifact")
    return max(candidates, key=artifact_sort_key)


def latest_optional_artifact(review_dir: Path, suffix: str) -> Path | None:
    candidates = list(review_dir.glob(f"*_{suffix}.json"))
    if not candidates:
        return None
    return max(candidates, key=artifact_sort_key)


def load_brooks_module() -> Any:
    spec = importlib.util.spec_from_file_location("brooks_price_action_market_study_script", BROOKS_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot_load_brooks_price_action_module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def gate_rows_by_symbol(gate_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    rows = (gate_payload or {}).get("symbols") or []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        out[symbol] = row
    return out


def selection_score_map(selection_rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    mapping: dict[tuple[str, str], float] = {}
    for row in selection_rows:
        asset_class = str(row.get("asset_class") or "").strip().lower()
        strategy_id = str(row.get("strategy_id") or "").strip()
        if not asset_class or not strategy_id:
            continue
        mapping[(asset_class, strategy_id)] = float(row.get("selection_score") or 0.0)
    return mapping


def current_candidates(
    *,
    feature_frames: dict[str, Any],
    selected_routes_by_asset_class: dict[str, list[str]],
    selection_rows: list[dict[str, Any]],
    gate_payload: dict[str, Any] | None,
    brooks_module: Any,
    recent_signal_age_bars: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    selection_scores = selection_score_map(selection_rows)
    gate_by_symbol = gate_rows_by_symbol(gate_payload)
    for symbol, frame in sorted(feature_frames.items()):
        asset_class = str(frame["asset_class"].iloc[0]).strip().lower()
        selected_strategies = selected_routes_by_asset_class.get(asset_class, [])
        if not selected_strategies:
            continue
        signal_map = brooks_module.generate_symbol_signals(frame)
        best: dict[str, Any] | None = None
        for strategy_id in selected_strategies:
            for row in signal_map.get(strategy_id, []):
                signal_age_bars = int(len(frame) - 1 - int(row["signal_index"]))
                if signal_age_bars < 0 or signal_age_bars > int(recent_signal_age_bars):
                    continue
                selection_score = float(selection_scores.get((asset_class, strategy_id), 0.0))
                candidate = {
                    "symbol": symbol,
                    "asset_class": asset_class,
                    "strategy_id": strategy_id,
                    "strategy_label": str(row.get("strategy_label") or strategy_id),
                    "direction": str(row.get("direction") or "-"),
                    "signal_ts": row.get("signal_ts"),
                    "signal_age_bars": signal_age_bars,
                    "signal_score": int(row.get("score") or 0),
                    "route_selection_score": selection_score,
                    "note": str(row.get("note") or ""),
                    "current_close": float(frame.iloc[-1]["close"]),
                    "signal_high": float(row.get("signal_high") or 0.0),
                    "signal_low": float(row.get("signal_low") or 0.0),
                    "signal_close": float(row.get("signal_close") or 0.0),
                    "atr14": float(row.get("atr14") or 0.0),
                    "reward_r": float(row.get("reward_r") or 0.0),
                    "max_hold_bars": int(row.get("max_hold_bars") or 0),
                    "structure_readiness": "structure_ready",
                    "route_bridge_status": "manual_structure_route",
                    "route_bridge_blocker_detail": "Brooks structure route is active for this asset class; execution remains manual unless a dedicated execution lane exists.",
                    "route_bridge_done_when": "review signal against live market context before any manual deployment",
                }
                if candidate["direction"] == "LONG":
                    entry_price = candidate["signal_high"]
                    stop_price = candidate["signal_low"] - (0.05 * candidate["atr14"])
                    risk_per_unit = max(0.0, entry_price - stop_price)
                    target_price = entry_price + (risk_per_unit * candidate["reward_r"])
                else:
                    entry_price = candidate["signal_low"]
                    stop_price = candidate["signal_high"] + (0.05 * candidate["atr14"])
                    risk_per_unit = max(0.0, stop_price - entry_price)
                    target_price = entry_price - (risk_per_unit * candidate["reward_r"])
                candidate["entry_price"] = float(entry_price)
                candidate["stop_price"] = float(stop_price)
                candidate["target_price"] = float(target_price)
                candidate["risk_per_unit"] = float(risk_per_unit)
                candidate["rr_ratio"] = float(candidate["reward_r"]) if risk_per_unit > 0.0 else 0.0
                if asset_class == "crypto":
                    gate_row = gate_by_symbol.get(symbol, {})
                    candidate["route_bridge_status"] = "blocked_shortline_gate"
                    candidate["structure_readiness"] = str(gate_row.get("execution_state") or "Bias_Only")
                    candidate["route_bridge_blocker_detail"] = str(gate_row.get("blocker_detail") or "missing_shortline_gate")
                    candidate["route_bridge_done_when"] = str(gate_row.get("done_when") or "complete shortline gate stack")
                if best is None or (
                    candidate["route_selection_score"],
                    candidate["signal_score"],
                    -candidate["signal_age_bars"],
                    candidate["symbol"],
                ) > (
                    best["route_selection_score"],
                    best["signal_score"],
                    -best["signal_age_bars"],
                    best["symbol"],
                ):
                    best = candidate
        if best is not None:
            candidates.append(best)
    candidates.sort(
        key=lambda row: (
            0 if row["route_bridge_status"] == "manual_structure_route" else 1,
            row["asset_class"],
            -float(row["route_selection_score"]),
            -int(row["signal_score"]),
            int(row["signal_age_bars"]),
            row["symbol"],
        )
    )
    return candidates


def asset_class_candidate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["asset_class"]), []).append(row)
    out: list[dict[str, Any]] = []
    for asset_class, items in sorted(grouped.items()):
        top = items[0]
        out.append(
            {
                "asset_class": asset_class,
                "candidate_count": int(len(items)),
                "top_symbol": str(top["symbol"]),
                "top_strategy_id": str(top["strategy_id"]),
                "top_direction": str(top["direction"]),
                "top_signal_age_bars": int(top["signal_age_bars"]),
                "top_route_selection_score": float(top["route_selection_score"]),
                "bridge_status": str(top["route_bridge_status"]),
            }
        )
    return out


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Brooks Price Action Route Report",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- study_artifact: `{payload.get('study_artifact') or ''}`",
        f"- shortline_gate_artifact: `{payload.get('shortline_gate_artifact') or '-'}`",
        f"- selected_routes: `{payload.get('selected_routes_brief') or '-'}`",
        f"- candidate_count: `{payload.get('candidate_count', 0)}`",
        "",
        "## Asset Class Route Summary",
    ]
    for row in payload.get("asset_class_route_summary", []):
        lines.append(
            f"- `{row.get('asset_class')}` candidates=`{row.get('candidate_count')}` top=`{row.get('top_symbol')}:{row.get('top_strategy_id')}:{row.get('top_direction')}` age_bars=`{row.get('top_signal_age_bars')}` bridge=`{row.get('bridge_status')}`"
        )
    lines.extend(["", "## Current Candidates"])
    for row in payload.get("current_candidates", [])[:15]:
        lines.append(
            f"- `{row.get('symbol')}` asset=`{row.get('asset_class')}` strategy=`{row.get('strategy_id')}` direction=`{row.get('direction')}` age_bars=`{row.get('signal_age_bars')}` score=`{row.get('signal_score')}` route_score=`{float(row.get('route_selection_score') or 0.0):.2f}` bridge=`{row.get('route_bridge_status')}`"
        )
        lines.append(
            f"  - plan: entry={float(row.get('entry_price') or 0.0):.4f} stop={float(row.get('stop_price') or 0.0):.4f} target={float(row.get('target_price') or 0.0):.4f} rr={float(row.get('rr_ratio') or 0.0):.2f}"
        )
        lines.append(f"  - blocker: {row.get('route_bridge_blocker_detail')}")
    lines.extend(["", "## Conclusions"])
    for line in payload.get("conclusions", []):
        lines.append(f"- {line}")
    return "\n".join(lines).strip() + "\n"


def selected_routes_brief(selected_routes_by_asset_class: dict[str, list[str]]) -> str:
    if not selected_routes_by_asset_class:
        return "-"
    parts: list[str] = []
    for asset_class, rows in sorted(selected_routes_by_asset_class.items()):
        parts.append(f"{asset_class}:{'/'.join(rows)}")
    return " | ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a current route report from the Brooks all-market adaptive study.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--min-bars", type=int, default=120)
    parser.add_argument("--recent-signal-age-bars", type=int, default=3)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    review_dir = Path(args.review_dir).expanduser().resolve()
    runtime_now = parse_now(args.now)
    brooks_module = load_brooks_module()

    study_path = latest_artifact(review_dir, "brooks_price_action_market_study")
    study_payload = load_json_mapping(study_path)
    gate_path = latest_optional_artifact(review_dir, "crypto_shortline_execution_gate")
    gate_payload = load_json_mapping(gate_path) if gate_path else None

    adaptive_route = dict(study_payload.get("adaptive_route_strategy") or {})
    selected_routes_by_asset_class = {
        str(asset_class): [str(item) for item in rows]
        for asset_class, rows in dict(adaptive_route.get("selected_routes_by_asset_class") or {}).items()
    }

    frames, coverage, scanned, skipped = brooks_module.load_market_dataset(
        output_root=output_root,
        min_bars=max(60, int(args.min_bars)),
        asset_classes=None,
        symbols=None,
    )
    feature_frames = {
        symbol: brooks_module.add_price_action_features(frame)
        for symbol, frame in sorted(frames.items())
    }
    candidates = current_candidates(
        feature_frames=feature_frames,
        selected_routes_by_asset_class=selected_routes_by_asset_class,
        selection_rows=list(adaptive_route.get("selection_rows") or []),
        gate_payload=gate_payload,
        brooks_module=brooks_module,
        recent_signal_age_bars=max(1, int(args.recent_signal_age_bars)),
    )
    route_summary = asset_class_candidate_summary(candidates)
    conclusions = [
        (
            "Adaptive Brooks routing currently favors non-crypto assets."
            if "crypto" not in selected_routes_by_asset_class
            else "Adaptive Brooks routing still allows at least one crypto route."
        ),
        (
            "Current crypto shortline gate remains non-blocking because no crypto Brooks route was selected."
            if "crypto" not in selected_routes_by_asset_class
            else "Crypto Brooks routes still require the shortline gate before any execution."
        ),
        (
            "Current scan found no fresh Brooks route candidates inside the recent signal window."
            if not candidates
            else f"Current scan found {len(candidates)} fresh Brooks route candidates across {len(route_summary)} asset classes."
        ),
        (
            "Equity/future routes are manual structure lanes only; they still need a dedicated execution layer before automation."
        ),
    ]

    payload: dict[str, Any] = {
        "action": "build_brooks_price_action_route_report",
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "study_artifact": str(study_path),
        "shortline_gate_artifact": str(gate_path) if gate_path else None,
        "selected_routes_by_asset_class": selected_routes_by_asset_class,
        "selected_routes_brief": selected_routes_brief(selected_routes_by_asset_class),
        "coverage_summary": {
            "included_symbol_count": int(len(coverage)),
            "scanned_symbol_count": int(len(scanned)),
            "skipped_symbol_count": int(len(skipped)),
        },
        "candidate_count": int(len(candidates)),
        "asset_class_route_summary": route_summary,
        "current_candidates": candidates[:40],
        "conclusions": conclusions,
    }

    review_dir.mkdir(parents=True, exist_ok=True)
    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_brooks_price_action_route_report.json"
    markdown_path = review_dir / f"{stamp}_brooks_price_action_route_report.md"
    checksum_path = review_dir / f"{stamp}_brooks_price_action_route_report_checksum.json"

    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "artifact": str(artifact_path),
        "markdown": str(markdown_path),
        "sha256": sha256_file(artifact_path),
        "generated_at": payload.get("as_of"),
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="brooks_price_action_route_report",
        current_paths=[artifact_path, markdown_path, checksum_path],
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=runtime_now,
    )
    payload["artifact"] = str(artifact_path)
    payload["markdown"] = str(markdown_path)
    payload["checksum"] = str(checksum_path)
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
