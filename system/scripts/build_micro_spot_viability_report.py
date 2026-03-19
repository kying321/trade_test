#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_now(raw: str) -> datetime | None:
    text_value = str(raw or "").strip()
    if not text_value:
        return None
    try:
        parsed = datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def resolve_path(raw: str, *, anchor: Path) -> Path:
    path = Path(str(raw).strip())
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return anchor / path


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest(), int(path.stat().st_size)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def find_latest(directory: Path, pattern: str) -> Path | None:
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_ticket_payload(review_dir: Path, ticket_json: str) -> tuple[dict[str, Any], Path]:
    if str(ticket_json).strip():
        path = resolve_path(ticket_json, anchor=review_dir)
    else:
        path = find_latest(review_dir, "*_signal_to_order_tickets.json")
        if path is None:
            raise FileNotFoundError("missing signal_to_order_tickets artifact")
    if not path.exists():
        raise FileNotFoundError(f"ticket artifact not found: {path}")
    payload = read_json(path, None)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid ticket payload: {path}")
    return payload, path


def select_focus_ticket(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    tickets = payload.get("tickets", [])
    if not isinstance(tickets, list):
        return {}
    if str(symbol).strip():
        wanted = str(symbol).strip().upper()
        for row in tickets:
            if str(row.get("symbol", "")).upper() == wanted:
                return row
    for row in tickets:
        reasons = list(row.get("reasons", []))
        if "size_below_min_notional" in reasons:
            return row
    for row in tickets:
        if not bool(row.get("allowed", False)):
            return row
    return tickets[0] if tickets else {}


def _parse_candidate_params(raw: str) -> dict[str, Any]:
    try:
        payload = ast.literal_eval(raw)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def collect_accepted_crypto_candidates(research_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(research_root.glob("strategy_lab_*/candidates.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if not to_bool(row.get("accepted")):
                        continue
                    name = str(row.get("name", "")).strip()
                    if "crypto" not in name:
                        continue
                    params = _parse_candidate_params(str(row.get("params", "")))
                    if not params:
                        continue
                    out.append(
                        {
                            "path": str(path),
                            "name": name,
                            "signal_confidence_min": to_float(params.get("signal_confidence_min"), 0.0),
                            "convexity_min": to_float(params.get("convexity_min"), 0.0),
                            "hold_days": to_float(params.get("hold_days"), 0.0),
                            "max_daily_trades": to_float(params.get("max_daily_trades"), 0.0),
                            "score": to_float(row.get("score"), 0.0),
                            "robustness_score": to_float(row.get("robustness_score"), 0.0),
                            "validation_metrics": str(row.get("validation_metrics", "")),
                        }
                    )
        except Exception:
            continue
    out.sort(
        key=lambda x: (
            x["signal_confidence_min"],
            x["convexity_min"],
            -x["score"],
            -x["robustness_score"],
        )
    )
    return out


def render_markdown(payload: dict[str, Any]) -> str:
    current = payload.get("current_ticket", {})
    viability = payload.get("viability", {})
    candidate = payload.get("lowest_threshold_accepted_crypto_candidate", {})
    lines = [
        "# Micro Spot Viability Report",
        "",
        f"- generated_at_utc: `{payload.get('generated_at_utc', '')}`",
        f"- ticket_artifact: `{payload.get('ticket_artifact', '')}`",
        f"- focus_symbol: `{current.get('symbol', '')}`",
        f"- current_reasons: `{current.get('reasons', [])}`",
        f"- current_confidence: `{current.get('signal', {}).get('confidence')}`",
        f"- current_convexity: `{current.get('signal', {}).get('convexity_ratio')}`",
        f"- current_quote_usdt: `{current.get('sizing', {}).get('quote_usdt')}`",
        f"- min_notional_usdt: `{current.get('sizing', {}).get('min_notional_usdt')}`",
        f"- threshold_source_kind: `{payload.get('thresholds', {}).get('threshold_source_kind', '')}`",
        f"- threshold_source: `{payload.get('thresholds', {}).get('threshold_source', '')}`",
        "",
        "## Accepted Crypto Candidate",
        "",
        f"- candidate_name: `{candidate.get('name', '')}`",
        f"- candidate_path: `{candidate.get('path', '')}`",
        f"- candidate_confidence_min: `{candidate.get('signal_confidence_min')}`",
        f"- candidate_convexity_min: `{candidate.get('convexity_min')}`",
        f"- candidate_score: `{candidate.get('score')}`",
        "",
        "## Viability",
        "",
        f"- threshold_only_passes: `{viability.get('threshold_only_passes')}`",
        f"- size_still_blocked: `{viability.get('size_still_blocked')}`",
        f"- expected_loss_at_min_notional_usdt: `{viability.get('expected_loss_at_min_notional_usdt')}`",
        f"- required_base_risk_pct_current_signal: `{viability.get('required_base_risk_pct_current_signal')}`",
        f"- required_base_risk_pct_full_conviction: `{viability.get('required_base_risk_pct_full_conviction')}`",
        "",
        "## Recommendation",
        "",
        f"- `{payload.get('recommendation', '')}`",
    ]
    return "\n".join(lines) + "\n"


def build_report(
    *,
    review_dir: Path,
    research_root: Path,
    ticket_json: str,
    symbol: str,
    reference_now: datetime,
) -> dict[str, Any]:
    ticket_payload, ticket_path = load_ticket_payload(review_dir, ticket_json)
    focus = select_focus_ticket(ticket_payload, symbol)
    if not focus:
        raise ValueError("missing focus ticket")

    thresholds = ticket_payload.get("thresholds", {})
    candidates = collect_accepted_crypto_candidates(research_root)
    lowest = candidates[0] if candidates else {}

    signal = focus.get("signal", {}) if isinstance(focus.get("signal", {}), dict) else {}
    sizing = focus.get("sizing", {}) if isinstance(focus.get("sizing", {}), dict) else {}
    confidence = to_float(signal.get("confidence"))
    convexity = to_float(signal.get("convexity_ratio"))
    quote_usdt = to_float(sizing.get("quote_usdt"))
    min_notional = to_float(sizing.get("min_notional_usdt"))
    equity_usdt = to_float(sizing.get("equity_usdt"))
    risk_multiplier = to_float(sizing.get("risk_multiplier"))
    risk_budget_usdt = to_float(sizing.get("risk_budget_usdt"))
    conviction = to_float(sizing.get("conviction"))
    risk_per_unit_pct = to_float(focus.get("levels", {}).get("risk_per_unit_pct"))

    expected_loss_at_min_notional = float(min_notional * risk_per_unit_pct / 100.0)
    denom_current = equity_usdt * max(risk_multiplier, 0.0) * max(conviction, 0.0)
    denom_full = equity_usdt * max(risk_multiplier, 0.0)
    required_base_risk_pct_current_signal = (
        float(expected_loss_at_min_notional * 100.0 / denom_current) if denom_current > 0.0 else 0.0
    )
    required_base_risk_pct_full_conviction = (
        float(expected_loss_at_min_notional * 100.0 / denom_full) if denom_full > 0.0 else 0.0
    )

    threshold_only_passes = False
    if lowest:
        threshold_only_passes = bool(
            confidence >= to_float(lowest.get("signal_confidence_min"))
            and convexity >= to_float(lowest.get("convexity_min"))
        )

    size_still_blocked = bool(quote_usdt + 1e-12 < min_notional)
    recommendation = (
        "仅放宽 confidence/convexity 阈值不会解除 size_below_min_notional；下一步应设计 micro_spot sizing contract，"
        "而不是直接改 live params。"
        if size_still_blocked
        else "当前 size 已可行，可进一步评估低门槛 crypto candidate 是否值得进入 SIM_ONLY profile。"
    )

    return {
        "generated_at_utc": reference_now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ticket_artifact": str(ticket_path),
        "thresholds": dict(thresholds) if isinstance(thresholds, dict) else {},
        "focus_symbol": str(focus.get("symbol", "")),
        "current_ticket": focus,
        "accepted_crypto_candidate_count": int(len(candidates)),
        "lowest_threshold_accepted_crypto_candidate": lowest,
        "viability": {
            "threshold_only_passes": bool(threshold_only_passes),
            "size_still_blocked": bool(size_still_blocked),
            "quote_usdt": float(quote_usdt),
            "min_notional_usdt": float(min_notional),
            "risk_budget_usdt": float(risk_budget_usdt),
            "expected_loss_at_min_notional_usdt": float(expected_loss_at_min_notional),
            "required_base_risk_pct_current_signal": float(required_base_risk_pct_current_signal),
            "required_base_risk_pct_full_conviction": float(required_base_risk_pct_full_conviction),
        },
        "recommendation": recommendation,
        "research_root": str(research_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a micro spot viability report from ticket and research artifacts.")
    parser.add_argument("--review-dir", default="output/review")
    parser.add_argument("--research-root", default="output/research")
    parser.add_argument("--ticket-json", default="")
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--now", default="")
    args = parser.parse_args()

    system_root = Path(__file__).resolve().parents[1]
    review_dir = resolve_path(args.review_dir, anchor=system_root)
    research_root = resolve_path(args.research_root, anchor=system_root)
    reference_now = parse_now(args.now) or now_utc()

    payload = build_report(
        review_dir=review_dir,
        research_root=research_root,
        ticket_json=str(args.ticket_json),
        symbol=str(args.symbol),
        reference_now=reference_now,
    )
    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    out_json = review_dir / f"{stamp}_micro_spot_viability_report.json"
    out_md = review_dir / f"{stamp}_micro_spot_viability_report.md"
    out_checksum = review_dir / f"{stamp}_micro_spot_viability_report_checksum.json"
    write_json(out_json, payload)
    write_text(out_md, render_markdown(payload))
    json_sha, json_size = sha256_file(out_json)
    md_sha, md_size = sha256_file(out_md)
    write_json(
        out_checksum,
        {
            "files": [
                {"path": str(out_json), "sha256": json_sha, "size": json_size},
                {"path": str(out_md), "sha256": md_sha, "size": md_size},
            ]
        },
    )
    print(
        json.dumps(
            {"json": str(out_json), "md": str(out_md), "checksum": str(out_checksum), "focus_symbol": payload["focus_symbol"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
