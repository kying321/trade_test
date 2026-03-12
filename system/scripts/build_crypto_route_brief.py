#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5


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
    return value.astimezone(dt.timezone.utc).isoformat()


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def parsed_artifact_stamp(path: Path) -> dt.datetime | None:
    stamp = artifact_stamp(path)
    if not stamp:
        return None
    return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def artifact_sort_key(path: Path, reference_now: dt.datetime | None = None) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    effective_now = reference_now or now_utc()
    future_cutoff = effective_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    return (0 if is_future else 1, artifact_stamp(path), path.stat().st_mtime, path.name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_symbol_route_handoff(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_binance_indicator_symbol_route_handoff.json"))
    if not candidates:
        raise FileNotFoundError("no_binance_indicator_symbol_route_handoff_artifact")
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_bnb_flow_focus(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob("*_binance_indicator_bnb_flow_focus.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


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
    cutoff = effective_now - dt.timedelta(hours=max(1.0, ttl_hours))
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
    for path in survivors[keep:]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def build_brief(source_payload: dict[str, Any], bnb_focus_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    deploy_now = [str(x).strip().upper() for x in source_payload.get("deploy_now_symbols", []) if str(x).strip()]
    watch_priority = [str(x).strip().upper() for x in source_payload.get("watch_priority_symbols", []) if str(x).strip()]
    watch_only = [str(x).strip().upper() for x in source_payload.get("watch_only_symbols", []) if str(x).strip()]
    review = [str(x).strip().upper() for x in source_payload.get("review_symbols", []) if str(x).strip()]
    next_focus_symbol = str(source_payload.get("next_focus_symbol", "") or "").strip().upper()
    next_focus_action = str(source_payload.get("next_focus_action", "") or "").strip()
    next_focus_reason = str(source_payload.get("next_focus_reason", "") or "").strip()
    operator_status = str(source_payload.get("operator_status", "") or "").strip()
    route_stack_brief = str(source_payload.get("route_stack_brief", "") or "").strip()
    overall_takeaway = str(source_payload.get("overall_takeaway", "") or "").strip()
    focus_window_gate = ""
    focus_window_gate_reason = ""
    focus_window_verdict = ""
    focus_window_floor = ""
    price_state_window_floor = ""
    comparative_window_takeaway = ""
    xlong_flow_window_floor = ""
    xlong_comparative_window_takeaway = ""
    focus_short_flow_combo = ""
    focus_short_flow_combo_canonical = ""
    focus_long_flow_combo = ""
    focus_long_flow_combo_canonical = ""
    focus_long_top_combo = ""
    focus_long_top_combo_canonical = ""
    focus_brief = ""
    next_retest_action = ""
    next_retest_reason = ""
    if bnb_focus_payload and next_focus_symbol == "BNBUSDT":
        focus_window_gate = str(bnb_focus_payload.get("promotion_gate", "") or "").strip()
        focus_window_gate_reason = str(bnb_focus_payload.get("promotion_gate_reason", "") or "").strip()
        focus_short_flow_combo = str(bnb_focus_payload.get("short_flow_combo", "") or "").strip()
        focus_short_flow_combo_canonical = str(bnb_focus_payload.get("short_flow_combo_canonical", "") or "").strip()
        focus_long_flow_combo = str(bnb_focus_payload.get("long_flow_combo", "") or "").strip()
        focus_long_flow_combo_canonical = str(bnb_focus_payload.get("long_flow_combo_canonical", "") or "").strip()
        focus_long_top_combo = str(bnb_focus_payload.get("long_top_combo", "") or "").strip()
        focus_long_top_combo_canonical = str(bnb_focus_payload.get("long_top_combo_canonical", "") or "").strip()
        focus_window_verdict = str(bnb_focus_payload.get("flow_window_verdict", "") or "").strip()
        focus_window_floor = str(bnb_focus_payload.get("flow_window_floor", "") or "").strip()
        price_state_window_floor = str(bnb_focus_payload.get("price_state_window_floor", "") or "").strip()
        comparative_window_takeaway = str(bnb_focus_payload.get("comparative_window_takeaway", "") or "").strip()
        xlong_flow_window_floor = str(bnb_focus_payload.get("xlong_flow_window_floor", "") or "").strip()
        xlong_comparative_window_takeaway = str(
            bnb_focus_payload.get("xlong_comparative_window_takeaway", "") or ""
        ).strip()
        focus_brief = str(bnb_focus_payload.get("brief", "") or "").strip()
        next_retest_action = str(bnb_focus_payload.get("next_retest_action", "") or "").strip()
        next_retest_reason = str(bnb_focus_payload.get("next_retest_reason", "") or "").strip()

    brief_lines = [
        f"status: {operator_status or '-'}",
        f"routes: {route_stack_brief or '-'}",
        f"focus: {next_focus_symbol or '-'}",
        f"action: {next_focus_action or '-'}",
        f"reason: {next_focus_reason or '-'}",
    ]
    if focus_window_gate:
        brief_lines.append(f"focus-gate: {focus_window_gate}")
    if focus_short_flow_combo_canonical:
        brief_lines.append(f"focus-short-flow: {focus_short_flow_combo_canonical}")
    if focus_long_flow_combo_canonical:
        brief_lines.append(f"focus-long-flow: {focus_long_flow_combo_canonical}")
    if focus_long_top_combo_canonical:
        brief_lines.append(f"focus-long-top: {focus_long_top_combo_canonical}")
    if focus_window_verdict:
        brief_lines.append(f"focus-window: {focus_window_verdict}")
    if focus_window_floor:
        brief_lines.append(f"focus-window-floor: {focus_window_floor}")
    if price_state_window_floor:
        brief_lines.append(f"price-window-floor: {price_state_window_floor}")
    if focus_window_gate_reason:
        brief_lines.append(f"focus-gate-reason: {focus_window_gate_reason}")
    if comparative_window_takeaway:
        brief_lines.append(f"focus-window-note: {comparative_window_takeaway}")
    if xlong_flow_window_floor:
        brief_lines.append(f"xlong-flow-floor: {xlong_flow_window_floor}")
    if xlong_comparative_window_takeaway:
        brief_lines.append(f"xlong-flow-note: {xlong_comparative_window_takeaway}")
    if next_retest_action:
        brief_lines.append(f"next-retest: {next_retest_action}")
    if next_retest_reason:
        brief_lines.append(f"next-retest-reason: {next_retest_reason}")
    if focus_brief:
        brief_lines.append(f"focus-brief: {focus_brief}")
    if overall_takeaway:
        brief_lines.append(f"takeaway: {overall_takeaway}")

    return {
        "operator_status": operator_status,
        "route_stack_brief": route_stack_brief,
        "deploy_now_symbols": deploy_now,
        "watch_priority_symbols": watch_priority,
        "watch_only_symbols": watch_only,
        "review_symbols": review,
        "deploy_count": len(deploy_now),
        "watch_priority_count": len(watch_priority),
        "watch_only_count": len(watch_only),
        "review_count": len(review),
        "next_focus_symbol": next_focus_symbol,
        "next_focus_action": next_focus_action,
        "next_focus_reason": next_focus_reason,
        "focus_window_gate": focus_window_gate,
        "focus_window_gate_reason": focus_window_gate_reason,
        "focus_short_flow_combo": focus_short_flow_combo,
        "focus_short_flow_combo_canonical": focus_short_flow_combo_canonical,
        "focus_long_flow_combo": focus_long_flow_combo,
        "focus_long_flow_combo_canonical": focus_long_flow_combo_canonical,
        "focus_long_top_combo": focus_long_top_combo,
        "focus_long_top_combo_canonical": focus_long_top_combo_canonical,
        "focus_window_verdict": focus_window_verdict,
        "focus_window_floor": focus_window_floor,
        "price_state_window_floor": price_state_window_floor,
        "comparative_window_takeaway": comparative_window_takeaway,
        "xlong_flow_window_floor": xlong_flow_window_floor,
        "xlong_comparative_window_takeaway": xlong_comparative_window_takeaway,
        "focus_brief": focus_brief,
        "next_retest_action": next_retest_action,
        "next_retest_reason": next_retest_reason,
        "overall_takeaway": overall_takeaway,
        "brief_lines": brief_lines,
        "brief_text": "\n".join(brief_lines),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Crypto Route Brief",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_artifact: `{payload.get('source_artifact') or ''}`",
        f"- operator_status: `{payload.get('operator_status') or ''}`",
        f"- route_stack: `{payload.get('route_stack_brief') or ''}`",
        f"- next_focus_symbol: `{payload.get('next_focus_symbol') or '-'}`",
        f"- next_focus_action: `{payload.get('next_focus_action') or '-'}`",
        f"- next_focus_reason: `{payload.get('next_focus_reason') or '-'}`",
        f"- focus_window_gate: `{payload.get('focus_window_gate') or '-'}`",
        f"- focus_short_flow_combo_canonical: `{payload.get('focus_short_flow_combo_canonical') or '-'}`",
        f"- focus_long_flow_combo_canonical: `{payload.get('focus_long_flow_combo_canonical') or '-'}`",
        f"- focus_long_top_combo_canonical: `{payload.get('focus_long_top_combo_canonical') or '-'}`",
        f"- focus_window_verdict: `{payload.get('focus_window_verdict') or '-'}`",
        f"- focus_window_floor: `{payload.get('focus_window_floor') or '-'}`",
        f"- price_state_window_floor: `{payload.get('price_state_window_floor') or '-'}`",
        f"- comparative_window_takeaway: `{payload.get('comparative_window_takeaway') or '-'}`",
        f"- xlong_flow_window_floor: `{payload.get('xlong_flow_window_floor') or '-'}`",
        f"- xlong_comparative_window_takeaway: `{payload.get('xlong_comparative_window_takeaway') or '-'}`",
        f"- next_retest_action: `{payload.get('next_retest_action') or '-'}`",
        f"- next_retest_reason: `{payload.get('next_retest_reason') or '-'}`",
        "",
        "## Buckets",
        f"- deploy_now: `{', '.join(payload.get('deploy_now_symbols', [])) or '-'}`",
        f"- watch_priority: `{', '.join(payload.get('watch_priority_symbols', [])) or '-'}`",
        f"- watch_only: `{', '.join(payload.get('watch_only_symbols', [])) or '-'}`",
        f"- review: `{', '.join(payload.get('review_symbols', [])) or '-'}`",
        "",
        "## Brief",
    ]
    for line in payload.get("brief_lines", []):
        lines.append(f"- {line}")
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a compact crypto route brief from the latest symbol route handoff.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = parse_now(args.now)

    source_path = latest_symbol_route_handoff(review_dir, runtime_now)
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    bnb_focus_path = latest_bnb_flow_focus(review_dir, runtime_now)
    bnb_focus_payload = json.loads(bnb_focus_path.read_text(encoding="utf-8")) if bnb_focus_path else None
    brief = build_brief(source_payload, bnb_focus_payload)

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_crypto_route_brief.json"
    md_path = review_dir / f"{stamp}_crypto_route_brief.md"
    checksum_path = review_dir / f"{stamp}_crypto_route_brief_checksum.json"

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_artifact": str(source_path),
        "bnb_flow_focus_artifact": str(bnb_focus_path) if bnb_focus_path else None,
        **brief,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "generated_at_utc": payload["as_of"],
        "artifact_ttl_hours": max(1.0, float(args.artifact_ttl_hours)),
        "files": [
            {"path": str(json_path), "sha256": sha256_file(json_path)},
            {"path": str(md_path), "sha256": sha256_file(md_path)},
        ],
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="crypto_route_brief",
        current_paths=[json_path, md_path, checksum_path],
        keep=max(3, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=runtime_now,
    )

    payload.update(
        {
            "artifact": str(json_path),
            "markdown": str(md_path),
            "checksum": str(checksum_path),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
