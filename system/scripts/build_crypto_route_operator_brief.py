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


def latest_crypto_route_brief(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_crypto_route_brief.json"))
    if not candidates:
        raise FileNotFoundError("no_crypto_route_brief_artifact")
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


def build_operator_brief(source_payload: dict[str, Any]) -> dict[str, Any]:
    operator_status = str(source_payload.get("operator_status") or "")
    route_stack_brief = str(source_payload.get("route_stack_brief") or "")
    next_focus_symbol = str(source_payload.get("next_focus_symbol") or "")
    next_focus_action = str(source_payload.get("next_focus_action") or "")
    next_focus_reason = str(source_payload.get("next_focus_reason") or "")
    focus_window_gate = str(source_payload.get("focus_window_gate") or "")
    focus_short_flow_combo_canonical = str(source_payload.get("focus_short_flow_combo_canonical") or "")
    focus_long_flow_combo_canonical = str(source_payload.get("focus_long_flow_combo_canonical") or "")
    focus_long_top_combo_canonical = str(source_payload.get("focus_long_top_combo_canonical") or "")
    focus_window_verdict = str(source_payload.get("focus_window_verdict") or "")
    focus_window_floor = str(source_payload.get("focus_window_floor") or "")
    price_state_window_floor = str(source_payload.get("price_state_window_floor") or "")
    comparative_window_takeaway = str(source_payload.get("comparative_window_takeaway") or "")
    xlong_flow_window_floor = str(source_payload.get("xlong_flow_window_floor") or "")
    xlong_comparative_window_takeaway = str(source_payload.get("xlong_comparative_window_takeaway") or "")
    focus_brief = str(source_payload.get("focus_brief") or "")
    next_retest_action = str(source_payload.get("next_retest_action") or "")
    next_retest_reason = str(source_payload.get("next_retest_reason") or "")
    deploy_now_symbols = list(source_payload.get("deploy_now_symbols") or [])
    watch_priority_symbols = list(source_payload.get("watch_priority_symbols") or [])
    watch_only_symbols = list(source_payload.get("watch_only_symbols") or [])

    operator_lines = [
        f"status: {operator_status or '-'}",
        f"routes: {route_stack_brief or '-'}",
        f"focus: {next_focus_symbol or '-'}",
        f"action: {next_focus_action or '-'}",
        f"focus-gate: {focus_window_gate or '-'}",
        f"focus-short-flow: {focus_short_flow_combo_canonical or '-'}",
        f"focus-long-flow: {focus_long_flow_combo_canonical or '-'}",
        f"focus-long-top: {focus_long_top_combo_canonical or '-'}",
        f"focus-window: {focus_window_verdict or '-'}",
        f"focus-window-floor: {focus_window_floor or '-'}",
        f"price-state-window-floor: {price_state_window_floor or '-'}",
        f"next-retest: {next_retest_action or '-'}",
        f"reason: {next_focus_reason or '-'}",
    ]
    if next_retest_reason:
        operator_lines.append(f"next-retest-reason: {next_retest_reason}")
    if comparative_window_takeaway:
        operator_lines.append(f"focus-window-note: {comparative_window_takeaway}")
    if xlong_flow_window_floor:
        operator_lines.append(f"xlong-flow-floor: {xlong_flow_window_floor}")
    if xlong_comparative_window_takeaway:
        operator_lines.append(f"xlong-flow-note: {xlong_comparative_window_takeaway}")
    if focus_brief:
        operator_lines.append(f"focus-brief: {focus_brief}")
    operator_text = " | ".join(operator_lines)
    return {
        "operator_status": operator_status,
        "route_stack_brief": route_stack_brief,
        "next_focus_symbol": next_focus_symbol,
        "next_focus_action": next_focus_action,
        "next_focus_reason": next_focus_reason,
        "focus_window_gate": focus_window_gate,
        "focus_short_flow_combo_canonical": focus_short_flow_combo_canonical,
        "focus_long_flow_combo_canonical": focus_long_flow_combo_canonical,
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
        "deploy_now_symbols": deploy_now_symbols,
        "watch_priority_symbols": watch_priority_symbols,
        "watch_only_symbols": watch_only_symbols,
        "operator_lines": operator_lines,
        "operator_text": operator_text,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Crypto Route Operator Brief",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_artifact: `{payload.get('source_artifact') or ''}`",
        "",
        "## Brief",
    ]
    for line in payload.get("operator_lines", []):
        lines.append(f"- {line}")
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a short operator brief from the latest crypto route brief artifact.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = parse_now(args.now)

    source_path = latest_crypto_route_brief(review_dir, runtime_now)
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    brief = build_operator_brief(source_payload)

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_crypto_route_operator_brief.json"
    md_path = review_dir / f"{stamp}_crypto_route_operator_brief.md"
    checksum_path = review_dir / f"{stamp}_crypto_route_operator_brief_checksum.json"

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_artifact": str(source_path),
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
        stem="crypto_route_operator_brief",
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
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
