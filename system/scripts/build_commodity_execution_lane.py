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


def _list_text(values: list[str], limit: int = 4) -> str:
    items = [str(v).strip() for v in values if str(v).strip()]
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" (+{len(items) - limit})"


def _non_crypto(values: list[str]) -> list[str]:
    return [str(x).strip() for x in values if str(x).strip() and "crypto" not in str(x).lower()]


def _collect_leaders_from_research(payload: dict[str, Any], batches: list[str]) -> list[str]:
    regime_playbook = payload.get("regime_playbook", {})
    if not isinstance(regime_playbook, dict):
        return []
    batch_rules = regime_playbook.get("batch_rules", [])
    if not isinstance(batch_rules, list):
        return []
    wanted = set(batches)
    seen: set[str] = set()
    leaders: list[str] = []
    for row in batch_rules:
        if not isinstance(row, dict):
            continue
        batch = str(row.get("batch") or "").strip()
        if batch not in wanted:
            continue
        for symbol in row.get("leader_symbols", []):
            tag = str(symbol).strip().upper()
            if tag and tag not in seen:
                seen.add(tag)
                leaders.append(tag)
    return leaders


def _route_stack(primary: list[str], regime: list[str], shadow: list[str]) -> str:
    parts: list[str] = []
    if primary:
        parts.append("paper-primary:" + ",".join(primary))
    if regime:
        parts.append("regime-filter:" + ",".join(regime))
    if shadow:
        parts.append("shadow:" + ",".join(shadow))
    return " | ".join(parts)


def latest_hot_universe_commodity_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob("*_hot_universe_research.json"))
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda path: artifact_sort_key(path, reference_now), reverse=True)
    ranked: list[tuple[int, tuple[int, str, float, str], Path]] = []
    for path in ordered:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ladder = payload.get("research_action_ladder", {})
        if not isinstance(ladder, dict):
            continue
        primary = _non_crypto(ladder.get("focus_primary_batches", []))
        regime = _non_crypto(ladder.get("focus_with_regime_filter_batches", []))
        shadow = _non_crypto(ladder.get("shadow_only_batches", []))
        score = len(primary) * 4 + len(regime) * 3 + len(shadow) * 2
        if score <= 0:
            continue
        if str(payload.get("status") or "").strip() == "dry_run":
            score -= 1
        ranked.append((score, artifact_sort_key(path, reference_now), path))
    if not ranked:
        return None
    return max(ranked, key=lambda item: (item[0], item[1]))[2]


def latest_live_gate_blocker_report(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob("*_live_gate_blocker_report.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def derive_from_hot_universe(payload: dict[str, Any]) -> dict[str, Any]:
    ladder = dict(payload.get("research_action_ladder") or {})
    primary = _non_crypto(ladder.get("focus_primary_batches", []))
    regime = _non_crypto(ladder.get("focus_with_regime_filter_batches", []))
    shadow = _non_crypto(ladder.get("shadow_only_batches", []))
    avoid = _non_crypto(ladder.get("avoid_batches", []))
    leaders_primary = _collect_leaders_from_research(payload, primary)
    leaders_regime = _collect_leaders_from_research(payload, regime)
    next_focus_batch = primary[0] if primary else (regime[0] if regime else "")
    next_focus_symbols = leaders_primary if next_focus_batch in set(primary) else leaders_regime
    return {
        "route_status": "paper-first",
        "execution_mode": "paper_first",
        "design_status": str(payload.get("status") or "").strip() or "ok",
        "focus_primary_batches": primary,
        "focus_with_regime_filter_batches": regime,
        "shadow_only_batches": shadow,
        "avoid_batches": avoid,
        "leader_symbols_primary": leaders_primary,
        "leader_symbols_regime_filter": leaders_regime,
        "next_focus_batch": next_focus_batch,
        "next_focus_symbols": next_focus_symbols,
        "next_stage": "paper_ticket_lane",
        "route_stack_brief": _route_stack(primary, regime, shadow),
        "stage_plan": [
            {
                "stage": "paper_ticket_lane",
                "batches": primary + regime,
                "rule": "Route validated commodity sleeves into paper tickets only.",
            }
        ],
    }


def derive_from_blocker_report(payload: dict[str, Any]) -> dict[str, Any]:
    path = dict(payload.get("commodity_execution_path") or {})
    primary = [str(x).strip() for x in path.get("focus_primary_batches", []) if str(x).strip()]
    regime = [str(x).strip() for x in path.get("focus_with_regime_filter_batches", []) if str(x).strip()]
    shadow = [str(x).strip() for x in path.get("shadow_only_batches", []) if str(x).strip()]
    avoid = [str(x).strip() for x in path.get("avoid_batches", []) if str(x).strip()]
    leaders_primary = [str(x).strip().upper() for x in path.get("leader_symbols_primary", []) if str(x).strip()]
    leaders_regime = [str(x).strip().upper() for x in path.get("leader_symbols_regime_filter", []) if str(x).strip()]
    next_focus_batch = primary[0] if primary else (regime[0] if regime else "")
    next_focus_symbols = leaders_primary if next_focus_batch in set(primary) else leaders_regime
    return {
        "route_status": "paper-first",
        "execution_mode": str(path.get("execution_mode") or "paper_first"),
        "design_status": str(path.get("design_status") or payload.get("status") or "").strip() or "proposed",
        "focus_primary_batches": primary,
        "focus_with_regime_filter_batches": regime,
        "shadow_only_batches": shadow,
        "avoid_batches": avoid,
        "leader_symbols_primary": leaders_primary,
        "leader_symbols_regime_filter": leaders_regime,
        "next_focus_batch": next_focus_batch,
        "next_focus_symbols": next_focus_symbols,
        "next_stage": "paper_ticket_lane",
        "route_stack_brief": _route_stack(primary, regime, shadow),
        "stage_plan": list(path.get("stages") or []),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Commodity Execution Lane",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_mode: `{payload.get('source_mode') or ''}`",
        f"- source_artifact: `{payload.get('source_artifact') or ''}`",
        f"- route_status: `{payload.get('route_status') or ''}`",
        f"- execution_mode: `{payload.get('execution_mode') or ''}`",
        f"- route_stack: `{payload.get('route_stack_brief') or ''}`",
        f"- next_focus_batch: `{payload.get('next_focus_batch') or '-'}`",
        f"- next_focus_symbols: `{_list_text(payload.get('next_focus_symbols', []))}`",
        "",
        "## Summary",
    ]
    for line in payload.get("summary_lines", []):
        lines.append(f"- {line}")
    lines.extend(["", "## Stages"])
    for row in payload.get("stage_plan", []):
        if not isinstance(row, dict):
            continue
        stage = str(row.get("stage") or "").strip()
        rule = str(row.get("rule") or "").strip()
        batches = _list_text(row.get("batches", []))
        lines.append(f"- `{stage}`: `{batches}`")
        if rule:
            lines.append(f"  - {rule}")
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a commodity paper-first execution lane artifact.")
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

    source_path = latest_hot_universe_commodity_source(review_dir, runtime_now)
    source_mode = "hot-universe-action-ladder"
    source_payload: dict[str, Any]
    if source_path is not None:
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        lane = derive_from_hot_universe(source_payload)
    else:
        blocker_path = latest_live_gate_blocker_report(review_dir, runtime_now)
        if blocker_path is None:
            raise FileNotFoundError("no_commodity_execution_source")
        source_mode = "blocker-report-commodity-path"
        source_path = blocker_path
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        lane = derive_from_blocker_report(source_payload)

    summary_lines = [
        f"status: {lane.get('route_status') or '-'}",
        f"execution-mode: {lane.get('execution_mode') or '-'}",
        f"primary: {_list_text(lane.get('focus_primary_batches', []))}",
        f"regime-filter: {_list_text(lane.get('focus_with_regime_filter_batches', []))}",
        f"shadow: {_list_text(lane.get('shadow_only_batches', []))}",
        f"leaders-primary: {_list_text(lane.get('leader_symbols_primary', []))}",
        f"leaders-regime: {_list_text(lane.get('leader_symbols_regime_filter', []))}",
        f"next-focus-batch: {lane.get('next_focus_batch') or '-'}",
        f"next-focus-symbols: {_list_text(lane.get('next_focus_symbols', []))}",
        f"next-stage: {lane.get('next_stage') or '-'}",
    ]

    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_commodity_execution_lane.json"
    md_path = review_dir / f"{stamp}_commodity_execution_lane.md"
    checksum_path = review_dir / f"{stamp}_commodity_execution_lane_checksum.json"

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_mode": source_mode,
        "source_artifact": str(source_path),
        "source_status": str(source_payload.get("status") or ""),
        **lane,
        "summary_lines": summary_lines,
        "summary_text": " | ".join(summary_lines),
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
        stem="commodity_execution_lane",
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
    checksum_payload["files"][0]["sha256"] = sha256_file(json_path)
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
