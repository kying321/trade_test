#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"
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


def artifact_sort_key(path: Path) -> tuple[str, float, str]:
    return (artifact_stamp(path), path.stat().st_mtime, path.name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _tier_for_plan_status(plan_status: str) -> str:
    plan_status_text = str(plan_status or "").strip()
    if plan_status_text == "manual_structure_review_now":
        return "review_queue_now"
    if plan_status_text == "blocked_shortline_gate":
        return "blocked_queue"
    if plan_status_text == "route_candidate_only":
        return "route_candidate_only"
    return "informational_queue"


def _priority_for_row(
    *,
    plan_status: str,
    route_selection_score: float,
    signal_score: int,
    signal_age_bars: int,
) -> tuple[int, str]:
    plan_status_text = str(plan_status or "").strip()
    if plan_status_text == "manual_structure_review_now":
        base = 60
    elif plan_status_text == "blocked_shortline_gate":
        base = 25
    elif plan_status_text == "route_candidate_only":
        base = 15
    else:
        base = 10
    route_component = min(25, max(0, int(round(float(route_selection_score or 0.0) / 4.0))))
    signal_component = min(10, max(0, int(round(int(signal_score or 0) / 10.0))))
    age_component = max(0, 10 - min(max(0, int(signal_age_bars or 0)), 10))
    priority_score = min(100, base + route_component + signal_component + age_component)
    if plan_status_text == "manual_structure_review_now":
        priority_tier = "review_queue_now"
    elif plan_status_text == "blocked_shortline_gate":
        priority_tier = "blocked_review"
    elif plan_status_text == "route_candidate_only":
        priority_tier = "route_candidate_only"
    else:
        priority_tier = "informational_review"
    return priority_score, priority_tier


def _queue_row_from_plan_item(row: dict[str, Any]) -> dict[str, Any]:
    plan_status = str(row.get("plan_status") or "").strip()
    route_selection_score = float(row.get("route_selection_score") or 0.0)
    signal_score = int(row.get("signal_score") or 0)
    signal_age_bars = int(row.get("signal_age_bars") or 0)
    priority_score, priority_tier = _priority_for_row(
        plan_status=plan_status,
        route_selection_score=route_selection_score,
        signal_score=signal_score,
        signal_age_bars=signal_age_bars,
    )
    return {
        "symbol": str(row.get("symbol") or "").strip().upper(),
        "asset_class": str(row.get("asset_class") or "").strip().lower(),
        "strategy_id": str(row.get("strategy_id") or "").strip(),
        "direction": str(row.get("direction") or "").strip().upper(),
        "tier": _tier_for_plan_status(plan_status),
        "plan_status": plan_status,
        "execution_action": str(row.get("execution_action") or "").strip(),
        "route_selection_score": route_selection_score,
        "signal_score": signal_score,
        "signal_age_bars": signal_age_bars,
        "priority_score": priority_score,
        "priority_tier": priority_tier,
        "blocker_detail": str(row.get("plan_blocker_detail") or "").strip(),
        "done_when": str(row.get("plan_done_when") or "").strip(),
    }


def _queue_row_from_route_candidate(row: dict[str, Any]) -> dict[str, Any]:
    route_selection_score = float(row.get("route_selection_score") or 0.0)
    signal_score = int(row.get("signal_score") or 0)
    signal_age_bars = int(row.get("signal_age_bars") or 0)
    priority_score, priority_tier = _priority_for_row(
        plan_status="route_candidate_only",
        route_selection_score=route_selection_score,
        signal_score=signal_score,
        signal_age_bars=signal_age_bars,
    )
    return {
        "symbol": str(row.get("symbol") or "").strip().upper(),
        "asset_class": str(row.get("asset_class") or "").strip().lower(),
        "strategy_id": str(row.get("strategy_id") or "").strip(),
        "direction": str(row.get("direction") or "").strip().upper(),
        "tier": "route_candidate_only",
        "plan_status": "route_candidate_only",
        "execution_action": "review_route_only",
        "route_selection_score": route_selection_score,
        "signal_score": signal_score,
        "signal_age_bars": signal_age_bars,
        "priority_score": priority_score,
        "priority_tier": priority_tier,
        "blocker_detail": str(row.get("route_bridge_blocker_detail") or "").strip(),
        "done_when": "Brooks route candidate receives a concrete execution plan head.",
    }


def _queue_sort_key(row: dict[str, Any]) -> tuple[int, int, float, int, int, str]:
    plan_status = str(row.get("plan_status") or "").strip()
    status_rank = {
        "manual_structure_review_now": 0,
        "blocked_shortline_gate": 1,
        "route_candidate_only": 2,
    }.get(plan_status, 3)
    return (
        -int(row.get("priority_score") or 0),
        status_rank,
        -float(row.get("route_selection_score") or 0.0),
        -int(row.get("signal_score") or 0),
        int(row.get("signal_age_bars") or 0),
        str(row.get("symbol") or ""),
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Brooks Structure Review Queue",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- route_report_artifact: `{payload.get('route_report_artifact') or ''}`",
        f"- execution_plan_artifact: `{payload.get('execution_plan_artifact') or ''}`",
        f"- queue_status: `{payload.get('queue_status') or ''}`",
        f"- queue_count: `{payload.get('queue_count', 0)}`",
        f"- priority_status: `{payload.get('priority_status') or ''}`",
        f"- priority_brief: `{payload.get('priority_brief') or ''}`",
        "",
        "## Head",
    ]
    head = payload.get("head") or {}
    if head:
        lines.extend(
            [
                f"- `{head.get('symbol')}` `{head.get('strategy_id')}` `{head.get('direction')}` tier=`{head.get('tier')}` status=`{head.get('plan_status')}` action=`{head.get('execution_action')}`",
                f"- route_score=`{float(head.get('route_selection_score') or 0.0):.6f}` signal_score=`{int(head.get('signal_score') or 0)}` age_bars=`{int(head.get('signal_age_bars') or 0)}`",
                f"- priority_score=`{int(head.get('priority_score') or 0)}` priority_tier=`{head.get('priority_tier') or ''}`",
                f"- blocker: {head.get('blocker_detail') or '-'}",
            ]
        )
    lines.extend(["", "## Queue"])
    for row in payload.get("queue", [])[:20]:
        lines.append(
            f"- `{row.get('rank')}` `{row.get('symbol')}` strategy=`{row.get('strategy_id')}` tier=`{row.get('tier')}` "
            f"status=`{row.get('plan_status')}` priority=`{row.get('priority_score')}`/{row.get('priority_tier')} "
            f"route_score=`{float(row.get('route_selection_score') or 0.0):.6f}` signal_score=`{int(row.get('signal_score') or 0)}` age_bars=`{int(row.get('signal_age_bars') or 0)}`"
        )
        lines.append(f"  - blocker: {row.get('blocker_detail') or '-'}")
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a prioritized Brooks structure review queue across all current candidates.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    runtime_now = parse_now(args.now)

    execution_plan_path = latest_artifact(review_dir, "brooks_price_action_execution_plan")
    route_report_path = latest_artifact(review_dir, "brooks_price_action_route_report")
    execution_plan = load_json_mapping(execution_plan_path)
    route_report = load_json_mapping(route_report_path)

    queue: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in list(execution_plan.get("plan_items") or []):
        if not isinstance(row, dict):
            continue
        queue_row = _queue_row_from_plan_item(row)
        key = (queue_row["symbol"], queue_row["strategy_id"])
        if key in seen:
            continue
        seen.add(key)
        queue.append(queue_row)
    for row in list(route_report.get("current_candidates") or []):
        if not isinstance(row, dict):
            continue
        queue_row = _queue_row_from_route_candidate(row)
        key = (queue_row["symbol"], queue_row["strategy_id"])
        if key in seen:
            continue
        seen.add(key)
        queue.append(queue_row)

    queue.sort(key=_queue_sort_key)
    for idx, row in enumerate(queue, start=1):
        row["rank"] = idx

    head = dict(queue[0]) if queue else {}
    status = "inactive"
    brief = "inactive:-"
    blocker_detail = "No Brooks structure review item is currently active."
    done_when = "a fresh Brooks route candidate appears with an execution plan item"
    if head:
        head_tier = str(head.get("tier") or "").strip()
        if head_tier == "review_queue_now":
            status = "ready"
        elif head_tier == "blocked_queue":
            status = "blocked"
        elif head_tier == "route_candidate_only":
            status = "route_candidate_only"
        else:
            status = "informational"
        brief = f"{status}:{head.get('symbol') or '-'}:{head.get('strategy_id') or '-'}:{head.get('plan_status') or '-'}"
        blocker_detail = str(head.get("blocker_detail") or "").strip() or blocker_detail
        done_when = str(head.get("done_when") or "").strip() or done_when

    actionable_count = sum(1 for row in queue if str(row.get("plan_status") or "").strip() == "manual_structure_review_now")
    blocked_count = sum(1 for row in queue if str(row.get("plan_status") or "").strip() == "blocked_shortline_gate")
    route_candidate_only_count = sum(1 for row in queue if str(row.get("plan_status") or "").strip() == "route_candidate_only")
    queue_brief = " | ".join(
        [
            f"{int(row.get('rank') or 0)}:{str(row.get('symbol') or '-')}"
            f":{str(row.get('tier') or '-')}"
            f":{str(row.get('plan_status') or '-')}"
            f":{int(row.get('priority_score') or 0)}"
            for row in queue[:6]
        ]
    ) or "-"
    priority_status = "ready" if queue else "inactive"
    priority_brief = (
        f"ready:{str(head.get('symbol') or '-')}:{int(head.get('priority_score') or 0)}:{str(head.get('priority_tier') or '-')}"
        if queue
        else "inactive:-"
    )
    payload: dict[str, Any] = {
        "action": "build_brooks_structure_review_queue",
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "route_report_artifact": str(route_report_path),
        "execution_plan_artifact": str(execution_plan_path),
        "review_status": status,
        "review_brief": brief,
        "queue_status": "ready" if queue else "inactive",
        "queue_count": len(queue),
        "queue_brief": queue_brief,
        "queue": queue,
        "head": head,
        "priority_status": priority_status,
        "priority_brief": priority_brief,
        "actionable_count": actionable_count,
        "blocked_count": blocked_count,
        "route_candidate_only_count": route_candidate_only_count,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
    }

    review_dir.mkdir(parents=True, exist_ok=True)
    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_brooks_structure_review_queue.json"
    markdown_path = review_dir / f"{stamp}_brooks_structure_review_queue.md"
    checksum_path = review_dir / f"{stamp}_brooks_structure_review_queue_checksum.json"
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
        stem="brooks_structure_review_queue",
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
