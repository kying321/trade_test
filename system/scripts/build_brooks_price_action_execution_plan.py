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


def classify_plan_item(row: dict[str, Any]) -> dict[str, Any]:
    bridge_status = str(row.get("route_bridge_status") or "")
    symbol = str(row.get("symbol") or "").strip().upper()
    asset_class = str(row.get("asset_class") or "").strip().lower()
    direction = str(row.get("direction") or "").strip().upper()
    if bridge_status == "manual_structure_route":
        plan_status = "manual_structure_review_now"
        execution_action = "review_manual_stop_entry"
        blocker_detail = "Structure route is valid, but this asset class has no automated execution bridge in-system."
        done_when = "manual trader confirms venue, sizing, and lower-timeframe trigger before placing a discretionary order"
    elif bridge_status == "blocked_shortline_gate":
        plan_status = "blocked_shortline_gate"
        execution_action = "wait_for_shortline_setup_ready"
        blocker_detail = str(row.get("route_bridge_blocker_detail") or "shortline gate blocked")
        done_when = str(row.get("route_bridge_done_when") or "shortline gate stack completes")
    else:
        plan_status = "informational_only"
        execution_action = "watch_only"
        blocker_detail = str(row.get("route_bridge_blocker_detail") or "no execution bridge")
        done_when = str(row.get("route_bridge_done_when") or "execution bridge becomes explicit")
    risk_per_unit = float(row.get("risk_per_unit") or 0.0)
    entry_price = float(row.get("entry_price") or 0.0)
    stop_price = float(row.get("stop_price") or 0.0)
    target_price = float(row.get("target_price") or 0.0)
    stop_gap_pct = 0.0
    target_gap_pct = 0.0
    if entry_price > 0.0:
        stop_gap_pct = abs(entry_price - stop_price) / entry_price * 100.0
        target_gap_pct = abs(target_price - entry_price) / entry_price * 100.0
    return {
        "symbol": symbol,
        "asset_class": asset_class,
        "direction": direction,
        "strategy_id": str(row.get("strategy_id") or ""),
        "strategy_label": str(row.get("strategy_label") or ""),
        "signal_ts": row.get("signal_ts"),
        "signal_age_bars": int(row.get("signal_age_bars") or 0),
        "route_selection_score": float(row.get("route_selection_score") or 0.0),
        "signal_score": int(row.get("signal_score") or 0),
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "risk_per_unit": risk_per_unit,
        "rr_ratio": float(row.get("rr_ratio") or 0.0),
        "stop_gap_pct": float(stop_gap_pct),
        "target_gap_pct": float(target_gap_pct),
        "execution_action": execution_action,
        "plan_status": plan_status,
        "plan_blocker_detail": blocker_detail,
        "plan_done_when": done_when,
        "route_bridge_status": bridge_status,
        "note": str(row.get("note") or ""),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Brooks Price Action Execution Plan",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- route_report_artifact: `{payload.get('route_report_artifact') or ''}`",
        f"- actionable_count: `{payload.get('actionable_count', 0)}`",
        f"- blocked_count: `{payload.get('blocked_count', 0)}`",
        "",
        "## Head Plan",
    ]
    head = payload.get("head_plan_item") or {}
    if head:
        lines.extend(
            [
                f"- `{head.get('symbol')}` `{head.get('strategy_id')}` `{head.get('direction')}` status=`{head.get('plan_status')}` action=`{head.get('execution_action')}`",
                f"- entry=`{float(head.get('entry_price') or 0.0):.4f}` stop=`{float(head.get('stop_price') or 0.0):.4f}` target=`{float(head.get('target_price') or 0.0):.4f}` rr=`{float(head.get('rr_ratio') or 0.0):.2f}`",
                f"- blocker: {head.get('plan_blocker_detail')}",
            ]
        )
    lines.extend(["", "## Plan Items"])
    for row in payload.get("plan_items", [])[:15]:
        lines.append(
            f"- `{row.get('symbol')}` asset=`{row.get('asset_class')}` strategy=`{row.get('strategy_id')}` direction=`{row.get('direction')}` status=`{row.get('plan_status')}` action=`{row.get('execution_action')}` rr=`{float(row.get('rr_ratio') or 0.0):.2f}` age_bars=`{row.get('signal_age_bars')}`"
        )
        lines.append(f"  - blocker: {row.get('plan_blocker_detail')}")
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Brooks route candidates into standardized execution plan items.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    runtime_now = parse_now(args.now)

    route_report_path = latest_artifact(review_dir, "brooks_price_action_route_report")
    route_report = load_json_mapping(route_report_path)
    plan_items = [
        classify_plan_item(row)
        for row in list(route_report.get("current_candidates") or [])
        if isinstance(row, dict)
    ]
    plan_items.sort(
        key=lambda row: (
            0 if row["plan_status"] == "manual_structure_review_now" else 1,
            -float(row["route_selection_score"]),
            -int(row["signal_score"]),
            int(row["signal_age_bars"]),
            row["symbol"],
        )
    )
    actionable_count = sum(1 for row in plan_items if row["plan_status"] == "manual_structure_review_now")
    blocked_count = sum(1 for row in plan_items if row["plan_status"] != "manual_structure_review_now")
    payload: dict[str, Any] = {
        "action": "build_brooks_price_action_execution_plan",
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "route_report_artifact": str(route_report_path),
        "actionable_count": int(actionable_count),
        "blocked_count": int(blocked_count),
        "plan_items": plan_items,
        "head_plan_item": plan_items[0] if plan_items else {},
    }

    review_dir.mkdir(parents=True, exist_ok=True)
    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_brooks_price_action_execution_plan.json"
    markdown_path = review_dir / f"{stamp}_brooks_price_action_execution_plan.md"
    checksum_path = review_dir / f"{stamp}_brooks_price_action_execution_plan_checksum.json"
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
        stem="brooks_price_action_execution_plan",
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
