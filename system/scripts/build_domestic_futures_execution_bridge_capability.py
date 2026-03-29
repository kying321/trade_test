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
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


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
    for path in survivors[max(1, int(keep)) :]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def latest_execution_plan_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_brooks_price_action_execution_plan.json"))
    if not candidates:
        raise FileNotFoundError("no_brooks_price_action_execution_plan_artifact")
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def _list_text(values: list[str], limit: int = 6) -> str:
    items = [str(v).strip() for v in values if str(v).strip()]
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" (+{len(items) - limit})"


def bridge_stage_counts(capabilities: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "research_only": 0,
        "paper_only": 0,
        "manual_only": 0,
        "guarded_canary": 0,
        "executable": 0,
    }
    for row in capabilities:
        stage = str(row.get("bridge_stage") or "").strip()
        if stage in counts:
            counts[stage] += 1
    return counts


def _normalize_future_capability(row: dict[str, Any], *, owner_artifact: Path, as_of: str | None) -> dict[str, Any]:
    symbol = str(row.get("symbol") or "").strip().upper()
    plan_status = str(row.get("plan_status") or "").strip()
    execution_action = str(row.get("execution_action") or "").strip()
    route_bridge_status = str(row.get("route_bridge_status") or "").strip()

    if plan_status == "manual_structure_review_now":
        bridge_stage = "manual_only"
        account_scope = "manual"
        adapter_kind = "manual"
        allowed_actions = ["review_manual_stop_entry", "queue_review", "source_refresh"]
        blocked_actions = ["auto_route", "auto_send", "live_promotion"]
        blocker_code = "no_automated_execution_bridge"
        blocker_detail = str(
            row.get("plan_blocker_detail")
            or "Structure route is valid, but this asset class has no automated execution bridge in-system."
        ).strip()
        execution_truth_source = "manual_confirmation"
        promotion_gates = [
            "paper_bridge_artifact_ready",
            "venue_account_adapter_explicit",
            "execution_truth_reconcile_available",
        ]
    else:
        bridge_stage = "research_only"
        account_scope = "research"
        adapter_kind = "none"
        allowed_actions = ["research_refresh", "watch_only", "source_rebuild"]
        blocked_actions = ["paper_apply", "auto_route", "auto_send", "live_promotion"]
        blocker_code = route_bridge_status or "bridge_stage_unresolved"
        blocker_detail = str(row.get("plan_blocker_detail") or "domestic futures bridge stage unresolved at source").strip()
        execution_truth_source = "research_artifact"
        promotion_gates = ["route_structure_ready", "bridge_stage_explicit"]

    return {
        "symbol": symbol,
        "asset_class": "future",
        "source_family": "brooks_structure",
        "venue": str(row.get("venue") or "").strip(),
        "account_scope": account_scope,
        "adapter_kind": adapter_kind,
        "bridge_stage": bridge_stage,
        "allowed_actions": allowed_actions,
        "blocked_actions": blocked_actions,
        "blocker_code": blocker_code,
        "blocker_detail": blocker_detail,
        "promotion_gates": promotion_gates,
        "demotion_triggers": [
            "source_of_truth_ambiguity",
            "execution_truth_source_missing",
            "adapter_unavailable",
        ],
        "execution_truth_source": execution_truth_source,
        "owner_artifact": str(owner_artifact),
        "as_of": as_of,
        "evidence_refs": [str(owner_artifact)],
        "strategy_id": str(row.get("strategy_id") or "").strip(),
        "direction": str(row.get("direction") or "").strip().upper(),
        "signal_ts": row.get("signal_ts"),
        "signal_age_bars": int(row.get("signal_age_bars") or 0),
        "route_selection_score": float(row.get("route_selection_score") or 0.0),
        "signal_score": int(row.get("signal_score") or 0),
        "consumer_mappings": {
            "plan_status": plan_status,
            "execution_action": execution_action,
            "route_bridge_status": route_bridge_status,
        },
    }


def build_capabilities(execution_plan: dict[str, Any], *, owner_artifact: Path) -> list[dict[str, Any]]:
    as_of = str(execution_plan.get("as_of") or "").strip() or None
    capabilities = [
        _normalize_future_capability(row, owner_artifact=owner_artifact, as_of=as_of)
        for row in execution_plan.get("plan_items", [])
        if isinstance(row, dict) and str(row.get("asset_class") or "").strip().lower() == "future"
    ]
    capabilities.sort(
        key=lambda row: (
            0 if str(row.get("bridge_stage") or "") == "manual_only" else 1,
            -float(row.get("route_selection_score") or 0.0),
            -int(row.get("signal_score") or 0),
            int(row.get("signal_age_bars") or 0),
            str(row.get("symbol") or ""),
        )
    )
    return capabilities


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Domestic Futures Execution Bridge Capability",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_execution_plan_artifact: `{payload.get('source_execution_plan_artifact') or ''}`",
        f"- capability_count: `{payload.get('capability_count', 0)}`",
        f"- head_symbol: `{payload.get('head_symbol') or '-'}`",
        (
            "- bridge_stage_counts: "
            f"`research_only={payload.get('bridge_stage_counts', {}).get('research_only', 0)} "
            f"paper_only={payload.get('bridge_stage_counts', {}).get('paper_only', 0)} "
            f"manual_only={payload.get('bridge_stage_counts', {}).get('manual_only', 0)} "
            f"guarded_canary={payload.get('bridge_stage_counts', {}).get('guarded_canary', 0)} "
            f"executable={payload.get('bridge_stage_counts', {}).get('executable', 0)}`"
        ),
        "",
        "## Summary",
    ]
    for line in payload.get("summary_lines", []):
        lines.append(f"- {line}")
    lines.extend(["", "## Capabilities"])
    for row in payload.get("capabilities", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('symbol')}` stage=`{row.get('bridge_stage')}` action=`{row.get('consumer_mappings', {}).get('execution_action') or '-'}` "
            f"blocker=`{row.get('blocker_code')}` truth=`{row.get('execution_truth_source')}`"
        )
        lines.append(f"  - detail: {row.get('blocker_detail') or '-'}")
        lines.append(f"  - allowed: {_list_text(row.get('allowed_actions', []), limit=10)}")
        lines.append(f"  - blocked: {_list_text(row.get('blocked_actions', []), limit=10)}")
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a source-owned domestic futures execution bridge capability artifact from current execution-plan evidence."
    )
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

    execution_plan_path = latest_execution_plan_source(review_dir, runtime_now)
    execution_plan = json.loads(execution_plan_path.read_text(encoding="utf-8"))
    capabilities = build_capabilities(execution_plan, owner_artifact=execution_plan_path)
    counts = bridge_stage_counts(capabilities)
    head_symbol = str(capabilities[0].get("symbol") or "") if capabilities else ""
    payload = {
        "action": "build_domestic_futures_execution_bridge_capability",
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_execution_plan_artifact": str(execution_plan_path),
        "capability_count": len(capabilities),
        "head_symbol": head_symbol,
        "bridge_stage_counts": counts,
        "capabilities": capabilities,
        "summary_lines": [
            f"head-symbol: {head_symbol or '-'}",
            f"manual-only-count: {counts['manual_only']}",
            f"research-only-count: {counts['research_only']}",
            f"future-symbols: {_list_text([str(row.get('symbol') or '') for row in capabilities], limit=10)}",
        ],
    }

    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_domestic_futures_execution_bridge_capability.json"
    md_path = review_dir / f"{stamp}_domestic_futures_execution_bridge_capability.md"
    checksum_path = review_dir / f"{stamp}_domestic_futures_execution_bridge_capability_checksum.json"
    payload["artifact"] = str(json_path)
    payload["markdown"] = str(md_path)
    payload["checksum"] = str(checksum_path)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "artifact": str(json_path),
        "markdown": str(md_path),
        "sha256": sha256_file(json_path),
        "generated_at_utc": payload["as_of"],
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="domestic_futures_execution_bridge_capability",
        current_paths=[json_path, md_path, checksum_path],
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=runtime_now,
    )
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_payload["sha256"] = sha256_file(json_path)
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
