#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_mapping(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(
        review_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def resolve_queue_profile_source(
    review_dir: Path,
    explicit_path: str,
) -> tuple[Path | None, dict[str, Any] | None, dict[str, Any] | None]:
    if str(explicit_path).strip():
        queue_path = Path(explicit_path).expanduser().resolve()
        queue_payload = load_json_mapping(queue_path)
        queue_profile = (
            queue_payload.get("crypto_cvd_queue_profile", {})
            if isinstance(queue_payload, dict)
            else None
        )
        return queue_path, queue_payload, queue_profile if isinstance(queue_profile, dict) else None

    queue_path = find_latest(review_dir, "*_crypto_cvd_queue_profile.json")
    queue_payload = load_json_mapping(queue_path)
    queue_profile = (
        queue_payload.get("crypto_cvd_queue_profile", {})
        if isinstance(queue_payload, dict)
        else None
    )
    if isinstance(queue_profile, dict) and queue_profile:
        return queue_path, queue_payload, queue_profile

    hot_path = find_latest(review_dir, "*_hot_universe_research.json")
    hot_payload = load_json_mapping(hot_path)
    hot_profile = (
        hot_payload.get("crypto_cvd_queue_profile", {})
        if isinstance(hot_payload, dict)
        else None
    )
    if isinstance(hot_profile, dict) and hot_profile:
        return hot_path, hot_payload, hot_profile

    return queue_path, queue_payload, queue_profile if isinstance(queue_profile, dict) else None


def prune_artifacts(
    review_dir: Path,
    *,
    current_artifact: Path,
    current_markdown: Path,
    current_checksum: Path,
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    review_dir.mkdir(parents=True, exist_ok=True)
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, ttl_hours))
    protected = {current_artifact.name, current_markdown.name, current_checksum.name}
    candidates: list[Path] = []
    for pattern in (
        "*_crypto_cvd_queue_handoff.json",
        "*_crypto_cvd_queue_handoff.md",
        "*_crypto_cvd_queue_handoff_checksum.json",
    ):
        candidates.extend(review_dir.glob(pattern))

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
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


def summarize_batch_runtime(batch_row: dict[str, Any], symbol_state: dict[str, dict[str, Any]]) -> dict[str, Any]:
    batch = str(batch_row.get("batch", "")).strip()
    eligible_symbols = [str(x).strip().upper() for x in batch_row.get("cvd_eligible_symbols", []) if str(x).strip()]
    matching: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    locality_counts: dict[str, int] = {}
    attack_side_counts: dict[str, int] = {}
    drift_risk_symbols: list[str] = []
    for symbol in eligible_symbols:
        row = symbol_state.get(symbol)
        if not row:
            continue
        klass = str(row.get("classification", "unclear")).strip() or "unclear"
        locality = str(row.get("cvd_locality_status", "unknown")).strip() or "unknown"
        attack_side = str(row.get("cvd_attack_side", "unknown")).strip() or "unknown"
        class_counts[klass] = int(class_counts.get(klass, 0)) + 1
        locality_counts[locality] = int(locality_counts.get(locality, 0)) + 1
        attack_side_counts[attack_side] = int(attack_side_counts.get(attack_side, 0)) + 1
        if bool(row.get("cvd_drift_risk", False)):
            drift_risk_symbols.append(symbol)
        matching.append(
            {
                "symbol": symbol,
                "classification": klass,
                "cvd_context_mode": str(row.get("cvd_context_mode", "unclear")),
                "cvd_trust_tier_hint": str(row.get("cvd_trust_tier_hint", "unavailable")),
                "cvd_veto_hint": str(row.get("cvd_veto_hint", "")),
                "cvd_locality_status": locality,
                "cvd_drift_risk": bool(row.get("cvd_drift_risk", False)),
                "cvd_attack_side": attack_side,
                "active_reasons": list(row.get("active_reasons", [])) if isinstance(row.get("active_reasons"), list) else [],
            }
        )

    queue_mode = str(batch_row.get("queue_mode", "")).strip()
    runtime_status = "no_live_symbols"
    runtime_note = "No overlapping symbols were present in the latest micro snapshot."
    if matching:
        if len(drift_risk_symbols) == len(matching):
            runtime_status = "local_window_drift_risk"
            runtime_note = "All overlapping symbols fail the local CVD window or drift guard."
        elif class_counts.get("watch_only", 0) == len(matching):
            runtime_status = "watch_only"
            runtime_note = "All overlapping symbols are downgraded to watch-only in the latest micro snapshot."
        elif queue_mode == "trend_confirmation" and class_counts.get("trend_confirmation_watch", 0) > 0:
            runtime_status = "aligned"
            runtime_note = "Latest micro snapshot has trend-confirmation support for this queue."
        elif queue_mode == "reversal_absorption_watch" and class_counts.get("reversal_absorption_watch", 0) > 0:
            runtime_status = "aligned"
            runtime_note = "Latest micro snapshot has reversal/absorption support for this queue."
        else:
            runtime_status = "mixed"
            runtime_note = "Latest micro snapshot is mixed relative to the queue's preferred contexts."

    return {
        "batch": batch,
        "status_label": str(batch_row.get("status_label", "")),
        "queue_mode": queue_mode,
        "trust_requirement": str(batch_row.get("trust_requirement", "")),
        "dominant_regime": str(batch_row.get("dominant_regime", "")),
        "leader_symbols": list(batch_row.get("leader_symbols", [])) if isinstance(batch_row.get("leader_symbols"), list) else [],
        "preferred_contexts": list(batch_row.get("preferred_contexts", [])) if isinstance(batch_row.get("preferred_contexts"), list) else [],
        "veto_biases": list(batch_row.get("veto_biases", [])) if isinstance(batch_row.get("veto_biases"), list) else [],
        "eligible_symbols": eligible_symbols,
        "matching_symbols": matching,
        "classification_counts": class_counts,
        "locality_counts": locality_counts,
        "attack_side_counts": attack_side_counts,
        "drift_risk_symbols": drift_risk_symbols,
        "runtime_status": runtime_status,
        "runtime_note": runtime_note,
    }


def runtime_priority_score(row: dict[str, Any]) -> float:
    runtime_status = str(row.get("runtime_status", "")).strip()
    queue_mode = str(row.get("queue_mode", "")).strip()
    trust_requirement = str(row.get("trust_requirement", "")).strip()
    eligible_count = len(row.get("eligible_symbols", [])) if isinstance(row.get("eligible_symbols"), list) else 0
    matching_count = len(row.get("matching_symbols", [])) if isinstance(row.get("matching_symbols"), list) else 0

    score = 0.0
    score += {
        "aligned": 4.0,
        "mixed": 2.0,
        "watch_only": 1.0,
        "local_window_drift_risk": 0.75,
        "no_live_symbols": 0.5,
    }.get(runtime_status, 0.0)
    score += {
        "trend_confirmation": 1.0,
        "reversal_absorption_watch": 0.8,
        "mixed_bridge_filter": 0.3,
    }.get(queue_mode, 0.0)
    score += {
        "dual_leader_alignment": 0.6,
        "basket_consensus": 0.5,
        "leader_plus_index_alignment": 0.3,
    }.get(trust_requirement, 0.0)
    score += min(1.0, 0.1 * float(eligible_count))
    score += min(0.5, 0.1 * float(matching_count))
    return round(score, 4)


def classify_queue_action(row: dict[str, Any]) -> tuple[str, str]:
    runtime_status = str(row.get("runtime_status", "")).strip()
    queue_mode = str(row.get("queue_mode", "")).strip()
    if runtime_status == "aligned":
        if queue_mode == "trend_confirmation":
            return "inspect_first", "Latest micro snapshot aligns with a trend-confirmation queue."
        return "inspect_first", "Latest micro snapshot aligns with a reversal/absorption queue."
    if runtime_status == "local_window_drift_risk":
        return "defer_until_local_cvd_recovers", "Overlapping symbols only show stale or drift-risk CVD context near the queue."
    if runtime_status == "mixed":
        return "review_after_refresh", "Queue remains valid, but the latest micro snapshot is mixed."
    if runtime_status == "watch_only":
        return "defer_until_micro_recovers", "All overlapping symbols are watch-only in the latest micro snapshot."
    return "defer_no_overlap", "No overlapping symbols were present in the latest micro snapshot."


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Crypto CVD Queue Handoff",
        "",
        f"- queue source: `{payload.get('queue_source_artifact') or ''}`",
        f"- semantic source: `{payload.get('semantic_source_artifact') or ''}`",
        f"- queue state: `{payload.get('queue_status') or ''}`",
        f"- semantic state: `{payload.get('semantic_status') or ''}`",
        f"- operator status: `{payload.get('operator_status') or ''}`",
        f"- takeaway: {payload.get('takeaway') or ''}",
        "",
        "## Priority",
        f"- priority batches: `{', '.join(payload.get('priority_batches', [])) or '-'}`",
        f"- ready now: `{', '.join(payload.get('ready_now_batches', [])) or '-'}`",
        f"- watch only: `{', '.join(payload.get('watch_only_batches', [])) or '-'}`",
        f"- next focus batch: `{payload.get('next_focus_batch') or '-'}`",
        f"- next focus action: `{payload.get('next_focus_action') or '-'}`",
        f"- queue stack: `{payload.get('queue_stack_brief') or '-'}`",
        "",
        "## Runtime Queue",
    ]
    for row in payload.get("runtime_queue", []):
        if not isinstance(row, dict):
            continue
        lines.extend(
            [
                f"- `{row.get('batch')}`",
                f"  - rank: `{row.get('queue_rank')}`",
                f"  - score: `{row.get('priority_score')}`",
                f"  - action: `{row.get('queue_action')}`",
                f"  - reason: `{row.get('queue_action_reason')}`",
            ]
        )
    lines.append("")
    lines.append("## Batch Runtime")
    for row in payload.get("batch_runtime_profiles", []):
        if not isinstance(row, dict):
            continue
        lines.extend(
            [
                f"- `{row.get('batch')}`",
                f"  - queue: `{row.get('queue_mode')}`",
                f"  - trust: `{row.get('trust_requirement')}`",
                f"  - dominant_regime: `{row.get('dominant_regime')}`",
                f"  - runtime_status: `{row.get('runtime_status')}`",
                f"  - runtime_note: `{row.get('runtime_note')}`",
                f"  - leaders: `{', '.join(row.get('leader_symbols', [])) or '-'}`",
                f"  - matching: `{', '.join(m.get('symbol', '') for m in row.get('matching_symbols', [])) or '-'}`",
                f"  - localities: `{json.dumps(row.get('locality_counts', {}), ensure_ascii=False, sort_keys=True)}`",
                f"  - attacks: `{json.dumps(row.get('attack_side_counts', {}), ensure_ascii=False, sort_keys=True)}`",
                f"  - drift_risk_symbols: `{', '.join(row.get('drift_risk_symbols', [])) or '-'}`",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge crypto CVD queue profile with the latest micro semantic snapshot.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--queue-profile-file", default="")
    parser.add_argument("--semantic-snapshot-file", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=16)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    now_ts = dt.datetime.fromisoformat(args.now) if str(args.now).strip() else now_utc()

    queue_path, queue_payload, queue_profile = resolve_queue_profile_source(review_dir, args.queue_profile_file)
    semantic_path = (
        Path(args.semantic_snapshot_file).expanduser().resolve()
        if str(args.semantic_snapshot_file).strip()
        else find_latest(review_dir, "*_crypto_cvd_semantic_snapshot.json")
    )

    semantic_payload = load_json_mapping(semantic_path)

    ok = True
    status = "ok"
    if not isinstance(queue_payload, dict):
        ok = False
        status = "queue_profile_missing"
    else:
        if not isinstance(queue_profile, dict) or not queue_profile:
            ok = False
            status = "queue_profile_invalid"

    if ok and not isinstance(semantic_payload, dict):
        ok = False
        status = "semantic_snapshot_missing"
    elif ok and not bool(semantic_payload.get("ok", False)):
        ok = False
        status = "semantic_snapshot_invalid"

    symbol_state = {
        str(row.get("symbol", "")).strip().upper(): row
        for row in (semantic_payload.get("symbols", []) if isinstance(semantic_payload, dict) else [])
        if isinstance(row, dict) and str(row.get("symbol", "")).strip()
    }

    batch_runtime_profiles: list[dict[str, Any]] = []
    priority_batches = list(queue_profile.get("priority_batches", [])) if isinstance(queue_profile, dict) else []
    for row in queue_profile.get("batch_profiles", []) if isinstance(queue_profile, dict) else []:
        if not isinstance(row, dict):
            continue
        batch_runtime_profiles.append(summarize_batch_runtime(row, symbol_state))

    ready_now_batches = [str(row.get("batch", "")) for row in batch_runtime_profiles if str(row.get("runtime_status", "")) == "aligned"]
    watch_only_batches = [str(row.get("batch", "")) for row in batch_runtime_profiles if str(row.get("runtime_status", "")) == "watch_only"]
    drift_risk_batches = [
        str(row.get("batch", ""))
        for row in batch_runtime_profiles
        if str(row.get("runtime_status", "")) == "local_window_drift_risk"
    ]
    mixed_batches = [str(row.get("batch", "")) for row in batch_runtime_profiles if str(row.get("runtime_status", "")) == "mixed"]
    runtime_queue: list[dict[str, Any]] = []
    for row in batch_runtime_profiles:
        queue_action, queue_action_reason = classify_queue_action(row)
        queue_row = dict(row)
        queue_row["priority_score"] = runtime_priority_score(row)
        queue_row["queue_action"] = queue_action
        queue_row["queue_action_reason"] = queue_action_reason
        runtime_queue.append(queue_row)
    runtime_queue.sort(
        key=lambda row: (
            -float(row.get("priority_score", 0.0) or 0.0),
            str(row.get("batch", "")),
        )
    )
    for idx, row in enumerate(runtime_queue, start=1):
        row["queue_rank"] = idx

    deferred_batches = [
        str(row.get("batch", ""))
        for row in runtime_queue
        if str(row.get("queue_action", "")).startswith("defer")
    ]
    next_focus_batch = str((runtime_queue[0] if runtime_queue else {}).get("batch", ""))
    next_focus_action = str((runtime_queue[0] if runtime_queue else {}).get("queue_action", ""))
    next_focus_reason = str((runtime_queue[0] if runtime_queue else {}).get("queue_action_reason", ""))
    queue_stack_brief = " -> ".join(str(row.get("batch", "")) for row in runtime_queue[:3] if str(row.get("batch", "")).strip())

    if not ok:
        takeaway = "Crypto CVD queue handoff is incomplete because one of the source artifacts is missing."
        operator_status = status
    elif ready_now_batches:
        takeaway = "Use the listed ready-now batches as the first crypto queue to inspect; the latest micro snapshot aligns with their preferred contexts."
        operator_status = "queue-ready"
    elif drift_risk_batches:
        takeaway = "Queue priorities remain valid, but the latest micro snapshot is stale near the key level; wait for a fresh local CVD window before using it."
        operator_status = "queue-local-cvd-drift-risk"
    elif watch_only_batches:
        takeaway = "Queue priorities remain valid, but the latest micro snapshot downgrades all overlapping crypto symbols to watch-only."
        operator_status = "queue-watch-only"
    elif mixed_batches:
        takeaway = "Queue priorities remain valid, but the latest micro snapshot is mixed; keep CVD-lite as advisory only."
        operator_status = "queue-mixed"
    else:
        takeaway = "Queue priorities exist, but the latest micro snapshot has no overlapping symbols."
        operator_status = "queue-no-live-overlap"

    out: dict[str, Any] = {
        "action": "build_crypto_cvd_queue_handoff",
        "ok": ok,
        "status": status,
        "generated_at": fmt_utc(now_ts),
        "queue_source_artifact": None if queue_path is None else str(queue_path),
        "semantic_source_artifact": None if semantic_path is None else str(semantic_path),
        "queue_status": str(queue_payload.get("status", "")) if isinstance(queue_payload, dict) else None,
        "semantic_status": str(semantic_payload.get("source_status", "")) if isinstance(semantic_payload, dict) else None,
        "operator_status": operator_status,
        "priority_batches": priority_batches,
        "ready_now_batches": ready_now_batches,
        "drift_risk_batches": drift_risk_batches,
        "watch_only_batches": watch_only_batches,
        "mixed_batches": mixed_batches,
        "deferred_batches": deferred_batches,
        "focus_stack_brief": "queue_priority -> live_semantics -> watch_only_gate",
        "queue_stack_brief": queue_stack_brief,
        "next_focus_batch": next_focus_batch,
        "next_focus_action": next_focus_action,
        "next_focus_reason": next_focus_reason,
        "queue_overall_takeaway": str(queue_profile.get("overall_takeaway", "")) if isinstance(queue_profile, dict) else "",
        "semantic_takeaway": str(semantic_payload.get("takeaway", "")) if isinstance(semantic_payload, dict) else "",
        "takeaway": takeaway,
        "batch_runtime_profiles": batch_runtime_profiles,
        "runtime_queue": runtime_queue,
        "artifact_status_label": "crypto-cvd-queue-handoff-ok" if ok else status,
        "artifact_label": None,
        "artifact_tags": [],
        "artifact": None,
        "markdown": None,
        "checksum": None,
        "pruned_keep": [],
        "pruned_age": [],
    }

    suffix = str(out.get("operator_status") or status)
    out["artifact_label"] = f"crypto-cvd-queue-handoff:{suffix}"
    out["artifact_tags"] = ["crypto-cvd", "queue-handoff", suffix]

    review_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_ts.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_crypto_cvd_queue_handoff.json"
    markdown_path = review_dir / f"{stamp}_crypto_cvd_queue_handoff.md"
    checksum_path = review_dir / f"{stamp}_crypto_cvd_queue_handoff_checksum.json"
    out["artifact"] = str(artifact_path)
    out["markdown"] = str(markdown_path)
    out["checksum"] = str(checksum_path)

    artifact_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(out), encoding="utf-8")
    checksum_path.write_text(
        json.dumps(
            {
                "generated_at": fmt_utc(now_ts),
                "artifact_ttl_hours": max(1.0, float(args.artifact_ttl_hours)),
                "files": [
                    {
                        "path": str(artifact_path),
                        "sha256": sha256_file(artifact_path),
                        "size_bytes": int(artifact_path.stat().st_size),
                    },
                    {
                        "path": str(markdown_path),
                        "sha256": sha256_file(markdown_path),
                        "size_bytes": int(markdown_path.stat().st_size),
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_artifact=artifact_path,
        current_markdown=markdown_path,
        current_checksum=checksum_path,
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )
    out["pruned_keep"] = pruned_keep
    out["pruned_age"] = pruned_age
    artifact_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
