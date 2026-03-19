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
COMBO_CANONICAL_ALIASES = {
    "ad_breakout": "cvd_breakout",
    "ad_rsi_breakout": "cvd_rsi_breakout",
    "ad_rsi_vol_breakout": "cvd_rsi_vol_breakout",
    "ad_rsi_reclaim": "cvd_rsi_reclaim",
    "taker_oi_ad_breakout": "taker_oi_cvd_breakout",
    "taker_oi_ad_rsi_breakout": "taker_oi_cvd_rsi_breakout",
}


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_combo_id(combo_id: Any) -> str:
    combo = str(combo_id or "")
    return COMBO_CANONICAL_ALIASES.get(combo, combo)


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


def artifact_status_rank(payload: dict[str, Any]) -> int:
    status = str(payload.get("status") or "").strip().lower()
    if status == "ok" or payload.get("ok") is True:
        return 2
    if status:
        return 0
    return 1


def latest_artifact(review_dir: Path, suffix: str, reference_now: dt.datetime | None = None) -> Path:
    candidates = sorted(
        review_dir.glob(f"*_{suffix}.json"),
        key=lambda path: artifact_sort_key(path, reference_now),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"missing_artifact:{suffix}")
    best_ok: Path | None = None
    best_fallback: Path | None = None
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            if best_fallback is None:
                best_fallback = path
            continue
        rank = artifact_status_rank(payload)
        if rank >= 2:
            best_ok = path
            break
        if best_fallback is None and rank >= 1:
            best_fallback = path
    return best_ok or best_fallback or candidates[0]


def latest_crypto_route_refresh(
    review_dir: Path,
    reference_now: dt.datetime | None = None,
) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = list(review_dir.glob("*_crypto_route_refresh.json"))
    if not candidates:
        return None, None
    path = max(candidates, key=lambda item: artifact_sort_key(item, reference_now))
    return path, json.loads(path.read_text(encoding="utf-8"))


def _crypto_route_refresh_audit_lane(
    review_dir: Path,
    reference_now: dt.datetime | None = None,
) -> dict[str, Any]:
    refresh_path, refresh_payload = latest_crypto_route_refresh(review_dir, reference_now)
    if refresh_path is None or refresh_payload is None:
        return {
            "status": "not_available",
            "brief": "not_available",
            "artifact": None,
            "as_of": None,
            "native_mode": None,
            "native_step_count": 0,
            "reused_native_count": 0,
            "missing_reused_count": 0,
            "note": "no crypto_route_refresh artifact is available yet",
            "done_when": "run refresh_crypto_route_state to record refresh audit",
        }

    native_mode = str(refresh_payload.get("native_refresh_mode") or "").strip() or None
    steps = list(refresh_payload.get("steps") or [])
    native_steps = [step for step in steps if str(step.get("name") or "").startswith("native_")]
    reused_native_count = sum(
        1 for step in native_steps if str(step.get("status") or "").strip() == "reused_previous_artifact"
    )
    native_step_count = len(native_steps)
    missing_reused_count = max(0, native_step_count - reused_native_count)
    if native_step_count == 0:
        status = "native_audit_unavailable"
        brief = "native_audit_unavailable"
    elif reused_native_count == native_step_count:
        status = "reused_native_inputs"
        brief = f"reused_native_inputs:{native_mode or 'unknown'}:{reused_native_count}/{native_step_count}"
    elif reused_native_count > 0:
        status = "mixed_native_inputs"
        brief = f"mixed_native_inputs:{native_mode or 'unknown'}:{reused_native_count}/{native_step_count}"
    else:
        status = "fresh_native_inputs"
        brief = f"fresh_native_inputs:{native_mode or 'unknown'}:{reused_native_count}/{native_step_count}"

    return {
        "status": status,
        "brief": brief,
        "artifact": str(refresh_path),
        "as_of": refresh_payload.get("as_of"),
        "native_mode": native_mode,
        "native_step_count": native_step_count,
        "reused_native_count": reused_native_count,
        "missing_reused_count": missing_reused_count,
        "note": f"latest crypto_route_refresh reports {reused_native_count}/{native_step_count} native steps reused.",
        "done_when": "run full native refresh only when fresh native recomputation is required",
    }


def _crypto_route_refresh_reuse_gate(audit: dict[str, Any]) -> dict[str, Any]:
    status = str(audit.get("status") or "").strip()
    brief = str(audit.get("brief") or "").strip()
    native_mode = str(audit.get("native_mode") or "").strip() or "unknown"
    reused_native_count = int(audit.get("reused_native_count") or 0)
    native_step_count = int(audit.get("native_step_count") or 0)
    if status == "reused_native_inputs":
        return {
            "level": "informational",
            "status": "reuse_non_blocking",
            "brief": f"reuse_non_blocking:{native_mode}:{reused_native_count}/{native_step_count}",
            "blocking": False,
            "blocker_detail": "latest crypto_route_refresh reused all tracked native steps; current route/reporting reuse is safe enough for operator consumption.",
            "done_when": "run full native refresh only when fresh native recomputation is explicitly required",
        }
    if status == "fresh_native_inputs":
        return {
            "level": "informational",
            "status": "fresh_non_blocking",
            "brief": f"fresh_non_blocking:{native_mode}:{reused_native_count}/{native_step_count}",
            "blocking": False,
            "blocker_detail": "latest crypto_route_refresh used fresh native inputs across tracked native steps.",
            "done_when": "keep using current fresh native inputs until the next required recomputation window",
        }
    if status in {"mixed_native_inputs", "native_audit_unavailable", "not_available"}:
        gate_status = (
            "mixed_requires_full_native_refresh"
            if status == "mixed_native_inputs"
            else "audit_missing_requires_full_native_refresh"
        )
        return {
            "level": "blocking",
            "status": gate_status,
            "brief": f"{gate_status}:{native_mode}:{reused_native_count}/{native_step_count}",
            "blocking": True,
            "blocker_detail": f"latest crypto_route_refresh audit is not fully reusable-safe ({brief or status}); force a full native refresh before trusting route reuse downstream.",
            "done_when": "rerun refresh_crypto_route_state without skip_native_refresh and confirm all native steps are either fresh or intentionally reused end-to-end",
        }
    return {
        "level": "blocking",
        "status": "unknown_requires_review",
        "brief": f"unknown_requires_review:{brief or status or 'unknown'}",
        "blocking": True,
        "blocker_detail": "latest crypto_route_refresh audit returned an unknown reuse status; manual review or a clean full native refresh is required.",
        "done_when": "rerun refresh_crypto_route_state with a known-good native refresh outcome",
    }


def prune_review_artifacts(
    review_dir: Path,
    *,
    stem: str,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, ttl_hours))
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


def taker_status_from_family(family_payload: dict[str, Any]) -> dict[str, Any]:
    ranked = list(family_payload.get("ranked_combos", []))
    discarded = list(family_payload.get("discarded_combos", []))
    taker_ranked = [
        row for row in ranked if str(row.get("combo_id") or "").startswith("taker_oi") or str(row.get("combo_id") or "").startswith("crowding_filtered")
    ]
    taker_discarded = [
        row for row in discarded if str(row.get("combo_id") or "").startswith("taker_oi") or str(row.get("combo_id") or "").startswith("crowding_filtered")
    ]
    if taker_ranked:
        best = taker_ranked[0]
        return {
            "status": "ranked",
            "best_combo": best.get("combo_id"),
            "best_combo_canonical": canonical_combo_id(best.get("combo_id")),
            "best_return": float(best.get("avg_total_return") or 0.0),
            "best_timely_hit_rate": float(best.get("avg_timely_hit_rate") or 0.0),
            "trade_count": int(best.get("trade_count") or 0),
        }
    if taker_discarded:
        best = taker_discarded[0]
        return {
            "status": "discarded",
            "best_combo": best.get("combo_id"),
            "best_combo_canonical": canonical_combo_id(best.get("combo_id")),
            "best_return": float(best.get("avg_total_return") or 0.0),
            "best_timely_hit_rate": float(best.get("avg_timely_hit_rate") or 0.0),
            "trade_count": int(best.get("trade_count") or 0),
        }
    return {
        "status": "absent",
        "best_combo": None,
        "best_combo_canonical": None,
        "best_return": 0.0,
        "best_timely_hit_rate": 0.0,
        "trade_count": 0,
    }


def classify_control_result(etf_payload: dict[str, Any], native_payload: dict[str, Any]) -> dict[str, Any]:
    etf_family = dict(etf_payload.get("crypto_family") or {})
    native_family = dict(native_payload.get("native_crypto_family") or {})

    etf_top = etf_family.get("ranked_combos", [{}])[0] if etf_family.get("ranked_combos") else {}
    native_top = native_family.get("ranked_combos", [{}])[0] if native_family.get("ranked_combos") else {}
    etf_taker = taker_status_from_family(etf_family)
    native_taker = taker_status_from_family(native_family)

    if etf_taker["status"] == "discarded" and native_taker["status"] == "ranked":
        verdict = "partial_recovery_without_leadership"
        takeaway = (
            "Switching from ETF proxies to native 24/7 Binance futures bars materially improved taker/OI usability: the flow stacks stopped failing the lag filter and became ranked candidates, but they still did not overtake price-state breakout."
        )
    elif etf_taker["status"] == "discarded" and native_taker["status"] == "discarded":
        verdict = "no_recovery"
        takeaway = "Changing the price source did not rescue taker/OI; the flow-heavy stacks stayed too laggy or sparse."
    elif native_taker["status"] == "ranked" and str(native_top.get("combo_id") or "").startswith("taker_oi"):
        verdict = "full_recovery"
        takeaway = "Native 24/7 bars fully restored taker/OI leadership; venue mismatch was the main blocker."
    else:
        verdict = "mixed"
        takeaway = "Native bars changed ranking quality, but the source-control result is mixed and needs manual review."

    return {
        "etf_top_combo": etf_top.get("combo_id"),
        "etf_top_combo_canonical": canonical_combo_id(etf_top.get("combo_id")),
        "native_top_combo": native_top.get("combo_id"),
        "native_top_combo_canonical": canonical_combo_id(native_top.get("combo_id")),
        "etf_taker_status": etf_taker,
        "native_taker_status": native_taker,
        "control_verdict": verdict,
        "control_takeaway": takeaway,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Binance Indicator Source Control Report",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- ETF artifact: `{payload.get('etf_artifact') or ''}`",
        f"- native artifact: `{payload.get('native_artifact') or ''}`",
        "",
        "## Control Result",
        f"- verdict: `{payload.get('control_verdict')}`",
        f"- takeaway: {payload.get('control_takeaway')}",
        "",
        "## Latest Crypto Route Refresh Audit",
        f"- status: `{payload.get('latest_crypto_route_refresh_status') or '-'}`",
        f"- brief: `{payload.get('latest_crypto_route_refresh_brief') or '-'}`",
        f"- reuse_gate: `{payload.get('latest_crypto_route_refresh_reuse_gate_brief') or '-'}`",
        f"- reuse_level: `{payload.get('latest_crypto_route_refresh_reuse_level') or '-'}`",
        f"- artifact: `{payload.get('latest_crypto_route_refresh_artifact') or '-'}`",
        f"- native_mode: `{payload.get('latest_crypto_route_refresh_native_mode') or '-'}`",
        f"- native_reuse: `{payload.get('latest_crypto_route_refresh_reused_native_count') or 0}/{payload.get('latest_crypto_route_refresh_native_step_count') or 0}`",
        "",
        "## Top Combo Comparison",
        f"- ETF top: `{payload.get('etf_top_combo')}` canonical=`{payload.get('etf_top_combo_canonical')}`",
        f"- native top: `{payload.get('native_top_combo')}` canonical=`{payload.get('native_top_combo_canonical')}`",
        "",
        "## Taker/OI Status",
        f"- ETF: `{payload.get('etf_taker_status', {}).get('status')}` combo=`{payload.get('etf_taker_status', {}).get('best_combo')}` canonical=`{payload.get('etf_taker_status', {}).get('best_combo_canonical')}` timely=`{payload.get('etf_taker_status', {}).get('best_timely_hit_rate'):.2%}`",
        f"- native: `{payload.get('native_taker_status', {}).get('status')}` combo=`{payload.get('native_taker_status', {}).get('best_combo')}` canonical=`{payload.get('native_taker_status', {}).get('best_combo_canonical')}` timely=`{payload.get('native_taker_status', {}).get('best_timely_hit_rate'):.2%}`",
    ]
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare ETF-proxy and native-crypto control results for Binance indicator combos.")
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

    etf_path = latest_artifact(review_dir, "binance_indicator_combo_etf", runtime_now)
    native_path = latest_artifact(review_dir, "binance_indicator_combo_native_crypto", runtime_now)
    etf_payload = json.loads(etf_path.read_text(encoding="utf-8"))
    native_payload = json.loads(native_path.read_text(encoding="utf-8"))
    control = classify_control_result(etf_payload, native_payload)
    latest_refresh_audit = _crypto_route_refresh_audit_lane(review_dir, runtime_now)
    latest_refresh_reuse_gate = _crypto_route_refresh_reuse_gate(latest_refresh_audit)

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_binance_indicator_source_control_report.json"
    md_path = review_dir / f"{stamp}_binance_indicator_source_control_report.md"
    checksum_path = review_dir / f"{stamp}_binance_indicator_source_control_report_checksum.json"

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "etf_artifact": str(etf_path),
        "native_artifact": str(native_path),
        **control,
        "latest_crypto_route_refresh_status": latest_refresh_audit["status"],
        "latest_crypto_route_refresh_brief": latest_refresh_audit["brief"],
        "latest_crypto_route_refresh_artifact": latest_refresh_audit["artifact"],
        "latest_crypto_route_refresh_as_of": latest_refresh_audit["as_of"],
        "latest_crypto_route_refresh_native_mode": latest_refresh_audit["native_mode"],
        "latest_crypto_route_refresh_native_step_count": latest_refresh_audit["native_step_count"],
        "latest_crypto_route_refresh_reused_native_count": latest_refresh_audit["reused_native_count"],
        "latest_crypto_route_refresh_missing_reused_count": latest_refresh_audit["missing_reused_count"],
        "latest_crypto_route_refresh_note": latest_refresh_audit["note"],
        "latest_crypto_route_refresh_done_when": latest_refresh_audit["done_when"],
        "latest_crypto_route_refresh_reuse_level": latest_refresh_reuse_gate["level"],
        "latest_crypto_route_refresh_reuse_gate_status": latest_refresh_reuse_gate["status"],
        "latest_crypto_route_refresh_reuse_gate_brief": latest_refresh_reuse_gate["brief"],
        "latest_crypto_route_refresh_reuse_gate_blocking": latest_refresh_reuse_gate["blocking"],
        "latest_crypto_route_refresh_reuse_gate_blocker_detail": latest_refresh_reuse_gate["blocker_detail"],
        "latest_crypto_route_refresh_reuse_gate_done_when": latest_refresh_reuse_gate["done_when"],
        "artifact_label": "binance-indicator-source-control:ok",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "artifact": str(json_path),
        "artifact_sha256": sha256_file(json_path),
        "markdown": str(md_path),
        "markdown_sha256": sha256_file(md_path),
        "generated_at": fmt_utc(runtime_now),
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="binance_indicator_source_control_report",
        current_paths=[json_path, md_path, checksum_path],
        keep=max(3, int(args.artifact_keep)),
        ttl_hours=max(24.0, float(args.artifact_ttl_hours)),
    )
    payload["artifact"] = str(json_path)
    payload["markdown"] = str(md_path)
    payload["checksum"] = str(checksum_path)
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_payload["artifact_sha256"] = sha256_file(json_path)
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
