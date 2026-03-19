#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(r"(?P<stamp>\d{8}T\d{6}Z)")
COMBO_CANONICAL_ALIASES = {
    "ad_breakout": "cvd_breakout",
    "ad_rsi_breakout": "cvd_rsi_breakout",
    "ad_rsi_vol_breakout": "cvd_rsi_vol_breakout",
    "ad_rsi_reclaim": "cvd_rsi_reclaim",
    "taker_oi_ad_breakout": "taker_oi_cvd_breakout",
    "taker_oi_ad_rsi_breakout": "taker_oi_cvd_rsi_breakout",
}


def canonical_combo_id(combo_id: Any) -> str:
    combo = str(combo_id or "")
    return COMBO_CANONICAL_ALIASES.get(combo, combo)


def artifact_status_rank(payload: dict[str, Any]) -> int:
    status = str(payload.get("status") or "").strip().lower()
    if status == "ok" or payload.get("ok") is True:
        return 2
    if status:
        return 0
    return 1


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


def artifact_stamp(path: Path) -> dt.datetime:
    match = TIMESTAMP_RE.search(path.name)
    if not match:
        return dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    return dt.datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


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


def latest_group_artifact(review_dir: Path, group_name: str) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(
        review_dir.glob("*_binance_indicator_combo_native_crypto.json"),
        key=lambda item: (artifact_stamp(item), item.name),
        reverse=True,
    )
    fallback: tuple[Path, dict[str, Any]] | None = None
    partial_fallback: tuple[Path, dict[str, Any]] | None = None
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("symbol_group") or "") == group_name:
            rank = artifact_status_rank(payload)
            if rank >= 2:
                return path, payload
            if fallback is None and rank >= 1:
                fallback = (path, payload)
            if partial_fallback is None:
                partial_fallback = (path, payload)
    if fallback is not None:
        return fallback
    if partial_fallback is not None:
        return partial_fallback
    raise FileNotFoundError(f"missing_native_group_artifact:{group_name}")


def best_taker_ranked(payload: dict[str, Any]) -> dict[str, Any] | None:
    for row in payload.get("native_crypto_family", {}).get("ranked_combos", []):
        combo_id = str(row.get("combo_id") or "")
        if combo_id.startswith("taker_oi") or combo_id.startswith("crowding_filtered"):
            return row
    return None


def classify_group(payload: dict[str, Any]) -> dict[str, Any]:
    top = (payload.get("native_crypto_family", {}).get("ranked_combos") or [{}])[0]
    taker = best_taker_ranked(payload)
    if taker is None:
        return {
            "status": "no_taker_recovery",
            "top_combo": top.get("combo_id"),
            "top_combo_canonical": canonical_combo_id(top.get("combo_id")),
            "top_return": float(top.get("avg_total_return") or 0.0),
            "taker_best_combo": None,
            "taker_best_combo_canonical": None,
            "taker_best_return": 0.0,
            "reason": "No taker/OI stack survived into the ranked set.",
        }

    taker_return = float(taker.get("avg_total_return") or 0.0)
    if taker_return > 0.0:
        status = "taker_positive_secondary"
        reason = "Taker/OI recovered and produced positive aggregate return, but did not take first place."
    else:
        status = "taker_ranked_negative"
        reason = "Taker/OI recovered into the ranked set, but aggregate return stayed non-positive."

    return {
        "status": status,
        "top_combo": top.get("combo_id"),
        "top_combo_canonical": canonical_combo_id(top.get("combo_id")),
        "top_return": float(top.get("avg_total_return") or 0.0),
        "taker_best_combo": taker.get("combo_id"),
        "taker_best_combo_canonical": canonical_combo_id(taker.get("combo_id")),
        "taker_best_return": taker_return,
        "taker_timely_hit_rate": float(taker.get("avg_timely_hit_rate") or 0.0),
        "reason": reason,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Binance Native Crypto Group Report",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        "",
        "## Group Comparison",
    ]
    for key in ("majors", "beta"):
        row = payload.get("groups", {}).get(key, {})
        lines.append(
            f"- `{key}` status=`{row.get('status')}` top=`{row.get('top_combo')}` top_canonical=`{row.get('top_combo_canonical')}` top_return=`{row.get('top_return')}` taker=`{row.get('taker_best_combo')}` taker_canonical=`{row.get('taker_best_combo_canonical')}` taker_return=`{row.get('taker_best_return')}`"
        )
        lines.append(f"  - reason: {row.get('reason')}")
    lines.extend(["", "## Takeaway", f"- {payload.get('overall_takeaway')}"])
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize majors vs beta native crypto indicator-combo results.")
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

    majors_path, majors_payload = latest_group_artifact(review_dir, "majors")
    beta_path, beta_payload = latest_group_artifact(review_dir, "beta")
    majors = classify_group(majors_payload)
    beta = classify_group(beta_payload)

    if str(beta.get("status")) == "taker_positive_secondary" and str(majors.get("status")) != "taker_positive_secondary":
        overall_takeaway = "Treat taker/OI as a beta-crypto secondary confirmation layer before considering it for majors."
    elif str(beta.get("status")).startswith("taker_") and str(majors.get("status")).startswith("taker_"):
        overall_takeaway = "Taker/OI recovered across both groups, but still belongs in secondary confirmation unless it overtakes price-state breakout."
    else:
        overall_takeaway = "Native group split did not produce a clean taker/OI leadership lane; keep taker/OI in research mode."

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_artifacts": {
            "majors": str(majors_path),
            "beta": str(beta_path),
        },
        "groups": {
            "majors": majors,
            "beta": beta,
        },
        "overall_takeaway": overall_takeaway,
        "artifact_label": "binance-native-crypto-group-report:ok",
    }

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_binance_indicator_native_group_report.json"
    md_path = review_dir / f"{stamp}_binance_indicator_native_group_report.md"
    checksum_path = review_dir / f"{stamp}_binance_indicator_native_group_report_checksum.json"
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
        stem="binance_indicator_native_group_report",
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
