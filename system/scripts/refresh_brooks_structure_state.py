#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
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


def fmt_utc(value: dt.datetime | None) -> str:
    effective = value or now_utc()
    return effective.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def step_now(base: dt.datetime, offset_seconds: int) -> dt.datetime:
    return base + dt.timedelta(seconds=int(offset_seconds))


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


def latest_review_json_artifact(review_dir: Path, suffix: str, reference_now: dt.datetime) -> Path | None:
    candidates = sorted(
        review_dir.glob(f"*_{suffix}.json"),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    for path in candidates:
        stamp_dt = parsed_artifact_stamp(path)
        if stamp_dt is None:
            continue
        if stamp_dt <= reference_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES):
            return path
    return None


def latest_stamped_datetime(review_dir: Path, suffixes: list[str]) -> dt.datetime | None:
    latest_dt: dt.datetime | None = None
    for suffix in suffixes:
        for path in review_dir.glob(f"*_{suffix}.json"):
            stamp_dt = parsed_artifact_stamp(path)
            if stamp_dt is None:
                continue
            if latest_dt is None or stamp_dt > latest_dt:
                latest_dt = stamp_dt
    return latest_dt


def derive_runtime_now(review_dir: Path, requested_now: str | None) -> dt.datetime:
    if str(requested_now or "").strip():
        return parse_now(requested_now)
    runtime_now = now_utc()
    latest_dt = latest_stamped_datetime(
        review_dir,
        [
            "brooks_price_action_market_study",
            "brooks_price_action_route_report",
            "brooks_price_action_execution_plan",
            "brooks_structure_review_queue",
            "brooks_structure_refresh",
        ],
    )
    if latest_dt is not None and latest_dt >= runtime_now:
        return latest_dt + dt.timedelta(seconds=1)
    return runtime_now


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


def run_json_step(*, step_name: str, cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{step_name}_failed: {(proc.stderr or proc.stdout or '').strip() or f'returncode={proc.returncode}'}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{step_name}_invalid_json: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{step_name}_invalid_payload")
    return payload


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def reuse_previous_artifact_payload(
    *,
    review_dir: Path,
    suffix: str,
    reference_now: dt.datetime,
    control_note: str,
) -> dict[str, Any]:
    artifact_path = latest_review_json_artifact(review_dir, suffix, reference_now)
    if artifact_path is None:
        raise FileNotFoundError(f"no_{suffix}_artifact")
    payload = load_json_mapping(artifact_path)
    payload["artifact"] = str(artifact_path)
    payload["status"] = "carry_over_previous_artifact"
    payload["control_note"] = control_note
    if not str(payload.get("as_of") or "").strip():
        stamp_dt = parsed_artifact_stamp(artifact_path)
        if stamp_dt is not None:
            payload["as_of"] = fmt_utc(stamp_dt)
    return payload


def script_path(name: str) -> Path:
    return SYSTEM_ROOT / "scripts" / name


def payload_as_of(payload: dict[str, Any]) -> dt.datetime | None:
    raw = str(payload.get("as_of") or "").strip()
    if not raw:
        return None
    try:
        return parse_now(raw)
    except ValueError:
        return None


def payload_or_artifact_as_of(payload: dict[str, Any]) -> str:
    effective = payload_as_of(payload)
    if effective is not None:
        return fmt_utc(effective)
    artifact_text = str(payload.get("artifact") or "").strip()
    if artifact_text:
        stamp_dt = parsed_artifact_stamp(Path(artifact_text))
        if stamp_dt is not None:
            return fmt_utc(stamp_dt)
    return ""


def render_markdown(payload: dict[str, Any]) -> str:
    steps = list(payload.get("steps") or [])
    lines = [
        "# Brooks Structure Refresh",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- review_status: `{payload.get('review_status') or '-'}`",
        f"- review_brief: `{payload.get('review_brief') or '-'}`",
        f"- queue_count: `{int(payload.get('queue_count') or 0)}`",
        f"- head_symbol: `{payload.get('head_symbol') or '-'}`",
        f"- head_action: `{payload.get('head_action') or '-'}`",
        f"- head_priority_score: `{payload.get('head_priority_score') if payload.get('head_priority_score') is not None else '-'}`",
        "",
        "## Artifacts",
        f"- study: `{payload.get('study_artifact') or '-'}`",
        f"- route_report: `{payload.get('route_report_artifact') or '-'}`",
        f"- execution_plan: `{payload.get('execution_plan_artifact') or '-'}`",
        f"- review_queue: `{payload.get('review_queue_artifact') or '-'}`",
        "",
        "## Steps",
    ]
    for row in steps:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('name')}` now=`{row.get('now') or '-'}` as_of=`{row.get('as_of') or '-'}` artifact=`{row.get('artifact') or '-'}`"
        )
    head = dict(payload.get("head") or {})
    if head:
        lines.extend(
            [
                "",
                "## Head",
                f"- `{head.get('symbol')}` `{head.get('strategy_id')}` `{head.get('direction')}`",
                f"- tier=`{head.get('tier') or '-'}` plan_status=`{head.get('plan_status') or '-'}` action=`{head.get('execution_action') or '-'}`",
                f"- route_score=`{head.get('route_selection_score')}` signal_score=`{head.get('signal_score')}` age_bars=`{head.get('signal_age_bars')}` priority=`{head.get('priority_score')}`/{head.get('priority_tier')}`",
                f"- blocker: {head.get('blocker_detail') or '-'}",
                f"- done_when: {head.get('done_when') or '-'}",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh the Brooks structure research chain from study to review queue.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--min-bars", type=int, default=120)
    parser.add_argument("--asset-classes", default="")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--recent-signal-age-bars", type=int, default=3)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--now", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    review_dir = Path(args.review_dir).expanduser().resolve()
    runtime_now = derive_runtime_now(review_dir, args.now)

    study_now = step_now(runtime_now, 0)
    try:
        study_payload = run_json_step(
            step_name="build_brooks_price_action_market_study",
            cmd=[
                "python3",
                str(script_path("backtest_brooks_price_action_all_market.py")),
                "--output-root",
                str(output_root),
                "--review-dir",
                str(review_dir),
                "--min-bars",
                str(int(args.min_bars)),
                "--asset-classes",
                str(args.asset_classes or ""),
                "--symbols",
                str(args.symbols or ""),
                "--now",
                fmt_utc(study_now),
            ],
        )
    except RuntimeError as exc:
        if "no_local_market_frames" not in str(exc):
            raise
        study_payload = reuse_previous_artifact_payload(
            review_dir=review_dir,
            suffix="brooks_price_action_market_study",
            reference_now=runtime_now,
            control_note="Study refresh skipped because no_local_market_frames; reusing latest Brooks market study artifact.",
        )

    route_now = step_now(runtime_now, 1)
    route_payload = run_json_step(
        step_name="build_brooks_price_action_route_report",
        cmd=[
            "python3",
            str(script_path("build_brooks_price_action_route_report.py")),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--min-bars",
            str(int(args.min_bars)),
            "--recent-signal-age-bars",
            str(int(args.recent_signal_age_bars)),
            "--now",
            fmt_utc(route_now),
        ],
    )

    plan_now = step_now(runtime_now, 2)
    execution_plan_payload = run_json_step(
        step_name="build_brooks_price_action_execution_plan",
        cmd=[
            "python3",
            str(script_path("build_brooks_price_action_execution_plan.py")),
            "--review-dir",
            str(review_dir),
            "--now",
            fmt_utc(plan_now),
        ],
    )

    queue_now = step_now(runtime_now, 3)
    review_queue_payload = run_json_step(
        step_name="build_brooks_structure_review_queue",
        cmd=[
            "python3",
            str(script_path("build_brooks_structure_review_queue.py")),
            "--review-dir",
            str(review_dir),
            "--now",
            fmt_utc(queue_now),
        ],
    )

    steps = [
        {
            "name": "build_brooks_price_action_market_study",
            "now": fmt_utc(study_now),
            "status": str(study_payload.get("status") or study_payload.get("ok") or ""),
            "as_of": payload_or_artifact_as_of(study_payload),
            "artifact": str(study_payload.get("artifact") or ""),
        },
        {
            "name": "build_brooks_price_action_route_report",
            "now": fmt_utc(route_now),
            "status": str(route_payload.get("status") or route_payload.get("ok") or ""),
            "as_of": payload_or_artifact_as_of(route_payload),
            "artifact": str(route_payload.get("artifact") or ""),
        },
        {
            "name": "build_brooks_price_action_execution_plan",
            "now": fmt_utc(plan_now),
            "status": str(execution_plan_payload.get("status") or execution_plan_payload.get("ok") or ""),
            "as_of": payload_or_artifact_as_of(execution_plan_payload),
            "artifact": str(execution_plan_payload.get("artifact") or ""),
        },
        {
            "name": "build_brooks_structure_review_queue",
            "now": fmt_utc(queue_now),
            "status": str(review_queue_payload.get("status") or review_queue_payload.get("ok") or ""),
            "as_of": payload_or_artifact_as_of(review_queue_payload),
            "artifact": str(review_queue_payload.get("artifact") or ""),
        },
    ]

    payload: dict[str, Any] = {
        "action": "refresh_brooks_structure_state",
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "study_artifact": str(study_payload.get("artifact") or ""),
        "route_report_artifact": str(route_payload.get("artifact") or ""),
        "execution_plan_artifact": str(execution_plan_payload.get("artifact") or ""),
        "review_queue_artifact": str(review_queue_payload.get("artifact") or ""),
        "review_status": str(review_queue_payload.get("review_status") or ""),
        "review_brief": str(review_queue_payload.get("review_brief") or ""),
        "queue_status": str(review_queue_payload.get("queue_status") or ""),
        "queue_count": int(review_queue_payload.get("queue_count") or 0),
        "priority_status": str(review_queue_payload.get("priority_status") or ""),
        "priority_brief": str(review_queue_payload.get("priority_brief") or ""),
        "head": dict(review_queue_payload.get("head") or {}),
        "head_symbol": str((review_queue_payload.get("head") or {}).get("symbol") or ""),
        "head_action": str((review_queue_payload.get("head") or {}).get("execution_action") or ""),
        "head_priority_score": (review_queue_payload.get("head") or {}).get("priority_score"),
        "steps": steps,
    }

    review_dir.mkdir(parents=True, exist_ok=True)
    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_brooks_structure_refresh.json"
    markdown_path = review_dir / f"{stamp}_brooks_structure_refresh.md"
    checksum_path = review_dir / f"{stamp}_brooks_structure_refresh_checksum.json"
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
        stem="brooks_structure_refresh",
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
