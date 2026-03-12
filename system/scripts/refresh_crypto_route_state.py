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

DEFAULT_NATIVE_GROUP_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "native_custom",
        "symbol_group": "custom",
        "symbols": "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT",
        "lookback_bars": 300,
        "sample_windows": 3,
        "window_bars": 40,
    },
    {
        "name": "native_majors",
        "symbol_group": "majors",
        "symbols": "BTCUSDT,ETHUSDT",
        "lookback_bars": 300,
        "sample_windows": 3,
        "window_bars": 40,
    },
    {
        "name": "native_beta",
        "symbol_group": "beta",
        "symbols": "SOLUSDT,BNBUSDT",
        "lookback_bars": 300,
        "sample_windows": 3,
        "window_bars": 40,
    },
    {
        "name": "native_sol",
        "symbol_group": "sol",
        "symbols": "SOLUSDT",
        "lookback_bars": 300,
        "sample_windows": 3,
        "window_bars": 40,
    },
    {
        "name": "native_bnb",
        "symbol_group": "bnb",
        "symbols": "BNBUSDT",
        "lookback_bars": 300,
        "sample_windows": 3,
        "window_bars": 40,
    },
    {
        "name": "native_majors_long",
        "symbol_group": "majors_long",
        "symbols": "BTCUSDT,ETHUSDT",
        "lookback_bars": 720,
        "sample_windows": 6,
        "window_bars": 40,
    },
    {
        "name": "native_beta_long",
        "symbol_group": "beta_long",
        "symbols": "SOLUSDT,BNBUSDT",
        "lookback_bars": 720,
        "sample_windows": 6,
        "window_bars": 40,
    },
    {
        "name": "native_sol_long",
        "symbol_group": "sol_long",
        "symbols": "SOLUSDT",
        "lookback_bars": 720,
        "sample_windows": 6,
        "window_bars": 40,
    },
    {
        "name": "native_bnb_long",
        "symbol_group": "bnb_long",
        "symbols": "BNBUSDT",
        "lookback_bars": 720,
        "sample_windows": 4,
        "window_bars": 80,
    },
)


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
            "crypto_route_brief",
            "crypto_route_operator_brief",
            "binance_indicator_symbol_route_handoff",
            "binance_indicator_combo_playbook",
            "binance_indicator_source_control_report",
            "crypto_route_refresh",
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


def script_path(name: str) -> Path:
    return SYSTEM_ROOT / "scripts" / name


def payload_as_of(payload: dict[str, Any]) -> dt.datetime | None:
    raw = str(payload.get("as_of") or "").strip()
    if not raw:
        return None
    return parse_now(raw)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Crypto Route Refresh",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- crypto_route_brief_artifact: `{payload.get('crypto_route_brief_artifact') or ''}`",
        f"- crypto_route_operator_candidate_artifact: `{payload.get('crypto_route_operator_candidate_artifact') or ''}`",
        f"- source_control_artifact: `{payload.get('source_control_artifact') or ''}`",
        f"- route_handoff_artifact: `{payload.get('route_handoff_artifact') or ''}`",
        "",
        "## Steps",
    ]
    for row in payload.get("steps", []):
        lines.append(
            f"- `{row.get('rank')}` `{row.get('name')}` status=`{row.get('status')}` artifact=`{row.get('artifact')}`"
        )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh crypto route artifacts in a stable guarded sequence.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--now", default="")
    parser.add_argument("--rpm", type=int, default=20)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--per-symbol-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--summarize-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--job-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_now = derive_runtime_now(review_dir, args.now)
    python_bin = "python3"

    steps: list[dict[str, Any]] = []
    next_now = runtime_now

    def record_step(rank: int, name: str, payload: dict[str, Any], status_key: str = "status") -> None:
        steps.append(
            {
                "rank": rank,
                "name": name,
                "artifact": str(payload.get("artifact") or ""),
                "status": str(payload.get(status_key) or payload.get("ok") or ""),
                "now": str(payload.get("as_of") or ""),
            }
        )

    def advance_now(payload: dict[str, Any]) -> None:
        nonlocal next_now
        effective = payload_as_of(payload)
        candidate = next_now + dt.timedelta(seconds=1)
        if effective is not None:
            candidate = max(candidate, effective + dt.timedelta(seconds=1))
        next_now = candidate

    for group_rank, spec in enumerate(DEFAULT_NATIVE_GROUP_SPECS, start=1):
        step_dt = next_now
        cmd = [
            python_bin,
            str(script_path("run_backtest_binance_indicator_combo_native_crypto_guarded.py")),
            "--review-dir",
            str(review_dir),
            "--symbols",
            str(spec["symbols"]),
            "--symbol-group",
            str(spec["symbol_group"]),
            "--interval",
            "1h",
            "--lookback-bars",
            str(int(spec["lookback_bars"])),
            "--sample-windows",
            str(int(spec["sample_windows"])),
            "--window-bars",
            str(int(spec["window_bars"])),
            "--hold-bars",
            "4",
            "--binance-limit",
            "300",
            "--binance-period",
            "1h",
            "--rpm",
            str(int(args.rpm)),
            "--timeout-ms",
            str(int(args.timeout_ms)),
            "--per-symbol-timeout-seconds",
            str(float(args.per_symbol_timeout_seconds)),
            "--summarize-timeout-seconds",
            str(float(args.summarize_timeout_seconds)),
            "--job-timeout-seconds",
            str(float(args.job_timeout_seconds)),
            "--artifact-ttl-hours",
            str(float(args.artifact_ttl_hours)),
            "--artifact-keep",
            str(int(args.artifact_keep)),
            "--now",
            fmt_utc(step_dt),
        ]
        payload = run_json_step(step_name=str(spec["name"]), cmd=cmd)
        record_step(group_rank, str(spec["name"]), payload)
        advance_now(payload)

    scripted_steps: tuple[tuple[str, str], ...] = (
        ("build_source_control_report", "build_binance_indicator_source_control_report.py"),
        ("build_native_group_report", "build_binance_indicator_native_group_report.py"),
        ("build_native_lane_stability_report", "build_binance_indicator_native_lane_stability_report.py"),
        ("build_native_beta_leg_report", "build_binance_indicator_beta_leg_report.py"),
        ("build_native_lane_playbook", "build_binance_indicator_native_lane_playbook.py"),
        ("build_native_beta_leg_window_report", "build_binance_indicator_beta_leg_window_report.py"),
        ("build_bnb_flow_focus", "build_binance_indicator_bnb_flow_focus.py"),
        ("build_combo_playbook", "build_binance_indicator_combo_playbook.py"),
        ("build_symbol_route_handoff", "build_binance_indicator_symbol_route_handoff.py"),
        ("build_crypto_route_brief", "build_crypto_route_brief.py"),
    )

    scripted_payloads: dict[str, dict[str, Any]] = {}
    for rank_index, (step_name, script_name) in enumerate(scripted_steps, start=len(steps) + 1):
        step_dt = next_now
        cmd = [
            python_bin,
            str(script_path(script_name)),
            "--review-dir",
            str(review_dir),
        ]
        if step_name in {"build_bnb_flow_focus", "build_crypto_route_brief"}:
            cmd.extend(["--now", fmt_utc(step_dt)])
        payload = run_json_step(step_name=step_name, cmd=cmd)
        scripted_payloads[step_name] = payload
        record_step(rank_index, step_name, payload)
        advance_now(payload)

    final_now = next_now
    stamp = final_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_crypto_route_refresh.json"
    md_path = review_dir / f"{stamp}_crypto_route_refresh.md"
    checksum_path = review_dir / f"{stamp}_crypto_route_refresh_checksum.json"

    source_control_payload = scripted_payloads["build_source_control_report"]
    route_handoff_payload = scripted_payloads["build_symbol_route_handoff"]
    brief_payload = scripted_payloads["build_crypto_route_brief"]

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(final_now),
        "output_root": str(output_root),
        "review_dir": str(review_dir),
        "crypto_route_brief_artifact": str(brief_payload.get("artifact") or ""),
        "crypto_route_operator_candidate_artifact": str(review_dir / "*_crypto_route_operator_brief.json"),
        "source_control_artifact": str(source_control_payload.get("artifact") or ""),
        "route_handoff_artifact": str(route_handoff_payload.get("artifact") or ""),
        "steps": steps,
        "artifact_label": "crypto-route-refresh:ok",
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
        stem="crypto_route_refresh",
        current_paths=[json_path, md_path, checksum_path],
        keep=max(3, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=final_now,
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
