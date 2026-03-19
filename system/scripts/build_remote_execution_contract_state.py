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


def text(value: Any) -> str:
    return str(value or "").strip()


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prune_artifacts(
    review_dir: Path,
    *,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, float(ttl_hours)))
    protected = {path.name for path in current_paths}
    candidates: list[Path] = []
    for pattern in (
        "*_remote_execution_contract_state.json",
        "*_remote_execution_contract_state.md",
        "*_remote_execution_contract_state_checksum.json",
    ):
        candidates.extend(review_dir.glob(pattern))

    existing: list[tuple[float, Path]] = []
    for path in candidates:
        try:
            existing.append((path.stat().st_mtime, path))
        except OSError:
            continue

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for _, path in sorted(existing, key=lambda item: item[0], reverse=True):
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


def unwrap_operator_handoff(payload: dict[str, Any]) -> dict[str, Any]:
    operator = payload.get("operator_handoff")
    return dict(operator) if isinstance(operator, dict) else payload


def build_payload(
    *,
    handoff_path: Path,
    handoff_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    operator = unwrap_operator_handoff(handoff_payload)
    account_scope_alignment = as_dict(operator.get("account_scope_alignment"))
    contract = as_dict(operator.get("execution_contract"))
    return {
        "action": "build_remote_execution_contract_state",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "contract_status": text(contract.get("status")),
        "contract_brief": text(contract.get("brief")),
        "contract_mode": text(contract.get("mode")),
        "guarded_probe_allowed": bool(contract.get("guarded_probe_allowed", False)),
        "live_orders_allowed": bool(contract.get("live_orders_allowed", False)),
        "reason_codes": [text(code) for code in as_list(contract.get("reason_codes")) if text(code)],
        "executor_mode": text(contract.get("executor_mode")),
        "executor_mode_source": text(contract.get("executor_mode_source")),
        "target_market": text(contract.get("target_market")) or text(operator.get("ready_check_scope_market")),
        "target_source": text(contract.get("target_source")) or text(operator.get("ready_check_scope_source")),
        "executable_lane_market": text(contract.get("executable_lane_market")),
        "account_scope_alignment_status": text(contract.get("account_scope_alignment_status"))
        or text(account_scope_alignment.get("status")),
        "account_scope_alignment_brief": text(account_scope_alignment.get("brief")),
        "blocker_detail": text(contract.get("blocker_detail")),
        "done_when": text(contract.get("done_when")),
        "artifacts": {
            "remote_live_handoff": str(handoff_path),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Remote Execution Contract State",
            "",
            f"- brief: `{text(payload.get('contract_brief'))}`",
            f"- status: `{text(payload.get('contract_status'))}`",
            f"- mode: `{text(payload.get('contract_mode'))}`",
            f"- guarded_probe_allowed: `{payload.get('guarded_probe_allowed')}`",
            f"- executor_mode: `{text(payload.get('executor_mode'))}` source=`{text(payload.get('executor_mode_source')) or '-'}`",
            f"- live_orders_allowed: `{payload.get('live_orders_allowed')}`",
            f"- target: `{text(payload.get('target_market'))}` source=`{text(payload.get('target_source'))}`",
            f"- executable lane: `{text(payload.get('executable_lane_market'))}`",
            f"- scope alignment: `{text(payload.get('account_scope_alignment_brief'))}`",
            f"- reasons: `{', '.join(as_list(payload.get('reason_codes'))) or '-'}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote execution contract state for OpenClaw.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--handoff-json", default="")
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)
    handoff_path = (
        Path(args.handoff_json).expanduser().resolve()
        if text(args.handoff_json)
        else find_latest(review_dir, "*_remote_live_handoff.json")
    )
    if handoff_path is None:
        raise SystemExit("missing_required_artifacts:remote_live_handoff")

    payload = build_payload(
        handoff_path=handoff_path,
        handoff_payload=load_json_mapping(handoff_path),
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_execution_contract_state.json"
    markdown = review_dir / f"{stamp}_remote_execution_contract_state.md"
    checksum = review_dir / f"{stamp}_remote_execution_contract_state_checksum.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_paths=[artifact, markdown, checksum],
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )
    payload.update(
        {
            "artifact": str(artifact),
            "markdown": str(markdown),
            "checksum": str(checksum),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
