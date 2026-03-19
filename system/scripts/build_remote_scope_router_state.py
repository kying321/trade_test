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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def text(value: Any) -> str:
    return str(value or "").strip()


def is_remote_executable_row(row: dict[str, Any], remote_market: str) -> bool:
    area = text(row.get("area"))
    action = text(row.get("action"))
    if remote_market.strip().lower() not in {"spot", "portfolio_margin_um"}:
        return False
    if area != "crypto_route":
        return False
    return bool(action)


def route_status_for_candidate(*, operator_head: dict[str, Any], candidate: dict[str, Any], remote_market: str) -> str:
    if is_remote_executable_row(operator_head, remote_market):
        return "operator_head_inside_remote_scope"
    action = text(candidate.get("action"))
    if action in {"deprioritize_flow", "refresh_source_before_use", "watch_priority_until_long_window_confirms"}:
        return "review_candidate_inside_scope_not_trade_ready"
    return "review_candidate_inside_remote_scope"


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
        "*_remote_scope_router_state.json",
        "*_remote_scope_router_state.md",
        "*_remote_scope_router_state_checksum.json",
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


def build_payload(
    *,
    cross_market_path: Path,
    cross_market_payload: dict[str, Any],
    identity_path: Path | None,
    identity_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    operator_head = as_dict(cross_market_payload.get("operator_head"))
    review_head = as_dict(cross_market_payload.get("review_head"))
    review_backlog = [dict(row) for row in as_list(cross_market_payload.get("review_backlog")) if isinstance(row, dict)]
    remote_market = text(identity_payload.get("ready_check_scope_market")) or "portfolio_margin_um"
    route_candidates = [row for row in review_backlog if is_remote_executable_row(row, remote_market)]
    preferred = dict(route_candidates[0]) if route_candidates else (
        dict(review_head) if is_remote_executable_row(review_head, remote_market) else {}
    )
    if preferred:
        scope_router_status = route_status_for_candidate(
            operator_head=operator_head,
            candidate=preferred,
            remote_market=remote_market,
        )
        scope_router_brief = ":".join(
            [
                scope_router_status,
                text(preferred.get("symbol")),
                text(preferred.get("action")),
                remote_market,
            ]
        )
        route_recommendation = (
            "hold_remote_idle_until_ticket_ready"
            if scope_router_status == "review_candidate_inside_scope_not_trade_ready"
            else "seed_remote_intent_queue_from_review_head"
        )
        blocker_detail = text(preferred.get("blocker_detail")) or text(
            cross_market_payload.get("remote_live_operator_alignment_blocker_detail")
        )
        done_when = text(preferred.get("done_when")) or text(
            cross_market_payload.get("remote_live_operator_alignment_done_when")
        )
    else:
        scope_router_status = "no_remote_scope_candidate"
        scope_router_brief = f"no_remote_scope_candidate:-:-:{remote_market}"
        route_recommendation = "keep_remote_idle"
        blocker_detail = text(cross_market_payload.get("remote_live_operator_alignment_blocker_detail"))
        done_when = text(cross_market_payload.get("remote_live_operator_alignment_done_when"))

    route_candidates_brief = " | ".join(
        [
            ":".join(
                [
                    str(int(row.get("rank") or 0)),
                    text(row.get("symbol")),
                    text(row.get("action")),
                    str(int(row.get("priority_score") or 0)),
                ]
            )
            for row in route_candidates
            if text(row.get("symbol"))
        ]
    )
    return {
        "action": "build_remote_scope_router_state",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "scope_router_status": scope_router_status,
        "scope_router_brief": scope_router_brief,
        "remote_market": remote_market,
        "current_operator_head_brief": text(cross_market_payload.get("operator_head_brief")),
        "current_review_head_brief": text(cross_market_payload.get("review_head_brief")),
        "remote_live_operator_alignment_brief": text(
            cross_market_payload.get("remote_live_operator_alignment_brief")
        ),
        "remote_live_takeover_gate_brief": text(cross_market_payload.get("remote_live_takeover_gate_brief")),
        "route_candidates_count": len(route_candidates),
        "route_candidates_brief": route_candidates_brief or "-",
        "preferred_route_target_slot": "review_head" if preferred else "-",
        "preferred_route_symbol": text(preferred.get("symbol")),
        "preferred_route_action": text(preferred.get("action")),
        "preferred_route_priority_score": int(preferred.get("priority_score") or 0) if preferred else 0,
        "route_recommendation": route_recommendation,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "artifacts": {
            "cross_market_operator_state": str(cross_market_path),
            "remote_execution_identity_state": str(identity_path) if identity_path else "",
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Remote Scope Router State",
            "",
            f"- brief: `{text(payload.get('scope_router_brief'))}`",
            f"- remote market: `{text(payload.get('remote_market'))}`",
            f"- current operator head: `{text(payload.get('current_operator_head_brief'))}`",
            f"- current review head: `{text(payload.get('current_review_head_brief'))}`",
            f"- candidates: `{text(payload.get('route_candidates_brief'))}`",
            f"- route recommendation: `{text(payload.get('route_recommendation'))}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote scope router state for OpenClaw.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)
    cross_market_path = find_latest(review_dir, "*_cross_market_operator_state.json")
    identity_path = find_latest(review_dir, "*_remote_execution_identity_state.json")
    if cross_market_path is None:
        raise SystemExit("missing_required_artifacts:cross_market_operator_state")

    payload = build_payload(
        cross_market_path=cross_market_path,
        cross_market_payload=load_json_mapping(cross_market_path),
        identity_path=identity_path,
        identity_payload=load_json_mapping(identity_path) if identity_path and identity_path.exists() else {},
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_scope_router_state.json"
    markdown = review_dir / f"{stamp}_remote_scope_router_state.md"
    checksum = review_dir / f"{stamp}_remote_scope_router_state_checksum.json"
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
