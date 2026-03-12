#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5
SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
DEFAULT_CRYPTO_ROUTE_REFRESH_BATCHES = ("crypto_hot", "crypto_majors", "crypto_beta")


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


def _payload_richness(payload: dict[str, Any]) -> int:
    action_ladder = dict(payload.get("research_action_ladder") or {})
    crypto_route_brief = dict(payload.get("crypto_route_brief") or {})
    score = 0
    score += len([x for x in action_ladder.get("focus_primary_batches", []) if str(x).strip()]) * 4
    score += len([x for x in action_ladder.get("focus_with_regime_filter_batches", []) if str(x).strip()]) * 3
    score += len([x for x in action_ladder.get("shadow_only_batches", []) if str(x).strip()]) * 2
    if str(crypto_route_brief.get("operator_status") or "").strip():
        score += 3
    if str(crypto_route_brief.get("route_stack_brief") or "").strip():
        score += 2
    if str(crypto_route_brief.get("next_focus_symbol") or "").strip():
        score += 1
    return score


def _action_richness(payload: dict[str, Any]) -> int:
    action_ladder = dict(payload.get("research_action_ladder") or {})
    score = 0
    score += len([x for x in action_ladder.get("focus_primary_batches", []) if str(x).strip()]) * 5
    score += len([x for x in action_ladder.get("focus_with_regime_filter_batches", []) if str(x).strip()]) * 4
    score += len([x for x in action_ladder.get("research_queue_batches", []) if str(x).strip()]) * 3
    score += len([x for x in action_ladder.get("shadow_only_batches", []) if str(x).strip()]) * 2
    score += len([x for x in action_ladder.get("avoid_batches", []) if str(x).strip()])
    return score


def _crypto_richness(payload: dict[str, Any]) -> int:
    crypto_route_brief = dict(payload.get("crypto_route_brief") or {})
    crypto_route_operator_brief = dict(payload.get("crypto_route_operator_brief") or {})
    score = 0
    if str(crypto_route_brief.get("operator_status") or "").strip():
        score += 4
    if str(crypto_route_brief.get("route_stack_brief") or "").strip():
        score += 3
    if str(crypto_route_brief.get("next_focus_symbol") or "").strip():
        score += 2
    if str(crypto_route_brief.get("next_retest_action") or "").strip():
        score += 1
    if str(crypto_route_operator_brief.get("focus_window_floor") or "").strip():
        score += 2
    if str(crypto_route_operator_brief.get("price_state_window_floor") or "").strip():
        score += 1
    if str(crypto_route_operator_brief.get("comparative_window_takeaway") or "").strip():
        score += 2
    if str(crypto_route_operator_brief.get("xlong_flow_window_floor") or "").strip():
        score += 3
    if str(crypto_route_operator_brief.get("xlong_comparative_window_takeaway") or "").strip():
        score += 2
    return score


def _hot_universe_status_priority(status: str) -> int:
    text = str(status or "").strip()
    if text == "ok":
        return 3
    if text == "partial_failure":
        return 2
    if text == "dry_run":
        return 1
    return 0


def latest_hot_universe_research(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_hot_universe_research.json"))
    if not candidates:
        raise FileNotFoundError("no_hot_universe_research_artifact")
    ordered = sorted(candidates, key=lambda path: artifact_sort_key(path, reference_now), reverse=True)
    preferred: list[tuple[int, tuple[int, str, float, str], int, Path]] = []
    all_ranked: list[tuple[int, int, tuple[int, str, float, str], str, Path]] = []
    for path in ordered:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = str(payload.get("status") or "").strip()
        richness = _payload_richness(payload)
        status_priority = _hot_universe_status_priority(status)
        sort_key = artifact_sort_key(path, reference_now)
        all_ranked.append((status_priority, richness, sort_key, status, path))
        if status != "dry_run":
            preferred.append((status_priority, sort_key, richness, path))
    if preferred:
        best_non_dry = max(preferred, key=lambda item: (item[0], item[1], item[2]))
        best_overall = max(all_ranked, key=lambda item: (item[1], item[0], item[2]))
        # Prefer a richer dry-run artifact over a sparse older "ok" artifact.
        if best_non_dry[2] > 0 or best_overall[3] != "dry_run":
            return best_non_dry[3]
        return best_overall[4]
    return ordered[0]


def latest_hot_universe_action_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_hot_universe_research.json"))
    if not candidates:
        raise FileNotFoundError("no_hot_universe_research_artifact")
    ordered = sorted(candidates, key=lambda path: artifact_sort_key(path, reference_now), reverse=True)
    preferred: list[tuple[int, tuple[int, str, float, str], int, Path]] = []
    fallback: list[tuple[int, tuple[int, str, float, str], int, Path]] = []
    for path in ordered:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        richness = _action_richness(payload)
        status_priority = _hot_universe_status_priority(str(payload.get("status") or ""))
        sort_key = artifact_sort_key(path, reference_now)
        fallback.append((status_priority, sort_key, richness, path))
        if str(payload.get("status") or "").strip() != "dry_run":
            preferred.append((status_priority, sort_key, richness, path))
    if preferred:
        best_non_dry = max(preferred, key=lambda item: (item[0], item[1], item[2]))
        if best_non_dry[2] > 0:
            return best_non_dry[3]
    return max(fallback, key=lambda item: (item[1], item[0], item[2]))[3]


def latest_hot_universe_crypto_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_hot_universe_research.json"))
    if not candidates:
        raise FileNotFoundError("no_hot_universe_research_artifact")
    ordered = sorted(candidates, key=lambda path: artifact_sort_key(path, reference_now), reverse=True)
    preferred: list[tuple[int, tuple[int, str, float, str], int, Path]] = []
    ranked: list[tuple[int, tuple[int, str, float, str], int, Path]] = []
    for path in ordered:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = str(payload.get("status") or "").strip()
        entry = (
            _hot_universe_status_priority(status),
            artifact_sort_key(path, reference_now),
            _crypto_richness(payload),
            path,
        )
        ranked.append(entry)
        if status != "dry_run":
            preferred.append(entry)
    if preferred:
        best_non_dry = max(preferred, key=lambda item: (item[0], item[1], item[2]))
        if best_non_dry[2] > 0:
            return best_non_dry[3]
    return max(ranked, key=lambda item: (item[1], item[0], item[2]))[3]


def latest_crypto_route_focus_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    operator_candidates = list(review_dir.glob("*_crypto_route_operator_brief.json"))
    if operator_candidates:
        return max(operator_candidates, key=lambda path: artifact_sort_key(path, reference_now))
    brief_candidates = list(review_dir.glob("*_crypto_route_brief.json"))
    if brief_candidates:
        return max(brief_candidates, key=lambda path: artifact_sort_key(path, reference_now))
    try:
        return latest_hot_universe_crypto_source(review_dir, reference_now)
    except FileNotFoundError:
        return None


def latest_commodity_execution_lane_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_execution_lane.json"))
    if candidates:
        return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))
    blocker_candidates = list(review_dir.glob("*_live_gate_blocker_report.json"))
    if blocker_candidates:
        return max(blocker_candidates, key=lambda path: artifact_sort_key(path, reference_now))
    return None


def latest_commodity_paper_ticket_lane_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_ticket_lane.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_ticket_book_source(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_ticket_book.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_execution_preview_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_execution_preview.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_execution_artifact_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_execution_artifact.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_execution_queue_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_execution_queue.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_execution_review_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_execution_review.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_execution_retro_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_execution_retro.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_execution_gap_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_execution_gap_report.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_commodity_paper_execution_bridge_source(
    review_dir: Path, reference_now: dt.datetime | None = None
) -> Path | None:
    candidates = list(review_dir.glob("*_commodity_paper_execution_bridge.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def resolve_explicit_source(raw: str | None) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _normalize_commodity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "commodity_execution_path" not in payload:
        return dict(payload)
    route = dict(payload.get("commodity_execution_path") or {})
    primary = [str(x).strip() for x in route.get("focus_primary_batches", []) if str(x).strip()]
    regime = [str(x).strip() for x in route.get("focus_with_regime_filter_batches", []) if str(x).strip()]
    shadow = [str(x).strip() for x in route.get("shadow_only_batches", []) if str(x).strip()]
    leaders_primary = [str(x).strip().upper() for x in route.get("leader_symbols_primary", []) if str(x).strip()]
    leaders_regime = [str(x).strip().upper() for x in route.get("leader_symbols_regime_filter", []) if str(x).strip()]
    next_focus_batch = primary[0] if primary else (regime[0] if regime else "")
    next_focus_symbols = leaders_primary if next_focus_batch in set(primary) else leaders_regime
    route_stack = []
    if primary:
        route_stack.append("paper-primary:" + ",".join(primary))
    if regime:
        route_stack.append("regime-filter:" + ",".join(regime))
    if shadow:
        route_stack.append("shadow:" + ",".join(shadow))
    return {
        "status": "ok",
        "route_status": "paper-first",
        "execution_mode": str(route.get("execution_mode") or "paper_first"),
        "focus_primary_batches": primary,
        "focus_with_regime_filter_batches": regime,
        "shadow_only_batches": shadow,
        "avoid_batches": [str(x).strip() for x in route.get("avoid_batches", []) if str(x).strip()],
        "leader_symbols_primary": leaders_primary,
        "leader_symbols_regime_filter": leaders_regime,
        "next_focus_batch": next_focus_batch,
        "next_focus_symbols": next_focus_symbols,
        "next_stage": "paper_ticket_lane",
        "route_stack_brief": " | ".join(route_stack),
        "summary_text": "",
    }


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


def _list_text(values: list[str], limit: int = 4) -> str:
    items = [str(v).strip() for v in values if str(v).strip()]
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" (+{len(items) - limit})"


def _mapping_text(values: dict[str, Any], limit: int = 4) -> str:
    items = [
        f"{str(key).strip().upper()}:{str(value).strip()}"
        for key, value in sorted((values or {}).items())
        if str(key).strip() and str(value).strip()
    ]
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" (+{len(items) - limit})"


def _watch_items_text(items: list[dict[str, Any]], limit: int = 8) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        signal_date = str(row.get("signal_date") or "").strip()
        age = row.get("signal_age_days")
        if not symbol:
            continue
        detail = symbol
        if age not in (None, ""):
            detail += f":{age}d"
        if signal_date:
            detail += f"@{signal_date}"
        parts.append(detail)
    if not parts:
        return "-"
    if len(items) <= limit:
        return ", ".join(parts)
    return ", ".join(parts) + f" (+{len(items) - limit})"


def _research_embedding_quality(payload: dict[str, Any]) -> dict[str, Any]:
    action_ladder = dict(payload.get("research_action_ladder") or {})
    batch_summary = dict(payload.get("batch_summary") or {})
    focus_primary = [str(x).strip() for x in action_ladder.get("focus_primary_batches", []) if str(x).strip()]
    focus_regime = [
        str(x).strip() for x in action_ladder.get("focus_with_regime_filter_batches", []) if str(x).strip()
    ]
    research_queue = [str(x).strip() for x in action_ladder.get("research_queue_batches", []) if str(x).strip()]
    avoid_batches = [str(x).strip() for x in action_ladder.get("avoid_batches", []) if str(x).strip()]
    ranked_batches = [dict(row) for row in batch_summary.get("ranked_batches", []) if isinstance(row, dict)]
    deprioritized_batches = [
        str(row.get("batch") or "").strip()
        for row in ranked_batches
        if str(row.get("status_label") or "").strip() == "deprioritize" and str(row.get("batch") or "").strip()
    ]
    zero_trade_deprioritized_batches = [
        str(row.get("batch") or "").strip()
        for row in ranked_batches
        if str(row.get("status_label") or "").strip() == "deprioritize"
        and int(row.get("research_trades", 0) or 0) <= 0
        and int(row.get("accepted_count", 0) or 0) <= 0
        and str(row.get("batch") or "").strip()
    ]
    active_batches: list[str] = []
    for value in [*focus_primary, *focus_regime, *research_queue]:
        if value and value not in active_batches:
            active_batches.append(value)
    avoid_only_batches: list[str] = []
    for value in [*avoid_batches, *deprioritized_batches]:
        if value and value not in avoid_only_batches:
            avoid_only_batches.append(value)

    status = "no_signal"
    brief = "no_signal:-"
    blocker_detail = "latest hot_universe_research has no focus, queue, or avoid classification."
    done_when = "latest hot_universe_research promotes at least one focus_primary or research_queue batch"
    if focus_primary or focus_regime:
        status = "focus_ready"
        brief = f"focus_ready:{_list_text([*focus_primary, *focus_regime], limit=6)}"
        blocker_detail = (
            "latest hot_universe_research keeps focus batches active: "
            + _list_text([*focus_primary, *focus_regime], limit=6)
        )
        done_when = "keep at least one focus batch active in hot_universe_research"
    elif research_queue:
        status = "queue_ready"
        brief = f"queue_ready:{_list_text(research_queue, limit=6)}"
        blocker_detail = "latest hot_universe_research keeps queue batches active: " + _list_text(
            research_queue,
            limit=6,
        )
        done_when = "keep at least one research_queue batch active in hot_universe_research"
    elif avoid_only_batches:
        status = "avoid_only"
        brief = f"avoid_only:{_list_text(avoid_only_batches, limit=6)}"
        blocker_detail = (
            "latest hot_universe_research is fresh and ok, but all tracked batches are avoid/inactive: "
            + _list_text(avoid_only_batches, limit=6)
        )
        if zero_trade_deprioritized_batches:
            blocker_detail += " | zero_trades=" + _list_text(zero_trade_deprioritized_batches, limit=6)

    return {
        "status": status,
        "brief": brief,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "active_batches": active_batches,
        "avoid_batches": avoid_only_batches,
        "zero_trade_deprioritized_batches": zero_trade_deprioritized_batches,
    }


def _crypto_route_embedding_alignment(
    *,
    secondary_focus_area: str,
    secondary_focus_symbol: str,
    secondary_focus_action: str,
    quality_brief: str,
    active_batches: list[str],
) -> dict[str, Any]:
    area_text = str(secondary_focus_area or "").strip()
    symbol_text = str(secondary_focus_symbol or "").strip().upper()
    action_text = str(secondary_focus_action or "").strip()
    quality_brief_text = str(quality_brief or "").strip() or "-"
    active_crypto_batches = [
        str(batch).strip()
        for batch in active_batches
        if str(batch).strip().startswith("crypto_")
    ]
    if area_text != "crypto_route" or not symbol_text:
        return {
            "status": "not_applicable",
            "brief": "not_applicable:-",
            "blocker_detail": "secondary focus is not currently driven by crypto_route.",
            "done_when": "secondary focus returns to crypto_route if crypto alignment becomes relevant again",
        }
    if active_crypto_batches:
        return {
            "status": "aligned",
            "brief": f"aligned:{symbol_text}:{_list_text(active_crypto_batches, limit=6)}",
            "blocker_detail": (
                f"dedicated crypto_route and hot_universe_research both remain usable for {symbol_text}: "
                f"{_list_text(active_crypto_batches, limit=6)}"
            ),
            "done_when": "keep crypto_route and hot_universe_research aligned",
        }
    return {
        "status": "route_ahead_of_embedding",
        "brief": f"route_ahead_of_embedding:{symbol_text}:{quality_brief_text}",
        "blocker_detail": (
            f"dedicated crypto_route still points to {symbol_text}:{action_text or '-'}, "
            f"but latest hot_universe_research has no active crypto batches ({quality_brief_text})."
        ),
        "done_when": (
            "hot_universe_research promotes at least one crypto batch or crypto_route focus degrades to match embedding"
        ),
    }


def _crypto_route_alignment_focus(payload: dict[str, Any]) -> dict[str, str]:
    for prefix in ("next_focus", "followup_focus", "secondary_focus"):
        area = str(payload.get(f"{prefix}_area") or "").strip()
        if area != "crypto_route":
            continue
        symbol = str(payload.get(f"{prefix}_symbol") or "").strip().upper()
        action = str(payload.get(f"{prefix}_action") or "").strip()
        if not symbol:
            continue
        return {
            "slot": prefix.removesuffix("_focus"),
            "area": area,
            "symbol": symbol,
            "action": action,
        }
    return {"slot": "", "area": "", "symbol": "", "action": ""}


def _crypto_route_alignment_recovery_recipe(
    *,
    alignment_status: str,
    source_artifact: str,
    avoid_batches: list[str],
    zero_trade_deprioritized_batches: list[str],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    recipe = {
        "script": "",
        "command_hint": "",
        "expected_status": "",
        "note": "",
        "followup_script": "",
        "followup_command_hint": "",
        "verify_hint": "",
        "window_days": None,
        "target_batches": [],
    }
    if str(alignment_status or "").strip() != "route_ahead_of_embedding":
        return recipe
    artifact_text = str(source_artifact or "").strip()
    artifact_path = Path(artifact_text).expanduser() if artifact_text else Path()
    review_dir = artifact_path.parent if artifact_path.suffix else DEFAULT_REVIEW_DIR
    output_root = review_dir.parent if review_dir.name == "review" else DEFAULT_OUTPUT_ROOT
    context_path = review_dir / "NEXT_WINDOW_CONTEXT_LATEST.md"
    research_script_path = Path(__file__).resolve().parent / "run_hot_universe_research.py"
    followup_script_path = Path(__file__).resolve().parent / "refresh_commodity_paper_execution_state.py"
    now_text = reference_now.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8")) if artifact_path.exists() else {}
    except Exception:
        payload = {}
    end_text = str(payload.get("end") or "").strip() or reference_now.date().isoformat()
    start_text = str(payload.get("start") or "").strip()
    try:
        end_date = dt.date.fromisoformat(end_text)
    except Exception:
        end_date = reference_now.date()
    try:
        start_date = dt.date.fromisoformat(start_text) if start_text else end_date - dt.timedelta(days=3)
    except Exception:
        start_date = end_date - dt.timedelta(days=3)
    current_window_days = max(1, (end_date - start_date).days + 1)
    target_window_days = max(21, current_window_days)
    recovery_start_date = end_date - dt.timedelta(days=target_window_days - 1)
    universe_file = _resolve_universe_file_path(
        review_dir=review_dir,
        payload=payload if isinstance(payload, dict) else {},
        reference_now=reference_now,
    )
    target_batches = [
        str(batch).strip()
        for batch in zero_trade_deprioritized_batches
        if str(batch).strip().startswith("crypto_")
    ]
    if not target_batches:
        target_batches = [str(batch).strip() for batch in avoid_batches if str(batch).strip().startswith("crypto_")]
    if not target_batches:
        target_batches = list(DEFAULT_CRYPTO_ROUTE_REFRESH_BATCHES)
    command = [
        "python3",
        str(research_script_path),
        "--output-root",
        str(output_root),
        "--review-dir",
        str(review_dir),
        "--start",
        recovery_start_date.isoformat(),
        "--end",
        end_date.isoformat(),
        "--now",
        now_text,
        "--hours-budget",
        "0.08",
        "--max-trials-per-mode",
        "5",
        "--review-days",
        "7",
        "--run-strategy-lab",
        "--strategy-lab-candidate-count",
        "6",
        "--batch-timeout-seconds",
        "30",
    ]
    if universe_file:
        command.extend(["--universe-file", universe_file])
    for batch_name in target_batches:
        command.extend(["--batch", batch_name])
    followup_command = [
        "python3",
        str(followup_script_path),
        "--review-dir",
        str(review_dir),
        "--output-root",
        str(output_root),
        "--context-path",
        str(context_path),
    ]
    recipe["script"] = str(research_script_path)
    recipe["command_hint"] = shlex.join(command)
    recipe["expected_status"] = "ok"
    recipe["note"] = (
        f"extend crypto embedding window to {target_window_days}d and enable strategy_lab "
        f"because current crypto batches are avoid_only with zero trades"
    )
    recipe["followup_script"] = str(followup_script_path)
    recipe["followup_command_hint"] = shlex.join(followup_command)
    recipe["verify_hint"] = (
        "confirm operator_research_embedding_quality_status leaves avoid_only or "
        "operator_crypto_route_alignment_status leaves route_ahead_of_embedding"
    )
    recipe["window_days"] = target_window_days
    recipe["target_batches"] = target_batches
    return recipe


def _crypto_route_alignment_recovery_outcome(
    *,
    alignment_status: str,
    quality_status: str,
    source_status: str,
    source_payload: dict[str, Any],
) -> dict[str, Any]:
    outcome = {
        "status": "not_needed",
        "brief": "not_needed:-",
        "blocker_detail": "crypto route alignment recovery outcome is not needed when alignment is not blocked.",
        "done_when": "crypto route alignment becomes blocked again before reassessing recovery outcome",
        "failed_batch_count": 0,
        "timed_out_batch_count": 0,
        "zero_trade_batches": [],
    }
    if str(alignment_status or "").strip() != "route_ahead_of_embedding":
        return outcome

    failed_batch_count = int(source_payload.get("failed_batch_count", 0) or 0)
    timed_out_batch_count = int(source_payload.get("timed_out_batch_count", 0) or 0)
    ranked_batches = [
        dict(row)
        for row in dict(source_payload.get("batch_summary") or {}).get("ranked_batches", [])
        if isinstance(row, dict)
    ]
    zero_trade_batches = [
        str(row.get("batch") or "").strip()
        for row in ranked_batches
        if str(row.get("batch") or "").strip()
        and int(row.get("research_trades", 0) or 0) <= 0
        and int(row.get("accepted_count", 0) or 0) <= 0
    ]
    if not zero_trade_batches:
        zero_trade_batches = [
            str(batch).strip()
            for batch in dict(source_payload.get("research_action_ladder") or {}).get("avoid_batches", [])
            if str(batch).strip()
        ]
    source_status_text = str(source_status or "").strip()
    quality_status_text = str(quality_status or "").strip()

    if source_status_text == "partial_failure" or failed_batch_count > 0 or timed_out_batch_count > 0:
        outcome["status"] = "recovery_partial_failure"
        outcome["brief"] = (
            f"recovery_partial_failure:failed={failed_batch_count}:timed_out={timed_out_batch_count}"
        )
        outcome["blocker_detail"] = (
            "latest crypto alignment recovery artifact still has batch failures: "
            f"failed_batch_count={failed_batch_count}, timed_out_batch_count={timed_out_batch_count}"
        )
        outcome["done_when"] = (
            "latest hot_universe_research finishes with failed_batch_count=0 and timed_out_batch_count=0"
        )
    elif source_status_text == "ok" and quality_status_text == "avoid_only":
        outcome["status"] = "recovery_completed_no_edge"
        outcome["brief"] = (
            "recovery_completed_no_edge:"
            + _list_text(zero_trade_batches or ["no_edge_batches"], limit=6)
        )
        outcome["blocker_detail"] = (
            "latest crypto alignment recovery artifact finished cleanly, but all targeted crypto batches still "
            "show zero research trades and zero accepted strategy_lab candidates: "
            + _list_text(zero_trade_batches, limit=6)
        )
        outcome["done_when"] = (
            "hot_universe_research promotes at least one crypto batch or dedicated crypto_route degrades to match the no-edge embedding"
        )
    elif source_status_text == "ok":
        outcome["status"] = "recovery_succeeded"
        outcome["brief"] = f"recovery_succeeded:{quality_status_text or '-'}"
        outcome["blocker_detail"] = "latest crypto alignment recovery artifact finished cleanly."
        outcome["done_when"] = "keep the recovery artifact fresh while the crypto route remains usable"
    else:
        outcome["status"] = "recovery_not_run"
        outcome["brief"] = f"recovery_not_run:{source_status_text or '-'}"
        outcome["blocker_detail"] = (
            "latest crypto alignment recovery artifact is not yet a fresh non-dry-run hot_universe_research result."
        )
        outcome["done_when"] = "a fresh non-dry-run hot_universe_research artifact becomes available"

    outcome["failed_batch_count"] = failed_batch_count
    outcome["timed_out_batch_count"] = timed_out_batch_count
    outcome["zero_trade_batches"] = zero_trade_batches
    return outcome


def _crypto_route_alignment_cooldown(
    *,
    alignment_status: str,
    recovery_status: str,
    source_status: str,
    source_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    cooldown = {
        "status": "not_needed",
        "brief": "not_needed:-",
        "blocker_detail": "crypto route alignment cooldown is not needed when recovery has not completed cleanly without edge.",
        "done_when": "crypto route alignment recovery completes without edge before reassessing cooldown",
        "last_research_end_date": "",
        "next_eligible_end_date": "",
    }
    if str(alignment_status or "").strip() != "route_ahead_of_embedding":
        return cooldown
    if str(recovery_status or "").strip() != "recovery_completed_no_edge":
        return cooldown
    if str(source_status or "").strip() != "ok":
        cooldown["status"] = "cooldown_waiting_for_clean_source"
        cooldown["brief"] = f"cooldown_waiting_for_clean_source:{source_status or '-'}"
        cooldown["blocker_detail"] = "latest hot_universe_research source is not a clean ok artifact yet."
        cooldown["done_when"] = "latest hot_universe_research source becomes ok before enforcing cooldown"
        return cooldown

    end_text = str(source_payload.get("end") or "").strip()
    try:
        end_date = dt.date.fromisoformat(end_text) if end_text else reference_now.date()
    except Exception:
        end_date = reference_now.date()
    next_date = end_date + dt.timedelta(days=1)
    cooldown["last_research_end_date"] = end_date.isoformat()
    cooldown["next_eligible_end_date"] = next_date.isoformat()

    if end_date >= reference_now.date():
        cooldown["status"] = "cooldown_active_wait_for_new_market_data"
        cooldown["brief"] = f"cooldown_active_wait_for_new_market_data:>{end_date.isoformat()}"
        cooldown["blocker_detail"] = (
            "latest clean crypto recovery already evaluated data through "
            f"{end_date.isoformat()} and still found no edge; rerunning before a later end date is unlikely to change the outcome"
        )
        cooldown["done_when"] = (
            f"hot_universe_research end date advances beyond {end_date.isoformat()} or crypto_route focus changes"
        )
        return cooldown

    cooldown["status"] = "cooldown_expired_rerun_allowed"
    cooldown["brief"] = f"cooldown_expired_rerun_allowed:{end_date.isoformat()}"
    cooldown["blocker_detail"] = (
        "latest clean crypto recovery found no edge, but its end date is now behind the current reference date."
    )
    cooldown["done_when"] = f"rerun hot_universe_research using data through at least {next_date.isoformat()}"
    return cooldown


def _crypto_route_alignment_recipe_gate(
    *,
    alignment_status: str,
    recipe_script: str,
    cooldown_status: str,
    cooldown_brief: str,
    cooldown_blocker_detail: str,
    cooldown_done_when: str,
    cooldown_next_eligible_end_date: str,
) -> dict[str, str]:
    gate = {
        "status": "not_needed",
        "brief": "not_needed:-",
        "blocker_detail": "crypto route alignment recovery recipe is not needed when alignment is not blocked.",
        "done_when": "crypto route alignment becomes blocked again before reassessing recovery recipe usage",
        "ready_on_date": "",
    }
    if str(alignment_status or "").strip() != "route_ahead_of_embedding":
        return gate
    if not str(recipe_script or "").strip():
        gate["status"] = "recipe_unavailable"
        gate["brief"] = "recipe_unavailable:-"
        gate["blocker_detail"] = "crypto route alignment recovery recipe is not available."
        gate["done_when"] = "a recovery recipe becomes available"
        return gate

    cooldown_status_text = str(cooldown_status or "").strip()
    next_eligible_text = str(cooldown_next_eligible_end_date or "").strip()
    if cooldown_status_text == "cooldown_active_wait_for_new_market_data":
        gate["status"] = "deferred_by_cooldown"
        gate["brief"] = f"deferred_by_cooldown:{next_eligible_text or '-'}"
        gate["blocker_detail"] = str(cooldown_blocker_detail or "").strip() or str(cooldown_brief or "").strip() or "-"
        gate["done_when"] = str(cooldown_done_when or "").strip() or "wait for cooldown to expire"
        gate["ready_on_date"] = next_eligible_text
        return gate
    if cooldown_status_text == "cooldown_expired_rerun_allowed":
        gate["status"] = "eligible_now"
        gate["brief"] = f"eligible_now:{next_eligible_text or '-'}"
        gate["blocker_detail"] = "recovery recipe is available and cooldown has expired."
        gate["done_when"] = "execute the recovery recipe and refresh the operator handoff"
        gate["ready_on_date"] = next_eligible_text
        return gate

    gate["status"] = "available_now"
    gate["brief"] = "available_now:-"
    gate["blocker_detail"] = "recovery recipe is available for the current crypto alignment blocker."
    gate["done_when"] = "execute the recovery recipe and refresh the operator handoff"
    gate["ready_on_date"] = next_eligible_text
    return gate


def _append_action_queue_item(
    queue: list[dict[str, Any]],
    *,
    area: str,
    target: str,
    symbol: str,
    action: str,
    reason: str,
) -> None:
    area_text = str(area or "").strip() or "-"
    target_text = str(target or "").strip() or "-"
    action_text = str(action or "").strip() or "-"
    symbol_text = str(symbol or "").strip().upper() or "-"
    reason_text = str(reason or "").strip() or "-"
    dedupe_key = (area_text, target_text, action_text)
    if dedupe_key == ("-", "-", "-"):
        return
    for row in queue:
        if (row.get("area"), row.get("target"), row.get("action")) == dedupe_key:
            return
    queue.append(
        {
            "area": area_text,
            "target": target_text,
            "symbol": symbol_text,
            "action": action_text,
            "reason": reason_text,
        }
    )


def _action_queue_brief(items: list[dict[str, Any]], limit: int = 4) -> str:
    parts: list[str] = []
    for idx, row in enumerate(items[:limit], start=1):
        if not isinstance(row, dict):
            continue
        parts.append(
            f"{idx}:{row.get('area') or '-'}:{row.get('target') or '-'}:{row.get('action') or '-'}"
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _action_checklist_state(*, area: str, action: str, rank: int) -> str:
    area_text = str(area or "").strip()
    action_text = str(action or "").strip()
    if action_text.startswith("watch_"):
        return "watch"
    if action_text.startswith("wait_"):
        return "waiting"
    if rank == 1:
        return "active"
    if area_text.startswith("commodity_"):
        return "queued"
    return "queued"


def _action_checklist_blocker(
    *,
    area: str,
    action: str,
    symbol: str,
    reason: str,
    stale_signal_dates: dict[str, str],
    stale_signal_age_days: dict[str, int],
    focus_evidence_summary: dict[str, Any],
    focus_area: str,
    focus_symbol: str,
    focus_lifecycle_status: str = "",
    focus_lifecycle_blocker_detail: str = "",
    crypto_route_alignment_status: str = "",
    crypto_route_alignment_recovery_status: str = "",
    crypto_route_alignment_recovery_zero_trade_batches: list[str] | None = None,
) -> str:
    symbol_text = str(symbol or "").strip().upper()
    action_text = str(action or "").strip()
    reason_text = str(reason or "").strip()
    stale_date = stale_signal_dates.get(symbol_text, "")
    stale_age = stale_signal_age_days.get(symbol_text)

    if action_text == "review_paper_execution_retro":
        evidence_present = (
            area == focus_area
            and symbol_text == focus_symbol
            and bool(focus_evidence_summary.get("paper_execution_evidence_present"))
        )
        if evidence_present:
            if str(focus_lifecycle_status or "").strip() == "open_position_wait_close_evidence":
                return str(focus_lifecycle_blocker_detail or "").strip() or (
                    "paper execution evidence is present, but position is still OPEN; retro should wait for close evidence"
                )
            return "paper execution evidence is present; retro item still pending"
        return "retro item still pending"
    if action_text == "wait_for_paper_execution_close_evidence":
        if area == focus_area and symbol_text == focus_symbol:
            return str(focus_lifecycle_blocker_detail or "").strip() or (
                "paper execution evidence is present, but position is still OPEN; waiting for close evidence"
            )
        return "paper execution close evidence is not written yet"
    if action_text == "review_paper_execution":
        return "review item still pending"
    if action_text == "wait_for_paper_execution_fill_evidence":
        if stale_age not in (None, "") and stale_date:
            return f"paper execution fill evidence not written; stale directional signal {stale_age}d since {stale_date}"
        return "paper execution fill evidence not written"
    if action_text == "restore_commodity_directional_signal":
        if stale_age not in (None, "") and stale_date:
            return f"directional signal is stale {stale_age}d since {stale_date}"
        return "directional signal ticket is missing or not fresh"
    if action_text == "normalize_commodity_execution_price_reference":
        return "execution price is still proxy-reference only"
    if action_text == "watch_priority_until_long_window_confirms":
        if (
            str(crypto_route_alignment_status or "").strip() == "route_ahead_of_embedding"
            and str(crypto_route_alignment_recovery_status or "").strip() == "recovery_completed_no_edge"
        ):
            return (
                "long-window confirmation still missing; clean crypto recovery still shows no edge in "
                + _list_text(list(crypto_route_alignment_recovery_zero_trade_batches or []), limit=6)
            )
        return "long-window confirmation still missing"
    if action_text == "apply_commodity_execution_bridge":
        return "bridge is ready but paper apply has not been executed"
    if reason_text:
        return reason_text.replace("_", " ")
    return "pending"


def _action_checklist_done_when(
    *,
    action: str,
    symbol: str,
    focus_lifecycle_status: str = "",
    focus_lifecycle_done_when: str = "",
    crypto_route_alignment_status: str = "",
    crypto_route_alignment_recovery_status: str = "",
) -> str:
    symbol_text = str(symbol or "").strip().upper() or "SYMBOL"
    action_text = str(action or "").strip()
    if action_text == "review_paper_execution_retro":
        if str(focus_lifecycle_status or "").strip() == "open_position_wait_close_evidence":
            return str(focus_lifecycle_done_when or "").strip() or (
                f"{symbol_text} paper_execution_status leaves OPEN and retro can evaluate close outcome"
            )
        return f"{symbol_text} leaves retro_pending_symbols"
    if action_text == "wait_for_paper_execution_close_evidence":
        return str(focus_lifecycle_done_when or "").strip() or (
            f"{symbol_text} paper_execution_status leaves OPEN and close evidence becomes available"
        )
    if action_text == "review_paper_execution":
        return f"{symbol_text} leaves review_pending_symbols"
    if action_text == "wait_for_paper_execution_fill_evidence":
        return f"{symbol_text} gains paper evidence and leaves fill_evidence_pending_symbols"
    if action_text == "restore_commodity_directional_signal":
        return f"{symbol_text} prints a fresh directional signal ticket"
    if action_text == "normalize_commodity_execution_price_reference":
        return f"{symbol_text} receives executable non-proxy price levels"
    if action_text == "apply_commodity_execution_bridge":
        return f"{symbol_text} is written into paper evidence artifacts"
    if action_text == "watch_priority_until_long_window_confirms":
        if (
            str(crypto_route_alignment_status or "").strip() == "route_ahead_of_embedding"
            and str(crypto_route_alignment_recovery_status or "").strip() == "recovery_completed_no_edge"
        ):
            return f"{symbol_text} gains supporting crypto research edge or leaves priority watch"
        return f"{symbol_text} upgrades from priority watch to deploy or leaves priority watch"
    return f"{symbol_text} completes {action_text or 'the current action'}"


def _action_checklist_brief(items: list[dict[str, Any]], limit: int = 4) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("rank") or "-"),
                    str(row.get("state") or "-"),
                    str(row.get("symbol") or "-"),
                    str(row.get("action") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _focus_slots_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("slot") or "-"),
                    str(row.get("state") or "-"),
                    str(row.get("symbol") or "-"),
                    str(row.get("action") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _focus_slot_source(payload: dict[str, Any], *, area: str, action: str) -> tuple[str, str]:
    area_text = str(area or "").strip()
    action_text = str(action or "").strip()
    mapping = [
        ("commodity_execution_close_evidence", "commodity_execution_retro", payload.get("source_commodity_execution_retro_artifact")),
        ("commodity_execution_retro", "commodity_execution_retro", payload.get("source_commodity_execution_retro_artifact")),
        ("commodity_execution_review", "commodity_execution_review", payload.get("source_commodity_execution_review_artifact")),
        ("commodity_fill_evidence", "commodity_execution_review", payload.get("source_commodity_execution_review_artifact")),
        ("commodity_execution_bridge", "commodity_execution_bridge", payload.get("source_commodity_execution_bridge_artifact")),
        ("commodity_execution_gap", "commodity_execution_gap", payload.get("source_commodity_execution_gap_artifact")),
        ("commodity_execution_queue", "commodity_execution_queue", payload.get("source_commodity_execution_queue_artifact")),
        ("commodity_execution_artifact", "commodity_execution_artifact", payload.get("source_commodity_execution_artifact")),
        ("commodity_execution_preview", "commodity_execution_preview", payload.get("source_commodity_execution_preview_artifact")),
        ("commodity_ticket_book", "commodity_ticket_book", payload.get("source_commodity_ticket_book_artifact")),
        ("commodity_route", "commodity_route", payload.get("source_commodity_artifact")),
        ("crypto_route", "crypto_route", payload.get("source_crypto_route_artifact") or payload.get("source_crypto_artifact")),
        ("research_queue", "action_research", payload.get("source_action_artifact")),
    ]
    for match_area, source_kind, source_artifact in mapping:
        if area_text == match_area:
            return source_kind, str(source_artifact or "")
    if action_text == "wait_for_paper_execution_close_evidence":
        return "commodity_execution_retro", str(payload.get("source_commodity_execution_retro_artifact") or "")
    if action_text == "wait_for_paper_execution_fill_evidence":
        return "commodity_execution_review", str(payload.get("source_commodity_execution_review_artifact") or "")
    return "-", ""


def _focus_slot_source_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("slot") or "-"),
                    str(row.get("source_kind") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _focus_slot_source_status(payload: dict[str, Any], *, source_kind: str) -> str:
    source_text = str(source_kind or "").strip()
    mapping = {
        "commodity_execution_retro": str(payload.get("source_commodity_execution_retro_status") or ""),
        "commodity_execution_review": str(payload.get("source_commodity_execution_review_status") or ""),
        "commodity_execution_bridge": str(payload.get("source_commodity_execution_bridge_status") or ""),
        "commodity_execution_gap": str(payload.get("source_commodity_execution_gap_status") or ""),
        "commodity_execution_queue": str(payload.get("source_commodity_execution_queue_status") or ""),
        "commodity_execution_artifact": str(payload.get("source_commodity_execution_artifact_status") or ""),
        "commodity_execution_preview": str(payload.get("source_commodity_execution_preview_status") or ""),
        "commodity_ticket_book": str(payload.get("source_commodity_ticket_book_status") or ""),
        "commodity_route": str(payload.get("source_commodity_status") or ""),
        "crypto_route": str(payload.get("source_crypto_route_status") or payload.get("source_crypto_status") or ""),
        "action_research": str(payload.get("source_action_status") or ""),
    }
    return mapping.get(source_text, "")


def _focus_slot_source_as_of(source_artifact: str) -> str:
    text = str(source_artifact or "").strip()
    if not text:
        return ""
    stamp_dt = parsed_artifact_stamp(Path(text))
    return fmt_utc(stamp_dt) or ""


def _focus_slot_status_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        source_status = str(row.get("source_status") or "").strip() or "-"
        source_as_of = str(row.get("source_as_of") or "").strip() or "-"
        parts.append(f"{row.get('slot') or '-'}:{source_status}@{source_as_of}")
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _focus_slot_source_age_minutes(*, reference_now: dt.datetime, source_as_of: str) -> int | None:
    text = str(source_as_of or "").strip()
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    delta = reference_now - parsed.astimezone(dt.timezone.utc)
    if delta.total_seconds() < 0:
        return 0
    return int(delta.total_seconds() // 60)


def _focus_slot_source_recency(*, source_age_minutes: int | None) -> str:
    if source_age_minutes is None:
        return "unknown"
    if source_age_minutes <= 15:
        return "fresh"
    if source_age_minutes <= 1440:
        return "carry_over"
    return "stale"


def _focus_slot_recency_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        recency = str(row.get("source_recency") or "").strip() or "-"
        age = row.get("source_age_minutes")
        age_text = str(age) if age not in (None, "") else "-"
        parts.append(f"{row.get('slot') or '-'}:{recency}:{age_text}m")
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _focus_slot_source_health(
    *,
    source_status: str,
    source_recency: str,
    source_kind: str = "",
    crypto_route_alignment_cooldown_status: str = "",
) -> str:
    status_text = str(source_status or "").strip()
    recency_text = str(source_recency or "").strip()
    if status_text == "ok" and recency_text == "fresh":
        return "ready"
    if (
        status_text == "ok"
        and recency_text == "carry_over"
        and str(source_kind or "").strip() == "crypto_route"
        and str(crypto_route_alignment_cooldown_status or "").strip() == "cooldown_active_wait_for_new_market_data"
    ):
        return "carry_over_ok"
    if status_text == "ok" and recency_text == "carry_over":
        return "carry_over_ok"
    if status_text == "dry_run" and recency_text == "fresh":
        return "dry_run_only"
    if status_text == "dry_run" and recency_text == "carry_over":
        return "refresh_required"
    if status_text == "dry_run":
        return "dry_run_only"
    if status_text:
        return f"status_{status_text}"
    return "unknown"


def _focus_slot_source_refresh_action(
    *,
    source_health: str,
    source_kind: str = "",
    crypto_route_alignment_cooldown_status: str = "",
) -> str:
    health_text = str(source_health or "").strip()
    if health_text == "ready":
        return "read_current_artifact"
    if health_text == "carry_over_ok":
        if (
            str(source_kind or "").strip() == "crypto_route"
            and str(crypto_route_alignment_cooldown_status or "").strip()
            == "cooldown_active_wait_for_new_market_data"
        ):
            return "wait_for_next_eligible_end_date"
        return "consider_refresh_before_promotion"
    if health_text == "dry_run_only":
        return "do_not_promote_without_non_dry_run"
    if health_text == "refresh_required":
        return "refresh_source_before_use"
    return "inspect_source_state"


def _focus_slot_health_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("slot") or "-"),
                    str(row.get("source_health") or "-"),
                    str(row.get("source_refresh_action") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _focus_slot_refresh_backlog(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    backlog: list[dict[str, Any]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        refresh_action = str(row.get("source_refresh_action") or "").strip()
        if refresh_action in ("", "read_current_artifact", "wait_for_next_eligible_end_date"):
            continue
        backlog.append(
            {
                "slot": str(row.get("slot") or "").strip() or "-",
                "symbol": str(row.get("symbol") or "").strip().upper(),
                "action": refresh_action,
                "source_kind": str(row.get("source_kind") or "").strip() or "-",
                "source_status": str(row.get("source_status") or "").strip() or "-",
                "source_recency": str(row.get("source_recency") or "").strip() or "-",
                "source_health": str(row.get("source_health") or "").strip() or "-",
                "source_age_minutes": row.get("source_age_minutes"),
                "source_as_of": str(row.get("source_as_of") or "").strip(),
                "source_artifact": str(row.get("source_artifact") or "").strip(),
            }
        )
    return backlog


def _focus_slot_refresh_backlog_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("slot") or "-"),
                    str(row.get("symbol") or "-"),
                    str(row.get("action") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _focus_slot_promotion_gate_status(*, total_count: int, ready_count: int) -> str:
    if total_count <= 0:
        return "unknown"
    if ready_count >= total_count:
        return "promotion_ready"
    return "promotion_guarded_by_source_freshness"


def _focus_slot_promotion_gate_blocker_detail(
    *,
    total_count: int,
    ready_count: int,
    refresh_backlog: list[dict[str, Any]],
) -> str:
    if total_count <= 0:
        return "no focus slots are present"
    if ready_count >= total_count or not refresh_backlog:
        return f"all {total_count} focus slots are covered by current source or cooldown state"
    head = dict(refresh_backlog[0])
    symbol = str(head.get("symbol") or "").strip().upper() or "-"
    slot = str(head.get("slot") or "").strip() or "-"
    action = str(head.get("action") or "").strip() or "-"
    source_kind = str(head.get("source_kind") or "").strip() or "-"
    source_status = str(head.get("source_status") or "").strip() or "-"
    source_recency = str(head.get("source_recency") or "").strip() or "-"
    age = head.get("source_age_minutes")
    age_text = f", age={age}m" if age not in (None, "") else ""
    return (
        f"{symbol} {slot} source requires {action} "
        f"({source_kind}, {source_status}, {source_recency}{age_text})"
    )


def _focus_slot_promotion_gate_done_when(*, total_count: int, ready_count: int) -> str:
    if total_count <= 0:
        return "operator_focus_slots becomes non-empty"
    if ready_count >= total_count:
        return "all focus slots remain covered by current source or cooldown state"
    return "operator_focus_slot_refresh_backlog_count reaches 0"


def _focus_slot_promotion_gate_brief(*, status: str, ready_count: int, total_count: int) -> str:
    status_text = str(status or "").strip() or "-"
    return f"{status_text}:{ready_count}/{total_count}"


def _focus_slot_actionability_backlog(
    items: list[dict[str, Any]],
    *,
    alignment_focus_slot: str,
    alignment_status: str,
    alignment_brief: str,
    alignment_recovery_status: str,
    alignment_recovery_brief: str,
) -> list[dict[str, Any]]:
    status_text = str(alignment_status or "").strip()
    if status_text != "route_ahead_of_embedding":
        return []
    focus_slot_text = str(alignment_focus_slot or "").strip()
    focus_slot_alias = {
        "next": "primary",
        "primary": "primary",
        "followup": "followup",
        "secondary": "secondary",
    }.get(focus_slot_text, focus_slot_text)
    backlog: list[dict[str, Any]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        area_text = str(row.get("area") or "").strip()
        slot_text = str(row.get("slot") or "").strip()
        if area_text != "crypto_route":
            continue
        if focus_slot_alias and slot_text != focus_slot_alias:
            continue
        backlog.append(
            {
                "slot": focus_slot_text or slot_text or "-",
                "symbol": str(row.get("symbol") or "").strip().upper() or "-",
                "action": str(row.get("action") or "").strip() or "-",
                "state": str(row.get("state") or "").strip() or "-",
                "blocker_detail": str(row.get("blocker_detail") or "").strip()
                or str(alignment_brief or "").strip()
                or "-",
                "done_when": str(row.get("done_when") or "").strip() or "-",
                "alignment_status": status_text,
                "alignment_brief": str(alignment_brief or "").strip() or "-",
                "alignment_recovery_status": str(alignment_recovery_status or "").strip() or "-",
                "alignment_recovery_brief": str(alignment_recovery_brief or "").strip() or "-",
            }
        )
    return backlog


def _focus_slot_actionability_backlog_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("slot") or "-"),
                    str(row.get("symbol") or "-"),
                    str(row.get("alignment_recovery_status") or row.get("alignment_status") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _focus_slot_actionability_gate_status(*, total_count: int, actionable_count: int) -> str:
    if total_count <= 0:
        return "unknown"
    if actionable_count >= total_count:
        return "actionability_ready"
    return "actionability_guarded_by_content"


def _focus_slot_actionability_gate_blocker_detail(
    *,
    total_count: int,
    actionable_count: int,
    backlog: list[dict[str, Any]],
) -> str:
    if total_count <= 0:
        return "no focus slots are present"
    if actionable_count >= total_count or not backlog:
        return f"all {total_count} focus slots have content-aligned decision state"
    head = dict(backlog[0])
    symbol = str(head.get("symbol") or "").strip().upper() or "-"
    slot = str(head.get("slot") or "").strip() or "-"
    alignment_status = str(head.get("alignment_status") or "").strip() or "-"
    recovery_status = str(head.get("alignment_recovery_status") or "").strip() or "-"
    blocker_detail = str(head.get("blocker_detail") or "").strip() or "-"
    return (
        f"{symbol} {slot} content state remains blocked "
        f"({alignment_status}, {recovery_status}): {blocker_detail}"
    )


def _focus_slot_actionability_gate_done_when(
    *,
    total_count: int,
    actionable_count: int,
    backlog: list[dict[str, Any]],
) -> str:
    if total_count <= 0:
        return "operator_focus_slots becomes non-empty"
    if actionable_count >= total_count or not backlog:
        return "all focus slots continue with content-aligned decision state"
    head = dict(backlog[0])
    return str(head.get("done_when") or "").strip() or "operator_crypto_route_alignment_status leaves route_ahead_of_embedding"


def _focus_slot_actionability_gate_brief(*, status: str, actionable_count: int, total_count: int) -> str:
    status_text = str(status or "").strip() or "-"
    return f"{status_text}:{actionable_count}/{total_count}"


def _focus_slot_readiness_gate_status(
    *,
    total_count: int,
    promotion_gate_status: str,
    actionability_gate_status: str,
) -> str:
    if total_count <= 0:
        return "unknown"
    if str(promotion_gate_status or "").strip() != "promotion_ready":
        return "readiness_guarded_by_source_freshness"
    if str(actionability_gate_status or "").strip() != "actionability_ready":
        return "readiness_guarded_by_content"
    return "readiness_ready"


def _focus_slot_readiness_gate_blocking_gate(*, status: str) -> str:
    status_text = str(status or "").strip()
    if status_text == "readiness_guarded_by_source_freshness":
        return "source_freshness"
    if status_text == "readiness_guarded_by_content":
        return "content_actionability"
    if status_text == "readiness_ready":
        return "none"
    return "unknown"


def _focus_slot_readiness_gate_ready_count(
    *,
    status: str,
    ready_count: int,
    actionable_count: int,
    total_count: int,
) -> int:
    status_text = str(status or "").strip()
    if status_text == "readiness_guarded_by_source_freshness":
        return ready_count
    if status_text == "readiness_guarded_by_content":
        return actionable_count
    if status_text == "readiness_ready":
        return total_count
    return 0


def _focus_slot_readiness_gate_blocker_detail(
    *,
    status: str,
    promotion_gate_blocker_detail: str,
    actionability_gate_blocker_detail: str,
    total_count: int,
) -> str:
    status_text = str(status or "").strip()
    if status_text == "readiness_guarded_by_source_freshness":
        return str(promotion_gate_blocker_detail or "").strip() or "focus slot source freshness is still blocked"
    if status_text == "readiness_guarded_by_content":
        return str(actionability_gate_blocker_detail or "").strip() or "focus slot content actionability is still blocked"
    if status_text == "readiness_ready":
        return f"all {total_count} focus slots have fresh sources and content-aligned decision state"
    return "focus slot readiness could not be determined"


def _focus_slot_readiness_gate_done_when(
    *,
    status: str,
    promotion_gate_done_when: str,
    actionability_gate_done_when: str,
) -> str:
    status_text = str(status or "").strip()
    if status_text == "readiness_guarded_by_source_freshness":
        return str(promotion_gate_done_when or "").strip() or "source freshness blocker clears"
    if status_text == "readiness_guarded_by_content":
        return str(actionability_gate_done_when or "").strip() or "content actionability blocker clears"
    if status_text == "readiness_ready":
        return "all focus slots remain fresh and content-aligned"
    return "focus slot readiness becomes known"


def _focus_slot_readiness_gate_brief(*, status: str, ready_count: int, total_count: int) -> str:
    status_text = str(status or "").strip() or "-"
    return f"{status_text}:{ready_count}/{total_count}"


def _source_refresh_queue(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for index, row in enumerate(items, start=1):
        if not isinstance(row, dict):
            continue
        item = dict(row)
        item["rank"] = index
        queue.append(item)
    return queue


def _source_refresh_queue_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("rank") or "-"),
                    str(row.get("slot") or "-"),
                    str(row.get("symbol") or "-"),
                    str(row.get("action") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _source_refresh_checklist_state(*, action: str) -> str:
    action_text = str(action or "").strip()
    if action_text == "consider_refresh_before_promotion":
        return "refresh_recommended"
    if action_text == "refresh_source_before_use":
        return "refresh_required"
    if action_text == "do_not_promote_without_non_dry_run":
        return "blocked_on_non_dry_run_refresh"
    return "inspect_required"


def _source_refresh_checklist_blocker(
    *,
    source_kind: str,
    source_status: str,
    source_recency: str,
    source_age_minutes: Any,
) -> str:
    kind_text = str(source_kind or "").strip() or "source"
    status_text = str(source_status or "").strip() or "unknown"
    recency_text = str(source_recency or "").strip() or "unknown"
    age_suffix = f", age={source_age_minutes}m" if source_age_minutes not in (None, "") else ""
    return f"{kind_text} artifact is {status_text} and {recency_text}{age_suffix}"


def _source_refresh_checklist_done_when(
    *,
    symbol: str,
    source_kind: str,
    source_status: str,
    action: str,
) -> str:
    symbol_text = str(symbol or "").strip().upper() or "SYMBOL"
    kind_text = str(source_kind or "").strip() or "source"
    status_text = str(source_status or "").strip()
    action_text = str(action or "").strip()
    if status_text == "dry_run" or action_text == "refresh_source_before_use":
        return f"{symbol_text} receives a fresh non-dry-run {kind_text} artifact"
    if action_text == "consider_refresh_before_promotion":
        return f"{symbol_text} receives a fresh {kind_text} artifact before promotion"
    return f"{symbol_text} leaves operator_source_refresh_queue"


def _latest_artifact_from_path_hint(path_hint: str, reference_now: dt.datetime) -> Path | None:
    hint_text = str(path_hint or "").strip()
    if not hint_text:
        return None
    hint_path = Path(hint_text).expanduser()
    parent = hint_path.parent if str(hint_path.parent) not in ("", ".") else DEFAULT_REVIEW_DIR
    pattern = hint_path.name or "*"
    candidates = list(parent.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def _artifact_status_from_path(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(payload.get("status") or "").strip()


def _resolve_universe_file_path(*, review_dir: Path, payload: dict[str, Any], reference_now: dt.datetime) -> str:
    universe_text = str(payload.get("universe_file") or "").strip()
    if universe_text:
        universe_path = Path(universe_text).expanduser()
        if universe_path.exists():
            return str(universe_path)
    latest_universe = _latest_artifact_from_path_hint(
        str(review_dir / "*_hot_research_universe.json"),
        reference_now,
    )
    return str(latest_universe) if latest_universe else ""


def _crypto_route_refresh_batches(universe_file: str) -> list[str]:
    universe_text = str(universe_file or "").strip()
    if universe_text:
        universe_path = Path(universe_text).expanduser()
        if universe_path.exists():
            try:
                payload = json.loads(universe_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            batches = payload.get("batches", {})
            if isinstance(batches, dict):
                selected = [
                    str(name).strip()
                    for name in batches.keys()
                    if str(name).strip().startswith("crypto_")
                ]
                if selected:
                    return selected
    return list(DEFAULT_CRYPTO_ROUTE_REFRESH_BATCHES)


def _recipe_step_checkpoint_state(*, expected_status: str, current_artifact: str, current_status: str, current_recency: str) -> str:
    if not str(current_artifact or "").strip():
        return "missing"
    expected_text = str(expected_status or "").strip()
    current_status_text = str(current_status or "").strip()
    if expected_text and current_status_text and current_status_text != expected_text:
        return f"status_{current_status_text}"
    current_recency_text = str(current_recency or "").strip()
    if current_recency_text == "fresh":
        return "current"
    if current_recency_text == "carry_over":
        return "carry_over"
    if current_recency_text == "stale":
        return "stale"
    return "unknown"


def _recipe_step_with_checkpoint(step: dict[str, Any], *, reference_now: dt.datetime) -> dict[str, Any]:
    item = dict(step)
    current_path = _latest_artifact_from_path_hint(str(item.get("expected_artifact_path_hint") or ""), reference_now)
    current_artifact = str(current_path) if current_path else ""
    current_status = _artifact_status_from_path(current_path)
    current_as_of = _focus_slot_source_as_of(current_artifact)
    current_age_minutes = _focus_slot_source_age_minutes(reference_now=reference_now, source_as_of=current_as_of)
    current_recency = _focus_slot_source_recency(source_age_minutes=current_age_minutes)
    item["current_artifact"] = current_artifact
    item["current_status"] = current_status
    item["current_as_of"] = current_as_of
    item["current_age_minutes"] = current_age_minutes
    item["current_recency"] = current_recency
    item["checkpoint_state"] = _recipe_step_checkpoint_state(
        expected_status=str(item.get("expected_status") or ""),
        current_artifact=current_artifact,
        current_status=current_status,
        current_recency=current_recency,
    )
    return item


def _recipe_step_checkpoint_brief(steps: list[dict[str, Any]], limit: int = 4) -> str:
    parts: list[str] = []
    for step in steps[:limit]:
        if not isinstance(step, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(step.get("rank") or "-"),
                    str(step.get("checkpoint_state") or "-"),
                    str(step.get("expected_artifact_kind") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(steps) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(steps) - limit}"


def _recipe_pending_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("checkpoint_state") or "").strip() == "current":
            continue
        pending.append(dict(step))
    return pending


def _recipe_deferred_steps(
    steps: list[dict[str, Any]],
    *,
    recipe_gate_status: str,
) -> list[dict[str, Any]]:
    if str(recipe_gate_status or "").strip() != "deferred_by_cooldown":
        return []
    deferred: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("checkpoint_state") or "").strip() == "current":
            continue
        deferred.append(dict(step))
    return deferred


def _recipe_pending_brief(steps: list[dict[str, Any]], limit: int = 4) -> str:
    parts: list[str] = []
    for step in steps[:limit]:
        if not isinstance(step, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(step.get("rank") or "-"),
                    str(step.get("name") or "-"),
                    str(step.get("checkpoint_state") or "-"),
                    str(step.get("expected_artifact_kind") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(steps) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(steps) - limit}"


def _source_refresh_recipe(*, source_kind: str, source_artifact: str, reference_now: dt.datetime) -> dict[str, Any]:
    kind_text = str(source_kind or "").strip()
    artifact_text = str(source_artifact or "").strip()
    artifact_path = Path(artifact_text).expanduser() if artifact_text else Path()
    review_dir = artifact_path.parent if artifact_path.suffix else DEFAULT_REVIEW_DIR
    output_root = review_dir.parent if review_dir.name == "review" else DEFAULT_OUTPUT_ROOT
    context_path = review_dir / "NEXT_WINDOW_CONTEXT_LATEST.md"
    now_text = reference_now.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    recipe = {
        "script": "",
        "command_hint": "",
        "expected_status": "",
        "expected_artifact_kind": "",
        "expected_artifact_path_hint": "",
        "note": "",
        "followup_script": "",
        "followup_command_hint": "",
        "verify_hint": "",
        "steps_brief": "",
        "step_checkpoint_brief": "",
        "steps": [],
    }
    if kind_text != "crypto_route":
        return recipe
    refresh_script_path = Path(__file__).resolve().parent / "refresh_crypto_route_state.py"
    brief_script_path = Path(__file__).resolve().parent / "build_crypto_route_brief.py"
    operator_script_path = Path(__file__).resolve().parent / "build_crypto_route_operator_brief.py"
    script_path = Path(__file__).resolve().parent / "run_hot_universe_research.py"
    followup_script_path = Path(__file__).resolve().parent / "refresh_commodity_paper_execution_state.py"
    start_text = ""
    end_text = ""
    universe_file = ""
    if artifact_path.exists():
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        start_text = str(payload.get("start") or "").strip()
        end_text = str(payload.get("end") or "").strip()
        universe_file = _resolve_universe_file_path(
            review_dir=review_dir,
            payload=payload,
            reference_now=reference_now,
        )
    if not end_text:
        end_text = reference_now.date().isoformat()
    if not start_text:
        start_text = (reference_now.date() - dt.timedelta(days=3)).isoformat()
    if not universe_file:
        universe_file = _resolve_universe_file_path(
            review_dir=review_dir,
            payload={},
            reference_now=reference_now,
        )
    crypto_refresh_batches = _crypto_route_refresh_batches(universe_file)
    refresh_command = [
        "python3",
        str(refresh_script_path),
        "--review-dir",
        str(review_dir),
        "--output-root",
        str(output_root),
        "--now",
        now_text,
    ]
    brief_command = [
        "python3",
        str(brief_script_path),
        "--review-dir",
        str(review_dir),
        "--now",
        now_text,
    ]
    operator_command = [
        "python3",
        str(operator_script_path),
        "--review-dir",
        str(review_dir),
        "--now",
        now_text,
    ]
    command = [
        "python3",
        str(script_path),
        "--output-root",
        str(output_root),
        "--review-dir",
        str(review_dir),
        "--start",
        start_text,
        "--end",
        end_text,
        "--now",
        now_text,
    ]
    if universe_file:
        command.extend(["--universe-file", universe_file])
    for batch_name in crypto_refresh_batches:
        command.extend(["--batch", batch_name])
    followup_command = [
        "python3",
        str(followup_script_path),
        "--review-dir",
        str(review_dir),
        "--output-root",
        str(output_root),
        "--context-path",
        str(context_path),
    ]
    steps = [
        {
            "rank": 1,
            "name": "refresh_crypto_route_brief",
            "script": str(refresh_script_path),
            "command_hint": shlex.join(refresh_command),
            "expected_status": "ok",
            "expected_artifact_kind": "crypto_route_brief",
            "expected_artifact_path_hint": str(review_dir / "*_crypto_route_brief.json"),
        },
        {
            "rank": 2,
            "name": "refresh_crypto_route_operator_brief",
            "script": str(operator_script_path),
            "command_hint": shlex.join(operator_command),
            "expected_status": "ok",
            "expected_artifact_kind": "crypto_route_operator_brief",
            "expected_artifact_path_hint": str(review_dir / "*_crypto_route_operator_brief.json"),
        },
        {
            "rank": 3,
            "name": "refresh_hot_universe_research_embedding",
            "script": str(script_path),
            "command_hint": shlex.join(command),
            "expected_status": "ok",
            "expected_artifact_kind": "hot_universe_research",
            "expected_artifact_path_hint": str(review_dir / "*_hot_universe_research.json"),
        },
        {
            "rank": 4,
            "name": "refresh_commodity_handoff",
            "script": str(followup_script_path),
            "command_hint": shlex.join(followup_command),
            "expected_status": "ok",
            "expected_artifact_kind": "commodity_paper_execution_refresh",
            "expected_artifact_path_hint": str(review_dir / "*_commodity_paper_execution_refresh.json"),
        },
    ]
    steps = [_recipe_step_with_checkpoint(step, reference_now=reference_now) for step in steps]
    recipe["script"] = str(refresh_script_path)
    recipe["command_hint"] = shlex.join(refresh_command)
    recipe["expected_status"] = "ok"
    recipe["expected_artifact_kind"] = "crypto_route_brief"
    recipe["expected_artifact_path_hint"] = str(review_dir / "*_crypto_route_brief.json")
    recipe["note"] = "guarded entrypoint refreshes native crypto route sources before the remaining pipeline steps"
    recipe["followup_script"] = str(followup_script_path)
    recipe["followup_command_hint"] = shlex.join(followup_command)
    recipe["verify_hint"] = "rerun commodity refresh and confirm BNBUSDT leaves operator_source_refresh_queue"
    recipe["steps_brief"] = " | ".join(f"{step['rank']}:{step['name']}" for step in steps)
    recipe["step_checkpoint_brief"] = _recipe_step_checkpoint_brief(steps)
    recipe["steps"] = steps
    return recipe


def _source_refresh_checklist(items: list[dict[str, Any]], *, reference_now: dt.datetime) -> list[dict[str, Any]]:
    checklist: list[dict[str, Any]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        recipe = _source_refresh_recipe(
            source_kind=str(item.get("source_kind") or ""),
            source_artifact=str(item.get("source_artifact") or ""),
            reference_now=reference_now,
        )
        item["state"] = _source_refresh_checklist_state(action=str(item.get("action") or ""))
        item["blocker_detail"] = _source_refresh_checklist_blocker(
            source_kind=str(item.get("source_kind") or ""),
            source_status=str(item.get("source_status") or ""),
            source_recency=str(item.get("source_recency") or ""),
            source_age_minutes=item.get("source_age_minutes"),
        )
        item["done_when"] = _source_refresh_checklist_done_when(
            symbol=str(item.get("symbol") or ""),
            source_kind=str(item.get("source_kind") or ""),
            source_status=str(item.get("source_status") or ""),
            action=str(item.get("action") or ""),
        )
        item["recipe_script"] = str(recipe.get("script") or "")
        item["recipe_command_hint"] = str(recipe.get("command_hint") or "")
        item["recipe_expected_status"] = str(recipe.get("expected_status") or "")
        item["recipe_expected_artifact_kind"] = str(recipe.get("expected_artifact_kind") or "")
        item["recipe_expected_artifact_path_hint"] = str(recipe.get("expected_artifact_path_hint") or "")
        item["recipe_note"] = str(recipe.get("note") or "")
        item["recipe_followup_script"] = str(recipe.get("followup_script") or "")
        item["recipe_followup_command_hint"] = str(recipe.get("followup_command_hint") or "")
        item["recipe_verify_hint"] = str(recipe.get("verify_hint") or "")
        item["recipe_steps_brief"] = str(recipe.get("steps_brief") or "")
        item["recipe_step_checkpoint_brief"] = str(recipe.get("step_checkpoint_brief") or "")
        item["recipe_steps"] = list(recipe.get("steps") or [])
        checklist.append(item)
    return checklist


def _source_refresh_checklist_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("rank") or "-"),
                    str(row.get("state") or "-"),
                    str(row.get("symbol") or "-"),
                    str(row.get("action") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _source_refresh_checklist_verify_hint(*, symbol: str) -> str:
    symbol_text = str(symbol or "").strip().upper() or "SYMBOL"
    return f"rerun commodity refresh and confirm {symbol_text} leaves operator_source_refresh_queue"


def _fallback_symbol_list(values: list[str], fallback_symbol: str, fallback_count: int) -> list[str]:
    items = [str(v).strip().upper() for v in values if str(v).strip()]
    if items:
        return items
    symbol = str(fallback_symbol or "").strip().upper()
    if symbol and int(fallback_count or 0) > 0:
        return [symbol]
    return []


def _fmt_num(raw: Any) -> str:
    try:
        value = float(raw)
    except Exception:
        text = str(raw or "").strip()
        return text or "-"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _find_execution_item(items: list[dict[str, Any]], *, execution_id: str, symbol: str) -> dict[str, Any]:
    execution_text = str(execution_id or "").strip()
    symbol_text = str(symbol or "").strip().upper()
    for row in items:
        if not isinstance(row, dict):
            continue
        if execution_text and str(row.get("execution_id") or "").strip() == execution_text:
            return dict(row)
    for row in items:
        if not isinstance(row, dict):
            continue
        if symbol_text and str(row.get("symbol") or "").strip().upper() == symbol_text:
            return dict(row)
    return {}


def _extract_paper_evidence_summary(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    return {
        "paper_execution_evidence_present": bool(row.get("paper_execution_evidence_present")),
        "paper_open_date": row.get("paper_open_date"),
        "paper_signal_date": row.get("paper_signal_date"),
        "paper_execution_side": row.get("paper_execution_side"),
        "paper_execution_status": row.get("paper_execution_status"),
        "paper_runtime_mode": row.get("paper_runtime_mode"),
        "paper_order_mode": row.get("paper_order_mode"),
        "paper_size_pct": row.get("paper_size_pct"),
        "paper_risk_pct": row.get("paper_risk_pct"),
        "paper_entry_price": row.get("paper_entry_price"),
        "paper_stop_price": row.get("paper_stop_price"),
        "paper_target_price": row.get("paper_target_price"),
        "paper_quote_usdt": row.get("paper_quote_usdt"),
        "paper_execution_price_normalization_mode": row.get("paper_execution_price_normalization_mode"),
        "paper_proxy_price_normalized": row.get("paper_proxy_price_normalized"),
        "paper_signal_price_reference_kind": row.get("paper_signal_price_reference_kind"),
        "paper_signal_price_reference_source": row.get("paper_signal_price_reference_source"),
        "paper_signal_price_reference_provider": row.get("paper_signal_price_reference_provider"),
        "paper_signal_price_reference_symbol": row.get("paper_signal_price_reference_symbol"),
    }


def _paper_execution_status(row: dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return ""
    evidence_snapshot = row.get("paper_execution_evidence_snapshot")
    executed_plan = evidence_snapshot.get("executed_plan", {}) if isinstance(evidence_snapshot, dict) else {}
    return str(row.get("paper_execution_status") or executed_plan.get("status") or "").strip().upper()


def _commodity_focus_lifecycle_gate(
    *,
    area: str,
    action: str,
    symbol: str,
    focus_evidence_summary: dict[str, Any],
) -> dict[str, str]:
    gate = {
        "status": "not_needed",
        "brief": "not_needed:-",
        "blocker_detail": "commodity focus lifecycle gate is not needed when the current focus is not a paper review/retro/close-evidence item.",
        "done_when": "current focus becomes a commodity paper review/retro/close-evidence item before reassessing lifecycle state",
    }
    area_text = str(area or "").strip()
    action_text = str(action or "").strip()
    symbol_text = str(symbol or "").strip().upper() or "SYMBOL"
    if area_text not in {
        "commodity_execution_review",
        "commodity_execution_retro",
        "commodity_execution_close_evidence",
    }:
        return gate
    if action_text not in {
        "review_paper_execution",
        "review_paper_execution_retro",
        "wait_for_paper_execution_close_evidence",
    }:
        return gate
    if not isinstance(focus_evidence_summary, dict) or not focus_evidence_summary:
        gate["status"] = "awaiting_execution_evidence_snapshot"
        gate["brief"] = f"awaiting_execution_evidence_snapshot:{symbol_text}"
        gate["blocker_detail"] = "current commodity focus does not yet have a paper execution evidence snapshot."
        gate["done_when"] = f"{symbol_text} gains a paper execution evidence snapshot"
        return gate

    paper_status = str(focus_evidence_summary.get("paper_execution_status") or "").strip().upper()
    open_date = str(focus_evidence_summary.get("paper_open_date") or "").strip()
    if action_text in {"review_paper_execution_retro", "wait_for_paper_execution_close_evidence"} and paper_status == "OPEN":
        blocker = "paper execution evidence is present, but position is still OPEN"
        if open_date:
            blocker += f" since {open_date}"
        if action_text == "wait_for_paper_execution_close_evidence":
            blocker += "; waiting for close evidence"
        else:
            blocker += "; retro should wait for close evidence"
        gate["status"] = "open_position_wait_close_evidence"
        gate["brief"] = f"open_position_wait_close_evidence:{symbol_text}"
        gate["blocker_detail"] = blocker
        if action_text == "wait_for_paper_execution_close_evidence":
            gate["done_when"] = (
                f"{symbol_text} paper_execution_status leaves OPEN and close evidence becomes available"
            )
        else:
            gate["done_when"] = f"{symbol_text} paper_execution_status leaves OPEN and retro can evaluate close outcome"
        return gate

    gate["status"] = "ready_for_pending_focus_action"
    gate["brief"] = f"ready_for_pending_focus_action:{symbol_text}:{paper_status or '-'}"
    if action_text == "review_paper_execution_retro":
        gate["blocker_detail"] = "paper execution evidence is present; retro item still pending"
        gate["done_when"] = f"{symbol_text} leaves retro_pending_symbols"
    elif action_text == "wait_for_paper_execution_close_evidence":
        gate["blocker_detail"] = "paper execution close evidence is present; wait-close item can clear"
        gate["done_when"] = f"{symbol_text} leaves wait_for_paper_execution_close_evidence state"
    else:
        gate["blocker_detail"] = "paper execution evidence is present; review item still pending"
        gate["done_when"] = f"{symbol_text} leaves review_pending_symbols"
    return gate


def build_operator_brief(
    action_payload: dict[str, Any],
    crypto_payload: dict[str, Any],
    crypto_route_payload: dict[str, Any] | None = None,
    commodity_payload: dict[str, Any] | None = None,
    commodity_ticket_payload: dict[str, Any] | None = None,
    commodity_ticket_book_payload: dict[str, Any] | None = None,
    commodity_execution_preview_payload: dict[str, Any] | None = None,
    commodity_execution_artifact_payload: dict[str, Any] | None = None,
    commodity_execution_queue_payload: dict[str, Any] | None = None,
    commodity_execution_review_payload: dict[str, Any] | None = None,
    commodity_execution_retro_payload: dict[str, Any] | None = None,
    commodity_execution_gap_payload: dict[str, Any] | None = None,
    commodity_execution_bridge_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    partial_bridge_statuses = {
        "bridge_partially_bridged_missing_remainder",
        "bridge_partially_bridged_stale_remainder",
        "bridge_partially_bridged_proxy_remainder",
    }
    pending_review_statuses = {
        "paper-execution-review-pending",
        "paper-execution-review-pending-close-remainder",
        "paper-execution-review-pending-close-fill-remainder",
        "paper-execution-review-pending-fill-remainder",
        "paper-execution-close-evidence-pending",
        "paper-execution-close-evidence-pending-fill-remainder",
    }
    pending_retro_statuses = {
        "paper-execution-retro-pending",
        "paper-execution-retro-pending-fill-remainder",
        "paper-execution-retro-pending-close-remainder",
        "paper-execution-retro-pending-close-fill-remainder",
    }
    bridge_blocked_statuses = {
        "blocked_missing_directional_signal",
        "blocked_stale_directional_signal",
        "blocked_proxy_price_reference_only",
        *partial_bridge_statuses,
    }
    action_ladder = dict(action_payload.get("research_action_ladder") or {})
    embedded_crypto_route_brief = dict(crypto_payload.get("crypto_route_brief") or {})
    embedded_crypto_route_operator_brief = dict(crypto_payload.get("crypto_route_operator_brief") or {})
    dedicated_crypto_route_payload = dict(crypto_route_payload or {})
    dedicated_crypto_route_operator_brief: dict[str, Any] = {}
    dedicated_crypto_route_brief: dict[str, Any] = {}
    if dedicated_crypto_route_payload:
        if any(
            str(dedicated_crypto_route_payload.get(key) or "").strip()
            for key in (
                "operator_status",
                "route_stack_brief",
                "next_focus_symbol",
                "next_focus_action",
                "operator_text",
                "operator_lines",
                "focus_window_floor",
                "xlong_flow_window_floor",
                "price_state_window_floor",
            )
        ):
            dedicated_crypto_route_operator_brief = dedicated_crypto_route_payload
        if any(
            str(dedicated_crypto_route_payload.get(key) or "").strip()
            for key in (
                "brief_text",
                "brief_lines",
                "route_stack_brief",
                "next_focus_symbol",
                "next_focus_action",
            )
        ):
            dedicated_crypto_route_brief = dedicated_crypto_route_payload
    crypto_route_brief = (
        dedicated_crypto_route_brief
        or embedded_crypto_route_brief
        or dedicated_crypto_route_operator_brief
    )
    crypto_route_operator_brief = (
        dedicated_crypto_route_operator_brief
        or embedded_crypto_route_operator_brief
        or dedicated_crypto_route_brief
    )
    commodity_payload = dict(commodity_payload or {})

    focus_primary = [str(x).strip() for x in action_ladder.get("focus_primary_batches", []) if str(x).strip()]
    focus_regime = [str(x).strip() for x in action_ladder.get("focus_with_regime_filter_batches", []) if str(x).strip()]
    shadow_only = [str(x).strip() for x in action_ladder.get("shadow_only_batches", []) if str(x).strip()]
    research_queue = [str(x).strip() for x in action_ladder.get("research_queue_batches", []) if str(x).strip()]
    avoid = [str(x).strip() for x in action_ladder.get("avoid_batches", []) if str(x).strip()]

    commodity_status = str(commodity_payload.get("route_status") or "").strip()
    commodity_execution_mode = str(commodity_payload.get("execution_mode") or "").strip()
    commodity_route_stack = str(commodity_payload.get("route_stack_brief") or "").strip()
    commodity_primary = [str(x).strip() for x in commodity_payload.get("focus_primary_batches", []) if str(x).strip()]
    commodity_regime = [
        str(x).strip() for x in commodity_payload.get("focus_with_regime_filter_batches", []) if str(x).strip()
    ]
    commodity_shadow = [str(x).strip() for x in commodity_payload.get("shadow_only_batches", []) if str(x).strip()]
    commodity_leaders_primary = [
        str(x).strip().upper() for x in commodity_payload.get("leader_symbols_primary", []) if str(x).strip()
    ]
    commodity_leaders_regime = [
        str(x).strip().upper() for x in commodity_payload.get("leader_symbols_regime_filter", []) if str(x).strip()
    ]
    commodity_focus_batch = str(commodity_payload.get("next_focus_batch") or "").strip()
    commodity_focus_symbols = [
        str(x).strip().upper() for x in commodity_payload.get("next_focus_symbols", []) if str(x).strip()
    ]
    commodity_next_stage = str(commodity_payload.get("next_stage") or "").strip()
    commodity_ticket_payload = dict(commodity_ticket_payload or {})
    commodity_ticket_status = str(commodity_ticket_payload.get("ticket_status") or "").strip()
    commodity_ticket_stack = str(commodity_ticket_payload.get("ticket_stack_brief") or "").strip()
    commodity_paper_ready_batches = [
        str(x).strip() for x in commodity_ticket_payload.get("paper_ready_batches", []) if str(x).strip()
    ]
    commodity_ticket_focus_batch = str(commodity_ticket_payload.get("next_ticket_batch") or "").strip()
    commodity_ticket_focus_symbols = [
        str(x).strip().upper() for x in commodity_ticket_payload.get("next_ticket_symbols", []) if str(x).strip()
    ]
    commodity_ticket_count = len([row for row in commodity_ticket_payload.get("tickets", []) if isinstance(row, dict)])
    commodity_ticket_book_payload = dict(commodity_ticket_book_payload or {})
    commodity_ticket_book_status = str(commodity_ticket_book_payload.get("ticket_book_status") or "").strip()
    commodity_ticket_book_stack = str(commodity_ticket_book_payload.get("ticket_book_stack_brief") or "").strip()
    commodity_actionable_batches = [
        str(x).strip() for x in commodity_ticket_book_payload.get("actionable_batches", []) if str(x).strip()
    ]
    commodity_shadow_batches = [
        str(x).strip() for x in commodity_ticket_book_payload.get("shadow_batches", []) if str(x).strip()
    ]
    commodity_next_ticket_id = str(commodity_ticket_book_payload.get("next_ticket_id") or "").strip()
    commodity_next_ticket_symbol = str(commodity_ticket_book_payload.get("next_ticket_symbol") or "").strip().upper()
    commodity_actionable_ticket_count = int(commodity_ticket_book_payload.get("actionable_ticket_count", 0) or 0)
    commodity_execution_preview_payload = dict(commodity_execution_preview_payload or {})
    commodity_execution_preview_status = str(commodity_execution_preview_payload.get("execution_preview_status") or "").strip()
    commodity_execution_preview_stack = str(commodity_execution_preview_payload.get("preview_stack_brief") or "").strip()
    commodity_preview_ready_batches = [
        str(x).strip() for x in commodity_execution_preview_payload.get("preview_ready_batches", []) if str(x).strip()
    ]
    commodity_preview_shadow_batches = [
        str(x).strip() for x in commodity_execution_preview_payload.get("shadow_only_batches", []) if str(x).strip()
    ]
    commodity_next_execution_batch = str(commodity_execution_preview_payload.get("next_execution_batch") or "").strip()
    commodity_next_execution_symbols = [
        str(x).strip().upper() for x in commodity_execution_preview_payload.get("next_execution_symbols", []) if str(x).strip()
    ]
    commodity_next_execution_ticket_ids = [
        str(x).strip() for x in commodity_execution_preview_payload.get("next_execution_ticket_ids", []) if str(x).strip()
    ]
    commodity_next_execution_regime_gate = str(
        commodity_execution_preview_payload.get("next_execution_regime_gate") or ""
    ).strip()
    commodity_next_execution_weight_hint_sum = float(
        commodity_execution_preview_payload.get("next_execution_weight_hint_sum", 0.0) or 0.0
    )
    commodity_execution_artifact_payload = dict(commodity_execution_artifact_payload or {})
    commodity_execution_artifact_status = str(
        commodity_execution_artifact_payload.get("execution_artifact_status") or ""
    ).strip()
    commodity_execution_artifact_stack = str(
        commodity_execution_artifact_payload.get("execution_stack_brief") or ""
    ).strip()
    commodity_execution_batch = str(commodity_execution_artifact_payload.get("execution_batch") or "").strip()
    commodity_execution_symbols = [
        str(x).strip().upper() for x in commodity_execution_artifact_payload.get("execution_symbols", []) if str(x).strip()
    ]
    commodity_execution_ticket_ids = [
        str(x).strip() for x in commodity_execution_artifact_payload.get("execution_ticket_ids", []) if str(x).strip()
    ]
    commodity_execution_regime_gate = str(commodity_execution_artifact_payload.get("execution_regime_gate") or "").strip()
    commodity_execution_weight_hint_sum = float(
        commodity_execution_artifact_payload.get("execution_weight_hint_sum", 0.0) or 0.0
    )
    commodity_execution_item_count = int(commodity_execution_artifact_payload.get("execution_item_count", 0) or 0)
    commodity_actionable_execution_item_count = int(
        commodity_execution_artifact_payload.get("actionable_execution_item_count", 0) or 0
    )
    commodity_execution_queue_payload = dict(commodity_execution_queue_payload or {})
    commodity_execution_queue_status = str(
        commodity_execution_queue_payload.get("execution_queue_status") or ""
    ).strip()
    commodity_execution_queue_stack = str(
        commodity_execution_queue_payload.get("queue_stack_brief") or ""
    ).strip()
    commodity_execution_queue_batch = str(
        commodity_execution_queue_payload.get("execution_batch") or ""
    ).strip()
    commodity_queue_depth = int(commodity_execution_queue_payload.get("queue_depth", 0) or 0)
    commodity_actionable_queue_depth = int(
        commodity_execution_queue_payload.get("actionable_queue_depth", 0) or 0
    )
    commodity_next_queue_execution_id = str(
        commodity_execution_queue_payload.get("next_execution_id") or ""
    ).strip()
    commodity_next_queue_execution_symbol = str(
        commodity_execution_queue_payload.get("next_execution_symbol") or ""
    ).strip().upper()
    commodity_execution_review_payload = dict(commodity_execution_review_payload or {})
    commodity_execution_review_items = [
        row for row in commodity_execution_review_payload.get("review_items", []) if isinstance(row, dict)
    ]
    commodity_execution_review_status = str(
        commodity_execution_review_payload.get("execution_review_status") or ""
    ).strip()
    commodity_execution_review_stack = str(
        commodity_execution_review_payload.get("review_stack_brief") or ""
    ).strip()
    commodity_review_item_count = int(commodity_execution_review_payload.get("review_item_count", 0) or 0)
    commodity_actionable_review_item_count = int(
        commodity_execution_review_payload.get("actionable_review_item_count", 0) or 0
    )
    commodity_review_fill_evidence_pending_count = int(
        commodity_execution_review_payload.get("fill_evidence_pending_count", 0) or 0
    )
    commodity_next_review_execution_id = str(
        commodity_execution_review_payload.get("next_review_execution_id") or ""
    ).strip()
    commodity_next_review_execution_symbol = str(
        commodity_execution_review_payload.get("next_review_execution_symbol") or ""
    ).strip().upper()
    commodity_next_review_fill_evidence_execution_id = str(
        commodity_execution_review_payload.get("next_fill_evidence_execution_id") or ""
    ).strip()
    commodity_next_review_fill_evidence_execution_symbol = str(
        commodity_execution_review_payload.get("next_fill_evidence_execution_symbol") or ""
    ).strip().upper()
    derived_review_close_evidence_items = [
        row
        for row in commodity_execution_review_payload.get("review_items", [])
        if isinstance(row, dict)
        and (
            str(row.get("review_status") or "").strip() == "awaiting_paper_execution_close_evidence"
            or (_paper_execution_status(row) == "OPEN" and bool(row.get("paper_execution_evidence_present")))
        )
    ]
    commodity_review_close_evidence_pending_count = int(
        commodity_execution_review_payload.get("close_evidence_pending_count", len(derived_review_close_evidence_items))
        or 0
    )
    commodity_next_review_close_evidence_execution_id = str(
        commodity_execution_review_payload.get("next_close_evidence_execution_id")
        or (derived_review_close_evidence_items[0].get("execution_id") if derived_review_close_evidence_items else "")
        or ""
    ).strip()
    commodity_next_review_close_evidence_execution_symbol = str(
        commodity_execution_review_payload.get("next_close_evidence_execution_symbol")
        or (derived_review_close_evidence_items[0].get("symbol") if derived_review_close_evidence_items else "")
        or ""
    ).strip().upper()
    commodity_execution_retro_payload = dict(commodity_execution_retro_payload or {})
    commodity_execution_retro_items = [
        row for row in commodity_execution_retro_payload.get("retro_items", []) if isinstance(row, dict)
    ]
    commodity_execution_retro_status = str(
        commodity_execution_retro_payload.get("execution_retro_status") or ""
    ).strip()
    commodity_execution_retro_stack = str(
        commodity_execution_retro_payload.get("retro_stack_brief") or ""
    ).strip()
    commodity_retro_item_count = int(commodity_execution_retro_payload.get("retro_item_count", 0) or 0)
    commodity_actionable_retro_item_count = int(
        commodity_execution_retro_payload.get("actionable_retro_item_count", 0) or 0
    )
    commodity_retro_fill_evidence_pending_count = int(
        commodity_execution_retro_payload.get("fill_evidence_pending_count", 0) or 0
    )
    commodity_next_retro_execution_id = str(
        commodity_execution_retro_payload.get("next_retro_execution_id") or ""
    ).strip()
    commodity_next_retro_execution_symbol = str(
        commodity_execution_retro_payload.get("next_retro_execution_symbol") or ""
    ).strip().upper()
    commodity_next_retro_fill_evidence_execution_id = str(
        commodity_execution_retro_payload.get("next_fill_evidence_execution_id") or ""
    ).strip()
    commodity_next_retro_fill_evidence_execution_symbol = str(
        commodity_execution_retro_payload.get("next_fill_evidence_execution_symbol") or ""
    ).strip().upper()
    derived_close_evidence_items = [
        row
        for row in commodity_execution_retro_items
        if (
            str(row.get("retro_status") or "").strip() == "awaiting_paper_execution_close_evidence"
            or (
                _paper_execution_status(row) == "OPEN"
                and bool(row.get("paper_execution_evidence_present"))
            )
        )
    ]
    commodity_close_evidence_pending_count = int(
        commodity_execution_retro_payload.get("close_evidence_pending_count", len(derived_close_evidence_items)) or 0
    )
    commodity_next_close_evidence_execution_id = str(
        commodity_execution_retro_payload.get("next_close_evidence_execution_id")
        or (derived_close_evidence_items[0].get("execution_id") if derived_close_evidence_items else "")
        or ""
    ).strip()
    commodity_next_close_evidence_execution_symbol = str(
        commodity_execution_retro_payload.get("next_close_evidence_execution_symbol")
        or (derived_close_evidence_items[0].get("symbol") if derived_close_evidence_items else "")
        or ""
    ).strip().upper()
    commodity_review_pending_symbols = _fallback_symbol_list(
        commodity_execution_review_payload.get("review_pending_symbols", []),
        commodity_next_review_execution_symbol,
        commodity_actionable_review_item_count,
    )
    commodity_review_close_evidence_pending_symbols = _fallback_symbol_list(
        commodity_execution_review_payload.get("close_evidence_pending_symbols", []),
        commodity_next_review_close_evidence_execution_symbol,
        commodity_review_close_evidence_pending_count,
    )
    commodity_review_fill_evidence_pending_symbols = _fallback_symbol_list(
        commodity_execution_review_payload.get("fill_evidence_pending_symbols", []),
        commodity_next_review_fill_evidence_execution_symbol,
        commodity_review_fill_evidence_pending_count,
    )
    commodity_retro_pending_symbols = _fallback_symbol_list(
        commodity_execution_retro_payload.get("retro_pending_symbols", []),
        commodity_next_retro_execution_symbol,
        commodity_actionable_retro_item_count,
    )
    commodity_close_evidence_pending_symbols = _fallback_symbol_list(
        commodity_execution_retro_payload.get("close_evidence_pending_symbols", []),
        commodity_next_close_evidence_execution_symbol,
        commodity_close_evidence_pending_count,
    )
    commodity_retro_fill_evidence_pending_symbols = _fallback_symbol_list(
        commodity_execution_retro_payload.get("fill_evidence_pending_symbols", []),
        commodity_next_retro_fill_evidence_execution_symbol,
        commodity_retro_fill_evidence_pending_count,
    )
    if commodity_close_evidence_pending_symbols:
        close_evidence_symbol_set = set(commodity_close_evidence_pending_symbols)
        commodity_retro_pending_symbols = [
            symbol for symbol in commodity_retro_pending_symbols if symbol not in close_evidence_symbol_set
        ]
        commodity_actionable_retro_item_count = len(commodity_retro_pending_symbols)
        if (
            commodity_next_retro_execution_symbol
            and commodity_next_retro_execution_symbol in close_evidence_symbol_set
            and commodity_actionable_retro_item_count == 0
        ):
            commodity_next_retro_execution_id = ""
            commodity_next_retro_execution_symbol = ""
    if not commodity_close_evidence_pending_symbols and commodity_review_close_evidence_pending_symbols:
        commodity_close_evidence_pending_count = commodity_review_close_evidence_pending_count
        commodity_close_evidence_pending_symbols = list(commodity_review_close_evidence_pending_symbols)
        commodity_next_close_evidence_execution_id = commodity_next_review_close_evidence_execution_id
        commodity_next_close_evidence_execution_symbol = commodity_next_review_close_evidence_execution_symbol
    review_close_evidence_symbol_set = set(commodity_review_close_evidence_pending_symbols)
    if review_close_evidence_symbol_set:
        commodity_review_pending_symbols = [
            symbol for symbol in commodity_review_pending_symbols if symbol not in review_close_evidence_symbol_set
        ]
        commodity_actionable_review_item_count = len(commodity_review_pending_symbols)
        if (
            commodity_next_review_execution_symbol
            and commodity_next_review_execution_symbol in review_close_evidence_symbol_set
            and commodity_actionable_review_item_count == 0
        ):
            commodity_next_review_execution_id = ""
            commodity_next_review_execution_symbol = ""
    if commodity_actionable_review_item_count > 0:
        if commodity_review_close_evidence_pending_count > 0 and commodity_review_fill_evidence_pending_count > 0:
            commodity_execution_review_status = "paper-execution-review-pending-close-fill-remainder"
        elif commodity_review_close_evidence_pending_count > 0:
            commodity_execution_review_status = "paper-execution-review-pending-close-remainder"
        elif commodity_review_fill_evidence_pending_count > 0:
            commodity_execution_review_status = "paper-execution-review-pending-fill-remainder"
        else:
            commodity_execution_review_status = "paper-execution-review-pending"
    elif commodity_review_close_evidence_pending_count > 0:
        if commodity_review_fill_evidence_pending_count > 0:
            commodity_execution_review_status = "paper-execution-close-evidence-pending-fill-remainder"
        else:
            commodity_execution_review_status = "paper-execution-close-evidence-pending"
    elif commodity_review_fill_evidence_pending_count > 0:
        commodity_execution_review_status = "paper-execution-awaiting-fill-evidence"
    else:
        commodity_execution_review_status = "paper-execution-review-empty"
    commodity_execution_gap_payload = dict(commodity_execution_gap_payload or {})
    commodity_execution_gap_status = str(
        commodity_execution_gap_payload.get("gap_status") or ""
    ).strip()
    commodity_execution_gap_decision = str(
        commodity_execution_gap_payload.get("current_decision") or ""
    ).strip()
    commodity_execution_gap_reason_codes = [
        str(x).strip() for x in commodity_execution_gap_payload.get("gap_reason_codes", []) if str(x).strip()
    ]
    commodity_execution_gap_batch = str(
        commodity_execution_gap_payload.get("execution_batch") or ""
    ).strip()
    commodity_execution_gap_next_execution_id = str(
        commodity_execution_gap_payload.get("next_execution_id") or ""
    ).strip()
    commodity_execution_gap_next_execution_symbol = str(
        commodity_execution_gap_payload.get("next_execution_symbol") or ""
    ).strip().upper()
    commodity_execution_gap_root_cause_lines = [
        str(x).strip() for x in commodity_execution_gap_payload.get("root_cause_lines", []) if str(x).strip()
    ]
    commodity_execution_gap_recommended_actions = [
        str(x).strip() for x in commodity_execution_gap_payload.get("recommended_actions", []) if str(x).strip()
    ]
    commodity_stale_signal_watch_items = [
        dict(row)
        for row in commodity_execution_gap_payload.get("stale_directional_signal_watch_items", [])
        if isinstance(row, dict)
    ]
    commodity_execution_bridge_stale_signal_dates = {
        str(key).strip().upper(): str(value).strip()
        for key, value in dict(
            commodity_execution_gap_payload.get("queue_symbols_with_stale_directional_signal_dates", {})
        ).items()
        if str(key).strip() and str(value).strip()
    }
    commodity_execution_bridge_stale_signal_age_days = {
        str(key).strip().upper(): int(value)
        for key, value in dict(
            commodity_execution_gap_payload.get("queue_symbols_with_stale_directional_signal_age_days", {})
        ).items()
        if str(key).strip() and str(value).strip()
    }
    commodity_gap_focus_batch = (
        commodity_execution_gap_batch
        or commodity_execution_batch
        or commodity_execution_queue_batch
        or commodity_next_execution_batch
    )
    commodity_gap_focus_execution_id = commodity_execution_gap_next_execution_id or commodity_next_queue_execution_id
    commodity_gap_focus_symbol = commodity_execution_gap_next_execution_symbol or commodity_next_queue_execution_symbol
    commodity_execution_bridge_payload = dict(commodity_execution_bridge_payload or {})
    commodity_execution_bridge_status = str(
        commodity_execution_bridge_payload.get("bridge_status") or ""
    ).strip()
    commodity_execution_bridge_next_ready_id = str(
        commodity_execution_bridge_payload.get("next_ready_execution_id") or ""
    ).strip()
    commodity_execution_bridge_next_ready_symbol = str(
        commodity_execution_bridge_payload.get("next_ready_symbol") or ""
    ).strip().upper()
    commodity_execution_bridge_next_blocked_id = str(
        commodity_execution_bridge_payload.get("next_blocked_execution_id") or ""
    ).strip()
    commodity_execution_bridge_next_blocked_symbol = str(
        commodity_execution_bridge_payload.get("next_blocked_symbol") or ""
    ).strip().upper()
    commodity_execution_bridge_signal_missing_count = int(
        commodity_execution_bridge_payload.get("signal_missing_count", 0) or 0
    )
    commodity_execution_bridge_signal_stale_count = int(
        commodity_execution_bridge_payload.get("signal_stale_count", 0) or 0
    )
    commodity_execution_bridge_signal_proxy_price_only_count = int(
        commodity_execution_bridge_payload.get("signal_proxy_price_only_count", 0) or 0
    )
    commodity_execution_bridge_already_present_count = int(
        commodity_execution_bridge_payload.get("already_present_count", 0) or 0
    )
    commodity_execution_bridge_items = [
        row for row in commodity_execution_bridge_payload.get("bridge_items", []) if isinstance(row, dict)
    ]
    commodity_execution_bridge_already_bridged_symbols = [
        str(x).strip().upper()
        for x in commodity_execution_bridge_payload.get("already_bridged_symbols", [])
        if str(x).strip()
    ]
    if not commodity_execution_bridge_stale_signal_dates:
        commodity_execution_bridge_stale_signal_dates = {
            str(row.get("symbol") or "").strip().upper(): str(row.get("signal_date") or "").strip()
            for row in commodity_execution_bridge_items
            if str(row.get("symbol") or "").strip()
            and str(row.get("signal_date") or "").strip()
            and "stale_signal" in [str(x).strip() for x in row.get("bridge_reasons", []) if str(x).strip()]
        }
    if not commodity_execution_bridge_stale_signal_age_days:
        commodity_execution_bridge_stale_signal_age_days = {
            str(row.get("symbol") or "").strip().upper(): int(row.get("signal_age_days"))
            for row in commodity_execution_bridge_items
            if str(row.get("symbol") or "").strip()
            and row.get("signal_age_days") is not None
            and "stale_signal" in [str(x).strip() for x in row.get("bridge_reasons", []) if str(x).strip()]
        }
    commodity_next_fill_evidence_execution_id = (
        commodity_next_retro_fill_evidence_execution_id or commodity_next_review_fill_evidence_execution_id
    )
    commodity_next_fill_evidence_execution_symbol = (
        commodity_next_retro_fill_evidence_execution_symbol or commodity_next_review_fill_evidence_execution_symbol
    )
    commodity_fill_evidence_pending_count = max(
        commodity_review_fill_evidence_pending_count,
        commodity_retro_fill_evidence_pending_count,
    )
    bridge_ready = commodity_execution_bridge_status == "bridge_ready"
    actionable_review_pending = (
        commodity_execution_review_status in pending_review_statuses
        and commodity_actionable_review_item_count > 0
    )
    actionable_retro_pending = (
        commodity_execution_retro_status in pending_retro_statuses
        and commodity_actionable_retro_item_count > 0
    )

    crypto_status = str(crypto_route_operator_brief.get("operator_status") or crypto_route_brief.get("operator_status") or "").strip()
    route_stack = str(crypto_route_operator_brief.get("route_stack_brief") or crypto_route_brief.get("route_stack_brief") or "").strip()
    focus_symbol = str(crypto_route_operator_brief.get("next_focus_symbol") or crypto_route_brief.get("next_focus_symbol") or "").strip()
    focus_action = str(crypto_route_operator_brief.get("next_focus_action") or crypto_route_brief.get("next_focus_action") or "").strip()
    focus_reason = str(crypto_route_operator_brief.get("next_focus_reason") or crypto_route_brief.get("next_focus_reason") or "").strip()
    focus_gate = str(crypto_route_operator_brief.get("focus_window_gate") or crypto_route_brief.get("focus_window_gate") or "").strip()
    focus_window = str(crypto_route_operator_brief.get("focus_window_verdict") or crypto_route_brief.get("focus_window_verdict") or "").strip()
    focus_window_floor = str(crypto_route_operator_brief.get("focus_window_floor") or "").strip()
    price_state_window_floor = str(crypto_route_operator_brief.get("price_state_window_floor") or "").strip()
    comparative_window_takeaway = str(crypto_route_operator_brief.get("comparative_window_takeaway") or "").strip()
    xlong_flow_window_floor = str(crypto_route_operator_brief.get("xlong_flow_window_floor") or "").strip()
    xlong_comparative_window_takeaway = str(crypto_route_operator_brief.get("xlong_comparative_window_takeaway") or "").strip()
    focus_brief = str(crypto_route_operator_brief.get("focus_brief") or crypto_route_brief.get("focus_brief") or "").strip()
    next_retest_action = str(crypto_route_operator_brief.get("next_retest_action") or crypto_route_brief.get("next_retest_action") or "").strip()
    next_retest_reason = str(crypto_route_operator_brief.get("next_retest_reason") or crypto_route_brief.get("next_retest_reason") or "").strip()

    operator_status = "focus-commodities-plus-crypto-watch"
    if not focus_primary and not focus_regime:
        operator_status = "research-queue-only"
    if research_queue and operator_status == "research-queue-only":
        operator_status = "research-queue-plus-crypto-watch"
    if crypto_status.startswith("deploy"):
        operator_status = "focus-commodities-plus-crypto-deploy-watch"
        if not focus_primary and not focus_regime:
            operator_status = "research-queue-plus-crypto-deploy-watch"
    if commodity_status:
        operator_status = "commodity-paper-first"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-first-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-first-plus-research-queue"
    if commodity_ticket_status == "paper-ready" or commodity_ticket_book_status == "paper-ready":
        operator_status = "commodity-paper-ticket-ready"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-ticket-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-ticket-plus-research-queue"
    if commodity_execution_preview_status == "paper-execution-ready":
        operator_status = "commodity-paper-execution-ready"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-plus-research-queue"
    if commodity_execution_artifact_status == "paper-execution-artifact-ready":
        operator_status = "commodity-paper-execution-artifact-ready"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-artifact-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-artifact-plus-research-queue"
    if commodity_execution_queue_status == "paper-execution-queued":
        operator_status = "commodity-paper-execution-queued"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-queued-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-queued-plus-research-queue"
    if commodity_execution_review_status in pending_review_statuses:
        operator_status = "commodity-paper-execution-review-pending"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-review-pending-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-review-pending-plus-research-queue"
    if commodity_close_evidence_pending_count > 0:
        operator_status = "commodity-paper-execution-close-evidence-pending"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-close-evidence-pending-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-close-evidence-pending-plus-research-queue"
    if commodity_execution_retro_status in pending_retro_statuses:
        operator_status = "commodity-paper-execution-retro-pending"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-retro-pending-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-retro-pending-plus-research-queue"
    if commodity_execution_gap_status == "blocking_gap_active":
        operator_status = "commodity-paper-execution-gap-blocked"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-gap-blocked-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-gap-blocked-plus-research-queue"
    if commodity_execution_bridge_status in bridge_blocked_statuses:
        operator_status = "commodity-paper-execution-bridge-blocked"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-bridge-blocked-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-bridge-blocked-plus-research-queue"
    elif commodity_execution_bridge_status == "bridge_ready":
        operator_status = "commodity-paper-execution-bridge-ready"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-bridge-ready-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-bridge-ready-plus-research-queue"
    if not bridge_ready and actionable_review_pending:
        operator_status = "commodity-paper-execution-review-pending"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-review-pending-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-review-pending-plus-research-queue"
    if not bridge_ready and actionable_retro_pending:
        operator_status = "commodity-paper-execution-retro-pending"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-retro-pending-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-retro-pending-plus-research-queue"
    if not bridge_ready and commodity_close_evidence_pending_count > 0:
        operator_status = "commodity-paper-execution-close-evidence-pending"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-close-evidence-pending-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-close-evidence-pending-plus-research-queue"

    commodity_route_brief = f"{commodity_status}:{commodity_focus_batch}" if commodity_focus_batch else (commodity_status or "-")
    commodity_ticket_brief = (
        f"{commodity_ticket_status}:{commodity_ticket_focus_batch}"
        if commodity_ticket_focus_batch
        else (commodity_ticket_status or "-")
    )
    commodity_execution_brief = "-"
    if commodity_next_close_evidence_execution_symbol:
        commodity_execution_brief = f"close-evidence:{commodity_next_close_evidence_execution_symbol}"
    elif commodity_next_retro_execution_symbol:
        commodity_execution_brief = f"retro:{commodity_next_retro_execution_symbol}"
    elif commodity_next_review_execution_symbol:
        commodity_execution_brief = f"review:{commodity_next_review_execution_symbol}"
    elif commodity_next_queue_execution_symbol:
        commodity_execution_brief = f"queue:{commodity_next_queue_execution_symbol}"
    elif commodity_execution_symbols:
        commodity_execution_brief = f"execution:{commodity_execution_symbols[0]}"
    elif commodity_next_execution_batch:
        commodity_execution_brief = f"preview:{commodity_next_execution_batch}"
    elif commodity_next_ticket_symbol:
        commodity_execution_brief = f"ticket:{commodity_next_ticket_symbol}"
    if commodity_execution_gap_status == "blocking_gap_active":
        if commodity_gap_focus_symbol:
            commodity_execution_brief = f"gap:{commodity_gap_focus_symbol}"
        elif commodity_gap_focus_batch:
            commodity_execution_brief = f"gap:{commodity_gap_focus_batch}"
        else:
            commodity_execution_brief = "gap"
    if commodity_execution_bridge_status in {
        "blocked_proxy_price_reference_only",
        "bridge_partially_bridged_proxy_remainder",
    }:
        if commodity_execution_bridge_next_blocked_symbol:
            commodity_execution_brief = f"bridge-proxy:{commodity_execution_bridge_next_blocked_symbol}"
        else:
            commodity_execution_brief = "bridge-proxy"
    elif commodity_execution_bridge_status in {
        "blocked_stale_directional_signal",
        "bridge_partially_bridged_stale_remainder",
    }:
        if commodity_execution_bridge_next_blocked_symbol:
            commodity_execution_brief = f"bridge-stale:{commodity_execution_bridge_next_blocked_symbol}"
        else:
            commodity_execution_brief = "bridge-stale"
    elif commodity_execution_bridge_status in {
        "blocked_missing_directional_signal",
        "bridge_partially_bridged_missing_remainder",
    }:
        if commodity_execution_bridge_next_blocked_symbol:
            commodity_execution_brief = f"bridge-blocked:{commodity_execution_bridge_next_blocked_symbol}"
        else:
            commodity_execution_brief = "bridge-blocked"
    elif commodity_execution_bridge_status == "bridge_ready":
        if commodity_execution_bridge_next_ready_symbol:
            commodity_execution_brief = f"bridge-ready:{commodity_execution_bridge_next_ready_symbol}"
        else:
            commodity_execution_brief = "bridge-ready"
    if not bridge_ready and actionable_review_pending:
        commodity_execution_brief = (
            f"review:{commodity_next_review_execution_symbol}" if commodity_next_review_execution_symbol else "review"
        )
    if not bridge_ready and commodity_close_evidence_pending_count > 0:
        commodity_execution_brief = (
            f"close-evidence:{commodity_next_close_evidence_execution_symbol}"
            if commodity_next_close_evidence_execution_symbol
            else "close-evidence"
        )
    if not bridge_ready and actionable_retro_pending:
        commodity_execution_brief = (
            f"retro:{commodity_next_retro_execution_symbol}" if commodity_next_retro_execution_symbol else "retro"
        )
    if not bridge_ready and commodity_close_evidence_pending_count > 0:
        commodity_execution_brief = (
            f"close-evidence:{commodity_next_close_evidence_execution_symbol}"
            if commodity_next_close_evidence_execution_symbol
            else "close-evidence"
        )

    crypto_route_short_brief = f"{focus_symbol}:{focus_action}" if focus_symbol else (route_stack or "-")

    next_focus_area = "-"
    next_focus_target = "-"
    next_focus_symbol = "-"
    next_focus_action = "-"
    next_focus_reason = "-"
    secondary_focus_area = "-"
    secondary_focus_target = "-"
    secondary_focus_symbol = "-"
    secondary_focus_action = "-"
    secondary_focus_reason = "-"
    commodity_remainder_focus_area = "-"
    commodity_remainder_focus_target = "-"
    commodity_remainder_focus_symbol = "-"
    commodity_remainder_focus_action = "-"
    commodity_remainder_focus_reason = "-"
    commodity_remainder_focus_signal_date = "-"
    commodity_remainder_focus_signal_age_days: int | str = "-"
    commodity_stale_signal_watch_brief = _watch_items_text(commodity_stale_signal_watch_items)
    commodity_stale_signal_watch_next_execution_id = "-"
    commodity_stale_signal_watch_next_symbol = "-"
    commodity_stale_signal_watch_next_signal_date = "-"
    commodity_stale_signal_watch_next_signal_age_days: int | str = "-"
    followup_focus_area = "-"
    followup_focus_target = "-"
    followup_focus_symbol = "-"
    followup_focus_action = "-"
    followup_focus_reason = "-"
    if commodity_stale_signal_watch_items:
        commodity_stale_watch_head = dict(commodity_stale_signal_watch_items[0])
        commodity_stale_signal_watch_next_execution_id = str(
            commodity_stale_watch_head.get("execution_id") or ""
        ).strip() or "-"
        commodity_stale_signal_watch_next_symbol = str(
            commodity_stale_watch_head.get("symbol") or ""
        ).strip().upper() or "-"
        commodity_stale_signal_watch_next_signal_date = str(
            commodity_stale_watch_head.get("signal_date") or ""
        ).strip() or "-"
        commodity_stale_signal_watch_next_signal_age_days = (
            commodity_stale_watch_head.get("signal_age_days")
            if commodity_stale_watch_head.get("signal_age_days") not in (None, "")
            else "-"
        )

    if commodity_execution_bridge_status == "bridge_ready":
        next_focus_area = "commodity_execution_bridge"
        next_focus_target = commodity_execution_bridge_next_ready_id or commodity_gap_focus_execution_id or "-"
        next_focus_symbol = commodity_execution_bridge_next_ready_symbol or commodity_gap_focus_symbol or "-"
        next_focus_action = "apply_commodity_execution_bridge"
        next_focus_reason = "commodity_bridge_ready"
    elif commodity_close_evidence_pending_count > 0:
        next_focus_area = "commodity_execution_close_evidence"
        next_focus_target = (
            commodity_next_close_evidence_execution_id
            or commodity_next_retro_execution_id
            or commodity_next_close_evidence_execution_symbol
            or "-"
        )
        next_focus_symbol = (
            commodity_next_close_evidence_execution_symbol
            or commodity_next_retro_execution_symbol
            or "-"
        )
        next_focus_action = "wait_for_paper_execution_close_evidence"
        next_focus_reason = "paper_execution_close_evidence_pending"
    elif actionable_retro_pending:
        next_focus_area = "commodity_execution_retro"
        next_focus_target = commodity_next_retro_execution_id or commodity_next_retro_execution_symbol or "-"
        next_focus_symbol = commodity_next_retro_execution_symbol or "-"
        next_focus_action = "review_paper_execution_retro"
        next_focus_reason = "paper_execution_retro_pending"
    elif actionable_review_pending:
        next_focus_area = "commodity_execution_review"
        next_focus_target = commodity_next_review_execution_id or commodity_next_review_execution_symbol or "-"
        next_focus_symbol = commodity_next_review_execution_symbol or "-"
        next_focus_action = "review_paper_execution"
        next_focus_reason = "paper_execution_review_pending"
    elif commodity_execution_bridge_status in bridge_blocked_statuses:
        next_focus_area = "commodity_execution_bridge"
        next_focus_target = commodity_execution_bridge_next_blocked_id or commodity_gap_focus_execution_id or "-"
        next_focus_symbol = commodity_execution_bridge_next_blocked_symbol or commodity_gap_focus_symbol or "-"
        if commodity_execution_bridge_status in {
            "blocked_proxy_price_reference_only",
            "bridge_partially_bridged_proxy_remainder",
        }:
            next_focus_action = "normalize_commodity_execution_price_reference"
            next_focus_reason = "commodity_bridge_blocked_proxy_price_reference_only"
        else:
            next_focus_action = "restore_commodity_directional_signal"
        if commodity_execution_bridge_status in {
            "blocked_stale_directional_signal",
            "bridge_partially_bridged_stale_remainder",
        }:
            next_focus_reason = "commodity_bridge_blocked_stale_directional_signal"
        elif commodity_execution_bridge_status in {
            "blocked_missing_directional_signal",
            "bridge_partially_bridged_missing_remainder",
        }:
            next_focus_reason = "commodity_bridge_blocked_missing_directional_signal"
    elif commodity_execution_gap_status == "blocking_gap_active":
        next_focus_area = "commodity_execution_gap"
        next_focus_target = commodity_gap_focus_execution_id or commodity_gap_focus_batch or "-"
        next_focus_symbol = commodity_gap_focus_symbol or "-"
        next_focus_action = "resolve_commodity_paper_execution_gap"
        next_focus_reason = "commodity_execution_gap_active"
    elif commodity_execution_queue_status == "paper-execution-queued":
        next_focus_area = "commodity_execution_queue"
        next_focus_target = commodity_next_queue_execution_id or commodity_next_queue_execution_symbol or "-"
        next_focus_symbol = commodity_next_queue_execution_symbol or "-"
        next_focus_action = "inspect_paper_execution_queue"
        next_focus_reason = "paper_execution_queued"
    elif commodity_execution_artifact_status == "paper-execution-artifact-ready":
        next_focus_area = "commodity_execution_artifact"
        next_focus_target = commodity_execution_batch or "-"
        next_focus_symbol = commodity_execution_symbols[0] if commodity_execution_symbols else "-"
        next_focus_action = "inspect_paper_execution_artifact"
        next_focus_reason = "paper_execution_artifact_ready"
    elif commodity_execution_preview_status == "paper-execution-ready":
        next_focus_area = "commodity_execution_preview"
        next_focus_target = commodity_next_execution_batch or "-"
        next_focus_symbol = commodity_next_execution_symbols[0] if commodity_next_execution_symbols else "-"
        next_focus_action = "prepare_paper_execution"
        next_focus_reason = "paper_execution_ready"
    elif commodity_ticket_book_status == "paper-ready":
        next_focus_area = "commodity_ticket_book"
        next_focus_target = commodity_next_ticket_id or commodity_ticket_focus_batch or "-"
        next_focus_symbol = commodity_next_ticket_symbol or "-"
        next_focus_action = "build_paper_execution_artifact"
        next_focus_reason = "paper_ticket_ready"
    elif commodity_status:
        next_focus_area = "commodity_route"
        next_focus_target = commodity_focus_batch or "-"
        next_focus_symbol = commodity_focus_symbols[0] if commodity_focus_symbols else "-"
        next_focus_action = "build_paper_ticket_lane"
        next_focus_reason = commodity_status
    elif focus_symbol:
        next_focus_area = "crypto_route"
        next_focus_target = focus_symbol
        next_focus_symbol = focus_symbol
        next_focus_action = focus_action or "-"
        next_focus_reason = focus_reason or crypto_status or "-"
    elif research_queue:
        next_focus_area = "research_queue"
        next_focus_target = research_queue[0]
        next_focus_action = "continue_research_queue"
        next_focus_reason = "research_queue_active"

    if next_focus_area.startswith("commodity_") and focus_symbol:
        secondary_focus_area = "crypto_route"
        secondary_focus_target = focus_symbol
        secondary_focus_symbol = focus_symbol
        secondary_focus_action = focus_action or "-"
        secondary_focus_reason = "secondary_focus"
    elif next_focus_area == "crypto_route" and research_queue:
        secondary_focus_area = "research_queue"
        secondary_focus_target = research_queue[0]
        secondary_focus_symbol = research_queue[0]
        secondary_focus_action = "continue_research_queue"
        secondary_focus_reason = "secondary_focus"
    elif next_focus_area == "research_queue" and focus_symbol:
        secondary_focus_area = "crypto_route"
        secondary_focus_target = focus_symbol
        secondary_focus_symbol = focus_symbol
        secondary_focus_action = focus_action or "-"
        secondary_focus_reason = "secondary_focus"

    if commodity_next_fill_evidence_execution_id or commodity_next_fill_evidence_execution_symbol:
        commodity_remainder_focus_area = "commodity_fill_evidence"
        commodity_remainder_focus_target = (
            commodity_next_fill_evidence_execution_id or commodity_next_fill_evidence_execution_symbol or "-"
        )
        commodity_remainder_focus_symbol = commodity_next_fill_evidence_execution_symbol or "-"
        commodity_remainder_focus_action = "wait_for_paper_execution_fill_evidence"
        commodity_remainder_focus_reason = "paper_execution_fill_evidence_pending"
    elif commodity_execution_bridge_status in bridge_blocked_statuses:
        commodity_remainder_focus_area = "commodity_execution_bridge"
        commodity_remainder_focus_target = commodity_execution_bridge_next_blocked_id or commodity_gap_focus_execution_id or "-"
        commodity_remainder_focus_symbol = commodity_execution_bridge_next_blocked_symbol or commodity_gap_focus_symbol or "-"
        if commodity_execution_bridge_status in {
            "blocked_proxy_price_reference_only",
            "bridge_partially_bridged_proxy_remainder",
        }:
            commodity_remainder_focus_action = "normalize_commodity_execution_price_reference"
            commodity_remainder_focus_reason = "commodity_bridge_blocked_proxy_price_reference_only"
        else:
            commodity_remainder_focus_action = "restore_commodity_directional_signal"
            commodity_remainder_focus_reason = (
                "commodity_bridge_blocked_stale_directional_signal"
                if commodity_execution_bridge_status in {
                    "blocked_stale_directional_signal",
                    "bridge_partially_bridged_stale_remainder",
                }
                else "commodity_bridge_blocked_missing_directional_signal"
            )
    if commodity_remainder_focus_symbol and commodity_remainder_focus_symbol != "-":
        commodity_remainder_focus_signal_date = (
            commodity_execution_bridge_stale_signal_dates.get(commodity_remainder_focus_symbol) or "-"
        )
        commodity_remainder_focus_signal_age_days = (
            commodity_execution_bridge_stale_signal_age_days.get(commodity_remainder_focus_symbol, "-")
        )

    checklist_research_embedding_quality = _research_embedding_quality(action_payload)
    checklist_alignment_focus = _crypto_route_alignment_focus(
        {
            "next_focus_area": next_focus_area,
            "next_focus_symbol": next_focus_symbol,
            "next_focus_action": next_focus_action,
            "followup_focus_area": followup_focus_area,
            "followup_focus_symbol": followup_focus_symbol,
            "followup_focus_action": followup_focus_action,
            "secondary_focus_area": secondary_focus_area,
            "secondary_focus_symbol": secondary_focus_symbol,
            "secondary_focus_action": secondary_focus_action,
        }
    )
    checklist_crypto_route_alignment = _crypto_route_embedding_alignment(
        secondary_focus_area=str(checklist_alignment_focus.get("area") or ""),
        secondary_focus_symbol=str(checklist_alignment_focus.get("symbol") or ""),
        secondary_focus_action=str(checklist_alignment_focus.get("action") or ""),
        quality_brief=str(checklist_research_embedding_quality.get("brief") or ""),
        active_batches=list(checklist_research_embedding_quality.get("active_batches") or []),
    )
    checklist_crypto_route_alignment_recovery = _crypto_route_alignment_recovery_outcome(
        alignment_status=str(checklist_crypto_route_alignment.get("status") or ""),
        quality_status=str(checklist_research_embedding_quality.get("status") or ""),
        source_status=str(crypto_payload.get("status") or ""),
        source_payload=crypto_payload if isinstance(crypto_payload, dict) else {},
    )

    action_queue: list[dict[str, Any]] = []
    operator_action_queue: list[dict[str, Any]] = []
    operator_action_queue_brief = "-"

    commodity_focus_evidence_item_source = "-"
    commodity_focus_evidence_summary: dict[str, Any] = {}
    if next_focus_area in {"commodity_execution_retro", "commodity_execution_close_evidence"}:
        focus_item = _find_execution_item(
            commodity_execution_retro_items,
            execution_id=next_focus_target,
            symbol=next_focus_symbol,
        )
        if focus_item:
            commodity_focus_evidence_item_source = "retro"
            commodity_focus_evidence_summary = _extract_paper_evidence_summary(focus_item)
    elif next_focus_area == "commodity_execution_review":
        focus_item = _find_execution_item(
            commodity_execution_review_items,
            execution_id=next_focus_target,
            symbol=next_focus_symbol,
        )
        if focus_item:
            commodity_focus_evidence_item_source = "review"
            commodity_focus_evidence_summary = _extract_paper_evidence_summary(focus_item)
    commodity_focus_lifecycle = _commodity_focus_lifecycle_gate(
        area=next_focus_area,
        action=next_focus_action,
        symbol=next_focus_symbol,
        focus_evidence_summary=commodity_focus_evidence_summary,
    )
    if (
        next_focus_action == "review_paper_execution_retro"
        and str(commodity_focus_lifecycle.get("status") or "").strip() == "open_position_wait_close_evidence"
    ):
        next_focus_area = "commodity_execution_close_evidence"
        next_focus_action = "wait_for_paper_execution_close_evidence"
        next_focus_reason = "paper_execution_close_evidence_pending"
        commodity_execution_brief = (
            f"close-evidence:{next_focus_symbol}" if next_focus_symbol and next_focus_symbol != "-" else "close-evidence"
        )
        operator_status = "commodity-paper-execution-close-evidence-pending"
        if crypto_status.startswith("deploy"):
            operator_status = "commodity-paper-execution-close-evidence-pending-plus-crypto-deploy-watch"
        elif research_queue:
            operator_status = "commodity-paper-execution-close-evidence-pending-plus-research-queue"
        commodity_focus_lifecycle = _commodity_focus_lifecycle_gate(
            area=next_focus_area,
            action=next_focus_action,
            symbol=next_focus_symbol,
            focus_evidence_summary=commodity_focus_evidence_summary,
        )

    action_queue = []
    _append_action_queue_item(
        action_queue,
        area=next_focus_area,
        target=next_focus_target,
        symbol=next_focus_symbol,
        action=next_focus_action,
        reason=next_focus_reason,
    )
    _append_action_queue_item(
        action_queue,
        area=commodity_remainder_focus_area,
        target=commodity_remainder_focus_target,
        symbol=commodity_remainder_focus_symbol,
        action=commodity_remainder_focus_action,
        reason=commodity_remainder_focus_reason,
    )
    _append_action_queue_item(
        action_queue,
        area=secondary_focus_area,
        target=secondary_focus_target,
        symbol=secondary_focus_symbol,
        action=secondary_focus_action,
        reason=secondary_focus_reason,
    )
    followup_focus_area = "-"
    followup_focus_target = "-"
    followup_focus_symbol = "-"
    followup_focus_action = "-"
    followup_focus_reason = "-"
    if len(action_queue) >= 2:
        followup_row = dict(action_queue[1])
        followup_focus_area = str(followup_row.get("area") or "-")
        followup_focus_target = str(followup_row.get("target") or "-")
        followup_focus_symbol = str(followup_row.get("symbol") or "-")
        followup_focus_action = str(followup_row.get("action") or "-")
        followup_focus_reason = str(followup_row.get("reason") or "-")
    operator_action_queue = [
        {
            **row,
            "rank": idx,
        }
        for idx, row in enumerate(action_queue, start=1)
    ]
    operator_action_queue_brief = _action_queue_brief(operator_action_queue)

    operator_action_checklist = [
        {
            **row,
            "rank": idx,
            "state": _action_checklist_state(
                area=str(row.get("area") or ""),
                action=str(row.get("action") or ""),
                rank=idx,
            ),
            "blocker_detail": _action_checklist_blocker(
                area=str(row.get("area") or ""),
                action=str(row.get("action") or ""),
                symbol=str(row.get("symbol") or ""),
                reason=str(row.get("reason") or ""),
                stale_signal_dates=commodity_execution_bridge_stale_signal_dates,
                stale_signal_age_days=commodity_execution_bridge_stale_signal_age_days,
                focus_evidence_summary=commodity_focus_evidence_summary,
                focus_area=next_focus_area,
                focus_symbol=next_focus_symbol,
                focus_lifecycle_status=str(commodity_focus_lifecycle.get("status") or ""),
                focus_lifecycle_blocker_detail=str(commodity_focus_lifecycle.get("blocker_detail") or ""),
                crypto_route_alignment_status=str(checklist_crypto_route_alignment.get("status") or ""),
                crypto_route_alignment_recovery_status=str(
                    checklist_crypto_route_alignment_recovery.get("status") or ""
                ),
                crypto_route_alignment_recovery_zero_trade_batches=list(
                    checklist_crypto_route_alignment_recovery.get("zero_trade_batches") or []
                ),
            ),
            "done_when": _action_checklist_done_when(
                action=str(row.get("action") or ""),
                symbol=str(row.get("symbol") or ""),
                focus_lifecycle_status=str(commodity_focus_lifecycle.get("status") or ""),
                focus_lifecycle_done_when=str(commodity_focus_lifecycle.get("done_when") or ""),
                crypto_route_alignment_status=str(checklist_crypto_route_alignment.get("status") or ""),
                crypto_route_alignment_recovery_status=str(
                    checklist_crypto_route_alignment_recovery.get("status") or ""
                ),
            ),
        }
        for idx, row in enumerate(action_queue, start=1)
    ]
    operator_action_checklist_brief = _action_checklist_brief(operator_action_checklist)
    next_focus_state = "-"
    next_focus_blocker_detail = "-"
    next_focus_done_when = "-"
    followup_focus_state = "-"
    followup_focus_blocker_detail = "-"
    followup_focus_done_when = "-"
    secondary_focus_state = "-"
    secondary_focus_blocker_detail = "-"
    secondary_focus_done_when = "-"
    if operator_action_checklist:
        active_row = dict(operator_action_checklist[0])
        next_focus_state = str(active_row.get("state") or "-")
        next_focus_blocker_detail = str(active_row.get("blocker_detail") or "-")
        next_focus_done_when = str(active_row.get("done_when") or "-")
    if len(operator_action_checklist) >= 2:
        followup_checklist_row = dict(operator_action_checklist[1])
        followup_focus_state = str(followup_checklist_row.get("state") or "-")
        followup_focus_blocker_detail = str(followup_checklist_row.get("blocker_detail") or "-")
        followup_focus_done_when = str(followup_checklist_row.get("done_when") or "-")
    if len(operator_action_checklist) >= 3:
        secondary_checklist_row = dict(operator_action_checklist[2])
        secondary_focus_area = str(secondary_checklist_row.get("area") or secondary_focus_area or "-")
        secondary_focus_target = str(secondary_checklist_row.get("target") or secondary_focus_target or "-")
        secondary_focus_symbol = str(secondary_checklist_row.get("symbol") or secondary_focus_symbol or "-")
        secondary_focus_action = str(secondary_checklist_row.get("action") or secondary_focus_action or "-")
        secondary_focus_reason = str(secondary_checklist_row.get("reason") or secondary_focus_reason or "-")
        secondary_focus_state = str(secondary_checklist_row.get("state") or "-")
        secondary_focus_blocker_detail = str(secondary_checklist_row.get("blocker_detail") or "-")
        secondary_focus_done_when = str(secondary_checklist_row.get("done_when") or "-")
    commodity_execution_close_evidence_status = "not_active"
    commodity_execution_close_evidence_brief = "not_active:-"
    commodity_execution_close_evidence_target = "-"
    commodity_execution_close_evidence_symbol = "-"
    commodity_execution_close_evidence_blocker_detail = (
        "close-evidence lane is not the current operator focus."
    )
    commodity_execution_close_evidence_done_when = (
        "operator focus moves to a close-evidence wait item before reassessing"
    )
    if (
        next_focus_area == "commodity_execution_close_evidence"
        and next_focus_action == "wait_for_paper_execution_close_evidence"
    ):
        commodity_execution_close_evidence_status = "close_evidence_pending"
        commodity_execution_close_evidence_target = next_focus_target or "-"
        commodity_execution_close_evidence_symbol = next_focus_symbol or "-"
        commodity_execution_close_evidence_brief = (
            f"close_evidence_pending:{commodity_execution_close_evidence_symbol}"
            if commodity_execution_close_evidence_symbol and commodity_execution_close_evidence_symbol != "-"
            else "close_evidence_pending"
        )
        commodity_execution_close_evidence_blocker_detail = (
            next_focus_blocker_detail
            or str(commodity_focus_lifecycle.get("blocker_detail") or "").strip()
            or commodity_execution_close_evidence_blocker_detail
        )
        commodity_execution_close_evidence_done_when = (
            next_focus_done_when
            or str(commodity_focus_lifecycle.get("done_when") or "").strip()
            or commodity_execution_close_evidence_done_when
        )
    operator_focus_slots = [
        {
            "slot": "primary",
            "area": next_focus_area,
            "target": next_focus_target,
            "symbol": next_focus_symbol,
            "action": next_focus_action,
            "reason": next_focus_reason,
            "state": next_focus_state,
            "blocker_detail": next_focus_blocker_detail,
            "done_when": next_focus_done_when,
        },
        {
            "slot": "followup",
            "area": followup_focus_area,
            "target": followup_focus_target,
            "symbol": followup_focus_symbol,
            "action": followup_focus_action,
            "reason": followup_focus_reason,
            "state": followup_focus_state,
            "blocker_detail": followup_focus_blocker_detail,
            "done_when": followup_focus_done_when,
        },
        {
            "slot": "secondary",
            "area": secondary_focus_area,
            "target": secondary_focus_target,
            "symbol": secondary_focus_symbol,
            "action": secondary_focus_action,
            "reason": secondary_focus_reason,
            "state": secondary_focus_state,
            "blocker_detail": secondary_focus_blocker_detail,
            "done_when": secondary_focus_done_when,
        },
    ]
    operator_focus_slots_brief = _focus_slots_brief(operator_focus_slots)

    operator_stack_brief = f"commodity:{commodity_execution_brief} | crypto:{crypto_route_short_brief}"

    summary_lines = [
        f"status: {operator_status}",
        f"stack: {operator_stack_brief}",
        f"next-focus: {next_focus_area}:{next_focus_target}",
        f"next-focus-action: {next_focus_action}",
        f"next-focus-state: {next_focus_state}",
        f"next-focus-blocker: {next_focus_blocker_detail}",
        f"next-focus-done-when: {next_focus_done_when}",
        f"followup-focus: {followup_focus_area}:{followup_focus_target}:{followup_focus_action}",
        f"followup-focus-state: {followup_focus_state}",
        f"followup-focus-blocker: {followup_focus_blocker_detail}",
        f"followup-focus-done-when: {followup_focus_done_when}",
        f"secondary-focus: {secondary_focus_area}:{secondary_focus_target}:{secondary_focus_action}",
        f"secondary-focus-state: {secondary_focus_state}",
        f"secondary-focus-blocker: {secondary_focus_blocker_detail}",
        f"secondary-focus-done-when: {secondary_focus_done_when}",
        f"focus-slots: {operator_focus_slots_brief}",
        f"action-queue: {operator_action_queue_brief}",
        f"action-checklist: {operator_action_checklist_brief}",
        f"primary: {_list_text(focus_primary)}",
        f"regime-filter: {_list_text(focus_regime)}",
        f"research-queue: {_list_text(research_queue)}",
        f"shadow: {_list_text(shadow_only)}",
        f"commodity-status: {commodity_status or '-'}",
        f"commodity-route: {commodity_route_stack or '-'}",
        f"commodity-primary: {_list_text(commodity_primary)}",
        f"commodity-regime-filter: {_list_text(commodity_regime)}",
        f"commodity-shadow: {_list_text(commodity_shadow)}",
        f"commodity-focus: {commodity_focus_batch or '-'}",
        f"commodity-focus-symbols: {_list_text(commodity_focus_symbols)}",
        f"commodity-next-stage: {commodity_next_stage or '-'}",
        f"commodity-ticket-status: {commodity_ticket_status or '-'}",
        f"commodity-ticket-stack: {commodity_ticket_stack or '-'}",
        f"commodity-ticket-focus: {commodity_ticket_focus_batch or '-'}",
        f"commodity-ticket-symbols: {_list_text(commodity_ticket_focus_symbols)}",
        f"commodity-ticket-book-status: {commodity_ticket_book_status or '-'}",
        f"commodity-ticket-book-stack: {commodity_ticket_book_stack or '-'}",
        f"commodity-ticket-book-actionable: {_list_text(commodity_actionable_batches)}",
        f"commodity-ticket-book-shadow: {_list_text(commodity_shadow_batches)}",
        f"commodity-next-ticket-id: {commodity_next_ticket_id or '-'}",
        f"commodity-next-ticket-symbol: {commodity_next_ticket_symbol or '-'}",
        f"commodity-execution-preview-status: {commodity_execution_preview_status or '-'}",
        f"commodity-execution-preview-stack: {commodity_execution_preview_stack or '-'}",
        f"commodity-preview-ready: {_list_text(commodity_preview_ready_batches)}",
        f"commodity-preview-shadow: {_list_text(commodity_preview_shadow_batches)}",
        f"commodity-next-execution-batch: {commodity_next_execution_batch or '-'}",
        f"commodity-next-execution-symbols: {_list_text(commodity_next_execution_symbols)}",
        f"commodity-execution-artifact-status: {commodity_execution_artifact_status or '-'}",
        f"commodity-execution-stack: {commodity_execution_artifact_stack or '-'}",
        f"commodity-execution-batch: {commodity_execution_batch or '-'}",
        f"commodity-execution-symbols: {_list_text(commodity_execution_symbols)}",
        f"commodity-execution-queue-status: {commodity_execution_queue_status or '-'}",
        f"commodity-execution-queue-stack: {commodity_execution_queue_stack or '-'}",
        f"commodity-next-queue-execution-id: {commodity_next_queue_execution_id or '-'}",
        f"commodity-next-queue-execution-symbol: {commodity_next_queue_execution_symbol or '-'}",
        f"commodity-execution-review-status: {commodity_execution_review_status or '-'}",
        f"commodity-execution-retro-status: {commodity_execution_retro_status or '-'}",
        f"commodity-execution-gap-status: {commodity_execution_gap_status or '-'}",
        f"commodity-execution-gap-decision: {commodity_execution_gap_decision or '-'}",
        f"commodity-execution-gap-reasons: {_list_text(commodity_execution_gap_reason_codes)}",
        f"commodity-execution-bridge-status: {commodity_execution_bridge_status or '-'}",
        f"commodity-execution-bridge-signal-missing-count: {commodity_execution_bridge_signal_missing_count}",
        f"commodity-execution-bridge-signal-stale-count: {commodity_execution_bridge_signal_stale_count}",
        f"commodity-execution-bridge-signal-proxy-price-only-count: {commodity_execution_bridge_signal_proxy_price_only_count}",
        f"commodity-execution-bridge-stale-signal-dates: {_mapping_text(commodity_execution_bridge_stale_signal_dates)}",
        f"commodity-execution-bridge-stale-signal-age-days: {_mapping_text(commodity_execution_bridge_stale_signal_age_days)}",
        f"commodity-stale-signal-watch: {commodity_stale_signal_watch_brief}",
        f"commodity-stale-signal-watch-next-id: {commodity_stale_signal_watch_next_execution_id}",
        f"commodity-stale-signal-watch-next-symbol: {commodity_stale_signal_watch_next_symbol}",
        f"commodity-stale-signal-watch-next-signal-date: {commodity_stale_signal_watch_next_signal_date}",
        f"commodity-stale-signal-watch-next-signal-age-days: {commodity_stale_signal_watch_next_signal_age_days}",
        f"commodity-execution-bridge-already-present-count: {commodity_execution_bridge_already_present_count}",
        f"commodity-execution-review-stack: {commodity_execution_review_stack or '-'}",
        f"commodity-execution-retro-stack: {commodity_execution_retro_stack or '-'}",
        f"commodity-next-review-execution-id: {commodity_next_review_execution_id or '-'}",
        f"commodity-next-review-execution-symbol: {commodity_next_review_execution_symbol or '-'}",
        f"commodity-review-pending-symbols: {_list_text(commodity_review_pending_symbols)}",
        f"commodity-review-close-evidence-pending-count: {commodity_review_close_evidence_pending_count}",
        f"commodity-review-close-evidence-pending-symbols: {_list_text(commodity_review_close_evidence_pending_symbols)}",
        f"commodity-next-fill-evidence-execution-id: {commodity_next_fill_evidence_execution_id or '-'}",
        f"commodity-next-fill-evidence-execution-symbol: {commodity_next_fill_evidence_execution_symbol or '-'}",
        f"commodity-fill-evidence-pending-count: {commodity_fill_evidence_pending_count}",
        f"commodity-fill-evidence-pending-symbols: {_list_text(commodity_retro_fill_evidence_pending_symbols or commodity_review_fill_evidence_pending_symbols)}",
        f"commodity-close-evidence-pending-count: {commodity_close_evidence_pending_count}",
        f"commodity-next-close-evidence-execution-id: {commodity_next_close_evidence_execution_id or '-'}",
        f"commodity-next-close-evidence-execution-symbol: {commodity_next_close_evidence_execution_symbol or '-'}",
        f"commodity-close-evidence-pending-symbols: {_list_text(commodity_close_evidence_pending_symbols)}",
        f"commodity-retro-item-count: {commodity_retro_item_count}",
        f"commodity-actionable-retro-item-count: {commodity_actionable_retro_item_count}",
        f"commodity-next-retro-execution-id: {commodity_next_retro_execution_id or '-'}",
        f"commodity-next-retro-execution-symbol: {commodity_next_retro_execution_symbol or '-'}",
        f"commodity-retro-pending-symbols: {_list_text(commodity_retro_pending_symbols)}",
        f"commodity-remainder-focus: {commodity_remainder_focus_area}:{commodity_remainder_focus_target}:{commodity_remainder_focus_action}",
        f"commodity-remainder-focus-signal-date: {commodity_remainder_focus_signal_date}",
        f"commodity-remainder-focus-signal-age-days: {commodity_remainder_focus_signal_age_days}",
        f"crypto-status: {crypto_status or '-'}",
        f"crypto-routes: {route_stack or '-'}",
        f"crypto-focus: {focus_symbol or '-'}",
        f"crypto-action: {focus_action or '-'}",
    ]
    if commodity_focus_evidence_summary:
        summary_lines.append(
            "commodity-focus-paper-evidence: "
            + " ".join(
                [
                    f"source={commodity_focus_evidence_item_source}",
                    f"entry={_fmt_num(commodity_focus_evidence_summary.get('paper_entry_price'))}",
                    f"stop={_fmt_num(commodity_focus_evidence_summary.get('paper_stop_price'))}",
                    f"target={_fmt_num(commodity_focus_evidence_summary.get('paper_target_price'))}",
                    f"quote={_fmt_num(commodity_focus_evidence_summary.get('paper_quote_usdt'))}",
                    f"status={commodity_focus_evidence_summary.get('paper_execution_status') or '-'}",
                    f"ref={commodity_focus_evidence_summary.get('paper_signal_price_reference_source') or '-'}",
                ]
            )
        )
    if str(commodity_focus_lifecycle.get("brief") or "").strip():
        summary_lines.append(
            f"commodity-focus-lifecycle: {str(commodity_focus_lifecycle.get('brief') or '').strip()}"
        )
    if commodity_execution_close_evidence_brief:
        summary_lines.append(f"commodity-close-evidence: {commodity_execution_close_evidence_brief}")
    if commodity_execution_bridge_already_bridged_symbols:
        summary_lines.append(
            f"commodity-execution-bridge-already-bridged-symbols: {_list_text(commodity_execution_bridge_already_bridged_symbols)}"
        )
    if commodity_execution_gap_root_cause_lines:
        summary_lines.append(f"commodity-gap-root-cause: {commodity_execution_gap_root_cause_lines[0]}")
    if commodity_leaders_primary:
        summary_lines.append(f"commodity-leaders-primary: {_list_text(commodity_leaders_primary)}")
    if commodity_leaders_regime:
        summary_lines.append(f"commodity-leaders-regime: {_list_text(commodity_leaders_regime)}")
    if focus_gate:
        summary_lines.append(f"crypto-focus-gate: {focus_gate}")
    if focus_window:
        summary_lines.append(f"crypto-focus-window: {focus_window}")
    if focus_window_floor:
        summary_lines.append(f"crypto-focus-window-floor: {focus_window_floor}")
    if price_state_window_floor:
        summary_lines.append(f"crypto-price-window-floor: {price_state_window_floor}")
    if comparative_window_takeaway:
        summary_lines.append(f"crypto-window-note: {comparative_window_takeaway}")
    if xlong_flow_window_floor:
        summary_lines.append(f"crypto-xlong-flow-floor: {xlong_flow_window_floor}")
    if xlong_comparative_window_takeaway:
        summary_lines.append(f"crypto-xlong-note: {xlong_comparative_window_takeaway}")
    if next_retest_action:
        summary_lines.append(f"crypto-next-retest: {next_retest_action}")
    if focus_brief:
        summary_lines.append(f"crypto-focus-brief: {focus_brief}")

    return {
        "operator_status": operator_status,
        "operator_stack_brief": operator_stack_brief,
        "commodity_route_brief": commodity_route_brief,
        "commodity_ticket_brief": commodity_ticket_brief,
        "commodity_execution_brief": commodity_execution_brief,
        "crypto_route_short_brief": crypto_route_short_brief,
        "next_focus_area": next_focus_area,
        "next_focus_target": next_focus_target,
        "next_focus_symbol": next_focus_symbol,
        "next_focus_action": next_focus_action,
        "next_focus_reason": next_focus_reason,
        "next_focus_state": next_focus_state,
        "next_focus_blocker_detail": next_focus_blocker_detail,
        "next_focus_done_when": next_focus_done_when,
        "followup_focus_area": followup_focus_area,
        "followup_focus_target": followup_focus_target,
        "followup_focus_symbol": followup_focus_symbol,
        "followup_focus_action": followup_focus_action,
        "followup_focus_reason": followup_focus_reason,
        "followup_focus_state": followup_focus_state,
        "followup_focus_blocker_detail": followup_focus_blocker_detail,
        "followup_focus_done_when": followup_focus_done_when,
        "operator_focus_slots": operator_focus_slots,
        "operator_focus_slots_brief": operator_focus_slots_brief,
        "operator_action_queue": operator_action_queue,
        "operator_action_queue_brief": operator_action_queue_brief,
        "operator_action_checklist": operator_action_checklist,
        "operator_action_checklist_brief": operator_action_checklist_brief,
        "secondary_focus_area": secondary_focus_area,
        "secondary_focus_target": secondary_focus_target,
        "secondary_focus_symbol": secondary_focus_symbol,
        "secondary_focus_action": secondary_focus_action,
        "secondary_focus_reason": secondary_focus_reason,
        "secondary_focus_state": secondary_focus_state,
        "secondary_focus_blocker_detail": secondary_focus_blocker_detail,
        "secondary_focus_done_when": secondary_focus_done_when,
        "commodity_remainder_focus_area": commodity_remainder_focus_area,
        "commodity_remainder_focus_target": commodity_remainder_focus_target,
        "commodity_remainder_focus_symbol": commodity_remainder_focus_symbol,
        "commodity_remainder_focus_action": commodity_remainder_focus_action,
        "commodity_remainder_focus_reason": commodity_remainder_focus_reason,
        "commodity_remainder_focus_signal_date": commodity_remainder_focus_signal_date,
        "commodity_remainder_focus_signal_age_days": commodity_remainder_focus_signal_age_days,
        "commodity_focus_evidence_item_source": commodity_focus_evidence_item_source,
        "commodity_focus_evidence_summary": commodity_focus_evidence_summary,
        "commodity_focus_lifecycle_status": str(commodity_focus_lifecycle.get("status") or ""),
        "commodity_focus_lifecycle_brief": str(commodity_focus_lifecycle.get("brief") or ""),
        "commodity_focus_lifecycle_blocker_detail": str(commodity_focus_lifecycle.get("blocker_detail") or ""),
        "commodity_focus_lifecycle_done_when": str(commodity_focus_lifecycle.get("done_when") or ""),
        "commodity_execution_close_evidence_status": commodity_execution_close_evidence_status,
        "commodity_execution_close_evidence_brief": commodity_execution_close_evidence_brief,
        "commodity_execution_close_evidence_target": commodity_execution_close_evidence_target,
        "commodity_execution_close_evidence_symbol": commodity_execution_close_evidence_symbol,
        "commodity_execution_close_evidence_blocker_detail": commodity_execution_close_evidence_blocker_detail,
        "commodity_execution_close_evidence_done_when": commodity_execution_close_evidence_done_when,
        "focus_primary_batches": focus_primary,
        "focus_with_regime_filter_batches": focus_regime,
        "research_queue_batches": research_queue,
        "shadow_only_batches": shadow_only,
        "avoid_batches": avoid,
        "commodity_route_status": commodity_status,
        "commodity_execution_mode": commodity_execution_mode,
        "commodity_route_stack_brief": commodity_route_stack,
        "commodity_focus_primary_batches": commodity_primary,
        "commodity_focus_with_regime_filter_batches": commodity_regime,
        "commodity_shadow_only_batches": commodity_shadow,
        "commodity_leader_symbols_primary": commodity_leaders_primary,
        "commodity_leader_symbols_regime_filter": commodity_leaders_regime,
        "commodity_focus_batch": commodity_focus_batch,
        "commodity_focus_symbols": commodity_focus_symbols,
        "commodity_next_stage": commodity_next_stage,
        "commodity_ticket_status": commodity_ticket_status,
        "commodity_ticket_stack_brief": commodity_ticket_stack,
        "commodity_paper_ready_batches": commodity_paper_ready_batches,
        "commodity_ticket_focus_batch": commodity_ticket_focus_batch,
        "commodity_ticket_focus_symbols": commodity_ticket_focus_symbols,
        "commodity_ticket_count": commodity_ticket_count,
        "commodity_ticket_book_status": commodity_ticket_book_status,
        "commodity_ticket_book_stack_brief": commodity_ticket_book_stack,
        "commodity_actionable_batches": commodity_actionable_batches,
        "commodity_shadow_batches": commodity_shadow_batches,
        "commodity_next_ticket_id": commodity_next_ticket_id,
        "commodity_next_ticket_symbol": commodity_next_ticket_symbol,
        "commodity_actionable_ticket_count": commodity_actionable_ticket_count,
        "commodity_execution_preview_status": commodity_execution_preview_status,
        "commodity_execution_preview_stack_brief": commodity_execution_preview_stack,
        "commodity_preview_ready_batches": commodity_preview_ready_batches,
        "commodity_preview_shadow_batches": commodity_preview_shadow_batches,
        "commodity_next_execution_batch": commodity_next_execution_batch,
        "commodity_next_execution_symbols": commodity_next_execution_symbols,
        "commodity_next_execution_ticket_ids": commodity_next_execution_ticket_ids,
        "commodity_next_execution_regime_gate": commodity_next_execution_regime_gate,
        "commodity_next_execution_weight_hint_sum": commodity_next_execution_weight_hint_sum,
        "commodity_execution_artifact_status": commodity_execution_artifact_status,
        "commodity_execution_stack_brief": commodity_execution_artifact_stack,
        "commodity_execution_batch": commodity_execution_batch,
        "commodity_execution_symbols": commodity_execution_symbols,
        "commodity_execution_ticket_ids": commodity_execution_ticket_ids,
        "commodity_execution_regime_gate": commodity_execution_regime_gate,
        "commodity_execution_weight_hint_sum": commodity_execution_weight_hint_sum,
        "commodity_execution_item_count": commodity_execution_item_count,
        "commodity_actionable_execution_item_count": commodity_actionable_execution_item_count,
        "commodity_execution_queue_status": commodity_execution_queue_status,
        "commodity_execution_queue_stack_brief": commodity_execution_queue_stack,
        "commodity_queue_depth": commodity_queue_depth,
        "commodity_actionable_queue_depth": commodity_actionable_queue_depth,
        "commodity_next_queue_execution_id": commodity_next_queue_execution_id,
        "commodity_next_queue_execution_symbol": commodity_next_queue_execution_symbol,
        "commodity_execution_review_status": commodity_execution_review_status,
        "commodity_execution_review_stack_brief": commodity_execution_review_stack,
        "commodity_review_item_count": commodity_review_item_count,
        "commodity_actionable_review_item_count": commodity_actionable_review_item_count,
        "commodity_review_pending_symbols": commodity_review_pending_symbols,
        "commodity_review_close_evidence_pending_count": commodity_review_close_evidence_pending_count,
        "commodity_review_close_evidence_pending_symbols": commodity_review_close_evidence_pending_symbols,
        "commodity_review_fill_evidence_pending_count": commodity_review_fill_evidence_pending_count,
        "commodity_review_fill_evidence_pending_symbols": commodity_review_fill_evidence_pending_symbols,
        "commodity_next_review_execution_id": commodity_next_review_execution_id,
        "commodity_next_review_execution_symbol": commodity_next_review_execution_symbol,
        "commodity_next_review_close_evidence_execution_id": commodity_next_review_close_evidence_execution_id,
        "commodity_next_review_close_evidence_execution_symbol": commodity_next_review_close_evidence_execution_symbol,
        "commodity_next_review_fill_evidence_execution_id": commodity_next_review_fill_evidence_execution_id,
        "commodity_next_review_fill_evidence_execution_symbol": commodity_next_review_fill_evidence_execution_symbol,
        "commodity_execution_retro_status": commodity_execution_retro_status,
        "commodity_execution_retro_stack_brief": commodity_execution_retro_stack,
        "commodity_retro_item_count": commodity_retro_item_count,
        "commodity_actionable_retro_item_count": commodity_actionable_retro_item_count,
        "commodity_close_evidence_pending_count": commodity_close_evidence_pending_count,
        "commodity_close_evidence_pending_symbols": commodity_close_evidence_pending_symbols,
        "commodity_next_close_evidence_execution_id": commodity_next_close_evidence_execution_id,
        "commodity_next_close_evidence_execution_symbol": commodity_next_close_evidence_execution_symbol,
        "commodity_retro_pending_symbols": commodity_retro_pending_symbols,
        "commodity_retro_fill_evidence_pending_count": commodity_retro_fill_evidence_pending_count,
        "commodity_retro_fill_evidence_pending_symbols": commodity_retro_fill_evidence_pending_symbols,
        "commodity_next_retro_execution_id": commodity_next_retro_execution_id,
        "commodity_next_retro_execution_symbol": commodity_next_retro_execution_symbol,
        "commodity_next_retro_fill_evidence_execution_id": commodity_next_retro_fill_evidence_execution_id,
        "commodity_next_retro_fill_evidence_execution_symbol": commodity_next_retro_fill_evidence_execution_symbol,
        "commodity_fill_evidence_pending_count": commodity_fill_evidence_pending_count,
        "commodity_next_fill_evidence_execution_id": commodity_next_fill_evidence_execution_id,
        "commodity_next_fill_evidence_execution_symbol": commodity_next_fill_evidence_execution_symbol,
        "commodity_execution_gap_status": commodity_execution_gap_status,
        "commodity_execution_gap_decision": commodity_execution_gap_decision,
        "commodity_execution_gap_reason_codes": commodity_execution_gap_reason_codes,
        "commodity_execution_gap_batch": commodity_execution_gap_batch,
        "commodity_execution_gap_next_execution_id": commodity_execution_gap_next_execution_id,
        "commodity_execution_gap_next_execution_symbol": commodity_execution_gap_next_execution_symbol,
        "commodity_gap_focus_batch": commodity_gap_focus_batch,
        "commodity_gap_focus_execution_id": commodity_gap_focus_execution_id,
        "commodity_gap_focus_symbol": commodity_gap_focus_symbol,
        "commodity_execution_gap_root_cause_lines": commodity_execution_gap_root_cause_lines,
        "commodity_execution_gap_recommended_actions": commodity_execution_gap_recommended_actions,
        "commodity_stale_signal_watch_items": commodity_stale_signal_watch_items,
        "commodity_stale_signal_watch_brief": commodity_stale_signal_watch_brief,
        "commodity_stale_signal_watch_next_execution_id": commodity_stale_signal_watch_next_execution_id,
        "commodity_stale_signal_watch_next_symbol": commodity_stale_signal_watch_next_symbol,
        "commodity_stale_signal_watch_next_signal_date": commodity_stale_signal_watch_next_signal_date,
        "commodity_stale_signal_watch_next_signal_age_days": commodity_stale_signal_watch_next_signal_age_days,
        "commodity_execution_bridge_status": commodity_execution_bridge_status,
        "commodity_execution_bridge_next_ready_id": commodity_execution_bridge_next_ready_id,
        "commodity_execution_bridge_next_ready_symbol": commodity_execution_bridge_next_ready_symbol,
        "commodity_execution_bridge_next_blocked_id": commodity_execution_bridge_next_blocked_id,
        "commodity_execution_bridge_next_blocked_symbol": commodity_execution_bridge_next_blocked_symbol,
        "commodity_execution_bridge_signal_missing_count": commodity_execution_bridge_signal_missing_count,
        "commodity_execution_bridge_signal_stale_count": commodity_execution_bridge_signal_stale_count,
        "commodity_execution_bridge_signal_proxy_price_only_count": commodity_execution_bridge_signal_proxy_price_only_count,
        "commodity_execution_bridge_stale_signal_dates": commodity_execution_bridge_stale_signal_dates,
        "commodity_execution_bridge_stale_signal_age_days": commodity_execution_bridge_stale_signal_age_days,
        "commodity_execution_bridge_already_present_count": commodity_execution_bridge_already_present_count,
        "commodity_execution_bridge_already_bridged_symbols": commodity_execution_bridge_already_bridged_symbols,
        "crypto_route_status": crypto_status,
        "crypto_route_stack_brief": route_stack,
        "crypto_focus_symbol": focus_symbol,
        "crypto_focus_action": focus_action,
        "crypto_focus_reason": focus_reason,
        "crypto_focus_window_gate": focus_gate,
        "crypto_focus_window_verdict": focus_window,
        "crypto_focus_window_floor": focus_window_floor,
        "crypto_price_state_window_floor": price_state_window_floor,
        "crypto_comparative_window_takeaway": comparative_window_takeaway,
        "crypto_xlong_flow_window_floor": xlong_flow_window_floor,
        "crypto_xlong_comparative_window_takeaway": xlong_comparative_window_takeaway,
        "crypto_focus_brief": focus_brief,
        "crypto_next_retest_action": next_retest_action,
        "crypto_next_retest_reason": next_retest_reason,
        "summary_lines": summary_lines,
        "summary_text": " | ".join(summary_lines),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Hot Universe Operator Brief",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_action_artifact: `{payload.get('source_action_artifact') or payload.get('source_artifact') or ''}`",
        f"- source_action_status: `{payload.get('source_action_status') or payload.get('source_status') or ''}`",
        f"- source_commodity_artifact: `{payload.get('source_commodity_artifact') or ''}`",
        f"- source_commodity_status: `{payload.get('source_commodity_status') or ''}`",
        f"- source_commodity_ticket_artifact: `{payload.get('source_commodity_ticket_artifact') or ''}`",
        f"- source_commodity_ticket_status: `{payload.get('source_commodity_ticket_status') or ''}`",
        f"- source_commodity_ticket_book_artifact: `{payload.get('source_commodity_ticket_book_artifact') or ''}`",
        f"- source_commodity_ticket_book_status: `{payload.get('source_commodity_ticket_book_status') or ''}`",
        f"- source_commodity_execution_preview_artifact: `{payload.get('source_commodity_execution_preview_artifact') or ''}`",
        f"- source_commodity_execution_preview_status: `{payload.get('source_commodity_execution_preview_status') or ''}`",
        f"- source_commodity_execution_artifact: `{payload.get('source_commodity_execution_artifact') or ''}`",
        f"- source_commodity_execution_artifact_status: `{payload.get('source_commodity_execution_artifact_status') or ''}`",
        f"- source_commodity_execution_queue_artifact: `{payload.get('source_commodity_execution_queue_artifact') or ''}`",
        f"- source_commodity_execution_queue_status: `{payload.get('source_commodity_execution_queue_status') or ''}`",
        f"- source_commodity_execution_review_artifact: `{payload.get('source_commodity_execution_review_artifact') or ''}`",
        f"- source_commodity_execution_review_status: `{payload.get('source_commodity_execution_review_status') or ''}`",
        f"- source_commodity_execution_retro_artifact: `{payload.get('source_commodity_execution_retro_artifact') or ''}`",
        f"- source_commodity_execution_retro_status: `{payload.get('source_commodity_execution_retro_status') or ''}`",
        f"- source_commodity_execution_gap_artifact: `{payload.get('source_commodity_execution_gap_artifact') or ''}`",
        f"- source_commodity_execution_gap_status: `{payload.get('source_commodity_execution_gap_status') or ''}`",
        f"- source_commodity_execution_bridge_artifact: `{payload.get('source_commodity_execution_bridge_artifact') or ''}`",
        f"- source_commodity_execution_bridge_status: `{payload.get('source_commodity_execution_bridge_status') or ''}`",
        f"- source_crypto_artifact: `{payload.get('source_crypto_artifact') or payload.get('source_artifact') or ''}`",
        f"- source_crypto_status: `{payload.get('source_crypto_status') or payload.get('source_status') or ''}`",
        f"- source_crypto_route_artifact: `{payload.get('source_crypto_route_artifact') or ''}`",
        f"- source_crypto_route_status: `{payload.get('source_crypto_route_status') or ''}`",
        f"- operator_research_embedding_quality_status: `{payload.get('operator_research_embedding_quality_status') or ''}`",
        f"- operator_research_embedding_quality_brief: `{payload.get('operator_research_embedding_quality_brief') or ''}`",
        f"- operator_research_embedding_quality_blocker_detail: `{payload.get('operator_research_embedding_quality_blocker_detail') or ''}`",
        f"- operator_research_embedding_quality_done_when: `{payload.get('operator_research_embedding_quality_done_when') or ''}`",
        f"- operator_research_embedding_active_batches: `{_list_text(payload.get('operator_research_embedding_active_batches', []), limit=20)}`",
        f"- operator_research_embedding_avoid_batches: `{_list_text(payload.get('operator_research_embedding_avoid_batches', []), limit=20)}`",
        f"- operator_research_embedding_zero_trade_deprioritized_batches: `{_list_text(payload.get('operator_research_embedding_zero_trade_deprioritized_batches', []), limit=20)}`",
        f"- operator_crypto_route_alignment_focus_slot: `{payload.get('operator_crypto_route_alignment_focus_slot') or ''}`",
        f"- operator_crypto_route_alignment_status: `{payload.get('operator_crypto_route_alignment_status') or ''}`",
        f"- operator_crypto_route_alignment_brief: `{payload.get('operator_crypto_route_alignment_brief') or ''}`",
        f"- operator_crypto_route_alignment_blocker_detail: `{payload.get('operator_crypto_route_alignment_blocker_detail') or ''}`",
        f"- operator_crypto_route_alignment_done_when: `{payload.get('operator_crypto_route_alignment_done_when') or ''}`",
        f"- operator_crypto_route_alignment_recovery_status: `{payload.get('operator_crypto_route_alignment_recovery_status') or ''}`",
        f"- operator_crypto_route_alignment_recovery_brief: `{payload.get('operator_crypto_route_alignment_recovery_brief') or ''}`",
        f"- operator_crypto_route_alignment_recovery_blocker_detail: `{payload.get('operator_crypto_route_alignment_recovery_blocker_detail') or ''}`",
        f"- operator_crypto_route_alignment_recovery_done_when: `{payload.get('operator_crypto_route_alignment_recovery_done_when') or ''}`",
        f"- operator_crypto_route_alignment_recovery_failed_batch_count: `{payload.get('operator_crypto_route_alignment_recovery_failed_batch_count')}`",
        f"- operator_crypto_route_alignment_recovery_timed_out_batch_count: `{payload.get('operator_crypto_route_alignment_recovery_timed_out_batch_count')}`",
        f"- operator_crypto_route_alignment_recovery_zero_trade_batches: `{_list_text(payload.get('operator_crypto_route_alignment_recovery_zero_trade_batches', []), limit=20)}`",
        f"- operator_crypto_route_alignment_cooldown_status: `{payload.get('operator_crypto_route_alignment_cooldown_status') or ''}`",
        f"- operator_crypto_route_alignment_cooldown_brief: `{payload.get('operator_crypto_route_alignment_cooldown_brief') or ''}`",
        f"- operator_crypto_route_alignment_cooldown_blocker_detail: `{payload.get('operator_crypto_route_alignment_cooldown_blocker_detail') or ''}`",
        f"- operator_crypto_route_alignment_cooldown_done_when: `{payload.get('operator_crypto_route_alignment_cooldown_done_when') or ''}`",
        f"- operator_crypto_route_alignment_cooldown_last_research_end_date: `{payload.get('operator_crypto_route_alignment_cooldown_last_research_end_date') or ''}`",
        f"- operator_crypto_route_alignment_cooldown_next_eligible_end_date: `{payload.get('operator_crypto_route_alignment_cooldown_next_eligible_end_date') or ''}`",
        f"- operator_crypto_route_alignment_recipe_status: `{payload.get('operator_crypto_route_alignment_recipe_status') or ''}`",
        f"- operator_crypto_route_alignment_recipe_brief: `{payload.get('operator_crypto_route_alignment_recipe_brief') or ''}`",
        f"- operator_crypto_route_alignment_recipe_blocker_detail: `{payload.get('operator_crypto_route_alignment_recipe_blocker_detail') or ''}`",
        f"- operator_crypto_route_alignment_recipe_done_when: `{payload.get('operator_crypto_route_alignment_recipe_done_when') or ''}`",
        f"- operator_crypto_route_alignment_recipe_ready_on_date: `{payload.get('operator_crypto_route_alignment_recipe_ready_on_date') or ''}`",
        f"- operator_crypto_route_alignment_recipe_script: `{payload.get('operator_crypto_route_alignment_recipe_script') or ''}`",
        f"- operator_crypto_route_alignment_recipe_command_hint: `{payload.get('operator_crypto_route_alignment_recipe_command_hint') or ''}`",
        f"- operator_crypto_route_alignment_recipe_expected_status: `{payload.get('operator_crypto_route_alignment_recipe_expected_status') or ''}`",
        f"- operator_crypto_route_alignment_recipe_note: `{payload.get('operator_crypto_route_alignment_recipe_note') or ''}`",
        f"- operator_crypto_route_alignment_recipe_followup_script: `{payload.get('operator_crypto_route_alignment_recipe_followup_script') or ''}`",
        f"- operator_crypto_route_alignment_recipe_followup_command_hint: `{payload.get('operator_crypto_route_alignment_recipe_followup_command_hint') or ''}`",
        f"- operator_crypto_route_alignment_recipe_verify_hint: `{payload.get('operator_crypto_route_alignment_recipe_verify_hint') or ''}`",
        f"- operator_crypto_route_alignment_recipe_window_days: `{payload.get('operator_crypto_route_alignment_recipe_window_days')}`",
        f"- operator_crypto_route_alignment_recipe_target_batches: `{_list_text(payload.get('operator_crypto_route_alignment_recipe_target_batches', []), limit=20)}`",
        "",
        "## Focus",
        f"- operator_stack_brief: `{payload.get('operator_stack_brief') or ''}`",
        f"- next_focus_area: `{payload.get('next_focus_area') or ''}`",
        f"- next_focus_target: `{payload.get('next_focus_target') or ''}`",
        f"- next_focus_symbol: `{payload.get('next_focus_symbol') or ''}`",
        f"- next_focus_action: `{payload.get('next_focus_action') or ''}`",
        f"- next_focus_reason: `{payload.get('next_focus_reason') or ''}`",
        f"- next_focus_state: `{payload.get('next_focus_state') or ''}`",
        f"- next_focus_blocker_detail: `{payload.get('next_focus_blocker_detail') or ''}`",
        f"- next_focus_done_when: `{payload.get('next_focus_done_when') or ''}`",
        f"- followup_focus_area: `{payload.get('followup_focus_area') or ''}`",
        f"- followup_focus_target: `{payload.get('followup_focus_target') or ''}`",
        f"- followup_focus_symbol: `{payload.get('followup_focus_symbol') or ''}`",
        f"- followup_focus_action: `{payload.get('followup_focus_action') or ''}`",
        f"- followup_focus_reason: `{payload.get('followup_focus_reason') or ''}`",
        f"- followup_focus_state: `{payload.get('followup_focus_state') or ''}`",
        f"- followup_focus_blocker_detail: `{payload.get('followup_focus_blocker_detail') or ''}`",
        f"- followup_focus_done_when: `{payload.get('followup_focus_done_when') or ''}`",
        f"- operator_focus_slots_brief: `{payload.get('operator_focus_slots_brief') or ''}`",
        f"- operator_focus_slot_sources_brief: `{payload.get('operator_focus_slot_sources_brief') or ''}`",
        f"- operator_focus_slot_status_brief: `{payload.get('operator_focus_slot_status_brief') or ''}`",
        f"- operator_focus_slot_recency_brief: `{payload.get('operator_focus_slot_recency_brief') or ''}`",
        f"- operator_focus_slot_health_brief: `{payload.get('operator_focus_slot_health_brief') or ''}`",
        f"- operator_focus_slot_refresh_backlog_brief: `{payload.get('operator_focus_slot_refresh_backlog_brief') or ''}`",
        f"- operator_focus_slot_refresh_backlog_count: `{payload.get('operator_focus_slot_refresh_backlog_count')}`",
        f"- operator_focus_slot_refresh_backlog: `{json.dumps(payload.get('operator_focus_slot_refresh_backlog', []), ensure_ascii=False, sort_keys=True)}`",
        f"- operator_focus_slot_ready_count: `{payload.get('operator_focus_slot_ready_count')}`",
        f"- operator_focus_slot_total_count: `{payload.get('operator_focus_slot_total_count')}`",
        f"- operator_focus_slot_promotion_gate_brief: `{payload.get('operator_focus_slot_promotion_gate_brief') or ''}`",
        f"- operator_focus_slot_promotion_gate_status: `{payload.get('operator_focus_slot_promotion_gate_status') or ''}`",
        f"- operator_focus_slot_promotion_gate_blocker_detail: `{payload.get('operator_focus_slot_promotion_gate_blocker_detail') or ''}`",
        f"- operator_focus_slot_promotion_gate_done_when: `{payload.get('operator_focus_slot_promotion_gate_done_when') or ''}`",
        f"- operator_focus_slot_actionability_backlog_brief: `{payload.get('operator_focus_slot_actionability_backlog_brief') or ''}`",
        f"- operator_focus_slot_actionability_backlog_count: `{payload.get('operator_focus_slot_actionability_backlog_count')}`",
        f"- operator_focus_slot_actionability_backlog: `{json.dumps(payload.get('operator_focus_slot_actionability_backlog', []), ensure_ascii=False, sort_keys=True)}`",
        f"- operator_focus_slot_actionable_count: `{payload.get('operator_focus_slot_actionable_count')}`",
        f"- operator_focus_slot_actionability_gate_brief: `{payload.get('operator_focus_slot_actionability_gate_brief') or ''}`",
        f"- operator_focus_slot_actionability_gate_status: `{payload.get('operator_focus_slot_actionability_gate_status') or ''}`",
        f"- operator_focus_slot_actionability_gate_blocker_detail: `{payload.get('operator_focus_slot_actionability_gate_blocker_detail') or ''}`",
        f"- operator_focus_slot_actionability_gate_done_when: `{payload.get('operator_focus_slot_actionability_gate_done_when') or ''}`",
        f"- operator_focus_slot_readiness_gate_ready_count: `{payload.get('operator_focus_slot_readiness_gate_ready_count')}`",
        f"- operator_focus_slot_readiness_gate_brief: `{payload.get('operator_focus_slot_readiness_gate_brief') or ''}`",
        f"- operator_focus_slot_readiness_gate_status: `{payload.get('operator_focus_slot_readiness_gate_status') or ''}`",
        f"- operator_focus_slot_readiness_gate_blocking_gate: `{payload.get('operator_focus_slot_readiness_gate_blocking_gate') or ''}`",
        f"- operator_focus_slot_readiness_gate_blocker_detail: `{payload.get('operator_focus_slot_readiness_gate_blocker_detail') or ''}`",
        f"- operator_focus_slot_readiness_gate_done_when: `{payload.get('operator_focus_slot_readiness_gate_done_when') or ''}`",
        f"- operator_source_refresh_queue_brief: `{payload.get('operator_source_refresh_queue_brief') or ''}`",
        f"- operator_source_refresh_queue_count: `{payload.get('operator_source_refresh_queue_count')}`",
        f"- operator_source_refresh_queue: `{json.dumps(payload.get('operator_source_refresh_queue', []), ensure_ascii=False, sort_keys=True)}`",
        f"- operator_source_refresh_checklist_brief: `{payload.get('operator_source_refresh_checklist_brief') or ''}`",
        f"- operator_source_refresh_checklist: `{json.dumps(payload.get('operator_source_refresh_checklist', []), ensure_ascii=False, sort_keys=True)}`",
        f"- operator_source_refresh_pipeline_steps_brief: `{payload.get('operator_source_refresh_pipeline_steps_brief') or ''}`",
        f"- operator_source_refresh_pipeline_step_checkpoint_brief: `{payload.get('operator_source_refresh_pipeline_step_checkpoint_brief') or ''}`",
        f"- operator_source_refresh_pipeline_pending_brief: `{payload.get('operator_source_refresh_pipeline_pending_brief') or ''}`",
        f"- operator_source_refresh_pipeline_pending_count: `{payload.get('operator_source_refresh_pipeline_pending_count')}`",
        f"- operator_source_refresh_pipeline_head_rank: `{payload.get('operator_source_refresh_pipeline_head_rank') or ''}`",
        f"- operator_source_refresh_pipeline_head_name: `{payload.get('operator_source_refresh_pipeline_head_name') or ''}`",
        f"- operator_source_refresh_pipeline_head_checkpoint_state: `{payload.get('operator_source_refresh_pipeline_head_checkpoint_state') or ''}`",
        f"- operator_source_refresh_pipeline_head_expected_artifact_kind: `{payload.get('operator_source_refresh_pipeline_head_expected_artifact_kind') or ''}`",
        f"- operator_source_refresh_pipeline_head_current_artifact: `{payload.get('operator_source_refresh_pipeline_head_current_artifact') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_brief: `{payload.get('operator_source_refresh_pipeline_deferred_brief') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_count: `{payload.get('operator_source_refresh_pipeline_deferred_count')}`",
        f"- operator_source_refresh_pipeline_deferred_status: `{payload.get('operator_source_refresh_pipeline_deferred_status') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_until: `{payload.get('operator_source_refresh_pipeline_deferred_until') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_reason: `{payload.get('operator_source_refresh_pipeline_deferred_reason') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_head_rank: `{payload.get('operator_source_refresh_pipeline_deferred_head_rank') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_head_name: `{payload.get('operator_source_refresh_pipeline_deferred_head_name') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_head_checkpoint_state: `{payload.get('operator_source_refresh_pipeline_deferred_head_checkpoint_state') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_head_expected_artifact_kind: `{payload.get('operator_source_refresh_pipeline_deferred_head_expected_artifact_kind') or ''}`",
        f"- operator_source_refresh_pipeline_deferred_head_current_artifact: `{payload.get('operator_source_refresh_pipeline_deferred_head_current_artifact') or ''}`",
        f"- operator_focus_slots: `{json.dumps(payload.get('operator_focus_slots', []), ensure_ascii=False, sort_keys=True)}`",
        f"- next_focus_source_kind: `{payload.get('next_focus_source_kind') or ''}`",
        f"- next_focus_source_artifact: `{payload.get('next_focus_source_artifact') or ''}`",
        f"- next_focus_source_status: `{payload.get('next_focus_source_status') or ''}`",
        f"- next_focus_source_as_of: `{payload.get('next_focus_source_as_of') or ''}`",
        f"- next_focus_source_age_minutes: `{payload.get('next_focus_source_age_minutes')}`",
        f"- next_focus_source_recency: `{payload.get('next_focus_source_recency') or ''}`",
        f"- next_focus_source_health: `{payload.get('next_focus_source_health') or ''}`",
        f"- next_focus_source_refresh_action: `{payload.get('next_focus_source_refresh_action') or ''}`",
        f"- followup_focus_source_kind: `{payload.get('followup_focus_source_kind') or ''}`",
        f"- followup_focus_source_artifact: `{payload.get('followup_focus_source_artifact') or ''}`",
        f"- followup_focus_source_status: `{payload.get('followup_focus_source_status') or ''}`",
        f"- followup_focus_source_as_of: `{payload.get('followup_focus_source_as_of') or ''}`",
        f"- followup_focus_source_age_minutes: `{payload.get('followup_focus_source_age_minutes')}`",
        f"- followup_focus_source_recency: `{payload.get('followup_focus_source_recency') or ''}`",
        f"- followup_focus_source_health: `{payload.get('followup_focus_source_health') or ''}`",
        f"- followup_focus_source_refresh_action: `{payload.get('followup_focus_source_refresh_action') or ''}`",
        f"- secondary_focus_source_kind: `{payload.get('secondary_focus_source_kind') or ''}`",
        f"- secondary_focus_source_artifact: `{payload.get('secondary_focus_source_artifact') or ''}`",
        f"- secondary_focus_source_status: `{payload.get('secondary_focus_source_status') or ''}`",
        f"- secondary_focus_source_as_of: `{payload.get('secondary_focus_source_as_of') or ''}`",
        f"- secondary_focus_source_age_minutes: `{payload.get('secondary_focus_source_age_minutes')}`",
        f"- secondary_focus_source_recency: `{payload.get('secondary_focus_source_recency') or ''}`",
        f"- secondary_focus_source_health: `{payload.get('secondary_focus_source_health') or ''}`",
        f"- secondary_focus_source_refresh_action: `{payload.get('secondary_focus_source_refresh_action') or ''}`",
        f"- operator_focus_slot_refresh_head_slot: `{payload.get('operator_focus_slot_refresh_head_slot') or ''}`",
        f"- operator_focus_slot_refresh_head_symbol: `{payload.get('operator_focus_slot_refresh_head_symbol') or ''}`",
        f"- operator_focus_slot_refresh_head_action: `{payload.get('operator_focus_slot_refresh_head_action') or ''}`",
        f"- operator_focus_slot_refresh_head_health: `{payload.get('operator_focus_slot_refresh_head_health') or ''}`",
        f"- operator_source_refresh_next_slot: `{payload.get('operator_source_refresh_next_slot') or ''}`",
        f"- operator_source_refresh_next_symbol: `{payload.get('operator_source_refresh_next_symbol') or ''}`",
        f"- operator_source_refresh_next_action: `{payload.get('operator_source_refresh_next_action') or ''}`",
        f"- operator_source_refresh_next_source_kind: `{payload.get('operator_source_refresh_next_source_kind') or ''}`",
        f"- operator_source_refresh_next_source_health: `{payload.get('operator_source_refresh_next_source_health') or ''}`",
        f"- operator_source_refresh_next_source_artifact: `{payload.get('operator_source_refresh_next_source_artifact') or ''}`",
        f"- operator_source_refresh_next_state: `{payload.get('operator_source_refresh_next_state') or ''}`",
        f"- operator_source_refresh_next_blocker_detail: `{payload.get('operator_source_refresh_next_blocker_detail') or ''}`",
        f"- operator_source_refresh_next_done_when: `{payload.get('operator_source_refresh_next_done_when') or ''}`",
        f"- operator_source_refresh_next_recipe_script: `{payload.get('operator_source_refresh_next_recipe_script') or ''}`",
        f"- operator_source_refresh_next_recipe_command_hint: `{payload.get('operator_source_refresh_next_recipe_command_hint') or ''}`",
        f"- operator_source_refresh_next_recipe_expected_status: `{payload.get('operator_source_refresh_next_recipe_expected_status') or ''}`",
        f"- operator_source_refresh_next_recipe_expected_artifact_kind: `{payload.get('operator_source_refresh_next_recipe_expected_artifact_kind') or ''}`",
        f"- operator_source_refresh_next_recipe_expected_artifact_path_hint: `{payload.get('operator_source_refresh_next_recipe_expected_artifact_path_hint') or ''}`",
        f"- operator_source_refresh_next_recipe_note: `{payload.get('operator_source_refresh_next_recipe_note') or ''}`",
        f"- operator_source_refresh_next_recipe_followup_script: `{payload.get('operator_source_refresh_next_recipe_followup_script') or ''}`",
        f"- operator_source_refresh_next_recipe_followup_command_hint: `{payload.get('operator_source_refresh_next_recipe_followup_command_hint') or ''}`",
        f"- operator_source_refresh_next_recipe_verify_hint: `{payload.get('operator_source_refresh_next_recipe_verify_hint') or ''}`",
        f"- operator_source_refresh_next_recipe_steps_brief: `{payload.get('operator_source_refresh_next_recipe_steps_brief') or ''}`",
        f"- operator_source_refresh_next_recipe_step_checkpoint_brief: `{payload.get('operator_source_refresh_next_recipe_step_checkpoint_brief') or ''}`",
        f"- operator_source_refresh_next_recipe_steps: `{json.dumps(payload.get('operator_source_refresh_next_recipe_steps', []), ensure_ascii=False, sort_keys=True)}`",
        f"- operator_action_queue_brief: `{payload.get('operator_action_queue_brief') or ''}`",
        f"- operator_action_queue: `{json.dumps(payload.get('operator_action_queue', []), ensure_ascii=False, sort_keys=True)}`",
        f"- operator_action_checklist_brief: `{payload.get('operator_action_checklist_brief') or ''}`",
        f"- operator_action_checklist: `{json.dumps(payload.get('operator_action_checklist', []), ensure_ascii=False, sort_keys=True)}`",
        f"- secondary_focus_area: `{payload.get('secondary_focus_area') or ''}`",
        f"- secondary_focus_target: `{payload.get('secondary_focus_target') or ''}`",
        f"- secondary_focus_symbol: `{payload.get('secondary_focus_symbol') or ''}`",
        f"- secondary_focus_action: `{payload.get('secondary_focus_action') or ''}`",
        f"- secondary_focus_reason: `{payload.get('secondary_focus_reason') or ''}`",
        f"- secondary_focus_state: `{payload.get('secondary_focus_state') or ''}`",
        f"- secondary_focus_blocker_detail: `{payload.get('secondary_focus_blocker_detail') or ''}`",
        f"- secondary_focus_done_when: `{payload.get('secondary_focus_done_when') or ''}`",
        f"- commodity_remainder_focus_area: `{payload.get('commodity_remainder_focus_area') or ''}`",
        f"- commodity_remainder_focus_target: `{payload.get('commodity_remainder_focus_target') or ''}`",
        f"- commodity_remainder_focus_symbol: `{payload.get('commodity_remainder_focus_symbol') or ''}`",
        f"- commodity_remainder_focus_action: `{payload.get('commodity_remainder_focus_action') or ''}`",
        f"- commodity_remainder_focus_reason: `{payload.get('commodity_remainder_focus_reason') or ''}`",
        f"- commodity_remainder_focus_signal_date: `{payload.get('commodity_remainder_focus_signal_date') or ''}`",
        f"- commodity_remainder_focus_signal_age_days: `{payload.get('commodity_remainder_focus_signal_age_days')}`",
        f"- commodity_focus_evidence_item_source: `{payload.get('commodity_focus_evidence_item_source') or ''}`",
        f"- commodity_focus_evidence_summary: `{json.dumps(payload.get('commodity_focus_evidence_summary', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- commodity_focus_lifecycle_status: `{payload.get('commodity_focus_lifecycle_status') or ''}`",
        f"- commodity_focus_lifecycle_brief: `{payload.get('commodity_focus_lifecycle_brief') or ''}`",
        f"- commodity_focus_lifecycle_blocker_detail: `{payload.get('commodity_focus_lifecycle_blocker_detail') or ''}`",
        f"- commodity_focus_lifecycle_done_when: `{payload.get('commodity_focus_lifecycle_done_when') or ''}`",
        f"- commodity_execution_gap_status: `{payload.get('commodity_execution_gap_status') or ''}`",
        f"- commodity_execution_gap_decision: `{payload.get('commodity_execution_gap_decision') or ''}`",
        f"- commodity_execution_gap_reason_codes: `{_list_text(payload.get('commodity_execution_gap_reason_codes', []), limit=20)}`",
        f"- commodity_execution_bridge_status: `{payload.get('commodity_execution_bridge_status') or ''}`",
        f"- commodity_execution_bridge_next_blocked_id: `{payload.get('commodity_execution_bridge_next_blocked_id') or ''}`",
        f"- commodity_execution_bridge_next_blocked_symbol: `{payload.get('commodity_execution_bridge_next_blocked_symbol') or ''}`",
        f"- commodity_execution_bridge_stale_signal_dates: `{_mapping_text(payload.get('commodity_execution_bridge_stale_signal_dates', {}), limit=20)}`",
        f"- commodity_execution_bridge_stale_signal_age_days: `{_mapping_text(payload.get('commodity_execution_bridge_stale_signal_age_days', {}), limit=20)}`",
        f"- commodity_stale_signal_watch_brief: `{payload.get('commodity_stale_signal_watch_brief') or ''}`",
        f"- commodity_stale_signal_watch_next_execution_id: `{payload.get('commodity_stale_signal_watch_next_execution_id') or ''}`",
        f"- commodity_stale_signal_watch_next_symbol: `{payload.get('commodity_stale_signal_watch_next_symbol') or ''}`",
        f"- commodity_stale_signal_watch_next_signal_date: `{payload.get('commodity_stale_signal_watch_next_signal_date') or ''}`",
        f"- commodity_stale_signal_watch_next_signal_age_days: `{payload.get('commodity_stale_signal_watch_next_signal_age_days')}`",
        f"- commodity_execution_bridge_already_present_count: `{payload.get('commodity_execution_bridge_already_present_count') or 0}`",
        f"- commodity_execution_bridge_already_bridged_symbols: `{_list_text(payload.get('commodity_execution_bridge_already_bridged_symbols', []), limit=20)}`",
        f"- commodity_next_fill_evidence_execution_id: `{payload.get('commodity_next_fill_evidence_execution_id') or ''}`",
        f"- commodity_next_fill_evidence_execution_symbol: `{payload.get('commodity_next_fill_evidence_execution_symbol') or ''}`",
        f"- commodity_fill_evidence_pending_count: `{payload.get('commodity_fill_evidence_pending_count') or 0}`",
        f"- commodity_review_pending_symbols: `{_list_text(payload.get('commodity_review_pending_symbols', []), limit=20)}`",
        f"- commodity_review_close_evidence_pending_symbols: `{_list_text(payload.get('commodity_review_close_evidence_pending_symbols', []), limit=20)}`",
        f"- commodity_retro_pending_symbols: `{_list_text(payload.get('commodity_retro_pending_symbols', []), limit=20)}`",
        f"- commodity_close_evidence_pending_symbols: `{_list_text(payload.get('commodity_close_evidence_pending_symbols', []), limit=20)}`",
        f"- commodity_review_fill_evidence_pending_symbols: `{_list_text(payload.get('commodity_review_fill_evidence_pending_symbols', []), limit=20)}`",
        f"- commodity_retro_fill_evidence_pending_symbols: `{_list_text(payload.get('commodity_retro_fill_evidence_pending_symbols', []), limit=20)}`",
        "",
        "## Summary",
    ]
    for line in payload.get("summary_lines", []):
        lines.append(f"- {line}")
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an operator-focused brief from the latest hot-universe research artifact.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--commodity-execution-queue-json", default="")
    parser.add_argument("--commodity-execution-review-json", default="")
    parser.add_argument("--commodity-execution-retro-json", default="")
    parser.add_argument("--commodity-execution-gap-json", default="")
    parser.add_argument("--commodity-execution-bridge-json", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = parse_now(args.now)

    action_source_path = latest_hot_universe_action_source(review_dir, runtime_now)
    crypto_source_path = latest_hot_universe_crypto_source(review_dir, runtime_now)
    crypto_route_source_path = latest_crypto_route_focus_source(review_dir, runtime_now)
    commodity_source_path = latest_commodity_execution_lane_source(review_dir, runtime_now)
    commodity_ticket_source_path = latest_commodity_paper_ticket_lane_source(review_dir, runtime_now)
    commodity_ticket_book_source_path = latest_commodity_paper_ticket_book_source(review_dir, runtime_now)
    commodity_execution_preview_source_path = latest_commodity_paper_execution_preview_source(review_dir, runtime_now)
    commodity_execution_artifact_source_path = latest_commodity_paper_execution_artifact_source(review_dir, runtime_now)
    commodity_execution_queue_source_path = resolve_explicit_source(args.commodity_execution_queue_json) or latest_commodity_paper_execution_queue_source(review_dir, runtime_now)
    commodity_execution_review_source_path = resolve_explicit_source(args.commodity_execution_review_json) or latest_commodity_paper_execution_review_source(review_dir, runtime_now)
    commodity_execution_retro_source_path = resolve_explicit_source(args.commodity_execution_retro_json) or latest_commodity_paper_execution_retro_source(review_dir, runtime_now)
    commodity_execution_gap_source_path = resolve_explicit_source(args.commodity_execution_gap_json) or latest_commodity_paper_execution_gap_source(review_dir, runtime_now)
    commodity_execution_bridge_source_path = resolve_explicit_source(args.commodity_execution_bridge_json) or latest_commodity_paper_execution_bridge_source(review_dir, runtime_now)
    action_source_payload = json.loads(action_source_path.read_text(encoding="utf-8"))
    crypto_source_payload = json.loads(crypto_source_path.read_text(encoding="utf-8"))
    crypto_route_source_payload = (
        json.loads(crypto_route_source_path.read_text(encoding="utf-8"))
        if crypto_route_source_path and crypto_route_source_path.exists()
        else None
    )
    commodity_source_payload = (
        _normalize_commodity_payload(json.loads(commodity_source_path.read_text(encoding="utf-8")))
        if commodity_source_path
        else None
    )
    commodity_ticket_source_payload = (
        json.loads(commodity_ticket_source_path.read_text(encoding="utf-8"))
        if commodity_ticket_source_path
        else None
    )
    commodity_ticket_book_source_payload = (
        json.loads(commodity_ticket_book_source_path.read_text(encoding="utf-8"))
        if commodity_ticket_book_source_path
        else None
    )
    commodity_execution_preview_source_payload = (
        json.loads(commodity_execution_preview_source_path.read_text(encoding="utf-8"))
        if commodity_execution_preview_source_path
        else None
    )
    commodity_execution_artifact_source_payload = (
        json.loads(commodity_execution_artifact_source_path.read_text(encoding="utf-8"))
        if commodity_execution_artifact_source_path
        else None
    )
    commodity_execution_queue_source_payload = (
        json.loads(commodity_execution_queue_source_path.read_text(encoding="utf-8"))
        if commodity_execution_queue_source_path
        else None
    )
    commodity_execution_review_source_payload = (
        json.loads(commodity_execution_review_source_path.read_text(encoding="utf-8"))
        if commodity_execution_review_source_path
        else None
    )
    commodity_execution_retro_source_payload = (
        json.loads(commodity_execution_retro_source_path.read_text(encoding="utf-8"))
        if commodity_execution_retro_source_path
        else None
    )
    commodity_execution_gap_source_payload = (
        json.loads(commodity_execution_gap_source_path.read_text(encoding="utf-8"))
        if commodity_execution_gap_source_path
        else None
    )
    commodity_execution_bridge_source_payload = (
        json.loads(commodity_execution_bridge_source_path.read_text(encoding="utf-8"))
        if commodity_execution_bridge_source_path
        else None
    )
    research_embedding_quality = _research_embedding_quality(action_source_payload)
    brief = build_operator_brief(
        action_source_payload,
        crypto_source_payload,
        crypto_route_source_payload,
        commodity_source_payload,
        commodity_ticket_source_payload,
        commodity_ticket_book_source_payload,
        commodity_execution_preview_source_payload,
        commodity_execution_artifact_source_payload,
        commodity_execution_queue_source_payload,
        commodity_execution_review_source_payload,
        commodity_execution_retro_source_payload,
        commodity_execution_gap_source_payload,
        commodity_execution_bridge_source_payload,
    )

    source_mode = "single-hot-universe-source"
    unique_sources = {str(action_source_path), str(crypto_source_path)}
    if commodity_source_path:
        unique_sources.add(str(commodity_source_path))
    if commodity_ticket_source_path:
        unique_sources.add(str(commodity_ticket_source_path))
    if commodity_ticket_book_source_path:
        unique_sources.add(str(commodity_ticket_book_source_path))
    if commodity_execution_preview_source_path:
        unique_sources.add(str(commodity_execution_preview_source_path))
    if commodity_execution_artifact_source_path:
        unique_sources.add(str(commodity_execution_artifact_source_path))
    if commodity_execution_queue_source_path:
        unique_sources.add(str(commodity_execution_queue_source_path))
    if commodity_execution_review_source_path:
        unique_sources.add(str(commodity_execution_review_source_path))
    if commodity_execution_retro_source_path:
        unique_sources.add(str(commodity_execution_retro_source_path))
    if commodity_execution_gap_source_path:
        unique_sources.add(str(commodity_execution_gap_source_path))
    if commodity_execution_bridge_source_path:
        unique_sources.add(str(commodity_execution_bridge_source_path))
    if len(unique_sources) > 1:
        if commodity_source_path or commodity_ticket_source_path or commodity_ticket_book_source_path:
            source_mode = "merged-action-commodity-crypto-sources"
        else:
            source_mode = "merged-action-crypto-sources"

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_hot_universe_operator_brief.json"
    md_path = review_dir / f"{stamp}_hot_universe_operator_brief.md"
    checksum_path = review_dir / f"{stamp}_hot_universe_operator_brief_checksum.json"

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_mode": source_mode,
        "source_artifact": str(action_source_path if action_source_path == crypto_source_path else crypto_source_path),
        "source_status": str(crypto_source_payload.get("status") or ""),
        "source_action_artifact": str(action_source_path),
        "source_action_status": str(action_source_payload.get("status") or ""),
        "source_commodity_artifact": str(commodity_source_path) if commodity_source_path else "",
        "source_commodity_status": str((commodity_source_payload or {}).get("status") or ""),
        "source_commodity_ticket_artifact": str(commodity_ticket_source_path) if commodity_ticket_source_path else "",
        "source_commodity_ticket_status": str((commodity_ticket_source_payload or {}).get("status") or ""),
        "source_commodity_ticket_book_artifact": str(commodity_ticket_book_source_path) if commodity_ticket_book_source_path else "",
        "source_commodity_ticket_book_status": str((commodity_ticket_book_source_payload or {}).get("status") or ""),
        "source_commodity_execution_preview_artifact": str(commodity_execution_preview_source_path) if commodity_execution_preview_source_path else "",
        "source_commodity_execution_preview_status": str((commodity_execution_preview_source_payload or {}).get("status") or ""),
        "source_commodity_execution_artifact": str(commodity_execution_artifact_source_path) if commodity_execution_artifact_source_path else "",
        "source_commodity_execution_artifact_status": str((commodity_execution_artifact_source_payload or {}).get("status") or ""),
        "source_commodity_execution_queue_artifact": str(commodity_execution_queue_source_path) if commodity_execution_queue_source_path else "",
        "source_commodity_execution_queue_status": str((commodity_execution_queue_source_payload or {}).get("status") or ""),
        "source_commodity_execution_review_artifact": str(commodity_execution_review_source_path) if commodity_execution_review_source_path else "",
        "source_commodity_execution_review_status": str((commodity_execution_review_source_payload or {}).get("status") or ""),
        "source_commodity_execution_retro_artifact": str(commodity_execution_retro_source_path) if commodity_execution_retro_source_path else "",
        "source_commodity_execution_retro_status": str((commodity_execution_retro_source_payload or {}).get("status") or ""),
        "source_commodity_execution_gap_artifact": str(commodity_execution_gap_source_path) if commodity_execution_gap_source_path else "",
        "source_commodity_execution_gap_status": str((commodity_execution_gap_source_payload or {}).get("status") or ""),
        "source_commodity_execution_bridge_artifact": str(commodity_execution_bridge_source_path) if commodity_execution_bridge_source_path else "",
        "source_commodity_execution_bridge_status": str((commodity_execution_bridge_source_payload or {}).get("status") or ""),
        "source_crypto_artifact": str(crypto_source_path),
        "source_crypto_status": str(crypto_source_payload.get("status") or ""),
        "source_crypto_route_artifact": str(crypto_route_source_path) if crypto_route_source_path else "",
        "source_crypto_route_status": str((crypto_route_source_payload or {}).get("status") or ""),
        "operator_research_embedding_quality_status": str(research_embedding_quality.get("status") or ""),
        "operator_research_embedding_quality_brief": str(research_embedding_quality.get("brief") or ""),
        "operator_research_embedding_quality_blocker_detail": str(
            research_embedding_quality.get("blocker_detail") or ""
        ),
        "operator_research_embedding_quality_done_when": str(research_embedding_quality.get("done_when") or ""),
        "operator_research_embedding_active_batches": list(research_embedding_quality.get("active_batches") or []),
        "operator_research_embedding_avoid_batches": list(research_embedding_quality.get("avoid_batches") or []),
        "operator_research_embedding_zero_trade_deprioritized_batches": list(
            research_embedding_quality.get("zero_trade_deprioritized_batches") or []
        ),
        **brief,
    }
    crypto_route_alignment_focus = _crypto_route_alignment_focus(payload)
    crypto_route_alignment = _crypto_route_embedding_alignment(
        secondary_focus_area=str(crypto_route_alignment_focus.get("area") or ""),
        secondary_focus_symbol=str(crypto_route_alignment_focus.get("symbol") or ""),
        secondary_focus_action=str(crypto_route_alignment_focus.get("action") or ""),
        quality_brief=str(payload.get("operator_research_embedding_quality_brief") or ""),
        active_batches=list(payload.get("operator_research_embedding_active_batches") or []),
    )
    payload["operator_crypto_route_alignment_focus_slot"] = str(
        crypto_route_alignment_focus.get("slot") or "-"
    )
    payload["operator_crypto_route_alignment_status"] = str(crypto_route_alignment.get("status") or "")
    payload["operator_crypto_route_alignment_brief"] = str(crypto_route_alignment.get("brief") or "")
    payload["operator_crypto_route_alignment_blocker_detail"] = str(
        crypto_route_alignment.get("blocker_detail") or ""
    )
    payload["operator_crypto_route_alignment_done_when"] = str(crypto_route_alignment.get("done_when") or "")
    crypto_route_alignment_recipe = _crypto_route_alignment_recovery_recipe(
        alignment_status=str(payload.get("operator_crypto_route_alignment_status") or ""),
        source_artifact=str(payload.get("source_crypto_artifact") or ""),
        avoid_batches=list(payload.get("operator_research_embedding_avoid_batches") or []),
        zero_trade_deprioritized_batches=list(
            payload.get("operator_research_embedding_zero_trade_deprioritized_batches") or []
        ),
        reference_now=runtime_now,
    )
    payload["operator_crypto_route_alignment_recipe_script"] = str(
        crypto_route_alignment_recipe.get("script") or ""
    )
    payload["operator_crypto_route_alignment_recipe_command_hint"] = str(
        crypto_route_alignment_recipe.get("command_hint") or ""
    )
    payload["operator_crypto_route_alignment_recipe_expected_status"] = str(
        crypto_route_alignment_recipe.get("expected_status") or ""
    )
    payload["operator_crypto_route_alignment_recipe_note"] = str(
        crypto_route_alignment_recipe.get("note") or ""
    )
    payload["operator_crypto_route_alignment_recipe_followup_script"] = str(
        crypto_route_alignment_recipe.get("followup_script") or ""
    )
    payload["operator_crypto_route_alignment_recipe_followup_command_hint"] = str(
        crypto_route_alignment_recipe.get("followup_command_hint") or ""
    )
    payload["operator_crypto_route_alignment_recipe_verify_hint"] = str(
        crypto_route_alignment_recipe.get("verify_hint") or ""
    )
    payload["operator_crypto_route_alignment_recipe_window_days"] = crypto_route_alignment_recipe.get("window_days")
    payload["operator_crypto_route_alignment_recipe_target_batches"] = list(
        crypto_route_alignment_recipe.get("target_batches") or []
    )
    crypto_route_alignment_recovery_outcome = _crypto_route_alignment_recovery_outcome(
        alignment_status=str(payload.get("operator_crypto_route_alignment_status") or ""),
        quality_status=str(payload.get("operator_research_embedding_quality_status") or ""),
        source_status=str(payload.get("source_crypto_status") or ""),
        source_payload=crypto_source_payload if isinstance(crypto_source_payload, dict) else {},
    )
    payload["operator_crypto_route_alignment_recovery_status"] = str(
        crypto_route_alignment_recovery_outcome.get("status") or ""
    )
    payload["operator_crypto_route_alignment_recovery_brief"] = str(
        crypto_route_alignment_recovery_outcome.get("brief") or ""
    )
    payload["operator_crypto_route_alignment_recovery_blocker_detail"] = str(
        crypto_route_alignment_recovery_outcome.get("blocker_detail") or ""
    )
    payload["operator_crypto_route_alignment_recovery_done_when"] = str(
        crypto_route_alignment_recovery_outcome.get("done_when") or ""
    )
    payload["operator_crypto_route_alignment_recovery_failed_batch_count"] = (
        crypto_route_alignment_recovery_outcome.get("failed_batch_count")
    )
    payload["operator_crypto_route_alignment_recovery_timed_out_batch_count"] = (
        crypto_route_alignment_recovery_outcome.get("timed_out_batch_count")
    )
    payload["operator_crypto_route_alignment_recovery_zero_trade_batches"] = list(
        crypto_route_alignment_recovery_outcome.get("zero_trade_batches") or []
    )
    crypto_route_alignment_cooldown = _crypto_route_alignment_cooldown(
        alignment_status=str(payload.get("operator_crypto_route_alignment_status") or ""),
        recovery_status=str(payload.get("operator_crypto_route_alignment_recovery_status") or ""),
        source_status=str(payload.get("source_crypto_status") or ""),
        source_payload=crypto_source_payload if isinstance(crypto_source_payload, dict) else {},
        reference_now=runtime_now,
    )
    payload["operator_crypto_route_alignment_cooldown_status"] = str(
        crypto_route_alignment_cooldown.get("status") or ""
    )
    payload["operator_crypto_route_alignment_cooldown_brief"] = str(
        crypto_route_alignment_cooldown.get("brief") or ""
    )
    payload["operator_crypto_route_alignment_cooldown_blocker_detail"] = str(
        crypto_route_alignment_cooldown.get("blocker_detail") or ""
    )
    payload["operator_crypto_route_alignment_cooldown_done_when"] = str(
        crypto_route_alignment_cooldown.get("done_when") or ""
    )
    payload["operator_crypto_route_alignment_cooldown_last_research_end_date"] = str(
        crypto_route_alignment_cooldown.get("last_research_end_date") or ""
    )
    payload["operator_crypto_route_alignment_cooldown_next_eligible_end_date"] = str(
        crypto_route_alignment_cooldown.get("next_eligible_end_date") or ""
    )
    crypto_route_alignment_recipe_gate = _crypto_route_alignment_recipe_gate(
        alignment_status=str(payload.get("operator_crypto_route_alignment_status") or ""),
        recipe_script=str(payload.get("operator_crypto_route_alignment_recipe_script") or ""),
        cooldown_status=str(payload.get("operator_crypto_route_alignment_cooldown_status") or ""),
        cooldown_brief=str(payload.get("operator_crypto_route_alignment_cooldown_brief") or ""),
        cooldown_blocker_detail=str(payload.get("operator_crypto_route_alignment_cooldown_blocker_detail") or ""),
        cooldown_done_when=str(payload.get("operator_crypto_route_alignment_cooldown_done_when") or ""),
        cooldown_next_eligible_end_date=str(
            payload.get("operator_crypto_route_alignment_cooldown_next_eligible_end_date") or ""
        ),
    )
    payload["operator_crypto_route_alignment_recipe_status"] = str(
        crypto_route_alignment_recipe_gate.get("status") or ""
    )
    payload["operator_crypto_route_alignment_recipe_brief"] = str(
        crypto_route_alignment_recipe_gate.get("brief") or ""
    )
    payload["operator_crypto_route_alignment_recipe_blocker_detail"] = str(
        crypto_route_alignment_recipe_gate.get("blocker_detail") or ""
    )
    payload["operator_crypto_route_alignment_recipe_done_when"] = str(
        crypto_route_alignment_recipe_gate.get("done_when") or ""
    )
    payload["operator_crypto_route_alignment_recipe_ready_on_date"] = str(
        crypto_route_alignment_recipe_gate.get("ready_on_date") or ""
    )
    focus_slots = []
    for row in payload.get("operator_focus_slots", []):
        if not isinstance(row, dict):
            continue
        slot = dict(row)
        source_kind, source_artifact = _focus_slot_source(
            payload,
            area=str(slot.get("area") or ""),
            action=str(slot.get("action") or ""),
        )
        slot["source_kind"] = source_kind
        slot["source_artifact"] = source_artifact
        focus_slots.append(slot)
    if focus_slots:
        payload["operator_focus_slots"] = focus_slots
        payload["operator_focus_slots_brief"] = _focus_slots_brief(focus_slots)
        payload["operator_focus_slot_sources_brief"] = _focus_slot_source_brief(focus_slots)
        enriched_slots: list[dict[str, Any]] = []
        for slot in focus_slots:
            enriched = dict(slot)
            enriched["source_status"] = _focus_slot_source_status(
                payload,
                source_kind=str(enriched.get("source_kind") or ""),
            )
            enriched["source_as_of"] = _focus_slot_source_as_of(str(enriched.get("source_artifact") or ""))
            enriched["source_age_minutes"] = _focus_slot_source_age_minutes(
                reference_now=runtime_now,
                source_as_of=str(enriched.get("source_as_of") or ""),
            )
            enriched["source_recency"] = _focus_slot_source_recency(
                source_age_minutes=enriched.get("source_age_minutes")
            )
            enriched["source_health"] = _focus_slot_source_health(
                source_status=str(enriched.get("source_status") or ""),
                source_recency=str(enriched.get("source_recency") or ""),
                source_kind=str(enriched.get("source_kind") or ""),
                crypto_route_alignment_cooldown_status=str(
                    payload.get("operator_crypto_route_alignment_cooldown_status") or ""
                ),
            )
            enriched["source_refresh_action"] = _focus_slot_source_refresh_action(
                source_health=str(enriched.get("source_health") or ""),
                source_kind=str(enriched.get("source_kind") or ""),
                crypto_route_alignment_cooldown_status=str(
                    payload.get("operator_crypto_route_alignment_cooldown_status") or ""
                ),
            )
            enriched_slots.append(enriched)
        focus_slots = enriched_slots
        payload["operator_focus_slots"] = focus_slots
        payload["operator_focus_slots_brief"] = _focus_slots_brief(focus_slots)
        payload["operator_focus_slot_sources_brief"] = _focus_slot_source_brief(focus_slots)
        payload["operator_focus_slot_status_brief"] = _focus_slot_status_brief(focus_slots)
        payload["operator_focus_slot_recency_brief"] = _focus_slot_recency_brief(focus_slots)
        payload["operator_focus_slot_health_brief"] = _focus_slot_health_brief(focus_slots)
        refresh_backlog = _focus_slot_refresh_backlog(focus_slots)
        source_refresh_queue = _source_refresh_queue(refresh_backlog)
        source_refresh_checklist = _source_refresh_checklist(source_refresh_queue, reference_now=runtime_now)
        payload["operator_focus_slot_refresh_backlog"] = refresh_backlog
        payload["operator_focus_slot_refresh_backlog_brief"] = _focus_slot_refresh_backlog_brief(refresh_backlog)
        payload["operator_focus_slot_refresh_backlog_count"] = len(refresh_backlog)
        payload["operator_source_refresh_queue"] = source_refresh_queue
        payload["operator_source_refresh_queue_brief"] = _source_refresh_queue_brief(source_refresh_queue)
        payload["operator_source_refresh_queue_count"] = len(source_refresh_queue)
        payload["operator_source_refresh_checklist"] = source_refresh_checklist
        payload["operator_source_refresh_checklist_brief"] = _source_refresh_checklist_brief(
            source_refresh_checklist
        )
        ready_count = sum(
            1
            for row in focus_slots
            if str(row.get("source_refresh_action") or "") in {"read_current_artifact", "wait_for_next_eligible_end_date"}
        )
        total_count = len(focus_slots)
        promotion_gate_status = _focus_slot_promotion_gate_status(
            total_count=total_count,
            ready_count=ready_count,
        )
        payload["operator_focus_slot_ready_count"] = ready_count
        payload["operator_focus_slot_total_count"] = total_count
        payload["operator_focus_slot_promotion_gate_brief"] = _focus_slot_promotion_gate_brief(
            status=promotion_gate_status,
            ready_count=ready_count,
            total_count=total_count,
        )
        payload["operator_focus_slot_promotion_gate_status"] = promotion_gate_status
        payload["operator_focus_slot_promotion_gate_blocker_detail"] = _focus_slot_promotion_gate_blocker_detail(
            total_count=total_count,
            ready_count=ready_count,
            refresh_backlog=refresh_backlog,
        )
        payload["operator_focus_slot_promotion_gate_done_when"] = _focus_slot_promotion_gate_done_when(
            total_count=total_count,
            ready_count=ready_count,
        )
        actionability_backlog = _focus_slot_actionability_backlog(
            focus_slots,
            alignment_focus_slot=str(payload.get("operator_crypto_route_alignment_focus_slot") or ""),
            alignment_status=str(payload.get("operator_crypto_route_alignment_status") or ""),
            alignment_brief=str(payload.get("operator_crypto_route_alignment_brief") or ""),
            alignment_recovery_status=str(payload.get("operator_crypto_route_alignment_recovery_status") or ""),
            alignment_recovery_brief=str(payload.get("operator_crypto_route_alignment_recovery_brief") or ""),
        )
        actionable_count = max(total_count - len(actionability_backlog), 0)
        actionability_gate_status = _focus_slot_actionability_gate_status(
            total_count=total_count,
            actionable_count=actionable_count,
        )
        payload["operator_focus_slot_actionability_backlog"] = actionability_backlog
        payload["operator_focus_slot_actionability_backlog_brief"] = _focus_slot_actionability_backlog_brief(
            actionability_backlog
        )
        payload["operator_focus_slot_actionability_backlog_count"] = len(actionability_backlog)
        payload["operator_focus_slot_actionable_count"] = actionable_count
        payload["operator_focus_slot_actionability_gate_brief"] = _focus_slot_actionability_gate_brief(
            status=actionability_gate_status,
            actionable_count=actionable_count,
            total_count=total_count,
        )
        payload["operator_focus_slot_actionability_gate_status"] = actionability_gate_status
        payload["operator_focus_slot_actionability_gate_blocker_detail"] = (
            _focus_slot_actionability_gate_blocker_detail(
                total_count=total_count,
                actionable_count=actionable_count,
                backlog=actionability_backlog,
            )
        )
        payload["operator_focus_slot_actionability_gate_done_when"] = (
            _focus_slot_actionability_gate_done_when(
                total_count=total_count,
                actionable_count=actionable_count,
                backlog=actionability_backlog,
            )
        )
        readiness_gate_status = _focus_slot_readiness_gate_status(
            total_count=total_count,
            promotion_gate_status=promotion_gate_status,
            actionability_gate_status=actionability_gate_status,
        )
        readiness_ready_count = _focus_slot_readiness_gate_ready_count(
            status=readiness_gate_status,
            ready_count=ready_count,
            actionable_count=actionable_count,
            total_count=total_count,
        )
        payload["operator_focus_slot_readiness_gate_ready_count"] = readiness_ready_count
        payload["operator_focus_slot_readiness_gate_brief"] = _focus_slot_readiness_gate_brief(
            status=readiness_gate_status,
            ready_count=readiness_ready_count,
            total_count=total_count,
        )
        payload["operator_focus_slot_readiness_gate_status"] = readiness_gate_status
        payload["operator_focus_slot_readiness_gate_blocking_gate"] = _focus_slot_readiness_gate_blocking_gate(
            status=readiness_gate_status
        )
        payload["operator_focus_slot_readiness_gate_blocker_detail"] = (
            _focus_slot_readiness_gate_blocker_detail(
                status=readiness_gate_status,
                promotion_gate_blocker_detail=str(
                    payload.get("operator_focus_slot_promotion_gate_blocker_detail") or ""
                ),
                actionability_gate_blocker_detail=str(
                    payload.get("operator_focus_slot_actionability_gate_blocker_detail") or ""
                ),
                total_count=total_count,
            )
        )
        payload["operator_focus_slot_readiness_gate_done_when"] = _focus_slot_readiness_gate_done_when(
            status=readiness_gate_status,
            promotion_gate_done_when=str(payload.get("operator_focus_slot_promotion_gate_done_when") or ""),
            actionability_gate_done_when=str(payload.get("operator_focus_slot_actionability_gate_done_when") or ""),
        )
        primary_slot = dict(focus_slots[0]) if len(focus_slots) >= 1 else {}
        followup_slot = dict(focus_slots[1]) if len(focus_slots) >= 2 else {}
        secondary_slot = dict(focus_slots[2]) if len(focus_slots) >= 3 else {}
        refresh_head = dict(refresh_backlog[0]) if refresh_backlog else {}
        source_refresh_head = dict(source_refresh_queue[0]) if source_refresh_queue else {}
        source_refresh_checklist_head = dict(source_refresh_checklist[0]) if source_refresh_checklist else {}
        pipeline_source_kind = str(
            secondary_slot.get("source_kind") or source_refresh_head.get("source_kind") or ""
        )
        pipeline_source_artifact = str(
            secondary_slot.get("source_artifact") or source_refresh_head.get("source_artifact") or ""
        )
        pipeline_recipe = _source_refresh_recipe(
            source_kind=pipeline_source_kind,
            source_artifact=pipeline_source_artifact,
            reference_now=runtime_now,
        )
        pipeline_steps = list(pipeline_recipe.get("steps") or [])
        pipeline_deferred_steps = _recipe_deferred_steps(
            pipeline_steps,
            recipe_gate_status=str(payload.get("operator_crypto_route_alignment_recipe_status") or ""),
        )
        pipeline_pending_steps = [] if pipeline_deferred_steps else _recipe_pending_steps(pipeline_steps)
        pipeline_head = dict(pipeline_pending_steps[0]) if pipeline_pending_steps else {}
        pipeline_deferred_head = dict(pipeline_deferred_steps[0]) if pipeline_deferred_steps else {}
        payload["operator_source_refresh_pipeline_steps_brief"] = str(pipeline_recipe.get("steps_brief") or "-")
        payload["operator_source_refresh_pipeline_step_checkpoint_brief"] = str(
            pipeline_recipe.get("step_checkpoint_brief") or "-"
        )
        payload["operator_source_refresh_pipeline_pending_brief"] = _recipe_pending_brief(pipeline_pending_steps)
        payload["operator_source_refresh_pipeline_pending_count"] = len(pipeline_pending_steps)
        payload["operator_source_refresh_pipeline_head_rank"] = str(pipeline_head.get("rank") or "-")
        payload["operator_source_refresh_pipeline_head_name"] = str(pipeline_head.get("name") or "-")
        payload["operator_source_refresh_pipeline_head_checkpoint_state"] = str(
            pipeline_head.get("checkpoint_state") or "-"
        )
        payload["operator_source_refresh_pipeline_head_expected_artifact_kind"] = str(
            pipeline_head.get("expected_artifact_kind") or "-"
        )
        payload["operator_source_refresh_pipeline_head_current_artifact"] = str(
            pipeline_head.get("current_artifact") or "-"
        )
        payload["operator_source_refresh_pipeline_deferred_brief"] = _recipe_pending_brief(pipeline_deferred_steps)
        payload["operator_source_refresh_pipeline_deferred_count"] = len(pipeline_deferred_steps)
        payload["operator_source_refresh_pipeline_deferred_status"] = str(
            payload.get("operator_crypto_route_alignment_recipe_status") or ""
        )
        payload["operator_source_refresh_pipeline_deferred_until"] = str(
            payload.get("operator_crypto_route_alignment_recipe_ready_on_date")
            or payload.get("operator_crypto_route_alignment_cooldown_next_eligible_end_date")
            or ""
        )
        payload["operator_source_refresh_pipeline_deferred_reason"] = str(
            payload.get("operator_crypto_route_alignment_recipe_blocker_detail")
            or payload.get("operator_crypto_route_alignment_cooldown_blocker_detail")
            or ""
        )
        payload["operator_source_refresh_pipeline_deferred_head_rank"] = str(
            pipeline_deferred_head.get("rank") or "-"
        )
        payload["operator_source_refresh_pipeline_deferred_head_name"] = str(
            pipeline_deferred_head.get("name") or "-"
        )
        payload["operator_source_refresh_pipeline_deferred_head_checkpoint_state"] = str(
            pipeline_deferred_head.get("checkpoint_state") or "-"
        )
        payload["operator_source_refresh_pipeline_deferred_head_expected_artifact_kind"] = str(
            pipeline_deferred_head.get("expected_artifact_kind") or "-"
        )
        payload["operator_source_refresh_pipeline_deferred_head_current_artifact"] = str(
            pipeline_deferred_head.get("current_artifact") or "-"
        )
        payload["next_focus_source_kind"] = str(primary_slot.get("source_kind") or "-")
        payload["next_focus_source_artifact"] = str(primary_slot.get("source_artifact") or "")
        payload["next_focus_source_status"] = str(primary_slot.get("source_status") or "-")
        payload["next_focus_source_as_of"] = str(primary_slot.get("source_as_of") or "")
        payload["next_focus_source_age_minutes"] = primary_slot.get("source_age_minutes")
        payload["next_focus_source_recency"] = str(primary_slot.get("source_recency") or "-")
        payload["next_focus_source_health"] = str(primary_slot.get("source_health") or "-")
        payload["next_focus_source_refresh_action"] = str(primary_slot.get("source_refresh_action") or "-")
        payload["followup_focus_source_kind"] = str(followup_slot.get("source_kind") or "-")
        payload["followup_focus_source_artifact"] = str(followup_slot.get("source_artifact") or "")
        payload["followup_focus_source_status"] = str(followup_slot.get("source_status") or "-")
        payload["followup_focus_source_as_of"] = str(followup_slot.get("source_as_of") or "")
        payload["followup_focus_source_age_minutes"] = followup_slot.get("source_age_minutes")
        payload["followup_focus_source_recency"] = str(followup_slot.get("source_recency") or "-")
        payload["followup_focus_source_health"] = str(followup_slot.get("source_health") or "-")
        payload["followup_focus_source_refresh_action"] = str(followup_slot.get("source_refresh_action") or "-")
        payload["secondary_focus_source_kind"] = str(secondary_slot.get("source_kind") or "-")
        payload["secondary_focus_source_artifact"] = str(secondary_slot.get("source_artifact") or "")
        payload["secondary_focus_source_status"] = str(secondary_slot.get("source_status") or "-")
        payload["secondary_focus_source_as_of"] = str(secondary_slot.get("source_as_of") or "")
        payload["secondary_focus_source_age_minutes"] = secondary_slot.get("source_age_minutes")
        payload["secondary_focus_source_recency"] = str(secondary_slot.get("source_recency") or "-")
        payload["secondary_focus_source_health"] = str(secondary_slot.get("source_health") or "-")
        payload["secondary_focus_source_refresh_action"] = str(secondary_slot.get("source_refresh_action") or "-")
        payload["operator_focus_slot_refresh_head_slot"] = str(refresh_head.get("slot") or "-")
        payload["operator_focus_slot_refresh_head_symbol"] = str(refresh_head.get("symbol") or "-")
        payload["operator_focus_slot_refresh_head_action"] = str(refresh_head.get("action") or "-")
        payload["operator_focus_slot_refresh_head_health"] = str(refresh_head.get("source_health") or "-")
        payload["operator_source_refresh_next_slot"] = str(source_refresh_head.get("slot") or "-")
        payload["operator_source_refresh_next_symbol"] = str(source_refresh_head.get("symbol") or "-")
        payload["operator_source_refresh_next_action"] = str(source_refresh_head.get("action") or "-")
        payload["operator_source_refresh_next_source_kind"] = str(source_refresh_head.get("source_kind") or "-")
        payload["operator_source_refresh_next_source_health"] = str(source_refresh_head.get("source_health") or "-")
        payload["operator_source_refresh_next_source_artifact"] = str(source_refresh_head.get("source_artifact") or "")
        payload["operator_source_refresh_next_state"] = str(source_refresh_checklist_head.get("state") or "-")
        payload["operator_source_refresh_next_blocker_detail"] = str(
            source_refresh_checklist_head.get("blocker_detail") or "-"
        )
        payload["operator_source_refresh_next_done_when"] = str(
            source_refresh_checklist_head.get("done_when") or "-"
        )
        payload["operator_source_refresh_next_recipe_script"] = str(
            source_refresh_checklist_head.get("recipe_script") or ""
        )
        payload["operator_source_refresh_next_recipe_command_hint"] = str(
            source_refresh_checklist_head.get("recipe_command_hint") or ""
        )
        payload["operator_source_refresh_next_recipe_expected_status"] = str(
            source_refresh_checklist_head.get("recipe_expected_status") or ""
        )
        payload["operator_source_refresh_next_recipe_expected_artifact_kind"] = str(
            source_refresh_checklist_head.get("recipe_expected_artifact_kind") or ""
        )
        payload["operator_source_refresh_next_recipe_expected_artifact_path_hint"] = str(
            source_refresh_checklist_head.get("recipe_expected_artifact_path_hint") or ""
        )
        payload["operator_source_refresh_next_recipe_note"] = str(
            source_refresh_checklist_head.get("recipe_note") or ""
        )
        payload["operator_source_refresh_next_recipe_followup_script"] = str(
            source_refresh_checklist_head.get("recipe_followup_script") or ""
        )
        payload["operator_source_refresh_next_recipe_followup_command_hint"] = str(
            source_refresh_checklist_head.get("recipe_followup_command_hint") or ""
        )
        payload["operator_source_refresh_next_recipe_verify_hint"] = str(
            source_refresh_checklist_head.get("recipe_verify_hint") or ""
        )
        payload["operator_source_refresh_next_recipe_steps_brief"] = str(
            source_refresh_checklist_head.get("recipe_steps_brief") or ""
        )
        payload["operator_source_refresh_next_recipe_step_checkpoint_brief"] = str(
            source_refresh_checklist_head.get("recipe_step_checkpoint_brief") or ""
        )
        payload["operator_source_refresh_next_recipe_steps"] = list(
            source_refresh_checklist_head.get("recipe_steps") or []
        )
        summary_lines = list(payload.get("summary_lines", []))
        summary_lines.append(f"focus-slot-sources: {payload.get('operator_focus_slot_sources_brief') or '-'}")
        summary_lines.append(f"focus-slot-source-status: {payload.get('operator_focus_slot_status_brief') or '-'}")
        summary_lines.append(f"focus-slot-source-recency: {payload.get('operator_focus_slot_recency_brief') or '-'}")
        summary_lines.append(f"focus-slot-source-health: {payload.get('operator_focus_slot_health_brief') or '-'}")
        summary_lines.append(f"focus-slot-refresh-backlog: {payload.get('operator_focus_slot_refresh_backlog_brief') or '-'}")
        summary_lines.append(f"focus-slot-promotion-gate: {payload.get('operator_focus_slot_promotion_gate_brief') or '-'}")
        summary_lines.append(
            f"focus-slot-actionability-gate: {payload.get('operator_focus_slot_actionability_gate_brief') or '-'}"
        )
        summary_lines.append(
            f"focus-slot-readiness-gate: {payload.get('operator_focus_slot_readiness_gate_brief') or '-'}"
        )
        summary_lines.append(
            f"research-embedding-quality: {payload.get('operator_research_embedding_quality_brief') or '-'}"
        )
        summary_lines.append(
            f"crypto-route-alignment: {payload.get('operator_crypto_route_alignment_brief') or '-'}"
        )
        summary_lines.append(
            f"crypto-route-alignment-slot: {payload.get('operator_crypto_route_alignment_focus_slot') or '-'}"
        )
        summary_lines.append(
            f"crypto-route-alignment-recovery-outcome: {payload.get('operator_crypto_route_alignment_recovery_brief') or '-'}"
        )
        summary_lines.append(
            f"crypto-route-alignment-cooldown: {payload.get('operator_crypto_route_alignment_cooldown_brief') or '-'}"
        )
        summary_lines.append(
            f"crypto-route-alignment-recovery-recipe: {payload.get('operator_crypto_route_alignment_recipe_brief') or '-'}"
        )
        if payload.get("operator_crypto_route_alignment_recipe_target_batches"):
            summary_lines.append(
                "crypto-route-alignment-recovery: "
                + _list_text(payload.get("operator_crypto_route_alignment_recipe_target_batches", []), limit=6)
                + f"@{payload.get('operator_crypto_route_alignment_recipe_window_days') or '-'}d"
            )
        summary_lines.append(f"source-refresh-queue: {payload.get('operator_source_refresh_queue_brief') or '-'}")
        summary_lines.append(f"source-refresh-checklist: {payload.get('operator_source_refresh_checklist_brief') or '-'}")
        summary_lines.append(
            f"source-refresh-pipeline: {payload.get('operator_source_refresh_pipeline_pending_brief') or '-'}"
        )
        summary_lines.append(
            f"source-refresh-pipeline-deferred: {payload.get('operator_source_refresh_pipeline_deferred_brief') or '-'}"
        )
        payload["summary_lines"] = summary_lines
        payload["summary_text"] = " | ".join(summary_lines)
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
        stem="hot_universe_operator_brief",
        current_paths=[json_path, md_path, checksum_path],
        keep=max(3, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=runtime_now,
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
