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


MODE_PRIORITY = {
    "architecture_review": 6,
    "live_guard_diagnostics": 5,
    "research_backtest": 4,
    "ui_rendering": 3,
    "source_first_implementation": 2,
    "default_implementation": 1,
}


FAILURE_SEVERITY_MATRIX: list[dict[str, str]] = [
    {
        "severity": "P0",
        "scope": "execution_path_or_position_state_ambiguity",
        "required_action": "trading_halt_or_panic_close",
        "notes": "Only execution-path failure or source-of-truth ambiguity may escalate to panic-close behavior.",
    },
    {
        "severity": "P1",
        "scope": "process_failure_or_critical_instance_fault",
        "required_action": "instance_recycle_or_process_kill",
        "notes": "Process-level failure may require recycle/kill, but not automatic trading halt unless execution truth is ambiguous.",
    },
    {
        "severity": "P2",
        "scope": "notification_metrics_ui_non_trading_failures",
        "required_action": "degrade_and_report_only",
        "notes": "Telegram 409, metrics sink issues, and UI failures must not directly trigger trading panic actions.",
    },
]


SOURCE_OF_TRUTH_REGISTRY: list[dict[str, str]] = [
    {
        "domain": "orders_and_positions",
        "rule": "Treat venue/account-backed order and position artifacts as source of truth; consumers may read but must not re-derive ownership.",
    },
    {
        "domain": "queues_and_lanes",
        "rule": "Queues, lanes, gates, and scores should be expressed in source artifacts first and propagated downstream with minimal consumer derivation.",
    },
    {
        "domain": "live_blockers",
        "rule": "ops_live_gate, risk_guard, readiness, and clearing conditions must come from blocker source artifacts, not UI reconstruction.",
    },
    {
        "domain": "research_and_backtests",
        "rule": "Strategy validity, OOS status, and comparison metrics must come from deterministic research artifacts; top briefs may summarize but not invent them.",
    },
]


CHANGE_CLASS_LIBRARY: list[dict[str, str]] = [
    {
        "name": "DOC_ONLY",
        "scope": "Docs, prompts, markdown, and presentation-only text changes.",
    },
    {
        "name": "RESEARCH_ONLY",
        "scope": "Reports, artifacts, orchestration summaries, and non-execution research wiring.",
    },
    {
        "name": "SIM_ONLY",
        "scope": "Backtests, replay, signal simulation, and non-live execution modeling.",
    },
    {
        "name": "LIVE_GUARD_ONLY",
        "scope": "Read-only probes, blockers, account scope, guards, and diagnostics around live execution.",
    },
    {
        "name": "LIVE_EXECUTION_PATH",
        "scope": "Order adapters, execution routing, portfolio mutation, secrets, deployment, or anything that can place/alter live orders.",
    },
]


VALIDATION_REPORT_REQUIREMENTS: list[str] = [
    "changed_files",
    "live_path_touched",
    "recommended_change_class",
    "validation_scope",
    "validation_commands",
    "validation_result",
    "residual_risk",
    "rollback_point",
]


ROLLOUT_LADDER: list[str] = [
    "shadow",
    "replay",
    "canary",
    "broader_rollout",
]


MODE_LIBRARY: dict[str, dict[str, Any]] = {
    "architecture_review": {
        "brief": "Prioritize source ownership, cycle detection, and state mismatch review.",
        "hard_guardrails": [
            "Do not widen consumer-side mirror fields before checking whether the source artifact can express the state directly.",
            "Prefer findings first; keep summaries secondary.",
            "If no concrete finding exists, stop instead of continuing abstract cleanup.",
        ],
        "execution_priorities": [
            "Identify source-consumer cycles or stale ownership chains.",
            "Find duplicated state derivation and lane/count drift.",
            "Fix the highest-risk architectural mismatch first.",
        ],
        "execution_path": [
            "Map the current source artifacts and their consumers.",
            "Check for cycles, duplicate derivation, and stale mirrors.",
            "Either report the findings or fix the highest-impact ownership issue.",
        ],
        "stop_rules": [
            "Stop after the top concrete architectural issues are explicit.",
            "Stop if the next step is only helper splitting or naming cleanup.",
        ],
        "success_criteria": [
            "The highest-risk architecture issue is concrete and file-specific.",
            "If a fix is applied, source ownership is simpler after the change.",
        ],
        "validation_plan": [
            "Run targeted tests around the affected producer and consumer pair.",
            "Re-check real artifacts for lane/count/state agreement.",
        ],
        "governance_overlays": [
            "Audit source-of-truth ownership explicitly before changing consumer summaries.",
            "Prefer documenting severity class and residual risk over broad refactor cleanup.",
        ],
        "forbidden_moves": [
            "Do not solve a cycle by adding a third mirror layer.",
            "Do not continue into UI cleanup before source ownership is stable.",
        ],
        "prompt_focus": "Operate in architecture-review mode. Findings and source ownership come before refactor breadth.",
    },
    "source_first_implementation": {
        "brief": "Add new behavior at the source artifact first, then propagate minimally.",
        "hard_guardrails": [
            "Do not add a consumer-side field if the source artifact can already represent it.",
            "Keep backward compatibility only where existing consumers need it.",
            "Stop once the source and one consumer are aligned.",
        ],
        "execution_priorities": [
            "Implement the new state or lane in the producer.",
            "Propagate it to one downstream consumer.",
            "Minimize mirror-field growth.",
        ],
        "execution_path": [
            "Locate the producer of truth.",
            "Implement and validate there first.",
            "Update one consumer and verify the handoff.",
        ],
        "stop_rules": [
            "Stop after the new capability exists in source and one consumer.",
            "Stop if the next step only adds summaries of the same source fact.",
        ],
        "success_criteria": [
            "The new capability is source-owned.",
            "A downstream consumer displays it correctly.",
        ],
        "validation_plan": [
            "Run the producer test and one downstream consumer test.",
            "Refresh the real artifact path and inspect the new field in output.",
        ],
        "governance_overlays": [
            "Pick the narrowest non-live change class that matches the touched path.",
            "If the touched path approaches live execution, stop at guard/diagnostic work unless explicitly authorized.",
        ],
        "forbidden_moves": [
            "Do not default to top-level brief derivation first.",
            "Do not add two new mirror layers for one state change.",
        ],
        "prompt_focus": "Operate in source-first mode. Prefer producer truth over consumer reconstruction.",
    },
    "ui_rendering": {
        "brief": "Build maintainable visibility using existing source artifacts and minimal backend assumptions.",
        "hard_guardrails": [
            "Do not invent backend state just to support the panel.",
            "Prefer direct source snapshots over debug mirror fields.",
            "Optimize for visibility of chain propagation and impact scope.",
        ],
        "execution_priorities": [
            "Use current source artifacts as the view model.",
            "Expose heads, queues, blockers, and chain propagation.",
            "Keep rendering maintainable and lightweight.",
        ],
        "execution_path": [
            "Inspect what source artifacts already expose.",
            "Build a view model from real artifacts.",
            "Render task progress, transmission path, and impact range in one panel.",
        ],
        "stop_rules": [
            "Stop once the user can see current head, queue, blockers, and transmission path.",
            "Stop if the next step requires backend semantics that are not actually needed for visibility.",
        ],
        "success_criteria": [
            "The panel reads current source artifacts and shows real state.",
            "A user can understand progress and chain impact without reading raw JSON.",
        ],
        "validation_plan": [
            "Run a focused UI/source test.",
            "Generate a real panel artifact and inspect current live state rendering.",
        ],
        "governance_overlays": [
            "UI changes must consume source artifacts rather than becoming a hidden source of truth.",
            "Prefer operator snapshots and propagation views over new backend semantics.",
        ],
        "forbidden_moves": [
            "Do not patch minified bundles when a maintainable source-driven path is available.",
            "Do not make the UI the new source of truth.",
        ],
        "prompt_focus": "Operate in UI-rendering mode. Favor direct source visibility over backend invention.",
    },
    "research_backtest": {
        "brief": "Evaluate strategy validity with deterministic, report-driven backtesting.",
        "hard_guardrails": [
            "No future functions or implicit look-ahead.",
            "Use explicit timestamps for temporal logic.",
            "Compare in-sample and out-of-sample instead of reporting one headline metric.",
        ],
        "execution_priorities": [
            "Protect against leakage first.",
            "Measure current performance and recent OOS behavior.",
            "State where the strategy does not generalize.",
        ],
        "execution_path": [
            "Run deterministic backtest or comparison script.",
            "Generate a recent comparison artifact.",
            "Extract practical conclusions, not only raw metrics.",
        ],
        "stop_rules": [
            "Stop after a current comparison artifact exists and caveats are explicit.",
            "Stop if the next step only adds more metrics without changing the conclusion.",
        ],
        "success_criteria": [
            "A recent report exists with specific trades, expectancy, and profit factor context.",
            "OOS conclusions are explicit and bounded.",
        ],
        "validation_plan": [
            "Run targeted test coverage for the backtest/report script.",
            "Inspect recent artifact output directly.",
        ],
        "governance_overlays": [
            "Use explicit event timestamps for business logic and monotonic clocks only for runtime deadlines or backoff.",
            "Prefer replay/shadow evidence before any claim that a research result should influence live behavior.",
        ],
        "forbidden_moves": [
            "Do not promote a strategy based only on in-sample results.",
            "Do not hide low-trade or weak-PF caveats.",
        ],
        "prompt_focus": "Operate in research-backtest mode. Determinism and OOS truth come before narrative.",
    },
    "live_guard_diagnostics": {
        "brief": "Run read-only diagnostics that separate profitability, account scope, and execution readiness.",
        "hard_guardrails": [
            "No order placement without explicit live gates and user approval.",
            "Account scope must be explicit: spot, futures, or portfolio margin.",
            "End with blockers and clearing conditions, not vague health summaries.",
        ],
        "execution_priorities": [
            "Determine the real account scope first.",
            "Separate profitability confirmation from trade readiness.",
            "Make ops_live_gate and risk_guard blockers concrete.",
        ],
        "execution_path": [
            "Probe the relevant account scope in read-only mode.",
            "Resolve live readiness blockers.",
            "Summarize what still prevents takeover or execution.",
        ],
        "stop_rules": [
            "Stop after account scope, blocker reasons, and clearing conditions are explicit.",
            "Stop if the next step would require live action.",
        ],
        "success_criteria": [
            "Readiness blockers are concrete and actionable.",
            "No live order path was touched.",
        ],
        "validation_plan": [
            "Run the targeted diagnostic script and its focused tests.",
            "Inspect the current handoff/blocker artifact for real blocker codes.",
        ],
        "governance_overlays": [
            "Separate account scope, profitability, and trade readiness; they are not interchangeable.",
            "Notification or observability failures must not be escalated as execution-path failures unless source-of-truth ambiguity appears.",
        ],
        "forbidden_moves": [
            "Do not infer readiness from profitability alone.",
            "Do not blur account scope with spot-only observations.",
        ],
        "prompt_focus": "Operate in live-guard diagnostics mode. Read-only, scope-first, blocker-explicit.",
    },
    "default_implementation": {
        "brief": "Apply the smallest useful change with explicit validation and early stop.",
        "hard_guardrails": [
            "Keep the change scoped to the user-visible goal.",
            "Do not expand into adjacent cleanup unless it blocks the task.",
        ],
        "execution_priorities": [
            "Identify the smallest useful implementation path.",
            "Validate the touched surface.",
            "Stop when the result is real and visible.",
        ],
        "execution_path": [
            "Inspect context.",
            "Implement the minimum viable change.",
            "Validate and report outcome.",
        ],
        "stop_rules": [
            "Stop when the requested user-visible capability exists and tests pass.",
            "Stop if further work is polish only.",
        ],
        "success_criteria": [
            "The requested capability exists.",
            "The touched path is validated.",
        ],
        "validation_plan": [
            "Run the narrowest relevant test pack.",
            "Check the real output path if one exists.",
        ],
        "governance_overlays": [
            "Choose the lowest-risk change class that still accomplishes the task.",
            "Prefer shadow/replay/canary progression over broad rollout when behavior could affect execution.",
        ],
        "forbidden_moves": [
            "Do not widen scope without a blocker.",
        ],
        "prompt_focus": "Operate in default implementation mode. Prefer fast convergence over broad cleanup.",
    },
}


LIVE_EXECUTION_PATH_HINTS = (
    "binance_live_takeover.py",
    "openclaw_cloud_bridge.sh",
    "trade_live",
    "order adapter",
    "order routing",
    "execution route",
    "place_market_order",
    "allow-live-order",
)

LIVE_GUARD_PATH_HINTS = (
    "live_gate",
    "remote_live",
    "ready-check",
    "risk_guard",
    "ops_live_gate",
    "probe",
    "takeover",
    "account scope",
)

SIMULATION_HINTS = (
    "backtest",
    "replay",
    "simulation",
    "oos",
    "out-of-sample",
)


MODE_SIGNALS: dict[str, dict[str, list[str]]] = {
    "architecture_review": {
        "summary": ["architecture", "drift", "ownership", "cycle", "review", "audit", "full architecture", "source of truth", "逻辑偏移", "架构"],
        "paths": ["ARCHITECTURE_REVIEW", "refresh_cross_market_operator_state.py", "build_hot_universe_operator_brief.py", "operator_context_sections.py"],
    },
    "source_first_implementation": {
        "summary": ["lane", "queue", "handoff", "source", "artifact", "brief", "refresh", "slot", "mirror field", "state field", "source-owned"],
        "paths": ["build_", "refresh_", "operator_brief_source_chunks.py", "output/review"],
    },
    "ui_rendering": {
        "summary": ["frontend", "panel", "dashboard", "visual", "window", "ui", "render", "可视化", "面板", "前端"],
        "paths": ["dashboard/web", "dashboard/api", ".html", ".css", ".tsx", ".jsx", ".ts", ".js"],
    },
    "research_backtest": {
        "summary": ["backtest", "study", "profitability", "out-of-sample", "expectancy", "strategy", "report", "research", "回测", "盈利能力"],
        "paths": ["backtest_", "strategy_lab", "recent_strategy_backtest_comparison_report", "brooks_price_action"],
    },
    "live_guard_diagnostics": {
        "summary": ["live", "probe", "ready-check", "takeover", "remote", "risk guard", "ops_live_gate", "portfolio margin", "实盘", "诊断"],
        "paths": ["openclaw_cloud_bridge.sh", "binance_live_takeover.py", "remote_live", "live_gate"],
    },
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


def fmt_utc(value: dt.datetime | None) -> str:
    effective = value or now_utc()
    return effective.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def score_mode(*, mode: str, task_summary: str, changed_paths: list[str]) -> tuple[int, list[str]]:
    summary_text = normalize(task_summary)
    path_texts = [normalize(path) for path in changed_paths]
    signals = MODE_SIGNALS.get(mode, {})
    score = 0
    reason_codes: list[str] = []
    for token in signals.get("summary", []):
        needle = normalize(token)
        if needle and needle in summary_text:
            score += 4
            reason_codes.append(f"summary:{token}")
    for token in signals.get("paths", []):
        needle = normalize(token)
        if any(needle in path for path in path_texts):
            score += 3
            reason_codes.append(f"path:{token}")
    if mode == "source_first_implementation":
        if any(path.endswith(".py") for path in path_texts):
            score += 1
            reason_codes.append("path:python_impl")
    if mode == "ui_rendering":
        if any(path.endswith((".html", ".css", ".js", ".ts", ".tsx", ".jsx")) for path in path_texts):
            score += 2
            reason_codes.append("path:web_impl")
    if mode == "research_backtest":
        if "out of sample" in summary_text or "oos" in summary_text:
            score += 2
            reason_codes.append("summary:oos")
    if mode == "live_guard_diagnostics":
        if "read-only" in summary_text or "只读" in summary_text:
            score += 2
            reason_codes.append("summary:read_only")
    return score, reason_codes


def select_modes(*, task_summary: str, changed_paths: list[str]) -> dict[str, Any]:
    scored: list[tuple[str, int, list[str]]] = []
    for mode in MODE_LIBRARY:
        if mode == "default_implementation":
            continue
        score, reason_codes = score_mode(mode=mode, task_summary=task_summary, changed_paths=changed_paths)
        if score > 0:
            scored.append((mode, score, reason_codes))
    if not scored:
        primary = "default_implementation"
        primary_score = 1
        primary_reasons = ["fallback:default"]
        secondary: list[dict[str, Any]] = []
    else:
        scored.sort(key=lambda row: (row[1], MODE_PRIORITY.get(row[0], 0), row[0]), reverse=True)
        primary, primary_score, primary_reasons = scored[0]
        secondary = [
            {"mode": mode, "score": score, "reason_codes": reasons}
            for mode, score, reasons in scored[1:3]
        ]
    return {
        "primary_mode": primary,
        "primary_score": primary_score,
        "primary_reason_codes": primary_reasons,
        "secondary_modes": secondary,
    }


def recommend_change_class(*, task_summary: str, changed_paths: list[str], primary_mode: str) -> dict[str, Any]:
    summary_text = normalize(task_summary)
    path_texts = [normalize(path) for path in changed_paths]
    combined = " ".join([summary_text, *path_texts])

    if any(hint in combined for hint in map(normalize, LIVE_EXECUTION_PATH_HINTS)):
        return {
            "recommended_change_class": "LIVE_EXECUTION_PATH",
            "rationale": "Touched paths or task text imply direct live execution behavior.",
        }
    if primary_mode == "live_guard_diagnostics" or any(
        hint in combined for hint in map(normalize, LIVE_GUARD_PATH_HINTS)
    ):
        return {
            "recommended_change_class": "LIVE_GUARD_ONLY",
            "rationale": "Task is scoped to probes, blockers, account scope, or live diagnostics.",
        }
    if primary_mode == "research_backtest" or any(
        hint in combined for hint in map(normalize, SIMULATION_HINTS)
    ):
        return {
            "recommended_change_class": "SIM_ONLY",
            "rationale": "Task is simulation/backtest/replay oriented and should not touch live behavior.",
        }
    if any(path.endswith((".md", ".txt")) for path in path_texts) and not any(
        path.endswith((".py", ".sh", ".js", ".ts", ".tsx", ".jsx", ".html", ".css")) for path in path_texts
    ):
        return {
            "recommended_change_class": "DOC_ONLY",
            "rationale": "Only documentation-like files appear touched.",
        }
    return {
        "recommended_change_class": "RESEARCH_ONLY",
        "rationale": "Task affects non-live orchestration, artifacts, or presentation layers.",
    }


def build_payload(*, task_summary: str, changed_paths: list[str], now_dt: dt.datetime) -> dict[str, Any]:
    selected = select_modes(task_summary=task_summary, changed_paths=changed_paths)
    primary_mode = str(selected["primary_mode"])
    template = MODE_LIBRARY[primary_mode]
    change_class = recommend_change_class(
        task_summary=task_summary,
        changed_paths=changed_paths,
        primary_mode=primary_mode,
    )
    payload = {
        "action": "build_adaptive_task_mode_report",
        "status": "ok",
        "ok": True,
        "as_of": fmt_utc(now_dt),
        "task_summary": task_summary,
        "changed_paths": changed_paths,
        "primary_mode": primary_mode,
        "primary_mode_brief": template["brief"],
        "primary_score": int(selected["primary_score"]),
        "primary_reason_codes": list(selected["primary_reason_codes"]),
        "secondary_modes": list(selected["secondary_modes"]),
        "hard_guardrails": list(template["hard_guardrails"]),
        "execution_priorities": list(template["execution_priorities"]),
        "execution_path": list(template["execution_path"]),
        "stop_rules": list(template["stop_rules"]),
        "success_criteria": list(template["success_criteria"]),
        "validation_plan": list(template["validation_plan"]),
        "governance_overlays": list(template.get("governance_overlays", [])),
        "forbidden_moves": list(template["forbidden_moves"]),
        "prompt_focus": str(template["prompt_focus"]),
        "global_guardrails": [
            "Protect live capital and keep live diagnostics read-only unless gates are explicitly clear.",
            "Use source-owned state before adding consumer-side mirrors.",
            "Stop if two consecutive turns do not change state, queue, gate, source ownership, or user-facing capability.",
        ],
        "failure_severity_matrix": FAILURE_SEVERITY_MATRIX,
        "source_of_truth_registry": SOURCE_OF_TRUTH_REGISTRY,
        "change_class_library": CHANGE_CLASS_LIBRARY,
        "recommended_change_class": str(change_class["recommended_change_class"]),
        "recommended_change_class_rationale": str(change_class["rationale"]),
        "validation_report_requirements": VALIDATION_REPORT_REQUIREMENTS,
        "rollout_ladder": ROLLOUT_LADDER,
    }
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join([f"- {item}" for item in items]) if items else "- -"

    secondary_lines = [
        f"- `{row.get('mode')}` score={int(row.get('score') or 0)} reasons={', '.join(row.get('reason_codes') or []) or '-'}"
        for row in payload.get("secondary_modes", [])
    ] or ["- -"]
    primary_reason_lines = [
        f"- `{reason}`" for reason in payload.get("primary_reason_codes", [])
    ] or ["- -"]

    changed_paths = [f"- `{path}`" for path in payload.get("changed_paths", [])] or ["- -"]
    failure_lines = [
        f"- `{row.get('severity')}` `{row.get('scope')}` -> `{row.get('required_action')}` | {row.get('notes')}"
        for row in payload.get("failure_severity_matrix", [])
    ] or ["- -"]
    sot_lines = [
        f"- `{row.get('domain')}`: {row.get('rule')}"
        for row in payload.get("source_of_truth_registry", [])
    ] or ["- -"]

    return "\n".join(
        [
            "# Adaptive Task Mode Report",
            "",
            f"- as_of: `{payload.get('as_of') or '-'}`",
            f"- primary_mode: `{payload.get('primary_mode') or '-'}`",
            f"- primary_mode_brief: `{payload.get('primary_mode_brief') or '-'}`",
            f"- primary_score: `{payload.get('primary_score') or 0}`",
            f"- prompt_focus: `{payload.get('prompt_focus') or '-'}`",
            "",
            "## Task Summary",
            "",
            f"{payload.get('task_summary') or '-'}",
            "",
            "## Changed Paths",
            "",
            *changed_paths,
            "",
            "## Primary Reasons",
            "",
            *primary_reason_lines,
            "",
            "## Secondary Modes",
            "",
            *secondary_lines,
            "",
            "## Global Guardrails",
            "",
            bullets(list(payload.get("global_guardrails", []))),
            "",
            "## Hard Guardrails",
            "",
            bullets(list(payload.get("hard_guardrails", []))),
            "",
            "## Governance Overlays",
            "",
            bullets(list(payload.get("governance_overlays", []))),
            "",
            "## Recommended Change Class",
            "",
            f"- class: `{payload.get('recommended_change_class') or '-'}`",
            f"- rationale: {payload.get('recommended_change_class_rationale') or '-'}",
            "",
            "## Failure Severity Matrix",
            "",
            *failure_lines,
            "",
            "## Source-of-Truth Registry",
            "",
            *sot_lines,
            "",
            "## Execution Priorities",
            "",
            bullets(list(payload.get("execution_priorities", []))),
            "",
            "## Execution Path",
            "",
            bullets(list(payload.get("execution_path", []))),
            "",
            "## Stop Rules",
            "",
            bullets(list(payload.get("stop_rules", []))),
            "",
            "## Success Criteria",
            "",
            bullets(list(payload.get("success_criteria", []))),
            "",
            "## Validation Plan",
            "",
            bullets(list(payload.get("validation_plan", []))),
            "",
            "## Validation Report Requirements",
            "",
            bullets(list(payload.get("validation_report_requirements", []))),
            "",
            "## Rollout Ladder",
            "",
            bullets(list(payload.get("rollout_ladder", []))),
            "",
            "## Forbidden Moves",
            "",
            bullets(list(payload.get("forbidden_moves", []))),
            "",
        ]
    ) + "\n"


def prune_review_artifacts(
    review_dir: Path,
    *,
    stem: str,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
    now_dt: dt.datetime,
) -> tuple[list[str], list[str]]:
    cutoff = now_dt - dt.timedelta(hours=max(1.0, ttl_hours))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an adaptive task-mode report.")
    parser.add_argument("--task-summary", required=True)
    parser.add_argument("--changed-path", action="append", default=[])
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-keep", type=int, default=30)
    parser.add_argument("--artifact-ttl-hours", type=float, default=240.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now_dt = parse_now(args.now)
    review_dir = args.review_dir.expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(task_summary=str(args.task_summary), changed_paths=list(args.changed_path), now_dt=now_dt)

    stamp = now_dt.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_adaptive_task_mode_report.json"
    markdown_path = review_dir / f"{stamp}_adaptive_task_mode_report.md"
    checksum_path = review_dir / f"{stamp}_adaptive_task_mode_report_checksum.json"

    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_path.write_text(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "markdown": str(markdown_path),
                "sha256_artifact": sha256_file(artifact_path),
                "sha256_markdown": sha256_file(markdown_path),
                "generated_at": payload.get("as_of"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="adaptive_task_mode_report",
        current_paths=[artifact_path, markdown_path, checksum_path],
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=now_dt,
    )
    payload["artifact"] = str(artifact_path)
    payload["markdown"] = str(markdown_path)
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
