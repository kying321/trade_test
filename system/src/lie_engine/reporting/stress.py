from __future__ import annotations

from datetime import date
from typing import Any


def render_mode_stress_matrix(as_of: date, payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# 模式压力矩阵 | {as_of.isoformat()}")
    lines.append("")
    lines.append(f"- 可用窗口: `{int(payload.get('window_count', 0))}`")
    lines.append(f"- 模式数量: `{int(payload.get('mode_count', 0))}`")
    lines.append(f"- 最优模式(当前规则): `{str(payload.get('best_mode', 'N/A'))}`")
    lines.append("")
    lines.append("## 模式汇总")
    summary_rows = payload.get("mode_summary", [])
    if isinstance(summary_rows, list) and summary_rows:
        for row in summary_rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                "- "
                + f"`{str(row.get('mode', ''))}`: "
                + f"score=`{float(row.get('robustness_score', 0.0)):.3f}`, "
                + f"annual=`{float(row.get('avg_annual_return', 0.0)):.2%}`, "
                + f"max_dd=`{float(row.get('worst_drawdown', 0.0)):.2%}`, "
                + f"win=`{float(row.get('avg_win_rate', 0.0)):.2%}`, "
                + f"pf=`{float(row.get('avg_profit_factor', 0.0)):.3f}`, "
                + f"pwr=`{float(row.get('avg_positive_window_ratio', 0.0)):.2%}`, "
                + f"violations=`{int(row.get('total_violations', 0))}`"
            )
    else:
        lines.append("- 无可用模式汇总")
    lines.append("")
    lines.append("## 窗口明细")
    detail_rows = payload.get("matrix", [])
    if isinstance(detail_rows, list) and detail_rows:
        for row in detail_rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                "- "
                + f"`{str(row.get('mode', ''))}` @ `{str(row.get('window', ''))}` "
                + f"({str(row.get('start', ''))}~{str(row.get('end', ''))})"
                + f": annual=`{float(row.get('annual_return', 0.0)):.2%}`, "
                + f"max_dd=`{float(row.get('max_drawdown', 0.0)):.2%}`, "
                + f"win=`{float(row.get('win_rate', 0.0)):.2%}`, "
                + f"pf=`{float(row.get('profit_factor', 0.0)):.3f}`, "
                + f"pwr=`{float(row.get('positive_window_ratio', 0.0)):.2%}`, "
                + f"trades=`{int(row.get('trades', 0))}`, "
                + f"status=`{str(row.get('status', 'ok'))}`"
            )
    else:
        lines.append("- 无窗口明细")
    lines.append("")
    lines.append("## 执行摩擦压力网格")
    friction = payload.get("execution_friction", {}) if isinstance(payload.get("execution_friction", {}), dict) else {}
    scorecard = friction.get("scorecard", {}) if isinstance(friction.get("scorecard", {}), dict) else {}
    lines.append(f"- 启用: `{bool(friction.get('enabled', False))}`")
    lines.append(f"- 参考模式: `{str(friction.get('reference_mode', 'N/A'))}`")
    lines.append(
        "- scorecard(active/status/gate_ok): "
        + f"`{bool(scorecard.get('active', False))}` / "
        + f"`{str(scorecard.get('status', 'inactive'))}` / "
        + f"`{bool(scorecard.get('gate_ok', True))}`"
    )
    scenario_rows = friction.get("scenario_summary", []) if isinstance(friction.get("scenario_summary", []), list) else []
    if scenario_rows:
        for row in scenario_rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                "- "
                + f"`{str(row.get('scenario', ''))}`: "
                + f"annual=`{float(row.get('avg_annual_return', 0.0)):.2%}`, "
                + f"max_dd=`{float(row.get('worst_drawdown', 0.0)):.2%}`, "
                + f"pf=`{float(row.get('avg_profit_factor', 0.0)):.3f}`, "
                + f"pwr=`{float(row.get('avg_positive_window_ratio', 0.0)):.2%}`, "
                + f"fail_ratio=`{float(row.get('fail_ratio', 0.0)):.2%}`"
            )
    else:
        lines.append("- 无执行摩擦场景汇总")
    friction_checks = scorecard.get("checks", {}) if isinstance(scorecard.get("checks", {}), dict) else {}
    if friction_checks:
        lines.append("- checks:")
        for k, v in friction_checks.items():
            lines.append(f"  - `{k}`: `{bool(v)}`")
    alerts = scorecard.get("alerts", []) if isinstance(scorecard.get("alerts", []), list) else []
    lines.append(f"- alerts: `{', '.join(str(x) for x in alerts) if alerts else 'NONE'}`")
    lines.append("")
    return "\n".join(lines) + "\n"
