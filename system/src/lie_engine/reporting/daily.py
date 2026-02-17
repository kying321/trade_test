from __future__ import annotations

from datetime import date
from typing import Any

from lie_engine.data.quality import DataQualityReport
from lie_engine.models import RegimeState, SignalCandidate, TradePlan


def _signal_emoji(confidence: float) -> str:
    if confidence >= 70:
        return "🟢"
    if confidence >= 50:
        return "🟡"
    return "🔴"


def render_daily_briefing(
    as_of: date,
    regime: RegimeState,
    signals: list[SignalCandidate],
    plans: list[TradePlan],
    quality: DataQualityReport,
    black_swan_items: list[str],
    next_events: list[str],
    black_swan_score: float = 0.0,
    non_trade_reasons: list[str] | None = None,
    mode_feedback: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# 📊 离厄反脆弱简报 | {as_of.isoformat()}")
    lines.append("")
    lines.append("## 🌡️ 市场温度计")
    lines.append(f"当前体制：**{regime.consensus.value}**；保护模式：**{'是' if regime.protection_mode else '否'}**")
    lines.append("")
    if mode_feedback:
        lines.append("## 🧭 模式引擎")
        runtime_mode = str(mode_feedback.get("runtime_mode", "base"))
        lines.append(f"- 当前运行模式：`{runtime_mode}`")
        runtime_params = mode_feedback.get("runtime_params", {})
        if isinstance(runtime_params, dict) and runtime_params:
            lines.append(
                "- 模式参数："
                + f"conf>={float(runtime_params.get('signal_confidence_min', 0.0)):.1f}, "
                + f"conv>={float(runtime_params.get('convexity_min', 0.0)):.2f}, "
                + f"hold={int(float(runtime_params.get('hold_days', 0.0)))}, "
                + f"max_trades={int(float(runtime_params.get('max_daily_trades', 0.0)))}"
            )
        today = mode_feedback.get("today", {})
        if isinstance(today, dict) and today:
            lines.append(
                "- 当日模式质量："
                + f"signals={int(today.get('signals', 0))}, "
                + f"plans={int(today.get('plans', 0))}, "
                + f"avg_conf={float(today.get('avg_confidence', 0.0)):.1f}%, "
                + f"avg_conv={float(today.get('avg_convexity', 0.0)):.2f}"
            )
        mode_health = mode_feedback.get("mode_health", {})
        if isinstance(mode_health, dict) and mode_health:
            lines.append(
                "- 模式健康："
                + f"passed={bool(mode_health.get('passed', True))}, "
                + f"active={bool(mode_health.get('active', False))}, "
                + f"reason={str(mode_health.get('reason', 'ok'))}"
            )
        risk_control = mode_feedback.get("risk_control", {})
        if isinstance(risk_control, dict) and risk_control:
            lines.append(
                "- 执行风控节流："
                + f"risk_mult={float(risk_control.get('risk_multiplier', 1.0)):.3f}, "
                + f"source_mult={float(risk_control.get('source_multiplier', 1.0)):.3f}, "
                + f"mode_mult={float(risk_control.get('mode_multiplier', 1.0)):.3f}, "
                + f"mode_reason={str(risk_control.get('mode_reason', 'healthy'))}"
            )
        history = mode_feedback.get("history", {})
        modes = history.get("modes", {}) if isinstance(history, dict) else {}
        if isinstance(modes, dict) and modes:
            lines.append("- 近窗回测模式统计：")
            for mode_name, item in sorted(modes.items()):
                if not isinstance(item, dict):
                    continue
                lines.append(
                    "- "
                    + f"`{mode_name}` "
                    + f"samples={int(item.get('samples', 0))}, "
                    + f"win={float(item.get('avg_win_rate', 0.0)):.2%}, "
                    + f"pf={float(item.get('avg_profit_factor', 0.0)):.3f}, "
                    + f"mdd={float(item.get('worst_drawdown', 0.0)):.2%}, "
                    + f"viol={int(item.get('total_violations', 0))}"
                )
        lines.append("")

    lines.append("## 🔬 体制诊断")
    lines.append(f"- Hurst: `{regime.hurst:.3f}`")
    lines.append(
        f"- HMM: `P(牛)={regime.hmm_probs.get('bull', 0):.1%}, P(震荡)={regime.hmm_probs.get('range', 0):.1%}, P(熊)={regime.hmm_probs.get('bear', 0):.1%}`"
    )
    lines.append(f"- ATR_Z: `{regime.atr_z:.3f}`")
    lines.append(f"- 共识理由：{regime.rationale}")
    lines.append("")

    lines.append("## 🧪 数据质量")
    lines.append(f"- 完整率：`{quality.completeness:.2%}`")
    lines.append(f"- 冲突占比：`{quality.unresolved_conflict_ratio:.2%}`")
    lines.append(f"- 源置信度：`{quality.source_confidence_score:.2%}`")
    lines.append(f"- 低置信源占比：`{quality.low_confidence_source_ratio:.2%}`")
    if quality.source_confidence:
        src_line = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(quality.source_confidence.items()))
        lines.append(f"- 各源评分：`{src_line}`")
    lines.append(f"- 质检标记：`{', '.join(quality.flags) if quality.flags else 'NONE'}`")
    lines.append("")

    lines.append("## 🎯 信号扫描")
    if not signals:
        lines.append("- 今日无可交易信号（或处于保护模式）")
    else:
        for s in signals:
            lines.append(
                f"- `{s.symbol}` {s.side.value} | 位置 `{s.position_score:.1f}` 结构 `{s.structure_score:.1f}` 动能 `{s.momentum_score:.1f}` | 置信 `{s.confidence:.1f}%` {_signal_emoji(s.confidence)} | 凸性 `{s.convexity_ratio:.2f}`"
            )
    lines.append("")

    lines.append("## 📐 交易计划")
    if not plans:
        lines.append("- 无执行计划")
    else:
        lines.append("| 标的 | 方向 | 仓位% | 风险% | 入场 | 止损 | 目标 | 状态 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for p in plans:
            lines.append(
                f"| {p.symbol} | {p.side.value} | {p.size_pct:.2f} | {p.risk_pct:.2f} | {p.entry_price:.3f} | {p.stop_price:.3f} | {p.target_price:.3f} | {p.status} |"
            )
    lines.append("")

    lines.append("## 📉 黑天鹅扫描")
    lines.append(f"- 风险评分：`{black_swan_score:.1f}/100`")
    if black_swan_items:
        for item in black_swan_items:
            lines.append(f"- {item}")
    else:
        lines.append("- 无新增高冲击事件")
    lines.append("")

    lines.append("## ⛔ 不交易条件")
    if non_trade_reasons:
        for reason in non_trade_reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- NONE")
    lines.append("")

    lines.append("## 🗓️ 明日路演")
    if next_events:
        for e in next_events:
            lines.append(f"- {e}")
    else:
        lines.append("- 暂无关键事件")

    return "\n".join(lines) + "\n"
