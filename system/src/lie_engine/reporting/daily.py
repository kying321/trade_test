from __future__ import annotations

from datetime import date

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
) -> str:
    lines: list[str] = []
    lines.append(f"# 📊 离厄反脆弱简报 | {as_of.isoformat()}")
    lines.append("")
    lines.append("## 🌡️ 市场温度计")
    lines.append(f"当前体制：**{regime.consensus.value}**；保护模式：**{'是' if regime.protection_mode else '否'}**")
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
