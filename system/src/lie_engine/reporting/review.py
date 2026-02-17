from __future__ import annotations

from datetime import date

from lie_engine.models import BacktestResult, ReviewDelta


def render_review_report(as_of: date, backtest: BacktestResult, review: ReviewDelta) -> str:
    lines: list[str] = []
    lines.append(f"# 盘后复盘与参数重估 | {as_of.isoformat()}")
    lines.append("")
    lines.append("## 回测摘要")
    lines.append(f"- 区间: `{backtest.start.isoformat()} ~ {backtest.end.isoformat()}`")
    lines.append(f"- 总收益: `{backtest.total_return:.2%}`")
    lines.append(f"- 年化: `{backtest.annual_return:.2%}`")
    lines.append(f"- 最大回撤: `{backtest.max_drawdown:.2%}`")
    lines.append(f"- 胜率: `{backtest.win_rate:.2%}`")
    lines.append(f"- 盈亏比: `{backtest.profit_factor:.3f}`")
    lines.append(f"- 正收益窗口占比: `{backtest.positive_window_ratio:.2%}`")
    lines.append(f"- 风控违规数: `{backtest.violations}`")
    lines.append("")
    lines.append("## 参数变更")
    for k, v in review.parameter_changes.items():
        lines.append(f"- `{k}` -> `{v:.6f}`")
    if review.change_reasons:
        lines.append("")
        lines.append("### 变更原因")
        for k, reason in review.change_reasons.items():
            lines.append(f"- `{k}`: {reason}")
    if review.rollback_anchor:
        lines.append(f"- 回滚锚点: `{review.rollback_anchor}`")
    lines.append(f"- 影响窗口: `{review.impact_window_days}` 交易日")
    lines.append("")
    lines.append("## 因子权重")
    for k, v in review.factor_weights.items():
        lines.append(f"- `{k}`: `{v:.2%}`")
    if review.factor_contrib_120d:
        lines.append("")
        lines.append("### 近120日边际贡献")
        for k, v in review.factor_contrib_120d.items():
            lines.append(f"- `{k}`: `{v:.4f}`")
    lines.append("")
    lines.append("## 审查结论")
    lines.append(f"- 通过门槛: `{review.pass_gate}`")
    lines.append(f"- 缺陷: `{', '.join(review.defects) if review.defects else 'NONE'}`")
    for note in review.notes:
        lines.append(f"- {note}")

    return "\n".join(lines) + "\n"
