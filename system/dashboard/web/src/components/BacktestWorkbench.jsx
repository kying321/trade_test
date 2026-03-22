import React from "react";
import { buildComparisonOption, buildEquityOption, ChartCard } from "./charts";
import { formatNumber, formatPercent, cardList } from "../lib/formatters";
import { MetricGrid, PanelCard, SimpleTable } from "./ui";
import { bilingualLabel } from "../lib/i18n";
import { labelFor } from "../utils/dictionary";

const t = (key) => labelFor(key);

export function BacktestWorkbench({ backtests, selectedBacktest, onSelectBacktest, comparisonReport }) {
  return (
    <div className="page-stack">
      <div className="two-column-grid">
        <PanelCard title={t("legacy_backtest_workbench_title")} tone="backtest" className="page-panel-small">
          <div className="control-stack">
            <label className="control-label">
              {t("legacy_backtest_select_sample_label")}
              <select value={selectedBacktest?.id || ""} onChange={(event) => onSelectBacktest(event.target.value)}>
                {backtests.map((item) => (
                  <option key={item.id} value={item.id}>{item.labelDisplay || item.label}</option>
                ))}
              </select>
            </label>
          </div>
          <MetricGrid
            items={cardList([
              { label: bilingualLabel("start"), value: selectedBacktest?.start },
              { label: bilingualLabel("end"), value: selectedBacktest?.end },
              { label: bilingualLabel("trades"), value: formatNumber(selectedBacktest?.trades, 0) },
              { label: bilingualLabel("total_return"), value: formatPercent(selectedBacktest?.total_return) },
              { label: bilingualLabel("annual_return"), value: formatPercent(selectedBacktest?.annual_return) },
              { label: bilingualLabel("max_drawdown"), value: formatPercent(selectedBacktest?.max_drawdown) },
              { label: bilingualLabel("profit_factor"), value: formatNumber(selectedBacktest?.profit_factor) },
              { label: bilingualLabel("positive_window_ratio"), value: formatPercent(selectedBacktest?.positive_window_ratio) },
            ])}
            compact
          />
        </PanelCard>

        <PanelCard title={t("legacy_backtest_equity_drawdown_title")} tone="sampled" className="page-panel-large">
          <ChartCard option={buildEquityOption(selectedBacktest)} height={430} />
        </PanelCard>
      </div>

      <div className="two-column-grid">
        <PanelCard title={t("legacy_backtest_comparison_title")} tone={comparisonReport?.comparison_takeaway || "neutral"} className="page-panel-large">
          <SimpleTable
            columns={[
              { key: "strategy_id_canonical", label: bilingualLabel("strategy_id_canonical") },
              { key: "market_scope", label: bilingualLabel("market_scope") },
              { key: "trade_count", label: bilingualLabel("trade_count"), render: (value) => formatNumber(value, 0) },
              { key: "win_rate", label: bilingualLabel("win_rate"), render: (value) => formatPercent(value) },
              { key: "total_return", label: bilingualLabel("total_return"), render: (value) => formatPercent(value) },
              { key: "profit_factor", label: bilingualLabel("profit_factor"), render: (value) => formatNumber(value) },
              { key: "expectancy_r", label: bilingualLabel("expectancy_r"), render: (value) => formatNumber(value) },
              { key: "recommendation", label: bilingualLabel("recommendation") },
            ]}
            rows={comparisonReport?.rows || []}
          />
        </PanelCard>
        <PanelCard title={t("legacy_backtest_chart_title")} tone="chart" className="page-panel-small">
          <ChartCard option={buildComparisonOption(comparisonReport?.rows || [])} height={360} />
        </PanelCard>
      </div>
    </div>
  );
}
