import React, { Suspense } from "react";
import * as echarts from "echarts/core";
import { BarChart, LineChart } from "echarts/charts";
import { GridComponent, LegendComponent, TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import { downsampleCurve } from "../lib/formatters";
import { labelFor } from "../utils/dictionary";

echarts.use([LineChart, BarChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

const ReactECharts = React.lazy(() => import("echarts-for-react/lib/core"));
const t = (key) => labelFor(key);

export function buildEquityOption(backtest) {
  const curve = downsampleCurve(backtest?.equity_curve_sample || []);
  let peak = 0;
  const drawdowns = curve.map((point) => {
    peak = Math.max(peak, point.equity);
    const dd = peak > 0 ? point.equity / peak - 1 : 0;
    return { date: point.date, drawdown: dd };
  });
  return {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    legend: { top: 0, textStyle: { color: "#41536b" } },
    grid: [
      { left: 50, right: 20, top: 40, height: "52%" },
      { left: 50, right: 20, top: "72%", height: "16%" },
    ],
    xAxis: [
      {
        type: "category",
        boundaryGap: false,
        data: curve.map((item) => item.date),
        axisLine: { lineStyle: { color: "rgba(77,99,122,0.24)" } },
        axisLabel: { color: "#5d718c", hideOverlap: true },
      },
      {
        type: "category",
        boundaryGap: false,
        gridIndex: 1,
        data: drawdowns.map((item) => item.date),
        axisLine: { lineStyle: { color: "rgba(77,99,122,0.24)" } },
        axisLabel: { color: "#5d718c", hideOverlap: true },
      },
    ],
    yAxis: [
      {
        type: "value",
        scale: true,
        axisLabel: { color: "#5d718c" },
        splitLine: { lineStyle: { color: "rgba(77,99,122,0.12)" } },
      },
      {
        type: "value",
        gridIndex: 1,
        axisLabel: { color: "#5d718c", formatter: (value) => `${(value * 100).toFixed(0)}%` },
        splitLine: { lineStyle: { color: "rgba(77,99,122,0.12)" } },
      },
    ],
    series: [
      {
        name: t("legacy_chart_equity_curve_series"),
        type: "line",
        smooth: true,
        showSymbol: false,
        data: curve.map((item) => item.equity),
        lineStyle: { width: 2.2, color: "#0b6bcb" },
        areaStyle: { color: "rgba(11,107,203,0.12)" },
      },
      {
        name: t("legacy_chart_drawdown_series"),
        type: "line",
        xAxisIndex: 1,
        yAxisIndex: 1,
        smooth: true,
        showSymbol: false,
        data: drawdowns.map((item) => item.drawdown),
        lineStyle: { width: 1.8, color: "#d46363" },
        areaStyle: { color: "rgba(212,99,99,0.12)" },
      },
    ],
  };
}

export function buildComparisonOption(rows = []) {
  const prepared = rows.slice(0, 10);
  return {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    legend: { top: 0, textStyle: { color: "#41536b" } },
    grid: { left: 50, right: 20, top: 42, bottom: 80 },
    xAxis: {
      type: "category",
      data: prepared.map((row) => row.strategy_id_canonical || row.strategy_id || row.row_id),
      axisLabel: { color: "#5d718c", rotate: 24, interval: 0 },
      axisLine: { lineStyle: { color: "rgba(77,99,122,0.24)" } },
    },
    yAxis: [
      {
        type: "value",
        axisLabel: { color: "#5d718c", formatter: (value) => `${(value * 100).toFixed(0)}%` },
        splitLine: { lineStyle: { color: "rgba(77,99,122,0.12)" } },
      },
      {
        type: "value",
        axisLabel: { color: "#5d718c" },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: t("total_return"),
        type: "bar",
        data: prepared.map((row) => Number(row.total_return ?? 0)),
        itemStyle: { color: "#5f7cff" },
      },
      {
        name: t("expectancy_r"),
        type: "line",
        yAxisIndex: 1,
        smooth: true,
        data: prepared.map((row) => Number(row.expectancy_r ?? 0)),
        itemStyle: { color: "#0f8f77" },
        lineStyle: { width: 2.2 },
      },
    ],
  };
}

export function ChartCard({ option, height = 340 }) {
  return (
    <Suspense fallback={<div className="chart-placeholder" style={{ height }}>{t("legacy_chart_loading")}</div>}>
      <ReactECharts echarts={echarts} option={option} style={{ height }} />
    </Suspense>
  );
}
