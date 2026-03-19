import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

vi.mock("echarts-for-react", () => ({
  default: ({ option }) => <div data-testid="echart-mock">{option ? "echart-ready" : "echart-empty"}</div>,
}));

const mockSnapshot = {
  status: "ok",
  meta: {
    generated_at_utc: "2026-03-19T05:00:00Z",
    artifact_payload_count: 4,
    backtest_artifact_count: 1,
  },
  ui_routes: {
    frontend_dev: "http://127.0.0.1:5173",
    operator_panel: "/operator_task_visual_panel.html",
  },
  backtest_artifacts: [
    {
      id: "bt1",
      label: "backtest_2015-01-01_2026-03-10",
      start: "2015-01-01",
      end: "2026-03-10",
      trades: 5,
      total_return: 0.12,
      annual_return: 0.03,
      max_drawdown: -0.02,
      profit_factor: 1.5,
      positive_window_ratio: 0.8,
      equity_curve_sample: [
        { date: "2026-03-01", equity: 1.0 },
        { date: "2026-03-02", equity: 1.1 },
      ],
    },
  ],
  artifact_payloads: {
    operator_panel: {
      label: "operator_panel",
      category: "operator-panel",
      path: "/tmp/operator_panel.json",
      summary: {
        status: "ok",
        change_class: "RESEARCH_ONLY",
        generated_at_utc: "2026-03-19T05:00:00Z",
        research_decision: "operator_ok",
        top_level_keys: ["summary", "action_queue"],
      },
      payload: {
        status: "ok",
        summary: {
          operator_head_brief: "waiting:XAUUSD",
          review_head_brief: "review:SC2603",
          repair_head_brief: "ready:SLOT_ANOMALY",
        },
        action_queue: [{ rank: 1, area: "ops", symbol: "XAUUSD", action: "wait", priority_score: 99, reason: "pending" }],
      },
    },
    recent_strategy_backtests: {
      label: "recent_strategy_backtests",
      category: "backtest",
      path: "/tmp/recent_backtests.json",
      summary: {
        status: "ok",
        change_class: "RESEARCH_ONLY",
        research_decision: "keep_best",
        top_level_keys: ["rows", "strongest_recent_backtest"],
      },
      payload: {
        status: "ok",
        row_count: 1,
        rows: [
          {
            row_id: "r1",
            strategy_id_canonical: "adaptive_route_strategy",
            market_scope: "crypto",
            trade_count: 12,
            win_rate: 0.6,
            total_return: 0.2,
            profit_factor: 1.4,
            expectancy_r: 0.18,
            recommendation: "keep",
          },
        ],
        strongest_recent_backtest: {
          strategy_id_canonical: "adaptive_route_strategy",
          profit_factor: 1.4,
          expectancy_r: 0.18,
        },
      },
    },
    hold_selection_handoff: {
      label: "hold_selection_handoff",
      category: "research",
      path: "/tmp/handoff.json",
      summary: {
        status: "ok",
        change_class: "SIM_ONLY",
        research_decision: "use_hold_selection_gate_as_canonical_head",
        top_level_keys: ["active_baseline", "local_candidate"],
      },
      payload: {
        active_baseline: "hold16_zero",
        local_candidate: "hold8_zero",
        transfer_watch: ["hold12_zero"],
        demoted_candidate: ["hold24_zero"],
        source_head_status: "gate_override_active",
        research_decision: "use_hold_selection_gate_as_canonical_head",
        recommended_brief: "ETHUSDT hold handoff",
      },
    },
    price_action_breakout_pullback: {
      label: "price_action_breakout_pullback",
      category: "sim-only",
      path: "/tmp/pabp.json",
      summary: {
        status: "ok",
        change_class: "SIM_ONLY",
        research_decision: "mixed_positive",
        top_level_keys: ["selected_params", "validation_metrics"],
      },
      payload: {
        focus_symbol: "ETHUSDT",
        selected_params: { max_hold_bars: 16 },
        validation_metrics: { cumulative_return: 0.02, profit_factor: 1.8 },
      },
    },
  },
  catalog: [
    { id: "operator_panel", payload_key: "operator_panel", category: "operator-panel", label: "operator_panel", status: "ok", path: "/tmp/operator_panel.json" },
    { id: "recent_strategy_backtests", payload_key: "recent_strategy_backtests", category: "backtest", label: "recent_strategy_backtests", status: "ok", path: "/tmp/recent_backtests.json" },
    { id: "hold_selection_handoff", payload_key: "hold_selection_handoff", category: "research", label: "hold_selection_handoff", status: "ok", path: "/tmp/handoff.json", research_decision: "use_hold_selection_gate_as_canonical_head" },
    { id: "price_action_breakout_pullback", payload_key: "price_action_breakout_pullback", category: "sim-only", label: "price_action_breakout_pullback", status: "ok", path: "/tmp/pabp.json", research_decision: "mixed_positive" },
  ],
};

describe("Fenlie dashboard app", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockSnapshot,
      })
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("renders and supports explorer drill-down with back navigation", async () => {
    render(<App />);

    await screen.findByText("前端可视化回测与多层次穿透面板");
    expect(screen.getByText("三层阅读框架")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "工件穿透" }));
    await screen.findByText("域导航");

    fireEvent.click(screen.getByTestId("domain-item-sim-only"));
    await waitFor(() => {
      expect(screen.getByTestId("domain-item-sim-only").className.includes("stack-item-active")).toBe(true);
    });

    fireEvent.click(screen.getByTestId("artifact-item-price_action_breakout_pullback"));
    await waitFor(() => {
      expect(screen.getByTestId("artifact-item-price_action_breakout_pullback").className.includes("stack-item-active")).toBe(true);
    });

    fireEvent.click(screen.getByTestId("detail-tab-fields"));
    fireEvent.click(screen.getByTestId("field-drill-selected_params"));
    await screen.findByText("max_hold_bars");
    fireEvent.click(screen.getByTestId("field-back-button"));
    await screen.findByText("selected_params");

    fireEvent.click(screen.getByRole("button", { name: "← 回退" }));
    await waitFor(() => {
      expect(screen.getByText("takeaway")).toBeTruthy();
    });
  });
});
