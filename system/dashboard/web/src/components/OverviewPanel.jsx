import React from "react";
import { buildComparisonOption, ChartCard } from "./charts";
import { MetricGrid, PanelCard, Badge } from "./ui";
import { buildOverviewCards } from "../lib/display-model";
import { cardList, safeDisplayValue } from "../lib/formatters";
import { bilingualLabel } from "../lib/i18n";
import { labelFor } from "../utils/dictionary";

const t = (key) => labelFor(key);

export function OverviewPanel({ meta, uiRoutes, comparisonReport, holdHandoff, holdGate, operatorPanel, domains, interfaces, sourceHeads, onJumpExplorer, onJumpBacktests, onJumpInterfaces }) {
  return (
    <div className="page-stack">
      <PanelCard title={t("overview_workflow_title")} tone="read_only" meta={t("overview_workflow_meta")}>
        <div className="framework-strip">
          <button className="framework-card" onClick={onJumpBacktests}>
            <span className="metric-label">{t("overview_layer_one_label")}</span>
            <strong>{t("legacy_tab_backtests_title")}</strong>
            <span className="metric-hint">{t("overview_layer_one_hint")}</span>
          </button>
          <button className="framework-card" onClick={() => onJumpExplorer("research")}> 
            <span className="metric-label">{t("overview_layer_two_label")}</span>
            <strong>{t("legacy_tab_explorer_title")}</strong>
            <span className="metric-hint">{t("overview_layer_two_hint")}</span>
          </button>
          <button className="framework-card" onClick={onJumpInterfaces}>
            <span className="metric-label">{t("overview_layer_three_label")}</span>
            <strong>{t("legacy_tab_interfaces_title")}</strong>
            <span className="metric-hint">{t("overview_layer_three_hint")}</span>
          </button>
        </div>
      </PanelCard>

      <div className="two-column-grid">
        <PanelCard title={t("overview_global_panel_title")} tone={meta?.change_class || "neutral"} className="page-panel-large">
          <MetricGrid items={buildOverviewCards(meta, uiRoutes, holdHandoff)} />
          <div className="chart-shell"><ChartCard option={buildComparisonOption(comparisonReport?.rows || [])} height={360} /></div>
        </PanelCard>

        <PanelCard title={t("overview_mainline_panel_title")} tone={holdHandoff?.source_head_status || holdGate?.gate_state?.hold16_baseline_anchor || "neutral"} className="page-panel-small">
          <MetricGrid
            items={cardList([
              { label: t("overview_active_baseline_metric"), value: holdHandoff?.active_baseline },
              { label: t("overview_local_candidate_metric"), value: holdHandoff?.local_candidate },
              { label: t("overview_transfer_watch_metric"), value: holdHandoff?.transfer_watch || [] },
              { label: t("overview_return_candidate_metric"), value: holdHandoff?.return_candidate || [] },
              { label: t("overview_demoted_candidate_metric"), value: holdHandoff?.demoted_candidate || [] },
              { label: t("overview_operator_head_metric"), value: operatorPanel?.summary?.operator_head_brief || operatorPanel?.operator_head_brief },
            ])}
            compact
          />
          <div className="note-block">
            <div className="note-title">{bilingualLabel("recommended_brief")}</div>
            <div>{holdHandoff?.recommended_brief || holdHandoff?.research_note || holdGate?.research_note || "—"}</div>
          </div>
        </PanelCard>
      </div>

      <div className="two-column-grid">
        <PanelCard title={t("overview_domain_map_title")} tone="ok" meta={t("overview_domain_map_meta")} className="page-panel-large">
          <div className="domain-summary-grid">
            {domains.filter((domain) => domain.id !== "all").map((domain) => (
              <button key={domain.id} className="domain-summary-card" onClick={() => onJumpExplorer(domain.id)}>
                <div className="stack-item-top">
                  <strong>{domain.labelDisplay || domain.label}</strong>
                  <Badge value={domain.headline || "neutral"}>{domain.count}</Badge>
                </div>
                <div className="stack-item-sub">{`${t("overview_domain_loaded_prefix")} ${domain.payloadCount} / ${t("overview_domain_total_prefix")} ${domain.count}`}</div>
                <div className="stack-item-note">{domain.headline || "—"}</div>
              </button>
            ))}
          </div>
        </PanelCard>

        <PanelCard title={t("overview_interfaces_title")} tone="public" meta={t("overview_interfaces_meta")} className="page-panel-small">
          <div className="interface-list">
            {interfaces.slice(0, 4).map((item) => (
              <a className="interface-card" href={item.url} target="_blank" rel="noreferrer" key={item.id}>
                <div className="stack-item-top">
                  <strong>{item.displayLabel || item.label}</strong>
                  <Badge value={item.kind}>{item.displayKind || item.kind}</Badge>
                </div>
                <div className="stack-item-sub">{item.url}</div>
                <div className="stack-item-note">{item.description}</div>
              </a>
            ))}
          </div>
        </PanelCard>
      </div>

      <PanelCard title={t("overview_source_heads_title")} tone="source" meta={t("overview_source_heads_meta")}>
        <div className="source-grid">
          {sourceHeads.map((head) => (
            <div className="source-head-card" key={head.id}>
              <div className="stack-item-top">
                <strong>{head.displayLabel || head.label}</strong>
                <Badge value={head.status}>{head.displayStatus || head.status}</Badge>
              </div>
              <div className="stack-item-note">{head.summary || "—"}</div>
              {head.baseline || head.candidate || head.transferWatch || head.returnCandidate || head.returnWatch ? (
                <div className="source-head-tags">
                  {head.baseline ? <span className="soft-chip">{bilingualLabel("active_baseline")}：{safeDisplayValue(head.baseline)}</span> : null}
                  {head.candidate ? <span className="soft-chip">{bilingualLabel("local_candidate")}：{safeDisplayValue(head.candidate)}</span> : null}
                  {head.transferWatch ? <span className="soft-chip">{bilingualLabel("transfer_watch")}：{safeDisplayValue(head.transferWatch)}</span> : null}
                  {head.returnCandidate ? <span className="soft-chip">{bilingualLabel("return_candidate")}：{safeDisplayValue(head.returnCandidate)}</span> : null}
                  {head.returnWatch ? <span className="soft-chip">{bilingualLabel("return_watch")}：{safeDisplayValue(head.returnWatch)}</span> : null}
                </div>
              ) : null}
            </div>
          ))}
        </div>
      </PanelCard>
    </div>
  );
}
