import React from "react";
import { PanelCard, SimpleTable, MetricGrid, Badge } from "./ui";
import { cardList } from "../lib/formatters";
import { bilingualLabel, zhChangeClass, zhLabel } from "../lib/i18n";
import { labelFor } from "../utils/dictionary";

const t = (key) => labelFor(key);

export function InterfacePanel({ interfaces, sourceHeads, uiRoutes, meta, snapshot }) {
  return (
    <div className="page-stack">
      <PanelCard title={t("legacy_interface_directory_title")} tone="public" meta={t("legacy_interface_directory_meta")}>
        <div className="interface-grid">
          {interfaces.map((item) => (
            <a className="interface-card" href={item.url} target="_blank" rel="noreferrer" key={item.id}>
              <div className="stack-item-top">
                <strong>{item.displayLabel || item.label}</strong>
                <Badge value={item.kind}>{item.displayKind || zhLabel(item.kind)}</Badge>
              </div>
              <div className="stack-item-sub">{item.url}</div>
              <div className="stack-item-note">{item.description || item.kind}</div>
            </a>
          ))}
        </div>
      </PanelCard>

      <div className="two-column-grid">
        <PanelCard title={t("legacy_interface_source_heads_title")} tone="source" className="page-panel-large">
          <SimpleTable
            columns={[
              { key: "displayLabel", label: t("source") },
              { key: "displayStatus", label: t("status") },
              { key: "summary", label: t("summary") },
              { key: "path", label: t("path") },
            ]}
            rows={sourceHeads}
          />
        </PanelCard>
        <PanelCard title={t("legacy_interface_snapshot_title")} tone={meta?.change_class || "neutral"} className="page-panel-small">
          <MetricGrid
            items={cardList([
              { label: t("frontend_dev"), value: uiRoutes?.frontend_dev },
              { label: t("frontend_public"), value: uiRoutes?.frontend_public },
              { label: t("frontend_pages"), value: uiRoutes?.frontend_pages },
              { label: t("link_operator_panel"), value: uiRoutes?.operator_panel },
              { label: t("snapshot_json"), value: uiRoutes?.snapshot_json },
              { label: t("legacy_generated_time_utc"), value: meta?.generated_at_utc },
              { label: t("artifact_payload_count"), value: meta?.artifact_payload_count },
              { label: t("backtest_artifact_count"), value: meta?.backtest_artifact_count },
              { label: t("change_class"), value: zhChangeClass(meta?.change_class) },
            ])}
            compact
          />
        </PanelCard>
      </div>

      <PanelCard title={t("legacy_interface_raw_contract_title")} tone="read_only" meta={t("legacy_interface_raw_contract_meta")}>
        <pre className="json-viewer">{JSON.stringify({
          ui_routes: snapshot?.ui_routes,
          interface_catalog: snapshot?.interface_catalog,
          source_heads: snapshot?.source_heads,
          domain_summary: snapshot?.domain_summary,
          experience_contract: snapshot?.experience_contract,
          field_translation_policy: {
            mode: t("legacy_translation_policy_mode"),
            examples: [
              bilingualLabel("research_decision"),
              bilingualLabel("active_baseline"),
              bilingualLabel("blocker_detail"),
            ],
          },
        }, null, 2)}</pre>
      </PanelCard>
    </div>
  );
}
