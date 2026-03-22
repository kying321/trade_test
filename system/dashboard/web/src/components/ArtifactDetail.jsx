import React from "react";
import { cardList, safeDisplayValue } from "../lib/formatters";
import { bilingualLabel, zhChangeClass } from "../lib/i18n";
import { FieldExplorer } from "./FieldExplorer";
import { MetricGrid, SimpleTable, Badge } from "./ui";
import { labelFor } from "../utils/dictionary";

const t = (key) => labelFor(key);

const DETAIL_TABS = [
  { id: "summary", label: t("summary") },
  { id: "fields", label: t("fields") },
  { id: "lineage", label: t("lineage") },
  { id: "raw", label: t("raw_json") },
];

export function ArtifactDetail({ artifact, detailTab, onDetailTabChange }) {
  if (!artifact) {
    return <div className="empty-card"><h3>{t("legacy_artifact_select_title")}</h3><p>{t("legacy_artifact_select_body")}</p></div>;
  }

  const summary = artifact.summaryPayload || {};
  const payload = artifact.payload;
  const topFields = cardList([
    { label: bilingualLabel("label"), value: artifact.labelDisplay || artifact.label },
    { label: bilingualLabel("category"), value: artifact.categoryDisplay || artifact.category },
    { label: bilingualLabel("status"), value: artifact.statusDisplay || artifact.status },
    { label: bilingualLabel("change_class"), value: artifact.changeClassDisplay || zhChangeClass(artifact.changeClass) },
    { label: bilingualLabel("research_decision"), value: artifact.researchDecisionDisplay || artifact.researchDecision },
    { label: bilingualLabel("generated_at_utc"), value: artifact.generatedAt },
    { label: bilingualLabel("action"), value: artifact.action },
    { label: bilingualLabel("payload"), value: artifact.hasPayload ? bilingualLabel("loaded") : bilingualLabel("summary-only") },
  ]);

  const lineageRows = [
    { field: bilingualLabel("source_path"), value: artifact.path },
    { field: bilingualLabel("payload_key"), value: artifact.payloadKey },
    { field: bilingualLabel("top_level_keys"), value: (artifact.topLevelKeys || []).join(", ") },
    { field: bilingualLabel("takeaway"), value: artifact.takeaway },
    { field: bilingualLabel("research_decision"), value: artifact.researchDecisionDisplay || artifact.researchDecision },
  ];

  return (
    <div className="detail-shell">
      <div className="panel-header">
        <div>
          <h2>{artifact.labelDisplay || artifact.label}</h2>
          <p className="panel-meta">{artifact.path}</p>
        </div>
        <Badge value={artifact.status || artifact.category}>{artifact.statusDisplay || artifact.status || artifact.categoryDisplay || artifact.category}</Badge>
      </div>

      <MetricGrid items={topFields} compact />

      <div className="detail-tabs">
        {DETAIL_TABS.map((tab) => (
          <button
            key={tab.id}
            className={`detail-tab ${detailTab === tab.id ? "detail-tab-active" : ""}`.trim()}
            onClick={() => onDetailTabChange(tab.id)}
            data-testid={`detail-tab-${tab.id}`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {detailTab === "summary" ? (
        <div className="detail-grid">
          <div className="note-block">
            <div className="note-title">{t("legacy_artifact_key_takeaway_title")}</div>
            <div>{artifact.takeaway || artifact.researchDecision || "—"}</div>
          </div>
          <div className="note-block">
            <div className="note-title">{t("legacy_artifact_top_level_keys_title")}</div>
            <div>{(artifact.topLevelKeys || []).join(", ") || "—"}</div>
          </div>
          <SimpleTable
            columns={[{ key: "field", label: t("field") }, { key: "value", label: t("value") }]}
            rows={Object.entries(summary).slice(0, 16).map(([field, value]) => ({ field: bilingualLabel(field), value: safeDisplayValue(value) }))}
            empty={t("legacy_artifact_empty_summary")}
          />
        </div>
      ) : null}

      {detailTab === "fields" ? (artifact.hasPayload ? <FieldExplorer payload={payload} /> : <div className="empty-card"><h3>{t("legacy_artifact_summary_only_title")}</h3><p>{t("legacy_artifact_summary_only_body")}</p></div>) : null}

      {detailTab === "lineage" ? (
        <div className="detail-grid">
          <SimpleTable columns={[{ key: "field", label: t("field") }, { key: "value", label: t("value") }]} rows={lineageRows} />
          <div className="note-block">
            <div className="note-title">{t("legacy_artifact_consumer_note_title")}</div>
            <div>{t("legacy_artifact_consumer_note_body")}</div>
          </div>
        </div>
      ) : null}

      {detailTab === "raw" ? <pre className="json-viewer">{JSON.stringify(artifact.hasPayload ? payload : summary, null, 2)}</pre> : null}
    </div>
  );
}
