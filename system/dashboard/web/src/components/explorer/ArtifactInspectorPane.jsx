import React from "react";
import { PanelCard } from "../ui";
import { ArtifactDetail } from "../ArtifactDetail";
import { labelFor } from "../../utils/dictionary";

const t = (key) => labelFor(key);

const DETAIL_TABS = [
  { id: "summary", label: t("summary") },
  { id: "fields", label: t("fields") },
  { id: "lineage", label: t("lineage") },
  { id: "raw", label: t("raw_json") },
];

export function ArtifactInspectorPane({ selectedArtifact, detailTab, onDetailTabChange }) {
  return (
    <PanelCard title={t("legacy_explorer_detail_title")} tone={selectedArtifact?.status || selectedArtifact?.category || "neutral"} className="explorer-detail">
      <ArtifactDetail artifact={selectedArtifact} detailTab={detailTab} onDetailTabChange={onDetailTabChange} detailTabs={DETAIL_TABS} />
    </PanelCard>
  );
}
