import React from "react";
import { Badge, PanelCard, SegmentedControl } from "../ui";
import { bilingualLabel } from "../../lib/i18n";
import { labelFor } from "../../utils/dictionary";

const t = (key) => labelFor(key);

const TONE_FILTERS = [
  { id: "all", label: t("all") },
  { id: "positive", label: t("positive") },
  { id: "warning", label: t("warning") },
  { id: "negative", label: t("blocked") },
];

export function ArtifactListPane({
  domains,
  selectedDomain,
  filteredArtifacts,
  selectedArtifact,
  search,
  onSearch,
  toneFilter,
  onToneFilter,
  onSelectArtifact,
}) {
  const domainLabel = selectedDomain === "all"
    ? t("legacy_explorer_all_domains")
    : (domains.find((domain) => domain.id === selectedDomain)?.labelDisplay || selectedDomain);

  return (
    <PanelCard title={t("legacy_explorer_list_title")} tone={selectedArtifact?.status || "read_only"} className="explorer-list">
      <div className="control-stack">
        <input
          type="search"
          className="search-input"
          value={search}
          onChange={(event) => onSearch(event.target.value)}
          placeholder={t("legacy_explorer_search_placeholder")}
          data-testid="artifact-search"
        />
        <SegmentedControl items={TONE_FILTERS} active={toneFilter} onChange={onToneFilter} />
      </div>
      <div className="panel-meta">{`${domainLabel} · ${t("workspace_summary_prefix_total")} ${filteredArtifacts.length} ${t("legacy_explorer_results_suffix")}`}</div>
      <div className="stack-list" data-testid="artifact-list">
        {filteredArtifacts.map((row) => (
          <button
            key={row.id}
            className={`stack-item ${selectedArtifact?.id === row.id ? "stack-item-active" : ""}`.trim()}
            onClick={() => onSelectArtifact(row.id)}
            data-testid={`artifact-item-${row.payloadKey || row.id}`}
          >
            <div className="stack-item-top">
              <strong>{row.labelDisplay || row.label}</strong>
              <Badge value={row.status || row.changeClass || row.category}>
                {row.statusDisplay || row.changeClassDisplay || bilingualLabel(row.category)}
              </Badge>
            </div>
            <div className="stack-item-sub">{row.categoryDisplay || row.category} · {row.path}</div>
            <div className="stack-item-note">{row.researchDecisionDisplay || row.researchDecision || row.takeaway || t("legacy_no_extra_note")}</div>
          </button>
        ))}
      </div>
    </PanelCard>
  );
}
