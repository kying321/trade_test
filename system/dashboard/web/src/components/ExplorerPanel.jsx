import React from "react";
import { PanelCard } from "./ui";
import { ExplorerBreadcrumbs } from "./explorer/ExplorerBreadcrumbs";
import { ExplorerDomainPane } from "./explorer/ExplorerDomainPane";
import { ArtifactListPane } from "./explorer/ArtifactListPane";
import { ArtifactInspectorPane } from "./explorer/ArtifactInspectorPane";

export function ExplorerPanel({
  domains,
  selectedDomain,
  filteredArtifacts,
  selectedArtifact,
  detailTab,
  onDetailTabChange,
  search,
  onSearch,
  toneFilter,
  onToneFilter,
  onSelectDomain,
  onSelectArtifact,
  breadcrumbs,
  onBack,
  canGoBack,
}) {
  return (
    <div className="page-stack explorer-page">
      <PanelCard className="page-panel-full">
        <ExplorerBreadcrumbs items={breadcrumbs} onBack={onBack} canGoBack={canGoBack} />
      </PanelCard>

      <div className="explorer-layout">
        <ExplorerDomainPane
          domains={domains}
          selectedDomain={selectedDomain}
          onSelectDomain={onSelectDomain}
        />

        <ArtifactListPane
          domains={domains}
          selectedDomain={selectedDomain}
          filteredArtifacts={filteredArtifacts}
          selectedArtifact={selectedArtifact}
          search={search}
          onSearch={onSearch}
          toneFilter={toneFilter}
          onToneFilter={onToneFilter}
          onSelectArtifact={onSelectArtifact}
        />

        <ArtifactInspectorPane
          selectedArtifact={selectedArtifact}
          detailTab={detailTab}
          onDetailTabChange={onDetailTabChange}
        />
      </div>
    </div>
  );
}
