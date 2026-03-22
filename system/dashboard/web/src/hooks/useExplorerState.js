import { useCallback, useEffect, useMemo, useState } from "react";
import { pickDefaultArtifact, searchArtifacts } from "../lib/dashboard-data";
import { bilingualLabel } from "../lib/i18n";
import { labelFor } from "../utils/dictionary";

export function useExplorerState({ artifactRows, domains }) {
  const [selectedDomain, setSelectedDomain] = useState("all");
  const [selectedArtifactId, setSelectedArtifactId] = useState("");
  const [artifactSearch, setArtifactSearch] = useState("");
  const [toneFilter, setToneFilter] = useState("all");
  const [detailTab, setDetailTab] = useState("summary");

  const filteredArtifacts = useMemo(
    () => searchArtifacts(artifactRows, selectedDomain, artifactSearch, toneFilter),
    [artifactRows, selectedDomain, artifactSearch, toneFilter]
  );

  useEffect(() => {
    if (!artifactRows.length) return;
    if (!filteredArtifacts.some((row) => row.id === selectedArtifactId)) {
      setSelectedArtifactId(filteredArtifacts[0]?.id || "");
      setDetailTab("summary");
    }
  }, [artifactRows, filteredArtifacts, selectedArtifactId]);

  const selectedArtifact = useMemo(
    () => filteredArtifacts.find((row) => row.id === selectedArtifactId)
      || artifactRows.find((row) => row.id === selectedArtifactId)
      || null,
    [artifactRows, filteredArtifacts, selectedArtifactId]
  );

  const selectedDomainRow = useMemo(
    () => domains.find((item) => item.id === selectedDomain),
    [domains, selectedDomain]
  );

  const canGoBack = detailTab !== "summary" || Boolean(selectedArtifactId) || selectedDomain !== "all";

  const resetExplorer = useCallback(() => {
    setSelectedDomain("all");
    setSelectedArtifactId(pickDefaultArtifact(artifactRows, "all"));
    setDetailTab("summary");
  }, [artifactRows]);

  const handleBack = useCallback(() => {
    if (detailTab !== "summary") {
      setDetailTab("summary");
      return;
    }
    if (selectedArtifactId) {
      setSelectedArtifactId("");
      return;
    }
    if (selectedDomain !== "all") setSelectedDomain("all");
  }, [detailTab, selectedArtifactId, selectedDomain]);

  const selectDomain = useCallback((domain) => {
    setSelectedDomain(domain);
    setSelectedArtifactId(pickDefaultArtifact(artifactRows, domain));
    setDetailTab("summary");
  }, [artifactRows]);

  const selectArtifact = useCallback((artifactId) => {
    setSelectedArtifactId(artifactId);
    setDetailTab("summary");
  }, []);

  const breadcrumbs = useMemo(() => ([
    { label: labelFor("artifact_library"), onClick: resetExplorer },
    ...(selectedDomain !== "all"
      ? [{
          label: selectedDomainRow?.labelDisplay || bilingualLabel(selectedDomain),
          onClick: () => {
            setSelectedArtifactId("");
            setDetailTab("summary");
          },
        }]
      : []),
    ...(selectedArtifact
      ? [{ label: selectedArtifact.labelDisplay || selectedArtifact.label, onClick: () => setDetailTab("summary") }]
      : []),
    ...(selectedArtifact && detailTab !== "summary"
      ? [{ label: bilingualLabel(detailTab), onClick: () => {}, disabled: true }]
      : []),
  ]), [detailTab, resetExplorer, selectedArtifact, selectedDomain, selectedDomainRow]);

  return {
    selectedDomain,
    selectedDomainRow,
    selectedArtifact,
    selectedArtifactId,
    detailTab,
    setDetailTab,
    artifactSearch,
    setArtifactSearch,
    toneFilter,
    setToneFilter,
    filteredArtifacts,
    canGoBack,
    breadcrumbs,
    handleBack,
    resetExplorer,
    selectDomain,
    selectArtifact,
  };
}
