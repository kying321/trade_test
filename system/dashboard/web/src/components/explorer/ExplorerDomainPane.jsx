import React from "react";
import { Badge, PanelCard } from "../ui";
import { labelFor } from "../../utils/dictionary";

const t = (key) => labelFor(key);

export function ExplorerDomainPane({ domains, selectedDomain, onSelectDomain }) {
  return (
    <PanelCard title={t("legacy_explorer_domain_title")} tone={selectedDomain} className="explorer-rail">
      <div className="stack-list" data-testid="domain-rail">
        {domains.map((domain) => (
          <button
            key={domain.id}
            className={`stack-item ${selectedDomain === domain.id ? "stack-item-active" : ""}`.trim()}
            onClick={() => onSelectDomain(domain.id)}
            data-testid={`domain-item-${domain.id}`}
          >
            <div className="stack-item-top">
              <strong>{domain.labelDisplay || domain.label}</strong>
              <Badge value={domain.headline || "neutral"}>{domain.count}</Badge>
            </div>
            <div className="stack-item-sub">{`${t("loaded")} ${domain.payloadCount} / ${t("total")} ${domain.count}`}</div>
            <div className="stack-item-note">{domain.headline || "—"}</div>
          </button>
        ))}
      </div>
    </PanelCard>
  );
}
