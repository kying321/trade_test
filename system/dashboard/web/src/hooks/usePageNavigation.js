import { useMemo, useState } from "react";
import { getActiveTabMeta } from "../lib/navigation";

export function usePageNavigation(defaultTab = "overview") {
  const [activeTab, setActiveTab] = useState(defaultTab);
  const pageMeta = useMemo(() => getActiveTabMeta(activeTab), [activeTab]);

  return {
    activeTab,
    setActiveTab,
    pageMeta,
  };
}
