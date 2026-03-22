import React from 'react';
import { PanelCard } from '../ui';
import { BreadcrumbBar } from './BreadcrumbBar';

export function ExplorerHero({ items, onBack, canGoBack }) {
  return (
    <PanelCard className="page-panel-full">
      <BreadcrumbBar items={items} onBack={onBack} canGoBack={canGoBack} />
    </PanelCard>
  );
}
