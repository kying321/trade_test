import React, { Suspense, lazy } from 'react';
import { ErrorBoundary } from './components/ErrorBoundary.legacy';
import { HeroSidebar } from './components/HeroSidebar';
import { EmptyCard } from './components/ui';
import { PageHeader } from './components/PageHeader';
import { useSnapshot } from './hooks/useSnapshot';
import { useBacktestSelection } from './hooks/useBacktestSelection';
import { useExplorerState } from './hooks/useExplorerState';
import { usePageNavigation } from './hooks/usePageNavigation';
import { MAIN_TABS } from './lib/navigation';
import { labelFor } from './utils/dictionary';

function lazyNamed(factory, name) {
  return lazy(() => factory().then((module) => ({ default: module[name] })));
}
const t = (key) => labelFor(key);

const OverviewPanel = lazyNamed(() => import('./components/OverviewPanel'), 'OverviewPanel');
const BacktestWorkbench = lazyNamed(() => import('./components/BacktestWorkbench'), 'BacktestWorkbench');
const ExplorerPanel = lazyNamed(() => import('./components/ExplorerPanel'), 'ExplorerPanel');
const InterfacePanel = lazyNamed(() => import('./components/InterfacePanel'), 'InterfacePanel');

function DashboardApp() {
  const { snapshot, normalized, loading, error, refresh, snapshotPath } = useSnapshot();
  const { meta, uiRoutes, interfaces, sourceHeads, domains, artifactRows, backtests, comparisonReport, holdHandoff, holdGate, operatorPanel } = normalized;
  const { activeTab, setActiveTab, pageMeta } = usePageNavigation('overview');
  const {
    selectedBacktest,
    setSelectedBacktestId,
  } = useBacktestSelection(backtests);
  const {
    selectedDomain,
    selectedArtifact,
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
  } = useExplorerState({ artifactRows, domains });

  function jumpToExplorer(domain = 'all') {
    setActiveTab('explorer');
    selectDomain(domain);
  }

  const headerBadges = [
    { label: t('read_only'), value: 'read_only' },
    { label: meta?.change_class || 'RESEARCH_ONLY', value: meta?.change_class || 'RESEARCH_ONLY' },
    { label: t('source_owned'), value: 'active' },
  ];

  return (
    <div className="app-shell">
      <div className="app-layout">
        <HeroSidebar
          meta={meta}
          uiRoutes={uiRoutes}
          holdHandoff={holdHandoff}
          sourceHeads={sourceHeads}
          activeTab={activeTab}
          navItems={MAIN_TABS}
          onTabChange={setActiveTab}
          onRefresh={refresh}
          loading={loading}
          snapshotPath={snapshotPath}
        />

        <main className="content-shell">
          <PageHeader
            eyebrow={`Fenlie / ${pageMeta.eyebrow || t('legacy_workspace_fallback')}`}
            title={pageMeta.title}
            description={pageMeta.subtitle}
            badges={headerBadges}
          />

          {error ? <div className="error-banner">{t('load_failed_prefix')}：{error}</div> : null}
          {!snapshot && loading ? <EmptyCard title={t('legacy_loading_title')} body={t('legacy_loading_snapshot_body')} /> : null}

          <Suspense fallback={<EmptyCard title={t('legacy_component_loading_title')} body={t('legacy_component_loading_body')} />}>
            {activeTab === 'overview' ? (
              <OverviewPanel
                meta={meta}
                uiRoutes={uiRoutes}
                comparisonReport={comparisonReport}
                holdHandoff={holdHandoff}
                holdGate={holdGate}
                operatorPanel={operatorPanel}
                domains={domains}
                interfaces={interfaces}
                sourceHeads={sourceHeads}
                onJumpExplorer={jumpToExplorer}
                onJumpBacktests={() => setActiveTab('backtests')}
                onJumpInterfaces={() => setActiveTab('interfaces')}
              />
            ) : null}

            {activeTab === 'backtests' ? (
              <BacktestWorkbench
                backtests={backtests}
                selectedBacktest={selectedBacktest}
                onSelectBacktest={setSelectedBacktestId}
                comparisonReport={comparisonReport}
              />
            ) : null}

            {activeTab === 'explorer' ? (
              <ExplorerPanel
                domains={domains}
                selectedDomain={selectedDomain}
                filteredArtifacts={filteredArtifacts}
                selectedArtifact={selectedArtifact}
                detailTab={detailTab}
                onDetailTabChange={setDetailTab}
                search={artifactSearch}
                onSearch={setArtifactSearch}
                toneFilter={toneFilter}
                onToneFilter={setToneFilter}
                onSelectDomain={selectDomain}
                onSelectArtifact={selectArtifact}
                breadcrumbs={breadcrumbs}
                onBack={handleBack}
                canGoBack={canGoBack}
              />
            ) : null}

            {activeTab === 'interfaces' ? (
              <InterfacePanel interfaces={interfaces} sourceHeads={sourceHeads} uiRoutes={uiRoutes} meta={meta} snapshot={snapshot} />
            ) : null}
          </Suspense>

            {activeTab === 'raw' ? (
              <div className="page-stack">
                <div className="panel-card page-panel-full">
                  <div className="panel-header">
                    <h2>{t('legacy_raw_output_title')}</h2>
                    <button className="ghost-button" onClick={resetExplorer}>{t('legacy_reset_browser_state')}</button>
                  </div>
                  <pre className="json-viewer">{JSON.stringify(snapshot || {}, null, 2)}</pre>
                </div>
            </div>
          ) : null}
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <DashboardApp />
    </ErrorBoundary>
  );
}
