import React from 'react';
import { buildOverviewCards } from '../lib/display-model';
import { safeDisplayValue } from '../lib/formatters';
import { bilingualLabel } from '../lib/i18n';
import { labelFor } from '../utils/dictionary';
import { MetricGrid, PanelCard, Badge } from './ui';

const t = (key) => labelFor(key);

export function HeroSidebar({ meta, uiRoutes, holdHandoff, sourceHeads, activeTab, navItems, onTabChange, onRefresh, loading, snapshotPath }) {
  return (
    <aside className="sidebar-shell">
      <div className="sidebar-hero">
        <p className="eyebrow">{t("sidebar_eyebrow")}</p>
        <h1>{t("sidebar_title")}</h1>
        <p className="hero-copy">
          {t("sidebar_copy")}
        </p>
        <div className="hero-actions hero-actions-vertical">
          <button className="primary-button" onClick={onRefresh}>{loading ? t('refreshing') : t('sidebar_refresh_snapshot')}</button>
          <a className="ghost-link" href={uiRoutes?.operator_panel || '/operator_task_visual_panel.html'} target="_blank" rel="noreferrer">{t('link_operator_panel')}</a>
          <a className="ghost-link" href={uiRoutes?.frontend_public || uiRoutes?.frontend_pages || uiRoutes?.frontend_dev} target="_blank" rel="noreferrer">{t('frontend_public')}</a>
          <a className="ghost-link" href={snapshotPath} target="_blank" rel="noreferrer">{t('sidebar_snapshot_json')}</a>
        </div>
      </div>

      <PanelCard title={t("sidebar_nav_title")} tone="framework">
        <div className="sidebar-nav">
          {navItems.map((item) => (
            <button
              key={item.id}
              type="button"
              data-testid={`nav-item-${item.id}`}
              aria-label={`${item.label} ${t("legacy_page_suffix")}`}
              className={`sidebar-nav-item ${activeTab === item.id ? 'sidebar-nav-item-active' : ''}`.trim()}
              onClick={() => onTabChange(item.id)}
            >
              <div className="sidebar-nav-copy">
                <strong>{item.label}</strong>
                <div className="stack-item-note">{item.subtitle}</div>
              </div>
              <Badge value={activeTab === item.id ? 'active' : 'neutral'}>{item.eyebrow || item.label}</Badge>
            </button>
          ))}
        </div>
      </PanelCard>

      <PanelCard title={t("sidebar_runtime_title")} tone={meta?.change_class || 'neutral'} meta={t("sidebar_runtime_meta")}>
        <MetricGrid items={buildOverviewCards(meta, uiRoutes, holdHandoff)} compact />
      </PanelCard>

      <PanelCard title={t("sidebar_source_heads_title")} tone={sourceHeads?.[0]?.status || 'neutral'} meta={t("sidebar_source_heads_meta")}>
        <div className="source-head-stack">
          {sourceHeads.map((head) => (
            <div className="source-head-card" key={head.id}>
              <div className="stack-item-top">
                <strong>{head.displayLabel || head.label}</strong>
                <Badge value={head.status}>{head.displayStatus || head.status}</Badge>
              </div>
              <div className="stack-item-note">{head.summary || '—'}</div>
              {head.baseline || head.candidate || head.transferWatch || head.returnCandidate || head.returnWatch ? (
                <div className="source-head-tags">
                  {head.baseline ? <span className="soft-chip">{bilingualLabel('active_baseline')}：{safeDisplayValue(head.baseline)}</span> : null}
                  {head.candidate ? <span className="soft-chip">{bilingualLabel('local_candidate')}：{safeDisplayValue(head.candidate)}</span> : null}
                  {head.transferWatch ? <span className="soft-chip">{bilingualLabel('transfer_watch')}：{safeDisplayValue(head.transferWatch)}</span> : null}
                  {head.returnCandidate ? <span className="soft-chip">{bilingualLabel('return_candidate')}：{safeDisplayValue(head.returnCandidate)}</span> : null}
                  {head.returnWatch ? <span className="soft-chip">{bilingualLabel('return_watch')}：{safeDisplayValue(head.returnWatch)}</span> : null}
                </div>
              ) : null}
            </div>
          ))}
        </div>
      </PanelCard>
    </aside>
  );
}
