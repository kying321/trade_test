import { cardList } from './formatters';
import { bilingualLabel, zhChangeClass, zhLabel } from './i18n';
const t = (key) => zhLabel(key);

function decorateArtifactRows(rows) {
  return rows.map((row) => ({
    ...row,
    labelDisplay: bilingualLabel(row.label),
    categoryDisplay: zhLabel(row.category || 'uncategorized'),
    statusDisplay: bilingualLabel(row.status || ''),
    changeClassDisplay: zhChangeClass(row.changeClass || ''),
    researchDecisionDisplay: bilingualLabel(row.researchDecision || ''),
  }));
}

function decorateDomainRows(rows) {
  return rows.map((row) => ({
    ...row,
    labelDisplay: zhLabel(row.label || row.id),
  }));
}

function decorateInterfaceCards(rows) {
  return rows.map((row) => ({
    ...row,
    displayLabel: zhLabel(row.label || row.id),
    displayKind: zhLabel(row.kind),
  }));
}

function decorateSourceHeadCards(rows) {
  return rows.map((row) => ({
    ...row,
    displayLabel: bilingualLabel(row.label || row.id),
    displayStatus: bilingualLabel(row.status || ''),
  }));
}

export function buildDisplayModel(model) {
  return {
    ...model,
    interfaces: decorateInterfaceCards(model.interfaces || []),
    sourceHeads: decorateSourceHeadCards(model.sourceHeads || []),
    artifactRows: decorateArtifactRows(model.artifactRows || []),
    domains: decorateDomainRows(model.domains || []),
  };
}

export function buildOverviewCards(meta, uiRoutes, holdHandoff) {
  return cardList([
    { label: t('overview_card_local_preview'), value: uiRoutes?.frontend_dev, hint: t('overview_card_local_preview_hint') },
    { label: t('overview_card_public_entry'), value: uiRoutes?.frontend_public || uiRoutes?.frontend_pages || '—', hint: t('overview_card_public_entry_hint') },
    { label: t('freshness_snapshot_generated'), value: meta?.generated_at_utc },
    { label: t('artifact_payload_count'), value: meta?.artifact_payload_count },
    { label: t('catalog_count'), value: meta?.catalog_count },
    { label: t('backtest_artifact_count'), value: meta?.backtest_artifact_count },
    { label: t('active_baseline'), value: holdHandoff?.active_baseline || '—' },
    { label: t('local_candidate'), value: holdHandoff?.local_candidate || '—' },
  ]);
}
