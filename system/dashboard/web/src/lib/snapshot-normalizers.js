import { statusTone } from './formatters';
import { labelFor } from '../utils/dictionary';

function normalizeArtifactRows(snapshot) {
  const payloads = snapshot?.artifact_payloads || {};
  const catalog = snapshot?.catalog || [];
  const rows = [];
  const seenPayloadKeys = new Set();

  catalog.forEach((item) => {
    const payloadEntry = item.payload_key ? payloads[item.payload_key] : null;
    if (payloadEntry) seenPayloadKeys.add(item.payload_key);
    rows.push({
      id: item.id || item.payload_key || item.path,
      payloadKey: item.payload_key || null,
      label: payloadEntry?.label || item.label || item.payload_key || item.id,
      category: item.category || payloadEntry?.category || 'uncategorized',
      status: item.status || payloadEntry?.summary?.status || '',
      changeClass: item.change_class || payloadEntry?.summary?.change_class || '',
      path: item.path || payloadEntry?.path || '',
      researchDecision: item.research_decision || payloadEntry?.summary?.research_decision || '',
      takeaway: item.takeaway || item.recommended_brief || payloadEntry?.summary?.recommended_brief || '',
      generatedAt: item.generated_at_utc || payloadEntry?.summary?.generated_at_utc || '',
      topLevelKeys: item.top_level_keys || payloadEntry?.summary?.top_level_keys || [],
      action: item.action || payloadEntry?.summary?.action || '',
      hasPayload: Boolean(payloadEntry),
      payload: payloadEntry?.payload || null,
      summaryPayload: payloadEntry?.summary || item,
    });
  });

  Object.entries(payloads).forEach(([key, entry]) => {
    if (seenPayloadKeys.has(key)) return;
    rows.push({
      id: key,
      payloadKey: key,
      label: entry.label || key,
      category: entry.category || 'uncategorized',
      status: entry.summary?.status || '',
      changeClass: entry.summary?.change_class || '',
      path: entry.path || '',
      researchDecision: entry.summary?.research_decision || '',
      takeaway: entry.summary?.recommended_brief || '',
      generatedAt: entry.summary?.generated_at_utc || '',
      topLevelKeys: entry.summary?.top_level_keys || [],
      action: entry.summary?.action || '',
      hasPayload: true,
      payload: entry.payload,
      summaryPayload: entry.summary,
    });
  });

  return rows;
}

function normalizeDomainRows(snapshot, artifactRows) {
  const sourceRows = Array.isArray(snapshot?.domain_summary) ? snapshot.domain_summary : null;
  if (sourceRows?.length) {
    return [
      {
        id: 'all',
        label: 'all',
        count: artifactRows.length,
        payloadCount: artifactRows.filter((row) => row.hasPayload).length,
        positive: artifactRows.filter((row) => statusTone(row.status || row.researchDecision) === 'positive').length,
        warning: artifactRows.filter((row) => statusTone(row.status || row.researchDecision) === 'warning').length,
        negative: artifactRows.filter((row) => statusTone(row.status || row.researchDecision) === 'negative').length,
        headline: labelFor('all_non_sensitive_domains_headline'),
        sourceOwned: true,
      },
      ...sourceRows.map((row) => ({
        id: row.id,
        label: row.label || row.id,
        count: row.count || 0,
        payloadCount: row.payload_count || row.payloadCount || 0,
        positive: row.positive || 0,
        warning: row.warning || 0,
        negative: row.negative || 0,
        headline: row.headline || row.research_decision || row.takeaway || '',
        sourceOwned: true,
      })),
    ];
  }

  const grouped = new Map();
  artifactRows.forEach((row) => {
    const key = row.category || 'uncategorized';
    if (!grouped.has(key)) {
      grouped.set(key, { id: key, label: key, count: 0, payloadCount: 0, positive: 0, warning: 0, negative: 0, headline: '' });
    }
    const item = grouped.get(key);
    item.count += 1;
    item.payloadCount += row.hasPayload ? 1 : 0;
    item[statusTone(row.status || row.researchDecision)] += 1;
    if (!item.headline && (row.researchDecision || row.takeaway)) item.headline = row.researchDecision || row.takeaway;
  });

  return [
    {
      id: 'all',
      label: 'all',
      count: artifactRows.length,
      payloadCount: artifactRows.filter((row) => row.hasPayload).length,
      positive: 0,
      warning: 0,
      negative: 0,
      headline: labelFor('all_non_sensitive_domains_headline'),
      sourceOwned: false,
    },
    ...Array.from(grouped.values()).sort((a, b) => b.count - a.count || a.label.localeCompare(b.label)),
  ];
}

function normalizeInterfaceCards(snapshot) {
  const interfaces = Array.isArray(snapshot?.interface_catalog) ? snapshot.interface_catalog : [];
  if (interfaces.length) {
    return interfaces.map((item) => ({
      id: item.id,
      label: item.label || item.id,
      kind: item.kind,
      url: item.url,
      description: item.description,
    }));
  }

  const uiRoutes = snapshot?.ui_routes || {};
  return Object.entries(uiRoutes).map(([id, url]) => ({
    id,
    label: id,
    kind: id.includes('public') ? 'public' : 'local',
    url,
    description: id,
  }));
}

function normalizeSourceHeadCards(snapshot) {
  const sourceHeads = snapshot?.source_heads;
  if (!sourceHeads || typeof sourceHeads !== 'object') return [];
  return Object.entries(sourceHeads).map(([id, value]) => ({
    id,
    label: value.label || id,
    status: value.status || value.research_decision || '',
    summary: value.summary || value.recommended_brief || value.takeaway || '',
    baseline: value.active_baseline,
    candidate: value.local_candidate,
    transferWatch: value.transfer_watch,
    returnCandidate: value.return_candidate,
    returnWatch: value.return_watch,
    path: value.path || '',
  }));
}

export function normalizeSnapshot(snapshot) {
  const artifactRows = normalizeArtifactRows(snapshot);
  return {
    meta: snapshot?.meta || {},
    uiRoutes: snapshot?.ui_routes || {},
    interfaces: normalizeInterfaceCards(snapshot),
    sourceHeads: normalizeSourceHeadCards(snapshot),
    artifactRows,
    domains: normalizeDomainRows(snapshot, artifactRows),
    backtests: snapshot?.backtest_artifacts || [],
    comparisonReport: snapshot?.artifact_payloads?.recent_strategy_backtests?.payload || {},
    holdHandoff: snapshot?.artifact_payloads?.hold_selection_handoff?.payload || {},
    holdGate: snapshot?.artifact_payloads?.hold_selection_gate?.payload || {},
    operatorPanel: snapshot?.artifact_payloads?.operator_panel?.payload || {},
    baseResearch: snapshot?.artifact_payloads?.price_action_breakout_pullback?.payload || {},
  };
}
