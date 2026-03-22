import { statusTone } from './formatters';

export function pickDefaultArtifact(rows, domain) {
  const candidates = rows.filter((row) => domain === 'all' || row.category === domain);
  return candidates[0]?.id || '';
}

export function searchArtifacts(rows, domain, query, tone = 'all') {
  const keyword = query.trim().toLowerCase();
  return rows.filter((row) => {
    if (domain !== 'all' && row.category !== domain) return false;
    if (tone !== 'all' && statusTone(row.status || row.researchDecision) !== tone) return false;
    if (!keyword) return true;
    return JSON.stringify({
      label: row.label,
      labelDisplay: row.labelDisplay,
      category: row.category,
      categoryDisplay: row.categoryDisplay,
      status: row.status,
      researchDecision: row.researchDecision,
      path: row.path,
      topLevelKeys: row.topLevelKeys,
      action: row.action,
    }).toLowerCase().includes(keyword);
  });
}
