import type { SearchScope } from './types';

export function buildSearchLink(query = '', scope: SearchScope = 'all'): string {
  const params = new URLSearchParams();
  const q = String(query || '').trim();
  if (q) params.set('q', q);
  if (scope && scope !== 'all') params.set('scope', scope);
  const nextQuery = params.toString();
  return `/search${nextQuery ? `?${nextQuery}` : ''}`;
}
