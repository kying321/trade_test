export type PrimaryDomain = 'overview' | 'graph' | 'ops' | 'research' | 'cpa';
export type OpsSurface = 'public' | 'internal';
export type ResearchLegacySection = 'artifacts' | 'alignment' | 'backtests' | 'contracts' | 'raw';

export type NavItem = {
  id: string;
  label: string;
  to: string;
  description?: string;
};
