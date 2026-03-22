export type PrimaryDomain = 'overview' | 'ops' | 'research';
export type OpsSurface = 'public' | 'internal';
export type ResearchLegacySection = 'artifacts' | 'alignment' | 'backtests' | 'contracts' | 'raw';

export type NavItem = {
  id: string;
  label: string;
  to: string;
  description?: string;
};
