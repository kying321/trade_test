import displayPolicyContract from '../contracts/dashboard-display-policy-contract.json';
import type { SurfaceKey } from '../types/contracts';

type SurfacePolicy = {
  metricStrips?: string[];
  tables?: Record<string, string[]>;
  drilldownFields?: Record<string, string[]>;
  keyValueFields?: Record<string, string[]>;
  compactPathFields?: Record<string, string[]>;
  valueTextFields?: Record<string, string[]>;
};

type DisplayFieldIntent = Record<string, Record<string, boolean>>;

type DisplayIntent = {
  metricStrips: Record<string, boolean>;
  tables: DisplayFieldIntent;
  drilldownFields: DisplayFieldIntent;
  keyValueFields: DisplayFieldIntent;
  compactPathFields: DisplayFieldIntent;
  valueTextFields: DisplayFieldIntent;
};

const POLICY = displayPolicyContract.surfaces as Record<SurfaceKey, SurfacePolicy>;

function includes(list: string[] | undefined, value: string): boolean {
  return Array.isArray(list) && (list.includes(value) || list.includes('*'));
}

function includesInMap(map: Record<string, string[]> | undefined, section: string, key: string): boolean {
  return includes(map?.[section], key);
}

function toBooleanMap(list: string[] | undefined): Record<string, boolean> {
  return Object.fromEntries((list || []).map((item) => [item, true]));
}

function toBooleanFieldIntent(map: Record<string, string[]> | undefined): DisplayFieldIntent {
  return Object.fromEntries(
    Object.entries(map || {}).map(([section, keys]) => [section, toBooleanMap(keys)]),
  );
}

export function resolveDisplayIntent(surface: SurfaceKey): DisplayIntent {
  const policy = POLICY[surface] || {};
  return {
    metricStrips: toBooleanMap(policy.metricStrips),
    tables: toBooleanFieldIntent(policy.tables),
    drilldownFields: toBooleanFieldIntent(policy.drilldownFields),
    keyValueFields: toBooleanFieldIntent(policy.keyValueFields),
    compactPathFields: toBooleanFieldIntent(policy.compactPathFields),
    valueTextFields: toBooleanFieldIntent(policy.valueTextFields),
  };
}

export function shouldShowRawMetricValues(surface: SurfaceKey, section: string): boolean {
  return includes(POLICY[surface]?.metricStrips, section);
}

export function shouldShowRawTableColumn(surface: SurfaceKey, section: string, key: string): boolean {
  return includesInMap(POLICY[surface]?.tables, section, key);
}

export function shouldShowRawDrilldownField(surface: SurfaceKey, section: string, key: string): boolean {
  return includesInMap(POLICY[surface]?.drilldownFields, section, key);
}

export function shouldShowRawKeyValueField(surface: SurfaceKey, section: string, key: string): boolean {
  return includesInMap(POLICY[surface]?.keyValueFields, section, key);
}

export function shouldShowRawCompactPath(surface: SurfaceKey, section: string, key: string): boolean {
  return includesInMap(POLICY[surface]?.compactPathFields, section, key);
}

export function shouldShowRawValueText(surface: SurfaceKey, section: string, key: string): boolean {
  return includesInMap(POLICY[surface]?.valueTextFields, section, key);
}
