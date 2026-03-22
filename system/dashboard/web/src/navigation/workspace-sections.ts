import type { TerminalReadModel } from '../types/contracts';
import type { WorkspaceSection } from '../utils/focus-links';
import { safeDisplayValue } from '../utils/formatters';

export type WorkspacePageSectionItem = {
  id: string;
  label: string;
  level: 1 | 2;
  groupId?: string;
  groupLabel?: string;
  accordionGroup?: string;
  accordionItemId?: string;
};

function buildBacktestsSections(): WorkspacePageSectionItem[] {
  return [
    { id: 'backtests-overview', label: '回测总览', level: 1 },
    { id: 'backtests-pool', label: '回测池', level: 1 },
    { id: 'backtests-comparison', label: '近期比较行', level: 1 },
  ];
}

function buildAlignmentSections(model: TerminalReadModel): WorkspacePageSectionItem[] {
  const domainGroups = model.feedback.domainGroups;
  return [
    { id: 'alignment-summary', label: '当前总览', level: 1, groupId: 'contracts-data', groupLabel: '对齐投射' },
    { id: 'alignment-trends', label: '趋势投射', level: 1, groupId: 'contracts-data', groupLabel: '对齐投射' },
    { id: 'alignment-events', label: '反馈事件', level: 1, groupId: 'contracts-data', groupLabel: '对齐投射' },
    ...domainGroups.slice(0, 4).map((group) => ({
      id: `alignment-domain-${safeDisplayValue(group.domain)}`,
      label: safeDisplayValue(group.domain || 'global'),
      level: 2 as const,
      groupId: 'contracts-data' as const,
      groupLabel: '对齐投射',
    })),
    { id: 'alignment-actions', label: '建议动作', level: 1, groupId: 'contracts-data', groupLabel: '对齐投射' },
  ];
}

function buildContractsSections(model: TerminalReadModel): WorkspacePageSectionItem[] {
  return [
    { id: 'contracts-topology', label: '公开入口拓扑', level: 1, groupId: 'contracts-acceptance', groupLabel: '验收链' },
    { id: 'contracts-acceptance-summary', label: '验收总览', level: 1, groupId: 'contracts-acceptance', groupLabel: '验收链' },
    { id: 'contracts-acceptance-checks', label: '子链状态', level: 1, groupId: 'contracts-acceptance', groupLabel: '验收链' },
    ...model.workspace.publicAcceptance.checks.map((row) => ({
      id: `contracts-check-${safeDisplayValue(row.id)}`,
      label: safeDisplayValue(row.label || row.id),
      level: 2 as const,
      groupId: 'contracts-acceptance' as const,
      groupLabel: '验收链',
      accordionGroup: 'contracts-acceptance-checks' as const,
      accordionItemId: safeDisplayValue(row.id),
    })),
    { id: 'contracts-acceptance-subcommands', label: '子命令证据', level: 1, groupId: 'contracts-subcommands', groupLabel: '子命令' },
    ...model.workspace.publicAcceptance.subcommands.map((row) => ({
      id: `contracts-subcommand-${safeDisplayValue(row.id)}`,
      label: safeDisplayValue(row.label || row.id),
      level: 2 as const,
      groupId: 'contracts-subcommands' as const,
      groupLabel: '子命令',
      accordionGroup: 'contracts-acceptance-subcommands' as const,
      accordionItemId: safeDisplayValue(row.id),
    })),
    { id: 'contracts-surfaces', label: '数据面断言', level: 1, groupId: 'contracts-data', groupLabel: '数据面' },
    { id: 'contracts-interface', label: '接口目录', level: 1, groupId: 'contracts-data', groupLabel: '数据面' },
    { id: 'contracts-source-heads', label: '主线总览', level: 1, groupId: 'contracts-source-heads', groupLabel: '源头主线' },
    ...model.workspace.sourceHeads.slice(0, 3).map((row) => ({
      id: `contracts-source-head-${safeDisplayValue(row.id)}`,
      label: safeDisplayValue(row.label || row.id),
      level: 2 as const,
      groupId: 'contracts-source-heads' as const,
      groupLabel: '源头主线',
      accordionGroup: 'contracts-source-heads' as const,
      accordionItemId: safeDisplayValue(row.id),
    })),
    { id: 'contracts-fallback', label: '回退链', level: 1, groupId: 'contracts-data', groupLabel: '数据面' },
  ];
}

function buildRawSections(model: TerminalReadModel): WorkspacePageSectionItem[] {
  const focusLabel = safeDisplayValue(model.workspace.artifactRows[0]?.payload_key || model.workspace.artifactRows[0]?.label || model.workspace.artifactRows[0]?.id || '焦点工件原始层');
  return [
    { id: 'raw-summary', label: '原始摘要', level: 1 },
    { id: 'raw-focus', label: focusLabel || '焦点工件原始层', level: 1 },
    { id: 'raw-snapshot', label: '全量原始快照', level: 1 },
  ];
}

export function getWorkspacePageSections(section: WorkspaceSection, model: TerminalReadModel): WorkspacePageSectionItem[] {
  if (section === 'alignment') return buildAlignmentSections(model);
  if (section === 'backtests') return buildBacktestsSections();
  if (section === 'contracts') return buildContractsSections(model);
  if (section === 'raw') return buildRawSections(model);
  return [];
}

export function findWorkspacePageSection(
  section: WorkspaceSection,
  model: TerminalReadModel,
  sectionId: string | undefined,
): WorkspacePageSectionItem | null {
  if (!sectionId) return null;
  return getWorkspacePageSections(section, model).find((item) => item.id === sectionId) || null;
}
