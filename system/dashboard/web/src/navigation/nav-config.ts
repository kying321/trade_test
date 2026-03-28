import type { NavItem, ResearchLegacySection } from '../pages/page-types';

export const PRIMARY_NAV: NavItem[] = [
  { id: 'overview', label: '总览', to: '/overview', description: '总状态 / 分流入口' },
  { id: 'graph', label: '图谱主页', to: '/graph-home', description: '交易中枢 / 自定义管道 / 关系图谱' },
  { id: 'ops', label: '操作终端', to: '/terminal/public', description: '调度 / 门禁 / 风险链路' },
  { id: 'research', label: '研究工作区', to: '/workspace/artifacts', description: '工件 / 回测 / 对比' },
];

export const OPS_SURFACE_NAV: NavItem[] = [
  { id: 'terminal-public', label: '公开面', to: '/terminal/public', description: '安全摘要' },
  { id: 'terminal-internal', label: '内部面', to: '/terminal/internal', description: '完整穿透' },
];

export const RESEARCH_LEGACY_NAV: Array<NavItem & { section: ResearchLegacySection }> = [
  { id: 'workspace-artifacts', section: 'artifacts', label: '工件池', to: '/workspace/artifacts', description: '工件目标池' },
  { id: 'workspace-alignment', section: 'alignment', label: '对齐页', to: '/workspace/alignment', description: '方向对齐投射' },
  { id: 'workspace-backtests', section: 'backtests', label: '回测池', to: '/workspace/backtests', description: '回测主池' },
  { id: 'workspace-contracts', section: 'contracts', label: '契约层', to: '/workspace/contracts', description: '公开入口拓扑' },
  { id: 'workspace-raw', section: 'raw', label: '原始层', to: '/workspace/raw', description: '原始快照' },
];
