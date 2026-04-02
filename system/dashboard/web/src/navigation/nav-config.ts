import type { NavItem, ResearchLegacySection } from '../pages/page-types';
import type { OpsPageId } from './route-contract';

export const PRIMARY_NAV: NavItem[] = [
  { id: 'graph', label: '图谱主页', to: '/graph-home', description: '交易中枢 / 自定义管道 / 关系图谱' },
  { id: 'overview', label: '总览', to: '/ops/overview', description: '90 秒总览 / 分流入口' },
  { id: 'ops', label: '操作终端', to: '/ops/risk', description: '风险驾驶舱 / 诊断入口' },
  { id: 'research', label: '研究工作区', to: '/workspace/artifacts', description: '工件 / 回测 / 对比' },
  { id: 'cpa', label: 'CPA 管理', to: '/cpa', description: '认证文件 / 配额 / 删除保护' },
];

export const OPS_SURFACE_NAV: Array<NavItem & { page: OpsPageId }> = [
  { id: 'ops-overview', page: 'overview', label: '90 秒总览', to: '/ops/overview', description: '状态 / 分流入口' },
  { id: 'ops-risk', page: 'risk', label: '风险驾驶舱', to: '/ops/risk', description: '观察 / 实时风险' },
  { id: 'ops-audits', page: 'audits', label: '审计诊断', to: '/ops/audits', description: '校准 / 趋势 / 验收' },
  { id: 'ops-runbooks', page: 'runbooks', label: '处置手册', to: '/ops/runbooks', description: '动作 / precheck / playbook' },
  { id: 'ops-workflow', page: 'workflow', label: '工作流', to: '/ops/workflow', description: 'proposal / approval / rollback' },
  { id: 'ops-traces', page: 'traces', label: '链路追踪', to: '/ops/traces', description: 'trace / event / artifact' },
];

export const RESEARCH_LEGACY_NAV: Array<NavItem & { section: ResearchLegacySection }> = [
  { id: 'workspace-artifacts', section: 'artifacts', label: '工件池', to: '/workspace/artifacts', description: '工件目标池' },
  { id: 'workspace-alignment', section: 'alignment', label: '对齐页', to: '/workspace/alignment', description: '方向对齐投射' },
  { id: 'workspace-backtests', section: 'backtests', label: '回测池', to: '/workspace/backtests', description: '回测主池' },
  { id: 'workspace-contracts', section: 'contracts', label: '契约层', to: '/workspace/contracts', description: '公开入口拓扑' },
  { id: 'workspace-raw', section: 'raw', label: '原始层', to: '/workspace/raw', description: '原始快照' },
];
