import { describe, expect, it } from 'vitest';
import { displayText, explainLabel } from './dictionary';

describe('dictionary 中文显示回归', () => {
  it('将清障摘要转成中文主显示', () => {
    expect(displayText('clearing_required:ops_live_gate+risk_guard')).toBe('需要清障 / 运维实盘门禁 + 风控守护');
  });

  it('将泳道优先序摘要转成中文排序说明', () => {
    expect(displayText('repair@1036:11 > waiting@197:2 > review@154:2 > watch@38:1 > blocked@0:0')).toBe(
      '修复 1036 分 / 11 条 ＞ 等待 197 分 / 2 条 ＞ 评审 154 分 / 2 条 ＞ 观察 38 分 / 1 条 ＞ 阻断 0 分 / 0 条',
    );
  });

  it('将英文完成条件转成中文主显示，并保留 raw 副标题用于审计', () => {
    const rendered = explainLabel('XAGUSD gains paper evidence and leaves fill_evidence_pending_symbols');
    expect(rendered.primary).toBe('XAGUSD 补齐纸面证据并移出成交证据待补标的清单');
    expect(rendered.secondary).toBe('XAGUSD gains paper evidence and leaves fill_evidence_pending_symbols');
  });

  it('将长英文研究结论转成中文主显示', () => {
    const rendered = explainLabel(
      'Recent measured evidence favors Brooks adaptive routing over current crypto ETF/native combo families; keep negative-edge crypto combos out of promotion, keep native majors-long as the only positive crypto subsegment, and treat Brooks execution heads as manual structure review until an execution bridge exists.',
    );
    expect(rendered.primary).toContain('最新实测证据更支持 Brooks 自适应路由');
    expect(rendered.primary).toContain('仅保留原生主流长持作为唯一正向子段');
    expect(rendered.secondary).toBeUndefined();
  });

  it('将 contracts 页面残余术语转成中文主显示', () => {
    expect(displayText('持有选择 Gate')).toBe('持有选择门禁');
    expect(displayText('本地 Vite 预览入口')).toBe('本地预览入口');
    expect(displayText('Cloudflare Pages 公开子域')).toBe('Cloudflare Pages 公开域名');
    expect(displayText('只读 operator 监控面板')).toBe('只读运维监控面板');
  });

  it('将外层壳层 header 与导航术语收口到共享合同', () => {
    expect(displayText('shell_brand_kicker')).toBe('Fenlie / 操作台');
    expect(displayText('nav_terminal_public')).toBe('公开面');
    expect(displayText('nav_terminal_internal')).toBe('内部面');
    expect(displayText('nav_workspace_artifacts')).toBe('工件池');
    expect(displayText('nav_workspace_backtests')).toBe('回测池');
    expect(displayText('nav_workspace_contracts')).toBe('契约层');
    expect(displayText('nav_workspace_raw')).toBe('原始层');
    expect(displayText('surface_watermark_public')).toBe('公开 / 公开摘要 / 安全定位');
    expect(displayText('fallback_shell_title')).toBe('前端渲染失败，已切到回退框架');
  });

  it('将 terminal/workspace panel 文案收口到共享合同', () => {
    expect(displayText('terminal_panel_orchestration_kicker')).toBe('面板 1 / 总线调度与全局门禁');
    expect(displayText('terminal_section_focus_slots_title')).toBe('穿透层 3 / 焦点槽位');
    expect(displayText('workspace_artifacts_controls_title')).toBe('穿透控件');
    expect(displayText('workspace_contracts_topology_kicker')).toBe('入口摘要 / 发布拓扑');
    expect(displayText('workspace_contracts_topology_node_prefix')).toBe('节点');
    expect(displayText('workspace_contracts_topology_tier_primary')).toBe('主入口层');
    expect(displayText('workspace_contracts_topology_tier_fallback')).toBe('回退入口层');
    expect(displayText('workspace_contracts_topology_tier_probe')).toBe('探针层');
    expect(displayText('workspace_contracts_topology_section_primary')).toBe('主入口链');
    expect(displayText('workspace_contracts_topology_section_fallback')).toBe('回退链');
    expect(displayText('workspace_contracts_topology_section_probe')).toBe('探针区');
    expect(displayText('workspace_contracts_topology_source_label')).toBe('控制源');
    expect(displayText('workspace_contracts_topology_chain_label')).toBe('退化至');
    expect(displayText('workspace_contracts_surface_kicker_public')).toBe('公开摘要 / 对外发布');
    expect(displayText('workspace_contracts_surface_kicker_internal')).toBe('内部摘要 / 本地剥离');
    expect(displayText('workspace_contracts_assertion_title_public')).toBe('穿透层 1 / 公开断言');
    expect(displayText('workspace_contracts_assertion_title_internal')).toBe('穿透层 1 / 本地断言');
    expect(displayText('workspace_contracts_publish_title_public')).toBe('穿透层 2 / 发布入口与默认回退');
    expect(displayText('workspace_contracts_publish_title_internal')).toBe('穿透层 2 / 剥离策略与私有回退');
    expect(displayText('workspace_contracts_acceptance_kicker')).toBe('验收摘要 / 公开回归');
    expect(displayText('workspace_contracts_interface_kicker')).toBe('接口分层 / 可见边界');
    expect(displayText('workspace_contracts_source_heads_kicker')).toBe('源头摘要 / 主线对照');
    expect(displayText('workspace_contracts_fallback_kicker')).toBe('回退摘要 / 失败退化');
    expect(displayText('workspace_contracts_fallback_meta')).toBe('当前 contracts 页的最终退化链，不参与研究判断。');
  });

  it('将表格列名与动态摘要键收口到共享合同', () => {
    expect(displayText('slot_trigger')).toBe('槽位/触发');
    expect(displayText('entry_route')).toBe('入口');
    expect(displayText('workspace_summary_payload_fields_suffix')).toBe('个负载字段');
    expect(displayText('terminal_summary_backlog_suffix')).toBe('个优化项');
  });
});
