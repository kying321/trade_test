import { describe, expect, it } from 'vitest';
import {
  bilingualLabel,
  translateFieldLabel,
  translateSourceHeadLabel,
  zhChangeClass,
  zhLabel,
  zhPath,
} from './i18n';

describe('legacy i18n 共享合同回归', () => {
  it('将 legacy 字段与状态显示收敛到共享合同', () => {
    expect(translateFieldLabel('change_class')).toBe('变更等级');
    expect(zhChangeClass('RESEARCH_ONLY')).toBe('仅研究');
    expect(zhLabel('summary-only')).toBe('仅摘要');
  });

  it('优先使用 source-owned 标签并在 source head 缺省时回落到 fallback', () => {
    expect(translateSourceHeadLabel('price_action_breakout_pullback')).toBe('ETH 15m 主线');
    expect(translateSourceHeadLabel('unknown_head', 'hold_selection_gate')).toBe('持有选择门禁');
  });

  it('将 legacy 页面路径/入口术语中文主显', () => {
    expect(bilingualLabel('frontend_pages')).toBe('Pages 地址');
    expect(zhPath(['public', 'frontend_public'])).toBe('公开 / 公开地址');
  });
});
