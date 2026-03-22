import { describe, expect, it } from 'vitest';
import { formatAgeMinutes, safeDisplayValue } from './formatters';

describe('formatters 中文显示回归', () => {
  it('将复杂数组摘要收口到共享合同', () => {
    expect(safeDisplayValue([{ a: 1 }, { b: 2 }, { c: 3 }])).toBe('[3 项]');
  });

  it('将对象摘要收口到共享合同', () => {
    expect(safeDisplayValue({ alpha: 1, beta: 2 })).toBe('{2 个键}');
  });

  it('将布尔值沿用共享合同', () => {
    expect(safeDisplayValue(true)).toBe('是');
    expect(safeDisplayValue(false)).toBe('否');
  });

  it('将时长摘要收口到中文短单位', () => {
    expect(formatAgeMinutes(null)).toBe('未接线');
    expect(formatAgeMinutes(18)).toBe('18分');
    expect(formatAgeMinutes(125)).toBe('2小时 5分');
  });
});
