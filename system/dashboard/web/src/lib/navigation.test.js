import { describe, expect, it } from 'vitest';
import { MAIN_TABS } from './navigation';

describe('legacy navigation contract', () => {
  it('reads legacy tab labels and titles from the shared term contract', () => {
    expect(MAIN_TABS[0]).toMatchObject({
      id: 'overview',
      label: '总览',
      title: '研究总览',
    });
    expect(MAIN_TABS[4]).toMatchObject({
      id: 'raw',
      label: '原始数据',
      eyebrow: '快照',
    });
  });
});
