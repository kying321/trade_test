import { describe, expect, it } from 'vitest';
import { buildOverviewCards } from './display-model';

describe('buildOverviewCards', () => {
  it('uses shared contract labels and hints for legacy overview cards', () => {
    const cards = buildOverviewCards(
      {
        generated_at_utc: '2026-03-20T00:00:00Z',
        artifact_payload_count: 20,
        catalog_count: 80,
        backtest_artifact_count: 10,
      },
      {
        frontend_dev: 'http://127.0.0.1:5173',
        frontend_public: 'https://fuuu.fun',
      },
      {
        active_baseline: 'hold16',
        local_candidate: 'hold8',
      },
    );

    expect(cards[0]).toMatchObject({
      label: '本地预览',
      hint: '开发预览入口',
    });
    expect(cards[1]).toMatchObject({
      label: '公开地址',
      hint: '公开访问面',
    });
    expect(cards[2].label).toBe('快照生成');
    expect(cards[6].label).toBe('当前基线');
    expect(cards[7].label).toBe('局部候选');
  });
});
