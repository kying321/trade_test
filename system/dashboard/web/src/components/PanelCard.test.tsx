import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { PanelCard } from './ui-kit';

describe('PanelCard', () => {
  it('把 kicker、标题、meta 与 actions 拆成明确头部槽位', () => {
    const { container } = render(
      <PanelCard
        kicker="研究地图"
        title="ETHUSDT 15m 主线"
        meta="保留 breakout-pullback，继续验证 exit/risk。"
        actions={<button type="button">切换视图</button>}
      >
        <div>内容区</div>
      </PanelCard>,
    );

    expect(container.querySelector('.panel-card-copy')).toBeTruthy();
    expect(container.querySelector('.panel-card-title')).toBeTruthy();
    expect(container.querySelector('.panel-card-meta')).toBeTruthy();
    expect(container.querySelector('.panel-actions')).toBeTruthy();
    expect(screen.getByText('研究地图')).toBeTruthy();
    expect(screen.getByRole('button', { name: '切换视图' })).toBeTruthy();
  });
});
