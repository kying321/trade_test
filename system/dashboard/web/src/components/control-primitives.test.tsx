import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import {
  ActionButton,
  DomainTab,
  EntityRowButton,
  FilterChip,
  SegmentedOption,
} from './control-primitives';

it('renders domain tabs separately from action buttons and filter chips', () => {
  render(
    <MemoryRouter>
      <DomainTab to="/overview" active>
        总览
      </DomainTab>
      <SegmentedOption active onClick={() => undefined}>
        public
      </SegmentedOption>
      <ActionButton onClick={() => undefined}>打开搜索</ActionButton>
      <FilterChip active onClick={() => undefined}>
        warning
      </FilterChip>
      <EntityRowButton
        title="持有选择主头"
        subtitle="research_hold_transfer"
        onClick={() => undefined}
      />
    </MemoryRouter>
  );

  expect(screen.getByRole('link', { name: '总览' }).className).toContain('control-domain-tab');
  expect(screen.getByRole('button', { name: 'public' }).className).toContain('control-segment-option');
  expect(screen.getByRole('button', { name: '打开搜索' }).className).toContain('control-action-button');
  expect(screen.getByRole('button', { name: 'warning' }).className).toContain('control-filter-chip');
  expect(screen.getByRole('button', { name: /持有选择主头/i }).className).toContain('control-entity-row');
});
