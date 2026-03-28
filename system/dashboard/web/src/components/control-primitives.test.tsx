import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { expect, it } from 'vitest';
import {
  ActionLink,
  ActionButton,
  DomainTab,
  EntityRowButton,
  FilterChip,
  SegmentedOption,
} from './control-primitives';

it('renders domain tabs separately from action buttons and filter chips', () => {
  render(
    <MemoryRouter initialEntries={['/outside']}>
      <DomainTab to="/overview" active>
        总览
      </DomainTab>
      <SegmentedOption active onClick={() => undefined}>
        public
      </SegmentedOption>
      <ActionLink to="/search">查看详情</ActionLink>
      <ActionButton onClick={() => undefined}>打开搜索</ActionButton>
      <FilterChip active onClick={() => undefined}>
        warning
      </FilterChip>
      <EntityRowButton
        title="持有选择主头"
        subtitle="research_hold_transfer"
        active
        onClick={() => undefined}
      />
    </MemoryRouter>
  );

  const domainTab = screen.getByRole('tab', { name: '总览' });
  const segmentedOption = screen.getByRole('button', { name: 'public' });
  const actionLink = screen.getByRole('link', { name: '查看详情' });
  const actionButton = screen.getByRole('button', { name: '打开搜索' });
  const filterChip = screen.getByRole('button', { name: 'warning' });
  const entityRow = screen.getByRole('button', { name: /持有选择主头/i });

  expect(domainTab.className).toContain('control-domain-tab');
  expect(domainTab.className).toContain('active');
  expect(segmentedOption.className).toContain('control-segment-option');
  expect(segmentedOption.getAttribute('aria-pressed')).toBe('true');
  expect(actionLink.className).toContain('control-action-link');
  expect(actionButton.className).toContain('control-action-button');
  expect(filterChip.className).toContain('control-filter-chip');
  expect(filterChip.getAttribute('aria-pressed')).toBe('true');
  expect(entityRow.className).toContain('control-entity-row');
  expect(entityRow.getAttribute('aria-pressed')).toBe('true');
});

it('defaults control button primitives to type=button', () => {
  render(
    <>
      <SegmentedOption>public</SegmentedOption>
      <ActionButton>打开搜索</ActionButton>
      <FilterChip>warning</FilterChip>
    </>
  );

  expect(screen.getByRole('button', { name: 'public' }).getAttribute('type')).toBe('button');
  expect(screen.getByRole('button', { name: '打开搜索' }).getAttribute('type')).toBe('button');
  expect(screen.getByRole('button', { name: 'warning' }).getAttribute('type')).toBe('button');
});

it('uses false semantic active state when not active', () => {
  render(
    <>
      <SegmentedOption>public</SegmentedOption>
      <FilterChip>warning</FilterChip>
      <EntityRowButton title="持有选择主头" onClick={() => undefined} />
    </>
  );

  expect(screen.getByRole('button', { name: 'public' }).getAttribute('aria-pressed')).toBe('false');
  expect(screen.getByRole('button', { name: 'warning' }).getAttribute('aria-pressed')).toBe('false');
  expect(screen.getByRole('button', { name: '持有选择主头' }).getAttribute('aria-pressed')).toBe('false');
});

it('merges route-active state with callback className for nav primitives', () => {
  render(
    <MemoryRouter initialEntries={['/search']}>
      <ActionLink to="/search" className={({ isActive }) => (isActive ? 'route-active-extra' : 'route-idle-extra')}>
        查看详情
      </ActionLink>
    </MemoryRouter>,
  );

  const actionLink = screen.getByRole('link', { name: '查看详情' });
  expect(actionLink.className).toContain('control-action-link');
  expect(actionLink.className).toContain('active');
  expect(actionLink.className).toContain('route-active-extra');
});

it('exposes selected tab semantics on the normal route-active path', () => {
  render(
    <MemoryRouter initialEntries={['/overview']}>
      <DomainTab to="/overview">总览</DomainTab>
    </MemoryRouter>,
  );

  const domainTab = screen.getByRole('tab', { name: '总览' });
  expect(domainTab.className).toContain('control-domain-tab');
  expect(domainTab.className).toContain('active');
  expect(domainTab.getAttribute('aria-selected')).toBe('true');
  expect(domainTab.getAttribute('aria-current')).toBe('page');
});

it('normalizes forced active state for DomainTab callback classes and semantics', () => {
  render(
    <MemoryRouter initialEntries={['/outside']}>
      <DomainTab
        to="/overview"
        active
        className={({ isActive }) => (isActive ? 'forced-active-extra' : 'forced-idle-extra')}
      >
        总览
      </DomainTab>
    </MemoryRouter>,
  );

  const domainTab = screen.getByRole('tab', { name: '总览' });
  expect(domainTab.className).toContain('control-domain-tab');
  expect(domainTab.className).toContain('active');
  expect(domainTab.className).toContain('forced-active-extra');
  expect(domainTab.className).not.toContain('forced-idle-extra');
  expect(domainTab.getAttribute('role')).toBe('tab');
  expect(domainTab.getAttribute('aria-selected')).toBe('true');
});

it('passes through standard button props on EntityRowButton', () => {
  render(
    <EntityRowButton
      title="持有选择主头"
      subtitle="research_hold_transfer"
      className="custom-row"
      data-testid="entity-row"
      disabled
      aria-label="自定义实体行"
      onClick={() => undefined}
    >
      <span>附加信息</span>
    </EntityRowButton>,
  );

  const row = screen.getByTestId('entity-row');
  expect(row.className).toContain('control-entity-row');
  expect(row.className).toContain('custom-row');
  expect(row.getAttribute('disabled')).not.toBeNull();
  expect(screen.getByText('持有选择主头').className).toContain('control-entity-row-title');
  expect(screen.getByText('research_hold_transfer').className).toContain('control-entity-row-subtitle');
  expect(screen.getByText('附加信息').className).not.toContain('control-entity-row-title');
  expect(screen.getByText('附加信息')).toBeTruthy();
});
