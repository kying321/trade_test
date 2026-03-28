import { ActionButton, DomainTab, SegmentedOption } from '../components/control-primitives';
import { ClampText } from '../components/ui-kit';
import type { NavItem } from '../pages/page-types';

export type ThemePreference = 'system' | 'dark' | 'light';
export type ResolvedTheme = 'dark' | 'light';

type GlobalTopbarProps = {
  primaryNav: NavItem[];
  globalSummary: {
    changeClass: string;
    chips?: Array<{ label: string; value: string }>;
  };
  currentPath: string;
  themePreference: ThemePreference;
  resolvedTheme: ResolvedTheme;
  onThemeChange: (next: ThemePreference) => void;
  onOpenSearch?: () => void;
};

const THEME_OPTIONS: Array<{ id: ThemePreference; label: string; ariaLabel: string }> = [
  { id: 'system', label: '跟随系统', ariaLabel: '跟随系统' },
  { id: 'dark', label: '夜间', ariaLabel: '夜间主题' },
  { id: 'light', label: '白天', ariaLabel: '白天主题' },
];

function resolvedThemeLabel(theme: ResolvedTheme): string {
  return theme === 'light' ? '白天' : '夜间';
}

function isPrimaryNavActive(item: NavItem, currentPath: string): boolean {
  if (item.id === 'overview') return currentPath === '/overview' || currentPath === '/';
  if (item.id === 'graph' || item.to.startsWith('/graph-home')) return currentPath.startsWith('/graph-home');
  if (item.id === 'ops' || item.to.startsWith('/terminal/')) return currentPath.startsWith('/terminal/');
  if (item.id === 'research' || item.to.startsWith('/workspace/')) return currentPath.startsWith('/workspace/');
  return currentPath === item.to;
}

export function GlobalTopbar({
  primaryNav,
  globalSummary,
  currentPath,
  themePreference,
  resolvedTheme,
  onThemeChange,
  onOpenSearch,
}: GlobalTopbarProps) {
  return (
    <div className="global-topbar-inner">
      <div className="global-topbar-row global-topbar-main global-topbar-global-only">
        <section className="global-brand" aria-label="global-brand-context">
          <p className="brand-kicker">Fenlie / 控制台</p>
          <strong><ClampText expandable={false} raw="只读操作与研究面">只读操作与研究面</ClampText></strong>
        </section>
        <nav className="primary-nav" aria-label="primary-domains">
          {primaryNav.map((item) => (
            <DomainTab
              key={item.id}
              className="primary-nav-link"
              active={isPrimaryNavActive(item, currentPath)}
              to={item.to}
            >
              <span>{item.label}</span>
            </DomainTab>
          ))}
        </nav>
        <section className="global-control-deck" aria-label="global-tool-strip">
          <div className="theme-switcher" role="group" aria-label="theme-switcher">
            <span className="theme-switcher-label">主题</span>
            <div className="theme-switcher-segment">
              {THEME_OPTIONS.map((option) => (
                <SegmentedOption
                  key={option.id}
                  className={`theme-switcher-button ${themePreference === option.id ? 'active' : ''}`.trim()}
                  aria-label={option.ariaLabel}
                  active={themePreference === option.id}
                  onClick={() => onThemeChange(option.id)}
                >
                  {option.label}
                </SegmentedOption>
              ))}
            </div>
            <span className="theme-switcher-state">当前：{resolvedThemeLabel(resolvedTheme)}</span>
          </div>
          {onOpenSearch ? (
            <ActionButton
              className="chip-button global-search-trigger"
              aria-label="打开全局搜索"
              onClick={onOpenSearch}
            >
              搜索 / ⌘K
            </ActionButton>
          ) : null}
        </section>
        <section className="global-summary-strip" aria-label="global-summary">
          {globalSummary.chips?.map((chip) => (
            <span className="summary-chip" key={`${chip.label}-${chip.value}`}>{chip.label}：{chip.value}</span>
          ))}
          <span className="summary-chip summary-chip-emphasis">变更级别：{globalSummary.changeClass}</span>
        </section>
      </div>
    </div>
  );
}
