import { NavLink } from 'react-router-dom';
import { ClampText } from '../components/ui-kit';
import type { NavItem } from '../pages/page-types';

export type ThemePreference = 'system' | 'dark' | 'light';
export type ResolvedTheme = 'dark' | 'light';
export type TopbarDomainTone = 'overview' | 'ops' | 'research';

export type GlobalTopbarDomainContext = {
  kicker: string;
  title: string;
  detail: string;
  tone: TopbarDomainTone;
};

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
  domainContext: GlobalTopbarDomainContext;
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

export function GlobalTopbar({
  primaryNav,
  globalSummary,
  currentPath,
  themePreference,
  resolvedTheme,
  onThemeChange,
  domainContext,
  onOpenSearch,
}: GlobalTopbarProps) {
  return (
    <div className="global-topbar-inner">
      <div className="global-topbar-row global-topbar-main">
        <section className="global-brand" aria-label="global-brand-context">
          <p className="brand-kicker">Fenlie / 控制台</p>
          <strong><ClampText expandable={false} raw="只读操作与研究面">只读操作与研究面</ClampText></strong>
        </section>
        <section className="global-control-deck" aria-label="global-control-deck">
          <div className="theme-switcher" role="group" aria-label="theme-switcher">
            <span className="theme-switcher-label">主题</span>
            <div className="theme-switcher-segment">
              {THEME_OPTIONS.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  className={`theme-switcher-button ${themePreference === option.id ? 'active' : ''}`.trim()}
                  aria-label={option.ariaLabel}
                  aria-pressed={themePreference === option.id}
                  onClick={() => onThemeChange(option.id)}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <span className="theme-switcher-state">当前：{resolvedThemeLabel(resolvedTheme)}</span>
          </div>
          {onOpenSearch ? (
            <button
              type="button"
              className="chip-button global-search-trigger"
              aria-label="打开全局搜索"
              onClick={onOpenSearch}
            >
              搜索 / ⌘K
            </button>
          ) : null}
          <div className="global-summary-strip" aria-label="global-summary">
            {globalSummary.chips?.map((chip) => (
              <span className="summary-chip" key={`${chip.label}-${chip.value}`}>{chip.label}：{chip.value}</span>
            ))}
            <span className="summary-chip summary-chip-emphasis">变更级别：{globalSummary.changeClass}</span>
          </div>
        </section>
      </div>
      <div className="global-topbar-row global-topbar-domain">
          <section className={`domain-context domain-context-${domainContext.tone}`.trim()} aria-label="global-domain-context">
          <span className="domain-context-kicker">{domainContext.kicker}</span>
          <strong><ClampText expandable={false} raw={domainContext.title}>{domainContext.title}</ClampText></strong>
          <small><ClampText expandable={false} raw={domainContext.detail}>{domainContext.detail}</ClampText></small>
        </section>
        <nav className="primary-nav" aria-label="primary-domains">
          {primaryNav.map((item) => (
            <NavLink
              key={item.id}
              className={({ isActive }) => `primary-nav-link ${isActive || currentPath === item.to ? 'active' : ''}`.trim()}
              to={item.to}
            >
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>
      </div>
    </div>
  );
}
