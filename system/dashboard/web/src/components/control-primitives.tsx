import type { AnchorHTMLAttributes, ButtonHTMLAttributes, CSSProperties, ReactNode } from 'react';
import { Link, NavLink, type LinkProps, type NavLinkProps, type NavLinkRenderProps, useMatch, useResolvedPath } from 'react-router-dom';

function joinControlClass(base: string, extra?: string) {
  return [base, extra].filter(Boolean).join(' ');
}

function normalizeNavRenderState(
  state: NavLinkRenderProps,
  forceActive?: boolean,
): NavLinkRenderProps {
  if (!forceActive || state.isActive) return state;
  return {
    ...state,
    isActive: true,
  };
}

function normalizeNavClassName(
  base: string,
  className?: NavLinkProps['className'],
  forceActive?: boolean,
): NavLinkProps['className'] {
  if (typeof className === 'function') {
    return (state) => {
      const effectiveState = normalizeNavRenderState(state, forceActive);
      return (
      joinControlClass(
        joinControlClass(base, effectiveState.isActive ? 'active' : undefined),
        className(effectiveState),
      ));
    };
  }

  return (state) =>
    joinControlClass(
      joinControlClass(base, state.isActive || forceActive ? 'active' : undefined),
      className,
    );
}

type DomainTabProps = Omit<LinkProps, 'className' | 'style' | 'children'> & {
  active?: boolean;
  caseSensitive?: boolean;
  end?: boolean;
  className?: NavLinkProps['className'];
  style?: CSSProperties | ((props: NavLinkRenderProps) => CSSProperties | undefined);
  children?: ReactNode | ((props: NavLinkRenderProps) => ReactNode);
};

function resolveDomainTabMatchEnd(pathname: string, end?: boolean) {
  if (typeof end === 'boolean') return end;
  return pathname === '/';
}

export function DomainTab({ className, active, children, style, to, end, caseSensitive, ...props }: DomainTabProps) {
  const resolvedPath = useResolvedPath(to, { relative: props.relative });
  const routeActive = Boolean(useMatch({
    path: resolvedPath.pathname,
    end: resolveDomainTabMatchEnd(resolvedPath.pathname, end),
    caseSensitive,
  }));
  const renderState = normalizeNavRenderState({
    isActive: routeActive,
    isPending: false,
    isTransitioning: false,
  }, active);

  return (
    <Link
      {...props}
      to={to}
      aria-current={renderState.isActive ? 'page' : undefined}
      className={typeof className === 'function'
        ? joinControlClass(
          joinControlClass('control-domain-tab', renderState.isActive ? 'active' : undefined),
          className(renderState),
        )
        : joinControlClass(
          joinControlClass('control-domain-tab', renderState.isActive ? 'active' : undefined),
          className,
        )}
      style={typeof style === 'function' ? style(renderState) : style}
    >
      {typeof children === 'function' ? children(renderState) : children}
    </Link>
  );
}

type ActionLinkProps = Omit<NavLinkProps, 'type'> &
  Omit<AnchorHTMLAttributes<HTMLAnchorElement>, keyof NavLinkProps | 'type'>;

export function ActionLink({ className, ...props }: ActionLinkProps) {
  return <NavLink {...props} className={normalizeNavClassName('control-action-link', className)} />;
}

export function SegmentedOption({
  className,
  active,
  type,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { active?: boolean }) {
  return (
    <button
      type={type ?? 'button'}
      {...props}
      aria-pressed={Boolean(active)}
      data-active={active ? 'true' : 'false'}
      className={joinControlClass('control-segment-option', className)}
    />
  );
}

export function ActionButton({ className, type, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button type={type ?? 'button'} {...props} className={joinControlClass('control-action-button', className)} />;
}

export function FilterChip({
  className,
  active,
  type,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { active?: boolean }) {
  return (
    <button
      type={type ?? 'button'}
      {...props}
      aria-pressed={Boolean(active)}
      data-active={active ? 'true' : 'false'}
      className={joinControlClass('control-filter-chip', className)}
    />
  );
}

export function EntityRowButton({
  title,
  subtitle,
  active,
  className,
  type,
  children,
  ...props
}: Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'title'> & {
  title: ReactNode;
  subtitle?: ReactNode;
  active?: boolean;
  children?: ReactNode;
}) {
  return (
    <button
      type={type ?? 'button'}
      {...props}
      data-active={active ? 'true' : 'false'}
      className={joinControlClass('control-entity-row', className)}
    >
      <span className="control-entity-row-title">{title}</span>
      {subtitle ? <small className="control-entity-row-subtitle">{subtitle}</small> : null}
      {children}
    </button>
  );
}
