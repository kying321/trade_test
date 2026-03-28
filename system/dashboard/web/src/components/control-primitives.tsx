import type { ButtonHTMLAttributes } from 'react';
import { NavLink, type NavLinkProps } from 'react-router-dom';

function joinControlClass(base: string, extra?: string) {
  return [base, extra].filter(Boolean).join(' ');
}

export function DomainTab({ className, active: _active, ...props }: NavLinkProps & { active?: boolean }) {
  return <NavLink {...props} className={joinControlClass('control-domain-tab', className)} />;
}

export function SegmentedOption({ className, active, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { active?: boolean }) {
  return (
    <button
      {...props}
      data-active={active ? 'true' : 'false'}
      className={joinControlClass('control-segment-option', className)}
    />
  );
}

export function ActionButton({ className, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button {...props} className={joinControlClass('control-action-button', className)} />;
}

export function FilterChip({ className, active, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { active?: boolean }) {
  return (
    <button
      {...props}
      data-active={active ? 'true' : 'false'}
      className={joinControlClass('control-filter-chip', className)}
    />
  );
}

export function EntityRowButton({ title, subtitle, active, onClick }: {
  title: string;
  subtitle?: string;
  active?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      data-active={active ? 'true' : 'false'}
      className="control-entity-row"
      onClick={onClick}
    >
      <span>{title}</span>
      {subtitle ? <small>{subtitle}</small> : null}
    </button>
  );
}
