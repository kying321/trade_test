import React from 'react';
import { labelFor } from '../../utils/dictionary';

const t = (key) => labelFor(key);

export function BreadcrumbBar({ items, onBack, canGoBack }) {
  return (
    <div className="breadcrumb-bar">
      <button className="ghost-button" disabled={!canGoBack} onClick={onBack}>{`← ${t('fallback')}`}</button>
      <div className="breadcrumb-track">
        {items.map((item, index) => (
          <React.Fragment key={`${item.label}-${index}`}>
            {index > 0 ? <span className="breadcrumb-sep">/</span> : null}
            <button className="breadcrumb-button" onClick={item.onClick} disabled={item.disabled}>{item.label}</button>
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
