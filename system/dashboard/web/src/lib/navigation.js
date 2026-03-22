import { labelFor } from '../utils/dictionary';

const t = (key) => labelFor(key);

export const MAIN_TABS = [
  { id: 'overview', label: t('legacy_tab_overview_label'), eyebrow: t('legacy_tab_overview_eyebrow'), title: t('legacy_tab_overview_title'), subtitle: t('legacy_tab_overview_subtitle') },
  { id: 'backtests', label: t('legacy_tab_backtests_label'), eyebrow: t('legacy_tab_backtests_eyebrow'), title: t('legacy_tab_backtests_title'), subtitle: t('legacy_tab_backtests_subtitle') },
  { id: 'explorer', label: t('legacy_tab_explorer_label'), eyebrow: t('legacy_tab_explorer_eyebrow'), title: t('legacy_tab_explorer_title'), subtitle: t('legacy_tab_explorer_subtitle') },
  { id: 'interfaces', label: t('legacy_tab_interfaces_label'), eyebrow: t('legacy_tab_interfaces_eyebrow'), title: t('legacy_tab_interfaces_title'), subtitle: t('legacy_tab_interfaces_subtitle') },
  { id: 'raw', label: t('legacy_tab_raw_label'), eyebrow: t('legacy_tab_raw_eyebrow'), title: t('legacy_tab_raw_title'), subtitle: t('legacy_tab_raw_subtitle') },
];

export function getActiveTabMeta(activeTab) {
  return MAIN_TABS.find((item) => item.id === activeTab) || MAIN_TABS[0];
}
