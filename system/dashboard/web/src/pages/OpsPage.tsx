import { TerminalPanels } from '../components/TerminalPanels';
import type { TerminalReadModel } from '../types/contracts';
import type { SharedFocusState } from '../utils/focus-links';

type OpsPageProps = {
  model: TerminalReadModel;
  focus: SharedFocusState;
  surface: 'public' | 'internal';
};

export function OpsPage({ model, focus, surface }: OpsPageProps) {
  return (
    <section className="ops-page" data-surface={surface}>
      <TerminalPanels model={model} focus={focus} />
    </section>
  );
}
