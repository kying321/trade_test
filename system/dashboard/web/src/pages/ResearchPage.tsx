import { WorkspacePanels } from '../components/WorkspacePanels';
import type { TerminalReadModel } from '../types/contracts';
import type { SharedFocusState, WorkspaceSection } from '../utils/focus-links';

type ResearchPageProps = {
  model: TerminalReadModel;
  focus: SharedFocusState;
  section: WorkspaceSection;
  pageSectionId?: string;
};

export function ResearchPage({ model, focus, section, pageSectionId }: ResearchPageProps) {
  return <WorkspacePanels model={model} section={section} focus={focus} pageSectionId={pageSectionId} />;
}
