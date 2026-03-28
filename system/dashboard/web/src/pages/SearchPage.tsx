import type { TerminalReadModel } from '../types/contracts';
import type { SearchResultEntry, SearchScope } from '../search/types';
import { SearchPageContent } from '../search/SearchResults';

type SearchPageProps = {
  model?: TerminalReadModel | null;
  query: string;
  scope: SearchScope;
  onPick: (entry: SearchResultEntry) => void;
  onQueryChange: (next: string) => void;
  onScopeChange: (next: SearchScope) => void;
};

export function SearchPage(props: SearchPageProps) {
  return <SearchPageContent {...props} />;
}
