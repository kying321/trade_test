export type SearchCategory = 'module' | 'route' | 'artifact';
export type SearchScope = 'all' | SearchCategory;

export interface SearchCatalogEntry {
  id: string;
  category: SearchCategory;
  title: string;
  subtitle?: string;
  keywords: string[];
  destination: string;
}

export interface SearchResultEntry extends SearchCatalogEntry {
  matchedFields: string[];
  score: number;
}
