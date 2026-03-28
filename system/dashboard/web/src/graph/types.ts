export type GraphNodeKind =
  | 'pipeline_stage'
  | 'market'
  | 'strategy'
  | 'artifact'
  | 'route'
  | 'custom_pipeline';

export type GraphEdgeRelation =
  | 'feeds'
  | 'validates'
  | 'triggers'
  | 'routes_to'
  | 'depends_on';

export type GraphNode = {
  id: string;
  kind: GraphNodeKind;
  label: string;
  subtitle?: string;
  status?: string;
  importance: number;
  destination?: string;
  upstream: string[];
  downstream: string[];
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  relation: GraphEdgeRelation;
  strength: number;
  active?: boolean;
};

export type GraphPipeline = {
  id: string;
  name: string;
  nodeIds: string[];
  layout: 'radial';
  updatedAt: string;
  isDefault: boolean;
};

export type GraphPipelineStore = {
  version: 1;
  pipelines: GraphPipeline[];
};

export type GraphHomeModel = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  defaultCenterId: string;
};

export type GraphLayoutNode = GraphNode & {
  x: number;
  y: number;
  z: number;
  depth: 0 | 1 | 2;
  emphasis: number;
};
