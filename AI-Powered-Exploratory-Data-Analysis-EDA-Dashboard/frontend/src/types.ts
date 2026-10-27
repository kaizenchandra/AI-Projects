export type Profile = {
  name: string;
  physical_type: string;
  inferred_type: string;
  semantic_type: string;
  override: string | null;
  count: number;
  non_null_count: number;
  null_count: number;
  missing_percentage: number;
  distinct_count: number;
  constant: boolean;
  high_cardinality: boolean;
  potential_sensitive: boolean;
  precision_warning?: boolean;
  invalid_count?: number;
  nonfinite_count?: number;
  top_values: { value: string; count: number; relative_frequency: number }[];
  other_count: number;
  statistics: Record<string, number | null>;
  distribution: {
    counts?: number[];
    edges?: number[];
    kde?: { x: number[]; y: number[]; n: number } | null;
    box?: number[];
  };
  outliers: {
    method: string;
    threshold: number;
    lower: number | null;
    upper: number | null;
    count: number;
    percentage: number;
    eligible: boolean;
  } | null;
  outlier_records?: { row_index: number; value: unknown }[];
  outlier_records_total?: number;
  temporal?: {
    earliest: string;
    latest: string;
    daily_counts: [string, number][];
    periods_total: number;
    cadence: string;
  };
};
export type Insight = {
  id: string;
  type: string;
  severity: string;
  title: string;
  affected_columns: string[];
  evidence: { metric: string; value: number };
  recommendation: string;
  method: string;
  limitations: string[];
};
export type Analysis = {
  run_id: string;
  fingerprint: string;
  created_at: string;
  overview: {
    rows: number;
    source_rows: number;
    columns: number;
    missing_percentage: number;
    duplicate_rows: number;
    memory_bytes: number;
    type_breakdown: Record<string, number>;
  };
  profiles: Profile[];
  insights: Insight[];
  correlations: {
    x: string;
    y: string;
    coefficient: number | null;
    n: number;
    points: [number, number][];
    plotted_n: number;
  }[];
  associations: { x: string; y: string; cramers_v: number; n: number }[];
  preview: Record<string, string | null>[];
  missingness: { columns: string[]; matrix: boolean[][]; shown_rows: number };
  methodology: Record<string, unknown>;
  config: Config;
};
export type Config = {
  filters: { column: string; op: string; value: string | null }[];
  overrides: Record<string, string>;
  correlation: string;
  outlier_method: string;
  threshold: number;
  bins: number;
  columns: string[];
};
export type Dataset = {
  id: string;
  name: string;
  rows: number;
  cols: number;
  sheets: string;
};
