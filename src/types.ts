export type OutputMode =
  | "MASTER_SOURCE_FAITHFUL"
  | "POD_READY"
  | "ALPHA_ONLY";

export type ToolMode = "pan" | "keep" | "remove" | "wand-keep" | "wand-remove";

export interface CanonicalImage {
  raw_hash: string;
  decoded_pixel_hash: string;
  width: number;
  height: number;
  original_orientation: number;
  canonical_orientation: number;
  source_mode: string;
  source_has_alpha: boolean;
  icc_profile_present: boolean;
  conversion_flags: string[];
}

export interface HistoryState {
  can_undo: boolean;
  can_redo: boolean;
  cursor: number;
  length: number;
}

export interface ComponentSummary {
  count: number;
  suspicious_count: number;
  components: Array<{
    id: number;
    area_px: number;
    bbox: [number, number, number, number];
    needs_review: boolean;
  }>;
}

export interface ProjectPayload {
  project_id: string;
  schema_version: string;
  source_path: string;
  project_path: string;
  preview_path: string;
  width: number;
  height: number;
  revision: string;
  canonical: CanonicalImage;
  history: HistoryState;
  components?: ComponentSummary;
  process?: {
    content_mode: string;
    quality_preset: string;
    subject_policy: string;
    diagnostics: Record<string, unknown>;
    ai_models_used: string[];
  } | null;
  warnings?: string[];
}

export interface PreflightItem {
  code: string;
  severity: "PASS" | "WARN" | "FAIL";
  message: string;
  value?: number | string | boolean;
}

export interface PreflightReport {
  status: "PASS" | "WARN" | "FAIL";
  effective_ppi?: number;
  warnings: Array<{ code: string; message: string; details?: unknown }>;
  failures: Array<{ code: string; message: string; details?: unknown }>;
  component_statistics: {
    count: number;
    small_count: number;
  };
  pixel_dimensions: { width: number; height: number };
  generated_at: string;
}

export interface ModelManifest {
  model_id: string;
  revision: string;
  weight_sha256: string | null;
  weight_size: number | null;
  code_revision: string;
  code_license: string;
  weight_license: string;
  commercial_pod_allowed: boolean;
  redistribution_allowed: boolean;
  preprocess_version: string;
  qualified_backends: string[];
  installed: boolean;
  status: string;
}

export interface HealthPayload {
  status: string;
  version: string;
  python: string;
  platform: string;
  projects_dir: string;
  processing_engine: string;
}

export interface ExportResult {
  path: string;
  mode: OutputMode;
  width: number;
  height: number;
  rgb_integrity?: boolean;
  bit_depth: number;
  warnings?: string[];
}
