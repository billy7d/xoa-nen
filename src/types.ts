export type OutputMode =
  | "MASTER_SOURCE_FAITHFUL"
  | "POD_READY"
  | "ALPHA_ONLY";

export type EngineProfile = "LEGACY_V1" | "V3_BALANCED" | "V3_AI_LOCAL";
export type WandAlgorithm = "LEGACY_COLOR" | "SMART";
export type SubjectPolicy = "ALL_DETECTED" | "SELECTED";
export type QualityPreset = "FAST" | "QUALITY";
export type ToolMode = "pan" | "subject" | "protect" | "keep" | "remove" | "wand-keep" | "wand-remove";

export interface ForegroundPoint {
  x: number;
  y: number;
}

export interface SubjectCandidate {
  id: number;
  area_px: number;
  bbox: [number, number, number, number];
  needs_review: boolean;
  selected: boolean;
  confidence: "detected" | "review";
}

export interface ReviewRegion {
  bbox: [number, number, number, number];
  area_proxy: number;
  reason: string;
}

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
    engine_profile: EngineProfile | "V2_ARCHIVED_RESULT";
    quality_preset: QualityPreset;
    subject_policy: SubjectPolicy;
    foreground_points?: ForegroundPoint[];
    background_points?: ForegroundPoint[];
    protection_mode?: "CONSERVATIVE";
    shadow_policy?: "REMOVE";
    result_status?: "READY" | "NEEDS_PROTECTION";
    diagnostics: Record<string, unknown>;
    ai_models_used: string[];
    subjects: SubjectCandidate[];
    selected_subject_ids: number[];
    review_regions: ReviewRegion[];
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
  effective_ppi?: { x: number; y: number } | null;
  print_dimensions?: {
    width: number;
    height: number;
    unit: "inch" | "cm";
    width_inch: number;
    height_inch: number;
  } | null;
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
  role?: string;
  adapter?: string;
  download_url?: string | null;
  signature_valid?: boolean;
  checksum_valid?: boolean;
  policy_valid?: boolean;
}

export interface WandPreview {
  selection_id: string;
  preview_path: string;
  selected_pixel_count: number;
  bounds: [number, number, number, number];
  mode: "keep" | "remove";
  wand_algorithm: WandAlgorithm;
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
