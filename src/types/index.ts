export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  service: string;
}

export interface CheckpointMetadata {
  epoch?: number;
  best_val_f1?: number;
  best_val_loss?: number;
  training_timestamp?: string;
  pytorch_version?: string;
  weights_sha256?: string;
}

export interface ModelStatusResponse {
  model_name: string;
  architecture: string;
  device: string;
  checkpoint_loaded: boolean;
  checkpoint_path: string;
  classes: string[];
  metadata?: CheckpointMetadata | null;
  error_message?: string | null;
  ready_for_inference: boolean;
}

export interface QualityMetrics {
  resolution: [number, number];
  channels: number;
  dynamic_range_min: number;
  dynamic_range_max: number;
  mean_intensity: number;
  std_intensity: number;
  contrast_ratio: number;
  sharpness_score: number;
  hotspot_ratio: number;
  quality_flag: 'optimal' | 'acceptable' | 'suboptimal';
  warnings: string[];
}

export interface ScreeningResponse {
  prediction_id: string;
  predicted_class: string;
  class_index: number;
  confidence: number;
  class_probabilities: Record<string, number>;
  quality_analysis: QualityMetrics;
  inference_time_ms: number;
  disclaimer: string;
  model_version: string;
}

export interface FocusRegion {
  region_id: number;
  center_normalized: [number, number];
  peak_intensity: number;
  relative_area_pct: number;
  intensity_category: string;
}

export interface ExplainResponse {
  prediction_id: string;
  predicted_class: string;
  confidence: number;
  target_layer: string;
  heatmap_base64: string;
  overlay_base64: string;
  original_processed_base64: string;
  focus_regions: FocusRegion[];
  inference_time_ms: number;
  disclaimer: string;
}

export interface ConfusionMatrixData {
  matrix: number[][];
  labels: string[];
  true_negatives: number;
  false_positives: number;
  false_negatives: number;
  true_positives: number;
}

export interface EvaluationMetrics {
  accuracy: number;
  precision: number;
  recall_sensitivity: number;
  specificity: number;
  f1_score: number;
  roc_auc?: number | null;
  test_sample_count: number;
}

export interface SplitCounts {
  train_images: number;
  val_images: number;
  test_images: number;
  total_images: number;
}

export interface EvaluationSummaryResponse {
  has_evaluation_data: boolean;
  evaluation_timestamp?: string | null;
  split_counts?: SplitCounts | null;
  metrics?: EvaluationMetrics | null;
  confusion_matrix?: ConfusionMatrixData | null;
  manifest_path: string;
  message: string;
}
