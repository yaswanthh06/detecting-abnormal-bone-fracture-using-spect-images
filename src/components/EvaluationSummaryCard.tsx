import React from 'react';
import { EvaluationSummaryResponse } from '../types';
import { BarChart3, Users, FileSpreadsheet, CheckCircle2, AlertTriangle, ShieldCheck, Terminal } from 'lucide-react';

interface EvaluationSummaryCardProps {
  evaluation: EvaluationSummaryResponse | null;
  isLoading: boolean;
  onRefresh: () => void;
}

export const EvaluationSummaryCard: React.FC<EvaluationSummaryCardProps> = ({
  evaluation,
  isLoading,
  onRefresh,
}) => {
  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center rounded-xl border border-slate-200 bg-white p-8">
        <div className="flex items-center gap-3 text-sm text-slate-600">
          <span className="h-5 w-5 animate-spin rounded-full border-2 border-cyan-700 border-t-transparent" />
          Fetching evaluation summary from backend...
        </div>
      </div>
    );
  }

  if (!evaluation || !evaluation.has_evaluation_data) {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50/50 p-6 shadow-xs space-y-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="h-6 w-6 shrink-0 text-amber-600 mt-0.5" />
          <div className="space-y-1">
            <h3 className="text-base font-bold text-amber-950">
              No Evaluation Summary Available Yet
            </h3>
            <p className="text-xs text-amber-800">
              {evaluation?.message ||
                'To maintain scientific honesty, metrics are not synthesized or fabricated. Run the evaluation script with a trained checkpoint and held-out test split.'}
            </p>
          </div>
        </div>

        {/* Manifest split counts if manifest exists */}
        {evaluation?.split_counts && (
          <div className="rounded-lg bg-white p-4 border border-amber-200">
            <p className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2">
              Detected Dataset Cohort Breakdown (From Manifest):
            </p>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 text-xs">
              <div className="rounded bg-slate-50 p-2 border border-slate-100">
                <span className="text-slate-500">Train Cohort:</span>
                <p className="font-bold text-slate-800 mt-0.5">
                  {evaluation.split_counts.train_images} images
                </p>
              </div>
              <div className="rounded bg-slate-50 p-2 border border-slate-100">
                <span className="text-slate-500">Validation:</span>
                <p className="font-bold text-slate-800 mt-0.5">
                  {evaluation.split_counts.val_images} images
                </p>
              </div>
              <div className="rounded bg-slate-50 p-2 border border-slate-100">
                <span className="text-slate-500">Held-Out Test:</span>
                <p className="font-bold text-slate-800 mt-0.5">
                  {evaluation.split_counts.test_images} images
                </p>
              </div>
              <div className="rounded bg-slate-50 p-2 border border-slate-100">
                <span className="text-slate-500">Total Scans:</span>
                <p className="font-bold text-slate-800 mt-0.5">{evaluation.split_counts.total_images} total</p>
              </div>
            </div>
          </div>
        )}

        <div className="rounded-lg border border-slate-200 bg-white p-4 space-y-2">
          <div className="flex items-center gap-2 text-xs font-bold text-slate-800">
            <Terminal className="h-4 w-4 text-slate-600" />
            <span>Command to Evaluate Held-Out Images</span>
          </div>
          <div className="rounded bg-slate-900 p-3 font-mono text-xs text-slate-100">
            <p className="text-cyan-400">
              python -m backend.ml.evaluate --checkpoint backend/checkpoints/best_resnet18_fracatlas.pt --manifest backend/data/manifest.csv
            </p>
          </div>
          <p className="text-[11px] text-slate-500">
            This independently computes accuracy, sensitivity, specificity, F1-score, ROC-AUC, and the confusion matrix on the held-out image-level test split.
          </p>
        </div>
      </div>
    );
  }

  const { metrics, confusion_matrix, split_counts, evaluation_timestamp } = evaluation;

  return (
    <div className="space-y-6">
      {/* Header & Verification Pill */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-slate-100 pb-4">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-cyan-700" />
            <div>
              <h3 className="text-base font-semibold text-slate-900">
                Independent Held-Out Test Evaluation
              </h3>
              <p className="text-xs text-slate-500">
                Stratified image-level test split • Evaluated: {evaluation_timestamp ? new Date(evaluation_timestamp).toLocaleString() : 'Recent'}
              </p>
            </div>
          </div>

          <button
            onClick={onRefresh}
            className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 shadow-xs hover:bg-slate-50"
          >
            Refresh Metrics
          </button>
        </div>

        {/* Primary Metrics Grid */}
        {metrics && (
          <div className="mt-6 grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
            {/* Accuracy */}
            <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Accuracy</p>
              <p className="mt-1 text-2xl font-black text-slate-900">
                {(metrics.accuracy * 100).toFixed(1)}%
              </p>
              <p className="text-[10px] text-slate-500">Overall correctness</p>
            </div>

            {/* Sensitivity / Recall */}
            <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Sensitivity (Recall)</p>
              <p className="mt-1 text-2xl font-black text-cyan-800">
                {(metrics.recall_sensitivity * 100).toFixed(1)}%
              </p>
              <p className="text-[10px] text-slate-500">True Fracture detection</p>
            </div>

            {/* Specificity */}
            <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Specificity</p>
              <p className="mt-1 text-2xl font-black text-slate-900">
                {(metrics.specificity * 100).toFixed(1)}%
              </p>
              <p className="text-[10px] text-slate-500">True Normal identification</p>
            </div>

            {/* Precision */}
            <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Precision (PPV)</p>
              <p className="mt-1 text-2xl font-black text-slate-900">
                {(metrics.precision * 100).toFixed(1)}%
              </p>
              <p className="text-[10px] text-slate-500">Positive predictive value</p>
            </div>

            {/* F1-Score */}
            <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">F1-Score</p>
              <p className="mt-1 text-2xl font-black text-slate-900">
                {(metrics.f1_score * 100).toFixed(1)}%
              </p>
              <p className="text-[10px] text-slate-500">Harmonic balance</p>
            </div>

            {/* ROC-AUC */}
            <div className="rounded-lg border border-slate-100 bg-slate-50 p-4">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">ROC-AUC</p>
              <p className="mt-1 text-2xl font-black text-slate-900">
                {metrics.roc_auc !== null && metrics.roc_auc !== undefined
                  ? metrics.roc_auc.toFixed(3)
                  : 'N/A'}
              </p>
              <p className="text-[10px] text-slate-500">Discrimination index</p>
            </div>
          </div>
        )}

        {/* Confusion Matrix & Split Breakdown */}
        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Confusion Matrix */}
          {confusion_matrix && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800 mb-3">
                Confusion Matrix (Held-Out Test Set)
              </h4>
              <div className="grid grid-cols-2 gap-3 text-center">
                <div className="rounded-lg border border-emerald-200 bg-emerald-50/70 p-3">
                  <p className="text-[11px] font-semibold text-emerald-800">True Negatives (TN)</p>
                  <p className="text-2xl font-black text-emerald-950 mt-1">{confusion_matrix.true_negatives}</p>
                  <p className="text-[10px] text-emerald-700">Normal correctly identified</p>
                </div>
                <div className="rounded-lg border border-rose-200 bg-rose-50/70 p-3">
                  <p className="text-[11px] font-semibold text-rose-800">False Positives (FP)</p>
                  <p className="text-2xl font-black text-rose-950 mt-1">{confusion_matrix.false_positives}</p>
                  <p className="text-[10px] text-rose-700">Type I error (False alarm)</p>
                </div>
                <div className="rounded-lg border border-rose-200 bg-rose-50/70 p-3">
                  <p className="text-[11px] font-semibold text-rose-800">False Negatives (FN)</p>
                  <p className="text-2xl font-black text-rose-950 mt-1">{confusion_matrix.false_negatives}</p>
                  <p className="text-[10px] text-rose-700">Type II error (Missed fracture)</p>
                </div>
                <div className="rounded-lg border border-emerald-200 bg-emerald-50/70 p-3">
                  <p className="text-[11px] font-semibold text-emerald-800">True Positives (TP)</p>
                  <p className="text-2xl font-black text-emerald-950 mt-1">{confusion_matrix.true_positives}</p>
                  <p className="text-[10px] text-emerald-700">Fracture correctly detected</p>
                </div>
              </div>
            </div>
          )}

          {/* Image-Level Split Distribution */}
          {split_counts && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800 mb-3">
                Image-Level Split Distribution
              </h4>
              <div className="space-y-3 text-xs">
                <div className="flex items-center justify-between border-b border-slate-200 pb-2">
                  <span className="text-slate-600">Train Cohort:</span>
                  <span className="font-semibold text-slate-900">
                    {split_counts.train_images} images
                  </span>
                </div>
                <div className="flex items-center justify-between border-b border-slate-200 pb-2">
                  <span className="text-slate-600">Validation Cohort:</span>
                  <span className="font-semibold text-slate-900">
                    {split_counts.val_images} images
                  </span>
                </div>
                <div className="flex items-center justify-between border-b border-slate-200 pb-2">
                  <span className="text-slate-600">Held-Out Test Cohort:</span>
                  <span className="font-semibold text-slate-900">
                    {split_counts.test_images} images
                  </span>
                </div>
                <div className="flex items-center justify-between pt-1">
                  <span className="font-bold text-slate-800">Total Scans in Manifest:</span>
                  <span className="font-bold text-cyan-800">{split_counts.total_images} scans</span>
                </div>
              </div>

              <div className="mt-4 flex items-center gap-1.5 text-[11px] text-emerald-700 font-medium">
                <CheckCircle2 className="h-3.5 w-3.5" />
                <span>Stratified image-level split; FracAtlas patient IDs are unavailable.</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
