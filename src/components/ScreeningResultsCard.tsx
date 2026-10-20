import React from 'react';
import { ScreeningResponse } from '../types';
import { AlertCircle, CheckCircle2, ShieldAlert, Clock, Hash, Activity } from 'lucide-react';

interface ScreeningResultsCardProps {
  screening: ScreeningResponse;
  onExplainClick: () => void;
  isExplaining: boolean;
}

export const ScreeningResultsCard: React.FC<ScreeningResultsCardProps> = ({
  screening,
  onExplainClick,
  isExplaining,
}) => {
  const isAbnormal = screening.class_index === 1 || screening.predicted_class.toLowerCase().includes('abnormal') || screening.predicted_class.toLowerCase().includes('fracture');

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-slate-100 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-cyan-700" />
            <h3 className="text-base font-semibold text-slate-900">ResNet-18 Screening Classification</h3>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-slate-500">
            <span className="inline-flex items-center gap-1">
              <Hash className="h-3.5 w-3.5" />
              ID: {screening.prediction_id}
            </span>
            <span className="inline-flex items-center gap-1">
              <Clock className="h-3.5 w-3.5" />
              Latency: {screening.inference_time_ms} ms
            </span>
            <span>Version: {screening.model_version}</span>
          </div>
        </div>

        <button
          onClick={onExplainClick}
          disabled={isExplaining}
          className="inline-flex items-center justify-center rounded-lg bg-slate-900 px-4 py-2 text-xs font-semibold text-white shadow-xs hover:bg-slate-800 disabled:opacity-50"
        >
          {isExplaining ? (
            <>
              <span className="mr-1.5 h-3.5 w-3.5 animate-spin rounded-full border-2 border-white border-t-transparent" />
              Computing Grad-CAM...
            </>
          ) : (
            'View Grad-CAM Activation Heatmap'
          )}
        </button>
      </div>

      {/* Main classification result banner */}
      <div
        className={`mt-4 rounded-xl p-5 border ${
          isAbnormal
            ? 'bg-rose-50/70 border-rose-200 text-rose-950'
            : 'bg-emerald-50/70 border-emerald-200 text-emerald-950'
        }`}
      >
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <div
              className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-full ${
                isAbnormal ? 'bg-rose-100 text-rose-700' : 'bg-emerald-100 text-emerald-700'
              }`}
            >
              {isAbnormal ? (
                <AlertCircle className="h-7 w-7" />
              ) : (
                <CheckCircle2 className="h-7 w-7" />
              )}
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Predicted Scintigraphic Classification
              </p>
              <h4 className="text-xl font-bold tracking-tight">
                {screening.predicted_class}
              </h4>
            </div>
          </div>

          <div className="text-right sm:border-l sm:border-slate-200 sm:pl-6">
            <p className="text-xs font-medium text-slate-500">Confidence Score</p>
            <p className="text-2xl font-black">
              {(screening.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>

      {/* Per-class probabilities */}
      <div className="mt-5 space-y-3">
        <p className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
          Class Probability Distribution
        </p>

        {Object.entries(screening.class_probabilities).map(([className, prob]) => {
          const isThisAbnormal = className.toLowerCase().includes('abnormal') || className.toLowerCase().includes('fracture');
          const isSelected = className === screening.predicted_class;

          return (
            <div key={className} className="space-y-1">
              <div className="flex justify-between text-xs font-medium">
                <span className={isSelected ? 'font-bold text-slate-900' : 'text-slate-600'}>
                  {className} {isSelected && ' (Predicted)'}
                </span>
                <span className="font-semibold text-slate-800">
                  {(prob * 100).toFixed(2)}%
                </span>
              </div>
              <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    isThisAbnormal ? 'bg-rose-600' : 'bg-emerald-600'
                  }`}
                  style={{ width: `${Math.min(Math.max(prob * 100, 0), 100)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Clinical Disclaimer */}
      <div className="mt-6 flex items-start gap-2.5 rounded-lg bg-slate-50 p-3 text-[11px] text-slate-600 border border-slate-200">
        <ShieldAlert className="h-4 w-4 shrink-0 text-slate-500 mt-0.5" />
        <p>{screening.disclaimer}</p>
      </div>
    </div>
  );
};
