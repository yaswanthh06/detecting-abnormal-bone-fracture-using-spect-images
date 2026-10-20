import React from 'react';
import { QualityMetrics } from '../types';
import { CheckCircle2, AlertTriangle, ShieldCheck, Gauge, Sliders, Activity } from 'lucide-react';

interface QualityAnalysisCardProps {
  quality: QualityMetrics;
}

export const QualityAnalysisCard: React.FC<QualityAnalysisCardProps> = ({ quality }) => {
  const flagColor = {
    optimal: 'bg-emerald-50 text-emerald-800 ring-emerald-200 border-emerald-300',
    acceptable: 'bg-blue-50 text-blue-800 ring-blue-200 border-blue-300',
    suboptimal: 'bg-amber-50 text-amber-800 ring-amber-200 border-amber-300',
  }[quality.quality_flag];

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs">
      <div className="flex items-center justify-between border-b border-slate-100 pb-3">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-5 w-5 text-cyan-700" />
          <h3 className="text-sm font-semibold text-slate-900">X-ray Image Quality</h3>
        </div>
        <span className={`inline-flex items-center gap-1 rounded-md px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wider ring-1 ${flagColor}`}>
          {quality.quality_flag === 'optimal' && <CheckCircle2 className="h-3.5 w-3.5" />}
          {quality.quality_flag !== 'optimal' && <AlertTriangle className="h-3.5 w-3.5" />}
          {quality.quality_flag} Quality
        </span>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {/* Resolution */}
        <div className="rounded-lg border border-slate-100 bg-slate-50 p-3">
          <p className="text-[11px] font-medium text-slate-500 uppercase tracking-wider">Resolution</p>
          <p className="mt-1 text-sm font-bold text-slate-800">
            {quality.resolution[1]} × {quality.resolution[0]} px
          </p>
          <p className="text-[10px] text-slate-500">{quality.channels}-channel input</p>
        </div>

        {/* Dynamic Range */}
        <div className="rounded-lg border border-slate-100 bg-slate-50 p-3">
          <p className="text-[11px] font-medium text-slate-500 uppercase tracking-wider">Dynamic Range</p>
          <p className="mt-1 text-sm font-bold text-slate-800">
            {quality.dynamic_range_min.toFixed(0)} - {quality.dynamic_range_max.toFixed(0)}
          </p>
          <p className="text-[10px] text-slate-500">Span: {(quality.dynamic_range_max - quality.dynamic_range_min).toFixed(0)} intensity levels</p>
        </div>

        {/* Michelson Contrast */}
        <div className="rounded-lg border border-slate-100 bg-slate-50 p-3">
          <p className="text-[11px] font-medium text-slate-500 uppercase tracking-wider">Michelson Contrast</p>
          <p className="mt-1 text-sm font-bold text-slate-800">
            {(quality.contrast_ratio * 100).toFixed(1)}%
          </p>
          <p className="text-[10px] text-slate-500">Image contrast estimate</p>
        </div>

        {/* Sharpness */}
        <div className="rounded-lg border border-slate-100 bg-slate-50 p-3">
          <p className="text-[11px] font-medium text-slate-500 uppercase tracking-wider">Laplacian Sharpness</p>
          <p className="mt-1 text-sm font-bold text-slate-800">
            {quality.sharpness_score.toFixed(1)}
          </p>
          <p className="text-[10px] text-slate-500">Edge gradient variance</p>
        </div>
      </div>

      {/* Hotspot & Intensity details */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">
        <span className="flex items-center gap-1">
          <Activity className="h-3.5 w-3.5 text-cyan-600" />
          High-Intensity Area Ratio: <strong>{(quality.hotspot_ratio * 100).toFixed(2)}%</strong> of image area
        </span>
        <span>
          Mean Intensity: <strong>{quality.mean_intensity}</strong> ± <strong>{quality.std_intensity}</strong>
        </span>
      </div>

      {/* Warnings */}
      {quality.warnings.length > 0 && (
        <div className="mt-3 space-y-1 rounded-lg bg-amber-50/70 p-3 text-xs text-amber-800">
          <p className="font-semibold text-amber-900">Quality Observations:</p>
          <ul className="list-inside list-disc space-y-0.5 text-amber-800">
            {quality.warnings.map((warning, idx) => (
              <li key={idx}>{warning}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
