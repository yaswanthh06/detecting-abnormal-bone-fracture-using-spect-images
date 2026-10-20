import React, { useState } from 'react';
import { ExplainResponse } from '../types';
import { Eye, Layers, Sliders, MapPin, Info, ShieldAlert } from 'lucide-react';

interface GradCamViewerProps {
  explanation: ExplainResponse;
}

export const GradCamViewer: React.FC<GradCamViewerProps> = ({ explanation }) => {
  const [viewMode, setViewMode] = useState<'overlay' | 'side-by-side'>('side-by-side');
  const [overlayAlpha, setOverlayAlpha] = useState<number>(0.5);

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-slate-100 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <Layers className="h-5 w-5 text-cyan-700" />
            <h3 className="text-base font-semibold text-slate-900">Grad-CAM Spatial Explainability</h3>
          </div>
          <p className="text-xs text-slate-500">
            Target Layer: <code className="rounded bg-slate-100 px-1 py-0.5">{explanation.target_layer}</code> • Class Target: <strong>{explanation.predicted_class}</strong>
          </p>
        </div>

        {/* View Mode controls */}
        <div className="flex items-center gap-2">
          <div className="inline-flex rounded-lg border border-slate-200 bg-slate-50 p-1 text-xs">
            <button
              onClick={() => setViewMode('side-by-side')}
              className={`rounded-md px-3 py-1 font-medium transition-colors ${
                viewMode === 'side-by-side'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Side-by-Side (3-View)
            </button>
            <button
              onClick={() => setViewMode('overlay')}
              className={`rounded-md px-3 py-1 font-medium transition-colors ${
                viewMode === 'overlay'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Interactive Overlay
            </button>
          </div>
        </div>
      </div>

      {/* Main Visual Display */}
      {viewMode === 'side-by-side' ? (
        <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-3">
          {/* 1. Original Input */}
          <div className="flex flex-col rounded-lg border border-slate-200 bg-slate-50 p-3">
            <div className="mb-2 flex items-center justify-between text-xs font-semibold text-slate-700">
              <span>Preprocessed X-ray</span>
              <span className="text-[10px] text-slate-500">224 × 224 px</span>
            </div>
            <div className="relative aspect-square w-full overflow-hidden rounded-md border border-slate-300 bg-black">
              <img
                src={explanation.original_processed_base64}
                alt="Original processed X-ray"
                className="h-full w-full object-contain"
              />
            </div>
            <p className="mt-2 text-[11px] text-slate-500">Standardized X-ray image used for inference</p>
          </div>

          {/* 2. Grad-CAM Activation Heatmap */}
          <div className="flex flex-col rounded-lg border border-slate-200 bg-slate-50 p-3">
            <div className="mb-2 flex items-center justify-between text-xs font-semibold text-slate-700">
              <span>Grad-CAM Activation Map</span>
              <span className="text-[10px] text-cyan-700 font-medium">Jet Colormap [0, 1]</span>
            </div>
            <div className="relative aspect-square w-full overflow-hidden rounded-md border border-slate-300 bg-black">
              <img
                src={explanation.heatmap_base64}
                alt="Grad-CAM Heatmap"
                className="h-full w-full object-contain"
              />
            </div>
            <p className="mt-2 text-[11px] text-slate-500">Red = high gradients driving prediction</p>
          </div>

          {/* 3. Blended Overlay */}
          <div className="flex flex-col rounded-lg border border-slate-200 bg-slate-50 p-3">
            <div className="mb-2 flex items-center justify-between text-xs font-semibold text-slate-700">
              <span>Superimposed Overlay</span>
              <span className="text-[10px] text-slate-500">α = 0.45</span>
            </div>
            <div className="relative aspect-square w-full overflow-hidden rounded-md border border-slate-300 bg-black">
              <img
                src={explanation.overlay_base64}
                alt="Grad-CAM Overlay"
                className="h-full w-full object-contain"
              />
            </div>
            <p className="mt-2 text-[11px] text-slate-500">Spatial correspondence with skeletal anatomy</p>
          </div>
        </div>
      ) : (
        /* Interactive Blended Overlay with Opacity Slider */
        <div className="mt-6 flex flex-col items-center">
          <div className="relative aspect-square w-full max-w-md overflow-hidden rounded-xl border border-slate-300 bg-black shadow-md">
            <img
              src={explanation.original_processed_base64}
              alt="Original X-ray base"
              className="absolute inset-0 h-full w-full object-contain"
            />
            <img
              src={explanation.heatmap_base64}
              alt="Heatmap layer"
              style={{ opacity: overlayAlpha }}
              className="absolute inset-0 h-full w-full object-contain mix-blend-screen transition-opacity"
            />
          </div>

          <div className="mt-4 flex w-full max-w-md items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
            <Sliders className="h-4 w-4 text-slate-600" />
            <span className="text-xs font-medium text-slate-700">Heatmap Opacity:</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={overlayAlpha}
              onChange={(e) => setOverlayAlpha(parseFloat(e.target.value))}
              className="flex-1 accent-cyan-700 cursor-pointer"
            />
            <span className="text-xs font-mono font-bold text-slate-800 w-12 text-right">
              {Math.round(overlayAlpha * 100)}%
            </span>
          </div>
        </div>
      )}

      {/* Focus Hotspot Regions Table */}
      {explanation.focus_regions && explanation.focus_regions.length > 0 && (
        <div className="mt-6 rounded-lg border border-slate-200 bg-slate-50 p-4">
          <div className="flex items-center gap-2 mb-3">
            <MapPin className="h-4 w-4 text-cyan-700" />
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800">
              Candidate High-Activation Regions (Connected Components)
            </h4>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-700">
              <thead className="border-b border-slate-200 text-[11px] font-semibold text-slate-500 uppercase">
                <tr>
                  <th className="py-2 px-3">Region #</th>
                  <th className="py-2 px-3">Normalized Center [Y, X]</th>
                  <th className="py-2 px-3">Peak Intensity</th>
                  <th className="py-2 px-3">Relative Image Area</th>
                  <th className="py-2 px-3">Activation Category</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {explanation.focus_regions.map((region) => (
                  <tr key={region.region_id} className="hover:bg-white/80">
                    <td className="py-2 px-3 font-semibold text-slate-900">ROI #{region.region_id}</td>
                    <td className="py-2 px-3 font-mono">
                      [{region.center_normalized[0].toFixed(2)}, {region.center_normalized[1].toFixed(2)}]
                    </td>
                    <td className="py-2 px-3 font-mono font-bold text-cyan-800">
                      {(region.peak_intensity * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-3">{region.relative_area_pct}%</td>
                    <td className="py-2 px-3 font-medium text-slate-800">{region.intensity_category}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Explainability Note */}
      <div className="mt-4 flex items-start gap-2.5 rounded-lg bg-cyan-50/70 p-3 text-xs text-cyan-950 border border-cyan-200">
        <Info className="h-4 w-4 shrink-0 text-cyan-700 mt-0.5" />
        <div>
          <p className="font-semibold text-cyan-900">X-ray Explainability Note:</p>
          <p className="mt-0.5 text-cyan-800 text-[11px]">
            Grad-CAM highlights image regions that most influenced this model output in <code>layer4</code>. These activations are not anatomical measurements, fracture localizations, or a substitute for radiologist interpretation.
          </p>
        </div>
      </div>
    </div>
  );
};
