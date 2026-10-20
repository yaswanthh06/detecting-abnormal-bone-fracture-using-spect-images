import React from 'react';
import { ModelStatusResponse } from '../types';
import { Cpu, CheckCircle2, AlertTriangle, Terminal, Layers, ShieldAlert, FileText, ArrowRight } from 'lucide-react';

interface ModelStatusCardProps {
  modelStatus: ModelStatusResponse | null;
  isBackendConnected: boolean;
  onRefresh: () => void;
}

export const ModelStatusCard: React.FC<ModelStatusCardProps> = ({
  modelStatus,
  isBackendConnected,
  onRefresh,
}) => {
  return (
    <div className="space-y-6">
      {/* Backend Connection Alert if Offline */}
      {!isBackendConnected && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-6 text-rose-900 shadow-xs">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-6 w-6 shrink-0 text-rose-600 mt-0.5" />
            <div className="space-y-2">
              <h3 className="text-base font-bold">FastAPI ML Backend Not Reachable</h3>
              <p className="text-xs text-rose-800">
                The frontend is attempting to query the PyTorch backend at <code className="font-mono bg-rose-100 px-1 py-0.5 rounded">http://127.0.0.1:8000</code>.
              </p>
              <div className="rounded-lg bg-slate-900 p-3 text-slate-100 font-mono text-xs">
                <p className="text-slate-400"># Start the FastAPI server:</p>
                <p className="text-emerald-400">uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Checkpoint Status Banner */}
      {isBackendConnected && modelStatus && (
        <div
          className={`rounded-xl border p-6 shadow-xs ${
            modelStatus.checkpoint_loaded
              ? 'border-emerald-200 bg-emerald-50/60 text-emerald-950'
              : 'border-amber-200 bg-amber-50/70 text-amber-950'
          }`}
        >
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start gap-3">
              {modelStatus.checkpoint_loaded ? (
                <CheckCircle2 className="h-6 w-6 shrink-0 text-emerald-600 mt-0.5" />
              ) : (
                <AlertTriangle className="h-6 w-6 shrink-0 text-amber-600 mt-0.5" />
              )}
              <div>
                <h3 className="text-base font-bold">
                  {modelStatus.checkpoint_loaded
                    ? 'Trained Checkpoint Active & Ready for Screening'
                    : 'Checkpoint File Not Found (Untrained Model Inference Halted)'}
                </h3>
                <p className="mt-1 text-xs text-slate-700">
                  {modelStatus.error_message ||
                    'Model weights have been successfully verified against the ResNet-18 FracAtlas X-ray configuration.'}
                </p>
              </div>
            </div>

            <button
              onClick={onRefresh}
              className="shrink-0 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 shadow-xs hover:bg-slate-50"
            >
              Verify Checkpoint
            </button>
          </div>

          {/* Setup Guide if Checkpoint Missing */}
          {!modelStatus.checkpoint_loaded && (
            <div className="mt-4 rounded-lg border border-amber-200 bg-white p-4 space-y-3">
              <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800 uppercase tracking-wider">
                <Terminal className="h-4 w-4 text-amber-600" />
                <span>How to Train or Mount the ResNet-18 Weights</span>
              </div>
              <p className="text-xs text-slate-600">
                To guarantee medical integrity, this system enforces zero fabricated predictions and does not permit uninitialized weights for diagnostic screening.
              </p>
              <div className="rounded-lg bg-slate-900 p-3 text-xs font-mono text-slate-100 space-y-2">
                <div>
                  <p className="text-slate-400"># 1. Run the image-level training pipeline:</p>
                  <p className="text-cyan-400">python -m backend.ml.train --manifest backend/data/manifest.csv --epochs 25 --device cpu</p>
                </div>
                <div>
                  <p className="text-slate-400"># 2. Or copy an existing trained checkpoint:</p>
                  <p className="text-cyan-400">cp /path/to/trained_weights.pt {modelStatus.checkpoint_path}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Model Architecture Specifications */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs">
        <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
          <Layers className="h-5 w-5 text-cyan-700" />
          <h3 className="text-sm font-semibold text-slate-900">PyTorch ResNet-18 Architecture Details</h3>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-lg border border-slate-100 bg-slate-50 p-3">
            <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Backbone</p>
            <p className="mt-1 text-sm font-bold text-slate-900">ResNet-18 (Residual CNN)</p>
            <p className="text-[11px] text-slate-500">11.7M trainable parameters</p>
          </div>

          <div className="rounded-lg border border-slate-100 bg-slate-50 p-3">
            <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Classification Head</p>
            <p className="mt-1 text-sm font-bold text-slate-900">Dropout(0.3) + Linear</p>
            <p className="text-[11px] text-slate-500">2 Output Classes</p>
          </div>

          <div className="rounded-lg border border-slate-100 bg-slate-50 p-3">
            <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Grad-CAM Hook Layer</p>
            <p className="mt-1 text-sm font-bold text-slate-900">layer4[-1]</p>
            <p className="text-[11px] text-slate-500">Last 512-channel residual block</p>
          </div>

          <div className="rounded-lg border border-slate-100 bg-slate-50 p-3">
            <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Active Device</p>
            <p className="mt-1 text-sm font-bold text-slate-900">{modelStatus?.device.toUpperCase() || 'CPU'}</p>
            <p className="text-[11px] text-slate-500">PyTorch Device Context</p>
          </div>
        </div>

        {/* Classes */}
        <div className="mt-4 rounded-lg bg-slate-50 p-3 border border-slate-100">
          <p className="text-xs font-semibold text-slate-700">Configured Target Classes:</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {modelStatus?.classes.map((c, i) => (
              <span
                key={i}
                className="inline-flex items-center rounded-md bg-white px-2.5 py-1 text-xs font-medium text-slate-800 border border-slate-200"
              >
                Index {i}: {c}
              </span>
            ))}
          </div>
        </div>

        {/* Checkpoint Metadata if available */}
        {modelStatus?.metadata && (
          <div className="mt-4 rounded-lg bg-slate-50 p-4 border border-slate-100">
            <p className="text-xs font-semibold text-slate-700 mb-2">Checkpoint Verification Metadata:</p>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 text-xs">
              <div>
                <span className="text-slate-500">Epoch:</span>{' '}
                <span className="font-semibold">{modelStatus.metadata.epoch ?? 'N/A'}</span>
              </div>
              <div>
                <span className="text-slate-500">Best Val F1:</span>{' '}
                <span className="font-semibold">
                  {modelStatus.metadata.best_val_f1 ? modelStatus.metadata.best_val_f1.toFixed(4) : 'N/A'}
                </span>
              </div>
              <div>
                <span className="text-slate-500">Best Val Loss:</span>{' '}
                <span className="font-semibold">
                  {modelStatus.metadata.best_val_loss ? modelStatus.metadata.best_val_loss.toFixed(4) : 'N/A'}
                </span>
              </div>
              <div>
                <span className="text-slate-500">Trained:</span>{' '}
                <span className="font-semibold truncate">
                  {modelStatus.metadata.training_timestamp
                    ? new Date(modelStatus.metadata.training_timestamp).toLocaleDateString()
                    : 'N/A'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
