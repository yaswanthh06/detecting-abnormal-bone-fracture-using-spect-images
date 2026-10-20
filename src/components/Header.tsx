import React from 'react';
import { Activity, Cpu, CheckCircle2, AlertTriangle, ShieldAlert, RefreshCw } from 'lucide-react';
import { ModelStatusResponse } from '../types';

interface HeaderProps {
  activeTab: 'screening' | 'status' | 'evaluation';
  setActiveTab: (tab: 'screening' | 'status' | 'evaluation') => void;
  modelStatus: ModelStatusResponse | null;
  isBackendConnected: boolean;
  isLoadingStatus: boolean;
  onRefreshStatus: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  modelStatus,
  isBackendConnected,
  isLoadingStatus,
  onRefreshStatus,
}) => (
  <header className="border-b border-slate-200 bg-white shadow-xs">
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
      <div className="flex flex-col gap-4 py-4 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-2.5">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-cyan-700 text-white shadow-xs">
            <Activity className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900">
              FracAtlas X-ray Bone Fracture Classification
            </h1>
            <p className="text-xs text-slate-500">
              FracAtlas X-ray images &bull; PyTorch ResNet-18 &bull; Grad-CAM spatial localization
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1 rounded-md bg-amber-50 px-2.5 py-1 text-xs font-medium text-amber-800 ring-1 ring-amber-200">
            <ShieldAlert className="h-3.5 w-3.5" />
            Research Use Only
          </span>
          <span
            className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium ring-1 ${
              isBackendConnected
                ? 'bg-emerald-50 text-emerald-800 ring-emerald-200'
                : 'bg-rose-50 text-rose-800 ring-rose-200'
            }`}
          >
            <span className={`h-2 w-2 rounded-full ${isBackendConnected ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
            {isBackendConnected ? 'FastAPI Connected' : 'FastAPI Offline'}
          </span>
          {isBackendConnected && (
            <span
              className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium ring-1 ${
                modelStatus?.checkpoint_loaded
                  ? 'bg-blue-50 text-blue-800 ring-blue-200'
                  : 'bg-amber-50 text-amber-800 ring-amber-200'
              }`}
            >
              {modelStatus?.checkpoint_loaded ? (
                <>
                  <CheckCircle2 className="h-3.5 w-3.5 text-blue-600" />
                  Checkpoint Loaded
                </>
              ) : (
                <>
                  <AlertTriangle className="h-3.5 w-3.5 text-amber-600" />
                  No Checkpoint
                </>
              )}
            </span>
          )}
          {isBackendConnected && modelStatus && (
            <span className="inline-flex items-center gap-1 rounded-md bg-slate-100 px-2 py-1 text-xs text-slate-700">
              <Cpu className="h-3.5 w-3.5 text-slate-500" />
              {modelStatus.device.toUpperCase()}
            </span>
          )}
          <button
            onClick={onRefreshStatus}
            disabled={isLoadingStatus}
            title="Refresh connection and checkpoint status"
            className="inline-flex items-center justify-center rounded-md border border-slate-300 bg-white p-1.5 text-slate-600 hover:bg-slate-50 disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isLoadingStatus ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      <div className="flex space-x-1 border-t border-slate-100 pt-2">
        {([
          ['screening', 'Clinical Screening & Grad-CAM'],
          ['status', 'Model & Weights Status'],
          ['evaluation', 'Dataset & Held-Out Evaluation'],
        ] as const).map(([tab, label]) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === tab
                ? 'border-cyan-600 text-cyan-700'
                : 'border-transparent text-slate-600 hover:border-slate-300 hover:text-slate-800'
            }`}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  </header>
);
