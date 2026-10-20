import React, { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { ImageUploadDropzone } from './components/ImageUploadDropzone';
import { QualityAnalysisCard } from './components/QualityAnalysisCard';
import { ScreeningResultsCard } from './components/ScreeningResultsCard';
import { GradCamViewer } from './components/GradCamViewer';
import { ModelStatusCard } from './components/ModelStatusCard';
import { EvaluationSummaryCard } from './components/EvaluationSummaryCard';
import { api, ApiError } from './services/api';
import {
  ModelStatusResponse,
  ScreeningResponse,
  ExplainResponse,
  EvaluationSummaryResponse,
} from './types';
import { AlertCircle, FileText, CheckCircle2, ShieldAlert } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState<'screening' | 'status' | 'evaluation'>('screening');
  const [modelStatus, setModelStatus] = useState<ModelStatusResponse | null>(null);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const [isLoadingStatus, setIsLoadingStatus] = useState<boolean>(false);

  // Screening and Explainability states
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isScreening, setIsScreening] = useState<boolean>(false);
  const [isExplaining, setIsExplaining] = useState<boolean>(false);
  const [screeningResult, setScreeningResult] = useState<ScreeningResponse | null>(null);
  const [explainResult, setExplainResult] = useState<ExplainResponse | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  // Evaluation states
  const [evaluationData, setEvaluationData] = useState<EvaluationSummaryResponse | null>(null);
  const [isLoadingEvaluation, setIsLoadingEvaluation] = useState<boolean>(false);

  // Fetch model and backend status
  const fetchStatus = useCallback(async () => {
    setIsLoadingStatus(true);
    try {
      const status = await api.getModelStatus();
      setModelStatus(status);
      setIsBackendConnected(true);
    } catch (err: unknown) {
      setIsBackendConnected(false);
      setModelStatus(null);
    } finally {
      setIsLoadingStatus(false);
    }
  }, []);

  // Fetch evaluation data
  const fetchEvaluation = useCallback(async () => {
    setIsLoadingEvaluation(true);
    try {
      const evalSummary = await api.getEvaluationSummary();
      setEvaluationData(evalSummary);
    } catch (err: unknown) {
      setEvaluationData(null);
    } finally {
      setIsLoadingEvaluation(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  useEffect(() => {
    if (activeTab === 'evaluation') {
      fetchEvaluation();
    }
  }, [activeTab, fetchEvaluation]);

  const handleFileSelect = (file: File | null) => {
    setSelectedFile(file);
    setScreeningResult(null);
    setExplainResult(null);
    setActionError(null);
  };

  const handleRunScreening = async () => {
    if (!selectedFile) return;
    setIsScreening(true);
    setActionError(null);

    try {
      const result = await api.screenImage(selectedFile);
      setScreeningResult(result);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setActionError(err.message);
      } else {
        setActionError('Failed to process image screening. Check backend connection.');
      }
    } finally {
      setIsScreening(false);
    }
  };

  const handleRunExplanation = async () => {
    if (!selectedFile) return;
    setIsExplaining(true);
    setActionError(null);

    try {
      const result = await api.explainImage(selectedFile);
      setExplainResult(result);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setActionError(err.message);
      } else {
        setActionError('Failed to generate Grad-CAM explanation. Check backend connection.');
      }
    } finally {
      setIsExplaining(false);
    }
  };

  const canRunInference = isBackendConnected && !!modelStatus?.checkpoint_loaded;
  let disabledReason: string | null = null;
  if (!isBackendConnected) {
    disabledReason = 'FastAPI backend is not running at http://127.0.0.1:8000. Start the server with uvicorn.';
  } else if (!modelStatus?.checkpoint_loaded) {
    disabledReason = 'No trained checkpoint loaded. Untrained model inference is disabled to prevent unvalidated predictions.';
  }

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        modelStatus={modelStatus}
        isBackendConnected={isBackendConnected}
        isLoadingStatus={isLoadingStatus}
        onRefreshStatus={fetchStatus}
      />

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Error notification banner */}
        {actionError && (
          <div className="mb-6 flex items-start justify-between rounded-xl border border-rose-200 bg-rose-50 p-4 text-rose-900 shadow-xs">
            <div className="flex items-start gap-3">
              <AlertCircle className="h-5 w-5 shrink-0 text-rose-600 mt-0.5" />
              <div>
                <p className="font-semibold text-sm">Operation Encountered An Issue</p>
                <p className="text-xs text-rose-800 mt-0.5">{actionError}</p>
              </div>
            </div>
            <button
              onClick={() => setActionError(null)}
              className="text-xs font-semibold text-rose-700 hover:text-rose-900"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Tab 1: Clinical Screening & Grad-CAM */}
        {activeTab === 'screening' && (
          <div className="space-y-6">
            <ImageUploadDropzone
              selectedFile={selectedFile}
              onFileSelect={handleFileSelect}
              isProcessing={isScreening || isExplaining}
              onRunScreening={handleRunScreening}
              onRunExplanation={handleRunExplanation}
              canRunInference={canRunInference}
              disabledReason={disabledReason}
            />

            {/* Quality analysis card when screening result is returned */}
            {screeningResult && (
              <QualityAnalysisCard quality={screeningResult.quality_analysis} />
            )}

            {/* Screening classification card */}
            {screeningResult && (
              <ScreeningResultsCard
                screening={screeningResult}
                onExplainClick={handleRunExplanation}
                isExplaining={isExplaining}
              />
            )}

            {/* Grad-CAM viewer card */}
            {explainResult && (
              <GradCamViewer explanation={explainResult} />
            )}
          </div>
        )}

        {/* Tab 2: Model & Weights Status */}
        {activeTab === 'status' && (
          <ModelStatusCard
            modelStatus={modelStatus}
            isBackendConnected={isBackendConnected}
            onRefresh={fetchStatus}
          />
        )}

        {/* Tab 3: Dataset & Held-Out Evaluation */}
        {activeTab === 'evaluation' && (
          <EvaluationSummaryCard
            evaluation={evaluationData}
            isLoading={isLoadingEvaluation}
            onRefresh={fetchEvaluation}
          />
        )}
      </main>

      {/* Footer disclaimer */}
      <footer className="mt-16 border-t border-slate-200 bg-white py-6 text-center text-xs text-slate-500">
        <div className="mx-auto max-w-7xl px-4">
          <p className="font-semibold text-slate-700">{/*
          */}FracAtlas X-ray Bone Fracture Classification &bull; Medical Machine Learning Research System</p>
          <p className="mt-1 text-[11px] text-slate-400">
            PyTorch ResNet-18 Model Architecture • Grad-CAM Layer-4 Heatmap Visualization • Patient-Level Group Isolation
          </p>
        </div>
      </footer>
    </div>
  );
}
