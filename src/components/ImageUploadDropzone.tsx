import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, X, FileCheck, AlertCircle } from 'lucide-react';

interface ImageUploadDropzoneProps {
  selectedFile: File | null;
  onFileSelect: (file: File | null) => void;
  isProcessing: boolean;
  onRunScreening: () => void;
  onRunExplanation: () => void;
  canRunInference: boolean;
  disabledReason?: string | null;
}

export const ImageUploadDropzone: React.FC<ImageUploadDropzoneProps> = ({
  selectedFile,
  onFileSelect,
  isProcessing,
  onRunScreening,
  onRunExplanation,
  canRunInference,
  disabledReason,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const allowedExtensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif'];
  const maxSizeBytes = 25 * 1024 * 1024; // 25MB

  const handleFile = (file: File) => {
    setValidationError(null);

    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!allowedExtensions.includes(ext)) {
      setValidationError(`Unsupported file type '${ext}'. Please upload PNG, JPG, or TIFF X-ray images.`);
      return;
    }

    if (file.size > maxSizeBytes) {
      setValidationError(`File size (${(file.size / (1024 * 1024)).toFixed(1)}MB) exceeds maximum allowed 25MB.`);
      return;
    }

    onFileSelect(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleClear = () => {
    onFileSelect(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setValidationError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold text-slate-900">FracAtlas X-ray Input</h2>
          <p className="text-xs text-slate-500">
            Upload an X-ray image for fracture classification (PNG, JPG, TIFF)
          </p>
        </div>
        {selectedFile && (
          <button
            onClick={handleClear}
            disabled={isProcessing}
            className="inline-flex items-center gap-1 rounded-md px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-50"
          >
            <X className="h-3.5 w-3.5" />
            Clear
          </button>
        )}
      </div>

      {validationError && (
        <div className="mb-4 flex items-center gap-2 rounded-lg bg-rose-50 p-3 text-xs text-rose-800 ring-1 ring-rose-200">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>{validationError}</span>
        </div>
      )}

      {!selectedFile ? (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors ${
            isDragOver
              ? 'border-cyan-500 bg-cyan-50/50'
              : 'border-slate-300 bg-slate-50/50 hover:border-slate-400 hover:bg-slate-100/50'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".png,.jpg,.jpeg,.tiff,.tif"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleFile(e.target.files[0]);
              }
            }}
          />
          <div className="rounded-full bg-white p-3 shadow-xs">
            <UploadCloud className="h-6 w-6 text-cyan-600" />
          </div>
          <p className="mt-3 text-sm font-medium text-slate-800">
            Click to upload or drag & drop X-ray image
          </p>
          <p className="mt-1 text-xs text-slate-500">
            PNG, JPG, TIFF up to 25MB • Temporary in-memory processing only
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex flex-col gap-4 rounded-lg border border-slate-200 bg-slate-50 p-4 sm:flex-row sm:items-center">
            {previewUrl && (
              <div className="relative h-28 w-28 shrink-0 overflow-hidden rounded-md border border-slate-300 bg-black">
                <img
                  src={previewUrl}
                  alt="X-ray preview"
                  className="h-full w-full object-contain"
                />
              </div>
            )}
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-900">
                <FileCheck className="h-4 w-4 text-cyan-600" />
                <span className="truncate">{selectedFile.name}</span>
              </div>
              <p className="mt-1 text-xs text-slate-500">
                Size: {(selectedFile.size / 1024).toFixed(1)} KB • Type: {selectedFile.type || 'image'}
              </p>
              <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-600">
                <span className="rounded bg-white px-2 py-0.5 border border-slate-200">
                  Target size: 224×224 px
                </span>
                <span className="rounded bg-white px-2 py-0.5 border border-slate-200">
                  Normalization: ImageNet stats
                </span>
              </div>
            </div>
          </div>

          {disabledReason && (
            <div className="flex items-start gap-2 rounded-lg bg-amber-50 p-3 text-xs text-amber-800 ring-1 ring-amber-200">
              <AlertCircle className="h-4 w-4 shrink-0 text-amber-600 mt-0.5" />
              <div>
                <p className="font-medium">Inference Restricted</p>
                <p className="mt-0.5 text-amber-700">{disabledReason}</p>
              </div>
            </div>
          )}

          <div className="flex flex-wrap gap-3">
            <button
              onClick={onRunScreening}
              disabled={isProcessing || !canRunInference}
              className="inline-flex flex-1 items-center justify-center rounded-lg bg-cyan-700 px-4 py-2.5 text-sm font-semibold text-white shadow-xs hover:bg-cyan-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isProcessing ? (
                <>
                  <span className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  Running ResNet-18 Screening...
                </>
              ) : (
                'Run Fracture Screening'
              )}
            </button>

            <button
              onClick={onRunExplanation}
              disabled={isProcessing || !canRunInference}
              className="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 shadow-xs hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Explain with Grad-CAM
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
