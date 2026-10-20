import {
  HealthResponse,
  ModelStatusResponse,
  ScreeningResponse,
  ExplainResponse,
  EvaluationSummaryResponse,
} from '../types';

const API_BASE = '/api';

export class ApiError extends Error {
  status: number;
  isBackendOffline: boolean;

  constructor(message: string, status: number = 500, isBackendOffline: boolean = false) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.isBackendOffline = isBackendOffline;
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  try {
    const response = await fetch(url, options);

    if (!response.ok) {
      let detail = `Request failed with status ${response.status}`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          detail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
        } else if (errorJson.error) {
          detail = errorJson.error;
        }
      } catch {
        // Fallback to text or default
      }
      throw new ApiError(detail, response.status, false);
    }

    return (await response.json()) as T;
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      throw err;
    }

    const errorMsg = err instanceof Error ? err.message : 'Network error';
    // Typical fetch error when backend is offline or unreachable
    const isOffline = errorMsg.includes('Failed to fetch') || errorMsg.includes('NetworkError') || errorMsg.includes('Load failed');
    throw new ApiError(
      isOffline
        ? 'Cannot connect to the FastAPI backend (http://127.0.0.1:8000). Please start the backend service.'
        : errorMsg,
      0,
      isOffline
    );
  }
}

export const api = {
  getHealth: async (): Promise<HealthResponse> => {
    return request<HealthResponse>('/health');
  },

  getModelStatus: async (): Promise<ModelStatusResponse> => {
    return request<ModelStatusResponse>('/model/status');
  },

  screenImage: async (file: File): Promise<ScreeningResponse> => {
    const formData = new FormData();
    formData.append('file', file, file.name);
    return request<ScreeningResponse>('/screen', {
      method: 'POST',
      body: formData,
    });
  },

  explainImage: async (file: File, targetClassIdx?: number): Promise<ExplainResponse> => {
    const formData = new FormData();
    formData.append('file', file, file.name);
    const query = targetClassIdx !== undefined ? `?target_class_idx=${targetClassIdx}` : '';
    return request<ExplainResponse>(`/explain${query}`, {
      method: 'POST',
      body: formData,
    });
  },

  getEvaluationSummary: async (): Promise<EvaluationSummaryResponse> => {
    return request<EvaluationSummaryResponse>('/evaluation/summary');
  },
};
