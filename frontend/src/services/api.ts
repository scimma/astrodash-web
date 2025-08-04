import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export interface SpectrumData {
  x: number[];
  y: number[];
}

export interface ProcessParams {
  smoothing: number;
  knownZ: boolean;
  zValue?: number;
  minWave: number;
  maxWave: number;
  calculateRlap: boolean;
  file?: File;
  oscRef?: string;
  modelType?: 'dash' | 'transformer';
  model_id?: string;
}

export interface ProcessResponse {
  spectrum: {
    x: number[];
    y: number[];
    redshift?: number;
  };
  classification: {
    best_matches: Array<{
      type: string;
      age: string;
      probability: number;
      redshift?: number;
      rlap?: number | string;
      reliable: boolean;
    }>;
    best_match: {
      type: string;
      age: string;
      probability: number;
      redshift?: number;
    };
    reliable_matches: boolean;
  };
  model_type?: string;
}

export interface AnalysisOptionsResponse {
  sn_types: string[];
  age_bins?: string[];
  age_bins_by_type?: { [sn_type: string]: string[] };
}

export interface TemplateSpectrumResponse {
  x: number[];
  y: number[];
}

export interface LineListResponse {
  [element: string]: number[];
}

class Api {
  async processSpectrum(params: ProcessParams): Promise<ProcessResponse> {
    console.log('API: Making request to /api/v1/process with params:', params);
    const formData = new FormData();

    // Add file if provided
    if (params.file) {
      console.log('API: Adding file to form data:', params.file.name);
      formData.append('file', params.file);
    }

    // Add other parameters
    const jsonParams: any = {
      smoothing: params.smoothing,
      knownZ: params.knownZ,
      zValue: params.zValue,
      minWave: params.minWave,
      maxWave: params.maxWave,
      calculateRlap: params.calculateRlap,
      oscRef: params.oscRef,
    };
    if (params.modelType) {
      jsonParams.modelType = params.modelType;
    } else {
      jsonParams.modelType = 'dash';
    }

    console.log('API: Adding JSON params to form data:', jsonParams);
    formData.append('params', JSON.stringify(jsonParams));

    // Add model_id as separate Form parameter if provided
    if (params.model_id) {
      formData.append('model_id', params.model_id);
    }

    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log('API: Received response:', response.data);
    return response.data;
    } catch (error) {
      console.error('API: Error making request:', error);
      throw error;
    }
  }



  async getAnalysisOptions(): Promise<AnalysisOptionsResponse> {
    const response = await axios.get(`${API_BASE_URL}/api/v1/analysis-options`);
    return response.data;
  }

  async getTemplateSpectrum(snType: string, age: string): Promise<TemplateSpectrumResponse> {
    console.log(`API: Fetching template for ${snType} ${age}`);
    try {
      const response = await axios.get(`${API_BASE_URL}/api/v1/template-spectrum`, {
        params: {
          sn_type: snType,
          age_bin: age
        }
      });
      console.log(`API: Template response received:`, response.data);
      return response.data;
    } catch (error: any) {
      console.error(`API: Template fetch failed for ${snType} ${age}:`, {
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data,
        message: error.message
      });
      throw error;
    }
  }

  async getLineList(): Promise<LineListResponse> {
    const response = await axios.get(`${API_BASE_URL}/api/v1/line-list`);
    return response.data;
  }

  async batchProcess({ zipFile, params }: { zipFile: File; params: any }): Promise<any> {
    const formData = new FormData();
    formData.append('zip_file', zipFile);

    // Ensure modelType is included in params
    const paramsWithModel = {
      ...params,
      modelType: params.modelType || 'dash'  // Default to dash if not specified
    };

    formData.append('params', JSON.stringify(paramsWithModel));

    // Add model_id as separate Form parameter if provided
    if (params.model_id) {
      formData.append('model_id', params.model_id);
    }
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/batch-process`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('API: Error making batch-process request:', error);
      throw error;
    }
  }

  async batchProcessMultiple({ files, params }: { files: File[]; params: any }): Promise<any> {
    const formData = new FormData();

    // Add each file to the form data
    files.forEach((file, index) => {
      formData.append(`files`, file);
    });

    // Ensure modelType is included in params
    const paramsWithModel = {
      ...params,
      modelType: params.modelType || 'dash'  // Default to dash if not specified
    };

    formData.append('params', JSON.stringify(paramsWithModel));

    // Add model_id as separate Form parameter if provided
    if (params.model_id) {
      formData.append('model_id', params.model_id);
    }
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/batch-process-multiple`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('API: Error making batch-process-multiple request:', error);
      throw error;
    }
  }

  async uploadModel({ file, classMapping, inputShape }: { file: File, classMapping: object, inputShape: number[] }) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('class_mapping', JSON.stringify(classMapping));
    formData.append('input_shape', JSON.stringify(inputShape));
    const response = await axios.post(`${API_BASE_URL}/api/v1/upload-model`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }

  async getUserModels(): Promise<any[]> {
    // Fetch user-uploaded models from the backend
    try {
      const response = await axios.get(`${API_BASE_URL}/api/v1/models`);
      return response.data.models || response.data || [];
    } catch (error) {
      console.warn('Failed to fetch user models:', error);
      // Return empty array if models endpoint fails
      return [];
    }
  }
}

export const api = new Api();
