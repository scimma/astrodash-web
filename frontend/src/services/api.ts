import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000';

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
      rlap?: number;
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
}

export interface AnalysisOptionsResponse {
  sn_types: string[];
  age_bins?: string[];
  age_bins_by_type?: { [sn_type: string]: string[] };
}

export interface TemplateSpectrumResponse {
  wave: number[];
  flux: number[];
  sn_type: string;
  age_bin: string;
}

class Api {
  async processSpectrum(params: ProcessParams): Promise<ProcessResponse> {
    console.log('API: Making request to /process with params:', params);
    const formData = new FormData();

    // Add file if provided
    if (params.file) {
      console.log('API: Adding file to form data:', params.file.name);
      formData.append('file', params.file);
    }

    // Add other parameters
    const jsonParams = {
      smoothing: params.smoothing,
      knownZ: params.knownZ,
      zValue: params.zValue,
      minWave: params.minWave,
      maxWave: params.maxWave,
      calculateRlap: params.calculateRlap,
      oscRef: params.oscRef
    };

    console.log('API: Adding JSON params to form data:', jsonParams);
    formData.append('params', JSON.stringify(jsonParams));

    try {
      const response = await axios.post(`${API_BASE_URL}/process`, formData, {
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

  async getOSCReferences(): Promise<string[]> {
    const response = await axios.get(`${API_BASE_URL}/api/osc-references`);
    if (response.data.status === 'success') {
      return response.data.references;
    }
    throw new Error(response.data.message || 'Failed to fetch OSC references');
  }

  async getAnalysisOptions(): Promise<AnalysisOptionsResponse> {
    const response = await axios.get(`${API_BASE_URL}/api/analysis-options`);
    return response.data;
  }

  async getTemplateSpectrum(snType: string, age: string): Promise<TemplateSpectrumResponse> {
    const response = await axios.get(`${API_BASE_URL}/api/template-spectrum`, {
      params: {
        sn_type: snType,
        age_bin: age
      }
    });
    return response.data;
  }
}

export const api = new Api();
