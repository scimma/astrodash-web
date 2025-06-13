import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export interface SpectrumData {
  wavelength: number;
  flux: number;
}

export interface ProcessParams {
  smoothing: number;
  knownZ: boolean;
  zValue?: number;
  minWave: number;
  maxWave: number;
  classifyHost: boolean;
  calculateRlap: boolean;
}

export interface ProcessResponse {
  x: number[];
  y: number[];
  snType?: string;
  age?: string;
  hostType?: string;
  probability?: number;
  redshift?: number;
  rlap?: number;
}

export const api = {
  // Health check
  checkHealth: async () => {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  },

  // Get spectrum data
  getSpectrum: async (): Promise<SpectrumData[]> => {
    const response = await axios.get(`${API_BASE_URL}/spectrum`);
    return response.data;
  },

  // Process spectrum
  processSpectrum: async (params: ProcessParams): Promise<ProcessResponse> => {
    const response = await axios.post(`${API_BASE_URL}/process`, params);
    return response.data;
  },
};
