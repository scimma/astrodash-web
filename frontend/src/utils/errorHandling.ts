// Error handling utility for frontend
export interface ApiError {
  detail?: string;
  message?: string;
  status_code?: number;
}

export interface ErrorResponse {
  detail?: string;
  message?: string;
  status?: string;
  error?: string;
}

/**
 * Extract error message from various error response formats
 */
export const extractErrorMessage = (error: any): string => {
  console.error('Error details:', error);

  // Handle different error response formats
  if (error?.message) {
    return error.message;
  }

  if (error?.response?.data?.detail) {
    return error.response.data.detail;
  }

  if (error?.response?.data?.message) {
    return error.response.data.message;
  }

  if (error?.response?.data?.error) {
    return error.response.data.error;
  }

  if (error?.detail) {
    return error.detail;
  }

  if (error?.error) {
    return error.error;
  }

  // Default error message
  return 'An unexpected error occurred. Please try again.';
};

/**
 * Get user-friendly error message based on error type
 */
export const getUserFriendlyErrorMessage = (error: any): string => {
  const errorMessage = extractErrorMessage(error);

  // Map specific error messages to user-friendly versions
  const errorMappings: { [key: string]: string } = {
    'No spectrum file or OSC reference provided': 'Please provide either a spectrum file or an OSC reference.',
    'File not found': 'The requested file could not be found. Please check the file path and try again.',
    'Model with ID': 'The requested model could not be found. Please check the model ID and try again.',
    'Template not found': 'The requested template could not be found. Please check the SN type and age bin.',
    'Line list file not found': 'The line list file is not available. Please contact support.',
    'Element not found in line list': 'The requested element could not be found in the line list.',
    'Classification failed': 'The classification process failed. Please check your input and try again.',
    'Spectrum processing failed': 'The spectrum processing failed. Please check your input and try again.',
    'Batch processing failed': 'The batch processing failed. Please check your files and try again.',
    'Validation failed': 'The input validation failed. Please check your input and try again.',
    'Model validation failed': 'The model validation failed. Please check your model and try again.',
    'File validation failed': 'The file validation failed. Please check your file format and try again.',
    'Storage error': 'A storage error occurred. Please try again later.',
    'Configuration error': 'A configuration error occurred. Please contact support.',
    'OSC service error': 'The OSC service is currently unavailable. Please try again later.',
    'Model with name already exists': 'A model with this name already exists. Please choose a different name.',
    'Unknown model type': 'The specified model type is not supported.',
    'Redshift value is required for Transformer model': 'Redshift value is required for Transformer model. Please enter a valid redshift.',
    'No files provided for batch processing': 'Please select at least one file for batch processing.',
    'Cannot provide both zip_file and files': 'Please provide either a zip file or individual files, not both.',
    'Must provide either zip_file or files parameter': 'Please provide either a zip file or individual files.',
    'Invalid files type': 'Invalid file type provided. Please check your files and try again.',
    'No valid spectrum data found in file': 'No valid spectrum data found in the uploaded file. Please check the file format.',
    'Unsupported file format': 'The file format is not supported. Please use .dat, .lnw, .txt, or .fits files.',
    'File extension not allowed': 'The file extension is not allowed. Please use .dat, .lnw, .txt, or .fits files.',
    'Wavelengths must be positive values': 'Wavelength values must be positive.',
    'Minimum wavelength must be less than or equal to maximum wavelength': 'Minimum wavelength must be less than or equal to maximum wavelength.',
    'Owner cannot be empty': 'Owner field cannot be empty.',
    'No updates provided': 'No updates were provided. Please make changes before saving.',
    'Model storage not available': 'Model storage is not available. Please contact support.',
    'Failed to get model info': 'Failed to retrieve model information. Please try again.',
    'Failed to delete model': 'Failed to delete the model. Please try again.',
    'Failed to update model': 'Failed to update the model. Please try again.',
    'Failed to list models': 'Failed to retrieve the list of models. Please try again.',
    'Failed to upload model': 'Failed to upload the model. Please check your model file and try again.',
    'Failed to load classification data': 'Failed to load classification data. Please try again later.',
    'Failed to fetch analysis options': 'Failed to fetch analysis options. Please try again later.',
    'Failed to fetch template statistics': 'Failed to fetch template statistics. Please try again later.',
    'Failed to process spectrum': 'Failed to process the spectrum. Please try again.',
    'Failed to estimate redshift': 'Failed to estimate redshift. Please try again.',
    'Failed to load line list': 'Failed to load the line list. Please try again later.',
    'Failed to get available elements': 'Failed to get available elements. Please try again later.',
    'Failed to get wavelengths for element': 'Failed to get wavelengths for the specified element. Please try again.',
    'Failed to filter line list': 'Failed to filter the line list. Please try again.',
    'Internal server error': 'An internal server error occurred. Please try again later.',
    'Request timed out': 'The request timed out. Please try again.',
    'Connection error': 'A connection error occurred. Please check your internet connection and try again.',
  };

  // Check for exact matches first
  for (const [key, value] of Object.entries(errorMappings)) {
    if (errorMessage.includes(key)) {
      return value;
    }
  }

  // Return the original error message if no mapping found
  return errorMessage;
};

/**
 * Check if error is a specific type
 */
export const isErrorType = (error: any, errorType: string): boolean => {
  const errorMessage = extractErrorMessage(error);
  return errorMessage.toLowerCase().includes(errorType.toLowerCase());
};

/**
 * Get error severity level
 */
export const getErrorSeverity = (error: any): 'error' | 'warning' | 'info' => {
  const errorMessage = extractErrorMessage(error);

  // Warning level errors
  const warningErrors = [
    'validation failed',
    'file validation failed',
    'model validation failed',
    'wavelengths must be positive',
    'minimum wavelength must be less than',
    'no updates provided',
    'owner cannot be empty'
  ];

  // Info level errors
  const infoErrors = [
    'no spectrum file or osc reference provided',
    'redshift value is required',
    'no files provided',
    'cannot provide both',
    'must provide either'
  ];

  const lowerMessage = errorMessage.toLowerCase();

  if (warningErrors.some(warning => lowerMessage.includes(warning))) {
    return 'warning';
  }

  if (infoErrors.some(info => lowerMessage.includes(info))) {
    return 'info';
  }

  return 'error';
};
