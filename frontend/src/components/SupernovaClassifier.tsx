import React, { useState, useEffect } from 'react';
import {
  Container,
  Box,
  Paper,
  Typography,
  Button,
  Slider,
  FormControlLabel,
  Checkbox,
  TextField,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Snackbar,
  Alert,
} from '@mui/material';
import Grid from '@mui/material/Grid';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip } from 'recharts';
import { api, ProcessParams, ProcessResponse } from '../services/api';
import { ResponsiveContainer } from 'recharts';

interface SupernovaClassifierProps {
  toggleColorMode: () => void;
  currentMode: 'light' | 'dark';
}

const SupernovaClassifier: React.FC<SupernovaClassifierProps> = ({ toggleColorMode, currentMode }) => {
  // State for file selection
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string>('Select SN File...');

  // State for priors
  const [knownZ, setKnownZ] = useState<boolean>(false);
  const [zValue, setZValue] = useState<string>('');
  const [minWave, setMinWave] = useState<string>('3500');
  const [maxWave, setMaxWave] = useState<string>('10000');
  const [smoothing, setSmoothing] = useState<number>(0);
  const [calculateRlap, setCalculateRlap] = useState<boolean>(false);

  // State for analysis
  const [snType, setSnType] = useState<string>('');
  const [age, setAge] = useState<string>('');
  const [spectrumData, setSpectrumData] = useState<ProcessResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Best matches state
  const [bestMatches, setBestMatches] = useState<Array<{
    type: string;
    age: string;
    probability: number;
    redshift?: number;
    rlap?: number;
    reliable: boolean;
  }>>([]);

  // Add OSC reference state
  const [oscRef, setOscRef] = useState<string>('');

  // Add state for dynamic dropdown options
  const [snTypeOptions, setSnTypeOptions] = useState<string[]>([]);
  const [ageOptions, setAgeOptions] = useState<string[]>([]);

  // Add state for template navigation
  const [currentMatchIndex, setCurrentMatchIndex] = useState<number>(0);

  // Add state for template spectrum
  type TemplateSpectrum = { wave: number[]; flux: number[]; sn_type: string; age_bin: string; };
  const [templateSpectrum, setTemplateSpectrum] = useState<TemplateSpectrum | null>(null);

  // Add state for error handling
  const [error, setError] = useState<string | null>(null);
  const [errorOpen, setErrorOpen] = useState(false);

  // Add useEffect to log spectrumData changes
  useEffect(() => {
    if (spectrumData) {
      console.log('spectrumData updated:', spectrumData);
      if (spectrumData.spectrum && spectrumData.spectrum.y) {
        console.log('First 10 flux values (y):', spectrumData.spectrum.y.slice(0, 10));
        console.log('Min flux:', Math.min(...spectrumData.spectrum.y));
        console.log('Max flux:', Math.max(...spectrumData.spectrum.y));
      }
    }
  }, [spectrumData]);

  // Fetch analysis options on mount
  useEffect(() => {
    api.getAnalysisOptions().then(res => {
      setSnTypeOptions(res.sn_types || []);
      setAgeOptions(res.age_bins || []);
    }).catch(error => {
      console.error("Failed to fetch analysis options:", error);
      setError("Failed to load classification data. Please try again later.");
      setErrorOpen(true);
    });
  }, []);

  // When dropdowns change, randomly select a best match and update analysis
  useEffect(() => {
    if (bestMatches.length > 0) {
      // Find all matches that match the selected SN type, age, and host
      const filtered = bestMatches.filter(m =>
        (!snType || m.type === snType) &&
        (!age || m.age === age)
      );
      if (filtered.length > 0) {
        // Randomly pick one
        const idx = Math.floor(Math.random() * filtered.length);
        setCurrentMatchIndex(idx);
        setSnType(filtered[idx].type);
        setAge(filtered[idx].age);
        // Optionally update plot/analysis here
      }
    }
  }, [snType, age, bestMatches]);

  // Fetch template spectrum when SN Type, Age, or Host changes
  useEffect(() => {
    if (snType && age) {
      api.getTemplateSpectrum(snType, age).then(res => {
        setTemplateSpectrum(res);
      }).catch(() => setTemplateSpectrum(null));
    } else {
      setTemplateSpectrum(null);
    }
  }, [snType, age]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setFileName(event.target.files[0].name);
      setOscRef('');
    }
  };

  const handleOscRefChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setOscRef(event.target.value);
    setSelectedFile(null);
  };

  const handleOscRefKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      handleProcess();
    }
  };

  const handleProcess = async () => {
    console.log('Starting process with params:', {
      smoothing,
      knownZ,
      zValue: knownZ ? parseFloat(zValue) : undefined,
      minWave: parseInt(minWave),
      maxWave: parseInt(maxWave),
      calculateRlap,
      file: selectedFile ? selectedFile.name : undefined,
      oscRef: oscRef || undefined
    });

    setLoading(true);
    setError(null);
    setErrorOpen(false);
    try {
      const params: ProcessParams = {
        smoothing,
        knownZ,
        zValue: knownZ ? parseFloat(zValue) : undefined,
        minWave: parseInt(minWave),
        maxWave: parseInt(maxWave),
        calculateRlap,
        file: selectedFile || undefined,
        oscRef: oscRef || undefined
      };

      console.log('Calling API with params:', params);
      const response = await api.processSpectrum(params);
      console.log('Received response:', response);

      // Type guard for error response
      const isErrorResponse = (resp: any): resp is { status?: string; error?: string; message?: string } => {
        return (
          (typeof resp === 'object') &&
          (
            (typeof resp.status === 'string' && resp.status === 'error') ||
            typeof resp.error === 'string' ||
            typeof resp.message === 'string'
          )
        );
      };

      if (isErrorResponse(response)) {
        setError(response.message || response.error || 'An unknown error occurred.');
        setErrorOpen(true);
        setLoading(false);
        return;
      }

      setSpectrumData(response);

      // Update best matches
      setBestMatches(response.classification.best_matches);

      // Update SN type and age from best match
      if (response.classification.best_match) {
        setSnType(response.classification.best_match.type);
        setAge(response.classification.best_match.age);
      }
    } catch (error: any) {
      // Try to extract error message from backend response
      let msg = 'An error occurred while processing the spectrum.';
      if (error.response && error.response.data) {
        msg = error.response.data.message || error.response.data.error || msg;
      } else if (error.message) {
        msg = error.message;
      }
      setError(msg);
      setErrorOpen(true);
      console.error('Error processing spectrum:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setFileName('Select SN File...');
    setSpectrumData(null);
    setBestMatches([]);
    setSnType('');
    setAge('');
    setError(null);
    setErrorOpen(false);
  };

  const handleDownload = () => {
    if (!spectrumData) return;

    const csvContent = "data:text/csv;charset=utf-8,"
      + "Wavelength,Flux\n"
      + spectrumData.spectrum.x.map((x: number, i: number) => `${x},${spectrumData.spectrum.y[i]}`).join("\n");

    // Create a download link
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "spectrum_data.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleHelp = () => {
    window.open('https://github.com/your-repo/astrodash-web/blob/main/README.md', '_blank');
  };

  return (
    <Container maxWidth="xl" sx={{ py: 2 }}>
      {/* Error Snackbar */}
      <Snackbar open={errorOpen} autoHideDuration={8000} onClose={() => setErrorOpen(false)} anchorOrigin={{ vertical: 'top', horizontal: 'center' }}>
        <Alert onClose={() => setErrorOpen(false)} severity="error" sx={{ width: '100%' }} variant="filled">
          {error}
        </Alert>
      </Snackbar>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" component="h1" gutterBottom>Supernova Classifier</Typography>
        <Box>
          <IconButton sx={{ ml: 1 }} onClick={toggleColorMode} color="inherit">
            {currentMode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
          </IconButton>
          <IconButton color="inherit" aria-label="help" onClick={handleHelp}>
            <HelpOutlineIcon />
          </IconButton>
        </Box>
      </Box>

      <Grid container spacing={2}>
        {/* Left Panel */}
        <Grid item xs={3}>
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>Select Spectrum</Typography>

            {/* Add OSC Reference Input */}
            <Box sx={{ mb: 2 }}>
              <TextField
                fullWidth
                label="OSC Reference"
                value={oscRef}
                onChange={handleOscRefChange}
                onKeyPress={handleOscRefKeyPress}
                placeholder="e.g., osc-sn2002er-8"
                size="small"
                sx={{ mb: 1 }}
              />
              <Typography variant="caption" color="text.secondary">
                Enter an OSC reference (e.g., osc-sn2002er-8) or upload a file
              </Typography>
            </Box>

            {/* Existing file upload UI */}
            <Box sx={{ mb: 2 }}>
              <input
                accept=".fits,.dat,.txt"
                style={{ display: 'none' }}
                id="file-input"
                type="file"
                onChange={handleFileSelect}
              />
              <Box sx={{ display: 'flex', gap: 1 }}>
                <label htmlFor="file-input">
                  <Button variant="contained" component="span" fullWidth>
                    Browse
                  </Button>
                </label>
                <Button
                  variant="outlined"
                  onClick={handleClear}
                  disabled={!selectedFile && !oscRef}
                >
                  Clear
                </Button>
              </Box>
              <Typography variant="body2" sx={{ mt: 1, textAlign: 'center' }}>
                {fileName}
              </Typography>
            </Box>
          </Paper>

          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Priors</Typography>
            <Box sx={{ mb: 2 }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={knownZ}
                    onChange={(e) => setKnownZ(e.target.checked)}
                  />
                }
                label="Known Redshift"
              />
              {knownZ && (
                <TextField
                  size="small"
                  value={zValue}
                  onChange={(e) => setZValue(e.target.value)}
                  sx={{ ml: 2 }}
                />
              )}
            </Box>

            <Box sx={{ mt: 2 }}>
              <Typography gutterBottom>Wavelength Range</Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  size="small"
                  label="Min wave"
                  value={minWave}
                  onChange={(e) => setMinWave(e.target.value)}
                  fullWidth
                />
                <TextField
                  size="small"
                  label="Max wave"
                  value={maxWave}
                  onChange={(e) => setMaxWave(e.target.value)}
                  fullWidth
                />
              </Box>
            </Box>

            <Box sx={{ mt: 2 }}>
              <Typography gutterBottom>Smooth</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <TextField
                  size="small"
                  value={smoothing}
                  onChange={(e) => setSmoothing(Number(e.target.value))}
                  sx={{ width: 60, mr: 2 }}
                />
                <Slider
                  value={typeof smoothing === 'number' ? smoothing : 0}
                  onChange={(_event, newValue) => setSmoothing(newValue as number)}
                  aria-labelledby="smooth-slider"
                  valueLabelDisplay="auto"
                  step={1}
                  marks
                  min={0}
                  max={50}
                />
              </Box>
            </Box>

            <FormControlLabel
              control={
                <Checkbox
                  checked={calculateRlap}
                  onChange={(e) => setCalculateRlap(e.target.checked)}
                />
              }
              label="Calculate rlap scores"
            />

            <Box sx={{ mt: 2 }}>
              <Button
                variant="contained"
                color="primary"
                onClick={handleProcess}
                disabled={loading || (!selectedFile && !oscRef)}
                fullWidth
                sx={{ py: 1.5 }}
              >
                {loading ? 'Processing...' : 'Process Spectrum'}
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* Right Panel */}
        <Grid item xs={9}>
          <Grid container spacing={2}>
            {/* Best Matches Section */}
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>Best Matches</Typography>
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <Box sx={{ width: '33%' }}>
                    <List>
                      {bestMatches.map((match, index) => (
                        <ListItem key={index}>
                          <ListItemText
                            primary={`${match.type} (${match.age})`}
                            secondary={`Prob: ${(match.probability * 100).toFixed(1)}%`}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                  <Box sx={{ width: '67%' }}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" gutterBottom>Best Match</Typography>
                      <Box sx={{ display: 'flex', justifyContent: 'space-around' }}>
                        <Box>
                          <Typography>SN Type</Typography>
                          <Typography variant="body2">{bestMatches[0]?.type || 'N/A'}</Typography>
                        </Box>
                        <Box>
                          <Typography>Age Range</Typography>
                          <Typography variant="body2">{bestMatches[0]?.age || 'N/A'}</Typography>
                        </Box>
                      </Box>
                    </Box>
                  </Box>
                </Box>
              </Paper>
            </Grid>

            {/* Analysis Section */}
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Analyse selection</Typography>
                  <Button
                    variant="outlined"
                    onClick={handleDownload}
                    disabled={!spectrumData}
                  >
                    Save
                  </Button>
                </Box>
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <FormControl size="small" sx={{ minWidth: 120 }}>
                      <InputLabel>SN Type</InputLabel>
                      <Select value={snType} onChange={(e) => setSnType(e.target.value)}>
                        {snTypeOptions.map((type) => (
                          <MenuItem key={type} value={type}>
                            {type}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <FormControl size="small" sx={{ minWidth: 120 }}>
                      <InputLabel>Age</InputLabel>
                      <Select value={age} onChange={(e) => setAge(e.target.value)}>
                        {ageOptions.map((ageBin) => (
                          <MenuItem key={ageBin} value={ageBin}>
                            {ageBin}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Box>
                </Box>
                <Box>
                  {spectrumData && (
                    <div className="mt-8">
                      <h2 className="text-xl font-bold mb-4">Spectrum Plot</h2>
                      <div className="bg-white p-4 rounded-lg shadow">
                        <ResponsiveContainer width="100%" height={400}>
                          <LineChart
                            key={spectrumData ? `spectrum-${spectrumData.spectrum.x[0]}` : 'no-spectrum'}
                            data={spectrumData.spectrum.x.map((x: number, i: number) => ({
                              x,
                              y: spectrumData.spectrum.y[i],
                              template: templateSpectrum && templateSpectrum.wave[i] === x ? templateSpectrum.flux[i] : undefined
                            }))}
                            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                              dataKey="x"
                              label={{ value: 'Wavelength (Å)', position: 'insideBottom', offset: -5 }}
                            />
                            <YAxis
                              label={{ value: 'Flux', angle: -90, position: 'insideLeft' }}
                            />
                            <RechartsTooltip
                              formatter={(value: number, name: string, props: any) => [`${props.payload.y.toFixed(6)}`, 'Flux']}
                              labelFormatter={(label: number) => `Wavelength: ${label.toFixed(2)} Å`}
                            />
                            <Line type="monotone" dataKey="y" stroke="#8884d8" dot={false} name="Observed" />
                            {templateSpectrum && (
                              <Line
                                type="monotone"
                                dataKey="template"
                                stroke="#d62728"
                                dot={false}
                                name="Template"
                                isAnimationActive={false}
                                connectNulls
                              />
                            )}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Container>
  );
};

export default SupernovaClassifier;
