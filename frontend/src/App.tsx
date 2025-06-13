import React, { useState } from 'react';
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
  Divider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tooltip,
} from '@mui/material';
import Grid from '@mui/material/Grid';
import HelpIcon from '@mui/icons-material/Help';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ReferenceLine } from 'recharts';
import { api, ProcessParams, ProcessResponse } from './services/api';
import './App.css';

function App() {
  // State for file selection
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string>('Select SN File...');

  // State for priors
  const [knownZ, setKnownZ] = useState<boolean>(true);
  const [zValue, setZValue] = useState<string>('0');
  const [classifyHost, setClassifyHost] = useState<boolean>(false);
  const [minWave, setMinWave] = useState<string>('3000');
  const [maxWave, setMaxWave] = useState<string>('10000');
  const [smoothing, setSmoothing] = useState<number>(6);
  const [calculateRlap, setCalculateRlap] = useState<boolean>(false);

  // State for analysis
  const [snType, setSnType] = useState<string>('');
  const [age, setAge] = useState<string>('');
  const [hostType, setHostType] = useState<string>('No Host');
  const [hostFraction, setHostFraction] = useState<number>(0);
  const [redshift, setRedshift] = useState<number>(0);
  const [spectrumData, setSpectrumData] = useState<ProcessResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Best matches state
  const [bestMatches, setBestMatches] = useState<Array<{
    type: string;
    age: string;
    host: string;
    probability: number;
    redshift?: number;
    rlap?: number;
  }>>([]);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setFileName(event.target.files[0].name);
    }
  };

  const handleProcess = async () => {
    setLoading(true);
    try {
      const params: ProcessParams = {
        smoothing,
        knownZ,
        zValue: knownZ ? parseFloat(zValue) : undefined,
        minWave: parseInt(minWave),
        maxWave: parseInt(maxWave),
        classifyHost,
        calculateRlap
      };
      const data = await api.processSpectrum(params);
      setSpectrumData(data);
      // Mock best matches data
      setBestMatches([
        { type: 'Ia-norm', age: '-20 to -18', host: 'No Host', probability: 0.95, redshift: 0.5, rlap: 0.8 },
        { type: 'Ia-91bg', age: '-15 to -13', host: 'No Host', probability: 0.85, redshift: 0.48, rlap: 0.75 },
      ]);
    } catch (error) {
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
    setHostType('No Host');
    setHostFraction(0);
    setRedshift(0);
  };

  const handleSave = () => {
    if (!spectrumData) return;

    // Create a CSV string with the spectrum data
    const csvContent = "data:text/csv;charset=utf-8,"
      + "Wavelength,Flux\n"
      + spectrumData.x.map((x, i) => `${x},${spectrumData.y[i]}`).join("\n");

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
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
        <Tooltip title="Help">
          <IconButton onClick={handleHelp}>
            <HelpIcon />
          </IconButton>
        </Tooltip>
      </Box>

      <Grid container spacing={2}>
        {/* Left Panel */}
        <Grid item xs={3}>
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>Select Spectrum</Typography>
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
                  disabled={!selectedFile}
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

            <FormControlLabel
              control={
                <Checkbox
                  checked={classifyHost}
                  onChange={(e) => setClassifyHost(e.target.checked)}
                />
              }
              label="Classify Host"
            />

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
                  value={smoothing}
                  onChange={(_, value) => setSmoothing(value as number)}
                  min={0}
                  max={20}
                  sx={{ flex: 1 }}
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
                onClick={handleProcess}
                disabled={loading || !selectedFile}
                fullWidth
              >
                {loading ? 'Processing...' : 'Fit with priors'}
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
                            secondary={`Host: ${match.host}, Prob: ${(match.probability * 100).toFixed(1)}%`}
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
                          <Typography>Host Type</Typography>
                          <Typography variant="body2">{bestMatches[0]?.host || 'N/A'}</Typography>
                        </Box>
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
                    onClick={handleSave}
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
                        <MenuItem value="Ia-norm">Ia-norm</MenuItem>
                        <MenuItem value="Ia-91bg">Ia-91bg</MenuItem>
                      </Select>
                    </FormControl>
                    <FormControl size="small" sx={{ minWidth: 120 }}>
                      <InputLabel>Age</InputLabel>
                      <Select value={age} onChange={(e) => setAge(e.target.value)}>
                        <MenuItem value="-20 to -18">-20 to -18</MenuItem>
                        <MenuItem value="-15 to -13">-15 to -13</MenuItem>
                      </Select>
                    </FormControl>
                    <FormControl size="small" sx={{ minWidth: 120 }}>
                      <InputLabel>Host</InputLabel>
                      <Select value={hostType} onChange={(e) => setHostType(e.target.value)}>
                        <MenuItem value="No Host">No Host</MenuItem>
                        <MenuItem value="E">E</MenuItem>
                        <MenuItem value="S0">S0</MenuItem>
                      </Select>
                    </FormControl>
                  </Box>
                </Box>
                <Box>
                  {spectrumData && (
                    <LineChart
                      width={800}
                      height={400}
                      data={spectrumData.x.map((x, i) => ({
                        x,
                        y: spectrumData.y[i]
                      }))}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="x" />
                      <YAxis />
                      <RechartsTooltip />
                      <Legend />
                      <Line type="monotone" dataKey="y" stroke="#8884d8" />
                      <ReferenceLine x={redshift} stroke="red" label="z" />
                    </LineChart>
                  )}
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Container>
  );
}

export default App;
