import React, { useState, useMemo } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

import LandingPage from './components/LandingPage';
import SupernovaClassifier from './components/SupernovaClassifier';

function App() {
  const [mode, setMode] = useState<'light' | 'dark'>('light');

  const colorMode = useMemo(
    () => ({
      toggleColorMode: () => {
        setMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'));
      },
    }),
    [],
  );

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode,
          primary: { main: mode === 'light' ? '#1976d2' : '#90caf9' },
          secondary: { main: mode === 'light' ? '#dc004e' : '#f48fb1' },
          background: {
            default: mode === 'light' ? '#f5f5f5' : '#121212',
            paper: mode === 'light' ? '#fff' : '#1e1e1e',
          },
        },
        typography: {
          fontFamily: 'Roboto, Arial, sans-serif',
          h1: { fontSize: '2.5rem' },
          h2: { fontSize: '2rem' },
          h3: { fontSize: '1.75rem' },
          h4: { fontSize: '1.5rem' },
          h5: { fontSize: '1.25rem' },
          h6: { fontSize: '1rem' },
        },
        shape: {
          borderRadius: 8,
        },
      }),
    [mode],
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline /> {/* CssBaseline handles baseline styles and adapts to theme.palette.mode */}
      <Router>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route
            path="/classify"
            element={<SupernovaClassifier
              toggleColorMode={colorMode.toggleColorMode}
              currentMode={mode}
            />}
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
