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
          // You can customize more palette properties here for light/dark mode
          // For example:
          // primary: { main: mode === 'light' ? '#1976d2' : '#90caf9' },
          // secondary: { main: mode === 'light' ? '#dc004e' : '#f48fb1' },
        },
      }),
    [mode],
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline /> {/* CssBaseline handles baseline styles and adapts to theme.palette.mode */}
      {/* The theme toggle button will be moved into SupernovaClassifier for specific placement */}
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
