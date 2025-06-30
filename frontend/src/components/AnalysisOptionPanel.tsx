import React from 'react';
import { Box, Typography } from '@mui/material';

interface AnalysisOptionPanelProps {
  title: string;
  children: React.ReactNode;
  sx?: object;
}

const AnalysisOptionPanel: React.FC<AnalysisOptionPanelProps> = ({ title, children, sx }) => (
  <Box sx={{ mb: 3, p: 2, border: '1px solid #e0e0e0', borderRadius: 1, ...sx }}>
    <Typography variant="h6" sx={{ mb: 2 }}>{title}</Typography>
    {children}
  </Box>
);

export default AnalysisOptionPanel;
