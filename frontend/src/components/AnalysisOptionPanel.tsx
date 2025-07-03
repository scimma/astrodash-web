import React from 'react';
import { Box, Typography, Paper } from '@mui/material';

interface AnalysisOptionPanelProps {
  title: string;
  controlRow?: React.ReactNode; // For checkboxes or switches in the header
  children: React.ReactNode;
  sx?: object;
}

const AnalysisOptionPanel: React.FC<AnalysisOptionPanelProps> = ({ title, controlRow, children, sx }) => (
  <Paper sx={{ p: 2, mb: 3, border: '1px solid #e0e0e0', borderRadius: 1, ...sx }}>
    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
      <Typography variant="h6">{title}</Typography>
      {controlRow}
    </Box>
    {children}
  </Paper>
);

export default AnalysisOptionPanel;
