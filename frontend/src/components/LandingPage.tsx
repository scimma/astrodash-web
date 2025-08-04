import React, { useState } from 'react';
import { Box, Button, Container, Typography, useTheme } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import ModelSelectionDialog, { ModelType } from './ModelSelectionDialog';

const LandingPage: React.FC = () => {
  const theme = useTheme();
  const navigate = useNavigate();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [pendingAction, setPendingAction] = useState<'classify' | 'batch' | null>(null);

  const handleButtonClick = (action: 'classify' | 'batch') => {
    setPendingAction(action);
    setDialogOpen(true);
  };

  const handleModelSelect = (model: ModelType) => {
    // Persist model selection in localStorage
    localStorage.setItem('astrodash_selected_model', JSON.stringify(model));
    if (pendingAction === 'classify') {
      navigate('/classify', { state: { model } });
    } else if (pendingAction === 'batch') {
      navigate('/batch', { state: { model } });
    }
  };

  const handleDialogClose = () => {
    setDialogOpen(false);
    setPendingAction(null);
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #1a237e 0%, #0d47a1 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'white',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <Container maxWidth="md">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <Typography
            variant="h1"
            component="h1"
            sx={{
              fontSize: { xs: '3rem', md: '4.5rem' },
              fontWeight: 700,
              textAlign: 'center',
              mb: 2,
              textShadow: '2px 2px 4px rgba(0,0,0,0.3)',
            }}
          >
            Welcome to Astrodash
          </Typography>

          <Typography
            variant="h5"
            sx={{
              textAlign: 'center',
              mb: 6,
              opacity: 0.9,
              maxWidth: '800px',
              mx: 'auto',
            }}
          >
            Poweful tool to classify SN spectra using machine learning
          </Typography>

          <Box
            sx={{
              display: 'flex',
              justifyContent: 'center',
              mt: 4,
              gap: 2,
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Button
                variant="contained"
                size="large"
                onClick={() => handleButtonClick('classify')}
                sx={{
                  backgroundColor: '#2196f3',
                  color: 'white',
                  fontSize: '1.2rem',
                  padding: '12px 32px',
                  borderRadius: '30px',
                  boxShadow: '0 4px 20px rgba(33, 150, 243, 0.3)',
                  '&:hover': {
                    backgroundColor: '#1976d2',
                    boxShadow: '0 6px 25px rgba(33, 150, 243, 0.4)',
                  },
                }}
              >
                Classify Supernovae
              </Button>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Button
                variant="outlined"
                size="large"
                onClick={() => handleButtonClick('batch')}
                sx={{
                  color: 'white',
                  borderColor: '#2196f3',
                  fontSize: '1.2rem',
                  padding: '12px 32px',
                  borderRadius: '30px',
                  mt: 2,
                  '&:hover': {
                    backgroundColor: '#1976d2',
                    borderColor: '#1976d2',
                  },
                }}
              >
                Batch Classify
              </Button>
            </motion.div>
          </Box>
        </motion.div>
      </Container>

      <ModelSelectionDialog
        open={dialogOpen}
        onClose={handleDialogClose}
        onModelSelect={handleModelSelect}
      />
    </Box>
  );
};

export default LandingPage;
