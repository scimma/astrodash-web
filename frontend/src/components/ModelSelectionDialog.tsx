import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Card,
  CardContent,
  CardActionArea,
  Chip,
} from '@mui/material';
import { motion } from 'framer-motion';
import ScienceIcon from '@mui/icons-material/Science';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';

export type ModelType = 'dash' | 'transformer';

interface ModelSelectionDialogProps {
  open: boolean;
  onClose: () => void;
  onModelSelect: (model: ModelType) => void;
}

const ModelSelectionDialog: React.FC<ModelSelectionDialogProps> = ({
  open,
  onClose,
  onModelSelect,
}) => {
  const handleModelSelect = (model: ModelType) => {
    onModelSelect(model);
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: {
          background: 'linear-gradient(135deg, #1a237e 0%, #0d47a1 100%)',
          color: 'white',
          borderRadius: '16px',
        },
      }}
    >
      <DialogTitle sx={{ textAlign: 'center', pb: 1 }}>
        <Typography variant="h4" component="h2" sx={{ fontWeight: 600 }}>
          Choose Your Model
        </Typography>
        <Typography variant="body1" sx={{ opacity: 0.8, mt: 1 }}>
          Select the machine learning model for classification
        </Typography>
      </DialogTitle>

      <DialogContent sx={{ pt: 2 }}>
        <Box sx={{ display: 'flex', gap: 2, flexDirection: { xs: 'column', sm: 'row' } }}>
          {/* Dash Model Card */}
          <motion.div
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            style={{ flex: 1 }}
          >
            <Card
              sx={{
                background: 'rgba(255, 255, 255, 0.1)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                color: 'white',
                height: '100%',
              }}
            >
              <CardActionArea
                onClick={() => handleModelSelect('dash')}
                sx={{ height: '100%', p: 2 }}
              >
                <CardContent sx={{ textAlign: 'center' }}>
                  <ScienceIcon sx={{ fontSize: 48, mb: 2, color: '#4caf50' }} />
                  <Typography variant="h5" component="h3" sx={{ mb: 1, fontWeight: 600 }}>
                    Dash Model
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2, opacity: 0.8 }}>
                    Traditional CNN-based model with comprehensive template matching
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, justifyContent: 'center' }}>
                    <Chip
                      label="CNN"
                      size="small"
                      sx={{ backgroundColor: 'rgba(76, 175, 80, 0.2)', color: '#4caf50' }}
                    />
                    <Chip
                      label="Template Matching"
                      size="small"
                      sx={{ backgroundColor: 'rgba(76, 175, 80, 0.2)', color: '#4caf50' }}
                    />
                    <Chip
                      label="RLap Scores"
                      size="small"
                      sx={{ backgroundColor: 'rgba(76, 175, 80, 0.2)', color: '#4caf50' }}
                    />
                  </Box>
                </CardContent>
              </CardActionArea>
            </Card>
          </motion.div>

          {/* Transformer Model Card */}
          <motion.div
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            style={{ flex: 1 }}
          >
            <Card
              sx={{
                background: 'rgba(255, 255, 255, 0.1)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                color: 'white',
                height: '100%',
              }}
            >
              <CardActionArea
                onClick={() => handleModelSelect('transformer')}
                sx={{ height: '100%', p: 2 }}
              >
                <CardContent sx={{ textAlign: 'center' }}>
                  <AutoAwesomeIcon sx={{ fontSize: 48, mb: 2, color: '#ff9800' }} />
                  <Typography variant="h5" component="h3" sx={{ mb: 1, fontWeight: 600 }}>
                    Transformer Model
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2, opacity: 0.8 }}>
                    Advanced transformer-based model with 5-class classification
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, justifyContent: 'center' }}>
                    <Chip
                      label="Transformer"
                      size="small"
                      sx={{ backgroundColor: 'rgba(255, 152, 0, 0.2)', color: '#ff9800' }}
                    />
                    <Chip
                      label="5 Classes"
                      size="small"
                      sx={{ backgroundColor: 'rgba(255, 152, 0, 0.2)', color: '#ff9800' }}
                    />
                    <Chip
                      label="Fast Inference"
                      size="small"
                      sx={{ backgroundColor: 'rgba(255, 152, 0, 0.2)', color: '#ff9800' }}
                    />
                  </Box>
                </CardContent>
              </CardActionArea>
            </Card>
          </motion.div>
        </Box>
      </DialogContent>

      <DialogActions sx={{ p: 3, pt: 1 }}>
        <Button
          onClick={onClose}
          sx={{
            color: 'white',
            borderColor: 'rgba(255, 255, 255, 0.3)',
            '&:hover': {
              borderColor: 'white',
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
            },
          }}
        >
          Cancel
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ModelSelectionDialog;
