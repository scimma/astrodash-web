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
  TextField,
} from '@mui/material';
import { motion } from 'framer-motion';
import ScienceIcon from '@mui/icons-material/Science';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { api } from '../services/api';

export type ModelType = 'dash' | 'transformer' | { user: string }; // user: model_id

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
  const [userModels, setUserModels] = React.useState<any[]>([]);
  const [uploading, setUploading] = React.useState(false);
  const [uploadError, setUploadError] = React.useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = React.useState<any | null>(null);
  const [file, setFile] = React.useState<File | null>(null);
  const [classMapping, setClassMapping] = React.useState('');
  const [inputShape, setInputShape] = React.useState('');

  React.useEffect(() => {
    // Fetch user models (stub for now)
    api.getUserModels().then(setUserModels);
  }, [uploadSuccess]);

  const handleModelSelect = (model: ModelType) => {
    onModelSelect(model);
    onClose();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);
    try {
      const parsedClassMapping = JSON.parse(classMapping);
      const parsedInputShape = JSON.parse(inputShape);
      const result = await api.uploadModel({ file: file!, classMapping: parsedClassMapping, inputShape: parsedInputShape });
      setUploadSuccess(result);
      setFile(null);
      setClassMapping('');
      setInputShape('');
    } catch (err: any) {
      setUploadError(err?.response?.data?.message || err?.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
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

        {/* Upload New Model Card */}
        <Box sx={{ mt: 4 }}>
          <Card
            sx={{
              background: 'rgba(255, 255, 255, 0.1)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              color: 'white',
              minWidth: 280,
              maxWidth: 400,
              margin: '0 auto',
              mb: 3,
            }}
          >
            <CardContent sx={{ textAlign: 'center' }}>
              <CloudUploadIcon sx={{ fontSize: 48, mb: 2, color: '#2196f3' }} />
              <Typography variant="h5" component="h3" sx={{ mb: 1, fontWeight: 600 }}>
                Upload Your Model
              </Typography>
              <Typography variant="body2" sx={{ mb: 2, opacity: 0.8 }}>
                Upload a PyTorch .pth/.pt file, class mapping, and input shape
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, alignItems: 'center', mb: 1 }}>
                <Button
                  variant="contained"
                  component="label"
                  sx={{ mb: 1, background: '#2196f3', color: 'white' }}
                  startIcon={<CloudUploadIcon />}
                >
                  {file ? file.name : 'Choose Model File'}
                  <input
                    type="file"
                    accept=".pth,.pt"
                    hidden
                    onChange={handleFileChange}
                  />
                </Button>
                <TextField
                  label="Class Mapping (JSON)"
                  value={classMapping}
                  onChange={e => setClassMapping(e.target.value)}
                  size="small"
                  fullWidth
                  multiline
                  minRows={2}
                  sx={{ background: 'rgba(255,255,255,0.08)', borderRadius: 1, input: { color: 'white' }, label: { color: 'white' } }}
                />
                <TextField
                  label="Input Shape (JSON, e.g. [1,1024])"
                  value={inputShape}
                  onChange={e => setInputShape(e.target.value)}
                  size="small"
                  fullWidth
                  sx={{ background: 'rgba(255,255,255,0.08)', borderRadius: 1, input: { color: 'white' }, label: { color: 'white' } }}
                />
                <Button
                  variant="outlined"
                  color="primary"
                  onClick={handleUpload}
                  disabled={!file || !classMapping || !inputShape || uploading}
                  sx={{ mt: 1 }}
                >
                  {uploading ? 'Uploading...' : 'Upload Model'}
                </Button>
                {uploadError && <Typography color="error" sx={{ mt: 1 }}>{uploadError}</Typography>}
                {uploadSuccess && (
                  <Box sx={{ mt: 1, color: '#90ee90', fontSize: 14 }}>
                    Uploaded! Model ID: <b>{uploadSuccess.model_id}</b>
                    <br />Output shape: {JSON.stringify(uploadSuccess.output_shape)}
                    <br />Input shape: {JSON.stringify(uploadSuccess.input_shape)}
                    <Button
                      variant="contained"
                      color="success"
                      size="small"
                      sx={{ mt: 1 }}
                      onClick={() => handleModelSelect({ user: uploadSuccess.model_id })}
                    >
                      Use This Model
                    </Button>
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Box>

        {/* User-uploaded Models */}
        {userModels.length > 0 && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" sx={{ mb: 2, color: 'white', fontWeight: 500 }}>
              User-Uploaded Models
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              {userModels.map((model) => (
                <Card
                  key={model.model_id}
                  sx={{
                    background: 'rgba(255,255,255,0.08)',
                    color: 'white',
                    minWidth: 260,
                    maxWidth: 320,
                  }}
                >
                  <CardActionArea onClick={() => handleModelSelect({ user: model.model_id })}>
                    <CardContent>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                        Model ID: {model.model_id}
                      </Typography>
                      <Typography variant="body2" sx={{ opacity: 0.8, mb: 1 }}>
                        Filename: {model.model_filename}
                      </Typography>
                      <Typography variant="body2" sx={{ opacity: 0.8, mb: 1 }}>
                        Input Shape: {JSON.stringify(model.input_shape)}
                      </Typography>
                      <Typography variant="body2" sx={{ opacity: 0.8, mb: 1 }}>
                        Classes: {Object.keys(model.class_mapping).join(', ')}
                      </Typography>
                    </CardContent>
                  </CardActionArea>
                </Card>
              ))}
            </Box>
          </Box>
        )}
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
