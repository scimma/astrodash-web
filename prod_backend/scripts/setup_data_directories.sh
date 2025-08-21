#!/bin/bash

# Setup script for external data directories
# This separates application code from data storage for production readiness

echo "Setting up external data directories for AstroDash..."

# Create main data directory
sudo mkdir -p /data
sudo chown $USER:$USER /data
sudo chmod 755 /data

# Create subdirectories
mkdir -p /data/user_models
mkdir -p /data/pre_trained_models/dash
mkdir -p /data/pre_trained_models/transformer
mkdir -p /data/pre_trained_models/templates
mkdir -p /data/logs

echo "Created directory structure:"
echo "  /data/"
echo "  ├── user_models/"
echo "  ├── pre_trained_models/"
echo "  │   ├── dash/"
echo "  │   ├── transformer/"
echo "  │   └── templates/"
echo "  └── logs/"

echo ""
echo "Next steps:"
echo "1. Move existing model files from app/astrodash_models/ to /data/"
echo "2. Update environment variables or .env file with new paths"
echo "3. Test the application with new paths"
echo ""
echo "Example environment variables:"
echo "export DATA_DIR=/data"
echo "export USER_MODEL_DIR=/data/user_models"
echo "export DASH_MODEL_PATH=/data/pre_trained_models/dash/pytorch_model.pth"
echo "export DASH_TRAINING_PARAMS_PATH=/data/pre_trained_models/dash/training_params.pickle"
echo "export TRANSFORMER_MODEL_PATH=/data/pre_trained_models/transformer/TF_wiserep_v6.pt"
echo "export TEMPLATE_PATH=/data/pre_trained_models/templates/snand_host_templates.npz"
echo "export LINE_LIST_PATH=/data/pre_trained_models/templates/sneLineList.txt"
