#!/bin/bash

# Migration script to move model files from app/astrodash_models to external /data directory
# This separates application code from data storage for production readiness

echo "Migrating model files to external storage..."

# Check if /data directory exists
if [ ! -d "/data" ]; then
    echo "Error: /data directory does not exist. Run setup_data_directories.sh first."
    exit 1
fi

# Source and destination directories
SOURCE_DIR="app/astrodash_models"
DEST_DIR="/data"

echo "Moving files from $SOURCE_DIR to $DEST_DIR..."

# Move DASH model files
if [ -d "$SOURCE_DIR/zeroZ" ]; then
    echo "Moving DASH model files..."
    cp -r "$SOURCE_DIR/zeroZ" "$DEST_DIR/pre_trained_models/dash/"
    echo "✓ DASH model files moved to $DEST_DIR/pre_trained_models/dash/"
else
    echo "⚠ DASH model directory not found at $SOURCE_DIR/zeroZ"
fi

if [ -d "$SOURCE_DIR/agnosticZ" ]; then
    echo "Moving agnosticZ DASH model files..."
    cp -r "$SOURCE_DIR/agnosticZ" "$DEST_DIR/pre_trained_models/dash/"
    echo "✓ agnosticZ DASH model files moved to $DEST_DIR/pre_trained_models/dash/"
else
    echo "⚠ agnosticZ DASH model directory not found at $SOURCE_DIR/agnosticZ"
fi

# Move Transformer model files
if [ -d "$SOURCE_DIR/yuqing_models" ]; then
    echo "Moving Transformer model files..."
    cp -r "$SOURCE_DIR/yuqing_models" "$DEST_DIR/pre_trained_models/transformer/"
    echo "✓ Transformer model files moved to $DEST_DIR/pre_trained_models/transformer/"
else
    echo "⚠ Transformer model directory not found at $SOURCE_DIR/yuqing_models"
fi

# Move template files
if [ -f "$SOURCE_DIR/sn_and_host_templates.npz" ]; then
    echo "Moving template files..."
    cp "$SOURCE_DIR/sn_and_host_templates.npz" "$DEST_DIR/pre_trained_models/templates/"
    echo "✓ Template files moved to $DEST_DIR/pre_trained_models/templates/"
else
    echo "⚠ Template file not found at $SOURCE_DIR/sn_and_host_templates.npz"
fi

if [ -f "$SOURCE_DIR/sneLineList.txt" ]; then
    cp "$SOURCE_DIR/sneLineList.txt" "$DEST_DIR/pre_trained_models/templates/"
    echo "✓ Line list file moved to $DEST_DIR/pre_trained_models/templates/"
else
    echo "⚠ Line list file not found at $SOURCE_DIR/sneLineList.txt"
fi

# Move user uploaded models (if any exist)
if [ -d "$SOURCE_DIR/user_uploaded" ] && [ "$(ls -A $SOURCE_DIR/user_uploaded)" ]; then
    echo "Moving user uploaded models..."
    cp -r "$SOURCE_DIR/user_uploaded"/* "$DEST_DIR/user_models/"
    echo "✓ User uploaded models moved to $DEST_DIR/user_models/"
else
    echo "ℹ No user uploaded models to move"
fi

echo ""
echo "Migration completed!"
echo ""
echo "Verification:"
echo "DASH models: $(ls -la $DEST_DIR/pre_trained_models/dash/ 2>/dev/null | wc -l) files"
echo "Transformer models: $(ls -la $DEST_DIR/pre_trained_models/transformer/ 2>/dev/null | wc -l) files"
echo "Templates: $(ls -la $DEST_DIR/pre_trained_models/templates/ 2>/dev/null | wc -l) files"
echo "User models: $(ls -la $DEST_DIR/user_models/ 2>/dev/null | wc -l) files"
echo ""
echo "Next steps:"
echo "1. Test the application with new paths"
echo "2. Once confirmed working, you can remove the old app/astrodash_models directory"
echo "3. Update your environment variables or .env file"
echo ""
echo "⚠ IMPORTANT: Keep the old files until you verify everything works!"
