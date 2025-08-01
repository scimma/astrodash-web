#!/bin/bash

# AstroDash Backend Deployment Script
set -e

echo "🚀 Starting AstroDash Backend Deployment..."

# Check if we're in the right directory
if [ ! -f "app/main.py" ]; then
    echo "❌ Error: Please run this script from the prod_backend directory"
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p storage/spectra storage/models logs app/astrodash_models/user_uploaded

# Set permissions
echo "🔐 Setting file permissions..."
chmod 755 storage logs
chmod 644 .env

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements/prod.txt

# Run database migrations
echo "🗄️ Running database migrations..."
alembic upgrade head

# Run tests
echo "🧪 Running tests..."
python -m pytest tests/test_api.py -v

# Build Docker image (optional)
if [ "$1" = "--docker" ]; then
    echo "🐳 Building Docker image..."
    docker build -t astrodash-backend .
    echo "✅ Docker image built successfully"
fi

echo "✅ Deployment completed successfully!"
echo "🎯 To start the server:"
echo "   micromamba activate astroweb && uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "🔗 API Documentation: http://localhost:8000/docs"
echo "🏥 Health Check: http://localhost:8000/health" 