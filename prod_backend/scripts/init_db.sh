#!/bin/bash

echo "Initializing database..."

# Activate the conda environment
source ~/micromamba/etc/profile.d/micromamba.sh
micromamba activate astroweb

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

echo "Database initialization complete!"
