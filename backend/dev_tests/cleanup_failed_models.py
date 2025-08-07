#!/usr/bin/env python3
"""
Script to cleanup failed user models from the database and file system.
"""

import os
import sys
import requests
import json
from pathlib import Path

# Add the prod_backend app to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../prod_backend'))

from app.infrastructure.storage.model_storage import ModelStorage
from app.config.settings import get_settings

API_URL = 'http://localhost:8000'

def list_models():
    """List all models from the API."""
    try:
        response = requests.get(f'{API_URL}/api/v1/models')
        if response.status_code == 200:
            models = response.json()
            print(f"Found {len(models)} models in database:")
            for model in models:
                print(f"  - ID: {model.get('id', 'N/A')}")
                print(f"    Name: {model.get('name', 'N/A')}")
                print(f"    Description: {model.get('description', 'N/A')}")
                print(f"    Owner: {model.get('owner', 'N/A')}")
                print()
            return models
        else:
            print(f"Failed to get models: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error getting models: {e}")
        return []

def list_storage_models():
    """List all models in the file storage."""
    try:
        settings = get_settings()
        storage = ModelStorage(settings.user_model_dir)
        model_ids = storage.list_models()
        print(f"Found {len(model_ids)} models in file storage:")
        for model_id in model_ids:
            print(f"  - {model_id}")
        return model_ids
    except Exception as e:
        print(f"Error listing storage models: {e}")
        return []

def delete_model(model_id):
    """Delete a model from the API."""
    try:
        response = requests.delete(f'{API_URL}/api/v1/models/{model_id}')
        if response.status_code == 200:
            print(f"✓ Successfully deleted model {model_id}")
            return True
        else:
            print(f"✗ Failed to delete model {model_id}: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error deleting model {model_id}: {e}")
        return False

def cleanup_storage_model(model_id):
    """Clean up model files from storage."""
    try:
        settings = get_settings()
        storage = ModelStorage(settings.user_model_dir)
        storage.cleanup_model_files(model_id)
        print(f"✓ Cleaned up storage files for {model_id}")
        return True
    except Exception as e:
        print(f"✗ Error cleaning up storage for {model_id}: {e}")
        return False

def main():
    print("=== User Model Cleanup Script ===\n")

    # List models in database
    print("1. Checking database models...")
    db_models = list_models()

    # List models in storage
    print("\n2. Checking file storage...")
    storage_models = list_storage_models()

    if not db_models and not storage_models:
        print("No models found to clean up.")
        return

    # Ask user which models to delete
    print("\n3. Select models to delete:")
    print("Enter model IDs separated by commas, or 'all' to delete all, or 'none' to cancel:")

    user_input = input().strip()

    if user_input.lower() == 'none':
        print("Cleanup cancelled.")
        return

    if user_input.lower() == 'all':
        models_to_delete = [model.get('id') for model in db_models if model.get('id')]
    else:
        models_to_delete = [mid.strip() for mid in user_input.split(',') if mid.strip()]

    if not models_to_delete:
        print("No models selected for deletion.")
        return

    # Confirm deletion
    print(f"\nAbout to delete {len(models_to_delete)} models:")
    for model_id in models_to_delete:
        print(f"  - {model_id}")

    confirm = input("\nAre you sure? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Deletion cancelled.")
        return

    # Delete models
    print("\n4. Deleting models...")
    success_count = 0

    for model_id in models_to_delete:
        print(f"\nDeleting {model_id}...")

        # Delete from database
        db_success = delete_model(model_id)

        # Clean up storage files
        storage_success = cleanup_storage_model(model_id)

        if db_success or storage_success:
            success_count += 1

    print(f"\n=== Cleanup Complete ===")
    print(f"Successfully processed {success_count}/{len(models_to_delete)} models.")

if __name__ == "__main__":
    main()
