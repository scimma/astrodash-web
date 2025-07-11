#!/usr/bin/env python3

import os
import sys
import torch
import logging

# Add the backend directory to the path
sys.path.append('./backend')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_transformer_model():
    """Test loading the transformer model"""
    try:
        # Check if model file exists
        model_path = "./backend/astrodash_models/yuqing_models/TF_wiserep_v6.pt"
        print(f"Checking if model file exists: {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")

        if os.path.exists(model_path):
            print(f"File size: {os.path.getsize(model_path)} bytes")

        # Try to load the model using the proper class
        print("\nAttempting to load model with spectraTransformerEncoder...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Import the model class
        from app.services.transformer_model import spectraTransformerEncoder

        # Model hyperparameters
        bottleneck_length = 1  # Changed from 8 to 1 based on saved model
        model_dim = 128
        num_heads = 4
        num_layers = 6  # Changed from 4 to 6 based on saved model (has blocks 0-5)
        num_classes = 5
        ff_dim = 256
        dropout = 0.1
        selfattn = False

        # Initialize the model
        model = spectraTransformerEncoder(
            bottleneck_length=bottleneck_length,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            ff_dim=ff_dim,
            dropout=dropout,
            selfattn=selfattn
        ).to(device)

        # Load the state dict
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        print(f"Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

        # Test with dummy input
        print("\nTesting with dummy input...")
        try:
            wavelength = torch.randn(1, 1024, dtype=torch.float32).to(device)
            flux = torch.randn(1, 1024, dtype=torch.float32).to(device)
            redshift = torch.tensor([[0.1]], dtype=torch.float32).to(device)

            print(f"Input shapes - wavelength: {wavelength.shape}, flux: {flux.shape}, redshift: {redshift.shape}")

            with torch.no_grad():
                output = model(wavelength, flux, redshift)
                print(f"Model output shape: {output.shape}")
                print(f"Model output: {output}")

                # Test softmax
                probs = torch.softmax(output, dim=-1)
                print(f"Probabilities: {probs}")
                print(f"Sum of probabilities: {probs.sum()}")

        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_transformer_model()
