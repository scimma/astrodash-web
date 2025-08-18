import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import scipy.interpolate


# Hard-coded Astrodash training parameters
redshifting = True  # Enable redshifting augmentation
minZ = 0
maxZ = 0.8

# Training parameters from original Astrodash
w0 = 3500.  # wavelength range in Angstroms
w1 = 10000.
nw = 1024  # number of wavelength bins
nIndexes = np.arange(0, int(nw))
dwlog = np.log(w1 / w0) / nw
outerVal = 0.5

class CNNSpectraClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),  # (B, 32, 1024)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.MaxPool1d(2),  # (B, 32, 512)
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),  # (B, 64, 512)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.MaxPool1d(2),  # (B, 64, 256)
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),  # (B, 128, 256)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.MaxPool1d(2),  # (B, 128, 128)
        )
        # Reduce the size of fully connected layers significantly
        self.fc1 = nn.Linear(128 * 128, 512)  # Reduced from 256
        self.fc2 = nn.Linear(512, 256)        # Reduced from 128
        self.fc_out = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, flux):
        # flux: (B, 1024)
        x = flux.unsqueeze(1)  # (B, 1, 1024)
        x = self.cnn(x)        # (B, 128, 128)
        x = x.view(x.size(0), -1)  # (B, 128*128)
        x = F.relu(self.fc1(x))    # (B, 512)
        x = self.dropout(x)        # Add dropout
        x = F.relu(self.fc2(x))    # (B, 256)
        x = self.dropout(x)        # Add dropout
        logits = self.fc_out(x)    # (B, num_classes)
        return logits

# Helper: accuracy calculation
def accuracy_fn(logits, labels):
    preds = torch.argmax(logits, 1)
    return torch.mean((preds == labels).float()).item()

# Ported exact Astrodash redshifting functions
def min_max_index(flux, outerVal=0):
    """Find min and max indices where flux != outerVal (Astrodash exact)"""
    nonZeros = np.where(flux != outerVal)[0]
    if nonZeros.size:
        minIndex, maxIndex = min(nonZeros), max(nonZeros)
    else:
        minIndex, maxIndex = len(flux), len(flux)
    return minIndex, maxIndex

def apodize(flux, minindex, maxindex, nw, outerVal=0):
    """Apodize with 5% cosine bell (Astrodash exact)"""
    percent = 0.05
    fluxout = np.copy(flux) - outerVal

    nsquash = int(nw * percent)
    for i in range(0, nsquash):
        arg = np.pi * i / (nsquash - 1)
        factor = 0.5 * (1 - np.cos(arg))
        if (minindex + i < nw) and (maxindex - i >= 0):
            fluxout[minindex + i] = factor * fluxout[minindex + i]
            fluxout[maxindex - i] = factor * fluxout[maxindex - i]
        else:
            print("INVALID FLUX IN APODIZE()")
            print("MININDEX=%d, i=%d" % (minindex, i))
            break

    if outerVal != 0:
        fluxout = fluxout + outerVal
        # Zero non-overlap parts
        for i in range(minindex):
            fluxout[i] = outerVal
        for i in range(maxindex, nw):
            fluxout[i] = outerVal

    return fluxout

def redshift_binned_spectrum(flux, z, nIndexes, dwlog, w0, w1, nw, outerVal=0.5):
    """Apply redshifting to a binned spectrum (Astrodash exact port)"""
    # Shift indices in log-wavelength space
    redshiftedIndexes = nIndexes + np.log(1 + z) / dwlog
    indexesInRange = redshiftedIndexes[redshiftedIndexes > 0]

    if len(indexesInRange) == 0:
        return np.zeros(nw) + outerVal

    # Interpolate flux at redshifted indices
    fluxInterp = scipy.interpolate.interp1d(indexesInRange, flux[redshiftedIndexes > 0], kind='linear')

    minWaveIndex = int(indexesInRange[0])

    # Create redshifted flux array
    fluxRedshifted = np.zeros(nw)
    fluxRedshifted[0:minWaveIndex] = outerVal * np.ones(minWaveIndex)

    # Fill in the redshifted flux
    valid_indices = indexesInRange[:nw - minWaveIndex]
    if len(valid_indices) > 0:
        fluxRedshifted[minWaveIndex:] = fluxInterp(valid_indices)

    # Apodize edges
    minIndex, maxIndex = min_max_index(fluxRedshifted, outerVal=outerVal)
    apodizedFlux = apodize(fluxRedshifted, minIndex, maxIndex, nw, outerVal=outerVal)

    return apodizedFlux

# Apply redshifting exactly as in original Astrodash
def apply_redshifting_batch(batch_xs, minZ, maxZ, nIndexes, dwlog, w0, w1, nw, outerVal=0.5):
    redshifts = np.random.uniform(low=minZ, high=maxZ, size=len(batch_xs))
    for j, z in enumerate(redshifts):
        batch_xs[j] = redshift_binned_spectrum(batch_xs[j], z, nIndexes, dwlog, w0, w1, nw, outerVal)
    return batch_xs



class OversampledAugmentedDataset(Dataset):
    """
    PyTorch Dataset that mimics Astrodash's OverSampling logic:
    - Balances classes by oversampling minority classes to match the majority.
    - Applies noise augmentation on-the-fly, with stddev depending on oversample amount.
    - Shuffles indices each epoch if DataLoader(shuffle=True) is used.
    """
    def __init__(self, images, labels, stddev_low=0.01, stddev_high=0.02, seed=None):  # Reduced noise
        self.images = images if isinstance(images, np.ndarray) else images.numpy()
        self.labels = labels if isinstance(labels, np.ndarray) else labels.numpy()
        self.stddev_low = stddev_low
        self.stddev_high = stddev_high
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self._build_oversample_indices()

    def _build_oversample_indices(self):
        # Use the mapped labels (which should be consecutive starting from 0)
        classes, counts = np.unique(self.labels, return_counts=True)
        max_count = counts.max()
        oversample_amount = np.rint(max_count / counts).astype(int)
        indices = []
        # Create a mapping from label to its index in the classes array
        label_to_idx = {label: idx for idx, label in enumerate(classes)}
        for label in classes:
            idxs = np.where(self.labels == label)[0]
            repeat = oversample_amount[label_to_idx[label]]
            indices.extend(list(np.repeat(idxs, repeat)))
        self.oversampled_indices = np.array(indices)
        np.random.shuffle(self.oversampled_indices)
        self.oversample_amount = oversample_amount
        self.label_to_amount = {label: oversample_amount[label_to_idx[label]] for label in classes}

    def __len__(self):
        return len(self.oversampled_indices)

    def __getitem__(self, idx):
        orig_idx = self.oversampled_indices[idx]
        image = self.images[orig_idx].copy()
        label = self.labels[orig_idx]  # This should now be the mapped label (0 to num_classes-1)
        stddev = self.stddev_low if self.label_to_amount[label] < 10 else self.stddev_high
        image = image + np.random.normal(0, stddev, size=image.shape)
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

def load_datasets():
    # Paths to saved arrays
    astroset_dir = os.path.join(os.path.dirname(__file__), 'astroset', 'training_set')
    train_images = np.load(os.path.join(astroset_dir, 'train_images.npy'))
    test_images = np.load(os.path.join(astroset_dir, 'test_images.npy'))

    # Load original string labels
    with open(os.path.join(astroset_dir, 'train_type_names.pkl'), 'rb') as f:
        train_type_names = pickle.load(f)
    with open(os.path.join(astroset_dir, 'test_type_names.pkl'), 'rb') as f:
        test_type_names = pickle.load(f)

    print(f"Loaded {train_images.shape[0]} train and {test_images.shape[0]} test samples.")

    # Create label mapping using string labels
    all_unique_labels = np.unique(np.concatenate([train_type_names, test_type_names]))
    label_to_idx = {label: idx for idx, label in enumerate(all_unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    # Map string labels to indices
    train_labels_mapped = np.array([label_to_idx[label] for label in train_type_names])
    test_labels_mapped = np.array([label_to_idx[label] for label in test_type_names])

    # Convert to torch tensors
    test_images_tensor = torch.tensor(test_images, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels_mapped, dtype=torch.long)

    # Create TensorDatasets
    train_dataset = OversampledAugmentedDataset(train_images, train_labels_mapped)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, train_labels_mapped, test_labels_mapped, idx_to_label

def cleanup_old_checkpoints(checkpoint_dir, keep_last=5):
    """Keep only the last N checkpoints to save disk space"""
    if not os.path.exists(checkpoint_dir):
        return

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if len(checkpoint_files) <= keep_last:
        return

    # Sort by epoch number and keep only the latest ones
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    files_to_remove = checkpoint_files[:-keep_last]

    for file_to_remove in files_to_remove:
        file_path = os.path.join(checkpoint_dir, file_to_remove)
        os.remove(file_path)
        print(f"Removed old checkpoint: {file_to_remove}")


def load_latest_checkpoint(model, optimizer, scheduler, checkpoint_dir):
    """Load the latest checkpoint if available"""
    if not os.path.exists(checkpoint_dir):
        return 0, float('inf'), 0  # start_epoch, best_val_loss, patience_counter

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoint_files:
        return 0, float('inf'), 0

    # Find the latest checkpoint
    try:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']

        print(f"Resuming from epoch {start_epoch}")
        print(f"Best validation loss so far: {best_val_loss:.4f}")
        print(f"Patience counter: {patience_counter}")

        return start_epoch, best_val_loss, patience_counter

    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        print("Starting training from scratch.")
        return 0, float('inf'), 0


def train_loop():
    num_epochs = 1000  # Train for 1000 epochs
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, train_labels, test_labels, idx_to_label = load_datasets()

    # Calculate number of classes based on the mapping
    num_classes = len(idx_to_label)
    print(f"Number of classes for model: {num_classes} (max label: {max(idx_to_label.keys())})")

    # Model setup with dropout
    model = CNNSpectraClassifier(num_classes=num_classes, dropout_rate=0.5).to(device)
    save_path = "cnn_spectra_model.pth"
    checkpoint_dir = "checkpoints"

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Try to load existing model, but handle errors gracefully
    try:
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            model.load_state_dict(torch.load(save_path, map_location=device))
            print(f"Loaded model weights from {save_path}, continuing training.")
        else:
            print("No existing model checkpoint found. Training from scratch.")
    except Exception as e:
        print(f"Could not load existing model: {e}")
        print("Starting training from scratch.")
        # Remove corrupted file if it exists
        if os.path.exists(save_path):
            os.remove(save_path)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Add weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    criterion = nn.CrossEntropyLoss()

    # Early stopping variables
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_epoch = 0

    print("Starting training loop...")
    t1 = time.time()
    test_accs = []
    val_losses = []

    # Load latest checkpoint if available
    start_epoch, best_val_loss, patience_counter = load_latest_checkpoint(model, optimizer, scheduler, checkpoint_dir)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_xs, batch_ys in train_loader:
            # Convert to numpy for redshifting (Astrodash exact)
            batch_xs_np = batch_xs.numpy()
            batch_ys_np = batch_ys.numpy()

            # Redshifting augmentation (Astrodash exact)
            if redshifting:
                batch_xs_np = apply_redshifting_batch(batch_xs_np, minZ, maxZ, nIndexes, dwlog, w0, w1, nw, outerVal)

            # Convert back to tensors and move to device
            batch_xs = torch.tensor(batch_xs_np, dtype=torch.float32).to(device)
            batch_ys = torch.tensor(batch_ys_np, dtype=torch.long).to(device)

            # Forward pass (pass flux directly)
            logits = model(batch_xs)
            loss = criterion(logits, batch_ys)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track epoch metrics
            epoch_loss += loss.item()
            pred = logits.argmax(dim=1, keepdim=True)
            epoch_correct += pred.eq(batch_ys.view_as(pred)).sum().item()
            epoch_total += batch_ys.size(0)

        # Compute epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        train_acc = epoch_correct / epoch_total

        # Evaluate on validation set every epoch
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for test_xs, test_ys in test_loader:
                test_xs, test_ys = test_xs.to(device), test_ys.to(device)
                test_logits = model(test_xs)
                val_loss += criterion(test_logits, test_ys).item()
                preds = torch.argmax(test_logits, 1)
                val_correct += preds.eq(test_ys).sum().item()
                val_total += test_ys.size(0)

        avg_val_loss = val_loss / len(test_loader)
        val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved at epoch {epoch} with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1

        # Print progress and save checkpoint every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Save checkpoint every 10 epochs
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            # Clean up old checkpoints (keep last 5)
            cleanup_old_checkpoints(checkpoint_dir, keep_last=5)

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best validation loss was {best_val_loss:.4f} at epoch {best_epoch}")
            break

        model.train()

    t2 = time.time()
    print(f"Training complete. Time spent: {t2-t1:.2f} seconds.")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # Load best model
    model.load_state_dict(torch.load(save_path, map_location=device))

    # Save the final model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save the label mapping for later evaluation
    import pickle
    with open("label_mapping.pkl", "wb") as f:
        pickle.dump(idx_to_label, f)
    print("Label mapping saved to label_mapping.pkl")

    return model, test_loader, device

def plot_results_from_checkpoint(model_path, test_loader, device):
    # Try to load the label mapping from file
    mapping_path = "label_mapping.pkl"
    if not os.path.exists(mapping_path):
        print(f"ERROR: {mapping_path} not found. You must save the label mapping during training.")
        return
    with open(mapping_path, "rb") as f:
        idx_to_label = pickle.load(f)
    print("[DEBUG] Loaded idx_to_label from label_mapping.pkl (first 10):", {k: idx_to_label[k] for k in list(idx_to_label)[:10]})

    # SN type list from original Astrodash
    sn_types = [
        'Ia-norm', 'Ia-91T', 'Ia-91bg', 'Ia-csm', 'Iax', 'Ia-pec',
        'Ib-norm', 'Ibn', 'IIb', 'Ib-pec', 'Ic-norm', 'Ic-broad',
        'Ic-pec', 'IIP', 'IIL', 'IIn', 'II-pec'
    ]

    def extract_sn_type(label_str):
        return str(label_str).split(':')[0].strip()

    # Load model
    num_classes = len(idx_to_label)
    model = CNNSpectraClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Collect predictions and true labels
    all_preds = []
    all_true = []
    with torch.no_grad():
        for test_xs, test_ys in test_loader:
            test_xs, test_ys = test_xs.to(device), test_ys.to(device)
            logits = model(test_xs)
            preds = torch.argmax(logits, 1)
            all_preds.append(preds.cpu().numpy())
            all_true.append(test_ys.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    # Map indices to original string labels
    all_pred_labels = [idx_to_label.get(idx, 'UNKNOWN') for idx in all_preds]
    all_true_labels = [idx_to_label.get(idx, 'UNKNOWN') for idx in all_true]

    # Map to SN types
    all_pred_types = [extract_sn_type(lbl) for lbl in all_pred_labels]
    all_true_types = [extract_sn_type(lbl) for lbl in all_true_labels]

    # Map SN types to indices in sn_types list, -1 if not found
    all_pred_type_indices = [sn_types.index(t) if t in sn_types else -1 for t in all_pred_types]
    all_true_type_indices = [sn_types.index(t) if t in sn_types else -1 for t in all_true_types]

    # Filter out any -1 (unknown types)
    valid = np.array([p != -1 and t != -1 for p, t in zip(all_pred_type_indices, all_true_type_indices)])
    all_pred_type_indices = np.array(all_pred_type_indices)[valid]
    all_true_type_indices = np.array(all_true_type_indices)[valid]

    # Plot confusion matrix
    cm = confusion_matrix(all_true_type_indices, all_pred_type_indices, labels=range(len(sn_types)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sn_types)
    disp.plot(xticks_rotation='vertical')
    plt.title('Confusion Matrix (Test Set) - SN Types Only')
    plt.tight_layout()
    plt.show()

    # Print accuracy by type
    print("\nAccuracy by SN Type:")
    for i, sn_type in enumerate(sn_types):
        type_mask = all_true_type_indices == i
        if np.sum(type_mask) > 0:
            type_acc = np.mean(all_pred_type_indices[type_mask] == all_true_type_indices[type_mask])
            print(f"{sn_type}: {type_acc:.3f} ({np.sum(type_mask)} samples)")
    overall_acc = np.mean(all_pred_type_indices == all_true_type_indices)
    print(f"\nOverall accuracy (by type): {overall_acc:.3f}")

# Main training function
if __name__ == "__main__":
    train_loop()
