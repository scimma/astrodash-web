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
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),  # (B, 32, 1024)
            nn.ReLU(),
            nn.MaxPool1d(2),  # (B, 32, 512)
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),  # (B, 64, 512)
            nn.ReLU(),
            nn.MaxPool1d(2),  # (B, 64, 256)
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),  # (B, 128, 256)
            nn.ReLU(),
            nn.MaxPool1d(2),  # (B, 128, 128)
        )
        self.fc1 = nn.Linear(128 * 128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, flux):
        # flux: (B, 1024)
        x = flux.unsqueeze(1)  # (B, 1, 1024)
        x = self.cnn(x)        # (B, 128, 128)
        x = x.view(x.size(0), -1)  # (B, 128*128)
        x = F.relu(self.fc1(x))    # (B, 256)
        x = F.relu(self.fc2(x))    # (B, 128)
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
    def __init__(self, images, labels, stddev_low=0.03, stddev_high=0.05, seed=None):
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
    train_labels = np.load(os.path.join(astroset_dir, 'train_labels.npy'))
    # with open(os.path.join(astroset_dir, 'train_filenames.pkl'), 'rb') as f:
    #     train_filenames = pickle.load(f)
    # with open(os.path.join(astroset_dir, 'train_type_names.pkl'), 'rb') as f:
    #     train_type_names = pickle.load(f)

    test_images = np.load(os.path.join(astroset_dir, 'test_images.npy'))
    test_labels = np.load(os.path.join(astroset_dir, 'test_labels.npy'))
    # with open(os.path.join(astroset_dir, 'test_filenames.pkl'), 'rb') as f:
    #     test_filenames = pickle.load(f)
    # with open(os.path.join(astroset_dir, 'test_type_names.pkl'), 'rb') as f:
    #     test_type_names = pickle.load(f)

    print(f"Loaded {train_images.shape[0]} train and {test_images.shape[0]} test samples.")

    # Create label mapping to ensure consecutive indices starting from 0
    all_unique_labels = np.unique(np.concatenate([train_labels, test_labels]))
    label_to_idx = {label: idx for idx, label in enumerate(all_unique_labels)}

    # Map labels to consecutive indices
    train_labels_mapped = np.array([label_to_idx[label] for label in train_labels])
    test_labels_mapped = np.array([label_to_idx[label] for label in test_labels])

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

    return train_loader, test_loader, train_labels_mapped

def train_loop():
    num_epochs = 1000  # Train for 1000 epochs
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, train_labels = load_datasets()

    # Calculate number of classes based on the maximum label value + 1
    num_classes = train_labels.max() + 1
    print(f"Number of classes for model: {num_classes} (max label: {train_labels.max()})")

    # Model setup
    model = CNNSpectraClassifier(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("Starting training loop...")
    t1 = time.time()
    test_accs = []

    for epoch in range(num_epochs):
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

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}, Training accuracy: {train_acc:.4f}")

        # Evaluate on test set every 50 epochs
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                all_preds = []
                all_true = []
                for test_xs, test_ys in test_loader:
                    test_xs, test_ys = test_xs.to(device), test_ys.to(device)
                    test_logits = model(test_xs)
                    preds = torch.argmax(test_logits, 1)
                    all_preds.append(preds.cpu().numpy())
                    all_true.append(test_ys.cpu().numpy())
                all_preds = np.concatenate(all_preds)
                all_true = np.concatenate(all_true)
                test_acc = np.mean(all_preds == all_true)
                print(f"Test accuracy: {test_acc:.4f}")
                test_accs.append(test_acc)
            model.train()

    t2 = time.time()
    print(f"Training complete. Time spent: {t2-t1:.2f} seconds.")

    # Save the model
    save_path = "cnn_spectra_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model, test_loader, device

def plot_results(model, test_loader, device):
    # Plot confusion matrix for test set by SN type only
    print("Computing confusion matrix for test set (by SN type only)...")
    model.eval()

    # SN type list from original Astrodash
    sn_types = ['Ia-norm', 'Ia-91T', 'Ia-91bg', 'Ia-csm', 'Iax', 'Ia-pec',
                'Ib-norm', 'Ibn', 'IIb', 'Ib-pec', 'Ic-norm', 'Ic-broad',
                'Ic-pec', 'IIP', 'IIL', 'IIn', 'II-pec']

    with torch.no_grad():
        all_preds = []
        all_true = []
        all_pred_types = []
        all_true_types = []

        for test_xs, test_ys in test_loader:
            test_xs, test_ys = test_xs.to(device), test_ys.to(device)
            test_logits = model(test_xs)
            preds = torch.argmax(test_logits, 1)
            all_preds.append(preds.cpu().numpy())
            all_true.append(test_ys.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)

        # Convert age bin labels back to original labels to extract SN types
        # We need to reverse the mapping to get original labels
        all_unique_labels = np.unique(np.concatenate([train_labels, test_labels]))
        idx_to_label = {idx: label for idx, label in enumerate(all_unique_labels)}

        # Convert mapped indices back to original labels
        original_preds = np.array([idx_to_label[pred] for pred in all_preds])
        original_true = np.array([idx_to_label[true] for true in all_true])

        # Extract SN types from age bin labels (e.g., "Ia-norm: 4 to 8" -> "Ia-norm")
        for pred_label, true_label in zip(original_preds, original_true):
            # Extract type from the label (assuming format like "Ia-norm: 4 to 8")
            pred_type = str(pred_label).split(':')[0].strip()
            true_type = str(true_label).split(':')[0].strip()

            # Map to type index
            if pred_type in sn_types:
                pred_type_idx = sn_types.index(pred_type)
            else:
                pred_type_idx = 0  # Default to first type if not found

            if true_type in sn_types:
                true_type_idx = sn_types.index(true_type)
            else:
                true_type_idx = 0  # Default to first type if not found

            all_pred_types.append(pred_type_idx)
            all_true_types.append(true_type_idx)

        # Create confusion matrix for SN types only
        cm = confusion_matrix(all_true_types, all_pred_types, labels=range(len(sn_types)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sn_types)
        disp.plot(xticks_rotation='vertical')
        plt.title('Confusion Matrix (Test Set) - SN Types Only')
        plt.tight_layout()
        plt.show()

        # Print accuracy by type
        print("\nAccuracy by SN Type:")
        for i, sn_type in enumerate(sn_types):
            type_mask = np.array(all_true_types) == i
            if np.sum(type_mask) > 0:
                type_acc = np.mean(np.array(all_pred_types)[type_mask] == np.array(all_true_types)[type_mask])
                print(f"{sn_type}: {type_acc:.3f} ({np.sum(type_mask)} samples)")

        # Overall accuracy
        overall_acc = np.mean(np.array(all_pred_types) == np.array(all_true_types))
        print(f"\nOverall accuracy (by type): {overall_acc:.3f}")

def plot_results_from_checkpoint(model_path, test_loader, train_labels, test_labels, device):
    import torch
    import numpy as np
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    # SN type list from original Astrodash
    sn_types = [
        'Ia-norm', 'Ia-91T', 'Ia-91bg', 'Ia-csm', 'Iax', 'Ia-pec',
        'Ib-norm', 'Ibn', 'IIb', 'Ib-pec', 'Ic-norm', 'Ic-broad',
        'Ic-pec', 'IIP', 'IIL', 'IIn', 'II-pec'
    ]

    # Helper to extract SN type from label string
    def extract_sn_type(label_str):
        return str(label_str).split(':')[0].strip()

    # Rebuild the mapping from index to original label and SN type
    all_unique_labels = np.unique(np.concatenate([train_labels, test_labels]))
    idx_to_label = {idx: label for idx, label in enumerate(all_unique_labels)}
    idx_to_type = {idx: extract_sn_type(label) for idx, label in idx_to_label.items()}

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

    # Map indices to SN types
    all_pred_types = [sn_types.index(idx_to_type[pred]) if idx_to_type[pred] in sn_types else -1 for pred in all_preds]
    all_true_types = [sn_types.index(idx_to_type[true]) if idx_to_type[true] in sn_types else -1 for true in all_true]

    # Filter out any -1 (unknown types)
    valid = np.array([p != -1 and t != -1 for p, t in zip(all_pred_types, all_true_types)])
    all_pred_types = np.array(all_pred_types)[valid]
    all_true_types = np.array(all_true_types)[valid]

    # Plot confusion matrix
    cm = confusion_matrix(all_true_types, all_pred_types, labels=range(len(sn_types)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sn_types)
    disp.plot(xticks_rotation='vertical')
    plt.title('Confusion Matrix (Test Set) - SN Types Only')
    plt.tight_layout()
    plt.show()

    # Print accuracy by type
    print("\nAccuracy by SN Type:")
    for i, sn_type in enumerate(sn_types):
        type_mask = all_true_types == i
        if np.sum(type_mask) > 0:
            type_acc = np.mean(all_pred_types[type_mask] == all_true_types[type_mask])
            print(f"{sn_type}: {type_acc:.3f} ({np.sum(type_mask)} samples)")
    overall_acc = np.mean(all_pred_types == all_true_types)
    print(f"\nOverall accuracy (by type): {overall_acc:.3f}")

# Main training function
if __name__ == "__main__":
    train_loop()
