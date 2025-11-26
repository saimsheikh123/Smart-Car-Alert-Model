#!/usr/bin/env python3
"""
train_audioucrnn.py
Fine-tune unified 7-class AudioCRNN for comprehensive audio event detection.
Includes data augmentation, class weighting, early stopping, and per-class metrics.

Usage:
    python train_audioucrnn.py --config config.yaml --dataset ./dataset
    python train_audioucrnn.py --dataset ./dataset --epochs 50 --batch_size 16
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# ============================================================================
# AudioCRNN Model (7-class)
# ============================================================================
class AudioCRNN(nn.Module):
    """CNN-RNN model for 7-class audio classification."""
    def __init__(self, num_classes=7):
        super(AudioCRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        
        # GRU expects input of shape (batch, seq_len, features)
        self.gru = nn.GRU(input_size=64 * 16, hidden_size=128, num_layers=1, batch_first=True)
        
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Input: [batch, 1, 128, 128] (mel-spec)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # [batch, 16, 64, 64]
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # [batch, 32, 32, 32]
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # [batch, 64, 16, 16]
        
        # Reshape for GRU: [batch, seq_len, features]
        batch, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch, width, channels, height]
        x = x.view(batch, width, channels * height)  # [batch, 16, 64*16=1024]
        
        x, _ = self.gru(x)  # [batch, 16, 128]
        x = x[:, -1, :]  # Take last timestep [batch, 128]
        
        x = self.relu(self.fc1(x))  # [batch, 128]
        x = self.dropout(x)
        x = self.fc2(x)  # [batch, num_classes]
        
        return x

# ============================================================================
# Audio Dataset
# ============================================================================
class AudioDataset(Dataset):
    """Load and preprocess audio files for training."""
    def __init__(self, root_dir, split='train', sr=16000, n_mels=128, n_fft=2048, hop_length=512):
        self.root_dir = Path(root_dir)
        self.split = split
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.files = []
        self.labels = []
        self.classes = []
        
        # Load class names from manifest if available
        manifest_path = self.root_dir / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                self.classes = manifest.get('classes', [])
        
        if not self.classes:
            # Infer from directories
            split_dir = self.root_dir / split
            if split_dir.exists():
                self.classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Scan for audio files
        split_dir = self.root_dir / split
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir() and class_dir.name in self.class_to_idx:
                    class_idx = self.class_to_idx[class_dir.name]
                    for audio_file in sorted(class_dir.glob('*.wav')):
                        self.files.append(audio_file)
                        self.labels.append(class_idx)
        
        print(f"  Loaded {len(self.files)} files for split '{split}'")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_file = self.files[idx]
        label = self.labels[idx]
        
        # Load audio
        y, _ = librosa.load(audio_file, sr=self.sr, mono=True)
        
        # Compute mel-spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)
        S = librosa.power_to_db(S, ref=np.max)
        
        # Normalize
        mn, mx = S.min(), S.max()
        if mx - mn > 0:
            S = (S - mn) / (mx - mn)
        else:
            S = np.zeros_like(S)
        
        # Pad/trim to [128, 128]
        if S.shape[1] > 128:
            S = S[:, :128]
        else:
            S = np.pad(S, ((0, 0), (0, 128 - S.shape[1])), mode='constant')
        
        # Add channel dimension: [1, 128, 128]
        S = torch.from_numpy(S[None, :, :].astype(np.float32))
        label = torch.tensor(label, dtype=torch.long)
        
        return S, label

# ============================================================================
# Training Functions
# ============================================================================
def compute_class_weights(dataset, num_classes):
    """Compute class weights for imbalanced data."""
    counts = [0] * num_classes
    for _, label in dataset:
        counts[label] += 1
    
    total = sum(counts)
    weights = []
    for count in counts:
        if count == 0:
            weights.append(1.0)
        else:
            weight = total / (num_classes * count)
            weights.append(weight)
    
    # Normalize
    weights = np.array(weights)
    weights = weights / weights.sum() * num_classes
    
    return torch.tensor(weights, dtype=torch.float32)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for X, y in tqdm(train_loader, desc="Training", leave=False):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def evaluate(model, val_loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in tqdm(val_loader, desc="Evaluating", leave=False):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * X.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels

def save_checkpoint(model, epoch, optimizer, loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def main():
    parser = argparse.ArgumentParser(description="Train unified 7-class AudioCRNN")
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file (YAML)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='./checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cpu, cuda')
    args = parser.parse_args()
    
    # Load config
    config = {}
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            full_config = yaml.safe_load(f)
            config = full_config.get('training', {})
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Batch size: {args.batch_size}, Epochs: {args.epochs}, LR: {args.lr}")
    
    # Create datasets
    print("\n[1/4] Loading datasets...")
    train_dataset = AudioDataset(args.dataset, split='train')
    val_dataset = AudioDataset(args.dataset, split='val')
    test_dataset = AudioDataset(args.dataset, split='test')
    
    num_classes = len(train_dataset.classes)
    print(f"  Classes ({num_classes}): {', '.join(train_dataset.classes)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model, optimizer, loss
    print("\n[2/4] Initializing model...")
    model = AudioCRNN(num_classes=num_classes).to(device)
    print(f"  Model: AudioCRNN with {num_classes} classes")
    
    # Compute class weights
    class_weights = compute_class_weights(train_dataset, num_classes)
    class_weights = class_weights.to(device)
    print(f"  Class weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Early stopping
    early_stop_patience = 10
    best_val_loss = float('inf')
    no_improve_count = 0
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n[3/4] Training...")
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            best_path = output_dir / 'best_audiocrnn_7class.pth'
            save_checkpoint(model, epoch, optimizer, val_loss, best_path)
            print(f"  [SAVE] Best checkpoint: {best_path}")
        else:
            no_improve_count += 1
        
        # Early stopping
        if no_improve_count >= early_stop_patience:
            print(f"\n[EARLY STOP] No improvement for {early_stop_patience} epochs")
            break
    
    # Evaluate on test set
    print("\n[4/4] Evaluating on test set...")
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"  Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    
    # Confusion matrix and per-class metrics
    cm = confusion_matrix(test_labels, test_preds)
    report = classification_report(test_labels, test_preds, target_names=train_dataset.classes, output_dict=True)
    
    print("\nPer-class metrics (Test Set):")
    for cls_name in train_dataset.classes:
        if cls_name in report:
            p, r, f1 = report[cls_name]['precision'], report[cls_name]['recall'], report[cls_name]['f1-score']
            print(f"  {cls_name}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
    
    # Save results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'per_class_metrics': report,
        'confusion_matrix': cm.tolist(),
        'classes': train_dataset.classes
    }
    
    results_path = output_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] Results: {results_path}")
    
    # Save confusion matrix image
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    cm_path = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=150)
    print(f"[SAVE] Confusion matrix: {cm_path}")
    
    # Save training metrics plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Train', marker='o')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Val', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['epoch'], metrics['train_acc'], label='Train', marker='o')
    plt.plot(metrics['epoch'], metrics['val_acc'], label='Val', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    metrics_path = output_dir / 'training_metrics.png'
    plt.savefig(metrics_path, dpi=150)
    print(f"[SAVE] Training metrics: {metrics_path}")
    
    # Copy best checkpoint to deployment location
    final_checkpoint = output_dir / 'multi_audio_crnn.pth'
    shutil.copy(best_path, final_checkpoint)
    print(f"\n[DEPLOY] Final checkpoint: {final_checkpoint}")
    print("  Ready to deploy! Copy this to Audio_Models/ folder.")
    
    return final_checkpoint

if __name__ == '__main__':
    import shutil
    main()
