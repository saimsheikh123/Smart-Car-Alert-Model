#!/usr/bin/env python3
"""
quick_test.py
Quickly test a checkpoint against a folder of audio files.
Useful for verifying model performance before deployment.

Usage:
    python quick_test.py --checkpoint checkpoints/best_audiocrnn_7class.pth --test_dir ./dataset/test
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================================
# AudioCRNN Model (must match training)
# ============================================================================
class AudioCRNN(nn.Module):
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
        self.gru = nn.GRU(input_size=64 * 16, hidden_size=128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        batch, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch, width, channels * height)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================================================
# Testing
# ============================================================================
def load_and_predict(model, audio_path, device, sr=16000):
    """Load audio and get model prediction."""
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        S = librosa.power_to_db(S, ref=np.max)
        
        mn, mx = S.min(), S.max()
        if mx - mn > 0:
            S = (S - mn) / (mx - mn)
        else:
            S = np.zeros_like(S)
        
        if S.shape[1] > 128:
            S = S[:, :128]
        else:
            S = np.pad(S, ((0, 0), (0, 128 - S.shape[1])), mode='constant')
        
        S = torch.from_numpy(S[None, None, :, :].astype(np.float32)).to(device)
        
        with torch.no_grad():
            logits = model(S)
            probs = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        return pred_class, confidence
    except Exception as e:
        print(f"  [ERROR] {audio_path}: {e}")
        return None, 0.0

def main():
    parser = argparse.ArgumentParser(description="Quick test of trained model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--test_dir', type=str, default='./dataset/test', help='Path to test directory')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cpu, cuda')
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print(f"[INFO] Test directory: {args.test_dir}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Determine num_classes from checkpoint
    model_state = checkpoint['model_state_dict']
    num_classes = model_state['fc2.weight'].shape[0]
    
    # Load model
    model = AudioCRNN(num_classes=num_classes).to(device)
    model.load_state_dict(model_state)
    model.eval()
    print(f"[OK] Model loaded with {num_classes} classes")
    
    # Scan test directory
    test_path = Path(args.test_dir)
    if not test_path.exists():
        print(f"[ERROR] Test directory not found: {args.test_dir}")
        return
    
    # Infer class names from subdirectories
    classes = sorted([d.name for d in test_path.iterdir() if d.is_dir()])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    idx_to_class = {i: cls for i, cls in enumerate(classes)}
    
    print(f"[INFO] Classes: {', '.join(classes)}")
    
    # Collect predictions
    all_preds = []
    all_labels = []
    per_class_preds = defaultdict(list)
    per_class_labels = defaultdict(list)
    
    print("\n[Testing...]")
    for class_dir in sorted(test_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        class_idx = class_to_idx[class_name]
        
        audio_files = list(class_dir.glob('*.wav'))
        print(f"\n  {class_name} ({len(audio_files)} files):")
        
        class_correct = 0
        for audio_file in audio_files:
            pred_idx, confidence = load_and_predict(model, audio_file, device)
            
            if pred_idx is not None:
                pred_class = idx_to_class[pred_idx]
                is_correct = pred_idx == class_idx
                class_correct += is_correct
                
                all_preds.append(pred_idx)
                all_labels.append(class_idx)
                per_class_preds[class_name].append(pred_class)
                per_class_labels[class_name].append(class_name)
        
        class_acc = class_correct / len(audio_files) if audio_files else 0
        print(f"    Accuracy: {class_acc:.1%} ({class_correct}/{len(audio_files)})")
    
    # Overall metrics
    print(f"\n[Results]")
    overall_acc = accuracy_score(all_labels, all_preds) if all_labels else 0
    print(f"  Overall Accuracy: {overall_acc:.1%}")
    
    # Per-class metrics
    print(f"\n  Per-class metrics:")
    report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    print(f"\n  Confusion Matrix:")
    print(f"    Rows: True class, Columns: Predicted class")
    print(f"    Classes: {', '.join(classes)}")
    print(cm)

if __name__ == '__main__':
    main()
