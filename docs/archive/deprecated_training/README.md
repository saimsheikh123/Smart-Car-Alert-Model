# Audio Classification Training Pipeline
## Option A: Unified 7-Class AudioCRNN

This directory contains scripts and configuration to retrain the AudioCRNN model to recognize **all 7 sound classes** with improved accuracy, especially for challenging classes like **glass_breaking**.

**Key improvements:**
- ✓ Unified multi-class model (instead of separate models)
- ✓ Data augmentation for minority classes (glass, scream, etc.)
- ✓ Class weighting for imbalanced data
- ✓ Early stopping and best-model checkpointing
- ✓ Comprehensive evaluation metrics (per-class precision/recall, confusion matrix)

---

## Quick Start (TL;DR)

If you just want to run the full pipeline:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare dataset (organize + augment)
python prepare_dataset.py --sources \
    c:\Users\Saim\cmpe-281-models\cmpe-281-models\audio_datasets \
    c:\Users\Saim\cmpe-281-models\cmpe-281-models\corrections_audio \
    --output ./dataset

# 3. Train model
python train_audioucrnn.py --dataset ./dataset --epochs 50 --batch_size 16

# 4. Deploy checkpoint
copy checkpoints\multi_audio_crnn.pth c:\Users\Saim\cmpe-281-models\cmpe-281-models\Audio_Models\Audio_Models\

# 5. Restart FastAPI server
# (your server will now load the new checkpoint)
```

---

## Step-by-Step Guide

### Step 1: Install Dependencies

Navigate to this directory and install required packages:

```bash
cd c:\Users\Saim\cmpe-281-models\cmpe-281-models\train
pip install -r requirements.txt
```

**Note:** If you already have `torch` installed, the command will skip it. If you want GPU support, install `torch` separately:
```bash
# For GPU (NVIDIA CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Organize and Prepare Dataset

The `prepare_dataset.py` script:
- Scans source directories (zipped datasets + feedback audio)
- Extracts and resamples all audio to 16kHz, mono
- Organizes into `train/val/test` folders by class
- **Augments minority classes** (glass, scream) to balance the dataset
- Saves a manifest with metadata

**Command:**

```bash
python prepare_dataset.py \
    --sources c:\Users\Saim\cmpe-281-models\cmpe-281-models\audio_datasets \
             c:\Users\Saim\cmpe-281-models\cmpe-281-models\corrections_audio \
    --output ./dataset \
    --train_split 0.70 \
    --val_split 0.15 \
    --augment_minority
```

**Options:**
- `--sources`: Source directories (can list multiple)
- `--output`: Output directory (default: `./dataset`)
- `--train_split`: Train ratio (default: 0.70, meaning 70% train, 15% val, 15% test)
- `--val_split`: Val ratio (default: 0.15)
- `--augment_minority`: Enable augmentation for underrepresented classes (default: True)

**Expected output:**
```
./dataset/
├── train/
│   ├── alert_sounds/
│   ├── car_crash/
│   ├── emergency_sirens/
│   ├── environmental_sounds/
│   ├── glass_breaking/
│   ├── human_scream/
│   └── road_traffic/
├── val/
│   └── [same structure]
├── test/
│   └── [same structure]
└── manifest.json
```

### Step 3: Train the Model

The `train_audioucrnn.py` script:
- Loads the organized dataset
- Initializes a 7-class AudioCRNN model
- Computes class weights (to handle imbalance)
- Trains with early stopping (stops if validation loss doesn't improve for 10 epochs)
- Saves best checkpoint and test metrics

**Command:**

```bash
python train_audioucrnn.py \
    --dataset ./dataset \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.001 \
    --device auto \
    --output ./checkpoints
```

**Options:**
- `--dataset`: Path to prepared dataset (required)
- `--epochs`: Max training epochs (default: 50)
- `--batch_size`: Batch size (default: 16; use 32 for faster training if GPU memory allows)
- `--lr`: Learning rate (default: 0.001)
- `--device`: `auto` (auto-detect), `cpu`, or `cuda`
- `--output`: Checkpoint output directory (default: `./checkpoints`)

**Expected output:**
```
[1/4] Loading datasets...
  Loaded 450 files for split 'train'
  Loaded 96 files for split 'val'
  Loaded 96 files for split 'test'
  Classes (7): alert_sounds, car_crash, emergency_sirens, environmental_sounds, glass_breaking, human_scream, road_traffic

[2/4] Initializing model...
  Model: AudioCRNN with 7 classes
  Class weights: [1.2, 1.5, 1.1, 0.9, 2.0, 2.3, 1.0]

[3/4] Training...
Epoch 1/50
  Train Loss: 1.8234, Acc: 0.3421
  Val Loss: 1.5623, Acc: 0.4102
  [SAVE] Best checkpoint: checkpoints/best_audiocrnn_7class.pth
...
[4/4] Evaluating on test set...
  Test Loss: 1.2345, Acc: 0.6234

Per-class metrics (Test Set):
  alert_sounds: Precision=0.720, Recall=0.680, F1=0.700
  car_crash: Precision=0.850, Recall=0.820, F1=0.835
  ...
  glass_breaking: Precision=0.780, Recall=0.750, F1=0.765
  ...

[SAVE] Results: checkpoints/test_results.json
[SAVE] Confusion matrix: checkpoints/confusion_matrix.png
[SAVE] Training metrics: checkpoints/training_metrics.png

[DEPLOY] Final checkpoint: checkpoints/multi_audio_crnn.pth
```

**Training tips:**
- **GPU acceleration:** If you have NVIDIA GPU, training is ~5-10x faster. Install `torch` with CUDA support (see Step 1).
- **Batch size:** Larger batch sizes (32, 64) train faster but need more memory. Start with 16.
- **Epochs:** 50 epochs is usually enough; early stopping will halt if validation loss plateaus.
- **Class imbalance:** The script automatically weights classes inversely by frequency, so minority classes (glass, scream) are emphasized.

### Step 4: Deploy New Checkpoint

After training completes, the best checkpoint is saved as `checkpoints/multi_audio_crnn.pth`. Copy it to the model directory:

```bash
# From PowerShell:
copy checkpoints\multi_audio_crnn.pth c:\Users\Saim\cmpe-281-models\cmpe-281-models\Audio_Models\Audio_Models\multi_audio_crnn.pth
```

### Step 5: Restart FastAPI Server and Test

1. **Stop** the old FastAPI server (if running):
   - Press Ctrl+C in the terminal where it's running.

2. **Start** the server again (it will load the new checkpoint):
   ```bash
   cd c:\Users\Saim\cmpe-281-models\cmpe-281-models\Audio_Models\Audio_Models
   python -m uvicorn multi_model_api:app --host 127.0.0.1 --port 8001
   ```

3. **Test** the API:
   - Open http://127.0.0.1:8001/ in your browser.
   - Upload a glass-breaking audio sample.
   - Verify the model now correctly identifies it as `glass_breaking`.

   **Or from command-line (PowerShell):**
   ```powershell
   $file = 'c:\path\to\glass_sample.wav'
   $resp = curl.exe -s -X POST -F "file=@$file" `
     -F "use_crnn=true" -F "use_siren=true" -F "use_dqn=true" `
     http://127.0.0.1:8001/classify | ConvertFrom-Json
   
   $resp.ensemble.primary_class  # Should be 'glass_breaking'
   $resp.ensemble.confidence     # Confidence score
   ```

---

## Configuration File (config.yaml)

The `config.yaml` file provides advanced knobs:

```yaml
# Dataset classes to train
dataset:
  classes:
    - alert_sounds
    - car_crash
    - emergency_sirens
    - environmental_sounds
    - glass_breaking
    - human_scream
    - road_traffic
  num_classes: 7

# Audio preprocessing
audio:
  sr: 16000               # Sample rate
  n_mels: 128             # Number of mel-frequency bins
  mel_spec_len: 128       # Time dimension of mel-spectrogram

# Data augmentation (training only)
augmentation:
  enabled: true
  apply_noise: true
  noise_prob: 0.3         # 30% chance to add noise to each sample
  apply_stretch: true
  stretch_factor: [0.8, 1.2]
  apply_shift: true
  shift_prob: 0.3

# Training hyperparameters
training:
  batch_size: 16
  num_epochs: 50
  learning_rate: 0.001
  loss_function: "weighted_crossentropy"  # Handles class imbalance
  class_weights: "balanced"
  early_stopping_patience: 10
```

To use a custom config:
```bash
python train_audioucrnn.py --config my_config.yaml --dataset ./dataset
```

---

## Evaluation and Metrics

After training, the script generates:

1. **`checkpoints/test_results.json`** — Test accuracy and per-class precision/recall/F1
2. **`checkpoints/confusion_matrix.png`** — Visual confusion matrix (helps identify which classes are confused)
3. **`checkpoints/training_metrics.png`** — Loss and accuracy curves over epochs

**Sample output:**
```json
{
  "test_loss": 1.2345,
  "test_accuracy": 0.6234,
  "per_class_metrics": {
    "glass_breaking": {
      "precision": 0.78,
      "recall": 0.75,
      "f1-score": 0.765
    },
    ...
  },
  "confusion_matrix": [[...], [...], ...]
}
```

Review these metrics to understand:
- **Precision:** Of all predictions for class X, how many were correct?
- **Recall:** Of all true samples of class X, how many did we find?
- **F1:** Harmonic mean of precision and recall (good overall metric).

If glass_breaking has low recall (< 0.7), you may need:
- More glass training data (collect more samples or augment more aggressively)
- Adjust augmentation parameters (use more pitch/stretch variations)
- Fine-tune hyperparameters (learning rate, batch size)

---

## Troubleshooting

### "ImportError: No module named 'yaml'"
→ Install PyYAML: `pip install PyYAML`

### "RuntimeError: CUDA out of memory"
→ Reduce batch size: `--batch_size 8` (or use `--device cpu`)

### "OSError: [Errno 2] No such file or directory: './dataset/train'"
→ Run `prepare_dataset.py` first to generate the dataset directory.

### Model accuracy is low / glass_breaking still misclassified
1. Check confusion matrix (`confusion_matrix.png`) to see what it's confused with
2. Collect more glass_breaking samples (or adjust augmentation settings)
3. Try training for more epochs (remove early stopping or increase patience)
4. Try a smaller learning rate (e.g., `--lr 0.0005`)

### Training very slow
→ Use GPU: Install `torch` with CUDA support (see Step 1)
→ Increase batch size: `--batch_size 32` (if memory allows)

---

## Advanced: Iterative Retraining

You can set up **automatic retraining** based on user feedback:

1. **Collect feedback:** Users mark misclassifications via the `/feedback/correction` endpoint.
2. **Retrain periodically:** Daily or weekly, run `prepare_dataset.py` (which includes feedback audio) and `train_audioucrnn.py`.
3. **Deploy:** Copy new checkpoint to deployment folder.

To automate (optional Windows Task Scheduler):
```bash
# Create a batch script: retrain.bat
@echo off
cd c:\Users\Saim\cmpe-281-models\cmpe-281-models\train
python prepare_dataset.py --sources ... --output ./dataset
python train_audioucrnn.py --dataset ./dataset --epochs 50
copy checkpoints\multi_audio_crnn.pth c:\Users\Saim\cmpe-281-models\cmpe-281-models\Audio_Models\Audio_Models\
echo Retraining complete!
```

Then schedule it to run daily via Windows Task Scheduler.

---

## Files in This Directory

- **`config.yaml`** — Training configuration (hyperparameters, augmentation settings, paths)
- **`prepare_dataset.py`** — Organize audio files, apply augmentation, create train/val/test split
- **`train_audioucrnn.py`** — Train the 7-class AudioCRNN model
- **`requirements.txt`** — Python dependencies
- **`README.md`** — This file

---

## Questions?

If retraining doesn't improve glass_breaking detection:
1. Check that the `audio_datasets/glass_breaking_dataset/` folder actually contains wav files.
2. Verify `prepare_dataset.py` found them: look at the console output or check `dataset/train/glass_breaking/` folder.
3. Review the confusion matrix: is glass confused with a specific other class? (May indicate a feature overlap.)
4. Try collecting more diverse glass examples via the UI feedback mechanism.

---

## Next Steps (After Deployment)

Once the new checkpoint is deployed:
1. **Test end-to-end:** Upload glass-breaking samples via the web UI. Confidence should be >70% for true glass.
2. **Monitor feedback:** Track `/feedback/stats` endpoint to see if glass_breaking errors decrease.
3. **Iterate:** Every week, run retraining on accumulated feedback to continuously improve.

Good luck! 🎯
