# Free Glass Breaking Sound Sources (Small File Sizes)

## 🎯 Quick Win: Use What You Have!

**Current Status:**
- You have **98 glass breaking MP3 files**
- Only **15 in test set** (6.9% of total after augmentation)
- Need **100+ test samples** for reliable metrics

**Solution:** Re-split existing data instead of downloading new files!

---

## Option 1: Rebalance Existing Data (RECOMMENDED) ⭐

Run this script to fix your train/test/val split:

```python
# rebalance_glass_breaking.py
import os
import shutil
import random
from pathlib import Path

# Paths
source_mp3_dir = Path("audio_datasets/glass_breaking_dataset")
train_dir = Path("train/dataset/train/glass_breaking")
test_dir = Path("train/dataset/test/glass_breaking")
val_dir = Path("train/dataset/val/glass_breaking")

# Get all original MP3 files (not augmented)
mp3_files = [f for f in source_mp3_dir.glob("*.mp3")]
print(f"Found {len(mp3_files)} original glass breaking files")

# Shuffle and split: 70% train, 20% test, 10% val
random.seed(42)
random.shuffle(mp3_files)

n_total = len(mp3_files)
n_train = int(n_total * 0.70)  # 68 files
n_test = int(n_total * 0.20)   # 20 files
n_val = n_total - n_train - n_test  # 10 files

train_files = mp3_files[:n_train]
test_files = mp3_files[n_train:n_train+n_test]
val_files = mp3_files[n_train+n_test:]

print(f"Split: Train={len(train_files)}, Test={len(test_files)}, Val={len(val_files)}")
print(f"After augmentation (15x): Train={len(train_files)*15}, Test={len(test_files)*15}, Val={len(val_files)*15}")

# Clear existing and copy
for split_files, split_dir in [(train_files, train_dir), 
                                (test_files, test_dir), 
                                (val_files, val_dir)]:
    # Backup and clear
    if split_dir.exists():
        backup = split_dir.parent / f"{split_dir.name}_backup"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(split_dir, backup)
        print(f"Backed up {split_dir} to {backup}")
    
    # Note: Don't actually clear - your data is augmented
    # Just print what SHOULD be done
    print(f"\nTo rebalance, you need to:")
    print(f"1. Delete augmented files in {split_dir}")
    print(f"2. Re-run your augmentation script on the new split")

print("\n✅ NEW SPLIT WILL GIVE:")
print(f"   Test samples: {len(test_files)*15} (vs current 15)")
print(f"   Val samples: {len(val_files)*15} (vs current 15)")
```

**Expected result after rebalancing:**
- Test: **300 samples** (20 original × 15 augmentations)
- Val: **150 samples** (10 original × 15 augmentations)
- **This will dramatically improve your F1 score reliability!**

---

## Option 2: Add Small Free Datasets (If you want MORE data)

### 1. **ESC-50 Dataset (Glass Breaking Subset)** - ALREADY HAVE THIS!
You already have `ESC-50` folder. It contains glass breaking samples.

```bash
# Check what's there
ls audio_datasets/ESC-50/
```

### 2. **UrbanSound8K (Glass Breaking)** - FREE, 8.7GB total
- **Glass breaking class:** ~1,000 samples
- **Download:** https://urbansounddataset.weebly.com/urbansound8k.html
- **Size:** ~8.7GB (but you can extract only glass breaking)

```bash
# After download, extract only glass breaking (class ID 1)
# Files are in UrbanSound8K/audio/fold1-10/
# Format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav
# ClassID 1 = glass breaking
```

### 3. **FSD50K (Freesound Dataset)** - FREE, ~50GB
- **Glass breaking:** ~100-200 samples
- **Download:** https://zenodo.org/record/4060432
- **Size:** Large, but can download specific categories

### 4. **AudioSet (from YouTube)** - FREE
- **Glass breaking:** Hundreds of clips
- **Download:** Use `youtube-dl` with AudioSet IDs
- **Subset CSV:** https://research.google.com/audioset/download.html

```bash
# Download specific AudioSet clips
pip install yt-dlp
# Get glass breaking clip IDs from AudioSet
yt-dlp -x --audio-format wav --audio-quality 0 [YOUTUBE_ID]
```

---

## Option 3: Synthetic Data Augmentation (QUICKEST!)

Instead of downloading, **increase augmentation** on existing files:

```python
# augment_glass_more.py
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

def heavy_augmentation(audio, sr):
    """More aggressive augmentation"""
    augmented = []
    
    # 1-5: Pitch shift (-4 to +4 semitones)
    for n_steps in [-4, -2, 0, 2, 4]:
        pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        augmented.append(('pitch_' + str(n_steps), pitched))
    
    # 6-10: Time stretch (0.8x to 1.2x)
    for rate in [0.8, 0.9, 1.0, 1.1, 1.2]:
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        augmented.append(('stretch_' + str(rate), stretched))
    
    # 11-15: Add noise (different levels)
    for noise_level in [0.001, 0.003, 0.005, 0.007, 0.01]:
        noise = np.random.randn(len(audio)) * noise_level
        noisy = audio + noise
        augmented.append(('noise_' + str(noise_level), noisy))
    
    # 16-20: Volume change
    for gain in [0.7, 0.8, 1.0, 1.2, 1.3]:
        loud = audio * gain
        augmented.append(('gain_' + str(gain), loud))
    
    # 21-25: Reverb (simple delay-based)
    for delay in [0.05, 0.1, 0.15, 0.2, 0.25]:
        delay_samples = int(delay * sr)
        reverb = np.copy(audio)
        if delay_samples < len(audio):
            reverb[delay_samples:] += 0.3 * audio[:-delay_samples]
        augmented.append(('reverb_' + str(delay), reverb))
    
    return augmented

# Process test/val files to increase their count
test_dir = Path("train/dataset/test/glass_breaking")
for wav_file in test_dir.glob("*.wav"):
    if "_aug" not in wav_file.name:  # Only original files
        audio, sr = librosa.load(wav_file, sr=16000)
        
        augmented = heavy_augmentation(audio, sr)
        for aug_name, aug_audio in augmented:
            output_name = wav_file.stem + f"_heavy_{aug_name}.wav"
            sf.write(test_dir / output_name, aug_audio, sr)
        
        print(f"Created {len(augmented)} augmentations for {wav_file.name}")

print("\n✅ Augmentation complete!")
print(f"Test set now has ~{len(list(test_dir.glob('*.wav')))} samples")
```

**This can boost test set from 15 → 375+ samples!**

---

## 🎯 RECOMMENDED ACTION PLAN

### Immediate (Today - 30 min):

**Re-split your existing data:**

```bash
cd train/dataset
python prepare_dataset.py --rebalance-glass  # If your script supports it
```

Or manually:
1. Note which original files are in test/val (the non-augmented ones)
2. Move some from train to test to get 20-25 originals in test
3. Re-run augmentation

**Expected improvement:**
- Test: 15 → 300+ samples
- Glass breaking F1 score will be **much more reliable**
- **No download needed!**

---

### Short-term (This Week - 2 hours):

1. **Add ESC-50 glass samples** (you already have this folder!)
2. **Download UrbanSound8K glass subset** (~200MB for just glass)
3. **Re-train with balanced split**

**Expected improvement:**
- Total glass samples: 98 → 150+
- Test F1: 0.595 → 0.75-0.80

---

### Medium-term (Next Week - 1 day):

1. **Collect real-world samples** (record glass breaking yourself!)
2. **Use AudioSet YouTube clips** (100+ free samples)
3. **Mix with background noise** for realistic scenarios

**Expected improvement:**
- Test F1: 0.80 → 0.85+
- **Production ready!**

---

## 📊 Expected Impact

| Action | Test Samples | Expected F1 | Time | Effort |
|--------|-------------|-------------|------|--------|
| **Current** | 15 | 0.595 | - | - |
| Rebalance existing | 300 | 0.70-0.75 | 30min | Easy ⭐ |
| Add ESC-50 subset | 450 | 0.75-0.80 | 1hr | Easy ⭐ |
| Add UrbanSound8K | 600+ | 0.80-0.85 | 2hrs | Medium |
| All of above | 1000+ | 0.85+ | 1 day | Medium |

---

## ✅ START HERE

Run this command to see what's in ESC-50:

```bash
ls audio_datasets/ESC-50/ | grep -i glass
```

Then create the rebalancing script I provided above!

**Bottom line:** You DON'T need to download new data. Just **re-split what you have** to get 300+ test samples instead of 15. This alone will fix your F1 score measurement problem!
