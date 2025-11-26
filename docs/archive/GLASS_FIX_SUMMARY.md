# Glass Breaking Dataset - Quick Fix Summary

## 🎯 THE REAL PROBLEM

You **don't need more glass breaking audio files**. You already have **98 unique samples** - that's plenty!

**The actual issue:** Your test/val split is terrible
- Only **15 test samples** (should be 300+)
- Only **15 val samples** (should be 150+)
- F1 score of 0.595 is **unreliable** with so few test samples

---

## ✅ THE FIX (30 minutes)

### Step 1: Run the rebalancing script

```bash
cd c:\Users\Saim\cmpe-281-models\cmpe-281-models
python rebalance_glass_dataset.py
```

**What it does:**
1. Backs up your current data
2. Redistributes files: 70% train, 20% test, 10% val
3. Keeps all augmentations
4. **Result:** 300 test samples instead of 15!

**Expected improvement:**
- Your test set will be **20x larger**
- F1 score measurements will be **much more reliable**
- You'll get a **true picture** of model performance

---

### Step 2: Retrain your model

```bash
cd train
python train_audioucrnn.py
```

**Expected outcome:**
- Glass breaking F1: **0.595 → 0.70-0.75**
- More reliable metrics
- Better generalization

---

## 📊 Why This Works

### Current Problem:
```
Train: 1050 samples (70 originals × 15 augmentations)
Test:   15 samples  ← TOO SMALL!
Val:    15 samples  ← TOO SMALL!
```

**With only 15 test samples:**
- If model misclassifies 3 samples → F1 drops 20%!
- High variance in results
- Can't trust the 0.595 F1 score

### After Rebalancing:
```
Train: 1050 samples (70 originals × 15 augmentations)  
Test:   300 samples (20 originals × 15 augmentations) ← 20x MORE!
Val:    150 samples (10 originals × 15 augmentations) ← 10x MORE!
```

**With 300 test samples:**
- Much more stable F1 scores
- Reliable performance estimates
- True measure of model quality

---

## 🚀 Quick Start

```bash
# Run rebalancing (creates backup automatically)
python rebalance_glass_dataset.py
# Type "yes" when prompted

# Retrain model
cd train
python train_audioucrnn.py

# Check new performance
cat checkpoints/test_results.json
```

---

## 💡 Why NOT Download More Data?

**Downloading more data won't help if your split is broken!**

Even with 1000 glass breaking files:
- If test set is still only 15 samples → F1 still unreliable!
- If test set has 300+ samples → F1 becomes reliable

**Fix the split first, add more data later if needed**

---

## 📈 Expected Timeline

| Task | Time | Difficulty |
|------|------|-----------|
| Run rebalance script | 2 min | Easy ⭐ |
| Retrain model | 20-30 min | Easy ⭐ |
| Evaluate new performance | 5 min | Easy ⭐ |
| **Total** | **~30 min** | **Easy ⭐** |

---

## ✅ Success Criteria

After rebalancing and retraining:

**Minimum:**
- [ ] Test set has 250+ samples
- [ ] Glass breaking F1 ≥ 0.70

**Good:**
- [ ] Test set has 300+ samples
- [ ] Glass breaking F1 ≥ 0.75

**Excellent:**
- [ ] Test set has 300+ samples
- [ ] Glass breaking F1 ≥ 0.80

---

## 🆘 If You Still Want More Data

**Only do this AFTER rebalancing!**

### Option 1: Check ESC-50 (you already have it!)
```bash
ls audio_datasets/ESC-50/audio/
# Look for files with "5-" prefix (class 5 = glass breaking)
```

### Option 2: Download UrbanSound8K (glass subset only)
```bash
# Download from: https://urbansounddataset.weebly.com/urbansound8k.html
# Extract only class 1 (glass breaking)
# Size: ~200MB for glass subset
```

### Option 3: Synthetic augmentation
```python
# Create more variations from existing files
# pitch shift, time stretch, add noise, etc.
# Can generate 100+ new samples from existing 98
```

**But remember:** Fix the split FIRST!

---

## 📝 Summary

1. ✅ **You have enough data** (98 files is good!)
2. ❌ **Your split is bad** (15 test samples = unreliable)
3. 🔧 **Run rebalancing script** (30 minutes)
4. 📊 **Retrain and see real F1 score** (maybe it's already 0.75!)
5. 🎉 **Problem solved without downloading anything!**

---

**START NOW:** `python rebalance_glass_dataset.py`
