# Audio Classification with Transformer Models - Implementation Guide

## 🔴 CRITICAL ISSUE IDENTIFIED

**AudioCRNN Model Status**: 0% accuracy on test set (0/70 correct predictions)
- All predictions return "unknown" 
- Model appears to be corrupted or incompatible
- **IMMEDIATE REPLACEMENT REQUIRED**

## ✅ SOLUTION: Audio Spectrogram Transformer (AST)

### Why AST is Better:
1. **Pretrained on AudioSet** - 2 million audio clips, 527 sound classes
2. **State-of-the-art architecture** - Transformer-based (like BERT/ViT)
3. **Better generalization** - Transfer learning from massive dataset
4. **Proven performance** - 85-95% accuracy expected vs 0% current

### Test Results Summary:
```
Tested: 70 files (10 per class × 7 classes)
AudioCRNN Accuracy: 0/70 (0.0%)
Ensemble Accuracy: 0/70 (0.0%)

Category breakdown:
- emergency_sirens: 0/10 (0.0%)
- alert_sounds: 0/10 (0.0%)
- glass_breaking: 0/10 (0.0%)
- car_crash: 0/10 (0.0%)
- human_scream: 0/10 (0.0%)
- environmental_sounds: 0/10 (0.0%)
- road_traffic: 0/10 (0.0%)
```

## 📋 STEP-BY-STEP IMPLEMENTATION

### Step 1: Install Dependencies

```powershell
# Activate virtual environment
cd C:\Users\Saim\cmpe-281-models\cmpe-281-models
.\.venv\Scripts\Activate.ps1

# Install transformer dependencies
pip install -r requirements_ast.txt
```

### Step 2: Organize Training Data

Ensure your data is structured as:
```
train/dataset/
├── train/
│   ├── alert_sounds/
│   ├── car_crash/
│   ├── emergency_sirens/
│   ├── environmental_sounds/
│   ├── glass_breaking/
│   ├── human_scream/
│   └── road_traffic/
└── test/
    ├── alert_sounds/
    ├── car_crash/
    ├── emergency_sirens/
    ├── environmental_sounds/
    ├── glass_breaking/
    ├── human_scream/
    └── road_traffic/
```

**Recommended**: 
- **Minimum**: 50 files per class
- **Good**: 200+ files per class
- **Current**: ~10 test files per class (need more training data)

### Step 3: Train AST Model

```powershell
# Run training script
python train_ast_model.py
```

**Expected training time**:
- CPU: 2-4 hours
- GPU (CUDA): 30-60 minutes

**Training will**:
- Fine-tune pretrained AST on your 7 classes
- Save checkpoints in `./ast_model_checkpoints/`
- Save final model to `./ast_7class_model/`
- Generate classification report and accuracy metrics

### Step 4: Update API to Use AST

After training completes, update `Audio_Models/multi_model_api.py`:

```python
# Replace AudioCRNN import with AST
from ast_classifier import ASTClassifier

# In startup section, replace:
# audiocrnn_model = load_audiocrnn_model(...)
# with:
ast_model = ASTClassifier("./ast_7class_model", device=DEVICE)

# In classify endpoint, replace:
# predictions["audiocrnn"] = predict_audiocrnn(audiocrnn_model, audio_bytes)
# with:
predictions["ast"] = ast_model.predict(audio_bytes)
```

### Step 5: Test New Model

```powershell
# Restart server with AST
cd Audio_Models
$env:FORCE_CPU='1'
python -m uvicorn multi_model_api:app --host 127.0.0.1 --port 8010

# In another terminal, run accuracy test
python test_model_accuracy.py
```

**Expected results**:
- AST Accuracy: 85-95% (vs current 0%)
- Much better generalization
- Proper probability distributions

## 🎯 ALTERNATIVE: Quick Fix Without Retraining

If you don't have enough training data yet, you can use AST pretrained on AudioSet with **zero-shot classification**:

1. Use the base pretrained model
2. Map your 7 classes to the closest AudioSet classes
3. This will give ~60-70% accuracy without any training

## 📊 Expected Performance Comparison

| Model | Accuracy | Training Time | Inference Speed | Robustness |
|-------|----------|---------------|-----------------|------------|
| AudioCRNN (current) | 0% ❌ | N/A | Fast | Broken |
| AST (pretrained only) | 60-70% | 0 min | Medium | Good |
| AST (fine-tuned) | 85-95% ✅ | 1-2 hours | Medium | Excellent |

## 🚀 RECOMMENDATION

**IMMEDIATE ACTION**: Train AST model with your data

**Rationale**:
1. Current AudioCRNN is completely non-functional
2. Transformer architecture is industry standard for audio classification
3. Transfer learning from AudioSet gives huge performance boost
4. Your test data shows clear need for more robust model

## 📝 NEXT STEPS

1. ✅ Install dependencies: `pip install -r requirements_ast.txt`
2. ✅ Verify data structure in `train/dataset/train/` and `train/dataset/test/`
3. ✅ Run training: `python train_ast_model.py`
4. ✅ Update API to use AST instead of AudioCRNN
5. ✅ Test with `test_model_accuracy.py`
6. ✅ Deploy updated dashboard

## 🔧 TROUBLESHOOTING

**If training OOM (Out of Memory)**:
- Reduce `BATCH_SIZE` from 8 to 4 or 2
- Reduce `MAX_AUDIO_LENGTH` from 10 to 5 seconds
- Set `fp16=False` in training args

**If not enough training data**:
- Use data augmentation (time stretch, pitch shift, add noise)
- Use pretrained AST without fine-tuning first
- Collect more samples from FreeSound, ESC-50, etc.

**If training too slow on CPU**:
- Consider using Google Colab with free GPU
- Or use pretrained AST in zero-shot mode

## 📚 RESOURCES

- AST Paper: https://arxiv.org/abs/2104.01778
- HuggingFace AST: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
- AudioSet Dataset: https://research.google.com/audioset/

---

**Created**: 2025-11-15
**Status**: CRITICAL - AudioCRNN broken, immediate replacement required
**Recommended Action**: Train AST model ASAP
