# 🚨 Audio Alert Classification (6 Classes)

Production-ready FastAPI service using fine‑tuned **Audio Spectrogram Transformer (AST)** for real‑time classification of six safety-critical audio categories.

**Classes:** `alert_sounds`, `collision_sounds`, `emergency_sirens`, `environmental_sounds`, `human_scream`, `road_traffic`

---
## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Audio Classification Pipeline                    │
└─────────────────────────────────────────────────────────────────────────┘

    Audio Input (.wav, .mp3, etc.)
           │
           ▼
    ┌──────────────────┐
    │  Preprocessing   │  • Resample → 16 kHz
    │   (librosa)      │  • Mono conversion
    │                  │  • Normalize amplitude
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │   Mel-Spectrogram │  • 128 mel bins
    │   Feature Extrac. │  • Pad/truncate to fixed length
    │                   │  • Log-scale transformation
    └────────┬──────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │  Audio Spectrogram Transformer (AST) │
    │  ────────────────────────────────────│
    │  • Patch Embedding (16×16 patches)   │
    │  • Positional Encoding               │
    │  • 12 Transformer Encoder Layers     │
    │    ├─ Multi-Head Self-Attention      │
    │    └─ Feed-Forward Networks          │
    │  • Fine-tuned Classification Head    │
    └────────┬─────────────────────────────┘
             │
             ▼
    ┌──────────────────┐
    │  Softmax Layer   │  • 6-class probabilities
    │                  │  • Confidence scores
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Severity Mapping │  • Threshold-based logic
    │                  │  • Critical class detection
    │                  │  • Confidence weighting
    └────────┬─────────┘
             │
             ▼
    JSON Response:
    {
      "class": "emergency_sirens",
      "confidence": 0.982,
      "severity": "severe",
      "all_probabilities": {...}
    }
```

**Key Components:**
- **Base Model:** MIT/ast-finetuned-audioset-10-10-0.4593 (pretrained on 2M AudioSet clips)
- **Fine-tuning:** 2 epochs on 6-class dataset (~6,843 training samples)
- **Inference:** CPU-optimized with ~200ms latency per audio file
- **Classes:** alert_sounds, collision_sounds, emergency_sirens, environmental_sounds, human_scream, road_traffic

---
## ✅ Current Model
- **Architecture:** Audio Spectrogram Transformer (AST)
- **Model Directory:** `ast_6class_model/` 
- **Test Accuracy:** 98.34% (1,206 test files)
- **Macro F1 Score:** 0.981
- **Average Confidence:** 99.82% (correct predictions)

---
## 🚀 Quick Start
```powershell
cd Audio_Models
set FORCE_CPU=1
python -m uvicorn multi_model_api:app --host 127.0.0.1 --port 8010
```
Open: http://127.0.0.1:8010 (dashboard in `index.html`)

Classify via curl:
```bash
curl -X POST http://127.0.0.1:8010/classify -F "file=@sample.wav"
```

Health check:
```bash
curl http://127.0.0.1:8010/health
```

---
## 📦 Project Structure
```
cmpe-281-models/
├── README.md
├── requirements_ast.txt            # Dependencies for training/inference
├── training_log.txt                # Latest fine-tune results
├── train_ast_model.py              # Fine-tuning script (6-class)
├── ast_6class_model/               # Fine-tuned AST weights (98.34% accuracy)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── class_names.json
│   └── training_summary.json
├── train/
│   └── dataset/                    # train/test split (6 classes)
│       ├── train/                  # Training data
│       └── test/                   # Test data (1,206 files)
└── Audio_Models/
    ├── ast_classifier.py           # AST model wrapper
    ├── multi_model_api.py          # FastAPI server
    ├── test_6class.py              # Sampled evaluation script
    ├── test_full_dataset.py        # Comprehensive test script
    ├── analyze_confidence.py       # Confidence calibration analysis
    └── full_test_results.json      # Complete test results
```

---
## 🔔 Severity Mapping Logic
Levels (`ignore`, `info`, `warning`, `high`, `severe`) determined by confidence thresholds and critical class membership:
- Critical: `collision_sounds`, `human_scream`, `emergency_sirens`
- Important: `alert_sounds`
- Low: `environmental_sounds`, `road_traffic`
See `multi_model_api.py` and `evaluate_samples.py` for threshold definitions.

Planned Enhancement: optional top‑2 escalation if a critical class appears as runner‑up with moderate confidence.

---
## 🏋️ Fine-Tuning
Retrain the model with custom parameters:
```powershell
$env:FORCE_CPU='1'
$env:BATCH_SIZE='8'
$env:EPOCHS='2'
$env:LR='5e-5'
python train_ast_model.py
```
Model artifacts saved to `ast_6class_model/` (weights, classification report, training_summary.json).

---
## 🧪 Model Evaluation

**Sampled Test (30 files per class):**
```powershell
cd Audio_Models
python test_6class.py --model ../ast_6class_model --data-root ../train/dataset/test --limit 30
```

**Comprehensive Test (ALL 1,206 test files):**
```powershell
python test_full_dataset.py --model ../ast_6class_model --test-root ../train/dataset/test
```

**Confidence Analysis:**
```powershell
python analyze_confidence.py
```

---
## 🔄 Updating the Model
1. Fine‑tune → new weights saved to `ast_6class_model/`
2. Restart server: `python -m uvicorn multi_model_api:app --host 127.0.0.1 --port 8010`
3. Verify via `/health` that AST model loaded successfully

---
## 📥 Dataset Expectations
- Audio: mono or stereo (auto downmix), resampled to 16 kHz
- Clip length: truncated/padded to 5–10s (configurable)
- Balanced train/test after merging collision sources.

---
## 🔧 Dependencies
Install all required dependencies:
```powershell
pip install -r requirements_ast.txt
```

---
## 📊 Performance Metrics

**Test Set Results (1,206 files):**
- **Overall Accuracy:** 98.34%
- **Macro F1 Score:** 0.981
- **Average Confidence:** 99.82% (correct predictions)

**Per-Class Accuracy:**
- alert_sounds: 92.2%
- collision_sounds: 100%
- emergency_sirens: 96.9%
- environmental_sounds: 100%
- human_scream: 99.5%
- road_traffic: 99.3%

---
## 🔜 Future Enhancements
- Top-2 severity escalation for close decisions
- Confidence-based adaptive thresholds
- Real-time model updates via feedback loop
- GPU acceleration support

---

## 📄 License & Attribution

This project uses:
- **PyTorch** (Meta AI)
- **Transformers** (Hugging Face)
- **Librosa** (Brian McFee et al.)
- **FastAPI** (Sebastián Ramírez)
- **AST Model** (Yuan Gong, MIT)

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| **Start API** | `cd Audio_Models; $env:FORCE_CPU='1'; python -m uvicorn multi_model_api:app --host 127.0.0.1 --port 8010` |
| **Test Classify** | `curl -X POST http://localhost:8010/classify -F "file=@audio.wav"` |
| **Health Check** | `curl http://localhost:8010/health` |
| **API Docs** | `http://localhost:8010/docs` |
| **Train Model** | `$env:FORCE_CPU='1'; python train_ast_model.py` |
| **Test Sampled** | `cd Audio_Models; python test_6class.py --model ../ast_6class_model --limit 30` |
| **Test Full** | `python test_full_dataset.py --model ../ast_6class_model --test-root ../train/dataset/test` |
| **Analyze Confidence** | `python analyze_confidence.py` |

---

**Last Updated:** November 15, 2025  
**Version:** 3.0.0  
**Model:** AST 6-Class (98.34% accuracy)  
**Status:** ✅ Production Ready
