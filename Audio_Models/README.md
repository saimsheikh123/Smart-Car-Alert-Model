# Audio Classification API

A FastAPI-based web service for classifying audio files using a fine-tuned Audio Spectrogram Transformer (AST) model into **6 safety-critical categories**:
- **Alert Sounds** (Alarms, beeps, warning signals)
- **Collision Sounds** (Car crashes, impacts, glass breaking combined)
- **Emergency Sirens** (Police, ambulance, fire truck sirens)
- **Environmental Sounds** (Rain, wind, nature, background ambience)
- **Human Scream** (Distress calls, screams)
- **Road Traffic** (Normal traffic noise, engines, horns)

## Features

- **High Accuracy**: 98.34% on full test set (1,206 files)
- **Well-Calibrated**: 99.82% average confidence on correct predictions
- **Transformer-Based**: Uses state-of-the-art Audio Spectrogram Transformer (AST)
- **Real-time Classification**: Fast inference via HTTP API
- **Multiple Audio Formats**: Supports WAV, MP3, FLAC, and more
- **Detailed Outputs**: Returns class probabilities and confidence scores
- **Production-Ready**: Comprehensive error handling and health monitoring

## Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager
- PyTorch (CPU or GPU)

### Installation

1. **Navigate to the project directory:**

   ```powershell
   cd C:\Users\Saim\cmpe-281-models\cmpe-281-models
   ```

2. **Install dependencies:**

   ```powershell
   pip install -r requirements_ast.txt
   ```

   Or, if using a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements_ast.txt
   ```

3. **Model location:**

   The fine-tuned AST model is located in `../ast_6class_model/`. The server automatically loads this model on startup.

## Running the Server

### Quick Start (CPU)

From PowerShell in the Audio_Models directory:

```powershell
cd C:\Users\Saim\cmpe-281-models\cmpe-281-models\Audio_Models
$env:FORCE_CPU='1'
python -m uvicorn multi_model_api:app --host 127.0.0.1 --port 8010
```

You should see output like:

```
[STARTUP] Audio Classification API (Transformer)
[AST] Loading model from ../ast_6class_model...
[AST] Model loaded successfully. Classes: ['alert_sounds', 'collision_sounds', ...]
[OK] AST model ready
INFO:     Uvicorn running on http://127.0.0.1:8010
```

### With Auto-Reload (Development)

```powershell
$env:FORCE_CPU='1'
python -m uvicorn multi_model_api:app --host 127.0.0.1 --port 8010 --reload
```

### GPU Acceleration

If you have a CUDA-capable GPU, omit the `FORCE_CPU` variable:

```powershell
python -m uvicorn multi_model_api:app --host 127.0.0.1 --port 8010
```

## API Usage

Once the server is running, you can access:

- **API Docs (Interactive):** http://localhost:8010/docs
- **ReDoc (Alternative Docs):** http://localhost:8010/redoc

### Endpoints

#### 1. **GET /health**

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model": "AST",
  "device": "cpu"
}
```

#### 2. **GET /**

Root endpoint with API info.

**Response:**
```json
{
  "service": "Audio Classification API (Transformer)",
  "model": "AST (Audio Spectrogram Transformer) 6-class",
  "version": "1.0.0",
  "classes": [
    "alert_sounds",
    "collision_sounds",
    "emergency_sirens",
    "environmental_sounds",
    "human_scream",
    "road_traffic"
  ]
}
```

#### 3. **POST /classify**

Classify an audio file.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `file` (audio file, WAV recommended)

**Response (Success):**
```json
{
  "predicted_class": "emergency_sirens",
  "confidence": 0.9987,
  "severity": "critical",
  "alert_type": "Emergency",
  "reason": "Emergency vehicle approaching",
  "probabilities": {
    "alert_sounds": 0.0005,
    "collision_sounds": 0.0002,
    "emergency_sirens": 0.9987,
    "environmental_sounds": 0.0001,
    "human_scream": 0.0003,
    "road_traffic": 0.0002
  }
}
```

**Response (Error):**
```json
{
  "error": "Classification failed: <error details>"
}
```

## Testing

### Automated Test Scripts

**Test sampled files from each class:**
```powershell
cd Audio_Models
python test_6class.py --model ../ast_6class_model --data-root ../datasets/test --limit 30
```

**Comprehensive Test (ALL 1,206 test files):**
```powershell
python test_full_dataset.py --model ../ast_6class_model --test-root ../datasets/test
```

**Analyze confidence calibration:**
```powershell
python analyze_confidence.py
```

### Using cURL (PowerShell)

```powershell
# Test health endpoint
curl http://localhost:8010/health

# Upload an audio file for classification
curl -X POST -F "file=@C:\path\to\audio.wav" http://localhost:8010/classify
```

### Using Python Requests

```python
import requests

# Test health
resp = requests.get("http://localhost:8010/health")
print(resp.json())

# Classify audio
with open("siren.wav", "rb") as f:
    files = {"file": f}
    resp = requests.post("http://localhost:8010/classify", files=files)
    result = resp.json()
    print(f"Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Severity: {result['severity']}")
```

### Using FastAPI's Interactive Docs

1. Start the server (see above)
2. Open http://localhost:8010/docs in your browser
3. Click on the `/classify` endpoint
4. Click **Try it out**
5. Select your audio file using the file picker
6. Click **Execute**

## Project Structure

```
Audio_Models/
├── multi_model_api.py          # Main FastAPI application
├── ast_classifier.py           # AST model wrapper
├── test_6class.py              # Sampled test script
├── test_full_dataset.py        # Comprehensive test script
├── analyze_confidence.py       # Confidence calibration analysis
├── compare_server_local.py     # Server vs local prediction comparison
├── requirements_ast.txt        # Python dependencies
├── README.md                   # This file
└── full_test_results.json      # Latest comprehensive test results

../ast_6class_model/            # Fine-tuned AST weights (6 classes)
├── config.json
├── preprocessor_config.json
├── pytorch_model.bin
├── class_names.json
└── training_summary.json
```

## Model Details

- **Architecture:** Audio Spectrogram Transformer (AST)
- **Base Model:** MIT/ast-finetuned-audioset-10-10-0.4593
- **Classes:** 6 (alert_sounds, collision_sounds, emergency_sirens, environmental_sounds, human_scream, road_traffic)
- **Input:** Audio waveform (resampled to 16 kHz)
- **Output:** Class probabilities + confidence score
- **Fine-tuning:** 2 epochs on custom 6-class dataset

## Performance Metrics

**Test Set Performance (1,206 files):**
- Overall Accuracy: **98.34%**
- Macro F1 Score: **0.981**
- Average Confidence (correct): **99.82%**

**Per-Class Accuracy:**
- alert_sounds: 92.2%
- collision_sounds: 100%
- emergency_sirens: 96.9%
- environmental_sounds: 100%
- human_scream: 99.5%
- road_traffic: 99.3%

**Inference Speed:**
- First request: ~3-5 seconds (model loading)
- Subsequent requests: ~0.5-1 second per file (CPU)
- GPU: ~0.2-0.4 seconds per file

## Troubleshooting

### Port already in use

Check what's using port 8010:
```powershell
Get-NetTCPConnection -LocalPort 8010 | Select-Object OwningProcess
```

Kill the process or use a different port:
```powershell
python -m uvicorn multi_model_api:app --host 127.0.0.1 --port 9000
```

### Module not found errors

Ensure all dependencies are installed:
```powershell
pip install -r requirements_ast.txt --upgrade
```

### Model loading errors

The server expects the fine-tuned model at `../ast_6class_model/`. If missing, it will fall back to the pretrained AudioSet model with aggregation (lower accuracy).

### Low accuracy on your data

The model is trained on specific audio classes. For best results:
- Use audio files similar to training data
- Ensure audio is at least 1 second long
- Avoid heavily compressed or noisy audio
- Check confidence scores (low confidence may indicate out-of-distribution data)

### Memory issues

If running out of memory on CPU:
1. Reduce batch size in training scripts
2. Limit concurrent API requests
3. Use shorter audio clips

## Advanced Usage

### Optional Collision Heuristic

Enable a heuristic that promotes collision predictions when they're narrowly behind road_traffic or environmental:

```powershell
$env:COLLISION_HEURISTIC='1'
python -m uvicorn multi_model_api:app --host 127.0.0.1 --port 8010
```

### Training Your Own Model

See `../train_ast_model.py` for fine-tuning on your own dataset:

```powershell
$env:FORCE_CPU='1'
$env:BATCH_SIZE='8'
$env:EPOCHS='2'
$env:LR='5e-5'
python ../train_ast_model.py
```

## Citation

This project uses the Audio Spectrogram Transformer (AST):
```
@inproceedings{gong21b_interspeech,
  author={Yuan Gong and Yu-An Chung and James Glass},
  title={{AST: Audio Spectrogram Transformer}},
  year=2021,
  booktitle={Proc. Interspeech 2021}
}
```

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions:
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [AST Model Card](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
