"""
FastAPI application supporting multiple audio classification models.
- AudioCRNN: 3-class (Glass Break, Traffic, Car Crash)
- SirenClassifier: 2-class (Emergency/Alert vs. Ambient)
- DQN Agents: Alert, Emergency, Environmental detection
- Ensemble: Voting-based consensus
Run with: uvicorn multi_model_api:app --host 0.0.0.0 --port 8000
"""

import os
import io
import sys
import json
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Optional, List
from pathlib import Path
import warnings
from ast_classifier import ASTClassifier

warnings.filterwarnings("ignore")

# Device: allow forcing CPU via environment variable FORCE_CPU=1 or FORCE_DEVICE=cpu
if os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes") or os.environ.get("FORCE_DEVICE", "").lower() == "cpu":
    DEVICE = torch.device("cpu")
    print("[INIT] FORCE_CPU set - forcing device to CPU")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Using device: {DEVICE}")


# ============================================================================
# Model 1: AudioCRNN (3-class: Glass, Traffic, Car Crash)
# ============================================================================
class AudioCRNN(nn.Module):
    """CNN-RNN model for multi-class audio classification."""
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
# Model 2: SirenClassifier (2-class: Emergency/Alert vs. Ambient)
# ============================================================================
class TransformerEncoder(nn.Module):
    """Transformer encoder for token sequences."""
    def __init__(self, d_model=64, n_heads=4, num_layers=2, max_len=400, vocab=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.zeros(max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x):
        emb = self.embedding(x) + self.pos_encoder[:x.size(1)]
        enc = self.encoder(emb.permute(1, 0, 2))
        return enc.permute(1, 0, 2)


class SirenClassifier(nn.Module):
    """Transformer-based classifier for alert/siren detection."""
    def __init__(self, d_model=64, hidden=128, num_classes=2):
        super().__init__()
        self.backbone = TransformerEncoder(d_model=d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = self.backbone(x)
        h = h.transpose(1, 2)
        h = self.pool(h).squeeze(-1)
        h = self.act(self.fc1(h))
        return self.fc2(h)


# ============================================================================
# Audio Preprocessing Functions
# ============================================================================
def wav_to_mel(
    audio_bytes: bytes,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop: int = 512,
    fixed: int = 128
) -> np.ndarray:
    """Convert WAV bytes to mel-spectrogram [1, 1, 128, 128]."""
    try:
        data, file_sr = sf.read(io.BytesIO(audio_bytes))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if file_sr != sr:
            data = librosa.resample(data.astype("float32"), orig_sr=file_sr, target_sr=sr)
        y = data
    except:
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True, duration=2.0)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop)
    S = librosa.power_to_db(S, ref=np.max)

    if S.shape[1] > fixed:
        S = S[:, :fixed]
    else:
        S = np.pad(S, ((0, 0), (0, fixed - S.shape[1])), mode="constant")

    mn, mx = S.min(), S.max()
    if mx - mn == 0 or np.isnan(mx - mn):
        S = np.zeros_like(S)
    else:
        S = (S - mn) / (mx - mn)

    return S.astype("float32")[None, None, ...]  # [1, 1, 128, 128]


def wav_to_mfcc(
    audio_bytes: bytes,
    sr: int = 16000,
    n_mfcc: int = 13,
    max_len: int = 400
) -> np.ndarray:
    """Convert WAV bytes to MFCC tokens [1, max_len]."""
    try:
        data, file_sr = sf.read(io.BytesIO(audio_bytes))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if file_sr != sr:
            data = librosa.resample(data.astype("float32"), orig_sr=file_sr, target_sr=sr)
        y = data
    except:
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True, duration=2.0)

    # Limit to 1 second for consistency
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)), mode='constant')
    else:
        y = y[:sr]

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T  # [time, n_mfcc]


# ============================================================================
# Model Loader Functions
# ============================================================================
def load_audiocrnn(model_path: str = "multi_audio_crnn.pth") -> Optional[AudioCRNN]:
    """Load AudioCRNN model. If a checkpoint exists, infer num_classes from the final linear layer
    in the checkpoint (fc2) and instantiate the model accordingly before loading weights.
    """
    # Default to 7-class model used in training
    default_classes = 7

    if os.path.exists(model_path):
        try:
            raw = torch.load(model_path, map_location=DEVICE)
            # raw may be a dict with metadata or a plain state_dict
            if isinstance(raw, dict) and 'model_state_dict' in raw:
                state = raw['model_state_dict']
            elif isinstance(raw, dict) and all(isinstance(k, str) for k in raw.keys()):
                # assume this is a state_dict
                state = raw
            else:
                raise ValueError('Unrecognized checkpoint format')

            # Infer num_classes from fc2.weight if available
            fc_key_candidates = [k for k in state.keys() if k.endswith('fc2.weight')]
            if fc_key_candidates:
                fc_key = fc_key_candidates[0]
                num_classes = state[fc_key].shape[0]
                print(f"[INFO] Inferred AudioCRNN num_classes={num_classes} from checkpoint key: {fc_key}")
            else:
                num_classes = default_classes
                print(f"[INFO] Could not find fc2 weight in checkpoint, defaulting to num_classes={num_classes}")

            model = AudioCRNN(num_classes=num_classes).to(DEVICE)
            model.load_state_dict(state)
            print(f"[OK] AudioCRNN loaded from {model_path}")
            return model
        except Exception as e:
            print(f"[WARN] Failed to load AudioCRNN from {model_path}: {e}")
    else:
        print(f"[INFO] AudioCRNN checkpoint not found: {model_path} (using random init)")

    # Fallback model
    model = AudioCRNN(num_classes=default_classes).to(DEVICE)
    return model


def load_siren_classifier(model_path: str = "alert-reinforcement_model.pth") -> Optional[SirenClassifier]:
    """Load SirenClassifier model."""
    model = SirenClassifier(d_model=64, hidden=128, num_classes=2).to(DEVICE)
    if os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state)
            print(f"[OK] SirenClassifier loaded from {model_path}")
            return model
        except Exception as e:
            print(f"[WARN] Failed to load SirenClassifier from {model_path}: {e}")
    else:
        print(f"[INFO] SirenClassifier checkpoint not found: {model_path} (using random init)")
    return model


def load_dqn_agent(agent_dir: str):
    """Load DQN agent from TorchScript traced model (bypasses Stable-Baselines3 Windows permission issues)."""
    try:
        if os.path.isdir(agent_dir):
            # Try TorchScript first (exported via export_dqn_to_onnx.py)
            agent_name = os.path.basename(agent_dir)
            torchscript_path = os.path.join(agent_dir, f"{agent_name}_traced.pt")
            
            if os.path.exists(torchscript_path):
                try:
                    model = torch.jit.load(torchscript_path, map_location=DEVICE)
                    print(f"[OK] DQN agent loaded from TorchScript: {torchscript_path}")
                    return model
                except Exception as e:
                    print(f"[WARN] Failed to load TorchScript: {e}")
            
            # Fallback: try Stable-Baselines3 (will likely fail on Windows)
            try:
                from stable_baselines3 import DQN
                policy_path = os.path.join(agent_dir, "policy.pth")
                if os.path.exists(policy_path):
                    model = DQN.load(agent_dir, device=DEVICE)
                    print(f"[OK] DQN agent loaded from Stable-Baselines3: {agent_dir}")
                    return model
            except PermissionError as pe:
                print(f"[WARN] PermissionError with Stable-Baselines3 (expected on Windows): {pe}")
                return None
            except Exception as sb_err:
                print(f"[WARN] Stable-Baselines3 load failed: {type(sb_err).__name__}")
                return None
            
            print(f"[WARN] No DQN model found in {agent_dir}")
            return None
        else:
            print(f"[WARN] Agent directory does not exist: {agent_dir}")
            return None
    except Exception as e:
        print(f"[WARN] Failed to load DQN agent from {agent_dir}: {e}")
        return None


# ============================================================================
# Inference Functions
# ============================================================================
def predict_audiocrnn(model: AudioCRNN, audio_bytes: bytes) -> Dict:
    """Predict with AudioCRNN."""
    try:
        mel = wav_to_mel(audio_bytes)
        mel_tensor = torch.from_numpy(mel).to(DEVICE)
        with torch.no_grad():
            logits = model(mel_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Updated 7-class label ordering used during training
        classes = [
            "alert_sounds",
            "car_crash",
            "emergency_sirens",
            "environmental_sounds",
            "glass_breaking",
            "human_scream",
            "road_traffic",
        ]
        idx = int(np.argmax(probs))
        # Compute top-2 classes for UI
        order = np.argsort(probs)[::-1]
        top2 = [
            {"class": classes[int(order[0])], "confidence": float(probs[int(order[0])])},
            {"class": classes[int(order[1])], "confidence": float(probs[int(order[1])])},
        ]
        # Debug: log all class probabilities to help diagnose prediction issues
        prob_str = ", ".join([f"{c}: {p:.1%}" for c, p in zip(classes, probs)])
        print(f"[AudioCRNN] Predicted: {classes[idx]} ({probs[idx]:.1%}) | All: {prob_str}")
        return {
            "model": "AudioCRNN",
            "class": classes[idx],
            "confidence": float(probs[idx]),
            "all_probs": {c: float(p) for c, p in zip(classes, probs)},
            "top2": top2,
            "success": True
        }
    except Exception as e:
        print(f"[ERROR] AudioCRNN prediction failed: {e}")
        return {"model": "AudioCRNN", "success": False, "error": str(e)}


def predict_siren_classifier(model: SirenClassifier, audio_bytes: bytes) -> Dict:
    """Predict with SirenClassifier."""
    try:
        mfcc = wav_to_mfcc(audio_bytes)
        
        # For simplicity, tokenize using energy-based bucketing (replace with KMeans if needed)
        rms = np.sqrt(np.mean(mfcc**2, axis=1))
        tokens = np.digitize(rms, np.linspace(rms.min(), rms.max(), 64)) - 1
        tokens = np.clip(tokens, 0, 63)
        
        # Pad or crop to 400
        if len(tokens) > 400:
            tokens = tokens[:400]
        else:
            tokens = np.pad(tokens, (0, 400 - len(tokens)), constant_values=0)
        
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tokens_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        classes = ["ambient", "alert_emergency"]
        idx = int(np.argmax(probs))
        return {
            "model": "SirenClassifier",
            "class": classes[idx],
            "confidence": float(probs[idx]),
            "all_probs": {c: float(p) for c, p in zip(classes, probs)},
            "success": True
        }
    except Exception as e:
        print(f"[ERROR] SirenClassifier prediction failed: {e}")
        return {"model": "SirenClassifier", "success": False, "error": str(e)}


def predict_dqn_agent(agent, audio_bytes: bytes, agent_name: str = "unknown") -> Dict:
    """Predict with DQN agent (TorchScript or Stable-Baselines3).

    Many DQN policies were trained on a specific feature vector length (e.g., 128 or 512).
    We adapt the input by downsampling a flattened mel-spectrogram to the expected size.
    """
    try:
        # Helper: build observation of target length from mel-spectrogram
        def build_obs(target_dim: int) -> torch.Tensor:
            mel = wav_to_mel(audio_bytes)  # [1,1,128,128]
            vec = mel.reshape(-1).astype("float32")  # 16384
            n = vec.shape[0]
            if target_dim <= 0:
                target_dim = n
            if n == target_dim:
                out = vec
            elif n > target_dim:
                # Average-pool into target_dim bins
                # Compute boundaries so all elements are used
                edges = np.linspace(0, n, num=target_dim + 1, dtype=np.int64)
                out = np.empty((target_dim,), dtype=np.float32)
                for i in range(target_dim):
                    s, e = edges[i], edges[i + 1]
                    if e > s:
                        out[i] = float(np.mean(vec[s:e]))
                    else:
                        out[i] = 0.0
            else:  # n < target_dim -> pad with zeros
                out = np.zeros((target_dim,), dtype=np.float32)
                out[:n] = vec
            # Normalize
            m = float(np.mean(out))
            sd = float(np.std(out))
            if sd < 1e-6:
                sd = 1e-6
            out = (out - m) / sd
            return torch.from_numpy(out[None, :]).to(DEVICE)

        # Determine expected input size
        target_dim = None
        try:
            # TorchScript models retain parameter names we set (fc1.weight)
            sd = agent.state_dict()
            for k, v in sd.items():
                if k.endswith("fc1.weight") and v.dim() == 2:
                    target_dim = int(v.shape[1])
                    break
        except Exception:
            target_dim = None

        obs = build_obs(target_dim or 16384).float()

        # TorchScript vs SB3
        if hasattr(agent, 'graph') or isinstance(agent, torch.jit.ScriptModule) or isinstance(agent, torch.jit.RecursiveScriptModule):
            q_values = agent(obs)
            action = int(torch.argmax(q_values, dim=1)[0])
        else:
            action, _ = agent.predict(obs, deterministic=True)
            action = int(action)

        return {
            "model": f"DQN_{agent_name}",
            "action": action,
            "action_label": "ALERT" if action == 1 else "WAIT",
            "success": True
        }
    except Exception as e:
        print(f"[ERROR] DQN prediction failed ({agent_name}): {e}")
        return {"model": f"DQN_{agent_name}", "success": False, "error": str(e)}


# ============================================================================
# FastAPI Lifespan & Setup
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load AST model at startup and disable legacy models."""
    print("\n" + "="*70)
    print("[STARTUP] Audio Classification API (Transformer)")
    print("="*70)

    # Disable legacy models (kept for backward compatibility in code paths)
    app.state.audiocrnn = None
    app.state.siren_classifier = None
    app.state.alert_dqn = None
    app.state.emergency_dqn = None
    app.state.environmental_dqn = None

    # Prefer locally fine-tuned model, else fall back to pretrained AudioSet model
    ast_local_dir = Path(__file__).parent.parent / "ast_6class_model"
    if ast_local_dir.exists():
        ast_source = str(ast_local_dir)
    else:
        ast_source = "MIT/ast-finetuned-audioset-10-10-0.4593"
        print(f"[INFO] Local AST model not found at {ast_local_dir}. Using pretrained {ast_source} with keyword-mapped aggregation.")

    try:
        app.state.ast = ASTClassifier(ast_source, device=str(DEVICE))
        print("[OK] AST model ready")
    except Exception as e:
        print(f"[ERROR] Failed to load AST model: {e}")
        app.state.ast = None

    yield

    print("\n" + "="*70)
    print("[SHUTDOWN] Audio Classification API (Transformer)")
    print("="*70)


# ============================================================================
# Create FastAPI App
# ============================================================================
app = FastAPI(
    title="Audio Classification API (Transformer)",
    description="Single-model Transformer classifier (AST) with severity levels",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files (index.html + assets)
# Mount static assets under /static and serve index.html at /
static_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(static_dir), html=False), name="static")



# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/")
async def root():
    """Serve the frontend index.html file.

    The static assets are mounted under `/static/`. The root path returns
    the `index.html` file located next to this module so visiting
    `http://<host>:<port>/` serves the web UI.
    """
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse(status_code=404, content={"error": "index.html not found"})


@app.get("/api")
async def api_info():
    """API information (kept for programmatic access)."""
    return {
        "message": "Audio Classification API (Transformer)",
        "version": "3.0.0",
        "models": [
            "AST (Audio Spectrogram Transformer) 6-class via fine-tune or aggregation"
        ],
        "endpoints": {
            "POST /classify": "Transformer prediction with severity",
            "POST /classify/crnn": "(legacy) AudioCRNN only",
            "POST /classify/siren": "(legacy) SirenClassifier only",
            "POST /classify/dqn": "(legacy) DQN agents only",
            "GET /health": "Health check",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    models_loaded = {
        "ast": getattr(app.state, "ast", None) is not None,
    }
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models_loaded": models_loaded,
    }


@app.post("/classify")
async def classify_ensemble(
    file: UploadFile,
):
    """Single-model prediction using AST with severity mapping."""
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            return JSONResponse(status_code=400, content={"error": "Empty audio file"})
        if not getattr(app.state, "ast", None):
            return JSONResponse(status_code=503, content={"error": "AST model not loaded"})

        ast_result = app.state.ast.predict(audio_bytes)

        # Severity levels (1-5) + label
        # Critical classes: collision_sounds, human_scream, emergency_sirens
        cls = ast_result.get("class", "unknown")
        conf = float(ast_result.get("confidence", 0.0))

        CRITICAL = {"collision_sounds", "human_scream", "emergency_sirens"}
        IMPORTANT = {"alert_sounds"}
        LOW = {"environmental_sounds", "road_traffic"}

        level = 1
        label = "ignore"
        justification = "Low confidence or low-importance class"
        if cls in CRITICAL:
            if conf >= 0.85:
                level, label = 5, "severe"
                justification = "High-confidence critical event"
            elif conf >= 0.60:
                level, label = 4, "high"
                justification = "Likely critical event"
            elif conf >= 0.40:
                level, label = 3, "warning"
                justification = "Possible critical event"
            else:
                level, label = 2, "info"
                justification = "Low-confidence critical class"
        elif cls in IMPORTANT:
            if conf >= 0.70:
                level, label = 3, "warning"
                justification = "Attention recommended"
            elif conf >= 0.50:
                level, label = 2, "info"
                justification = "Moderate confidence alert sound"
            else:
                level, label = 1, "ignore"
                justification = "Low confidence"
        elif cls in LOW:
            if conf >= 0.70:
                level, label = 2, "info"
                justification = "Ambient/non-actionable"
            else:
                level, label = 1, "ignore"
                justification = "Background noise"
        else:
            # unknown
            if conf >= 0.60:
                level, label = 2, "info"
                justification = "Unmapped but confident"

        ensemble_top2 = ast_result.get("top2", [])

        models_loaded = {"ast": True}

        return {
            "file_name": file.filename,
            "models_loaded": models_loaded,
            "ensemble": {
                "primary_class": cls,
                "final_prediction": cls,
                "confidence": conf,
                "votes": {cls: 1},
                "num_models": 1,
                "top2": ensemble_top2,
            },
            "detailed_predictions": {
                "ast": {
                    "prediction": cls,
                    "confidence": conf,
                    "all_probabilities": ast_result.get("all_probs", {}),
                    "top2": ensemble_top2,
                    "success": ast_result.get("success", True),
                }
            },
            "severity": {
                "level": label,
                "level_num": level,
                "justification": justification,
            },
            "most_likely": (ensemble_top2[0] if len(ensemble_top2) > 0 else None),
            "second_most_likely": (ensemble_top2[1] if len(ensemble_top2) > 1 else None)
        }
    
    except Exception as e:
        print(f"[ERROR] Ensemble classification failed: {e}")
        return JSONResponse(status_code=500, content={"error": f"Classification failed: {str(e)}"})


@app.post("/classify/crnn")
async def classify_crnn(file: UploadFile):
    """AudioCRNN prediction only."""
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            return JSONResponse(status_code=400, content={"error": "Empty audio file"})
        
        if not app.state.audiocrnn:
            return JSONResponse(status_code=503, content={"error": "AudioCRNN model not loaded"})
        
        result = predict_audiocrnn(app.state.audiocrnn, audio_bytes)
        return result
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/classify/siren")
async def classify_siren(file: UploadFile):
    """SirenClassifier prediction only."""
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            return JSONResponse(status_code=400, content={"error": "Empty audio file"})
        
        if not app.state.siren_classifier:
            return JSONResponse(status_code=503, content={"error": "SirenClassifier model not loaded"})
        
        result = predict_siren_classifier(app.state.siren_classifier, audio_bytes)
        return result
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/classify/dqn")
async def classify_dqn(file: UploadFile):
    """DQN agents prediction only."""
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            return JSONResponse(status_code=400, content={"error": "Empty audio file"})
        
        predictions = {}
        
        if app.state.alert_dqn:
            predictions["alert"] = predict_dqn_agent(app.state.alert_dqn, audio_bytes, "alert")
        if app.state.emergency_dqn:
            predictions["emergency"] = predict_dqn_agent(app.state.emergency_dqn, audio_bytes, "emergency")
        if app.state.environmental_dqn:
            predictions["environmental"] = predict_dqn_agent(app.state.environmental_dqn, audio_bytes, "environmental")
        
        if not predictions:
            return JSONResponse(status_code=503, content={"error": "No DQN models loaded"})
        
        return {"dqn_predictions": predictions}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ============================================================================
# User Feedback System for Misclassification Tracking
# ============================================================================

SOUND_CLASSES = {
    "alert_sounds": "Alert Sounds",
    "collision_sounds": "Collision/Impact",
    "human_scream": "Human Scream",
    "road_traffic": "Road Traffic",
    "other": "Other/Unsure"
}

CORRECTIONS_FILE = Path("user_corrections.json")
CORRECTIONS_AUDIO_DIR = Path("corrections_audio")

def load_corrections():
    """Load all user corrections from file."""
    if CORRECTIONS_FILE.exists():
        with open(CORRECTIONS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_corrections(corrections: list):
    """Save corrections to file."""
    CORRECTIONS_FILE.parent.mkdir(exist_ok=True)
    with open(CORRECTIONS_FILE, 'w') as f:
        json.dump(corrections, f, indent=2)

def calculate_adaptive_thresholds(corrections: list) -> Dict[str, float]:
    """
    Calculate per-class confidence thresholds based on correction history.
    For each class, find the minimum confidence of CORRECT predictions
    and maximum confidence of INCORRECT predictions to derive optimal thresholds.
    """
    thresholds = {}
    
    if len(corrections) < 5:
        # Not enough data; return defaults
        return {
            "alert_sounds": 0.60,
            "collision_sounds": 0.65,
            "emergency_sirens": 0.65,
            "environmental_sounds": 0.55,
            "human_scream": 0.60,
            "road_traffic": 0.65,
        }
    
    from collections import defaultdict
    
    # Organize by predicted class
    class_predictions = defaultdict(lambda: {"correct": [], "incorrect": []})
    
    for correction in corrections:
        predicted = correction.get("predicted", "")
        actual = correction.get("actual", "")
        confidence = correction.get("confidence", 0.0)
        
        if predicted and actual:
            is_correct = (predicted == actual)
            if is_correct:
                class_predictions[predicted]["correct"].append(confidence)
            else:
                class_predictions[predicted]["incorrect"].append(confidence)
    
    # Calculate threshold for each class
    for class_name in class_predictions:
        correct_confs = class_predictions[class_name]["correct"]
        incorrect_confs = class_predictions[class_name]["incorrect"]
        
        if correct_confs and incorrect_confs:
            # Threshold = midpoint between min(correct) and max(incorrect)
            min_correct = min(correct_confs)
            max_incorrect = max(incorrect_confs)
            threshold = (min_correct + max_incorrect) / 2
            thresholds[class_name] = max(0.5, min(threshold, 0.9))  # Clamp to [0.5, 0.9]
        elif correct_confs:
            # All predictions were correct; lower threshold slightly
            thresholds[class_name] = min(correct_confs) - 0.05
        else:
            # All predictions were wrong; raise threshold
            thresholds[class_name] = 0.80
    
    return thresholds

def save_correction_audio(audio_bytes: bytes, actual_class: str, predicted_class: Optional[str]) -> str:
    """Save audio file for correction analysis."""
    import time
    CORRECTIONS_AUDIO_DIR.mkdir(exist_ok=True)
    
    timestamp = int(time.time() * 1000)
    pred_label = predicted_class.replace(" ", "_") if predicted_class else "unknown"
    actual_label = actual_class.replace(" ", "_")
    filename = f"{timestamp}_{pred_label}_actual_{actual_label}.wav"
    filepath = CORRECTIONS_AUDIO_DIR / filename
    
    sf.write(filepath, audio_bytes, 16000)
    return str(filepath)

@app.post("/feedback/correction")
async def report_correction(
    file: UploadFile,
    predicted_class: Optional[str] = Query(None, description="What the model predicted"),
    actual_class: str = Query(..., description="What it actually is"),
    confidence: Optional[float] = Query(None, description="Model's confidence score")
):
    """
    Report a misclassification to help improve the models.
    
    **Parameters:**
    - `file`: Audio file that was misclassified
    - `predicted_class`: What the model predicted (optional)
    - `actual_class`: What it actually is (required) - must be one of: alert_sounds, car_crash, emergency_sirens, environmental_sounds, glass_breaking, human_scream, road_traffic, other
    - `confidence`: Model's confidence (optional)
    
    **Example:**
    ```
    POST /feedback/correction
    ?predicted_class=Traffic&actual_class=human_scream&confidence=0.85
    ```
    """
    try:
        # Validate inputs
        if actual_class not in SOUND_CLASSES:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Invalid actual_class. Must be one of: {', '.join(SOUND_CLASSES.keys())}",
                    "valid_classes": list(SOUND_CLASSES.keys())
                }
            )
        
        audio_bytes = await file.read()
        if not audio_bytes:
            return JSONResponse(status_code=400, content={"error": "Empty audio file"})
        
        # Save correction metadata
        import time
        from datetime import datetime
        
        correction_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "predicted": predicted_class,
            "actual": actual_class,
            "actual_label": SOUND_CLASSES[actual_class],
            "predicted_label": SOUND_CLASSES.get(predicted_class, predicted_class) if predicted_class else None,
            "confidence": confidence,
            "audio_filename": file.filename,
            "audio_size_bytes": len(audio_bytes),
            "audio_path": None
        }
        
        # Save audio file
        try:
            audio_path = save_correction_audio(audio_bytes, actual_class, predicted_class)
            correction_entry["audio_path"] = audio_path
        except Exception as e:
            print(f"[WARNING] Failed to save correction audio: {e}")
        
        # Append to corrections log
        corrections = load_corrections()
        corrections.append(correction_entry)
        save_corrections(corrections)
        
        return {
            "status": "success",
            "message": f"Correction recorded! Thank you for helping improve the model.",
            "correction": correction_entry,
            "total_corrections": len(corrections)
        }
    
    except Exception as e:
        print(f"[ERROR] Feedback correction failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/feedback/stats")
async def get_correction_stats():
    """
    Get statistics on misclassification corrections.
    Shows patterns in what the model gets wrong.
    """
    try:
        corrections = load_corrections()
        
        if not corrections:
            return {
                "total_corrections": 0,
                "message": "No corrections recorded yet"
            }
        
        from collections import Counter
        
        # Calculate statistics
        actual_counts = Counter(c["actual"] for c in corrections)
        predicted_counts = Counter(c["predicted"] for c in corrections if c["predicted"])
        
        # Find most common confusion (wrong predictions)
        confusion_pairs = [
            (c["predicted"], c["actual"]) 
            for c in corrections 
            if c["predicted"] and c["predicted"] != c["actual"]
        ]
        most_confused = Counter(confusion_pairs).most_common(5)
        
        # Average confidence when wrong
        wrong_predictions = [c["confidence"] for c in corrections if c["confidence"] and c["predicted"] != c["actual"]]
        avg_confidence_when_wrong = sum(wrong_predictions) / len(wrong_predictions) if wrong_predictions else 0
        
        # Breakdown by actual sound type
        by_actual_sound = {}
        for actual_sound in SOUND_CLASSES:
            count = sum(1 for c in corrections if c["actual"] == actual_sound)
            wrong_count = sum(1 for c in corrections if c["actual"] == actual_sound and c["predicted"] != c["actual"])
            if count > 0:
                by_actual_sound[actual_sound] = {
                    "total": count,
                    "misclassified": wrong_count,
                    "accuracy": (count - wrong_count) / count * 100
                }
        
        return {
            "total_corrections": len(corrections),
            "actual_sound_distribution": dict(actual_counts),
            "predicted_sound_distribution": dict(predicted_counts),
            "most_confused_pairs": [
                {
                    "predicted_as": pair[0],
                    "actually_was": pair[1],
                    "occurrences": count
                }
                for pair, count in most_confused
            ],
            "average_confidence_when_wrong": round(avg_confidence_when_wrong, 3),
            "accuracy_by_sound_type": by_actual_sound,
            "recommendation": analyze_correction_patterns(corrections)
        }
    
    except Exception as e:
        print(f"[ERROR] Failed to get correction stats: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

def analyze_correction_patterns(corrections: list) -> str:
    """Analyze patterns in corrections and provide recommendations."""
    if len(corrections) < 5:
        return "Collect more corrections (at least 5) to identify patterns"
    
    # Find the most misclassified sound type
    from collections import Counter
    actual_sounds = Counter(c["actual"] for c in corrections)
    most_misclassified = actual_sounds.most_common(1)[0][0]
    
    # Find common false predictions for most misclassified
    wrong_for_sound = [
        c["predicted"] 
        for c in corrections 
        if c["actual"] == most_misclassified and c["predicted"] != c["actual"] and c["predicted"]
    ]
    
    if wrong_for_sound:
        most_wrong = Counter(wrong_for_sound).most_common(1)[0][0]
        return f"PATTERN: '{SOUND_CLASSES.get(most_misclassified, most_misclassified)}' is frequently confused with '{SOUND_CLASSES.get(most_wrong, most_wrong)}'. Consider retraining with this distinction highlighted."
    
    return "Monitor more corrections to identify patterns"

@app.get("/feedback/get-all")
async def get_all_corrections():
    """Retrieve all recorded corrections."""
    try:
        corrections = load_corrections()
        return {
            "total": len(corrections),
            "corrections": corrections
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/feedback/export")
async def export_corrections(format: str = Query("json", description="Export format: json or csv")):
    """Export corrections for analysis or retraining."""
    try:
        corrections = load_corrections()
        
        if not corrections:
            return JSONResponse(status_code=400, content={"error": "No corrections to export"})
        
        if format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=["timestamp", "predicted", "actual", "confidence", "audio_filename"]
            )
            writer.writeheader()
            for c in corrections:
                writer.writerow({
                    "timestamp": c.get("timestamp", ""),
                    "predicted": c.get("predicted", ""),
                    "actual": c.get("actual", ""),
                    "confidence": c.get("confidence", ""),
                    "audio_filename": c.get("audio_filename", "")
                })
            
            return {
                "format": "csv",
                "data": output.getvalue(),
                "rows": len(corrections)
            }
        else:  # json
            return {
                "format": "json",
                "data": corrections,
                "total": len(corrections)
            }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
