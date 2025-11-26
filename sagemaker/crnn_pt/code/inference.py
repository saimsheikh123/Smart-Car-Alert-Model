import io
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import librosa


class AudioCRNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
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
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, w, c * h)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def _load_classes(model_dir: str):
    p = Path(model_dir) / "class_names.json"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return [
        "alert_sounds",
        "car_crash",
        "emergency_sirens",
        "environmental_sounds",
        "glass_breaking",
        "human_scream",
        "road_traffic",
    ]


def _wav_to_mel(audio_bytes: bytes, sr: int = 16000, n_mels: int = 128, n_fft: int = 2048, hop: int = 512, fixed: int = 128):
    try:
        data, file_sr = sf.read(io.BytesIO(audio_bytes))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if file_sr != sr:
            data = librosa.resample(data.astype("float32"), orig_sr=file_sr, target_sr=sr)
        y = data
    except Exception:
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
    return S.astype("float32")[None, None, ...]


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_name = os.environ.get("CRNN_CKPT", "multi_audio_crnn.pth")
    ckpt_path = Path(model_dir) / ckpt_name
    raw = torch.load(str(ckpt_path), map_location=device)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif isinstance(raw, dict):
        state = raw
    else:
        raise ValueError("Unrecognized checkpoint format for CRNN")
    # Infer class count from fc2.weight
    fc_key = next((k for k in state.keys() if k.endswith("fc2.weight")), None)
    num_classes = state[fc_key].shape[0] if fc_key else 7
    model = AudioCRNN(num_classes=num_classes).to(device)
    model.load_state_dict(state)
    model.eval()
    classes = _load_classes(model_dir)
    return {"model": model, "device": device, "classes": classes}


def input_fn(request_body, content_type=None):
    if content_type is None:
        content_type = "application/octet-stream"
    if content_type.startswith("audio/") or content_type == "application/octet-stream":
        if isinstance(request_body, (bytes, bytearray)):
            return bytes(request_body)
        return request_body.encode("latin1") if isinstance(request_body, str) else request_body
    if content_type == "application/json":
        obj = json.loads(request_body)
        import base64
        return base64.b64decode(obj["audio"]) if "audio" in obj else b""
    raise ValueError(f"Unsupported content_type: {content_type}")


def predict_fn(input_data, model_artifacts):
    mel = _wav_to_mel(input_data)
    x = torch.from_numpy(mel)
    device = model_artifacts["device"]
    model = model_artifacts["model"]
    classes = model_artifacts["classes"]
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    order = np.argsort(probs)[::-1]
    idx = int(order[0])
    top2 = [
        {"class": classes[int(order[0])], "confidence": float(probs[int(order[0])])},
        {"class": classes[int(order[1])], "confidence": float(probs[int(order[1])])},
    ]
    all_probs = {c: float(p) for c, p in zip(classes, probs)}
    return {
        "model": "AudioCRNN",
        "class": classes[idx],
        "confidence": float(probs[idx]),
        "top2": top2,
        "all_probs": all_probs,
        "success": True,
    }


def output_fn(prediction, accept=None):
    body = json.dumps(prediction)
    return body, "application/json"
