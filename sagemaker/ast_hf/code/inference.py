import io
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa
from transformers import ASTForAudioClassification, ASTFeatureExtractor


def _load_classes(model_dir: str):
    p = Path(model_dir) / "class_names.json"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Fallback classes (7-class ordering used in training)
    return [
        "alert_sounds",
        "car_crash",
        "emergency_sirens",
        "environmental_sounds",
        "glass_breaking",
        "human_scream",
        "road_traffic",
    ]


def _prepare_audio(audio_bytes: bytes, target_sr: int = 16000, max_seconds: int = 10):
    bio = io.BytesIO(audio_bytes)
    data, sr = sf.read(bio)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = librosa.resample(data.astype("float32"), orig_sr=sr, target_sr=target_sr)
    max_len = target_sr * max_seconds
    if len(data) > max_len:
        data = data[:max_len]
    else:
        pad = max_len - len(data)
        if pad > 0:
            data = np.pad(data, (0, pad), mode="constant")
    return data.astype("float32")


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASTForAudioClassification.from_pretrained(model_dir)
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    classes = _load_classes(model_dir)
    return {
        "model": model,
        "fx": feature_extractor,
        "device": device,
        "classes": classes,
    }


def input_fn(request_body, content_type=None):
    if content_type is None:
        content_type = "application/octet-stream"
    if content_type.startswith("audio/") or content_type == "application/octet-stream":
        if isinstance(request_body, (bytes, bytearray)):
            return bytes(request_body)
        return request_body.encode("latin1") if isinstance(request_body, str) else request_body
    if content_type == "application/json":
        obj = json.loads(request_body)
        # Expect base64 under key "audio"
        import base64
        return base64.b64decode(obj["audio"]) if "audio" in obj else b""
    raise ValueError(f"Unsupported content_type: {content_type}")


def predict_fn(input_data, model_artifacts):
    audio = _prepare_audio(input_data, target_sr=16000, max_seconds=10)
    fx = model_artifacts["fx"]
    model = model_artifacts["model"]
    device = model_artifacts["device"]
    classes = model_artifacts["classes"]

    inputs = fx(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # If model label count matches our classes, map directly; otherwise choose top index only
    if len(probs) == len(classes):
        order = np.argsort(probs)[::-1]
        idx = int(order[0])
        pred_class = classes[idx]
        top2 = [
            {"class": classes[int(order[0])], "confidence": float(probs[int(order[0])])},
            {"class": classes[int(order[1])], "confidence": float(probs[int(order[1])])},
        ]
        all_probs = {c: float(p) for c, p in zip(classes, probs)}
        return {
            "model": "AST",
            "class": pred_class,
            "confidence": float(probs[idx]),
            "top2": top2,
            "all_probs": all_probs,
            "success": True,
        }
    idx = int(np.argmax(probs))
    return {
        "model": "AST",
        "class": str(idx),
        "confidence": float(probs[idx]),
        "top2": [],
        "all_probs": {str(i): float(p) for i, p in enumerate(probs)},
        "success": True,
    }


def output_fn(prediction, accept=None):
    body = json.dumps(prediction)
    return body, "application/json"
