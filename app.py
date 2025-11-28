"""
FastAPI Application for Audio Classification using Local Model
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
from io import BytesIO
from pathlib import Path
import numpy as np
import os
import mimetypes

# Lazy import heavy ML libraries after app starts
app = FastAPI(
    title="Smart Car Alert - Audio Classification API",
    description="Audio classification using local AST model for safety-critical sound detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Load local model
MODEL_PATH = Path(__file__).parent / "ast_6class_model"
CLASS_NAMES_PATH = MODEL_PATH / "class_names.json"

# Load class names
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

# SageMaker configuration
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "audio-ast-v1")
AWS_REGION = os.getenv("AWS_REGION", "us-west-1")

# Model will be loaded on first request
model = None
feature_extractor = None

def load_model():
    """Load model on first use"""
    global model, feature_extractor
    if model is None:
        print(f"Loading model from {MODEL_PATH}...")
        from transformers import AutoFeatureExtractor, ASTForAudioClassification
        import torch
        
        feature_extractor = AutoFeatureExtractor.from_pretrained(str(MODEL_PATH))
        model = ASTForAudioClassification.from_pretrained(str(MODEL_PATH))
        model.eval()
        print(f"Model loaded successfully! Classes: {class_names}")
    return model, feature_extractor


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI"""
    index_file = Path(__file__).parent / "static" / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding='utf-8')
    return """
    <h1>Smart Car Alert API</h1>
    <p>API is running. Visit <a href="/docs">/docs</a> for API documentation.</p>
    <p>Status: Online | Model: AST-6class | Endpoint: audio-ast-v1</p>
    """

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "status": "online",
        "model": "Audio Spectrogram Transformer (AST)",
        "deployment": "Local",
        "model_path": str(MODEL_PATH),
        "classes": class_names
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    cloud_status = "unknown"
    try:
        import boto3
        sm = boto3.client('sagemaker', region_name=AWS_REGION)
        endpoint = sm.describe_endpoint(EndpointName=SAGEMAKER_ENDPOINT)
        cloud_status = endpoint['EndpointStatus']
    except:
        pass
    
    return {
        "api": "healthy",
        "model": "loaded",
        "model_path": str(MODEL_PATH),
        "classes": class_names,
        "deployment": "local",
        "cloud_endpoint": SAGEMAKER_ENDPOINT,
        "cloud_status": cloud_status,
        "ready": True
    }


@app.post("/classify")
async def classify_audio(file: UploadFile = File(...)):
    """
    Classify an audio file using local AST model
    
    Args:
        file: Audio file (.wav, .mp3, .flac)
    
    Returns:
        Classification result with predicted class and confidence
    """
    # Validate file type
    allowed_types = [
        'audio/wav', 'audio/x-wav', 'audio/wave', 'audio/x-wave', 'audio/vnd.wave',
        'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/ogg', 'audio/x-flac',
        'audio/aac', 'audio/mp4', 'audio/webm', 'application/octet-stream'
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        import torch
        import torchaudio
        import soundfile as sf
        
        # Load model if not already loaded
        mdl, feat_ext = load_model()
        
        # Read audio file
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Load audio with soundfile (more reliable than torchaudio.load)
        waveform_np, sample_rate = sf.read(BytesIO(audio_bytes), dtype='float32')
        
        # Convert to torch tensor and add channel dimension if mono
        if len(waveform_np.shape) == 1:
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        else:
            waveform = torch.from_numpy(waveform_np.T)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Prepare inputs
        inputs = feat_ext(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # Run inference
        with torch.no_grad():
            outputs = mdl(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get predictions
        predicted_class_idx = probabilities.argmax().item()
        predicted_class = class_names[predicted_class_idx]
        confidence = probabilities[0][predicted_class_idx].item()
        
        # Get all class probabilities
        all_probabilities = {}
        for idx, class_name in enumerate(class_names):
            all_probabilities[class_name] = float(probabilities[0][idx].item())
        
        result = {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "probabilities": all_probabilities,
            "model": "AST-6class-local",
            "file_size_bytes": len(audio_bytes),
            "filename": file.filename
        }
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in classify: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@app.post("/batch-classify")
async def batch_classify(files: list[UploadFile] = File(...)):
    """
    Classify multiple audio files using local model
    
    Args:
        files: List of audio files
    
    Returns:
        List of classification results
    """
    import torch
    import torchaudio
    import soundfile as sf
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    # Load model if not already loaded
    mdl, feat_ext = load_model()
    
    results = []
    for file in files:
        try:
            audio_bytes = await file.read()
            waveform_np, sample_rate = sf.read(BytesIO(audio_bytes), dtype='float32')
            
            if len(waveform_np.shape) == 1:
                waveform = torch.from_numpy(waveform_np).unsqueeze(0)
            else:
                waveform = torch.from_numpy(waveform_np.T)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            inputs = feat_ext(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = mdl(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            predicted_class_idx = probabilities.argmax().item()
            predicted_class = class_names[predicted_class_idx]
            confidence = probabilities[0][predicted_class_idx].item()
            
            results.append({
                "filename": file.filename,
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "success": True
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results, "total": len(files)})


@app.get("/test-files")
async def list_test_files():
    """List available test audio files from dataset (recursive)."""
    # Allow override via env var
    env_dir = os.getenv("TEST_DATA_DIR")
    base_dir = Path(env_dir) if env_dir else (Path(__file__).parent / "train" / "dataset" / "test")

    exists = base_dir.exists()
    if not exists:
        return {
            "files": [],
            "total": 0,
            "exists": False,
            "root": str(base_dir),
            "message": "Test directory not found"
        }

    suffixes = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
    files = []
    classes = []
    try:
        for class_dir in base_dir.iterdir():
            if class_dir.is_dir():
                classes.append(class_dir.name)
                for audio_file in class_dir.rglob("*"):
                    if audio_file.is_file() and audio_file.suffix.lower() in suffixes:
                        files.append({
                            "path": str(audio_file.relative_to(base_dir.parent.parent)),
                            "name": audio_file.name,
                            "class": class_dir.name,
                            "size": audio_file.stat().st_size
                        })
    except Exception as e:
        print(f"/test-files scan error: {e}")

    files_sorted = sorted(files, key=lambda x: (x["class"], x["name"]))
    return {
        "files": files_sorted[:100],  # limit for performance
        "total": len(files_sorted),
        "exists": True,
        "root": str(base_dir),
        "classes": classes
    }


@app.get("/test-file/{class_name}/{filename}")
async def get_test_file(class_name: str, filename: str):
    """Serve a test audio file"""
    from fastapi.responses import FileResponse
    test_file = Path(__file__).parent / "train" / "dataset" / "test" / class_name / filename
    if not test_file.exists():
        raise HTTPException(status_code=404, detail="File not found")
    content_type, _ = mimetypes.guess_type(str(test_file))
    return FileResponse(test_file, media_type=content_type or "application/octet-stream")


@app.post("/classify-cloud")
async def classify_cloud(file: UploadFile = File(...)):
    """
    Classify an audio file using SageMaker endpoint
    
    Args:
        file: Audio file (.wav, .mp3, .flac)
    
    Returns:
        Classification result from SageMaker
    """
    # Validate file type
    allowed_types = [
        'audio/wav', 'audio/x-wav', 'audio/wave', 'audio/x-wave', 'audio/vnd.wave',
        'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/ogg', 'audio/x-flac',
        'audio/aac', 'audio/mp4', 'audio/webm', 'application/octet-stream'
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        import boto3
        import traceback
        import soundfile as sf
        from io import BytesIO
        
        # Read audio file
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Calculate duration
        try:
            info = sf.info(BytesIO(audio_bytes))
            duration_seconds = info.duration
        except Exception:
            duration_seconds = 0.0
        
        # Initialize SageMaker runtime client
        runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
        
        # Invoke endpoint
        response = runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType=file.content_type or 'audio/wav',
            Body=audio_bytes
        )
        
        # Parse response
        body_bytes = response['Body'].read()
        text = body_bytes.decode('utf-8', errors='ignore') if isinstance(body_bytes, (bytes, bytearray)) else str(body_bytes)
        try:
            raw = json.loads(text)
        except Exception:
            raw = text

        # Normalize to expected schema
        def softmax(x):
            import math
            m = max(x)
            exps = [math.exp(v - m) for v in x]
            s = sum(exps)
            return [v / s for v in exps]

        normalized = None
        if isinstance(raw, dict):
            # Common variants
            if 'all_probs' in raw and isinstance(raw['all_probs'], dict):
                prob_map = {k: float(v) for k, v in raw['all_probs'].items()}
                best = max(prob_map.items(), key=lambda kv: kv[1])
                normalized = {
                    "predicted_class": raw.get('class', raw.get('predicted_class', best[0])),
                    "confidence": float(raw.get('confidence', best[1])),
                    "probabilities": prob_map
                }
            elif 'probabilities' in raw and isinstance(raw['probabilities'], dict):
                prob_map = {k: float(v) for k, v in raw['probabilities'].items()}
                best = max(prob_map.items(), key=lambda kv: kv[1])
                normalized = {
                    "predicted_class": raw.get('predicted_class', best[0]),
                    "confidence": float(raw.get('confidence', best[1])),
                    "probabilities": prob_map
                }
            elif 'probabilities' in raw and isinstance(raw['probabilities'], list):
                probs = [float(v) for v in raw['probabilities']]
                if len(probs) == len(class_names):
                    prob_map = {class_names[i]: probs[i] for i in range(len(probs))}
                else:
                    prob_map = {str(i): probs[i] for i in range(len(probs))}
                best_idx = max(range(len(probs)), key=lambda i: probs[i]) if probs else 0
                normalized = {
                    "predicted_class": class_names[best_idx] if best_idx < len(class_names) else str(best_idx),
                    "confidence": float(probs[best_idx]) if probs else 0.0,
                    "probabilities": prob_map
                }
            elif 'logits' in raw and isinstance(raw['logits'], (list, tuple)):
                probs = softmax([float(v) for v in raw['logits']])
                prob_map = {class_names[i]: probs[i] for i in range(min(len(probs), len(class_names)))}
                best_idx = max(range(len(probs)), key=lambda i: probs[i]) if probs else 0
                normalized = {
                    "predicted_class": class_names[best_idx] if best_idx < len(class_names) else str(best_idx),
                    "confidence": float(probs[best_idx]) if probs else 0.0,
                    "probabilities": prob_map
                }
            elif 'outputs' in raw and isinstance(raw['outputs'], (list, tuple)):
                arr = raw['outputs']
                arr = arr[0] if arr and isinstance(arr[0], (list, tuple)) else arr
                probs = [float(v) for v in arr]
                if sum(probs) == 0 or max(probs) > 1.0:
                    probs = softmax(probs)
                prob_map = {class_names[i]: probs[i] for i in range(min(len(probs), len(class_names)))}
                best_idx = max(range(len(probs)), key=lambda i: probs[i]) if probs else 0
                normalized = {
                    "predicted_class": class_names[best_idx] if best_idx < len(class_names) else str(best_idx),
                    "confidence": float(probs[best_idx]) if probs else 0.0,
                    "probabilities": prob_map
                }
        elif isinstance(raw, list):
            # Check if it's a list containing a JSON string (common in some SageMaker configs)
            # It might be [json_string] or [json_string, content_type]
            if len(raw) >= 1 and isinstance(raw[0], str):
                try:
                    inner_raw = json.loads(raw[0])
                    if isinstance(inner_raw, dict) and 'all_probs' in inner_raw:
                        prob_map = {k: float(v) for k, v in inner_raw['all_probs'].items()}
                        best = max(prob_map.items(), key=lambda kv: kv[1])
                        normalized = {
                            "predicted_class": inner_raw.get('class', inner_raw.get('predicted_class', best[0])),
                            "confidence": float(inner_raw.get('confidence', best[1])),
                            "probabilities": prob_map
                        }
                except:
                    pass

            if normalized is None:
                # Assume list of probabilities or logits
                try:
                    probs = [float(v) for v in raw]
                    if sum(probs) == 0 or max(probs) > 1.0:
                        probs = softmax(probs)
                    prob_map = {class_names[i]: probs[i] for i in range(min(len(probs), len(class_names)))}
                    best_idx = max(range(len(probs)), key=lambda i: probs[i]) if probs else 0
                    normalized = {
                        "predicted_class": class_names[best_idx] if best_idx < len(class_names) else str(best_idx),
                        "confidence": float(probs[best_idx]) if probs else 0.0,
                        "probabilities": prob_map
                    }
                except ValueError:
                    # Failed to convert to float, likely not a simple list of numbers
                    pass

        if normalized is None:
            normalized = {
                "predicted_class": "unknown",
                "confidence": 0.0,
                "probabilities": {},
                "raw": raw
            }

        # Add metadata
        normalized['endpoint'] = SAGEMAKER_ENDPOINT
        normalized['region'] = AWS_REGION
        normalized['filename'] = file.filename
        normalized['file_size_bytes'] = len(audio_bytes)
        normalized['duration_seconds'] = duration_seconds
        
        return JSONResponse(content=normalized)
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in classify-cloud: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Cloud classification failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("Starting Smart Car Alert API...")
    print(f"Using Local Model: {MODEL_PATH}")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Open your browser: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
