# New Installation Guide

This guide will help you set up and run the Smart Car Alert audio classification system from scratch.

## Prerequisites

- **Python 3.8 or higher**
- **Git** (to clone the repository)
- **4GB+ RAM** (for running the ML model)
- **Internet connection** (for downloading dependencies)

## Step 1: Clone the Repository

```bash
git clone https://github.com/saimsheikh123/Smart-Car-Alert-Model.git
cd Smart-Car-Alert-Model/cmpe-281-models
```

## Step 2: Create a Virtual Environment

### Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 3: Install Python Dependencies

```bash
pip install fastapi uvicorn transformers torch torchaudio soundfile numpy boto3
```

**Required packages:**
- `fastapi` - Web framework for the API
- `uvicorn` - ASGI server to run the application
- `transformers` - Hugging Face library for the ML model
- `torch` - PyTorch deep learning framework
- `torchaudio` - Audio processing for PyTorch
- `soundfile` - Audio file reading/writing
- `numpy` - Numerical computing
- `boto3` - AWS SDK (only needed for SageMaker cloud mode)

## Step 4: Verify Model Files

Ensure the `ast_6class_model/` directory exists with these files:
- `class_names.json`
- `config.json`
- `model.safetensors`
- `preprocessor_config.json`

These files should already be in the repository. If missing, contact the repository maintainer.

## Step 5: Run the Application

```bash
python app.py
```

You should see output like:
```
Starting Smart Car Alert API...
Using Local Model: /path/to/ast_6class_model
Classes: alert_sounds, collision_sounds, emergency_sirens, environmental_sounds, human_scream, road_traffic
Open your browser: http://127.0.0.1:8000
```

## Step 6: Access the Application

Open your web browser and navigate to:

- **Local Mode**: http://127.0.0.1:8000
  - Uses the local ML model for classification
  - No AWS credentials needed
  
- **Cloud Mode**: http://127.0.0.1:8000/static/sagemaker.html
  - Uses AWS SageMaker for classification
  - Requires AWS credentials (see below)

- **API Documentation**: http://127.0.0.1:8000/docs
  - Interactive API documentation

## Step 7: Using the Application

### Local Mode (Recommended for new users)

1. Go to http://127.0.0.1:8000
2. Drag and drop an audio file (WAV, MP3, FLAC) or click "Choose File"
3. Click the "Classify" button
4. View the classification results showing predicted sound class and confidence scores

### Cloud Mode (Optional - Requires AWS Setup)

Only use this if you have an AWS SageMaker endpoint deployed.

## Optional: AWS SageMaker Setup (Cloud Mode Only)

If you want to use the cloud-based classification via AWS SageMaker:

### 1. Create AWS Credentials File

**Windows**: Create file at `C:\Users\YourUsername\.aws\credentials`

**macOS/Linux**: Create file at `~/.aws/credentials`

```ini
[default]
aws_access_key_id = YOUR_ACCESS_KEY_HERE
aws_secret_access_key = YOUR_SECRET_KEY_HERE
```

### 2. Create AWS Config File

**Windows**: Create file at `C:\Users\YourUsername\.aws\config`

**macOS/Linux**: Create file at `~/.aws/config`

```ini
[default]
region = us-west-1
```

### 3. Verify SageMaker Endpoint

The application expects a SageMaker endpoint named `audio-ast-v1` in the `us-west-1` region. You can modify these values in `app.py` if your endpoint has a different name or region:

```python
SAGEMAKER_ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "audio-ast-v1")
AWS_REGION = os.getenv("AWS_REGION", "us-west-1")
```

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Make sure you've activated the virtual environment and installed all dependencies.

### Issue: "Port 8000 already in use"
**Solution**: 
- Windows: `$port = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue; if ($port) { Stop-Process -Id $port.OwningProcess -Force }`
- macOS/Linux: `lsof -ti:8000 | xargs kill -9`

### Issue: Model files not found
**Solution**: Verify the `ast_6class_model/` directory exists in the same location as `app.py`.

### Issue: AWS credentials error (Cloud Mode)
**Solution**: 
- Verify your credentials file is properly formatted
- Ensure you have permissions to invoke SageMaker endpoints
- Check that the endpoint name and region match your AWS setup

## Testing the API

### Using curl (Local Mode):
```bash
curl -X POST "http://127.0.0.1:8000/classify" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_audio_file.wav"
```

### Using curl (Cloud Mode):
```bash
curl -X POST "http://127.0.0.1:8000/classify-cloud" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_audio_file.wav"
```

## Stopping the Application

Press `Ctrl+C` in the terminal where the application is running.

## Next Steps

- Try uploading different audio files to test classification
- Check the API documentation at http://127.0.0.1:8000/docs
- Review the classification history in the web interface
- Provide feedback on predictions using the "Right/Wrong" buttons

## Support

For issues or questions:
- Check the GitHub repository: https://github.com/saimsheikh123/Smart-Car-Alert-Model
- Review the API documentation at `/docs`
- Check the `docs/SAGEMAKER_API_TESTING.md` for advanced AWS SageMaker usage

## Audio File Classes

The model can classify audio into these 6 categories:
1. **alert_sounds** - Alert and notification sounds
2. **collision_sounds** - Vehicle collision/crash sounds
3. **emergency_sirens** - Emergency vehicle sirens
4. **environmental_sounds** - General environmental audio
5. **human_scream** - Human screaming/distress sounds
6. **road_traffic** - Normal traffic and vehicle sounds
