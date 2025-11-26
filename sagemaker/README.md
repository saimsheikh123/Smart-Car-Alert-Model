SageMaker Deployment Guide

## Overview
Deploy your audio classification models (AST or CRNN) to AWS SageMaker with automated setup.

Variants supported:
- **AST** (Hugging Face DLC): packages `ast_6class_model` artifacts with custom inference handler
- **CRNN** (PyTorch DLC): packages `multi_audio_crnn.pth` with custom inference handler

## Step-by-Step Setup

### 1. Install Dependencies
```powershell
pip install boto3 sagemaker
```

### 2. Configure AWS Credentials
Login to your AWS account and configure credentials:
```powershell
aws configure
```
Or set environment variables:
```powershell
$env:AWS_ACCESS_KEY_ID='your-key'
$env:AWS_SECRET_ACCESS_KEY='your-secret'
$env:AWS_DEFAULT_REGION='us-east-1'
```

### 3. Set Up AWS Resources (First Time Only)
Run the setup script to create S3 bucket and SageMaker execution role:
```powershell
python sagemaker/setup_aws_resources.py --region us-east-1
```

This will:
- ✓ Verify AWS credentials
- ✓ Create S3 bucket (auto-named or specify with `--bucket-name`)
- ✓ Create SageMaker execution role with necessary permissions
- ✓ Save configuration to `aws_config.json`

Optional arguments:
- `--bucket-name`: Custom S3 bucket name
- `--role-name`: Custom IAM role name (default: SageMakerAudioModelRole)
- `--region`: AWS region (default: us-east-1)

### 4. Deploy Your Model
After setup completes, deploy using the generated config:

**Deploy AST model:**
```powershell
python sagemaker/deploy_to_sagemaker.py --variant ast --endpoint-name audio-ast-v1
```

**Deploy CRNN model:**
```powershell
python sagemaker/deploy_to_sagemaker.py --variant crnn --endpoint-name audio-crnn-v1
```

The script will:
- Package model artifacts into `model.tar.gz`
- Upload to S3
- Create SageMaker endpoint (takes 5-10 minutes)

Optional arguments:
- `--instance-type`: Instance type (default: ml.m5.xlarge)
- `--bucket`: Override S3 bucket from config
- `--role`: Override IAM role from config
- `--region`: Override region from config

### 5. Test Your Endpoint
Once deployment completes (status: InService), test with a sample audio file:
```powershell
python sagemaker/test_endpoint.py --endpoint audio-ast-v1 --audio path/to/sample.wav
```

Check endpoint status:
```powershell
aws sagemaker describe-endpoint --endpoint-name audio-ast-v1
```

## What's Deployed

### AST variant
- Model files: `config.json`, `model.safetensors`, `preprocessor_config.json`, `class_names.json`
- Custom handler: `code/inference.py` (audio preprocessing + AST inference)
- Dependencies: `librosa`, `soundfile`, `transformers`

### CRNN variant
- Model checkpoint: `multi_audio_crnn.pth`
- Model architecture: `AudioCRNN` class in `code/inference.py`
- Custom handler: mel-spectrogram preprocessing + PyTorch inference
- Dependencies: `librosa`, `soundfile`, `torch`

## Cost Estimation
- **ml.m5.xlarge**: ~$0.23/hour (~$166/month if running 24/7)
- **S3 storage**: Minimal (<$0.10/month for model artifacts)

Remember to delete endpoints when not in use:
```powershell
aws sagemaker delete-endpoint --endpoint-name audio-ast-v1
```

## Troubleshooting
- **Credentials error**: Run `aws configure` or check environment variables
- **Role permissions**: Ensure SageMaker role has S3 and SageMaker permissions
- **Deployment fails**: Check CloudWatch logs in AWS Console
- **Endpoint not ready**: Wait 5-10 minutes, check status with `describe-endpoint`
