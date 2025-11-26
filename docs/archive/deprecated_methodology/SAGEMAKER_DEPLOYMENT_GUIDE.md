# Deploying Audio Models to Amazon SageMaker

This guide walks you through deploying your multi-model audio classification system (AudioCRNN + SirenClassifier + DQN agents) to AWS SageMaker.

## Overview

Your system includes:
- **AudioCRNN**: 7-class audio classifier
- **SirenClassifier**: 2-class alert/emergency detector
- **3 DQN Agents**: TorchScript models for reinforcement learning (alert, emergency_siren, environmental)
- **FastAPI**: REST API with ensemble logic

## Deployment Options

### Option 1: SageMaker Real-Time Endpoint (Recommended for Production)
Best for low-latency, synchronous predictions with consistent traffic.

### Option 2: SageMaker Batch Transform
Best for processing large audio files in batches asynchronously.

### Option 3: SageMaker Serverless Inference
Best for intermittent traffic with automatic scaling to zero.

---

## Option 1: Real-Time Endpoint Deployment

### Step 1: Prepare Your Model Package

Create a SageMaker-compatible model structure:

```powershell
# Create deployment structure
New-Item -ItemType Directory -Force -Path "sagemaker_deployment/code"
New-Item -ItemType Directory -Force -Path "sagemaker_deployment/model"
```

**File: `sagemaker_deployment/code/inference.py`**

```python
import os
import json
import torch
import librosa
import soundfile as sf
import numpy as np
from io import BytesIO
import base64

# Import your model classes
import sys
sys.path.append('/opt/ml/code')

# Model architecture definitions (same as your local models)
class AudioCRNN(torch.nn.Module):
    # ... copy from your local code ...
    pass

class SirenClassifier(torch.nn.Module):
    # ... copy from your local code ...
    pass

# Global variables for models
audiocrnn_model = None
siren_model = None
dqn_models = {}
device = None

def model_fn(model_dir):
    """
    Load all models. Called once when endpoint starts.
    SageMaker expects this function to load and return the model(s).
    """
    global audiocrnn_model, siren_model, dqn_models, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[SAGEMAKER] Using device: {device}")
    
    # Load AudioCRNN
    audiocrnn_path = os.path.join(model_dir, 'multi_audio_crnn.pth')
    state_dict = torch.load(audiocrnn_path, map_location=device)
    num_classes = state_dict['fc2.weight'].shape[0]
    audiocrnn_model = AudioCRNN(num_classes=num_classes)
    audiocrnn_model.load_state_dict(state_dict)
    audiocrnn_model.to(device).eval()
    print(f"[OK] AudioCRNN loaded ({num_classes} classes)")
    
    # Load SirenClassifier
    siren_path = os.path.join(model_dir, 'alert-reinforcement_model.pth')
    siren_state = torch.load(siren_path, map_location=device)
    siren_model = SirenClassifier()
    siren_model.load_state_dict(siren_state)
    siren_model.to(device).eval()
    print("[OK] SirenClassifier loaded")
    
    # Load DQN agents (TorchScript)
    dqn_agents = ['alert_dqn_agent', 'emergency_siren_dqn_agent', 'environmental_dqn_agent']
    for agent_name in dqn_agents:
        traced_path = os.path.join(model_dir, agent_name, f"{agent_name}_traced.pt")
        if os.path.exists(traced_path):
            dqn_models[agent_name] = torch.jit.load(traced_path, map_location=device)
            dqn_models[agent_name].eval()
            print(f"[OK] {agent_name} loaded")
    
    return {
        'audiocrnn': audiocrnn_model,
        'siren': siren_model,
        'dqns': dqn_models,
        'device': device
    }

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare input for prediction.
    Handles both JSON (base64 audio) and binary audio data.
    """
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        # Expect base64-encoded audio
        audio_bytes = base64.b64decode(data['audio'])
        return BytesIO(audio_bytes)
    elif request_content_type in ['audio/wav', 'audio/mpeg', 'audio/ogg']:
        return BytesIO(request_body)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(audio_stream, models):
    """
    Make prediction on the deserialized data.
    """
    device = models['device']
    
    # Load and preprocess audio
    audio, sr = sf.read(audio_stream)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Resample to 16kHz
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Extract mel spectrogram (128 mel bins, matching training)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_norm = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    # Prepare tensor
    mel_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0).unsqueeze(0).to(device)
    
    # AudioCRNN prediction
    with torch.no_grad():
        crnn_output = models['audiocrnn'](mel_tensor)
        crnn_probs = torch.softmax(crnn_output, dim=1).cpu().numpy()[0]
        crnn_class = int(crnn_probs.argmax())
        top2_idx = np.argsort(crnn_probs)[-2:][::-1]
        
    crnn_labels = ['car_crash', 'glass_breaking', 'human_scream', 'emergency_siren', 
                   'alert_sounds', 'road_traffic', 'environmental_sounds']
    
    # SirenClassifier prediction
    with torch.no_grad():
        siren_output = models['siren'](mel_tensor)
        siren_probs = torch.softmax(siren_output, dim=1).cpu().numpy()[0]
        siren_class = int(siren_probs.argmax())
    
    siren_labels = ['normal_environment', 'alert_emergency']
    
    # DQN predictions
    dqn_results = {}
    mel_flat = mel_spec_norm.flatten()
    
    for agent_name, dqn_model in models['dqns'].items():
        try:
            # Get expected input size from first layer
            first_param = next(dqn_model.parameters())
            if hasattr(first_param, 'shape') and len(first_param.shape) >= 2:
                target_dim = first_param.shape[1] if len(first_param.shape) == 2 else first_param.shape[0]
            else:
                target_dim = 128  # default
            
            # Build observation
            if len(mel_flat) > target_dim:
                obs = np.mean(mel_flat[:len(mel_flat)//target_dim*target_dim].reshape(-1, target_dim), axis=0)
            else:
                obs = np.pad(mel_flat, (0, target_dim - len(mel_flat)), mode='constant')
            
            obs = (obs - obs.mean()) / (obs.std() + 1e-8)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values = dqn_model(obs_tensor)
                action = int(q_values.argmax(dim=1).item())
            
            dqn_results[agent_name] = {
                'action': action,
                'label': 'ALERT' if action == 1 else 'WAIT',
                'success': True
            }
        except Exception as e:
            dqn_results[agent_name] = {
                'success': False,
                'error': str(e)
            }
    
    # Ensemble logic
    dqn_alert_count = sum(1 for r in dqn_results.values() if r.get('action') == 1)
    
    return {
        'audiocrnn': {
            'predicted_class': crnn_labels[crnn_class],
            'confidence': float(crnn_probs[crnn_class]),
            'all_probabilities': {crnn_labels[i]: float(crnn_probs[i]) for i in range(len(crnn_labels))},
            'top2': [
                {'class': crnn_labels[top2_idx[0]], 'confidence': float(crnn_probs[top2_idx[0]])},
                {'class': crnn_labels[top2_idx[1]], 'confidence': float(crnn_probs[top2_idx[1]])}
            ]
        },
        'siren_classifier': {
            'predicted_class': siren_labels[siren_class],
            'confidence': float(siren_probs[siren_class]),
            'probabilities': {siren_labels[i]: float(siren_probs[i]) for i in range(len(siren_labels))}
        },
        'dqn_agents': dqn_results,
        'ensemble': {
            'final_decision': 'ALERT' if dqn_alert_count >= 2 or siren_class == 1 else 'WAIT',
            'dqn_vote': f"{dqn_alert_count}/{len(dqn_results)} agents recommend ALERT",
            'top2': {
                'most_likely': crnn_labels[top2_idx[0]],
                'second_most_likely': crnn_labels[top2_idx[1]]
            }
        }
    }

def output_fn(prediction, response_content_type):
    """
    Serialize the prediction result.
    """
    return json.dumps(prediction)
```

**File: `sagemaker_deployment/code/requirements.txt`**

```
torch>=2.0.0
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.24.0
```

### Step 2: Package Your Models

```powershell
# Copy models to deployment folder
Copy-Item "Audio_Models\multi_audio_crnn.pth" "sagemaker_deployment\model\"
Copy-Item "Audio_Models\alert-reinforcement_model.pth" "sagemaker_deployment\model\"

# Copy DQN agents
Copy-Item -Recurse "Reinforcement_learning_agents\alert_dqn_agent" "sagemaker_deployment\model\"
Copy-Item -Recurse "Reinforcement_learning_agents\emergency_siren_dqn_agent" "sagemaker_deployment\model\"
Copy-Item -Recurse "Reinforcement_learning_agents\environmental_dqn_agent" "sagemaker_deployment\model\"

# Create model tarball
cd sagemaker_deployment
tar -czf model.tar.gz model/ code/

# Upload to S3 (replace with your bucket)
aws s3 cp model.tar.gz s3://your-bucket-name/audio-models/model.tar.gz
```

### Step 3: Create SageMaker Model and Endpoint

**File: `sagemaker_deployment/deploy_sagemaker.py`**

```python
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from time import gmtime, strftime

# Configuration
role = 'arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole'  # Replace with your role
model_data = 's3://your-bucket-name/audio-models/model.tar.gz'
endpoint_name = f'audio-classifier-{strftime("%Y-%m-%d-%H-%M-%S", gmtime())}'

# Create PyTorch Model
pytorch_model = PyTorchModel(
    model_data=model_data,
    role=role,
    framework_version='2.0.0',
    py_version='py310',
    entry_point='inference.py',
    source_dir='code',
    name=f'audio-model-{strftime("%Y-%m-%d-%H-%M-%S", gmtime())}'
)

# Deploy to endpoint
predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',  # or ml.g4dn.xlarge for GPU
    endpoint_name=endpoint_name
)

print(f"Endpoint deployed: {endpoint_name}")
```

### Step 4: Test Your Endpoint

```python
import boto3
import json
import base64

runtime = boto3.client('sagemaker-runtime')
endpoint_name = 'your-endpoint-name'  # From Step 3

# Read audio file
with open('test_audio.wav', 'rb') as f:
    audio_bytes = f.read()

# Option 1: Send as JSON with base64
payload = json.dumps({
    'audio': base64.b64encode(audio_bytes).decode('utf-8')
})

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=payload
)

result = json.loads(response['Body'].read())
print(json.dumps(result, indent=2))

# Option 2: Send raw audio
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='audio/wav',
    Body=audio_bytes
)

result = json.loads(response['Body'].read())
print(json.dumps(result, indent=2))
```

---

## Option 2: SageMaker Batch Transform

For processing many audio files asynchronously:

```python
from sagemaker.pytorch import PyTorchModel

# Use same model from Option 1
pytorch_model = PyTorchModel(
    model_data=model_data,
    role=role,
    framework_version='2.0.0',
    py_version='py310',
    entry_point='inference.py',
    source_dir='code'
)

# Create transformer
transformer = pytorch_model.transformer(
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path='s3://your-bucket/batch-output/',
    accept='application/json'
)

# Start batch job
transformer.transform(
    data='s3://your-bucket/audio-files/',  # Folder with audio files
    content_type='audio/wav',
    split_type='None',
    wait=True
)
```

---

## Option 3: SageMaker Serverless Inference

For cost-effective deployment with auto-scaling:

```python
from sagemaker.serverless import ServerlessInferenceConfig

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=4096,  # 2GB-6GB
    max_concurrency=5
)

predictor = pytorch_model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name=endpoint_name
)
```

---

## Cost Optimization Tips

1. **Instance Selection**:
   - CPU: `ml.m5.xlarge` (~$0.23/hr) for most workloads
   - GPU: `ml.g4dn.xlarge` (~$0.74/hr) if you need faster inference
   - Serverless: Pay per inference (starts at $0.20 per 1M requests)

2. **Auto-Scaling** (Real-Time Endpoints):
```python
client = boto3.client('application-autoscaling')

# Register scalable target
client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=3
)

# Target tracking policy
client.put_scaling_policy(
    PolicyName='audio-classifier-scaling',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,  # 70% invocation rate
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        },
        'ScaleInCooldown': 300,
        'ScaleOutCooldown': 60
    }
)
```

3. **Use SageMaker Inference Recommender** to find optimal instance type:
```python
from sagemaker.inference_recommender import InferenceRecommender

recommender = InferenceRecommender(
    model_package_arn='arn:aws:sagemaker:...',
    role=role
)

job = recommender.run(
    instance_types=['ml.m5.xlarge', 'ml.m5.2xlarge', 'ml.g4dn.xlarge'],
    max_invocations=1000
)
```

---

## Monitoring & Logging

### CloudWatch Metrics
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Get endpoint metrics
metrics = cloudwatch.get_metric_statistics(
    Namespace='AWS/SageMaker',
    MetricName='ModelLatency',
    Dimensions=[
        {'Name': 'EndpointName', 'Value': endpoint_name},
        {'Name': 'VariantName', 'Value': 'AllTraffic'}
    ],
    StartTime=datetime.now() - timedelta(hours=1),
    EndTime=datetime.now(),
    Period=300,
    Statistics=['Average', 'Maximum']
)
```

### Enable Data Capture for Model Monitoring
```python
from sagemaker.model_monitor import DataCaptureConfig

data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri='s3://your-bucket/data-capture'
)

predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    data_capture_config=data_capture_config
)
```

---

## Security Best Practices

1. **VPC Configuration** (Keep models private):
```python
from sagemaker.network import NetworkConfig

network_config = NetworkConfig(
    subnets=['subnet-xxx', 'subnet-yyy'],
    security_group_ids=['sg-xxx']
)

predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    vpc_config={
        'Subnets': network_config.subnets,
        'SecurityGroupIds': network_config.security_group_ids
    }
)
```

2. **Encryption**:
   - At rest: Enable KMS encryption for S3 and EBS
   - In transit: HTTPS enforced by default

3. **IAM Role** (minimal permissions):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::your-bucket/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

---

## Troubleshooting

### Model Loading Issues
Check CloudWatch logs:
```bash
aws logs tail /aws/sagemaker/Endpoints/your-endpoint-name --follow
```

### Common Errors:

**"ModelError: Model not loaded"**
- Check model.tar.gz contains `model/` and `code/` directories
- Verify all model files are present

**"ResourceNotFound: Could not find model"**
- Ensure S3 path is correct and accessible by SageMaker role

**"Out of Memory"**
- Increase instance size or switch to GPU instance
- Reduce batch size in inference code

**"Inference timeout"**
- Audio files too large? Set `model_timeout` in deploy config
- Optimize mel spectrogram computation

---

## Integration with Your Frontend

Update your frontend to call SageMaker endpoint:

```javascript
async function classifyAudio(audioFile) {
    // Convert to base64
    const reader = new FileReader();
    reader.readAsDataURL(audioFile);
    
    reader.onload = async () => {
        const base64Audio = reader.result.split(',')[1];
        
        // Call Lambda function (see next section)
        const response = await fetch('https://your-api-gateway-url/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ audio: base64Audio })
        });
        
        const result = await response.json();
        displayResults(result);
    };
}
```

### Optional: Add API Gateway + Lambda

**Lambda function** to proxy requests:
```python
import json
import boto3
import base64

runtime = boto3.client('sagemaker-runtime')
ENDPOINT_NAME = 'your-endpoint-name'

def lambda_handler(event, context):
    try:
        # Parse request
        body = json.loads(event['body'])
        audio_b64 = body['audio']
        
        # Invoke SageMaker
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps({'audio': audio_b64})
        )
        
        result = json.loads(response['Body'].read())
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps(result)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

---

## Quick Start Checklist

- [ ] Install AWS CLI: `pip install awscli boto3 sagemaker`
- [ ] Configure credentials: `aws configure`
- [ ] Create S3 bucket: `aws s3 mb s3://your-audio-models-bucket`
- [ ] Create SageMaker execution role in IAM console
- [ ] Copy model architecture to `sagemaker_deployment/code/inference.py`
- [ ] Package models: `tar -czf model.tar.gz model/ code/`
- [ ] Upload to S3: `aws s3 cp model.tar.gz s3://your-bucket/`
- [ ] Run deployment script: `python deploy_sagemaker.py`
- [ ] Test endpoint with sample audio
- [ ] Set up auto-scaling (optional)
- [ ] Add monitoring/alarms (optional)

---

## Cost Estimate (Monthly)

**Real-Time Endpoint (24/7)**:
- ml.m5.xlarge: ~$165/month
- ml.g4dn.xlarge (GPU): ~$530/month

**Serverless** (1M requests/month, avg 2s per request):
- Compute: ~$8/month
- Requests: ~$0.20/month
- **Total: ~$8.20/month**

**Batch Transform** (100 hours/month):
- ml.m5.xlarge: ~$23/month

**Recommendation**: Start with **Serverless** for development/low traffic, migrate to **Real-Time** when you hit consistent high volume.

---

## Next Steps

1. Test locally first: Ensure `inference.py` works with your models
2. Start with Serverless for cost-effective testing
3. Monitor latency and adjust instance type as needed
4. Set up CI/CD for model updates (AWS CodePipeline)
5. Implement A/B testing with SageMaker Variants

Need help with any specific step? Let me know!
