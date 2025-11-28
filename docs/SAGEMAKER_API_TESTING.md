# Testing the SageMaker API

This document outlines how to test the Smart Car Alert SageMaker endpoint, including authentication, request formats, and tools.

## 1. Overview

The application uses a **FastAPI backend** (`app.py`) as a proxy to communicate with the AWS SageMaker Runtime. This architecture simplifies client-side logic and secures AWS credentials.

- **Frontend**: `static/sagemaker.html` (User Interface)
- **Backend Proxy**: `POST /classify-cloud` (FastAPI)
- **AWS Service**: SageMaker Runtime (`invoke_endpoint`)

## 2. AWS Credentials

To invoke the SageMaker endpoint, the backend (`app.py`) requires valid AWS credentials with `sagemaker:InvokeEndpoint` permissions.

### Configuration
The application uses the standard `boto3` credential chain. You can provide credentials via:

1.  **Environment Variables** (Recommended for containers/CI):
    ```bash
    export AWS_ACCESS_KEY_ID=AKIA...
    export AWS_SECRET_ACCESS_KEY=wJalr...
    export AWS_DEFAULT_REGION=us-west-1
    ```

2.  **AWS Credentials File** (Recommended for local dev):
    File: `~/.aws/credentials` (Linux/Mac) or `%USERPROFILE%\.aws\credentials` (Windows)
    ```ini
    [default]
    aws_access_key_id = AKIA...
    aws_secret_access_key = wJalr...
    region = us-west-1
    ```

### Required Permissions Policy
The IAM user or role used must have a policy similar to:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "sagemaker:InvokeEndpoint",
            "Resource": "arn:aws:sagemaker:us-west-1:123456789012:endpoint/audio-ast-v1"
        }
    ]
}
```

## 3. Testing via Application API (Proxy)

The easiest way to test is through the local FastAPI server, which handles the AWS authentication for you.

**Endpoint**: `POST http://127.0.0.1:8000/classify-cloud`
**Content-Type**: `multipart/form-data`

### Using `curl`
```bash
curl -X POST "http://127.0.0.1:8000/classify-cloud" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/audio.wav"
```

### Using Python (requests)
```python
import requests

url = "http://127.0.0.1:8000/classify-cloud"
files = {'file': open('test_audio.wav', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

## 4. Direct SageMaker Invocation (Advanced)

If you want to bypass the app and call SageMaker directly, you must sign your requests using **AWS Signature Version 4 (SigV4)**. Standard HTTP tools like Postman (without AWS auth config) or simple `curl` will fail.

### Using AWS CLI
The AWS CLI handles signing automatically.
```bash
aws sagemaker-runtime invoke-endpoint \
    --endpoint-name audio-ast-v1 \
    --body fileb://test_audio.wav \
    --content-type audio/wav \
    --region us-west-1 \
    output_file.json
```

### Using Python (boto3)
This is how `app.py` implements the call internally.

```python
import boto3

# Initialize client (uses credentials from env or ~/.aws/credentials)
runtime = boto3.client('sagemaker-runtime', region_name='us-west-1')

# Read file
with open('test_audio.wav', 'rb') as f:
    payload = f.read()

# Invoke
response = runtime.invoke_endpoint(
    EndpointName='audio-ast-v1',
    ContentType='audio/wav',
    Body=payload
)

# Read response
result = response['Body'].read().decode('utf-8')
print(result)
```

## 5. Request & Response Format

### Request
- **Method**: `POST`
- **Headers**:
    - `Content-Type`: `audio/wav` (or `audio/mp3`, `audio/flac`)
- **Body**: Binary audio data (raw bytes)

### Response
The SageMaker endpoint returns a JSON object.
```json
{
  "predicted_class": "emergency_sirens",
  "confidence": 0.999,
  "probabilities": {
    "emergency_sirens": 0.999,
    "road_traffic": 0.001,
    ...
  }
}
```
*Note: The exact JSON structure may vary slightly depending on the model's inference script (e.g., sometimes wrapped in a list).*
