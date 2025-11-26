"""
Test deployed SageMaker endpoint with sample audio
"""
import argparse
import json
from pathlib import Path

import boto3


def invoke_endpoint(endpoint_name, audio_file_path, region):
    """Send audio file to SageMaker endpoint and get prediction"""
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    # Read audio file
    with open(audio_file_path, 'rb') as f:
        audio_bytes = f.read()
    
    print(f"Sending {len(audio_bytes)} bytes to endpoint '{endpoint_name}'...")
    
    # Invoke endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='audio/wav',
        Body=audio_bytes
    )
    
    # Parse result
    result = json.loads(response['Body'].read().decode())
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Test SageMaker endpoint with audio file")
    parser.add_argument("--endpoint", required=True, help="SageMaker endpoint name")
    parser.add_argument("--audio", required=True, help="Path to audio file (.wav)")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--config", default="aws_config.json", help="Load region from config")
    
    args = parser.parse_args()
    
    # Load region from config if not specified
    if args.region == "us-east-1":
        config_path = Path(__file__).parent / args.config
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    args.region = config.get('region', args.region)
            except Exception:
                pass
    
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    print("=" * 60)
    print("Testing SageMaker Endpoint")
    print("=" * 60)
    print(f"Endpoint: {args.endpoint}")
    print(f"Audio: {audio_path.name}")
    print(f"Region: {args.region}")
    print()
    
    try:
        result = invoke_endpoint(args.endpoint, audio_path, args.region)
        
        print("\n" + "=" * 60)
        print("Prediction Result")
        print("=" * 60)
        print(json.dumps(result, indent=2))
        
        if result.get('success'):
            print(f"\n✓ Predicted class: {result['class']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            
            if 'top2' in result and len(result['top2']) > 1:
                print(f"\nTop 2 predictions:")
                for i, pred in enumerate(result['top2'][:2], 1):
                    print(f"  {i}. {pred['class']}: {pred['confidence']:.2%}")
        else:
            print(f"\n✗ Prediction failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"\n✗ Error invoking endpoint: {e}")
        print("\nTroubleshooting:")
        print("  - Verify endpoint is 'InService':")
        print(f"    aws sagemaker describe-endpoint --endpoint-name {args.endpoint} --region {args.region}")
        print("  - Check CloudWatch logs for errors")


if __name__ == "__main__":
    main()
