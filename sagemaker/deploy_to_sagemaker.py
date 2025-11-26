import argparse
import os
import tarfile
import tempfile
from pathlib import Path
import json
import time

import boto3


def _tar_dir(output_tar: Path, pairs):
    with tarfile.open(output_tar, mode="w:gz") as tar:
        for src, arcname in pairs:
            tar.add(str(src), arcname=str(arcname))


def package_ast(model_root: Path, code_root: Path, out_tar: Path):
    files = [
        (model_root / "config.json", Path("config.json")),
        (model_root / "preprocessor_config.json", Path("preprocessor_config.json")),
    ]
    # model file
    model_file = model_root / "model.safetensors"
    if not model_file.exists():
        model_file = model_root / "pytorch_model.bin"
    files.append((model_file, Path(model_file.name)))
    # optional class names
    if (model_root / "class_names.json").exists():
        files.append((model_root / "class_names.json", Path("class_names.json")))
    # code folder
    files.append((code_root / "inference.py", Path("code/inference.py")))
    if (code_root / "requirements.txt").exists():
        files.append((code_root / "requirements.txt", Path("code/requirements.txt")))
    _tar_dir(out_tar, files)


def package_crnn(ckpt_path: Path, code_root: Path, out_tar: Path, class_names: Path | None):
    pairs = [
        (ckpt_path, Path("multi_audio_crnn.pth")),
        (code_root / "inference.py", Path("code/inference.py")),
    ]
    req = code_root / "requirements.txt"
    if req.exists():
        pairs.append((req, Path("code/requirements.txt")))
    if class_names and class_names.exists():
        pairs.append((class_names, Path("class_names.json")))
    _tar_dir(out_tar, pairs)


def upload_to_s3(s3, bucket: str, key: str, file_path: Path):
    s3.upload_file(str(file_path), bucket, key)
    return f"s3://{bucket}/{key}"


def deploy_ast(sagemaker_client, role: str, model_data: str, instance_type: str, endpoint_name: str, region: str):
    # Hugging Face DLC container - using a stable version available in us-west-1
    account_map = {
        "us-west-1": "763104351884",
        "us-east-1": "763104351884",
        "us-west-2": "763104351884",
        "eu-west-1": "763104351884",
    }
    account = account_map.get(region, "763104351884")
    # Using PyTorch 2.0 + Transformers 4.28 (more widely available)
    image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04"
    
    model_name = f"{endpoint_name}-model-{int(time.time())}"
    
    sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data,
        },
        ExecutionRoleArn=role,
    )
    
    endpoint_config_name = f"{endpoint_name}-config-{int(time.time())}"
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
            }
        ],
    )
    
    sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )
    
    return endpoint_name


def deploy_crnn(sagemaker_client, role: str, model_data: str, instance_type: str, endpoint_name: str, region: str):
    # PyTorch DLC container for pytorch 2.3
    account_map = {
        "us-west-1": "763104351884",
        "us-east-1": "763104351884",
        "us-west-2": "763104351884",
        "eu-west-1": "763104351884",
    }
    account = account_map.get(region, "763104351884")
    image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.3.0-cpu-py310-ubuntu22.04"
    
    model_name = f"{endpoint_name}-model-{int(time.time())}"
    
    sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data,
        },
        ExecutionRoleArn=role,
    )
    
    endpoint_config_name = f"{endpoint_name}-config-{int(time.time())}"
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
            }
        ],
    )
    
    sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )
    
    return endpoint_name


def main():
    parser = argparse.ArgumentParser(description="Package and deploy model to SageMaker")
    parser.add_argument("--variant", choices=["ast", "crnn"], required=True)
    parser.add_argument("--region", required=False, default=os.environ.get("AWS_REGION", "us-west-1"))
    parser.add_argument("--role", required=False, help="SageMaker execution role ARN")
    parser.add_argument("--bucket", required=False, help="S3 bucket for model artifacts")
    parser.add_argument("--config", default="aws_config.json", help="Load bucket/role from config file")
    parser.add_argument("--prefix", default="models", help="S3 key prefix")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--endpoint-name", required=True)
    # AST inputs
    parser.add_argument("--ast-model-root", default=str(Path(__file__).parent.parent / "ast_6class_model"))
    # CRNN inputs
    parser.add_argument("--crnn-ckpt", default=str(Path(__file__).parent.parent / "Audio_Models" / "Audio_Models" / "multi_audio_crnn.pth"))
    parser.add_argument("--class-names", default=str(Path(__file__).parent.parent / "ast_6class_model" / "class_names.json"))

    args = parser.parse_args()

    # Load config file if exists and bucket/role not provided
    config_path = Path(__file__).parent / args.config
    if config_path.exists() and (not args.bucket or not args.role):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            if not args.bucket:
                args.bucket = config.get('bucket')
                print(f"Using bucket from config: {args.bucket}")
            if not args.role:
                args.role = config.get('role')
                print(f"Using role from config")
            if not args.region and 'region' in config:
                args.region = config['region']
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    if not args.bucket:
        print("Error: --bucket required (or run setup_aws_resources.py first)")
        return

    boto_sess = boto3.Session(region_name=args.region)
    s3 = boto_sess.client("s3")
    sagemaker_client = boto_sess.client("sagemaker")
    role = args.role

    out_dir = Path(tempfile.mkdtemp())
    tar_path = out_dir / "model.tar.gz"

    if args.variant == "ast":
        model_root = Path(args.ast_model_root).resolve()
        code_root = Path(__file__).parent / "ast_hf" / "code"
        package_ast(model_root, code_root, tar_path)
        key = f"{args.prefix}/ast/{args.endpoint_name}/model.tar.gz"
        model_data = upload_to_s3(s3, args.bucket, key, tar_path)
        deploy_ast(sagemaker_client, role, model_data, args.instance_type, args.endpoint_name, args.region)
    else:
        ckpt = Path(args.crnn_ckpt).resolve()
        code_root = Path(__file__).parent / "crnn_pt" / "code"
        class_names = Path(args.class_names)
        class_names = class_names if class_names.exists() else None
        package_crnn(ckpt, code_root, tar_path, class_names)
        key = f"{args.prefix}/crnn/{args.endpoint_name}/model.tar.gz"
        model_data = upload_to_s3(s3, args.bucket, key, tar_path)
        deploy_crnn(sagemaker_client, role, model_data, args.instance_type, args.endpoint_name, args.region)

    print("\n" + "="*60)
    print("✓ Deployment request submitted successfully!")
    print("="*60)
    print(f"\nEndpoint name: {args.endpoint_name}")
    print(f"Instance type: {args.instance_type}")
    print(f"Model variant: {args.variant.upper()}")
    print(f"\nMonitor deployment status:")
    print(f"  aws sagemaker describe-endpoint --endpoint-name {args.endpoint_name} --region {args.region}")
    print(f"\nOr check in AWS Console:")
    print(f"  https://{args.region}.console.aws.amazon.com/sagemaker/home?region={args.region}#/endpoints/{args.endpoint_name}")
    print("\nDeployment typically takes 5-10 minutes to complete.")



if __name__ == "__main__":
    main()
