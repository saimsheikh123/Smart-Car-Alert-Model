"""
Interactive AWS setup for SageMaker deployment
Creates S3 bucket, SageMaker execution role, and validates setup
"""
import argparse
import json
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def check_aws_credentials():
    """Verify AWS credentials are configured"""
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✓ AWS credentials verified")
        print(f"  Account: {identity['Account']}")
        print(f"  User ARN: {identity['Arn']}")
        return True
    except Exception as e:
        print(f"✗ AWS credentials not found or invalid: {e}")
        print("\nPlease configure AWS credentials:")
        print("  aws configure")
        print("Or set environment variables:")
        print("  $env:AWS_ACCESS_KEY_ID='...'")
        print("  $env:AWS_SECRET_ACCESS_KEY='...'")
        print("  $env:AWS_DEFAULT_REGION='us-east-1'")
        return False


def create_s3_bucket(bucket_name, region):
    """Create S3 bucket for model artifacts"""
    s3 = boto3.client('s3', region_name=region)
    
    try:
        # Check if bucket exists
        s3.head_bucket(Bucket=bucket_name)
        print(f"✓ S3 bucket '{bucket_name}' already exists")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            # Bucket doesn't exist, create it
            try:
                if region == 'us-east-1':
                    s3.create_bucket(Bucket=bucket_name)
                else:
                    s3.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                print(f"✓ Created S3 bucket: s3://{bucket_name}")
                
                # Enable versioning (optional but recommended)
                s3.put_bucket_versioning(
                    Bucket=bucket_name,
                    VersioningConfiguration={'Status': 'Enabled'}
                )
                print(f"  Enabled versioning on bucket")
                return True
            except ClientError as create_error:
                print(f"✗ Failed to create bucket: {create_error}")
                return False
        else:
            print(f"✗ Error checking bucket: {e}")
            return False


def create_sagemaker_role(role_name, region):
    """Create SageMaker execution role with necessary permissions"""
    iam = boto3.client('iam', region_name=region)
    
    # Trust policy allowing SageMaker to assume this role
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        # Check if role exists
        role = iam.get_role(RoleName=role_name)
        role_arn = role['Role']['Arn']
        print(f"✓ SageMaker role '{role_name}' already exists")
        print(f"  ARN: {role_arn}")
        return role_arn
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            # Role doesn't exist, create it
            try:
                response = iam.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(trust_policy),
                    Description='SageMaker execution role for audio model deployment'
                )
                role_arn = response['Role']['Arn']
                print(f"✓ Created SageMaker role: {role_name}")
                print(f"  ARN: {role_arn}")
                
                # Attach managed policies
                policies = [
                    'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
                    'arn:aws:iam::aws:policy/AmazonS3FullAccess'
                ]
                
                for policy_arn in policies:
                    iam.attach_role_policy(
                        RoleName=role_name,
                        PolicyArn=policy_arn
                    )
                    policy_name = policy_arn.split('/')[-1]
                    print(f"  Attached policy: {policy_name}")
                
                # Wait for role to propagate
                print("  Waiting for role to propagate (10s)...")
                time.sleep(10)
                
                return role_arn
            except ClientError as create_error:
                print(f"✗ Failed to create role: {create_error}")
                return None
        else:
            print(f"✗ Error checking role: {e}")
            return None


def verify_sagemaker_access(region):
    """Verify SageMaker service is accessible"""
    try:
        sm = boto3.client('sagemaker', region_name=region)
        # Simple API call to verify access
        sm.list_endpoints(MaxResults=1)
        print(f"✓ SageMaker service accessible in {region}")
        return True
    except Exception as e:
        print(f"✗ Cannot access SageMaker: {e}")
        return False


def save_config(bucket_name, role_arn, region, output_file):
    """Save configuration for deployment script"""
    config = {
        "bucket": bucket_name,
        "role": role_arn,
        "region": region,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Configuration saved to: {output_file}")
    print("\nYou can now deploy using:")
    print(f"  python sagemaker/deploy_to_sagemaker.py --variant ast --bucket {bucket_name} --role {role_arn} --region {region} --endpoint-name audio-ast-v1")


def main():
    parser = argparse.ArgumentParser(description="Set up AWS resources for SageMaker deployment")
    parser.add_argument("--bucket-name", help="S3 bucket name (default: auto-generated)")
    parser.add_argument("--role-name", default="SageMakerAudioModelRole", help="IAM role name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--config-file", default="aws_config.json", help="Output config file")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AWS SageMaker Setup for Audio Model Deployment")
    print("=" * 60)
    print()
    
    # Step 1: Check credentials
    print("[1/4] Checking AWS credentials...")
    if not check_aws_credentials():
        sys.exit(1)
    print()
    
    # Step 2: Create S3 bucket
    print("[2/4] Setting up S3 bucket...")
    if not args.bucket_name:
        # Auto-generate bucket name using account ID
        sts = boto3.client('sts')
        account_id = sts.get_caller_identity()['Account']
        args.bucket_name = f"sagemaker-audio-models-{account_id}"
        print(f"  Using auto-generated bucket name: {args.bucket_name}")
    
    if not create_s3_bucket(args.bucket_name, args.region):
        sys.exit(1)
    print()
    
    # Step 3: Create SageMaker role
    print("[3/4] Setting up SageMaker execution role...")
    role_arn = create_sagemaker_role(args.role_name, args.region)
    if not role_arn:
        sys.exit(1)
    print()
    
    # Step 4: Verify SageMaker access
    print("[4/4] Verifying SageMaker access...")
    if not verify_sagemaker_access(args.region):
        sys.exit(1)
    print()
    
    # Save configuration
    config_path = Path(__file__).parent / args.config_file
    save_config(args.bucket_name, role_arn, args.region, config_path)
    
    print("\n" + "=" * 60)
    print("✓ AWS setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
