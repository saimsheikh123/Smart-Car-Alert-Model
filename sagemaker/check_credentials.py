import boto3

try:
    sts = boto3.client('sts')
    identity = sts.get_caller_identity()
    print(f"✓ AWS credentials verified")
    print(f"  Account: {identity['Account']}")
    print(f"  User ARN: {identity['Arn']}")
except Exception as e:
    print(f"✗ AWS credentials not configured")
    print(f"  Error: {e}")
    print("\nTo configure AWS credentials:")
    print("  1. Get your AWS Access Key ID and Secret Access Key from AWS Console")
    print("  2. Set environment variables in PowerShell:")
    print("     $env:AWS_ACCESS_KEY_ID='your-access-key-id'")
    print("     $env:AWS_SECRET_ACCESS_KEY='your-secret-access-key'")
    print("     $env:AWS_DEFAULT_REGION='us-east-1'")
    print("\n  Or create ~/.aws/credentials file with:")
    print("     [default]")
    print("     aws_access_key_id = your-access-key-id")
    print("     aws_secret_access_key = your-secret-access-key")
