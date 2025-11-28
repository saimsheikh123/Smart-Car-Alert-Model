import boto3

sm = boto3.client('sagemaker', region_name='us-west-1')
endpoint = sm.describe_endpoint(EndpointName='audio-ast-v1')
print(f"Endpoint: audio-ast-v1")
print(f"Status: {endpoint['EndpointStatus']}")
print(f"Instance: {endpoint.get('InstanceType', 'N/A')}")
