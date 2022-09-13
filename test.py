import boto3
from datetime import datetime, timezone

today = datetime.now(timezone.utc)

s3 = boto3.client('s3')

objects = s3.list_objects(Bucket='sagemaker-vehicle-pricer-data')

print(objects["Contents"][0])
