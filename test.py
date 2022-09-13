<<<<<<< HEAD
import boto3
from datetime import datetime, timezone

today = datetime.now(timezone.utc)

s3 = boto3.client('s3')

objects = s3.list_objects(Bucket='sagemaker-vehicle-pricer-data')

print(objects["Contents"][0])
=======
from datetime import datetime

import boto3

s3 = boto3.resource('s3')

my_bucket = s3.Bucket('sagemaker-vehicle-pricer-data')

last_modified_date = datetime(1939, 9, 1).replace(tzinfo=None)
for file in my_bucket.objects.all():
    file_date = file.last_modified.replace(tzinfo=None)
    if last_modified_date < file_date:
        last_modified_date = file_date

print(last_modified_date)