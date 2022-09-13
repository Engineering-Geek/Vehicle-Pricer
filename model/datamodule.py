import pandas as pd
import os
import ast
import pandas as pd
from PIL import Image
import io
import boto3
os.environ['AWS_REGION'] = 'us-east-1' # set region, important for sagemaker

# AWS
from awsio.python.lib.io.s3.s3dataset import S3IterableDataset
# documentation: https://github.com/aws/amazon-s3-plugin-for-pytorch

from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms


class VehiclePricerDataset(IterableDataset):
    def __init__(self, bucket_uri: str, shuffle_urls: bool = False, transform: transforms.Compose = None) -> None:
        # check to see if url_list is a 2d list
        # s3://sagemaker-vehicle-pricer-data/master.csv
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(bucket_uri)
        bucket.download_file("master.csv", "master.csv")
        df = pd.read_csv("master.csv")
        dirpaths = df["Dirpath"].tolist()
        dirpaths = ["s3://{}/{}".format(bucket_uri, dirpath) for dirpath in dirpaths]
        msrp = df["MSRP"].tolist()
        print(dirpaths[0])
        exit()
        self.mapping = dict(zip(dirpaths, msrp))
        self.s3_iter_dataset = S3IterableDataset(dirpaths, shuffle_urls=shuffle_urls)
        self.transform = transform
    
    def data_generator(self):
        try:
            for uri, img in self.s3_iter_dataset_iterator: yield Image.open(io.BytesIO(img)).convert('RGB'), self.mapping[uri]
        except StopIteration:
            return
    
    def __iter__(self):
        self.s3_iter_dataset_iterator = iter(self.s3_iter_dataset)
        return self.data_generator()

    def __len__(self):
        return len(self.s3_iter_dataset)
    
    
    
if __name__ == "__main__":
    # test
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = VehiclePricerDataset("sagemaker-vehicle-pricer-data", shuffle_urls=True, transform=transforms)
    for i in dataset:
        print(i)
        break
    

