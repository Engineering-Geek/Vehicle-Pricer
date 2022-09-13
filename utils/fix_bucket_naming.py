from tkinter import image_types
import boto3
import pandas as pd
import ast
import os
from tqdm.auto import tqdm


def process_data(old_bucket_uri: str, new_bucket_uri: str):
    """
    Process data from old bucket to new bucket, specifically renaming the images

    Args:
        old_bucket_uri (str): _description_
        new_bucket_uri (str): _description_
    """
    input("DO NOT RUN THIS SCRIPT UNLESS YOU KNOW WHAT YOU ARE DOING. Press Enter to continue...")
    s3 = boto3.resource('s3')
    # create new bucket if it doesnt exist
    if new_bucket_uri not in [bucket.name for bucket in s3.buckets.all()]:
        s3.create_bucket(Bucket=new_bucket_uri)
    old_bucket = s3.Bucket(old_bucket_uri)
    new_bucket = s3.Bucket(new_bucket_uri)
    # dataframe located in old bucket
    old_bucket.download_file('master.csv', 'master.csv')
    df = pd.read_csv('master.csv')
    os.remove('master.csv')
    for _, row in tqdm(df.iterrows(), desc='processing image filepaths', total=len(df)):
        msrp, make, model, year = row['MSRP'], row['Make'], row['Model'], row['Year']
        old_filepaths = ast.literal_eval(row['Filepaths'])
        image_type = old_filepaths[0].split('.')[-1]
        new_filepaths = ["{}_{}_{}_{}_{}.{}".format(make, model, year, msrp, i, image_type) 
                         for i in range(len(old_filepaths))]
        for old_filepath, new_filepath in zip(old_filepaths, new_filepaths):
            copy_source = {
                'Bucket': old_bucket_uri,
                'Key': old_filepath
            }
            new_bucket.copy(copy_source, new_filepath)
    print('Done!')





