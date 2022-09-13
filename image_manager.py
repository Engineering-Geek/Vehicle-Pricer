# create a script to download images from a given url
from numpy import floor
import requests
import os
from tqdm.auto import tqdm
import pandas as pd
from ast import literal_eval
from random import choices
import boto3
from utils.scraper import Scraper

template = 'https://images.hgmsites.net/'

class ImageManager:
    def __init__(self, bucket_name):
        self.s3 = boto3.resource('s3')
        self.bucket = self.s3.Bucket(bucket_name)
        self.bucket_name = bucket_name
    
    def _url_image_to_bucket(self, url: str, filepath: str):
        """
        Download an image from a given url and save it to a filepath
        """
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            self.bucket.upload_fileobj(response.raw, filepath)
            return True
        else:
            print("Error downloading image")
            return False
    
    def _df_to_bucket(self, df: pd.DataFrame, filepath: str):
        """
        Upload a DataFrame to a bucket
        """
        df.to_csv(filepath, index=False)
        self.bucket.upload_file(filepath, filepath)
    
    def url_images_to_bucket(self, df: pd.DataFrame, fraction=0.0625):
        """
        Download images from a DataFrame of urls and save them to a bucket
        DataFrame should have columns 'Make', 'Model', 'Year', 'Image'
        """
        data = df.sample(frac=fraction)
        master_csv = pd.DataFrame(columns=['Make', 'Model', 'Year', 'MSRP', 'Filepaths', 'Dirpath'])
        for i, row in tqdm(data.iterrows(), desc='downloading images', total=len(data)):
            urls = row['Picture Links']
            image_filepaths = []
            for i, url in enumerate(tqdm(choices(urls, k=int(len(urls)*fraction)), 
                                         desc='downloading images for {} {} {}'.format(row['Make'], row['Model'], row['Year']), 
                                         total=int(len(urls)*fraction))):
                filepath = 'images/{}/{}/{}/{}.jpg'.format(row['Make'], row['Model'], row['Year'], i)
                image_filepaths.append(filepath)
                self._url_image_to_bucket(template + url, filepath)
                
            dirpath = 'images/{}/{}/{}'.format(row['Make'], row['Model'], row['Year'])
            master_csv.loc[len(master_csv)] = [
                row['Make'], 
                row['Model'], 
                row['Year'],
                row['MSRP'],
                image_filepaths,
                dirpath
            ]
        master_csv.to_csv('master.csv', index=False)
        self._df_to_bucket(master_csv, 'master.csv')
        os.remove('master.csv')
        return master_csv

    def download_image(self, uri: str, filepath: str):
        self.s3.meta.client.download_file(self.bucket_name, uri, filepath)
    
    def purge_bucket(self):
        self.bucket.objects.all().delete()
    
    def scrape(self, n_makes: int = None):
        scraper = Scraper()
        df = scraper.scrape(n_makes=n_makes)
        return df
        


if __name__ == "__main__":
    im = ImageManager('sagemaker-vehicle-pricer-data')
    print("purging...")
    im.purge_bucket()
    print("purged")
    df = im.scrape(n_makes=10)
    df_master = im.url_images_to_bucket(df, fraction=0.0625)
