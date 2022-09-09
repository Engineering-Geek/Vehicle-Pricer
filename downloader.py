# create a script to download images from a given url
from numpy import floor
import requests
import os
from tqdm.auto import tqdm
import pandas as pd
from ast import literal_eval
from pathlib import Path
from random import choices

template = 'https://images.hgmsites.net/'
FRACTION = 0.125

def download_image(url: str, filepath: str):
    """
    Download an image from a given url and save it to a filepath
    """
    response = requests.get(url)
    dirpath = os.path.dirname(filepath)
    if response.status_code == 200:
        Path(dirpath).mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        return True
    else:
        print("Error downloading image")
        return False


def download_images(filepath: str):
    os.chdir(filepath)
    data = pd.read_csv('data.csv')
    data['Picture Links'] = data['Picture Links'].apply(lambda x: literal_eval(x))
    master_csv = pd.DataFrame(columns=['Make', 'Model', 'Year', 'MSRP', 'Filepaths'])
    for i, row in tqdm(data.iterrows(), desc='downloading images', total=len(data)):
        urls = row['Picture Links']
        image_filepaths = []
        for i, url in enumerate(tqdm(choices(urls, k=int(len(urls)*FRACTION)), desc='downloading images for {} {} {}'.format(row['Make'], row['Model'], row['Year']), total=int(len(urls)*FRACTION))):
            image_filepath = os.path.join(filepath, 'images', row['Make'], row['Model'], str(row['Year']), str(i) + '.jpg')
            download_image(template + url, image_filepath)
            image_filepaths.append(image_filepath)
        master_csv.loc[len(master_csv)] = [
            row['Make'], 
            row['Model'], 
            row['Year'],
            row['MSRP'],
            image_filepaths
        ]
    master_csv.to_csv('master_data.csv', index=False)
        


if __name__ == "__main__":
    download_images("C:\\Users\\melgi\\portfolio\\data")