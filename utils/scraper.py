# credit goes to https://github.com/nicolas-gervais for the starter code
# Well, everything dealing with connecting to the website is his work. I overhauled most of his pandas
#   code to make it cleaner and readable using established pandas methods and techniques.

import bs4 as bs
from urllib.request import Request, urlopen
import pandas as pd
import os
import re
from tqdm.auto import tqdm
from random import choices

website = 'https://www.thecarconnection.com'
template = 'https://images.hgmsites.net/'

"""
Final DataFrame format:
-------------------------------------------
| Make| Model| Year| MSRP | Picture Links|
-------------------------------------------
"""


class Scraper:
    def __init__(self):
        self.df = None


    def _fetch(self, page, addition='')->bs.BeautifulSoup:
        try:
            soup = bs.BeautifulSoup(
                urlopen(Request(
                    page + addition,
                    headers={
                        'User-Agent': 'Opera/9.80 (X11; Linux i686; Ubuntu/14.10) Presto/2.12.388 Version/12.16'
                    })).read(),
                    'lxml'
                )
        except KeyboardInterrupt as e:
            exit()
        except Exception as e:
            soup = None
        return soup


    def _all_makes(self, n_makes: int)->list:
        all_makes_list = []
        for a in self._fetch(website, "/used-cars").find_all("a", {"class": ""}):
            if a: all_makes_list.append(a['href'])
        
        # only keep elements in list that start with "/make/"
        all_makes_list = [i for i in all_makes_list if i.startswith('/make/')]
        if n_makes:
            all_makes_list = choices(all_makes_list, k=n_makes)
        return all_makes_list


    def _make_menu(self, all_makes)->pd.DataFrame:
        makes_df = pd.DataFrame(columns=['Make', 'Models'])
        for i, make in enumerate(tqdm(all_makes, desc='getting hrefs for the makes for each model')): 
            for div in self._fetch(website, make).find_all("div", {"class": "name"}):
                if div: makes_df.loc[len(makes_df)] = [make, div.find_all("a")[0]['href']]
        return makes_df


    def _get_model_specification_links(self, df: pd.DataFrame)->pd.DataFrame:
        new_df = pd.DataFrame(columns=['Make', 'Model', 'Specifications Link'])
        for i, row in tqdm(df.iterrows(), desc='getting hrefs for the models for each make', total=len(df)):
            make, model = row['Make'], row['Models']
            soup = self._fetch(website, model)
            if soup:
                for div in soup.find_all("a", {"class": "btn avail-now first-item"}):
                    new_df.loc[len(new_df)] = [make, model, div['href'].replace('overview', 'specifications')]
                for div in soup.find_all("a", {"class": "btn 1"})[:8]:
                    new_df.loc[len(new_df)] = [make, model, div['href'].replace('overview', 'specifications')]
        return new_df


    def _specs_and_pics(self, df)->pd.DataFrame:
        df['Picture Links'] = df['Specifications Link'].apply(lambda x: x.replace('specifications', 'photos'))
        specifications_df = pd.DataFrame()
        for i, row in tqdm(
                df[["Specifications Link", "Picture Links"]].iterrows(), 
                desc='getting specs and pics',
                total=len(df)
            ):
            specification_link = row['Specifications Link']
            pic_link = row['Picture Links']

            try:
                soup = self._fetch(website, specification_link)
                make = soup.find_all('a', {'id': 'a_bc_1'})[0].text.strip()
                model = soup.find_all('a', {'id': 'a_bc_2'})[0].text.strip()
                year = soup.find_all('a', {'id': 'a_bc_3'})[0].text.strip()
                msrp = float(soup.find_all('span', {'class': 'msrp'})[0].text.replace('$', '').replace(',', '').strip())
                spec_names, spec_values = ['Make', 'Model', 'Year', 'MSRP'], [make, model, year, msrp]
                fetch_pics_url = str(self._fetch(website, pic_link))
                picture_links = []
                try:
                    for photo in re.findall('sml.+?_s.jpg', fetch_pics_url)[:150]: picture_links.append(photo.replace('\\', ''))
                except:
                    print('Error with {}.'.format(template + photo))
                spec_names.append('Picture Links')
                spec_values.append(list(tuple(picture_links)))
                if i == 0: specifications_df = pd.DataFrame(columns=spec_names)
                specifications_df.loc[len(specifications_df)] = spec_values
            except:
                print('Problem with {}.'.format(website + specification_link))
            
        return specifications_df


    def scrape(self, n_makes: int = None)->pd.DataFrame:
        makes_list = self._all_makes(n_makes)
        make_menu = self._make_menu(makes_list)
        specs = self._get_model_specification_links(make_menu)
        self.df = self._specs_and_pics(specs)
        return self.df
