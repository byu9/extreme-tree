from glob import glob
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd

from nyiso.load_zones import months
from nyiso.load_zones import years
from nyiso.load_zones import zone_filenames

fetch_urls = [
    f'http://mis.nyiso.com/public/csv/pal/{year:4}{month:02}01pal_csv.zip'
    for year in years
    for month in months
]

filenames = [
    Path(f'fetch_load/nyiso-{year}-{month}.zip')
    for year in years
    for month in months
]

for url, filename in zip(fetch_urls, filenames):
    if not filename.is_file():
        urlretrieve(url, filename=filename)

for filename in filenames:
    with ZipFile(filename, 'r') as file:
        file.extractall('unpack_load')

load_fragments = [
    pd.read_csv(filename, index_col='Time Stamp', parse_dates=['Time Stamp'])
    for filename in glob('unpack_load/*.csv')
]
load_data = pd.concat(load_fragments, axis='index')
load_data = load_data.tz_localize('America/New_York', ambiguous=(load_data['Time Zone'] == 'EDT'))
load_data.index.name = 'Time'
load_data.drop(columns=['Time Zone', 'PTID'], inplace=True)
load_data.rename(inplace=True, columns={'Name': 'Zone'})
load_data = pd.pivot(load_data, columns='Zone')
load_data.columns = load_data.columns.swaplevel()

for zone_name, zone_filename in zone_filenames.items():
    load_data[zone_name].to_csv(f'combined_load/{zone_filename}.csv')
