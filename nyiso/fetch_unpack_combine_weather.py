from glob import glob
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd

from load_zones import weather_station_mapping
from nyiso.load_zones import months
from nyiso.load_zones import years
from nyiso.load_zones import zone_filenames

fetch_urls = [
    f'http://mis.nyiso.com/public/csv/lfweather/{year:4}{month:02}01lfweather_csv.zip'
    for year in years
    for month in months
]

filenames = [
    Path(f'fetch_weather/nyiso-{year}-{month}.zip')
    for year in years
    for month in months
]

for url, filename in zip(fetch_urls, filenames):
    if not filename.is_file():
        urlretrieve(url, filename=filename)

for filename in filenames:
    with ZipFile(filename, 'r') as file:
        file.extractall('unpack_weather')

weather_station_zones = pd.DataFrame.from_dict(weather_station_mapping, orient='index')

weather_fragments = [
    pd.read_csv(filename, index_col='Forecast Date')
    for filename in glob('unpack_weather/*.csv')
]
weather_data = pd.concat(weather_fragments, axis='index')
weather_data.index = pd.to_datetime(weather_data.index).tz_localize('America/New_York')
weather_data.index.name = 'Time'
weather_data = weather_data[weather_data['Vintage'] == 'Forecast']
weather_data = weather_data[['Station ID', 'Max Temp', 'Min Temp', 'Max Wet Bulb', 'Min Wet Bulb']]

weather_data = weather_data.merge(weather_station_zones, left_on='Station ID', right_index=True)
weather_data = weather_data.drop(columns='Station ID')
weather_data = weather_data.rename(columns={0: 'Zone'})
weather_data = weather_data.reset_index().groupby(['Time', 'Zone']).mean().round(0)
weather_data = weather_data.reset_index().set_index('Time')
weather_data = pd.pivot(weather_data, columns='Zone')
weather_data.columns = weather_data.columns.swaplevel()

for zone_name, zone_filename in zone_filenames.items():
    weather_data[zone_name].to_csv(f'combined_weather/{zone_filename}.csv')
