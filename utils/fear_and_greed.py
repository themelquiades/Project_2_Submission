import pandas as pd
import requests

from utils.url_exists import (
    url_exists
)

# Function to Get the Fear and Greed Index from https://alternative.me
def get_fng():
    url = 'https://api.alternative.me/fng/?limit='+str(365*10)
    valid=url_exists(url)
    fng=requests.get(url).json()
    fng_data_df = pd.DataFrame(fng["data"])
    fng_data_df['timestamp'] = pd.to_datetime(fng_data_df['timestamp'], unit='s')
    fng_data_df = fng_data_df.drop(columns=['time_until_update'])
    fng_data_df['value'] = fng_data_df['value'].astype(int)
    return fng_data_df