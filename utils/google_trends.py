# Import libraries
import pandas as pd
import time
from datetime import datetime,timezone,timedelta
from pytrends.request import TrendReq
pytrend = TrendReq()

# Import our utils libraries
from utils.mongo import (
    update_trends_db,
    insert_trends_db
)

# Processes the Google Trends and returns the dataframe with the most relevant column
def process_trends(df):
    # Select non partial data
    new_df = df.copy()
    new_df = new_df[new_df['isPartial'] != True]
    
    # Drop Columns that are not needed
    new_df = new_df.drop(columns=['isPartial'])
    
    return new_df

def get_google_trends(db, coinpair, keywords, coin_name):
    # Gets current time in UTC
    now_utc = datetime.now(timezone.utc)
    
    # Pull the last date from Database
    collection_name ="trends_"+coin_name
    latest = db[collection_name].find().sort('date', -1 ).limit(1)
    
    # Sets start and end Dates
    date_start = latest[0]["date"] + timedelta(hours=1)
    date_end = now_utc
    
    # Pulls data from Google Trends
    try:
        df = pytrend.get_historical_interest(keywords, year_start=date_start.year, month_start=date_start.month, day_start=date_start.day, hour_start=date_start.hour, year_end=date_end.year, month_end=date_end.month, day_end=date_end.day, hour_end=date_end.hour, cat=0, geo='', gprop='', sleep=60)
        trends_df = process_trends(df)
        update_trends_db(db, coin_name, trends_df)
    except:
        print(f"Error with Google Trends, waiting 3 seconds..")
        time.sleep(3)
        
def populate_google_trends(db, coinpair, keywords, coin_name, months_back=24):
    i = months_back
    now_utc = datetime.now(timezone.utc)
    while i > 0:
        date_start = now_utc - pd.offsets.DateOffset(months=i)
        date_end = now_utc - pd.offsets.DateOffset(months=i-1)
        print(f"{date_start} - {date_end} | {keywords}")

        try:
            df = pytrend.get_historical_interest(keywords, year_start=date_start.year, month_start=date_start.month, day_start=date_start.day, hour_start=date_start.hour, year_end=date_end.year, month_end=date_end.month, day_end=date_end.month, hour_end=date_end.hour, cat=0, geo='', gprop='', sleep=0)
            trends_df = process_trends(df)
            insert_trends_db(db, coin_name, trends_df)
            i -=1
        except:
            print(f"Error with Google Trends, waiting 3 seconds..")
            time.sleep(3)