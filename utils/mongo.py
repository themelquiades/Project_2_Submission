# Import needed libraries
from pymongo import MongoClient
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Import DateOffSet
from pandas.tseries.offsets import DateOffset

def connect_to_db(connection_str):
    # Gets MongoDB Connection String
    # MDB_CONNECTION_STRING = os.getenv('MDB_CONNECTION_STRING')

    # Function to connect to the Mongo DB
    def get_database():
        try:
            client = MongoClient(connection_str)
            db = client["project-02"]
            return db
        except Exception as e:
            print(e)

    # Connect to the db
    db = get_database()

    # Test Connection
    serverStatusResult=db.command("serverStatus")
    print(serverStatusResult["version"])
    
    return db

# Gets the coinpairs to process from the database
def get_coinpairs(db):
    # Get the coinpairs from the Database
    db_coinpairs = db["coinpairs"].find({"exchange" : "binance"})

    # convert the dictionary objects to dataframe
    binance_coinpairs_df = pd.DataFrame(db_coinpairs)

    # see the magic
    coinpair_list = list(binance_coinpairs_df['pair'])
    return coinpair_list

# Read MongoDB and return a Pandas DataFrame with the coin pair data
def get_coinpair_df_mongo(db, coinpair, timeframe, last_date):
    
    # Defines collection name
    collection_name = coinpair + "_" + timeframe
    
    # Get the coinpairs from the Database
    coinpair = db[collection_name].find({}).sort("open_time", 1)

    # convert the dictionary objects to dataframe
    coinpair_df = pd.DataFrame(coinpair)
    
    # Drop columns we don't need
    coinpair_df = coinpair_df.drop(columns=['_id',"close_time"])
    
    # Set Open Time as Index
    coinpair_df = coinpair_df.set_index('open_time')
    
    return coinpair_df

# Pulls the data from the Fear and Greed Index and returns it as a Pandas DataFrame
def get_google_trends_mongo(db, timeframe, coin_name, last_date):
    # Replaces "m" (minute) to "T" for Pandas resample
    timeframe = timeframe.replace("m", "T")
    
    # Get the coinpairs from the Database
    trends = db["trends_"+coin_name].find({}, {"_id": 0 })
    
    # Convert the dictionary objects to dataframe
    df_trends = pd.DataFrame(trends)
    
    # Rename "date" to "timestamp" for better understanding
    df_trends.columns = df_trends.columns.str.replace('date', 'timestamp')
    
    # Resample date given our timeframe using FILL
    df_trends = df_trends.resample(on="timestamp",rule=timeframe).first().ffill().drop('timestamp', axis=1).reset_index()

    # Sets Timestamp as Index
    df_trends = df_trends.set_index('timestamp')

    return df_trends

# Pulls the data from the Fear and Greed Index and returns it as a Pandas DataFrame
def get_fng_mongo(db, coinpair, timeframe, last_date):
    # Replaces "m" (minute) to "T" for Pandas resample
    timeframe = timeframe.replace("m", "T")
    
    # Get the coinpairs from the Database
    fng = db["fear_greed_index"].find({}, {"_id": 0, "value_classification": 0 })
    
    # Convert the dictionary objects to dataframe
    df_fg = pd.DataFrame(fng)

    # Rename "value" to "fear_greed" for better understanding
    df_fg.columns = df_fg.columns.str.replace('value', 'fear_greed')
    
    # Resample date given our timeframe
    df_fg = df_fg.resample(on="timestamp",rule=timeframe).first().ffill().drop('timestamp', axis=1).reset_index()
    
    # Sets Timestamp as Index
    df_fg = df_fg.set_index('timestamp')
    
    return df_fg

# Gets the coinpairs to process from the database
def get_keywords(db, coinpair):
    # Get the coinpairs from the Database
    db_coinpairs = db["coinpairs"].find({"pair": coinpair, "exchange" : "binance"})

    # convert the dictionary objects to dataframe
    binance_coinpairs_df = pd.DataFrame(db_coinpairs)

    # see the magic
    coinpair_list = list(binance_coinpairs_df['keywords'][0])
    
    return coinpair_list

# Gets the main coin name to process from the database
def get_main_coin(db, coinpair):
    # Get the coin name from the Database
    db_coinpairs = db["coinpairs"].find({"pair": coinpair, "exchange" : "binance"})

    # convert the dictionary objects to dataframe
    binance_coinpairs_df = pd.DataFrame(db_coinpairs)

    # see the magic
    coinpair_list = binance_coinpairs_df['main'][0]
    
    return coinpair_list

# Insert MongoDB and return a Pandas DataFrame with the coin pair info
def save_models_mongo(db, data):
    # Define the column name
    collection_name = "predictions"
    data["timestamp"] = datetime.utcnow()
    
    # Get the coinpairs from the Database
    predictions = db[collection_name].insert_one(data)

    return

# Read MongoDB and return a Pandas DataFrame with the coin pair data
def get_last_coinpair_update(db, coinpair, timeframe, date_range={"training_start":12, "testing_range": 2, "period": "weeks"}):
    
    # Defines collection name
    collection_name = coinpair + "_" + timeframe

    # Get the coinpairs from the Database
    coinpair = db[collection_name].find({}, {"_id": 0, "open_time": 1}).sort("open_time", -1).limit(1)

    # convert the dictionary objects to dataframe
    coinpair_df = pd.DataFrame(coinpair)
    
    # Last updated in coinpair
    last_updated = coinpair_df["open_time"][0]
    
    # Use monthly
    if date_range["period"] == "months":
        start_date = last_updated - DateOffset(months=date_range["training_start"]) - DateOffset(months=12) 
        
    # Use weekly
    elif date_range["period"] == "weeks":
        start_date = last_updated - DateOffset(weeks=date_range["training_start"]) - DateOffset(weeks=52) 
        
    # Use hours
    elif date_range["period"] == "hours":
        start_date = last_updated - DateOffset(hours=date_range["training_start"]) - DateOffset(weeks=12) 
        
    # Use minutes
    elif date_range["period"] == "minutes":
        start_date = last_updated - DateOffset(minutes=date_range["training_start"]) - DateOffset(weeks=3) 
    
    return start_date

# Function to get the timeframes stored in the database
def get_binance_timeframes_list(db):
    # Get the timeframes from the Database
    db_binance_timeframes = db["binance_timeframes"].find()

    # Convert the dictionary objects to dataframe
    binance_timeframes_df = pd.DataFrame(db_binance_timeframes)

    # Return the timeframes from the db
    return list(binance_timeframes_df['timeframe'])

# Get latest inserted kline date for each Coin Pair
def get_pair_data(db, binance_timeframes_list, coinpair_list):
    complete_pair_tf = []
    from_timestamp = "3 months ago UTC"
    for timeframe in binance_timeframes_list:
        for pair in coinpair_list:
            # Get the coinpairs from the Database
            collection_name = pair+"_"+timeframe
            latest = db[collection_name].find().sort('open_time', -1 ).limit(1)
            # Exists, update collection by setting from_timestamp to lastest timestamp
            if collection_name in db.list_collection_names():
                complete_pair_tf.append([timeframe, pair, latest[0]["open_time"].strftime("%m/%d/%Y, %H:%M:%S"),collection_name, "is_update"])
            # Does not exists, import from csv
            else:
                complete_pair_tf.append([timeframe, pair, from_timestamp, collection_name, "is_new"])
    return complete_pair_tf

# Save trends to Database
def update_trends_db(db, coin_name, trends_df):
    # Resets the index
    trends_df = trends_df.reset_index()
    
    # Checks if there no data to update then skips it
    if len(trends_df) > 0:
        # Update the trends for the Main Coin in the Database
        db["trends_"+coin_name].insert_many(trends_df.to_dict("records"))

# Save trends to Database
def insert_trends_db(db, coin_name, trends_df):
    # Resets the index
    trends_df = trends_df.reset_index()
    
    # Insert the coinpairs in the Database
    db["trends_"+coin_name].insert_many(trends_df.to_dict("records"))
    db["trends_"+coin_name].create_index([ ("date", -1) ])

    return

# Updates the coinpairs in the database
def update_coinpairs(db, coinpair_object):
    coinpair_object["exchange"] = "binance"
    # Update the coinpairs in the Database
    db["coinpairs"].update_one({"pair" : coinpair_object['pair']},
    {"$setOnInsert": coinpair_object},
    upsert=True)

    return

# Updates the timeframes in the database
def update_timeframes(db, timeframe):
    # Update the timeframes in the Database
    db["binance_timeframes"].update_one({"timeframe" : timeframe},
    {"$setOnInsert": {"timeframe": timeframe}},
    upsert=True)

    return timeframe

# Function to Update the Fear and Greed Index into MongoDB
def update_fng(db, fng_data_df):
    collection_name = "fear_greed_index"
    data = fng_data_df.to_dict("records")
    db[collection_name].drop()
    db[collection_name].insert_many(data)
    db[collection_name].create_index([ ("timestamp", -1) ])
    print("Done downloading Fear and Greed Index")
    
# Uploads CSV to database
def csv_to_mongo(db, csv_filename, collection_name):
    headers=["open_time", "open", "high", "low","close","volume","close_time","quote_asset_volume","numer_trades","taker_base_volume","taker_quote_volume","ignore"]
    kline_df = pd.read_csv(Path(csv_filename), parse_dates = True, infer_datetime_format = True, names=headers)
    kline_df['open_time'] = kline_df['open_time'].values.astype(dtype='datetime64[ms]')
    kline_df['close_time'] = kline_df['close_time'].values.astype(dtype='datetime64[ms]')
    kline_df[["open", "high", "low","close","volume","quote_asset_volume","taker_base_volume","taker_quote_volume"]] = kline_df[["open", "high", "low","close","volume","quote_asset_volume","taker_base_volume","taker_quote_volume"]].astype(float)
    kline_df = kline_df.drop(columns=['ignore'])
    kline_dict = kline_df.to_dict("records")
    db[collection_name].insert_many(kline_dict)
    db[collection_name].create_index([ ("open_time", -1) ])