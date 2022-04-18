from binance.client import Client
from binance import AsyncClient
    
# Function to get the kline data from Binance
async def get_coinpair_kline(BINANCE_API_KEY, BINANCE_API_SECRET, pair, timeframe, from_timestamp):
    client = await AsyncClient.create(BINANCE_API_KEY, BINANCE_API_SECRET)
    if timeframe == "1m":
        klines = await client.get_historical_klines(pair, Client.KLINE_INTERVAL_1MINUTE, from_timestamp)
    elif timeframe == "5m":
        klines = await client.get_historical_klines(pair, Client.KLINE_INTERVAL_5MINUTE, from_timestamp)
    elif timeframe == "30m":
        klines = await client.get_historical_klines(pair, Client.KLINE_INTERVAL_30MINUTE, from_timestamp)
    elif timeframe == "1h":
        klines = await client.get_historical_klines(pair, Client.KLINE_INTERVAL_1HOUR, from_timestamp)
    elif timeframe == "1d":
        klines = await client.get_historical_klines(pair, Client.KLINE_INTERVAL_1DAY, from_timestamp)
    else:
        return
    await client.close_connection()
    return klines


        