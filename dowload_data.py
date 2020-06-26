from configparser import ConfigParser  
from binance.client import Client
import pandas as pd

if __name__ == "__main__":
    config = ConfigParser()  
    config.read('config/config.ini')  
    api_key = config.get('auth', 'Bin_api_key')  
    api_secret = config.get('auth', 'Bin_api_secret') 

    client = Client(api_key, api_secret)
    prices = client.get_all_tickers()

    symbols = list(filter(lambda x: x["symbol"].endswith("BTC"), prices))
    symbols_to_remove = ["ETHBTC", "BNBBTC"]
    symbols = [elem["symbol"] for elem in symbols if elem["symbol"] not in symbols_to_remove]
    symbols.append("BTCUSDT")
    print("Number of symbol: %s" %len(symbols))
    for symbol in symbols:
        print("Retrive klines of %s" %symbol)
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_5MINUTE, "1 Jan, 2016")
        print("Retrive klines done")
        # klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_5MINUTE, "1 hour ago UTC")
        data = pd.DataFrame(klines, columns=['Open time','Open','High','Low','Close','Volume','Close time',
                                                            'Quote asset volume','Number of trades','Taker buy base asset volume',
                                                            'Taker buy quote asset volume','Ignore'])
        data = data.drop(columns=['Ignore'])
        print('Data saved to : %s'%symbol)        
        data.to_csv('data/RAW/%s.csv'%symbol, index=False)
