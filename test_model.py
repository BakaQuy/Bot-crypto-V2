import pandas as pd
import numpy as np
from collections import deque 
from configparser import ConfigParser  

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

from slack import WebClient
from slack.errors import SlackApiError
import random

from os import listdir
from os.path import isfile, join, splitext, basename


SEQ_LEN = 60 # Number of indexes for 24 hours
FUTURE_PERIOD_PREDICT = 12 # Number of indexes for 1 hour (12x5min=1hour)

def get_file_list(path, extension=None):
    if extension is None:
        file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
    else:
        file_list = [
            path + f
            for f in listdir(path)
            if isfile(join(path, f)) and splitext(f)[1] == extension
        ]
    return file_list

def slack_report(client, payload):
    try:
        response = client.chat_postMessage(
            channel='trading-bot',
            text=payload)
        # assert response["message"]["text"] == payload
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")

def preprocess_data2(data, target_pct, timestamp_test):
    df = data.drop(columns=["Open time", "Max next hour", "Vol 24h"]).pct_change()
    # df = data.drop(columns=["Open","High","Low","Close","Open time", "Max next hour", "Vol 24h"])
    df["Green"] = (data["Open"] <= data["Close"]).astype(int)
    df["Red"] = (data["Open"] > data["Close"]).astype(int)
    df["Change"] = data["Close"]/data["Open"]-1
    df["Change High"] = data["High"]/data["Low"]-1
    df[["Open time", "Vol 24h"]] = data[["Open time", "Vol 24h"]]
    df["Max next hour"] = data["Max next hour"] >= target_pct*data["Close"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    scaler = StandardScaler()
    for col in df.columns:
        if col != "Max next hour" and col != "Vol 24h" and col != "Open time" and col != "Green" and col != "Red":            
            train_scale = scaler.fit_transform(df[col][df['Open time'] <= timestamp_test].values.reshape(-1, 1))
            test_scale = scaler.transform(df[col][df['Open time'] > timestamp_test].values.reshape(-1, 1))
            df[col] = np.concatenate((train_scale,test_scale), axis=0)
        # elif col == "Volume":
        #     scaler = MinMaxScaler()
        #     train_scale = scaler.fit_transform(df[col][df['Open time'] <= timestamp_test].values.reshape(-1, 1))
        #     test_scale = scaler.transform(df[col][df['Open time'] > timestamp_test].values.reshape(-1, 1))
        #     df[col] = np.concatenate((train_scale,test_scale), axis=0)
    return df

def build_sequential(data):
    df = data.copy()
    print("Build sequential data")
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    for idx,i in enumerate(df.values):
        prev_days.append([n for n in i[:-3]])
        if len(prev_days) == SEQ_LEN: # Check if volume 24h higher than 100 BTC
            sequential_data.append(np.array(prev_days))
    print("Sequential data done")
    return sequential_data

def perform_simulation_fast(df, target_pct, timestamp_test, model, coin_name):
    # config = ConfigParser()
    # config.read('config/config.ini')
    # slack_token = config.get("auth", "SLACK_TOKEN")
    # print(slack_token)
    # slack = WebClient(token="")
    # slack_report(slack, "BOT INIT")
    data = preprocess_data2(df, target_pct, timestamp_test)
    data = data[data['Open time'] > timestamp_test]
    X_new = np.asarray(build_sequential(data))
    prediction = model.predict(X_new, verbose=0)
    data = data.iloc[SEQ_LEN-1:]

    money_init = 1000
    money = money_init
    stop_loss = 0.97
    sell_count_limit = 6
    fee = 0.001
    fees = 0
    position_take = False
    buy_idx = 0
    buy_price = 0
    sell_count = 0
    good_move = 0
    bad_move = 0
    stop_loss_activated = 0
    big_payload = ""
    for idx,i in enumerate(data.values):
        if i[-2] >= 200 and position_take == False:
            if prediction[idx][1] >= 0.7 and df.loc[data.index[idx],"Close"]/df.loc[data.index[idx-SEQ_LEN],"Close"]-1 < 0.15: # Buy
                if idx == buy_idx+1:
                    buy_price = df.loc[data.index[idx],"Close"]
                    fee_tmp = money*fee
                    fees += fee_tmp
                    money -= fee_tmp
                    sell_count = 0
                    position_take = True
                buy_idx = idx
        elif position_take == True:
            if stop_loss*buy_price > df.loc[data.index[idx],"Close"] or target_pct*buy_price <= df.loc[data.index[idx],"Close"]: # 10 False
                if stop_loss*buy_price > df.loc[data.index[idx],"Close"]: 
                    sell_price = stop_loss*buy_price
                    stop_loss_activated += 1
                else: sell_price = df.loc[data.index[idx],"Close"]
            # if target_pct*buy_price <= df.loc[data.index[idx],"Close"]: # 10 False
                # sell_price = df.loc[data.index[idx],"Close"]
                fee_tmp = (money*(sell_price/buy_price))*fee
                money = money*(sell_price/buy_price) - fee_tmp
                fees += fee_tmp
                position_take = False
                payload = f"Money: {money:.2f} \nBuy price: {buy_price} at {data.index[buy_idx]+2} \nSell price: {sell_price} at {data.index[idx]+2} \nBenefice: {((sell_price/buy_price)-1)*100:.2f}% \nInstance: {idx}/{len(data.values)} \n"
                big_payload += payload
                print(payload)
                if buy_price < sell_price: good_move += 1
                else: bad_move += 1
            elif prediction[idx][1] <= prediction[idx][0]: # Sell
                sell_count += 1
                if sell_count == sell_count_limit: # 10 False
                    sell_price = df.loc[data.index[idx],"Close"]
                    fee_tmp = (money*(sell_price/buy_price))*fee
                    money = money*(sell_price/buy_price) - fee_tmp
                    fees += fee_tmp
                    position_take = False
                    payload = f"Money: {money:.2f} \nBuy price: {buy_price} at {data.index[buy_idx]+2} \nSell price: {sell_price} at {data.index[idx]+2} \nBenefice: {((sell_price/buy_price)-1)*100:.2f}% \nInstance: {idx}/{len(data.values)} \n"
                    big_payload += payload
                    print(payload)
                    if buy_price < sell_price: good_move += 1
                    else: bad_move += 1
            elif prediction[idx][1] >= 0.7:
                sell_count = 0 
    summary = f"SUMMARY: \nCoin: {coin_name} \nMoney end: {money:.2f} \ngood moves: {good_move} \nbad moves: {bad_move} \nstop loss: {stop_loss_activated} \nfees: {fees:.2f} \n"
    big_payload += summary
    # slack_report(slack, summary)
    return summary, money

if __name__ == "__main__":
    # list_coin = ["data/TargetMax/ICXBTC.csv", "data/TargetMax/DATABTC.csv", "data/TargetMax/MATICBTC.csv", "data/TargetMax/NANOBTC.csv"] # MFT ICX DOCK HOT
    list_coin = get_file_list("data/TargetMax/")
    slack = WebClient(token="")
    big_summary = "Test: take profit=3%, stop loss=3%, MAX%=15%, Volume limit=200\n" # Best = take profit=3%, stop loss=3%, MAX%=15%, Volume limit=150 or maybe 100 :)
    slack_report(slack, big_summary)
    money = 0
    coins = 0
    target_pct = 1.03
    timestamp_test = 1560000000000 # 8 June 2019 (1560000000000)
    model = load_model("models/60-SEQ-12-PRED-1588326862")
    for i in range(len(list_coin)):
        data = pd.read_csv(list_coin[i])
        coin_name = list_coin[i][15:-4]
        data.sort_values(by=['Open time'], inplace=True)
        if data.iloc[0]['Open time'] > timestamp_test-10*SEQ_LEN or data.iloc[-1]['Open time'] < timestamp_test+10*SEQ_LEN: continue
        data["Vol 24h"] = data["Quote asset volume"].rolling(SEQ_LEN).sum()
        if data["Vol 24h"].max() < 200: continue
        data.drop(columns=["Volume","Close time","Quote asset volume",
                            "Number of trades","Taker buy base asset volume",
                            "Taker buy quote asset volume"], inplace=True)
        summary, money_coin = perform_simulation_fast(data, target_pct, timestamp_test, model, coin_name)
        big_summary += summary
        money += money_coin
        coins += 1
        slack_report(slack, summary+f"Instance: ({i+1}/{len(list_coin)})")
    print(big_summary)
    slack_report(slack, f"FINAL BENEF: {money-coins*1000:.2f} ({(money/coins*1000)-1:.2f})")
    slack_report(slack,"END")
