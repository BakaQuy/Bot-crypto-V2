from os import listdir
from os.path import isfile, join, splitext, basename
import multiprocessing
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import deque
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Activation
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint


"""
RUN WITH:
conda create -n tf-gpu
conda activate tf-gpu
conda install tensorflow-gpu=1.12
conda install cudatoolkit=9.0
"""

SEQ_LEN = 60 # Number of indexes for 5 hours
FUTURE_PERIOD_PREDICT = 12 # Number of indexes for 1 hour (12x5min=1hour)

EPOCHS = 10  # how many passes through our data
BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"  # a unique name for the model

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

def preprocess_data(data, target_pct, timestamp_test):
    df = data.copy()
    df["Max next hour"] = df["Max next hour"] >= target_pct*df["Close"]
    scaler = StandardScaler()
    for col in df.columns:
        if col != "Max next hour" and col != "Vol 24h" and col != "Open time":
            df[col] = df[col].pct_change()
            df.replace([np.inf, -np.inf], np.nan,inplace=True)
            df.dropna(inplace=True)
            train_scale = scaler.fit_transform(df[col][df['Open time'] <= timestamp_test].values.reshape(-1, 1))
            test_scale = scaler.transform(df[col][df['Open time'] > timestamp_test].values.reshape(-1, 1))
            df[col] = np.concatenate((train_scale,test_scale), axis=0)
    df.drop(columns=["Open time"], inplace=True)
    df.dropna(inplace=True)
    return df

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

def preprocess_data3(data, target_pct, timestamp_test):
    df = data.drop(columns=["Open time", "Max next hour", "Vol 24h"]).pct_change()
    df[["Open time", "Vol 24h"]] = data[["Open time", "Vol 24h"]]
    df["Max next hour"] = data["Max next hour"] >= target_pct*data["Close"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def build_sequential(data, timestamp_test):
    df = data.copy()
    print("Build sequential data")
    sequential_data_train = []
    sequential_data_test = []
    prev_days = deque(maxlen=SEQ_LEN)
    # scaler = MinMaxScaler()
    for idx,i in enumerate(df.values):
        prev_days.append([n for n in i[:-3]])
        if len(prev_days) == SEQ_LEN and i[-2] >= 150 and i[-3] <= timestamp_test: # Check if volume 24h higher than 100 BTC
            sequence = np.array(prev_days)
            # sequence[:,-1] = scaler.fit_transform(sequence[:,-1].reshape(-1, 1)).reshape(-1)
            sequential_data_train.append([sequence, i[-1]])
        elif len(prev_days) == SEQ_LEN and i[-2] >= 150 and i[-3] > timestamp_test: 
            sequence = np.array(prev_days)
            # sequence[:,-1] = scaler.fit_transform(sequence[:,-1].reshape(-1, 1)).reshape(-1)
            sequential_data_test.append([sequence, i[-1]])
    random.shuffle(sequential_data_train)
    random.shuffle(sequential_data_test)
    print("Sequential data done")
    return sequential_data_train, sequential_data_test

def process_sequential(sequential_data):
    targets_true = []  # list that will store our buy sequences and targets
    targets_false = []  # list that will store our sell sequences and targets
    for seq, target in sequential_data:  # iterate over the sequential data
        if target == False:  # if it's a "not buy"
            targets_false.append([seq, 0])  # append to sells list
        elif target == True:  # otherwise if the target is a 1...
            targets_true.append([seq, 1])  # it's a buy!

    random.shuffle(targets_false)  # shuffle the buys
    random.shuffle(targets_true)  # shuffle the sells!

    lower = min(len(targets_false), len(targets_true))  # what's the shorter length?

    targets_true = targets_true[:lower]  # make sure both lists are only up to the shortest length.
    targets_false = targets_false[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = targets_true+targets_false  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)
    print("Process sequential data done")
    return X, y  # return X and y...and make X a numpy array!

def process_sequential_test(sequential_data):
    targets_true = []  # list that will store our buy sequences and targets
    targets_false = []  # list that will store our sell sequences and targets
    for seq, target in sequential_data:  # iterate over the sequential data
        if target == False:  # if it's a "not buy"
            targets_false.append([seq, 0])  # append to sells list
        elif target == True:  # otherwise if the target is a 1...
            targets_true.append([seq, 1])  # it's a buy!

    random.shuffle(targets_false)  # shuffle the buys
    random.shuffle(targets_true)  # shuffle the sells!

    lower = min(len(targets_false), len(targets_true))  # what's the shorter length?

    targets_true = targets_true[:lower]  # make sure both lists are only up to the shortest length.
    targets_false = targets_false[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = targets_true+targets_false  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)
    print("Process sequential data done")
    return X, y  # return X and y...and make X a numpy array!
    

if __name__ == "__main__":
    # list_csv = ["data/TargetMax/GASBTC.csv", "data/TargetMax/NEOBTC.csv", "data/TargetMax/DATABTC.csv", "data/TargetMax/ADABTC.csv", "data/TargetMax/MATICBTC.csv"]
    list_csv = get_file_list("data/TargetMax/")
    random.shuffle(list_csv)
    target_pct = 1.1
    timestamp_test = 1560000000000
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(len(list_csv)):
        if len(train_x) > 40000: break
        data = pd.read_csv(list_csv[i])
        data.sort_values(by=['Open time'], inplace=True)
        if data.iloc[0]['Open time'] > timestamp_test-2*SEQ_LEN or data.iloc[-1]['Open time'] < timestamp_test+2*SEQ_LEN: continue
        data["Vol 24h"] = data["Quote asset volume"].rolling(SEQ_LEN).sum()
        data.drop(columns=["Volume","Quote asset volume","Number of trades",
                           "Taker buy quote asset volume",
                           "Close time","Taker buy base asset volume"], inplace=True)

        data = preprocess_data2(data, target_pct, timestamp_test)
        sequential_train, sequential_test = build_sequential(data, timestamp_test)
        train_x_tmp, train_y_tmp = process_sequential(sequential_train)
        test_x_tmp, test_y_tmp = process_sequential(sequential_test)

        train_x += train_x_tmp
        train_y += train_y_tmp
        test_x += test_x_tmp
        test_y += test_y_tmp

    train_x = np.asarray(train_x) 
    test_x = np.asarray(test_x)

    print('\nDATA READY')
    print(f"train data: {len(train_x)} validation: {len(test_x)}")
    print(f"TRAIN False +4%: {train_y.count(0)}, True +4%: {train_y.count(1)}")
    print(f"VALIDATION False +4%: {test_y.count(0)}, True +4%: {test_y.count(1)}")
    print('\n')

    model = Sequential()
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))
    
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones
    
    # Train model
    history = model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(test_x, test_y),
        callbacks=[tensorboard, checkpoint]
    )

    # Score model
    score = model.evaluate(test_x, test_y, verbose=0) # Change validation to test set
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Save model
    model.save("models/{}".format(NAME))
