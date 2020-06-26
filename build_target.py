from os import listdir
from os.path import isfile, join, splitext, basename
import multiprocessing
import time
import pandas as pd

SEQ_LEN = 288 # Number of indexes for 24 hours

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

def to_supervised(data):
    """ We want to predict if the price is going to increase by at least 20% 
        in the next 10 hours by analyzing the past 24 hours. Klines of 5 min.
    """
    # expectation = 1.04
    n_futur_idx = 12 # Number of index for 1 hour
    # target = [data["Close"].iloc[i:i+n_futur_idx].max() >= expectation*data["Close"].iloc[i] for i in range(0, data.shape[0]-n_futur_idx)] 
    target = [data["Close"].iloc[i+1:i+n_futur_idx+1].max() for i in range(0, data.shape[0]-(n_futur_idx+1))]
    return target

def build_target_data(CSV_file):
    data = pd.read_csv(CSV_file)
    target = to_supervised(data)
    data = data.iloc[0:len(target)]
    data["Max next hour"] = target
    data.to_csv(f"data/TargetMax/{basename(CSV_file)}", index=False)
    print(f"Target {basename(CSV_file)} is build")

if __name__ == "__main__":
    CSV_files = get_file_list("data/RAW/", extension=None)

    start = time.time()
    pool = multiprocessing.Pool()
    pool.map(build_target_data, CSV_files)
    pool.close() 
    pool.join()
    stop = time.time()
    print(f"Execution time: {stop-start}s")
