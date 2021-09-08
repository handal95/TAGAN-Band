import os
import json
import torch
import numpy as np
import pandas as pd

from utils.logger import Logger
from utils.device import init_device


logger = Logger(__file__)

MISSING = -1.0
ANOMALY = 1.0


class TimeseriesDataset:
    def __init__(self, config):
        # Load Data
        logger.info("Dataset Setting")
        self.device = init_device()
        self.set_config(config)

        # Dataset
        data, label = self.load_data()
        self.index = data.index
        self.origin = data.value

        (self.train, self.test) = self.split_data(data)
        data = self.train
        self.time = self.store_times(data)
        self.data = self.store_values(data, normalize=True)
        self.label = self.store_values(label, normalize=False)

        self.in_dim = len(data.columns)
        self.data_len = len(self.data)
        self.shape = (self.batch_size, self.seq_len, self.in_dim)

        self.replace = None

    def set_config(self, config):
        self.title = config["data"]
        self.workers = config["workers"]
        self.key = config["key"]

        self.stride = config["stride"]
        self.seq_len = config["seq_len"]
        self.batch_size = config["batch_size"]
        self.hidden_dim = config["hidden_dim"]
        
        self.train_option = config["train"]["opt"]
        self.split_rate = {
            "train": config["train"]["train_rate"],
            "valid": config["train"]["valid_rate"],
            "test": config["train"]["test_rate"], 
        }

        self.data_path = os.path.join(config["path"], f"{self.title}.csv")
        self.label_path = os.path.join(config["path"], f"{self.title}.json")

    def load_data(self):
        _path_checker(self.data_path, force=True)

        data = pd.read_csv(self.data_path)
        label = _json_load(self.label_path)["anomalies"]

        # Labeling missing values (and known anomalies)
        data, label = self.labeling(data, label)
        data = data.set_index(self.key)
        data = data.interpolate(method="time")

        return data, label

    def split_data(self, data):
        if self.train_option is False:
            return (data, None, None)
        
        logger.info(
            f"Data Split >> "
            f"(Train: {self.split_rate['train']}, "
            f"Valid: {self.split_rate['valid']})")
        
        self.train_idx = int(len(data) * (self.split_rate["train"]))
        self.valid_idx = self.train_idx + int(len(data) * (self.split_rate["valid"]))
        
        train_set = data[:self.valid_idx]
        test_set = data[self.valid_idx:]
        
        return (train_set, test_set)


    def labeling(self, data, anomalies):
        data, missings = self.check_missing_value(data)

        label = np.zeros(len(data))

        def _labeling(TAG, json_data):
            for time_span in json_data:
                start_time = pd.to_datetime(time_span[0])
                end_time = pd.to_datetime(time_span[1])

                for i in data.index:
                    cur_time = data[self.key][i]
                    if start_time <= cur_time and cur_time <= end_time:
                        label[i] = TAG
            return label

        label = _labeling(ANOMALY, anomalies)
        label = _labeling(MISSING, missings)

        label = pd.DataFrame({self.key: data[self.key], "value": label})
        label = label.set_index(self.key)

        return data, label

    def check_missing_value(self, data):
        # TODO : Need Refactoring
        def timestamp(index=0):
            return data[self.key][index]

        data[self.key] = pd.to_datetime(data[self.key])
        TIMEGAP = timestamp(1) - timestamp(0)

        missings = list()
        filled_count = 0
        for i in range(1, len(data)):
            if timestamp(i) - timestamp(i - 1) != TIMEGAP:
                start_time = timestamp(i - 1) + TIMEGAP
                end_time = timestamp(i) - TIMEGAP

                missings.append([str(start_time), str(end_time)])

                # Fill time gap
                cur_time = start_time
                while cur_time <= end_time:
                    filled_count += 1
                    data = data.append({self.key: cur_time}, ignore_index=True)
                    cur_time = cur_time + TIMEGAP

        # Resorting by timestamp
        logger.info(f"Checking Timegap - ({TIMEGAP}), Filled : {filled_count}")
        data = data.set_index(self.key).sort_index().reset_index()

        return data, missings

    def store_times(self, data):
        time = pd.to_datetime(data.index)
        time = time.strftime("%y%m%d:%H%M")
        time = time.values
        return time

    def store_values(self, data, normalize=False):
        data = self.windowing(data[["value"]])
        if normalize is True:
            data = self.normalize(data)
        data = torch.from_numpy(data).float()
        return data

    def windowing(self, x):
        stop = len(x) - self.seq_len
        return np.array([x[i : i + self.seq_len] for i in range(0, stop, self.stride)])

    def normalize(self, x):
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        self.max = x.max()
        self.min = x.min()
        
        output = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        return output

    def denormalize(self, x):
        """Revert [-1,1] normalization"""
        if not hasattr(self, "max") or not hasattr(self, "min"):
            raise Exception("Try to denormalize, but the input was not normalized")

        output = 0.5 * (x * self.max - x * self.min + self.max + self.min)
        return output

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    


def _path_checker(path, force=False):
    if os.path.exists(path):
        return True
    elif force:
        logger.warn(f"{path} is not founed")
        raise FileNotFoundError
    return False


def _json_load(path):
    with open(path) as f:
        data = json.load(f)
    return data
