import os
import json
import torch
import numpy as np
import pandas as pd
import typing
from utils import preprocess
from tabulate import tabulate

from utils.logger import Logger
from utils.device import init_device
from utils.preprocess import one_hot_encoding


logger = Logger(__file__)

MISSING = -1.0
ANOMALY = 1.0


class TAGANDataset:
    def __init__(self, config, device):
        logger.info("  Dataset: ")
        # Set device
        self.device = device

        # Set Config
        self.set_config(config)

        # Load Data
        x_data, y_data = self.load_data()

        # data = self.train
        self.time = self.store_times(self.data)
        # self.data = self.store_values(data, normalize=True)
        # self.label = self.store_values(label, normalize=False)

        self.in_dim = x_data[0].shape[2]
        self.data_len = len(self.data)
        self.shape = (self.batch_size, self.window_len, self.in_dim)

    def set_config(self, config):
        self.title = config["data"]
        self.workers = config["workers"]
        self.key = config["key"]
        self.weekday = config["weekday"]
        self.skip_weekend = config["skip_weekend"]

        self.stride = config["stride"]
        self.window_len = config["window_len"]
        self.future_len = config["future_len"]
        self.batch_size = config["batch_size"]
        self.hidden_dim = config["hidden_dim"]
        self.target_dim = config["target_dim"]

        self.train_option = config["train"]["opt"]
        self.split_rate = {
            "train": config["train"]["train_rate"],
            "valid": config["train"]["valid_rate"],
            "test": config["train"]["test_rate"],
        }

        self.data_path = os.path.join(config["path"], f"{self.title}.csv")
        # self.label_path = os.path.join(config["path"], f"{self.title}.json")

    def load_data(self) -> typing.Tuple:
        # Read csv data
        _path_checker(self.data_path, force=True)
        data = pd.read_csv(self.data_path)

        self.data_len = data.shape[0]
        self.columns = data.columns

        logger.info(f"  - File   : {self.data_path}")
        logger.info(f"  - Length : {self.data_len}")

        self.data = data
        x_data, y_data = self.preprocess(data)

        return x_data, y_data

    def preprocess(self, data: pd.DataFrame) -> typing.Tuple:
        # Indexing by Time key
        logger.info(f"  - Index  : {self.key}")
        data[self.key] = pd.to_datetime(data[self.key])

        # Weekday encoding
        if self.weekday in data.columns:
            data[self.weekday] = one_hot_encoding(data[self.weekday])
        data = data.set_index(self.key)

        # Normalize
        logger.info(f"  - Scaler : Min-Max")
        data = self.normalize(data)

        # X Y Split - Custom for dataset
        x_data = data.iloc[:, :]
        y_data = data.iloc[:, 3::2]

        # Windowing
        x_data = self.windowing(x_data)
        y_data = self.windowing(y_data)

        # When the train option is on, split train and valid data
        if self.train_option:
            split_rate = self.split_rate["valid"]
            logger.info(f"  - Split  : Train({1 - split_rate}), Valid({split_rate})")
            valid_idx = int(len(data) * split_rate)
            x_train = x_data[: -valid_idx - self.future_len]
            y_train = y_data[: -valid_idx - self.future_len]
            x_valid = x_data[-valid_idx:]
            y_valid = y_data[-valid_idx:]
            return (x_train, x_valid), (y_train, y_valid)

        return (x_data, None), (y_data, None)

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
        if data is None:
            return data

        if normalize is True:
            data = self.normalize(data)

        data = self.windowing(data)
        data = torch.from_numpy(data).float()
        return data

    def windowing(self, x):
        stop = len(x) - self.window_len
        output = [x[i : i + self.window_len] for i in range(0, stop, self.stride)]
        output = np.array(output)
        return output

    def normalize(self, data):
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        # 2 * (x - x.min) / (x.max - x.min) - 1
        self.max = data.iloc[:, 1:].max(0)
        self.min = data.iloc[:, 1:].min()

        data.iloc[:, 1:] = data.iloc[:, 1:] - self.min
        data.iloc[:, 1:] = data.iloc[:, 1:] / (self.max - self.min)
        data.iloc[:, 1:] = 2 * data.iloc[:, 1:] - 1

        self.max = torch.tensor(self.max)
        self.min = torch.tensor(self.min)

        # print("-----  Min Max information  -----")
        # df_minmax = pd.DataFrame({
        #     'MIN': self.min,
        #     'MAX': self.max
        # })
        # print(df_minmax.T)

        return data

    def denormalize(self, data):
        """Revert [-1,1] normalization"""
        if not hasattr(self, "max") or not hasattr(self, "min"):
            raise Exception("Try to denormalize, but the input was not normalized")

        for batch in range(data.shape[0]):
            data[batch, :, 1:] = 0.5 * data[batch, :, 1:] + 1
            data[batch, :, 1:] = data[batch, :, 1:] * (self.max - self.min)
            data[batch, :, 1:] = data[batch, :, 1:] + self.min

        return data

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.data[idx]


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
