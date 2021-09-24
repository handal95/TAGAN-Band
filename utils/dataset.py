import os
import json
import torch
import numpy as np
import pandas as pd
import typing
from tabulate import tabulate

from utils.logger import Logger
from utils.preprocess import one_hot, one_hot_encoding


logger = Logger(__file__)

MISSING = -1.0
ANOMALY = 1.0


class Dataset:
    def __init__(self, encoder, decoder):
        self.encoder = encoder[0]
        self.encoder_future = encoder[1]

        self.decoder = decoder[0]
        self.decoder_future = decoder[1]

        self.length = len(self.encoder)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "encoder": torch.tensor(self.encoder[idx], dtype=torch.float32),
            "decoder": torch.tensor(self.decoder[idx], dtype=torch.float32),
            "enc_future": torch.tensor(self.encoder_future[idx], dtype=torch.float32),
            "dec_future": torch.tensor(self.decoder_future[idx], dtype=torch.float32),
        }


class TAGANDataset:
    """
    TAGAN Dataset

    """

    def __init__(self, config: dict, device: torch.device) -> None:
        """
        params:
            config: Dataset configuration dict
            device: Torch device (cpu / cuda:0)
        """
        logger.info("  Dataset: ")

        # Set Config
        self.set_config(config)

        # Set device
        self.device = device

        # Load Data
        (x_train, x_valid), (y_train, y_valid) = self.load_data()

        logger.info(f"  - Train  : {x_train[0].shape}, {y_train[1].shape}")
        logger.info(f"  - Valid  : {x_valid[0].shape}, {y_valid[1].shape}")
        self.train_dataset = Dataset(x_train, y_train)
        self.valid_dataset = Dataset(x_valid, y_valid)

        # Feature info
        self.shape      = y_train[0].shape
        self.encode_dim = x_train[0].shape[2]
        self.decode_dim = y_train[0].shape[2]

    def set_config(self, config: dict) -> None:
        """
        Configure settings related to the data set.

        params:
            config: Dataset configuration dict
                `config['dataset']`
        """

        # Data file configuration
        self.path = config["path"]  # dirctory path 
        self.file = config["file"]  # csv format file
        self.workers = config["workers"]    # Thread workers

        # Columns
        self.key = config["key"]
        self.year = config["year"]
        self.month = config["month"]
        self.weekday = config["weekday"]
        self.targets = config["targets"]

        self.stride = config["stride"]
        self.window_len = config["window_len"]
        self.future_len = config["future_len"]
        self.batch_size = config["batch_size"]
        self.hidden_dim = config["hidden_dim"]

        self.train_option = config["train"]["opt"]
        self.split_rate = config["train"]["split_rate"]

        self.data_path = os.path.join(config["path"], f"{self.file}.csv")
        # self.label_path = os.path.join(config["path"], f"{self.title}.json")

    def load_data(self) -> typing.Tuple:
        # Read csv data
        data = pd.read_csv(self.data_path)

        self.length = data.shape[0]
        self.columns = data.columns

        logger.info(f"  - File   : {self.data_path}")
        logger.info(f"  - Length : {self.length}")

        self.data = data
        x_data, y_data = self.preprocess(data)

        return x_data, y_data

    def preprocess(self, data: pd.DataFrame) -> typing.Tuple:
        logger.info(f"  - Index  : {self.key}")
        logger.info(f"  - Target : ({len(self.targets)} items)\n{self.targets}")
        data[self.key] = pd.to_datetime(data[self.key])

        # Date information preprocess
        month_data = data[self.key].dt.month_name()
        if self.month in data.columns:
            month_data = data[self.month]
            data = data.drop(self.month, axis=1)

        weekday_data = data[self.key].dt.day_name()
        if self.weekday in data.columns:
            weekday_data = data[self.weekday]
            data = data.drop(self.weekday, axis=1)

        data.set_index(self.key)
        data = data.drop(self.key, axis=1)
        
        # Normalize
        logger.info(f"  - Scaler : Min-Max")
        data = self.normalize(data)

        month_encode = one_hot(month_data)
        weekday_encode = one_hot(weekday_data)
        data = pd.concat([data, month_encode], axis=1)
        data = pd.concat([data, weekday_encode], axis=1)
        
        # X Y Split - Custom for dataset
        x_data = data
        y_data = data[self.targets]
        
        # Windowing
        x_data, x_future = self.windowing(x_data)
        y_data, y_future = self.windowing(y_data)

        # When the train option is on, split train and valid data
        if self.train_option:
            split_idx = max(int(len(data) * self.split_rate), self.future_len + 1)
            logger.info(f"  - Split  : Train({self.split_rate:.2f}), Valid({1 - self.split_rate:.2f})")

            x_train = x_data[:split_idx - self.future_len]
            y_train = y_data[:split_idx - self.future_len]
            xf_train = x_future[:split_idx - self.future_len]
            yf_train = y_future[:split_idx - self.future_len]

            x_valid = x_data[split_idx:]
            y_valid = y_data[split_idx:]
            xf_valid = x_future[split_idx:]
            yf_valid = y_future[split_idx:]

            return (
                ((x_train, xf_train), (x_valid, xf_valid)),
                ((y_train, yf_train), (y_valid, yf_valid)),
            )

        return ((x_data, x_future), (None, None)), ((y_data, y_future), (None, None))

    def onehot_encoding(self, data: pd.DataFrame, value: pd.DataFrame, cat_col: str) -> pd.DataFrame:
        if cat_col in data.columns:
            cat_data = data[cat_col]
            data = data.drop(cat_col, axis=1)
            
        encoded = one_hot(cat_data)
        return data, encoded

    def windowing(self, x):
        stop = len(x) - self.window_len - self.future_len

        data = []
        target = []
        for i in range(0, stop, self.stride):
            j = i + self.window_len

            data.append(x[i : i + self.window_len])
            target.append(x[j : j + self.future_len])

        data = np.array(data)
        target = np.array(target)

        return data, target

    def normalize(self, data):
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        # 2 * (log(x + 1) - log(x + 1).min) / (log(x + 1).max - log(x + 1).min) - 1
        norm = data
        # norm = np.log10(norm + 1)

        self.max = norm.max(0)
        self.min = norm.min()

        norm = norm - self.min
        norm = norm / (self.max - self.min)
        norm = 2 * norm - 1
        
        data = norm
        self.decoder_min = torch.tensor(self.min[self.targets])
        self.decoder_max = torch.tensor(self.max[self.targets])
        self.min = torch.tensor(self.min)
        self.max = torch.tensor(self.max)
        
        print("-----  Min Max information  -----")
        df_minmax = pd.DataFrame({
            'min': self.decoder_min,
            'max': self.decoder_max,
        }, index=self.targets)
        print(df_minmax.T)
        # print(df_minmax2.T)

        return data

    # def denormalize(self, data):
    #     """Revert [-1,1] normalization"""
    #     if not hasattr(self, "max") or not hasattr(self, "min"):
    #         raise Exception("Try to denormalize, but the input was not normalized")

    #     delta = self.max - self.min
    #     for batch in range(data.shape[0]):
    #         batch_denorm = data[batch]

    #         batch_denorm = 0.5 * (batch_denorm + 1)
    #         batch_denorm = batch_denorm * delta
    #         batch_denorm = batch_denorm + self.min
    #         batch_denorm[:, :] = np.power(10, batch_denorm[:, :]) - 1

    #         data[batch] = batch_denorm

    #     return data

    def decoder_denormalize(self, data):
        """Revert [-1,1] normalization"""
        if not hasattr(self, "max") or not hasattr(self, "min"):
            raise Exception("Try to denormalize, but the input was not normalized")

        delta = self.decoder_max - self.decoder_min
        for batch in range(data.shape[0]):
            batch_denorm = data[batch]
            print(batch_denorm)

            batch_denorm = 0.5 * (batch_denorm + 1)
            print(batch_denorm)
            batch_denorm = batch_denorm * delta
            print(batch_denorm)
            batch_denorm = batch_denorm + self.decoder_min
            print(batch_denorm)
            batch_denorm = torch.pow(10, batch_denorm) - 1

            print(batch_denorm)
            data[batch] = batch_denorm
            input()

        return data

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.train_dataset[idx]
