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
    def __init__(self, encoder, decoder, answer):
        self.encoder = encoder[0]
        self.encoder_future = encoder[1]

        self.decoder = decoder[0]
        self.decoder_future = decoder[1]

        self.answer = answer[0]
        self.answer_future = answer[1]
        self.length = len(self.encoder)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "encoder": torch.tensor(self.encoder[idx], dtype=torch.float32),
            "decoder": torch.tensor(self.decoder[idx], dtype=torch.float32),
            "answer": torch.tensor(self.answer[idx], dtype=torch.float32),
            "enc_future": torch.tensor(self.encoder_future[idx], dtype=torch.float32),
            "dec_future": torch.tensor(self.decoder_future[idx], dtype=torch.float32),
            "ans_future": torch.tensor(self.answer_future[idx], dtype=torch.float32),
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
        (x_train, x_valid), (y_train, y_valid), (a_train, a_valid)= self.load_data()

        logger.info(f"  - Train  : {x_train[0].shape}, {y_train[1].shape}")
        logger.info(f"  - Valid  : {x_valid[0].shape}, {y_valid[1].shape}")
        self.train_dataset = Dataset(x_train, y_train, a_train)
        self.valid_dataset = Dataset(x_valid, y_valid, a_valid)

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
        x_data, y_data, a_data = self.preprocess(data)

        return x_data, y_data, a_data

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
            # weekday_data = data[self.weekday]
            data = data.drop(self.weekday, axis=1)

        data.set_index(self.key)
        data = data.drop(self.key, axis=1)

        # Normalize
        logger.info(f"  - Scaler : Min-Max")
        answer = data[self.targets].copy()
        
        data = self.imputate_zero_value(data)
        
        # print(data.iloc[:60])
        data.to_csv("./data/outputs/imputed.csv")
        data = self.normalize(data)

        month_encode = one_hot(month_data)
        weekday_encode = one_hot(weekday_data)
        data = pd.concat([data, month_encode], axis=1)
        data = pd.concat([data, weekday_encode], axis=1)
        data.to_csv("./data/outputs/processed.csv")
        
        # X Y Split - Custom for dataset
        x_data = data
        y_data = data[self.targets].copy()
        
        # Windowing
        a_data, a_future = self.windowing(answer)
        x_data, x_future = self.windowing(x_data)
        y_data, y_future = self.windowing(y_data)
        
        # When the train option is on, split train and valid data
        if self.train_option:
            split_idx = max(int(len(data) * self.split_rate), self.future_len + 1)
            logger.info(f"  - Split  : Train({self.split_rate:.2f}), Valid({1 - self.split_rate:.2f})")

            x_train = x_data[:split_idx - self.future_len]
            y_train = y_data[:split_idx - self.future_len]
            a_train = a_data[:split_idx - self.future_len]
            xf_train = x_future[:split_idx - self.future_len]
            yf_train = y_future[:split_idx - self.future_len]
            af_train = a_future[:split_idx - self.future_len]

            x_valid = x_data[split_idx:]
            y_valid = y_data[split_idx:]
            a_valid = a_data[split_idx:]

            xf_valid = x_future[split_idx:]
            yf_valid = y_future[split_idx:]
            af_valid = a_future[split_idx:]

            return (
                ((x_train, xf_train), (x_valid, xf_valid)),
                ((y_train, yf_train), (y_valid, yf_valid)),
                ((a_train, af_train), (a_valid, af_valid)),
            )

        return (
            ((x_data, x_future), (None, None)), 
            ((y_data, y_future), (None, None)),
            ((a_data, a_future), (None, None))
        )

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
        # data = np.log10(data + 1)

        self.origin_max = data.max(0)
        self.origin_min = data.min()

        cutting_data = data # .copy()
        for col in data.columns:
            unique_values = sorted(cutting_data[col].unique())
            percent = 0.02
            cut_idx = int(len(unique_values) * percent) 
            front_two = unique_values[:cut_idx]
            back_two = unique_values[-cut_idx:]
            for i, value in enumerate(cutting_data[col]):
                if value in front_two:
                    cutting_data[col][i] = unique_values[cut_idx - 1]
                elif value in back_two:
                    cutting_data[col][i] = unique_values[-cut_idx]
                    
        self.max = cutting_data.max(0) * 1.01
        self.min = cutting_data.min(0) * 0.99

        data = data - self.min
        data = data / (self.max - self.min)
        data = 2 * data - 1
        
        self.decoder_max = torch.tensor(self.max[self.targets])
        self.decoder_min = torch.tensor(self.min[self.targets])
        
        self.min = torch.tensor(self.min)
        self.max = torch.tensor(self.max)
        
        print("-----  Min Max information  -----")
        df_minmax = pd.DataFrame({
            'MIN': self.origin_min,
            f'min ({(percent) * 100})': self.decoder_min,
            f'max ({(1 - percent) * 100})': self.decoder_max,
            'MAX': self.origin_max,
        }, index=self.targets)
        print(df_minmax.T)

        return data

    def denormalize(self, data):
        """Revert [-1,1] normalization"""
        if not hasattr(self, "max") or not hasattr(self, "min"):
            raise Exception("Try to denormalize, but the input was not normalized")

        delta = self.decoder_max - self.decoder_min
        for batch in range(data.shape[0]):
            batch_denorm = data[batch]

            batch_denorm = 0.5 * (batch_denorm + 1)
            batch_denorm = batch_denorm * delta
            batch_denorm = batch_denorm + self.decoder_min
            # batch_denorm = torch.pow(10, batch_denorm) - 1

            data[batch] = batch_denorm

        return data

    def imputate_zero_value(self, data):
        for col in data:
            for row in range(len(data[col])):
                if data[col][row] <= 0:
                    yesterday = data[col][max(0, row - 1)]
                    last_week = data[col][max(0, row - 7)]
                    last_year = data[col][max(0, row - 365)]
                    candidates = [yesterday, last_week, last_year]
                    try:
                        while 0 in candidates:
                            candidates.remove(0)
                    except ValueError:
                        pass
                    if len(candidates) == 0:
                        mean_value = 0
                    else:
                        mean_value = np.mean(candidates)
                    data[col][row] = mean_value

        return data


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.train_dataset[idx]
