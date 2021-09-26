import os
import torch
import typing
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from utils.logger import Logger
from utils.preprocess import one_hot


logger = Logger(__file__)


class Dataset:
    def __init__(self, dataset, device):
        self.encodes = dataset["encode"]
        self.decodes = dataset["decode"]

        self.windows = dataset["window"]
        self.futures = dataset["future"]

        self.length = len(self.encodes)
        self.device = device

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = {
            "encode": torch.tensor(self.encodes[idx], dtype=torch.float32),
            "decode": torch.tensor(self.decodes[idx], dtype=torch.float32),

            "window": torch.tensor(self.windows[idx], dtype=torch.float32),
            "future": torch.tensor(self.futures[idx], dtype=torch.float32)
        }

        return data


class TAGANDataset:
    """
    TAGAN Dataset

    """

    def __init__(self, config: dict, device: torch.device) -> None:
        """
        TAGAN Dataset

        Args:
            config: Dataset configuration dict
            device: Torch device (cpu / cuda:0)
        """
        logger.info("  Dataset: ")

        # Set Config
        self.set_config(config)

        # Set device
        self.device = device

        # Load Data
        train_set, valid_set, preds_set = self.load_data()
        
        # Feature info
        self.shape = train_set['decode'].shape
        self.encode_dims = train_set['encode'].shape[2]
        self.decode_dims = train_set['decode'].shape[2]
        self.dims = {
            "encode": self.encode_dims, 
            "decode": self.decode_dims
        }
        
        # Dataset
        self.train_dataset = Dataset(train_set, self.device)
        self.valid_dataset = Dataset(valid_set, self.device)
        self.preds_dataset = Dataset(preds_set, self.device)

    def set_config(self, config: dict) -> None:
        """
        Configure settings related to the data set.

        params:
            config: Dataset configuration dict
                `config['dataset']`
        """

        # Data file configuration
        self.directory = config["directory"]  # dirctory path
        self.data_name = config["data_name"]  # csv format file
        self.data_path = os.path.join(self.directory, self.data_name)

        # Columns
        self.index_col = config["index_col"]
        self.year = config["year"]
        self.month = config["month"]
        self.weekday = config["weekday"]
        self.targets = config["targets"]

        self.stride = config["stride"]
        self.window_len = config["window_len"]
        self.future_len = config["future_len"]

        self.split_rate = config["split_rate"]
        
        # Preprocess
        self.log = config["log"]
        self.cutoff_min = config["cutoff"]["min"]
        self.cutoff_max = config["cutoff"]["max"]

    def load_data(self) -> typing.Tuple:
        # Read csv data
        data = pd.read_csv(f"{self.data_path}.csv")

        self.length = data.shape[0]
        self.columns = data.columns

        logger.info(f"  - File   : {self.data_path}")
        logger.info(f"  - Length : {self.length}")
        logger.info(f"  - Index  : {self.index_col}")
        logger.info(f"  - Target : ({len(self.targets)} items)\n{self.targets}")
        data[self.index_col] = pd.to_datetime(data[self.index_col])

        # Date information preprocess
        month_data = data[self.index_col].dt.month_name()
        if self.month in data.columns:
            month_data = data[self.month]
            data = data.drop(self.month, axis=1)

        weekday_data = data[self.index_col].dt.day_name()
        if self.weekday in data.columns:
            # weekday_data = data[self.weekday]
            data = data.drop(self.weekday, axis=1)

        data.set_index(self.index_col)
        self.times = data[self.index_col]
        
        data = data.drop(self.index_col, axis=1)
        # data = data.drop("팽이버섯_wind", axis=1)

        # KEEP ANSWER DATA
        real_data = data[self.targets].copy()

        # Zero Value Imputating
        logger.info(f"  - Imputer : mean(-1d, -1w, -1y)")
        data = self.imputate_zero_value(data)
        data.to_csv("./data/outputs/imputed.csv")

        # Normalize
        logger.info(f"  - Scaler : Min-Max")
        data = self.normalize(data)
        month_encode = one_hot(month_data)
        weekday_encode = one_hot(weekday_data)
        data = pd.concat([data, month_encode], axis=1)
        data = pd.concat([data, weekday_encode], axis=1)
        data.to_csv("./data/outputs/processed.csv")

        # Windowing
        encode_data = data
        decode_data = data[self.targets].copy()

        # Windowing
        encode, _ = self.windowing(encode_data)
        _, decode = self.windowing(decode_data)
        window, future = self.windowing(real_data)
        
        # When the train option is on, split train and valid data
        split_idx = max(int(len(data) * self.split_rate), self.future_len + 1)
        logger.info(
            f"  - Split  : Train({self.split_rate:.2f}), Valid({1 - self.split_rate:.2f}) / Real({-2 * self.future_len})"
        )

        # Train
        train_set = {
            "encode": encode[: split_idx - self.future_len],
            "decode": decode[: split_idx - self.future_len],
            "window": window[: split_idx - self.future_len],
            "future": future[: split_idx - self.future_len]
        }
        self.train_times = self.get_time_arange(self.times[:split_idx - 2 * self.future_len])

        # Train
        valid_set = {
            "encode": encode[split_idx: -1],
            "decode": decode[split_idx: -1],
            "window": window[split_idx: -1],
            "future": future[split_idx: -1]
        }
        self.valid_times = self.get_time_arange(self.times[split_idx: - 1])
        
        # Preds
        preds_set = {
            "encode": encode[-2 * self.future_len:],
            "decode": decode[-2 * self.future_len:],
            "window": window[-2 * self.future_len:],
            "future": future[-2 * self.future_len:],
        }
        self.preds_times = self.get_time_arange(self.times[-2 * self.future_len:])

        return train_set, valid_set, preds_set

    def get_time_arange(self, times):
        first_day = times.values[0] + np.timedelta64(1, 'D')
        last_day = times.values[-1] + np.timedelta64(1 + self.future_len, 'D')
        time_arange = np.arange(first_day, last_day, dtype='datetime64[D]')
        return time_arange


    def onehot_encoding(
        self, data: pd.DataFrame, value: pd.DataFrame, cat_col: str
    ) -> pd.DataFrame:
        if cat_col in data.columns:
            cat_data = data[cat_col]
            data = data.drop(cat_col, axis=1)

        encoded = one_hot(cat_data)
        return data, encoded

    def windowing(self, x):
        stop = len(x) - self.window_len - self.future_len

        window = []
        future = []
        for i in range(0, stop, self.stride):
            j = i + self.window_len

            window.append(x[i : i + self.window_len])
            future.append(x[j : j + self.future_len])

        window = np.array(window)
        future = np.array(future)

        return window, future

    def normalize(self, data):
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        # 2 * (x - x.min) / (x.max - x.min) - 1

        origin_max = data.max(0)
        origin_min = data.min()
        min_p = self.cutoff_min
        max_p = self.cutoff_max
        logger.info(f"Cutting : Min {min_p*100} % , Max {max_p*100} %")

        for col in data.columns:
            uniques = sorted(data[col].unique())
            pivot_min = uniques[1 + int(len(uniques) * min_p)]
            pivot_max = uniques[-1 - int(len(uniques) * max_p)]

            data[col][data[col] < pivot_min] = pivot_min
            data[col][data[col] > pivot_max] = pivot_max

        if self.log:
            data = np.log10(data + 1)

        self.min = data.min()
        self.max = data.max(0)

        data = data - self.min
        data = data / (self.max - self.min)
        data = 2 * data - 1

        self.decode_min = torch.tensor(self.min[self.targets])
        self.decode_max = torch.tensor(self.max[self.targets])

        self.min = torch.tensor(self.min)
        self.max = torch.tensor(self.max)

        logger.info(f"- Log   : {self.log}")
        logger.info("-----  Min Max information  -----")
        df_minmax = pd.DataFrame(
            {
                f"MIN": origin_min,
                f"min ({min_p * 100})": self.decode_min,
                f"max ({max_p * 100})": self.decode_max,
                f"MAX": origin_max,
            },
            index=self.targets,
        )
        logger.info(df_minmax.T)

        return data

    def denormalize(self, data):
        """Revert [-1,1] normalization"""
        if not hasattr(self, "max") or not hasattr(self, "min"):
            raise Exception("Try to denormalize, but the input was not normalized")

        delta = self.decode_max - self.decode_min
        for batch in range(data.shape[0]):
            batch_denorm = data[batch]

            batch_denorm = 0.5 * (batch_denorm + 1)
            batch_denorm = batch_denorm * delta
            batch_denorm = batch_denorm + self.decode_min
            
            if self.log:
                batch_denorm = torch.pow(10, batch_denorm) - 1

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

    def get_random(self):
        idx = np.random.randint(self.shape)
        data = self.train_dataset[idx]
        
        encode = data["encode"].to(self.device)
        decode = data["decode"].to(self.device)
        
        return encode, decode

    def loader(self, batch_size: int, n_workers: int, train: bool = False, preds: bool=False):
        dataset = self.train_dataset if train else self.valid_dataset
        dataset = self.preds_dataset if preds else dataset
        
        dataloader = DataLoader(dataset, batch_size, num_workers=n_workers)
        return dataloader

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.train_dataset[idx]
