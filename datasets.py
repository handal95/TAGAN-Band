import os
from pickle import TRUE
from numpy.core.numeric import NaN
import yaml
import json
import torch
import numpy as np
import pandas as pd
from logger import Logger

logger = Logger(__file__)

MISSING = -1.0
ANOMALY = 1.0

class TimeseriesDataset:
    def __init__(self, config, device, temp=False):
        # Load Data
        logger.info("Loading Dataset")
        self.device = device
        self.init_config(config, temp)
        self.temp = temp

        # Dataset
        data, label = self.load_data()

        self.n_feature = len(data.columns)
        self.time = self.store_times(data)
        self.data = self.store_values(data, normalize=True)
        self.label = self.store_values(label, normalize=False)
        
        self.data_len = len(self.data)
        
        self.replace = None
        
    def init_config(self, config, temp):
        self.title = config["data"]
        self.label = config["anomaly"]
        self.workers = config["workers"]
        self.index = config["index"]

        self.stride = config["stride"]
        self.seq_len = config["seq_len"]
        self.shuffle = config["shuffle"]
        self.batch_size = config["batch_size"]
        self.hidden_dim = config["hidden_dim"]
        
        if temp is True:
            self.data_path = os.path.join(config["path"], "ambient_temperature_system_failure_origin.csv")
        else:
            self.data_path = os.path.join(config["path"], config["data"])
        self.anomaly_path = os.path.join(config["path"], config["anomaly"])
        
        
    def load_data(self):
        data = pd.read_csv(self.data_path)
        data, label = self.fill_timegap(data)

        label = self.load_anomaly(data)
        data = data.set_index(self.index)

        data = data.interpolate(method='time')

        return data, label
    
    def fill_timegap(self, data):
        # TODO : Need Refactoring 
        data[self.index] = pd.to_datetime(data[self.index])
        timegap = data[self.index][1] - data[self.index][0]
        label = pd.DataFrame()

        length = len(data)
        if os.path.exists(self.anomaly_path) is True:
            with open(self.anomaly_path, mode="r") as f:
                json_data = json.load(f)
            json_data["missings"] = list()
        else:
            json_data = {
                "missings": list(),
                "anomalies": list(),
            }

        for i in range(1, len(data)):
            if data[self.index][i] - data[self.index][i - 1] != timegap:
                start_time = data[self.index][i - 1] + timegap
                end_time = data[self.index][i] - timegap

                json_data["missings"].append([str(start_time), str(end_time)])

                # Fill time gap
                for _ in range(1 + (end_time - start_time) // timegap):
                    time = start_time + _ * timegap
                    data = data.append({self.index: time}, ignore_index=True)

        data = data.set_index(self.index).sort_index().reset_index()

        if self.temp is False:
            with open(self.anomaly_path, mode="w") as f:
                json.dump(json_data, f)

        filled_length = len(data) - length
        logger.info(f"Filling Time Gap : Filled records : {filled_length} : timegap : {timegap}")
                
        return data, label

    def load_anomaly(self, data):
        with open(self.anomaly_path, mode='r') as f:
            json_data = json.load(f)
            
        label = np.zeros(len(data))
        for ano_span in json_data["anomalies"]:
            ano_start = pd.to_datetime(ano_span[0])
            ano_end = pd.to_datetime(ano_span[1])
            for idx in data.index:
                if data.loc[idx, self.index] >= ano_start and data.loc[idx, self.index] <= ano_end:
                    label[idx] = ANOMALY

        for miss_span in json_data["missings"]:
            nan_start = pd.to_datetime(miss_span[0])
            nan_end = pd.to_datetime(miss_span[1])
            for idx in data.index:
                if data.loc[idx, self.index] >= nan_start and data.loc[idx, self.index] <= nan_end:
                    label[idx] = MISSING

        label = pd.DataFrame({self.index : data[self.index], "value": label})
        label = label.set_index(self.index)
        label.to_csv("labels.csv")
        
        return label

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

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

        return 2 * (x - x.min()) / (x.max() - x.min()) - 1

    def denormalize(self, x):
        """Revert [-1,1] normalization"""
        if not hasattr(self, "max") or not hasattr(self, "min"):
            raise Exception(
                "You are calling denormalize, but the input was not normalized"
            )
            
        output = 0.5 * (x * self.max - x * self.min + self.max + self.min)
        
        return output

    def get_samples(self, netG, shape, cond):
        idx = np.random.randint(self.data.shape[0], size=shape[0])
        x = self.data[idx].to(self.device)
        z = torch.randn(shape).to(self.device)
        if cond > 0:
            z[:, :cond, :] = x[:, :cond, :]

        y = netG(z).to(self.device)

        return y, x
    
    def get_sample(self, x, netG, shape, cond):
        z = torch.randn(shape).to(self.device)
        if cond > 0:
            z[:, :cond, :] = x[:, :cond, :]
        
        y = netG(z).to(self.device)
        
        return y
    
    def fill_missing_value(self, x, y, label, idx, cond):
        
        def isin(label, latest=False):
            label_ = label[:, -1] if latest is True else label
            return (True in np.isin(label_, [MISSING]) )#or True in np.isin(label_, [ANOMALY]))
                    
        if isin(label):
            print(
                f"{self.time[idx]}: "
                f"{self.denormalize(x.cpu().detach().numpy().ravel())}  " 
                f"{label.cpu().detach().numpy().ravel()}")

            if self.replace is None:
                self.replace = x
            else:
                self.replace[:, :-1] = self.replace[:, 1:].clone()

            if isin(label, latest=True):
                y = y.cpu().detach().numpy()
                n = (np.median(y[:, cond:-1], axis=0) + y[:, -1]) / 2
                self.replace[:, -1:] = torch.Tensor(n[-1])
                print(
                    f"{self.time[idx]}: "
                    f"{self.denormalize(y.ravel())}  " 
                    f"{self.denormalize(n.ravel())}  " 
                )
            else:
                self.replace[:, -1:] = x[:, -1:, :]

            x[0] = self.replace.to(self.device)

        else:
            # # TODO: Temp Flag
            if self.replace is not None:
                self.replace = None


        return x
