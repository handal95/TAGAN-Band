import torch
import numpy as np
import pandas as pd
from utils.logger import Logger
from utils.device import init_device

logger = Logger(__file__)
MISSING = -1.0
ANOMALY = 1.0
WARNING = 2.0
OUTLIER = 3.0

class TAGAN_Bander:
    def __init__(self, dataset, config):
        logger.info("Bander Setting")

        self.device = init_device()
        self.config = self.set_config(config)
        self.dataset = dataset

        self.label = None
        self.label_info = []
        self.data = None
        self.pred = None
        self.bands = {
            "init": False,
            "median": None,
            "upper": [None, None, None],
            "lower": [None, None, None],
        }
        self.detects = {"analized": [], "labeled": []}
        self.replace = None

    def set_config(self, config):
        self.pivot = config["pivot"]
        self.gp_weight = config["gp_weight"]
        self.l1_gamma = config["l1_gamma"]
        self.l2_gamma = config["l2_gamma"]
        
        sigma = config["sigmas"]
        self.sigmas = [sigma["inner"], sigma["normal"], sigma["warning"]]

    def process(self, x, y, label, normalized=True):
        if normalized:
            x = self._denormalize(x)[0].numpy().ravel()
            y = self._denormalize(y)[0].numpy().ravel()
            label = label[0].detach().numpy().ravel()

        if self.bands["init"] is False:
            self.bands["init"] = True
            self.bands["median"] = x[: self.pivot]
            for i in range(3):
                self.bands["upper"][i] = x[: self.pivot]
                self.bands["lower"][i] = x[: self.pivot]

        std = y.std()
        median = np.median(y[self.pivot + 1 :])

        self.bands["median"] = np.append(self.bands["median"], median)
        for i in range(3):
            self.bands["upper"][i] = np.append(
                self.bands["upper"][i], median + self.sigmas[i] * std
            )
            self.bands["lower"][i] = np.append(
                self.bands["lower"][i], median - self.sigmas[i] * std
            )

        self.data = self.data_concat(self.data, x)
        self.pred = self.pred_concat(self.pred, y)
        self.label = self.pred_concat(self.label, label)
        self.detecting(self.data[-1], label)

        return self.data, self.pred, label, self.bands, self.detects

    def detecting(self, ypos, label):
        # Analized Unlabeled points
        pivot = self.pivot
        xpos = len(self.data) - 1

        if True in np.isin(label[pivot], [MISSING]):
            self.detects["labeled"].append((xpos, "black"))

        elif True in np.isin(label[pivot], [ANOMALY]):
            self.detects["labeled"].append((xpos, "red"))
            
        for i in range(1, 3):
            color = "red" if i == 2 else "black"
            LABEL = OUTLIER if i == 2 else WARNING
            if ypos < self.bands["lower"][i][-1] or ypos > self.bands["upper"][i][-1]:
                self.detects["analized"].append((xpos, ypos, color))
                self.label[-1] = LABEL

    def _denormalize(self, x):
        if self.device != torch.device("cpu"):
            x = x.cpu().detach()
        return self.dataset.denormalize(x)

    def get_sample(self, x, netG):
        shape = (x.size(0), x.size(1), x.size(2))

        z = torch.randn(shape).to(self.device)
        if self.pivot > 0:
            z[:, : self.pivot, :] = x[:, : self.pivot, :]

        y = netG(z).to(self.device)
        return y

    def impute_missing_value(self, x, y, label, pivot):
        # TODO : Need to refactoring
        if self.check_missing(label):
            if self.replace is None:
                self.replace = x
            else:
                self.replace[:, :-1] = self.replace[:, 1:].clone()

            if self.check_missing(label, latest=True):
                y_ = y.cpu().detach().numpy()
                m = np.median(y_[:, :, :], axis=1)
                std = y_[:, pivot:, :].std()
                y = np.random.normal(m, std / 2, y.shape)
                self.replace[:, -1:] = torch.Tensor(y[:, -1:, :])
            else:
                self.replace[:, -1:] = x[:, -1:, :]

            x[0] = self.replace.to(self.device)
        else:
            if self.replace is not None:
                self.replace = None

        return x

    def check_missing(self, label, latest=False):
        label_ = label[:, -1] if latest is True else label
        return True in np.isin(label_, [MISSING])

    def data_concat(self, target, x):
        if target is None:
            target = x[: self.pivot]

        return np.concatenate((target[: 1 - self.pivot], x[: self.pivot]))

    def pred_concat(self, target, y):
        if target is None:
            target = y

        length = len(target) - self.dataset.seq_len + self.pivot
        return np.concatenate((target[: length + 1], y[self.pivot :]))
