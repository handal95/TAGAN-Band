import torch
import numpy as np


def metric_NMAE(pred, true):
    pred = pred[:, [6, 13, 27]]
    true = true[:, [6, 13, 27]]

    target = torch.where(true != 0)
    true = true[target]
    pred = pred[target]

    if len(true) == 0:
        return torch.tensor(0.0)

    score = torch.mean(torch.abs((true - pred)) / (true))
    return score
