import torch
import numpy as np

def metric_NMAE(pred, true):
    true = true[:len(pred)-16]
    pred = torch.from_numpy(pred[16:])
    true = torch.from_numpy(true)

    target = torch.where(true!=0)
    true = true[target]
    pred = pred[target]
    score = torch.mean(torch.abs((true-pred))/(true))

    return score
