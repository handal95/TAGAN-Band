import torch


def metric_NMAE(pred, true):
    pred = pred[:, [6, 13, 27]]
    true = true[:, [6, 13, 27]]
    target = torch.where(true != 0)
    true = true[target]
    pred = pred[target]
    score = torch.mean(torch.abs((true - pred)) / (true))

    return score
