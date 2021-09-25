import torch
import torch.nn as nn


class TAGAN_Metric:
    def __init__(self, config, device):
        self.set_config(config)
        self.device = device
        self.criterion_l1n = nn.SmoothL1Loss().to(self.device)
        self.criterion_l2n = nn.MSELoss().to(self.device)
        self.criterion_adv = GANLoss(real_label=0.9, fake_label=0.1).to(self.device)

    def set_config(self, config):
        self.window_len = config["window_len"]
        self.gp_weight = config["gp_weight"]
        self.l1_gamma = config["l1_gamma"]
        self.l2_gamma = config["l2_gamma"]
        self.zero_ignoring = config["zero_ignoring"]

    def GANloss(self, D, target_is_real):
        return self.criterion_adv(D, target_is_real)

    def NMAE(self, pred, true, real_test=False):
        if real_test:
            pred = pred[:, [6, 13, 27]]
            true = true[:, [6, 13, 27]]
            pred, true = self._ignore_zero(pred, true)
            return torch.mean(torch.abs((true - pred)) / (true))

        pred, true = self._ignore_zero(pred, true)
        return torch.mean((true - pred) / (true))

    def l1loss(self, pred, true):
        pred, true = self._ignore_zero(pred, true)
        return self.l1_gamma * self.criterion_l1n(pred, true)

    def l2loss(self, pred, true):
        pred, true = self._ignore_zero(pred, true)
        return self.l2_gamma * self.criterion_l2n(pred, true)

    def grad_penalty(self, pred, true):
        pred, true = self._ignore_zero(pred, true)
        return self.gp_weight * self._grad_penalty(pred, true)

    def _ignore_zero(self, pred, true):
        if self.zero_ignoring:
            target = torch.where(true != 0)
            true = true[target]
            pred = pred[target]
        return pred, true

    def _grad_penalty(self, pred, true):
        gradients = pred - true
        gradients_sqr = torch.square(gradients)
        gradients_sqr_sum = torch.sum(gradients_sqr)
        gradients_l2_norm = torch.sqrt(gradients_sqr_sum)
        gradients_penalty = torch.square(1 - gradients_l2_norm) / true.size(0)
        return gradients_penalty


class GANLoss(nn.Module):
    def __init__(self, real_label=0.9, fake_label=0.1):
        super(GANLoss, self).__init__()
        self.register_buffer("none_label", torch.tensor(0.5))
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        target_tensor = target_tensor.to(input.device)
        return self.loss(input, target_tensor)
