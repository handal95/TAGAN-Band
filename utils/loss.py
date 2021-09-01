import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self, real_label=0.9, fake_label=0.1):
        super(GANLoss, self).__init__()
        self.register_buffer("none_label", torch.tensor(0.5))
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real, imputate):
        if imputate is False:
            if target_is_real:
                target_tensor = self.real_label
            else:
                target_tensor = self.fake_label
        else:
            target_tensor = self.none_label

        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real, imputate=False):
        target_tensor = self.get_target_tensor(input, target_is_real, imputate).to(
            input.device
        )
        return self.loss(input, target_tensor)
