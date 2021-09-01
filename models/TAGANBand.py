import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from models.bander import TAGAN_Bander
from models.LSTMGAN import LSTMGenerator, LSTMDiscriminator

from utils.args import load_yaml
from utils.loss import GANLoss
from utils.logger import Logger
from utils.device import init_device
from utils.dashboard import Dashboard
from utils.dataset import TimeseriesDataset

logger = Logger(__file__)

MISSING = -1.0
ANOMALY = 1.0


class TAGANBand:
    """
    TAGANBand : Timeseries Analysis using GAN Band

    The Model for Detecting anomalies / Imputating missing value in timeseries data

    """

    def __init__(self, config=None):
        # Set device
        self.device = init_device()
        logger.info(f"Device setting - {self.device}")

        # Config option
        config = self.set_config(config=config)

        # Dataset option
        self.dataset = TimeseriesDataset(config=self.dataset_config)
        self.dataloader = self.init_dataloader(self.dataset)

        # Model option
        self.bander = TAGAN_Bander(self.dataset, config=self.bander_config)
        (self.netG, self.netD) = self.init_model()
        self.losses = {"G": 0.0, "D": 0.0, "l1": 0.0, "l2": 0.0, "GP": 0.0}

        # Data option
        self.shape = self.dataset.shape
        self.in_dim = self.dataset.in_dim

    def set_config(self, config=None):
        """
        Setting configuration

        If config/config.json is not exists,
        Use default config 'config.yml'
        """
        if config is None:
            logger.info(
                "Config Setting - JSON config is not entered, Use Default settings"
            )
            config = load_yaml()
        else:
            logger.info("Loaded JSON configuration")

        # Configuration Categories
        self.dataset_config = config["dataset"]
        model_cfg = config["model"]
        train_cfg = config["train"]
        print_cfg = config["print"]
        # result_cfg = config["result"]

        # Model option
        self.save = model_cfg["save"]
        self.load = model_cfg["load"]
        self.model_tag = model_cfg["tag"]
        self.model_path = model_cfg["path"]
        self.model_interval = model_cfg["interval"]

        # Train option
        self.lr = train_cfg["learning_rate"]["base"]
        self.lr_gammaG = train_cfg["learning_rate"]["gammaG"]
        self.lr_gammaD = train_cfg["learning_rate"]["gammaD"]

        self.epochs = train_cfg["epochs"]["iter"]
        self.base_epochs = train_cfg["epochs"]["base"]
        self.iter_epochs = train_cfg["epochs"]["iter"]
        self.iter_critic = train_cfg["epochs"]["critic"]

        self.bander_config = train_cfg["bander"]
        self.pivot = train_cfg["bander"]["pivot"]
        self.gp_weight = train_cfg["bander"]["gp_weight"]
        self.l1_gamma = train_cfg["bander"]["l1_gamma"]
        self.l2_gamma = train_cfg["bander"]["l2_gamma"]

        # Print option
        self.print_verbose = print_cfg["verbose"]
        self.print_newline = print_cfg["newline"]

        # Visual option
        self.visual = config["visual"]

        return config

    def init_dataloader(self, dataset):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.batch_size,
            num_workers=dataset.workers,
        )
        return dataloader

    def init_model(self):
        netG, netD = self.load_model(self.load)

        # Set Oprimizer
        self.optimizerD = optim.RMSprop(netD.parameters(), lr=self.lr * self.lr_gammaD)
        self.optimizerG = optim.RMSprop(netG.parameters(), lr=self.lr * self.lr_gammaG)

        # Set Criterion
        self.criterion_adv = GANLoss(real_label=0.9, fake_label=0.1).to(self.device)
        self.criterion_l1n = nn.SmoothL1Loss().to(self.device)
        self.criterion_l2n = nn.MSELoss().to(self.device)

        return (netG, netD)

    def load_model(self, load_option=False):
        hidden_dim = self.dataset.hidden_dim
        in_dim = self.dataset.in_dim
        device = self.device

        if load_option is True:
            logger.info("Loading Pretrained Models..")
            netG_path = os.path.join(self.model_path, f"netG_{self.model_tag}.pth")
            netD_path = os.path.join(self.model_path, f"netD_{self.model_tag}.pth")

            if os.path.exists(netG_path) and os.path.exists(netD_path):
                logger.info(f" - Loaded net D : {netD_path}, net G: {netG_path}")
                return (torch.load(netG_path), torch.load(netD_path))
            else:
                logger.info(
                    f"Pretrained Model File ('{netG_path}', '{netD_path}') is not found"
                )

        netG = LSTMGenerator(
            in_dim, out_dim=in_dim, hidden_dim=hidden_dim, device=device
        )
        netD = LSTMDiscriminator(in_dim, hidden_dim=hidden_dim, device=device)
        return (netG, netD)

    def run(self):
        logger.info("Evaluate the model")

        errD_real = None
        dashboard = Dashboard(self.dataset)
        for i, (data, label) in enumerate(self.dataloader, 0):
            self.optimizerD.zero_grad()
            self.optimizerG.zero_grad()

            x = data.to(self.device)

            # Train with Fake Data z
            y = self.bander.get_sample(x, self.netG)
            x = self.bander.impute_missing_value(x, y, label, self.pivot)

            Dx = self.netD(x)
            imputate = self.bander.check_missing(label, latest=True)
            errD_real = self.criterion_adv(Dx, target_is_real=True, imputate=imputate)
            errD_real.backward(retain_graph=True)

            y = self.bander.get_sample(x, self.netG)
            Gy = self.netD(y)
            errD_fake = self.criterion_adv(Gy, target_is_real=False)
            errD_fake.backward(retain_graph=True)

            errD = errD_fake + errD_real
            self.optimizerD.step()

            Dy = self.netD(y)
            err_G = self.criterion_adv(Dy, target_is_real=False)
            err_l1 = self.l1_gamma * self.criterion_l1n(y, x)
            err_l2 = self.l2_gamma * self.criterion_l2n(y, x)
            err_gp = self.gp_weight * self._grad_penalty(y, x)

            errG = err_G + err_l1 + err_l2 + err_gp
            errG.backward(retain_graph=True)
            self.optimizerG.step()

            self.losses["G"] += err_G
            self.losses["D"] += errD
            self.losses["l1"] += err_l1
            self.losses["l2"] += err_l2
            self.losses["GP"] += err_gp

            if self.print_verbose > 0:
                print(f"{self._loss_message(i)}", end="\r")

            (x, y, label, bands, detects) = self.bander.process(x, y, label)
            
            if self.visual:
                dashboard.visualize(x, y, label, bands, detects, pivot=self.pivot)

        logger.info(f"\n{self._loss_message()}")
        
    def get_labels(self, text=False):
        origin = self.dataset.origin
        pred = self.bander.pred
        median = self.bander.bands["median"]
        pred[:len(median)] = median
        labels = self.bander.label
        
        output_path = f"output_{self.dataset.title}.csv"
        
        if text:
            text_label = list()
            LABELS = {
                -1.0: "Imputed",
                0.0: "Normal",
                1.0: "Labeled-Anomal",
                2.0: "Detected-Warning",
                3.0: "Detected-Outlier"
            }

            for label in labels:
                text_label.append(LABELS[label])
            labels = text_label

        label_info = pd.DataFrame({"value": origin, "pred": pred, "label": labels})
        label_info.to_csv(output_path)

        logger.info(f"Labeling File is saved to {output_path}")

    def _runtime(self, epoch, time):
        mean_time = time / (epoch - self.base_epochs)
        left_epoch = self.iter_epochs - epoch
        done_time = time + mean_time * left_epoch

        runtime = f"{time:4.2f} / {done_time:4.2f} sec "
        return runtime

    def _grad_penalty(self, y, x):
        gradients = y - x
        gradients_sqr = torch.square(gradients)
        gradients_sqr_sum = torch.sum(gradients_sqr)
        gradients_l2_norm = torch.sqrt(gradients_sqr_sum)
        gradients_penalty = torch.square(1 - gradients_l2_norm) / x.size(0)
        return gradients_penalty

    def _loss_message(self, i=None):
        if i is None:
            i = len(self.dataset)

        message = (
            f"[{i + 1:4d}/{len(self.dataloader):4d}] "
            f"D   {self.losses['D']/(i + 1):.4f} "
            f"G   {self.losses['G']/(i + 1):.4f} "
            f"L1  {self.losses['l1']/(i + 1):.3f} "
            f"L2  {self.losses['l2']/(i + 1):.3f} "
            f"GP  {self.losses['GP']/(i + 1):.3f} ",
        )
        return message
