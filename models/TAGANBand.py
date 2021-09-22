import os
import time
from numpy.random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from tqdm import tqdm
from models.bander import TAGAN_Bander
from models.LSTMGAN import LSTMGenerator, LSTMDiscriminator
import matplotlib.pyplot as plt

from utils.args import load_yaml
from utils.loss import GANLoss
from utils.logger import Logger
from utils.device import init_device
from utils.dashboard import Dashboard
from utils.dataset import TAGANDataset
from utils.metric import metric_NMAE

logger = Logger(__file__)

MISSING = -1.0
ANOMALY = 1.0


class TAGANBand:
    """
    TAGANBand : Timeseries Analysis using GAN Band

    The Model for Detecting anomalies / Imputating missing value in timeseries data

    """

    def __init__(self, config: dict = None) -> None:
        # Set device
        self.device = init_device()

        # Set Config
        config = self.set_config(config=config)

        # Dataset option
        self.dataset = TAGANDataset(self.dataset_config, device=self.device)
        self.train_loader = self.init_dataloader(self.dataset, train=True)
        self.valid_loader = self.init_dataloader(self.dataset)

        # Model option
        self.bander = TAGAN_Bander(self.dataset, config=self.bander_config)
        (self.netG, self.netD) = self.init_model()
        self.losses = {"G": 0, "D": 0, "l1": 0, "l2": 0, "GP": 0, "Score": 0}

        # Data option
        self.shape = self.dataset.shape
        self.in_dim = self.dataset.in_dim

    def set_config(self, config: dict = None) -> dict:
        """
        Setting configuration

        If config/config.json is not exists,
        Use default config 'config.yml'
        """
        if config is None:
            logger.info("  Config : Default configs")
            config = load_yaml()
        else:
            logger.info("  Config : JSON configs")

        # Configuration Categories
        self.dataset_config = config["dataset"]
        model_cfg = config["model"]
        train_cfg = config["train"]
        print_cfg = config["print"]
        # result_cfg = config["result"]

        # Model option
        self.load = model_cfg["load"]
        self.save = model_cfg["save"]
        self.save_interval = model_cfg["interval"]
        self.model_tag = model_cfg["tag"]
        self.model_path = model_cfg["path"]

        # Train option
        self.lr = train_cfg["learning_rate"]["base"]
        self.lr_gammaG = train_cfg["learning_rate"]["gammaG"]
        self.lr_gammaD = train_cfg["learning_rate"]["gammaD"]

        self.epochs = train_cfg["epochs"]["iter"]
        self.base_epochs = train_cfg["epochs"]["base"]
        self.iter_epochs = train_cfg["epochs"]["iter"]
        self.iter_critic = train_cfg["epochs"]["critic"]

        self.bander_config = train_cfg["bander"]
        self.window_len = train_cfg["bander"]["window_len"]
        self.gp_weight = train_cfg["bander"]["gp_weight"]
        self.l1_gamma = train_cfg["bander"]["l1_gamma"]
        self.l2_gamma = train_cfg["bander"]["l2_gamma"]

        # Print option
        self.print_verbose = print_cfg["verbose"]
        self.print_newline = print_cfg["newline"]

        # Visual option
        self.visual = config["visual"]

        return config

    def init_dataloader(self, dataset, train=False):
        data = dataset.train_dataset if train else dataset.valid_dataset

        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=dataset.batch_size,
            num_workers=dataset.workers,
            shuffle=True if train else False,
        )
        return dataloader

    def init_model(self):
        logger.info(f"Torch Model")
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
        target_dim = self.dataset.target_dim
        device = self.device

        if load_option is True:
            logger.info("Loading Pretrained Models..")
            netG_path = os.path.join(self.model_path, f"{self.model_tag}/netG.pth")
            netD_path = os.path.join(self.model_path, f"{self.model_tag}/netD.pth")

            if os.path.exists(netG_path) and os.path.exists(netD_path):
                logger.info(f" - Loaded net D : {netD_path}, net G: {netG_path}")
                return (torch.load(netG_path), torch.load(netD_path))
            else:
                logger.info(
                    f"Pretrained Model File ('{netG_path}', '{netD_path}') is not found"
                )

        netG = LSTMGenerator(
            in_dim, out_dim=target_dim, hidden_dim=hidden_dim, device=device
        ).to(device)
        netD = LSTMDiscriminator(target_dim, hidden_dim=hidden_dim, device=device).to(
            device
        )

        logger.info(f"   - Network : \n{netG} \n{netD}")
        return (netG, netD)

    def train(self):
        logger.info("Train the model")
        train_score_plot = []
        valid_score_plot = []

        # dashboard = Dashboard(self.dataset)
        EPOCHS = self.base_epochs + self.iter_epochs
        for epoch in range(self.base_epochs, EPOCHS):
            # Train Section
            total_score = 0
            losses = {"G": 0, "D": 0, "l1": 0, "l2": 0, "GP": 0, "Score": 0}
            tqdm_train = tqdm(self.train_loader, loss_info("Train", epoch, losses, 0))

            for i, data in enumerate(tqdm_train):
                x_window = data["encoder"].to(self.device)
                y_window = data["decoder"].to(self.device)
                x_future = data["enc_future"].to(self.device)
                y_future = data["dec_future"].to(self.device)

                # Critic
                for critic in range(self.iter_critic):
                    y, x = self.bander.get_random_sample(self.netG)

                    Dx = self.netD(x)
                    Dy = self.netD(y)
                    self.optimizerD.zero_grad()

                    loss_GP = self.gp_weight * self._grad_penalty(y, x)
                    loss_D_ = Dy.mean() - Dx.mean()

                    loss_D = loss_D_ + loss_GP
                    loss_D.backward()
                    self.optimizerD.step()

                # Optimizer initialize
                self.optimizerD.zero_grad()
                self.optimizerG.zero_grad()

                Dx = self.netD(y_future)
                errD_real = self.criterion_adv(Dx, target_is_real=True)
                errD_real.backward(retain_graph=True)

                y_gen = self.bander.get_sample(x_window, y_future, self.netG)
                Dy = self.netD(y_gen)
                errD_fake = self.criterion_adv(Dy, target_is_real=False)
                errD_fake.backward(retain_graph=True)
                errD = errD_real + errD_fake
                self.optimizerD.step()

                Dy = self.netD(y_gen)
                err_G = self.criterion_adv(Dy, target_is_real=False)
                err_l1 = self.l1_gamma * self.criterion_l1n(y_future, y_gen)
                err_l2 = self.l2_gamma * self.criterion_l2n(y_future, y_gen)
                err_gp = self.gp_weight * self._grad_penalty(y_future, y_gen)
                errG = err_G + err_l1 + err_l2 + err_gp
                errG.backward(retain_graph=True)
                self.optimizerG.step()

                # Scoring
                true = self.dataset.decoder_denormalize(y_future.cpu())
                pred = self.dataset.decoder_denormalize(y_gen.cpu())
                score = metric_NMAE(pred, true).detach().numpy()

                # Losses Log
                losses["D"] += errD
                losses["G"] += err_G
                losses["l1"] += err_l1
                losses["l2"] += err_l2
                losses["GP"] += err_gp
                losses["Score"] += score
                tqdm_train.set_description(loss_info("Train", epoch, losses, i))
            train_score_plot.append((losses["Score"]/i + 1))

            # Valid Section
            losses = {"G": 0, "D": 0, "l1": 0, "l2": 0, "GP": 0, "Score": 0}
            tqdm_valid = tqdm(self.valid_loader, loss_info("Valid", epoch, losses, 0))
            for i, data in enumerate(tqdm_valid):
                x_window = data["encoder"].to(self.device)
                y_window = data["decoder"].to(self.device)
                x_future = data["enc_future"].to(self.device)
                y_future = data["dec_future"].to(self.device)

                # Optimizer initialize
                self.optimizerD.zero_grad()
                self.optimizerG.zero_grad()

                Dx = self.netD(y_future)
                errD_real = self.criterion_adv(Dx, target_is_real=True)

                y_gen = self.bander.get_sample(x_window, y_future, self.netG)
                Dy = self.netD(y_gen)
                errD_fake = self.criterion_adv(Dy, target_is_real=False)
                errD = errD_real + errD_fake

                Dy = self.netD(y_gen)
                err_G = self.criterion_adv(Dy, target_is_real=False)
                err_l1 = self.l1_gamma * self.criterion_l1n(y_future, y_gen)
                err_l2 = self.l2_gamma * self.criterion_l2n(y_future, y_gen)
                err_gp = self.gp_weight * self._grad_penalty(y_future, y_gen)
                errG = err_G + err_l1 + err_l2 + err_gp

                # Scoring
                true = self.dataset.decoder_denormalize(y_future.cpu())
                pred = self.dataset.decoder_denormalize(y_gen.cpu())
                score = metric_NMAE(pred, true).detach().numpy()

                # Loss Log
                losses["D"] += errD
                losses["G"] += err_G
                losses["l1"] += err_l1
                losses["l2"] += err_l2
                losses["GP"] += err_gp
                losses["Score"] += score
                tqdm_valid.set_description(loss_info("Valid", epoch, losses, i))

            valid_score_plot.append((losses["Score"]/i + 1))

            # for i, (data) in enumerate(tqdm_dataset):
            #     for critic in range(self.iter_critic):
            #         y, x = self.bander.get_random_sample(self.netG)

            #         Dx = self.netD(x)
            #         Dy = self.netD(y)
            #         self.optimizerD.zero_grad()

            #         loss_GP = self.gp_weight * self._grad_penalty(y, x)
            #         loss_D_ = Dy.mean() - Dx.mean()

            #         loss_D = loss_D_ + loss_GP
            #         loss_D.backward()
            #         self.optimizerD.step()

            #         # if i == self.iter_critic - 1:
            #         #     self.losses["D"] += loss_D
            #         #     self.losses["GP"] += loss_GP

            #     # Vanilla
            #     self.optimizerD.zero_grad()
            #     self.optimizerG.zero_grad()

            #     x = data.to(self.device)
            #     Dx = self.netD(x)
            #     errD_real = self.criterion_adv(Dx, target_is_real=True)
            #     errD_real.backward(retain_graph=True)

            #     y = self.bander.get_sample(x, self.netG)
            #     Dy = self.netD(y)
            #     errD_fake = self.criterion_adv(Dy, target_is_real=False)
            #     errD_fake.backward(retain_graph=True)

            #     errD = errD_fake + errD_real
            #     self.optimizerD.step()

            #     Dy = self.netD(y)
            #     err_G = self.criterion_adv(Dy, target_is_real=False)
            #     err_l1 = self.l1_gamma * self.criterion_l1n(y, x)
            #     err_l2 = self.l2_gamma * self.criterion_l2n(y, x)
            #     err_gp = self.gp_weight * self._grad_penalty(y, x)
            #     errG = err_G + err_l1 + err_l2 + err_gp
            #     errG.backward(retain_graph=True)
            #     self.optimizerG.step()

            #     true = self.bander.variables(x)
            #     y = self.bander.get_sample(x, self.netG)
            #     pred = self.bander.variables(y)

            #     score = metric_NMAE(pred, true)

            #     losses["G"] += err_G
            #     losses["D"] += errD
            #     losses["l1"] += err_l1
            #     losses["l2"] += err_l2
            #     losses["GP"] += err_gp
            #     losses["Score"] = score

            #     # Print loss
            #     if self.print_verbose > 0:
            #         tqdm_dataset.set_postfix(
            #             {
            #                 "Epoch": epoch + 1,
            #                 "Score": f'{losses["Score"]:2.4f}',
            #                 "D": f'{losses["D"]:2.4f}',
            #                 "G": f'{losses["G"]:2.4f}',
            #                 "L1": f'{losses["l1"]:2.4f}',
            #                 "L2": f'{losses["l2"]:2.4f}',
            #                 "GP": f'{losses["GP"]:2.4f}',
            #                 # 'Score': '{:06f}'.format(batch_score.item()),
            #                 # 'Total Score' : '{:06f}'.format(total_score/(batch+1)),
            #             }
            #         )

            #     # Visualize

            #     if self.visual is True and (epoch) % 10 == 0:
            #         dashboard.train_vis(pred)
            
            
            if self.save is True and (epoch + 1) % self.save_interval == 0:
                logger.info(f"Epcoh {epoch + 1} Model is saved")
                true_info = true[0, 0, :].detach().numpy()
                pred_info = pred[0, 0, :].detach().numpy()
                result_info = pd.DataFrame({
                    'true': true_info,
                    'pred': pred_info,
                    'diff': true_info - pred_info,
                    'perc': ((true_info - pred_info) / true_info) * 100,
                })
                logger.info(
                    f"-----  Result  -----"
                    f"\n{result_info.T}\n"
                    f"Sum(Diff/True) ({abs(sum(result_info['diff'])):.2f}/{sum(result_info['true']):.2f})"
                    f"({sum(abs(result_info['diff']))/sum(result_info['true'])*100:.2f}%)"
                )

                logger.info(loss_info("Valid", epoch, losses, i))
                model_path = os.path.join(self.model_path, self.model_tag)
                if not os.path.exists(model_path):
                    os.mkdir(model_path)

                netD_path = os.path.join(model_path, "netD.pth")
                netG_path = os.path.join(model_path, "netG.pth")
                torch.save(self.netD, netD_path)
                torch.save(self.netG, netG_path)
        plt.plot(train_score_plot, label="train_score")
        plt.plot(valid_score_plot, label="val_score")
        plt.xlabel("epoch")
        plt.ylabel("score(nmae)")
        plt.title("score_plot")
        plt.legend()
        plt.show()

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
            Dy = self.netD(y)
            errD_fake = self.criterion_adv(Dy, target_is_real=False)
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
        origin = self.dataset.data
        pred = self.bander.pred
        median = self.bander.bands["median"]
        pred[: len(median)] = median
        labels = self.bander.label

        output_path = f"output_{self.dataset.title}.csv"

        if text:
            text_label = list()
            LABELS = {
                -1.0: "Imputed",
                0.0: "Normal",
                1.0: "Labeled-Anomal",
                2.0: "Detected-Warning",
                3.0: "Detected-Outlier",
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


def loss_info(process, epoch, losses, i=0):
    return (
        f"[{process} e{epoch + 1:4d}]"
        f"Score {losses['Score']/(i+1):7.4f} "
        f"D {losses['D']/(i+1):7.3f} "
        f"G {losses['G']/(i+1):7.3f} "
        f"L1 {losses['l1']/(i+1):6.3f} "
        f"L2 {losses['l2']/(i+1):6.3f} "
        f"GP {losses['GP']/(i+1):7.3f} "
    )
