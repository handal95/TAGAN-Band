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
from utils.logger import Logger
from utils.device import init_device
from utils.dashboard import Dashboard_v2
from utils.metric import TAGAN_Metric
from utils.dataset import TAGANDataset

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
        self.metric = TAGAN_Metric(self.bander_config, self.device)
        (self.netG, self.netD) = self.init_model()
        self.trainer = TAGAN_Trainer(
            (self.trainer_config, self.lr_config, self.print_cfg),
            self.dataset,
            self.metric,
            self.bander,
            self.netD,
            self.netG,
            self.device,
        )
        self.losses = {"G": 0, "D": 0, "l1": 0, "l2": 0, "GP": 0, "Score": 0}

        # Data option
        self.encode_dim = self.dataset.encode_dim

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
        self.print_cfg = print_cfg = config["print"]
        # result_cfg = config["result"]

        # Model option
        self.load = model_cfg["load"]
        self.save = model_cfg["save"]
        self.save_interval = model_cfg["interval"]
        self.model_tag = model_cfg["tag"]
        self.model_path = model_cfg["path"]

        # Train option
        self.lr_config = train_cfg["learning_rate"]
        self.lr = train_cfg["learning_rate"]["base"]
        self.lr_gammaG = train_cfg["learning_rate"]["gammaG"]
        self.lr_gammaD = train_cfg["learning_rate"]["gammaD"]

        self.trainer_config = train_cfg["epochs"]
        self.epochs = train_cfg["epochs"]["iter"]
        self.base_epochs = train_cfg["epochs"]["base"]
        self.iter_epochs = train_cfg["epochs"]["iter"]
        self.iter_critic = train_cfg["epochs"]["critic"]

        self.bander_config = train_cfg["bander"]

        # Print option
        self.print_verbose = print_cfg["verbose"]
        self.print_interval = print_cfg["interval"]

        # Visual option
        self.visual = config["visual"]

        return config

    def init_dataloader(self, dataset, train=False):
        data = dataset.train_dataset if train else dataset.valid_dataset

        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=dataset.batch_size,
            num_workers=dataset.workers,
            shuffle=False
        )
        return dataloader

    def init_model(self):
        logger.info(f"Torch Model")

        hidden_dim = self.dataset.hidden_dim
        encode_dim = self.dataset.encode_dim
        decode_dim = self.dataset.decode_dim
        device = self.device

        if self.load is True:
            logger.info("Loading Pretrained Models..")
            model_path = os.path.join(self.model_path, self.model_tag)
            netG_path = os.path.join(model_path, "netG.pth")
            netD_path = os.path.join(model_path, "netD.pth")

            if os.path.exists(netG_path) and os.path.exists(netD_path):
                logger.info(f" - Loaded net D : {netD_path}, net G: {netG_path}")
                return (torch.load(netG_path), torch.load(netD_path))
            else:
                logger.info(
                    f"Pretrained Model ('{netG_path}', '{netD_path}') is not found"
                )

        netG = LSTMGenerator(encode_dim, decode_dim, hidden_dim=hidden_dim, device=device)
        netD = LSTMDiscriminator(decode_dim, hidden_dim=hidden_dim, device=device)
        netG = netG.to(device)
        netD = netD.to(device)

        logger.info(f"   - Network : \n{netG} \n{netD}")
        return (netG, netD)

    def train(self):
        logger.info("Train the model")
        try:
            train_score_plot = []
            valid_score_plot = []

            BEST_COUNT = 20
            BEST_SCORE = 0.28

            EPOCHS = self.base_epochs + self.iter_epochs
            for epoch in range(self.base_epochs, EPOCHS):
                # Train Section
                tqdm_train = tqdm(self.train_loader, loss_info("Train", epoch))
                train_score = self.trainer.train_step(tqdm_train, epoch, training=True)
                train_score_plot.append(train_score)

                # Valid Section
                tqdm_valid = tqdm(self.valid_loader, loss_info("Valid", epoch))
                valid_score = self.trainer.train_step(tqdm_valid, epoch, training=False)
                valid_score_plot.append(valid_score)

                if self.save:
                    # periodic Model save
                    model_path = os.path.join(self.model_path, self.model_tag)
                    if not os.path.exists(model_path):
                        os.mkdir(model_path)
                    if ((epoch + 1) % self.save_interval) == 0:
                        logger.info(
                            f"[Epoch {epoch + 1:4d}]*** MODEL IS SAVED ***"
                            f"(T {train_score:.4f}, V {valid_score:.4f})"
                        )

                        # Model save
                        self.save_model(model_path)
                        
                    # Best Model save
                    if valid_score < BEST_SCORE:
                        BEST_SCORE = valid_score
                        BEST_COUNT = 20
                        logger.info(f"*** BEST SCORE MODEL ({BEST_SCORE:.4f}) IS SAVED ***")
                        self.save_model(model_path, postfix=f"_{BEST_SCORE:.4f}")
                    else:
                        BEST_COUNT -= 1
                        if BEST_COUNT == 0:
                            BEST_COUNT = 20
                            logger.info(
                                f"*** BEST SCORE MODEL ({BEST_SCORE:.4f}) IS RELOADED ***"
                            )
                            self.netD = torch.load(f"{model_path}/netD_{BEST_SCORE:.4f}.pth")
                            self.netG = torch.load(f"{model_path}/netG_{BEST_SCORE:.4f}.pth")
        except:
            pass
        finally:
            self.netD = torch.load(f"{model_path}/netD_{BEST_SCORE:.4f}.pth")
            self.netG = torch.load(f"{model_path}/netG_{BEST_SCORE:.4f}.pth")
            logger.info("Saving Best model...")                
            self.save_model(model_path)

        plt.plot(train_score_plot, label="train_score")
        plt.plot(valid_score_plot, label="val_score")
        plt.xlabel("epoch")
        plt.ylabel("score(nmae)")
        plt.title("score_plot")
        plt.legend()
        plt.show()

    def save_model(self, model_path, postfix=""):
        netD_path = os.path.join(model_path, f"netD{postfix}.pth")
        netG_path = os.path.join(model_path, f"netG{postfix}.pth")
        torch.save(self.netD, netD_path)
        torch.save(self.netG, netG_path)

    def run(self):
        logger.info("Evaluate the model")

        errD_real = None
        dashboard = Dashboard_v2(self.dataset)
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

    def get_labels(self, text=False):
        origin = self.dataset.data
        pred = self.bander.pred
        median = self.bander.bands["median"]
        pred[: len(median)] = median
        labels = self.bander.label

        output_path = f"output_{self.dataset.file}.csv"

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


class TAGAN_Trainer:
    def __init__(self, configs, dataset, metric, bander, netD, netG, device):
        self.set_config(configs)
        self.dataset = dataset
        self.metric = metric
        self.bander = bander
        self.device = device

        self.dashboard = Dashboard_v2(self.dataset)
        self.train_score_plot = []
        self.valid_score_plot = []

        # Set Models
        self.netD = netD
        self.netG = netG

        # Set Oprimizer
        self.optimizerD = optim.RMSprop(netD.parameters(), lr=self.lr * self.lr_gammaD)
        self.optimizerG = optim.RMSprop(netG.parameters(), lr=self.lr * self.lr_gammaG)

    def set_config(self, configs):
        config1, config2, config3 = configs

        self.epochs = config1["iter"]
        self.base_epochs = config1["base"]
        self.iter_epochs = config1["iter"]
        self.iter_critic = config1["critic"]

        self.lr = config2["base"]
        self.lr_gammaG = config2["gammaG"]
        self.lr_gammaD = config2["gammaD"]

        self.print_verbose = config3["verbose"]
        self.print_interval = config3["interval"]

        self.visual = False

    def train_step(self, tqdm, epoch, training=True):
        TAG = "Train" if training else "Valid"
        losses = {
            "G": 0, "D": 0, "l1": 0, "l2": 0, "GP": 0, 
            "Score": 0,
            "Score1": 0,
            "Score2": 0,
            "Score3": 0,
            "Score4": 0,
            "ScoreAll": 0,
        }

        if not training and self.visual:
            self.dashboard.close_figure()
            
        i = 0
        for i, data in enumerate(tqdm):
            x_window = data["encoder"].to(self.device)
            y_window = data["decoder"].to(self.device)
            a_window = data["answer"].to(self.device)
            x_future = data["enc_future"].to(self.device)
            y_future = data["dec_future"].to(self.device)
            a_future = data["ans_future"].to(self.device)

            # Critic
            if training:
                for _ in range(self.iter_critic):
                    y, x = self.bander.get_random_sample(self.netG)

                    Dx = self.netD(x)
                    Dy = self.netD(y)
                    self.optimizerD.zero_grad()

                    loss_GP = self.metric.grad_penalty(y, x)
                    loss_D_ = Dy.mean() - Dx.mean()

                    loss_D = loss_D_ + loss_GP
                    loss_D.backward()
                    self.optimizerD.step()

            # Optimizer initialize
            self.optimizerD.zero_grad()
            self.optimizerG.zero_grad()

            Dx = self.netD(y_future)
            errD_real = self.metric.GANloss(Dx, target_is_real=True)

            y_gen = self.bander.get_sample(x_window, y_future, self.netG)
            Dy = self.netD(y_gen)
            errD_fake = self.metric.GANloss(Dy, target_is_real=False)
            errD = errD_real + errD_fake

            if training:
                errD_real.backward(retain_graph=True)
                errD_fake.backward(retain_graph=True)
                self.optimizerD.step()

            Dy = self.netD(y_gen)
            err_G = self.metric.GANloss(Dy, target_is_real=False)
            err_l1 = self.metric.l1loss(y_future, y_gen)
            err_l2 = self.metric.l2loss(y_future, y_gen)
            err_gp = self.metric.grad_penalty(y_future, y_gen)
            errG = err_G + err_l1 + err_l2 + err_gp

            if training:
                errG.backward(retain_graph=True)
                self.optimizerG.step()

            # Scoring
            window = a_window.cpu()
            # true = a_future.cpu()
            true = self.dataset.denormalize(y_future.cpu())
            if 0 in true:
                print(true)
            pred = self.dataset.denormalize(y_gen.cpu())
            
            score = self.metric.NMAE(pred, true, real_test=True).detach().numpy()
            score1 = self.metric.NMAE(pred[:, 0], true[:, 0]).detach().numpy()
            score2 = self.metric.NMAE(pred[:, 6], true[:, 6]).detach().numpy()
            score3 = self.metric.NMAE(pred[:, 13], true[:, 13]).detach().numpy()
            score4 = self.metric.NMAE(pred[:, 27], true[:, 27]).detach().numpy()
            score_all = self.metric.NMAE(pred, true).detach().numpy()

            # Losses Log
            losses["D"] += errD
            losses["G"] += err_G
            losses["l1"] += err_l1
            losses["l2"] += err_l2
            losses["GP"] += err_gp
            losses["Score"] += score
            losses["Score1"] += score1
            losses["Score2"] += score2
            losses["Score3"] += score3
            losses["Score4"] += score4
            losses["ScoreAll"] += score_all

            tqdm.set_description(loss_info(TAG, epoch, losses, i))
            if not training and self.visual is True:
                self.dashboard.initalize(window)
                self.dashboard.visualize(window, true, pred)

        if not training and ((epoch + 1) % self.print_interval) == 0:
            true_info = true[0, 0, :].detach().numpy()
            pred_info = pred[0, 0, :].detach().numpy()
            diff_info = true_info - pred_info
            results = pd.DataFrame(
                {
                    "true": true_info,
                    "pred": pred_info,
                    "diff": diff_info,
                    "perc": 100 * diff_info / true_info,
                }, index=self.dataset.targets
            )
            logger.info(
                f"-----  Result (e{epoch + 1:4d}) +1 day -----"
                f"\n{results.T}\n"
                f"Sum(Diff/True) {abs(sum(results['diff'])):.2f}/{sum(results['true']):.2f}"
                f"({sum(abs(results['diff']))/sum(results['true'])*100:.2f}%)"
            )

        return losses["Score"] / (i + 1)


def loss_info(process, epoch, losses=None, i=0):
    if losses is None:
        losses = {
            "G": 0, "D": 0, "l1": 0, "l2": 0, "GP": 0, "Score": 0,
            "Score1": 0, "Score2": 0, "Score3": 0, "Score4": 0, "ScoreAll": 0
        }

    return (
        f"[{process} e{epoch + 1:4d}]"
        f"Score {losses['Score']/(i+1):7.4f}("
        f"all {losses['ScoreAll']/(i+1):5.2f} "
        f"1d {losses['Score1']/(i+1):5.2f} "
        f"1w {losses['Score2']/(i+1):5.2f} "
        f"2w {losses['Score3']/(i+1):5.2f} "
        f"4w {losses['Score4']/(i+1):5.2f})"
        f"D {losses['D']/(i+1):7.3f} "
        f"G {losses['G']/(i+1):7.3f} "
        f"L1 {losses['l1']/(i+1):6.3f} "
        f"L2 {losses['l2']/(i+1):6.3f} "
        f"GP {losses['GP']/(i+1):7.3f} "
    )
