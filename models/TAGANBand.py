import os
import torch
import torch.optim as optim
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.args import load_yaml
from utils.logger import Logger
from utils.device import init_device
from utils.dashboard import Dashboard_v2
from utils.metric import TAGAN_Metric
from models.dataset import TAGANDataset
from models.lstm_layer import LSTMGenerator, LSTMDiscriminator

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
        # self.bander = TAGAN_Bander(self.dataset, config=self.metric_config)
        self.metric = TAGAN_Metric(self.metric_config, self.device)
        self.netG, self.netD = self.init_model()
        # Set Oprimizer
        self.optimizerD = optim.RMSprop(self.netD.parameters(), lr=self.lr * self.lr_gammaD)
        self.optimizerG = optim.RMSprop(self.netG.parameters(), lr=self.lr * self.lr_gammaG)

        self.dashboard = Dashboard_v2(self.dataset)

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
        metric_cfg = config["metric"]
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

        self.metric_config = metric_cfg

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
                netG = torch.load(netG_path)
                netD = torch.load(netD_path)
                return (netG, netD)
            else:
                logger.info(f"Pretrained Model is not found")

        netG = LSTMGenerator(encode_dim, decode_dim, hidden_dim=hidden_dim, device=device)
        netD = LSTMDiscriminator(decode_dim, hidden_dim=hidden_dim, device=device)
        netG = netG.to(device)
        netD = netD.to(device)

        logger.info(f"   - Network : \n{netG} \n{netD}")
        return (netG, netD)

    def train(self):
        logger.info("Train the model")
        model_path = os.path.join(self.model_path, self.model_tag)
        try:
            train_score_plot = []
            valid_score_plot = []

            BEST_COUNT = 20
            BEST_SCORE = 0.2358

            EPOCHS = self.base_epochs + self.iter_epochs
            for epoch in range(self.base_epochs, EPOCHS):
                # Train Section
                tqdm_train = tqdm(self.train_loader, loss_info("Train", epoch))
                train_score = self.train_step(tqdm_train, epoch, training=True)
                train_score_plot.append(train_score)

                # Valid Section
                tqdm_valid = tqdm(self.valid_loader, loss_info("Valid", epoch))
                valid_score = self.train_step(tqdm_valid, epoch, training=False)
                valid_score_plot.append(valid_score)

                if self.save:
                    # periodic Model save
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
            raise
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
            # Critic
            if training:
                for _ in range(self.iter_critic):
                    y_fake, y_real = self.dataset.get_random_sample(self.netG)

                    Dx = self.netD(y_real)
                    Dy = self.netD(y_fake)
                    self.optimizerD.zero_grad()

                    loss_GP = self.metric.grad_penalty(y_fake, y_real)
                    loss_D_ = Dy.mean() - Dx.mean()

                    loss_D = loss_D_ + loss_GP
                    loss_D.backward()
                    self.optimizerD.step()
            
            x_window = data["encoder"].to(self.device)
            y_window = data["decoder"].to(self.device)
            a_window = data["answer"].to(self.device)
            x_future = data["enc_future"].to(self.device)
            y_future = data["dec_future"].to(self.device)
            a_future = data["ans_future"].to(self.device)
            
            # Optimizer initialize
            self.optimizerD.zero_grad()
            self.optimizerG.zero_grad()

            Dx = self.netD(y_future)
            errD_real = self.metric.GANloss(Dx, target_is_real=True)

            y_fake = self.dataset.get_sample(x_window, self.netG)
            Dy = self.netD(y_fake)
            errD_fake = self.metric.GANloss(Dy, target_is_real=False)
            errD = errD_real + errD_fake

            if training:
                errD_real.backward(retain_graph=True)
                errD_fake.backward(retain_graph=True)
                self.optimizerD.step()

            Dy = self.netD(y_fake)
            err_G = self.metric.GANloss(Dy, target_is_real=False)
            err_l1 = self.metric.l1loss(y_future, y_fake)
            err_l2 = self.metric.l2loss(y_future, y_fake)
            err_gp = self.metric.grad_penalty(y_future, y_fake)
            errG = err_G + err_l1 + err_l2 + err_gp

            if training:
                errG.backward(retain_graph=True)
                self.optimizerG.step()

            # Scoring
            window = a_window.cpu()
            true = a_future.cpu()
            pred = self.dataset.denormalize(y_fake.cpu())
            
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
    
    def save_model(self, model_path, postfix=""):
        netD_path = os.path.join(model_path, f"netD{postfix}.pth")
        netG_path = os.path.join(model_path, f"netG{postfix}.pth")
        torch.save(self.netD, netD_path)
        torch.save(self.netG, netG_path)


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
