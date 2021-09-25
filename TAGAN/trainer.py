import torch
import pandas as pd

from tqdm import tqdm
from torch.optim import RMSprop
import matplotlib.pyplot as plt

from utils.logger import Logger
from TAGAN.model import TAGANModel
from TAGAN.metric import TAGANMetric
from TAGAN.dataset import TAGANDataset

logger = Logger(__file__)


class TAGANTrainer:
    def __init__(
        self,
        config: dict,
        dataset: TAGANDataset,
        models: TAGANModel,
        metric: TAGANMetric,
        device: torch.device,
    ) -> None:
        # Set device
        self.device = device

        self.dataset = dataset
        self.models = models
        self.metric = metric

        # Set Config
        config = self.set_config(config=config)
        

    def set_config(self, config: dict = None) -> dict:
        """
        Configure settings related to the data set.

        params:
            config: Trainer configuration dict
                `config['trainer']`
        """

        self.print_cfg = print_cfg = config["print"]

        # Train option
        self.lr_config = config["learning_rate"]
        self.lr = config["learning_rate"]["base"]
        self.lr_gammaG = config["learning_rate"]["gammaG"]
        self.lr_gammaD = config["learning_rate"]["gammaD"]

        self.trainer_config = config["epochs"]
        self.epochs = config["epochs"]["iter"]
        self.base_epochs = config["epochs"]["base"]
        self.iter_epochs = config["epochs"]["iter"]
        self.iter_critic = config["epochs"]["critic"]

        # Print option
        self.print_verbose = print_cfg["verbose"]
        self.print_interval = print_cfg["interval"]

        # Visual option
        self.visual = config["visual"]
        self.print_cfg = config["print"]

    def run(self, netD, netG, trainset, validset):
        logger.info("Train the model")

        train_score_plot = []
        valid_score_plot = []

        self.optimD = RMSprop(netD.parameters(), lr=self.lr * self.lr_gammaD)
        self.optimG = RMSprop(netG.parameters(), lr=self.lr * self.lr_gammaG)

        EPOCHS = self.base_epochs + self.iter_epochs
        
        self.netD = netD
        self.netG = netG
        for epoch in range(self.base_epochs, EPOCHS):
            # Train Section
            losses = init_loss()
            train_tqdm = tqdm(trainset, loss_info("Train", epoch, losses))
            train_score = self.train_step(train_tqdm, epoch, training=True)
            train_score_plot.append(train_score)

            # # Valid Section
            losses = init_loss()
            valid_tqdm = tqdm(validset, loss_info("Valid", epoch))
            valid_score = self.train_step(valid_tqdm, epoch, training=False)
            valid_score_plot.append(valid_score)

            if (epoch + 1) % self.models.save_interval == 0:
                self.models.save(self.netD, self.netG)

            # Best Model save
            self.netD, self.netG = self.models.update(
                self.netD, self.netG, valid_score
            )

        best_score = self.models.best_score
        netD_best, netG_best = self.models.load(postfix=f"{best_score:.4f}")
        self.models.save(netD_best, netG_best)

        self.plot_score(train_score_plot, valid_score_plot)
        
    def train_step(self, tqdm, epoch, training=True):
        def discriminate(x):
            return self.netD(x).to(self.device)

        def generate(x):
            return self.netG(x).to(self.device)

        # if not training and self.visual:
        #     self.dashboard.close_figure()
        losses = init_loss()
        TAG = "Train" if training else "Valid"
        for i, data in enumerate(tqdm):
            if training:
                for _ in range(self.iter_critic):
                    true_x, true_y = self.dataset.get_random()
                    fake_y = generate(true_x)

                    Dx = discriminate(true_y)
                    Dy = discriminate(fake_y)
                    self.optimD.zero_grad()

                    loss_GP = self.metric.grad_penalty(fake_y, true_y)
                    loss_D_ = Dy.mean() - Dx.mean()

                    loss_D = loss_D_ + loss_GP
                    loss_D.backward()
                    self.optimD.step()

            # Data
            true_x = data["encode"].to(self.device)
            real = data["window"].to(self.device)
            
            true_y = data["decode"].to(self.device)

            # Optimizer initialize
            self.optimD.zero_grad()
            self.optimG.zero_grad()

            # #######################            
            # Discriminator Training
            # #######################            
            Dx = discriminate(true_y)
            errD_real = self.metric.GANloss(Dx, target_is_real=True)

            fake_y = generate(true_x)
            Dy = discriminate(fake_y)
            errD_fake = self.metric.GANloss(Dy, target_is_real=False)
            errD = errD_real + errD_fake
            
            if training:
                errD_real.backward(retain_graph=True)
                errD_fake.backward(retain_graph=True)
                self.optimD.step() 

            # Discriminator Loss
            losses["D"] += errD
            
            # #######################            
            # Generator Trainining
            # #######################            
            Dy = self.netD(fake_y)
            err_G = self.metric.GANloss(Dy, target_is_real=False)
            err_l1 = self.metric.l1loss(true_y, fake_y)
            err_l2 = self.metric.l2loss(true_y, fake_y)
            err_gp = self.metric.grad_penalty(true_y, fake_y)
            errG = err_G + err_l1 + err_l2 + err_gp
            
            if training:
                errG.backward(retain_graph=True)
                self.optimG.step()

            # Generator Loss
            losses["G"] += err_G
            losses["l1"] += err_l1
            losses["l2"] += err_l2
            losses["GP"] += err_gp
            

            # #######################            
            # Scoring
            # #######################          
            pred = self.dataset.denormalize(fake_y.cpu())
            real = data["future"]

            score = self.metric.NMAE(pred, real, real_test=True).detach().numpy()
            score1 = self.metric.NMAE(pred[:, 0], real[:, 0]).detach().numpy()
            score2 = self.metric.NMAE(pred[:, 6], real[:, 6]).detach().numpy()
            score3 = self.metric.NMAE(pred[:, 13], real[:, 13]).detach().numpy()
            score4 = self.metric.NMAE(pred[:, 27], real[:, 27]).detach().numpy()
            score_all = self.metric.NMAE(pred, real).detach().numpy()

            # Losses Log
            losses["Score"] += score
            losses["Score1"] += score1
            losses["Score2"] += score2
            losses["Score3"] += score3
            losses["Score4"] += score4
            losses["ScoreAll"] += score_all

            tqdm.set_description(loss_info(TAG, epoch, losses, i))
            # if not training and self.visual is True:
            #     self.dashboard.initalize(window)
            #     self.dashboard.visualize(window, true, pred)

        if not training and ((epoch + 1) % self.print_interval) == 0:
            real_info = real[0, 0, :].cpu().detach().numpy()
            pred_info = pred[0, 0, :].cpu().detach().numpy()
            diff_info = real_info - pred_info
            results = pd.DataFrame(
                {
                    "real": real_info,
                    "pred": pred_info,
                    "diff": diff_info,
                    "perc": 100 * diff_info / real_info,
                },
                index=self.dataset.targets,
            )
            logger.info(
                f"-----  Result (e{epoch + 1:4d}) +1 day -----"
                f"\n{results.T}\n"
                f"Sum(Diff/True) {abs(sum(results['diff'])):.2f}/{sum(results['real']):.2f}"
                f"({sum(abs(results['diff']))/sum(results['real'])*100:.2f}%)"
            )

        return losses["Score"] / (i + 1)

    def plot_score(self, train_scores, valid_scores):
        plt.plot(train_scores, label="train_score")
        plt.plot(valid_scores, label="valid_score")
        plt.xlabel("epoch")
        plt.ylabel("score(nmae)")
        plt.title("score_plot")
        plt.legend()
        plt.show()

def loss_info(process, epoch, losses=None, i=0):
    if losses is None:
        losses = init_loss()

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


def init_loss() -> dict:
    return {
        "G": 0,
        "D": 0,
        "l1": 0,
        "l2": 0,
        "GP": 0,
        "Score": 0,
        "Score1": 0,
        "Score2": 0,
        "Score3": 0,
        "Score4": 0,
        "ScoreAll": 0,
    }
