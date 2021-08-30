import os
import time
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad as torch_grad
import numpy as np
import pandas as pd

from timeseries.args import init_arguments
from timeseries.logger import Logger
from timeseries.bandgan import BandGAN
from timeseries.datasets import TimeseriesDataset
from timeseries.layers.LSTMGAN import LSTMGenerator, LSTMDiscriminator

from utils.loss import GANLoss
from utils.visualize import Dashboard


logger = Logger(__file__)

torch.manual_seed(31)


class BandGAN:
    """
    Band GAN
    """

    def __init__(self, config=None):
        # Config option
        self.device = self.init_device()
        config = self.set_config(config=config)

        # Model option
        self.dataset = TimeseriesDataset(config["dataset"], self.device)
        self.origin = TimeseriesDataset(config["dataset"], self.device)
        self.dataloader = self.init_dataloader(self.dataset)
        (self.netG, self.netD) = self.init_model()

        self.batch_size = self.dataset.batch_size
        self.seq_len = self.dataset.seq_len
        self.in_dim = self.dataset.n_feature
        self.shape = (self.batch_size, self.seq_len, self.in_dim)

        self.losses = {"G": 0.0, "D": 0.0, "l1": 0.0, "l2": 0.0, "GP": 0.0}
        self.visual = True if self.batch_size == 1 else False

    def set_config(self, config=None):
        if config is None:
            logger.info("JSON configuration is None, Use Default Config Settings")
            config = init_arguments()
        else:
            logger.info("Loaded JSON configuration")

        # Sub config
        model_cfg = config["model"]
        train_cfg = config["train"]
        print_cfg = config["print"]
        result_cfg = config["result"]
        
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
        
        self.gp_weight = train_cfg["wgan"]["gp_weight"]
        self.cond = train_cfg["wgan"]["cond"]
        
        # Print option
        self.print_verbose = print_cfg["verbose"]
        self.print_newline = print_cfg["newline"]
        
        # Result option
        self.sigma = {
            "safety": result_cfg["bands_sigma"]["safety"],
            "normal": result_cfg["bands_sigma"]["normal"],
            "notion": result_cfg["bands_sigma"]["notion"],
            "danger": result_cfg["bands_sigma"]["danger"],
        }
        
        return config

    def init_device(self):
        """
        Setting device option
        """

        device = torch.device("cpu")

        if torch.cuda.is_available():
            device = torch.device("cuda:0")

        logger.info(f"Set torch device `{device}`")
        return device

    def init_dataloader(self, dataset):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.batch_size,
            # shuffle=dataset.shuffle,
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
        in_dim = self.dataset.n_feature
        device = self.device
        
        if load_option is True:
            logger.info("Loading Pretrained Models..")
            netG_path = os.path.join(self.model_path, f"netG_{self.model_tag}.pth")
            netD_path = os.path.join(self.model_path, f"netD_{self.model_tag}.pth")
            if os.path.exists(netG_path) and os.path.exists(netD_path):
                logger.info(f" - Loaded net D : {netD_path}, net G: {netG_path}")
                return (torch.load(netG_path), torch.load(netD_path))
            else:
                logger.info(f"Pretrained Model File ('{netG_path}', '{netD_path}') is not found")

        netG = LSTMGenerator(in_dim, out_dim=in_dim, hidden_dim=hidden_dim, device=device).to(device)
        netD = LSTMDiscriminator(in_dim, hidden_dim=hidden_dim, device=device).to(device)
        return (netG, netD)       
        
        
    def train(self):
        logger.info("Train the model")

        device = self.device

        runtime = 0
        samples = None
        data_samples = None
        dashboard = Dashboard(self.dataset)

        for epoch in range(self.base_epochs, self.iter_epochs):
            start_time = time.time()
            # for i in range(self.iter_critic):
            #     y, x = self.dataset.get_samples(self.netG, shape=self.shape, cond=self.cond)
                
            #     Dx = self.netD(x)
            #     DGz = self.netD(y)
            #     self.optimizerD.zero_grad()
                
            #     with torch.backends.cudnn.flags(enabled=False):
            #         grad_penalty = self._grad_penalty(x, y)
            #     loss_D = DGz.mean() - Dx.mean() + grad_penalty
            #     loss_D.backward()
            #     self.optimizerD.step()

            #     if i == self.iter_critic - 1:
            #         self.losses["D"].append(loss_D)
            #         self.losses["GP"].append(grad_penalty)

            # self.optimizerG.zero_grad()
            # y, x = self.dataset.get_samples(self.netG, shape=self.shape, cond=self.cond)
            # Dy = self.netD(y)
            
            # errl1 = self.criterion_l1n(y, x)
            # errl2 = self.criterion_l2n(y, x)

            # loss_G = -Dy.mean()
            # loss_G.backward()

            # self.optimizerG.step()
            # self.losses["G"].append(loss_G)
            # self.losses["l1"] = errl1
            # self.losses["l2"] = errl2
            
            # if self.print_verbose > 0:
            #     print(
            #         f"[{(epoch + 1):4d}/{self.iter_epochs:4d}]"
            #         f" D  {self.losses['D'][-1]:2.4f}"
            #         f" G  {self.losses['G'][-1]:2.4f}",
            #         f" L1 {self.losses['l1']:2.4f} ",
            #         f" L2 {self.losses['l2']:2.4f} ",
            #         f" GP  {self.losses['GP'][-1]:.4f}",
            #         end="\r",
            #     )
            #     if (epoch + 1) % self.print_newline == 0:
            #         print()

            running_loss = {"D": 0, "G": 0, "Dx": 0, "l1": 0, "l2": 0, "GP": 0}
            for i, data in enumerate(self.dataloader, 0):
                self.optimizerD.zero_grad()
                self.optimizerG.zero_grad()

                x = data.to(self.device)
                shape = (x.size(0), x.size(1), self.in_dim)
                
                Dx = self.netD(x)
                errD_real = self.criterion_adv(Dx, target_is_real=True)
                errD_real.backward()

                # Train with Fake Data z
                y = self.netG(torch.randn(shape).to(device))
                DGz1 = self.netD(y)

                errD_fake = self.criterion_adv(DGz1, target_is_real=False)
                errD_fake.backward()
                errD = errD_real + errD_fake
                self.optimizerD.step() 

                y = self.netG(torch.randn(shape).to(device))
                Dy = self.netD(y)

                errG_ = self.criterion_adv(Dy, target_is_real=False)

                gradients = y - x
                gradients_sqr = torch.square(gradients)
                gradients_sqr_sum = torch.sum(gradients_sqr)
                gradients_l2_norm = torch.sqrt(gradients_sqr_sum)
                gradients_penalty = torch.square(1 - gradients_l2_norm)
                gradients_penalty = gradients_penalty / shape[0]

                errl1 = self.criterion_l1n(y, x) * 10.0
                errl2 = self.criterion_l2n(y, x) * 10.0
                errG = errG_ + errl1 + errl2 + gradients_penalty
                errG.backward()

                self.optimizerG.step()
                running_loss["D"] += errD
                running_loss["G"] += errG_
                running_loss["l1"] += errl1
                running_loss["l2"] += errl2
                running_loss["GP"] += gradients_penalty
                
                print(
                    f"[{i + 1:4d}/{len(self.dataloader):4d}] ", end='\r')

                if self.print_verbose > 0 and (i + 1) == len(self.dataloader):
                    runtime += (time.time() - start_time) 
                    print(
                        f"[{i + 1:4d}/{len(self.dataloader):4d}] "
                        f"D  {running_loss['D']/(i + 1):.4f} ",
                        f"G  {running_loss['G']/(i + 1):.4f} ",
                        f"L1  {running_loss['l1']/(i + 1):.3f} ",
                        f"L2  {running_loss['l2']/(i + 1):.3f} ",
                        f"GP  {running_loss['GP']/(i + 1):.3f} ",
                        f" || {self._runtime(epoch + 1, runtime)}",
                        end="\r"
                    )

                if self.visual is True:
                    y1 = self.netG(torch.randn(shape).to(device))
                    y_ = pd.DataFrame(y1[0].T.cpu().detach().numpy())
                    x_ = pd.DataFrame(x[0].T.cpu().detach().numpy())
                    if samples is None:
                        samples = y_
                        data_samples = x_
                    else:
                        data_samples = data_samples.append(self.dataset.denormalize(x_))
                        samples = samples.append(self.dataset.denormalize(y_))
                        data_samples.to_csv("x.csv")
                        samples.to_csv("y.csv")
                        
                    dashboard._visualize(
                        self.dataset.time[i],
                        self.dataset.denormalize(x.cpu()),
                        self.dataset.denormalize(y1.cpu()),
                    )
                    # dashboard.visualize(
                    #     x.cpu(),
                    #     y.cpu(),
                    #     self.cond,
                    #     normalize=True
                    # )
            
            if (epoch + 1) % (self.print_newline // 10) == 0:
                print()

            if self.save["opt"] is True and (epoch + 1) % self.save["interval"] == 0:
                logger.info(f"Model saved Epochs {epoch+1}")
                torch.save(self.netG, f"pretrained/netG_e{epoch + 1}.pth")
                torch.save(self.netD, f"pretrained/netD_e{epoch + 1}.pth")
                torch.save(self.netG, "pretrained/netG_latest.pth")
                torch.save(self.netD, "pretrained/netD_latest.pth")

        if self.save["opt"] is True:
            logger.info(f"Model saved Epochs {epoch+1}")
            torch.save(self.netG, "pretrained/netG_latest.pth")
            torch.save(self.netD, "pretrained/netD_latest.pth")
        
    def _grad_penalty(self, x, y):
        batch_size = x.size()[0]

        alpha = torch.rand((batch_size, 1, 1), requires_grad=True)
        alpha = alpha.expand_as(x).to(self.device)

        # mixed sample from real and fake; make approx of the 'true' gradient norm
        interpolates = alpha * x.data + (1 - alpha) * y.data
        interpolates = interpolates.to(self.device)

        D_interpolates = self.netD(interpolates)
        fake = torch.ones(D_interpolates.size()).to(self.device)

        gradients = torch_grad(
            outputs=D_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = self.gp_weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def _runtime(self, epoch, time):
        mean_time = time / (epoch - self.base_epochs)
        left_epoch = self.iter_epochs - epoch
        done_time = time + mean_time * left_epoch

        runtime = f"{time:4.2f} / {done_time:4.2f} sec "
        return runtime

    def interpolate(self):
        pass
    
    def evaluate(self):
        logger.info("Evaluate the model")
        
        bands = {
            "upper": list(),
            "lower": list()
        }
        
        errD_real = None
        dashboard = Dashboard(self.dataset)
        for i, (data, label) in enumerate(self.dataloader, 0):
            self.optimizerD.zero_grad()
            self.optimizerG.zero_grad()

            x = data.to(self.device)
            shape = (x.size(0), x.size(1), self.in_dim)

            # Train with Fake Data z
            y = self.dataset.get_sample(x, self.netG, shape, self.cond)
            x = self.dataset.fill_missing_value(x, y, label, i)

            Dx = self.netD(x)
            errD_real = self.criterion_adv(Dx, target_is_real=not(True in np.isin(label, [1]) or True in np.isin(label, [-1])))
            errD_real.backward(retain_graph=True)

            Dy = self.netD(y)
            errD_fake = self.criterion_adv(Dy, target_is_real=False)
            errD_fake.backward(retain_graph=True)
            
            errD = errD_fake + errD_real
            self.optimizerD.step()
            
            # if True in np.isin(label, [1]) or True in np.isin(label, [-1]):
            y = self.dataset.get_sample(x, self.netG, shape, self.cond)
            Dy = self.netD(y)

            errG_ = self.criterion_adv(Dy, target_is_real=False)
            errl1 = self.criterion_l1n(y, x) * 10.0
            errl2 = self.criterion_l2n(y, x) * 10.0

            gradients = y - x
            gradients_sqr = torch.square(gradients)
            gradients_sqr_sum = torch.sum(gradients_sqr)
            gradients_l2_norm = torch.sqrt(gradients_sqr_sum)
            gradients_penalty = torch.square(1 - gradients_l2_norm)
            gradients_penalty = gradients_penalty / x.size(0) 
                    
            errG = errG_ + errl1 + errl2 + gradients_penalty

            errG.backward(retain_graph=True)
            self.optimizerG.step()
            
            self.losses["G"] += errG_
            self.losses["D"] += errD
            self.losses["l1"] += errl1
            self.losses["l2"] += errl2
            self.losses["GP"] += gradients_penalty

            if self.print_verbose > 0:
                print(self.print_verbose)
                print(
                    f"[{i + 1:4d}/{len(self.dataloader):4d}] "
                    f"D   {self.losses['D']/(i + 1):.4f} "
                    f"G   {self.losses['G']/(i + 1):.4f} "
                    f"L1  {self.losses['l1']/(i + 1):.3f} "
                    f"L2  {self.losses['l2']/(i + 1):.3f} "
                    f"GP  {self.losses['GP']/(i + 1):.3f} ",
                    end="\r",
                )
            
            
            # dashboard._visualize(
            #     self.dataset.time[i],
            #     self.dataset.denormalize(x.cpu()),
            #     self.dataset.denormalize(y.cpu()),
            # )
            
            dashboard.visualize(
                self.origin[i][0].cpu(),
                x.cpu(),
                y.cpu(),
                label.cpu(),
                cond=self.cond,
                normalize=True,
            )

        pass
        

if __name__ == "__main__":
    # Argument options
    with open("config/config.json") as f:
        config = json.load(f)

    model = BandGAN(config=config)
    try:   
        model.evaluate()
    except AttributeError as e:
        print(e)
        print("Except")
    # model.train()
