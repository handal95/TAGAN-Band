import time
import torch
from utils.visualize import plt_loss


class BandGAN:
    def __init__(
        self,
        batch_size,
        seq_len,
        in_dim,
        device,
        dataloader,
        dataset,
        netD,
        netG,
        optimD,
        optimG,
        gp_weight=10,
        critic_iter=5,
    ):
        self.critic_iter = critic_iter
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.device = device
        self.dataloader = dataloader
        self.data = dataset
        self.netD = netD
        self.netG = netG
        self.gp_weight = gp_weight
        self.optimD = optimD
        self.optimG = optimG
        self.cond = seq_len // 2
        self.scorepath = "logs/"
        self.losses = {
            "G": [],
            "D": [],
            "GP": [],
            "gradient_norm": [],
            "LR_G": [],
            "LR_D": [],
        }

    def train(self, epochs):
        device = self.device
        runtime = 0
        # dashboard = Dashboard()
        for epoch in range(epochs):
            plot_num = 0
            start_time = time.time()
            for i in range(self.critic_iter):
                y, x = self.data.get_samples(
                    netG=self.netG,
                    shape=(self.batch_size, self.seq_len, self.in_dim),
                    cond=self.cond,
                    device=device,
                )

                Dx = self.netD(x)
                DGz = self.netD(y)

                self.optimD.zero_grad()

                loss_D = DGz.mean() - Dx.mean()
                # + grad_penalty.to(torch.float32)
                loss_D.backward()
                self.optimD.step()

                if i == self.critic_iter - 1:
                    self.losses["D"].append(float(loss_D))

            self.optimG.zero_grad()
            y, x = self.data.get_samples(
                netG=self.netG,
                shape=(self.batch_size, self.seq_len, self.in_dim),
                cond=self.cond,
                device=device,
            )

            Dy = self.netD(y)
            loss_G = -Dy.mean()
            loss_G.backward()

            self.optimG.step()
            self.losses["G"].append(loss_G.item())
            end_time = time.time()
            runtime += end_time - start_time

            # plt_loss(self.losses["G"], self.losses["D"], self.scorepath, plot_num)

            print(
                f"[{epoch}/{epochs}] " f"D  {self.losses['D'][-1]/(i + 1):.4f}",
                f"G  {self.losses['G'][-1]/(i + 1):.4f}",
                f"|| {runtime:.2f}sec",
                end="\r",
            )

            if (epoch + 1) % 100 == 0:
                torch.save(self.netG, f"netG_band_e{epoch + 1}.pth")
                torch.save(self.netD, f"netD_band_e{epoch + 1}.pth")
                torch.save(self.netG, "netG_latest.pth")
                torch.save(self.netD, "netD_latest.pth")
        print()
        torch.save(self.netG, "netG_latest.pth")
        torch.save(self.netD, "netD_latest.pth")
