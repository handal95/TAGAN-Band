from numpy.lib import math, median
import torch
import numpy as np
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
        
        self.data = None
        self.answer = None
        

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

        self.data = None
        self.answer = None

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
            true_y = data["decode"].to(self.device)
            real_x = data["window"]
            real_y = data["future"]
            
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
            pred_y = self.dataset.denormalize(fake_y.cpu())
            
            pred_y = pred_y * 1.3
            if not training:
                self.data_concat(real_x, real_y, pred_y)
                self.predict(pred_y)
            
            score = self.metric.NMAE(pred_y, real_y, real_test=True).detach().numpy()
            score1 = self.metric.NMAE(pred_y[:, 0], real_y[:, 0]).detach().numpy()
            score2 = self.metric.NMAE(pred_y[:, 6], real_y[:, 6]).detach().numpy()
            score3 = self.metric.NMAE(pred_y[:, 13], real_y[:, 13]).detach().numpy()
            score4 = self.metric.NMAE(pred_y[:, 27], real_y[:, 27]).detach().numpy()
            score_all = self.metric.NMAE(pred_y, real_y).detach().numpy()

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

        # if not training and ((epoch + 1) % self.print_interval) == 0:
        #     real_info = real_y[0, 0, :].cpu().detach().numpy()
        #     pred_info = pred_y[0, 0, :].cpu().detach().numpy()
        #     diff_info = real_info - pred_info
        #     results = pd.DataFrame(
        #         {
        #             "real": real_info,
        #             "pred": pred_info,
        #             "diff": diff_info,
        #             "perc": 100 * diff_info / real_info,
        #         },
        #         index=self.dataset.targets,
        #     )
            # logger.info(
            #     f"-----  Result (e{epoch + 1:4d}) +1 day -----"
            #     f"\n{results.T}\n"
            #     f"Sum(Diff/True) {abs(sum(results['diff'])):.2f}/{sum(results['real']):.2f}"
            #     f"({sum(abs(results['diff']))/sum(results['real'])*100:.2f}%)"
            # )
        
        # for f in range(21):
        #     data = self.answer[:, :, f]
        #     data[data == 0] = np.nan

        #     predict_mean = np.nanmean(data, axis=1).reshape((-1, 1))
        #     predict_median = np.nanmedian(data, axis=1).reshape((-1, 1))
            
        #     truths = self.data["true"][:, f:f+1].detach().numpy()
            
        #     true = self.data["true"][:, f:f+1]

        #     mean = torch.tensor(predict_mean)
        #     median = torch.tensor(predict_median)
        #     week1 = torch.tensor(data[:, 6:7])
        #     week2 = torch.tensor(data[:, 13:14])
        #     week4 = torch.tensor(data[:, 27:28])
            
        #     target = torch.where(true != 0)
        #     mean[target] = (true[target] - mean[target]) / true[target] 
        #     median[target] = (true[target] - median[target]) / true[target] 
        #     week1[target] = (true[target] - week1[target]) / true[target] 
        #     week2[target] = (true[target] - week2[target]) / true[target] 
        #     week4[target] = (true[target] - week4[target]) / true[target] 
            
        #     target = torch.where(true == 0)
        #     mean[target] = 0
        #     median[target] = 0
        #     week1[target] = 0
        #     week2[target] = 0
        #     week4[target] = 0

        #     mean = mean.numpy()
        #     median = median.numpy()

        #     predict = self.answer[:, :, f]
        #     predict = np.concatenate([week4, predict], axis=1)
        #     predict = np.concatenate([week2, predict], axis=1)
        #     predict = np.concatenate([week1, predict], axis=1)
        #     predict = np.concatenate([median, predict], axis=1)
        #     predict = np.concatenate([mean, predict], axis=1)

        #     predict = np.concatenate([truths, predict], axis=1)
        #     predict_df = pd.DataFrame(predict)
        #     print(predict_df)
        
        if not training:
            self.result(real_y)

        return losses["Score"] / (i + 1)

    def data_concat(self, real, true, pred):
        batch_size = pred.shape[0]
        window_size = pred.shape[1]
        future_size = pred.shape[1]
        feature_dim = pred.shape[2]

        real = real.reshape((-1, window_size, feature_dim))
        true = true.reshape((-1, future_size, feature_dim))
        pred = pred.reshape((-1, future_size, feature_dim))
        
        if self.data is None:
            empty = torch.empty(pred.shape)
            self.data = {
                "real": torch.cat([real[0, :-1], real[:, -1]]),
                "true": torch.cat([true[0, :-1], true[:, -1]]),
                "pred": torch.cat([empty, pred])
            }
            return

        self.data["real"] = torch.cat([self.data["real"], real[:, -1]])
        self.data["true"] = torch.cat([self.data["true"], true[:, -1]])
        self.data["pred"] = torch.cat([self.data["pred"], pred])
        
    def predict(self, pred_y):
        batch_size = pred_y.shape[0]
        future_size = pred_y.shape[1]
        feature_dim = pred_y.shape[2]

        preds = np.concatenate([
            self.data["pred"][-batch_size-future_size:].detach().numpy(), 
            np.zeros((batch_size, future_size, feature_dim))
        ])
        
        if self.answer is None:
            self.answer = np.zeros((batch_size + future_size - 1, future_size, feature_dim))
            self.index = 0
        else:
            self.answer = np.concatenate(
                [self.answer, np.zeros((batch_size, future_size, feature_dim))], 
            )

        for f in range(future_size):
            self.answer[self.index + f: self.index + f + batch_size, f] = preds[batch_size:2*batch_size, f]

        self.index += batch_size

    def result(self, y_pred):
        #### Result
        
        results = None
        for f in range(y_pred.shape[2]):
            data = self.answer[:, :, f]
            data[data == 0] = np.nan

            predict_mean = np.nanmean(data, axis=1).reshape((-1, 1))
            predict_median = np.nanmedian(data, axis=1).reshape((-1, 1))

            # predict_mean_3d = np.nanmean(data[:, 3:], axis=1).reshape((-1, 1))
            # predict_median_3d = np.nanmedian(data[:, 3:], axis=1).reshape((-1, 1))
            
            truths = self.data["true"][:, f:f+1].detach().numpy()
            
            true = self.data["true"][:, f:f+1]

            mean = torch.tensor(predict_mean)
            median = torch.tensor(predict_median)
            week1 = torch.tensor(data[:, 6:7])
            week2 = torch.tensor(data[:, 13:14])
            week4 = torch.tensor(data[:, 27:28])
            # mean3d = torch.tensor(predict_mean_3d)
            # median3d = torch.tensor(predict_median_3d)
            
            target = torch.where(true != 0)
            mean[target] = (true[target] - mean[target]) / true[target] 
            median[target] = (true[target] - median[target]) / true[target] 
            week1[target] = (true[target] - week1[target]) / true[target] 
            week2[target] = (true[target] - week2[target]) / true[target] 
            week4[target] = (true[target] - week4[target]) / true[target] 

            # mean3d[target] = (true[target] - mean3d[target]) / true[target] 
            # median3d[target] = (true[target] - median3d[target]) / true[target] 
            
            target = torch.where(true == 0)
            mean[target] = 0
            median[target] = 0
            week1[target] = 0
            week1[target] = 0
            week2[target] = 0
            week4[target] = 0

            mean_score  = torch.sum(torch.abs(mean))/torch.count_nonzero(true)
            median_score= torch.sum(torch.abs(median))/torch.count_nonzero(true)
            week1_score = torch.sum(torch.abs(week1[~torch.any(week1.isnan(),dim=1)]))/torch.count_nonzero(true[~torch.any(week1.isnan(),dim=1)])
            week2_score = torch.sum(torch.abs(week2[~torch.any(week2.isnan(),dim=1)]))/torch.count_nonzero(true[~torch.any(week2.isnan(),dim=1)])
            week4_score = torch.sum(torch.abs(week4[~torch.any(week4.isnan(),dim=1)]))/torch.count_nonzero(true[~torch.any(week4.isnan(),dim=1)])
            
            _mean_score  = torch.sum(mean)/torch.count_nonzero(true)
            _median_score= torch.sum(median)/torch.count_nonzero(true)
            _week1_score = torch.sum(week1[~torch.any(week1.isnan(),dim=1)])/torch.count_nonzero(true[~torch.any(week1.isnan(),dim=1)])
            _week2_score = torch.sum(week2[~torch.any(week2.isnan(),dim=1)])/torch.count_nonzero(true[~torch.any(week2.isnan(),dim=1)])
            _week4_score = torch.sum(week4[~torch.any(week4.isnan(),dim=1)])/torch.count_nonzero(true[~torch.any(week4.isnan(),dim=1)])

            result = np.array([[
                mean_score,
                median_score,
                (week1_score+week2_score+week4_score)/3, 
                week1_score,
                week2_score,
                week4_score
            ]])

            _result = np.array([[
                _mean_score,
                _median_score,
                (_week1_score+_week2_score+_week4_score)/3, 
                _week1_score,
                _week2_score,
                _week4_score,
            ]])
            
            result = result.reshape(-1, 1)
            _result = _result.reshape(-1, 1)

            result = np.concatenate([result, _result], axis=1)
            if results is None:
                results = pd.DataFrame(result)
            else:
                results = pd.DataFrame(np.concatenate([results, result], axis=1))
        
        mean = results.mean(axis=1).values.reshape((-1, 1))
        results = pd.DataFrame(
            np.concatenate([results, mean], axis=1)
        )
        results = results.set_axis(['Mean', 'Median','Week124', 'Week1', 'Week2', 'Week4'], axis=0)
        logger.info(results)
            
            # mean = mean.numpy()
            # median = median.numpy()
            # # Log options
            # predict = self.answer[:, :, f]
            # predict = np.concatenate([week4, predict], axis=1)
            # predict = np.concatenate([week2, predict], axis=1)
            # predict = np.concatenate([week1, predict], axis=1)
            # predict = np.concatenate([median, predict], axis=1)
            # predict = np.concatenate([mean, predict], axis=1)

            # predict = np.concatenate([truths, predict], axis=1)
            # predict_df = pd.DataFrame(predict)
            # print(predict_df[predict_df.shape[0] // 2 : predict_df.shape[0] // 2 + 30])
            
        # predicted_value = np.median(self.answer, axis=1)
        
        # print(self.data["true"].shape)
        # print(predicted_value.shape)
        # print(df.shape)
        # print(df.head(60))
        # print(df.tail(60))
            
        # temp_torch = torch.empty(self.data["pred"].shape)
        # pred_index = batch_size + future_size
        # print(self.data["pred"][:, 0:2, 0])
        # for f in range(future_size):
        #     idx_s = f
        #     print(self.data["pred"][f + batch_size :f + batch_size + future_size, f, 0])
        #     temp_torch[f:f + batch_size, f, :] = self.data["pred"][f + batch_size:f + 2 * batch_size, f, :]
        #     input()

        # df = pd.DataFrame(temp_torch[:, :, 0].detach().numpy())
        # pred_mean = df[df > 0].mean(axis=1)
        # print(pred_mean.T)

        # input()

        # mean_df = np.empty((batch_size, future_size, feature_dim))
        # for b in range(batch_size):
        #     idx_s = epoch * batch_size + b
        #     mean_df[b, :, 0] = pred_mean[idx_s:idx_s + future_size]
        
        # real_data = pd.DataFrame(self.data["true"][:, 0].detach().numpy())
        # print(real_data.shape)
        # print(mean_df.shape)
        # prediction_df = pd.DataFrame(
        #     np.concatenate([
        #         real_data[epoch * batch_size: (epoch + 1)* batch_size],
        #         mean_df[:, :, 0]
        #     ], axis=1)
        # )
        # print(prediction_df)
        
        # print("======")
        
        # prediction_df = pd.DataFrame(
        #     np.concatenate([
        #         real_data[epoch * batch_size: (epoch + 1)* batch_size],
        #         pred_y[:, :, 0].detach().numpy()
        #     ], axis=1)
        # )

        # print(prediction_df)
        
        # print()
        # input()
        # pred_median = df[df > 0].median(axis=1)

        # diff_mean = (real_data - pred_mean).abs()
        # # diff_median = (real_data - pred_median).abs()

        # diff_mean = diff_mean.fillna(0)
        # # diff_median = diff_median.fillna(0)

        # df = pd.concat([diff_mean, df], axis=1)
        # # df = pd.concat([diff_median, df], axis=1)

        # df = pd.concat([pred_mean, df], axis=1)
        # df = pd.concat([pred_median, df], axis=1)

        # df = pd.concat([real_data, df], axis=1)
        # print(df.head(29))


    # def plot_score(self, train_scores, valid_scores):
    #     plt.plot(train_scores, label="train_score")
    #     plt.plot(valid_scores, label="valid_score")
    #     plt.xlabel("epoch")
    #     plt.ylabel("score(nmae)")
    #     plt.title("score_plot")
    #     plt.legend()
    #     plt.show()


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
