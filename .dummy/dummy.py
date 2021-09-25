# def check_missing_value(self, data):
#     # TODO : Need Refactoring
#     def timestamp(index=0):
#         return data[self.key][index]

#     data[self.key] = pd.to_datetime(data[self.key])
#     TIMEGAP = timestamp(1) - timestamp(0)

#     missings = list()
#     filled_count = 0
#     for i in range(1, len(data)):
#         if timestamp(i) - timestamp(i - 1) != TIMEGAP:
#             start_time = timestamp(i - 1) + TIMEGAP
#             end_time = timestamp(i) - TIMEGAP

#             missings.append([str(start_time), str(end_time)])

#             # Fill time gap
#             cur_time = start_time
#             while cur_time <= end_time:
#                 filled_count += 1
#                 data = data.append({self.key: cur_time}, ignore_index=True)
#                 cur_time = cur_time + TIMEGAP

#     # Resorting by timestamp
#     logger.info(f"Checking Timegap - ({TIMEGAP}), Filled : {filled_count}")
#     data = data.set_index(self.key).sort_index().reset_index()

#     return data, missings


# # BANDER DUMMY
# class TAGAN_Bander:
#     def variables(self, data):
#         batch_size = data.size(0)
#         data = self._denormalize(data).cpu().detach()

#         if self.mdata is None:
#             self.mdata = data[0, :0, :]

#         for batch in range(batch_size):
#             self.mdata = np.concatenate((self.mdata, data[batch, :1, :]))

#         return self.mdata

#     def single_process(self, data, normalized=True, predict=False):
#         if normalized:
#             if predict is False:
#                 data = self._denormalize(data[:, :, :]).numpy().ravel()
#             else:
#                 data = self._denormalize(data[:, :, :]).numpy().ravel()

#         if predict:
#             self.pred = self.data_concat(self.pred, data, predict=predict)
#             return self.pred
#         else:
#             self.data = self.data_concat(self.data, data, predict=predict)
#             return self.data


#     # def process(self, x, y, label, normalized=True):
#     #     if normalized:
#     #         x = self._denormalize(x)[0].numpy().ravel()
#     #         y = self._denormalize(y)[0].numpy().ravel()
#     #         label = label[0].detach().numpy().ravel()

#     #     if self.bands["init"] is False:
#     #         self.bands["init"] = True
#     #         self.bands["median"] = x[: self.window_len]
#     #         for i in range(3):
#     #             self.bands["upper"][i] = x[: self.window_len]
#     #             self.bands["lower"][i] = x[: self.window_len]

#     #     std = y.std()
#     #     median = np.median(y[self.window_len + 1 :])

#         # self.bands["median"] = np.append(self.bands["median"], median)
#         # for i in range(3):
#         #     self.bands["upper"][i] = np.append(
#         #         self.bands["upper"][i], median + self.sigmas[i] * std
#         #     )
#         #     self.bands["lower"][i] = np.append(
#         #         self.bands["lower"][i], median - self.sigmas[i] * std
#         #     )

#         # self.data = self.data_concat(self.data, x)
#         # self.pred = self.pred_concat(self.pred, y)
#         # self.label = self.pred_concat(self.label, label)
#         # self.detecting(self.data[-1], label)

#         # return self.data, self.pred, label, self.bands, self.detects

#     # def detecting(self, ypos, label):
#     #     # Analized Unlabeled points
#     #     window_len = self.window_len
#     #     xpos = len(self.data) - 1

#     #     if True in np.isin(label[window_len], [MISSING]):
#     #         self.detects["labeled"].append((xpos, "black"))

#     #     elif True in np.isin(label[window_len], [ANOMALY]):
#     #         self.detects["labeled"].append((xpos, "red"))

#     #     for i in range(1, 3):
#     #         color = "red" if i == 2 else "black"
#     #         LABEL = OUTLIER if i == 2 else WARNING
#     #         if ypos < self.bands["lower"][i][-1] or ypos > self.bands["upper"][i][-1]:
#     #             self.detects["analized"].append((xpos, ypos, color))
#     #             self.label[-1] = LABEL

#     # def _denormalize(self, x):
#     #     if self.device != torch.device("cpu"):
#     #         x = x.cpu().detach()
#     #     return self.dataset.denormalize(x)
#     #
#     # bel, window_len):
#     #     # TODO : Need to refactoring
#     #     if self.check_missing(label):
#     #         if self.replace is None:
#     #             self.replace = x
#     #         else:
#     #             self.replace[:, :-1] = self.replace[:, 1:].clone()

#     #         if self.check_missing(label, latest=True):
#     #             y_ = y.cpu().detach().numpy()
#     #             m = np.median(y_[:, :, :], axis=1)
#     #             std = y_[:, window_len:, :].std()
#     #             y = np.random.normal(m, std / 2, y.shape)
#     #             self.replace[:, -1:] = torch.Tensor(y[:, -1:, :])
#     #         else:
#     #             self.replace[:, -1:] = x[:, -1:, :]

#     #         x[0] = self.replace.to(self.device)
#     #     else:
#     #         if self.replace is not None:
#     #             self.replace = None

#     #     return x

#     # def check_missing(self, label, latest=False):
#     #     label_ = label[:, -1] if latest is True else label
#     #     return True in np.isin(label_, [MISSING])

#     # def data_concat(self, target, x, predict):
#     #     if target is None:
#     #         if predict is False:
#     #             return x[-1:]
#     #         return x

#     #     # if predict:
#     #     #     return np.concatenate((target[:], x[:]))

#     #     return np.concatenate((target, x))

#     # def pred_concat(self, target, y):
#     #     if target is None:
#     #         target = y

#     #     length = len(target) - self.dataset.seq_len + self.window_len
#     #     return np.concatenate((target[: length + 1], y[self.window_len :]))


# class TAGAN_MODEL:

#         origin = self.dataset.data
#         pred = self.bander.pred
#         median = self.bander.bands["median"]
#         pred[: len(median)] = median
#         labels = self.bander.label

#         output_path = f"output_{self.dataset.file}.csv"

#         if text:
#             text_label = list()
#             LABELS = {
#                 -1.0: "Imputed",
#                 0.0: "Normal",
#                 1.0: "Labeled-Anomal",
#                 2.0: "Detected-Warning",
#                 3.0: "Detected-Outlier",
#             }

#             for label in labels:
#                 text_label.append(LABELS[label])
#             labels = text_label

#         label_info = pd.DataFrame({"value": origin, "pred": pred, "label": labels})
#         label_info.to_csv(output_path)

#         logger.info(f"Labeling File is saved to {output_path}")

#     def _runtime(self, epoch, time):
#         mean_time = time / (epoch - self.base_epochs)
#         left_epoch = self.iter_epochs - epoch
#         done_time = time + mean_time * left_epoch

#         runtime = f"{time:4.2f} / {done_time:4.2f} sec "
#         return runtime

#     def _loss_message(self, i=None):
#         if i is None:
#             i = len(self.dataset)

#         message = (
#             f"[{i + 1:4d}/{len(self.dataloader):4d}] "
#             f"D   {self.losses['D']/(i + 1):.4f} "
#             f"G   {self.losses['G']/(i + 1):.4f} "
#             f"L1  {self.losses['l1']/(i + 1):.3f} "
#             f"L2  {self.losses['l2']/(i + 1):.3f} "
#             f"GP  {self.losses['GP']/(i + 1):.3f} ",
#         )
#         return message

#     def run(self):
#         logger.info("Evaluate the model")

#         errD_real = None
#         dashboard = Dashboard_v2(self.dataset)
#         for i, (data, label) in enumerate(self.dataloader, 0):
#             self.optimizerD.zero_grad()
#             self.optimizerG.zero_grad()

#             x = data.to(self.device)

#             # Train with Fake Data z
#             y = self.dataset.get_sample(x, self.netG)
#             x = self.dataset.impute_missing_value(x, y, label, self.pivot)

#             Dx = self.netD(x)
#             # imputate = self.bander.check_missing(label, latest=True)
#             errD_real = self.criterion_adv(Dx, target_is_real=True, imputate=imputate)
#             errD_real.backward(retain_graph=True)

#             y = self.dataset.get_sample(x, self.netG)
#             Dy = self.netD(y)
#             errD_fake = self.criterion_adv(Dy, target_is_real=False)
#             errD_fake.backward(retain_graph=True)

#             errD = errD_fake + errD_real
#             self.optimizerD.step()

#             Dy = self.netD(y)
#             err_G = self.criterion_adv(Dy, target_is_real=False)
#             err_l1 = self.l1_gamma * self.criterion_l1n(y, x)
#             err_l2 = self.l2_gamma * self.criterion_l2n(y, x)
#             err_gp = self.gp_weight * self._grad_penalty(y, x)

#             errG = err_G + err_l1 + err_l2 + err_gp
#             errG.backward(retain_graph=True)
#             self.optimizerG.step()

#             self.losses["G"] += err_G
#             self.losses["D"] += errD
#             self.losses["l1"] += err_l1
#             self.losses["l2"] += err_l2
#             self.losses["GP"] += err_gp

#             if self.print_verbose > 0:
#                 print(f"{self._loss_message(i)}", end="\r")

#             (x, y, label, bands, detects) = self.bander.process(x, y, label)

#             if self.visual:
#                 dashboard.visualize(x, y, label, bands, detects, pivot=self.pivot)
