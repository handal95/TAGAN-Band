import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


class Dashboard:
    def __init__(self, dataset):
        super(Dashboard).__init__()
        self.dataset = dataset
        self.seq_len = dataset.seq_len

        self.fig, self.ax = self.init_figure()

        self.band_flag = False
        self.bands = {"flag": False, "upper": list(), "lower": list()}

        self.area_upper = None
        self.area_lower = None
        self.time = self.initialize(dataset.time)
        self.scope = 480
        self.idx = 0

        self.detects = list()

    def init_figure(self):
        fig, ax = plt.subplots(figsize=(20, 6), facecolor="lightgray")
        fig.suptitle(self.dataset.title, fontsize=25)
        fig.set_facecolor("lightgray")

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        return fig, ax

    def concat(self, target, x, denormalize=False):
        if denormalize is True:
            x = self.dataset.denormalize(x)

        x = x.detach().numpy()
        if target is None:
            target = x
        else:
            target = np.concatenate((target, x))

        return target

    def initialize(self, value=None):
        if value is None:
            data_list = [list() for x in range(self.seq_len)]
            for i in range(len(data_list)):
                data_list[i] = np.zeros(i)
            return data_list

        return value

    def visualize(self, data, pred, label, bands, detects, pivot):
        def isnormal():
            return not (
                True in np.isin(label[:pivot], [1])
                or True in np.isin(label[:pivot], [-1])
            )

        fig, ax = self.fig, self.ax
        ax.clear()
        ax.grid()
        min_scope = max(self.seq_len, data.size - self.scope + 1)

        length = len(pred)

        for xpos, color in detects["labeled"]:
            if xpos >= min_scope:
                plt.axvline(xpos - min_scope, 0, 1, color=color, linewidth=4, alpha=0.2)

        for xpos, ypos, color in detects["analized"]:
            if xpos >= min_scope:
                plt.scatter(xpos - min_scope, ypos, color=color, s=10)

        base = max(0, length - self.seq_len - min_scope)
        plt.axvspan(
            base + pivot,
            length - min_scope,
            facecolor="lightblue" if isnormal() else "red",
            alpha=0.7,
        )
        plt.axvspan(
            base,
            base + pivot,
            facecolor="green" if isnormal() else "red",
            alpha=0.3,
        )

        length = np.arange(len(bands["upper"][0][min_scope:]))

        for i in range(3):
            ax.fill_between(
                length,
                bands["upper"][i][min_scope:],
                bands["lower"][i][min_scope:],
                color="blue" if i < 2 else "red",
                alpha=(3 - i) / 10,
            )

        ax.plot(data[min_scope:], "r-", linewidth=1, alpha=1, label="data")
        ax.plot(pred[min_scope:], "b-", linewidth=1, alpha=0.2, label="pred")
        ax.plot(
            bands["median"][min_scope:], "k-", linewidth=1, alpha=0.5, label="median"
        )

        xtick = np.arange(0, self.scope, 24)
        values = self.time[min_scope : min_scope + self.scope : 24]

        plt.ylim(self.dataset.min, self.dataset.max)
        plt.xticks(xtick, values, rotation=30)
        plt.legend()

        fig.show()
        fig.canvas.draw()
        fig.canvas.flush_events()
