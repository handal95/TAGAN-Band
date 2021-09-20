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

    def initialize(self, value=None):
        if value is None:
            data_list = [list() for x in range(self.seq_len)]
            for i in range(len(data_list)):
                data_list[i] = np.zeros(i)
            return data_list
        return value

    def reset_figure(self):
        self.ax.clear()
        self.ax.grid()
        return self.fig, self.ax

    def show_figure(self):
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        return self.fig

    def train_vis(self, origins):
        fig, ax = self.reset_figure()

        for i in range(1, 43):
            ax.plot(origins[:, i], alpha=1, label=f"Column{i} data")
        
        # # Fill Background Valid Area
        # plt.fill_between(
        #     (self.dataset.train_idx, self.dataset.valid_idx),
        #     self.dataset.min,
        #     self.dataset.max,
        #     alpha=0.2,
        #     label="Valid Set",
        # )

        # # Set Y limit by min-max
        # plt.ylim(self.dataset.min, self.dataset.max)

        self.show_figure()

    def visualize(self, data, pred, label, bands, detects, pivot):
        fig, ax = self.reset_figure()

        start = max(self.seq_len, data.size - self.scope + 1)

        length = len(pred)

        # Pivot and Predict Area
        base = max(0, length - self.seq_len - start)
        detected = True in np.isin(label[:pivot], [1]) or True in np.isin(
            label[:pivot], [-1]
        )
        pivot_color = "red" if detected else "lightblue"
        preds_color = "red" if detected else "green"
        plt.axvspan(base + pivot, length - start, facecolor=pivot_color, alpha=0.7)
        plt.axvspan(base, base + pivot, facecolor=preds_color, alpha=0.3)

        # Anomalies Line
        for xpos, color in detects["labeled"]:
            if xpos >= start:
                plt.axvline(xpos - start, 0, 1, color=color, linewidth=4, alpha=0.2)

        for xpos, ypos, color in detects["analized"]:
            if xpos >= start:
                plt.scatter(xpos - start, ypos, color=color, s=10)

        # Bands Area
        xscope = np.arange(len(bands["upper"][0][start:]))
        for i in range(3):
            color = "blue" if i < 2 else "red"
            alpha = (3 - i) / 10
            ax.fill_between(
                xscope,
                bands["upper"][i][start:],
                bands["lower"][i][start:],
                color=color,
                alpha=alpha,
            )

        # Data/Predict Line
        ax.plot(data[start:], "r-", alpha=1, label="data")
        ax.plot(pred[start:], "b-", alpha=0.2, label="pred")
        ax.plot(bands["median"][start:], "k-", alpha=0.5, label="median")

        xtick = np.arange(0, self.scope, 24)
        values = self.time[start : start + self.scope : 24]

        plt.ylim(self.dataset.min, self.dataset.max)
        plt.xticks(xtick, values, rotation=30)
        plt.legend()

        self.show_figure()
