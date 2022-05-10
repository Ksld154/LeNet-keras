import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime
import os

def multiplot(all_data, y_label, title, figure_idx):
    ax1 = plt.figure(num=figure_idx, figsize=(8, 6)).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    # plt.figure(num=figure_idx, figsize=(12, 8))
    plt.title(title)
    plt.ylabel(y_label)   # y label
    plt.xlabel("Epochs")  # x label
    font = {'size'   : 12}
    plt.rc('font', **font)

    for data in all_data:
        # print(data)
        plt.plot(data.get('acc'),
                 label=data.get('name'),
                 marker="o",
                 linestyle="-")
    plt.legend(loc='lower right')

def show():
    plt.show()


def save_figure(filepath:str):
    plt.savefig(filepath)
    # print(filepath)