import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from model_search_plot import TxtPlot
import ast
import sys
import os

def plot_figure(parsed_txt_data, figure_idx, filename):
    ax1 = plt.figure(figure_idx).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(f"Static Freezing Comparison")
    plt.ylabel("Accuracy")  # y label
    plt.xlabel("Epochs")  # x label
    
    for idx, data in enumerate(parsed_txt_data):
        if idx == 0:
            plt.plot(data, label='Baseline: No Freeze',marker="o", markersize=4)
        elif idx == 1:
            plt.plot(data, label='Static Freeze 1 Layers',marker="o", markersize=4)
        elif idx == 2:
            plt.plot(data, label='Static Freeze 2 Layers',marker="o", markersize=4)
        elif idx == 3:
            plt.plot(data, label='Static Freeze 4 Layers',marker="o", markersize=4)
        elif idx == 4:
            plt.plot(data, label='Our Dynamic Switch',marker="o", markersize=4)
        elif idx == 5:
            plt.plot(data[2:], label='Our Dynamic Switch, approach2',marker="o", markersize=4)
    plt.legend()

    script_dir = os.path.dirname(__file__)
    image_path = os.path.join(script_dir, 'results/', filename.replace('.txt', '.png'))
    plt.savefig(image_path)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('no argument')
        sys.exit()
    
    font = {'size'   : 14}
    plt.rc('font', **font)

    filename = sys.argv[1]
    plot_obj = TxtPlot(filename)
    data = plot_obj.txt_parser()
    plot_figure(data, 1, filename)
    # plot_obj.plot_figure(plot_obj.result, 1)

    # filename2 = ''
    # if 'accuracy' in filename:
    #     filename2 = filename.replace('accuracy', 'loss', 1)
    # elif 'loss' in filename:
    #     filename2 = filename.replace('loss', 'accuracy', 1)
    
    # plot_obj2 = TxtPlot(filename2)
    # plot_obj2.txt_parser()
    # plot_obj2.plot_figure(plot_obj2.result, 2)

    plt.show()