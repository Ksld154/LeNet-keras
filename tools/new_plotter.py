import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def multiplot(all_data, y_label, title, figure_idx):
    ax1 = plt.figure(figure_idx).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # integer x-axis

    plt.figure(figure_idx)
    plt.title(title)
    plt.ylabel(y_label)   # y label
    plt.xlabel("Epochs")  # x label
    
    for data in all_data:
        # print(data)
        plt.plot(data.get('acc'),
                 label=data.get('name'),
                 marker="o",
                 linestyle="-")
    plt.legend()

def show():
    plt.show()