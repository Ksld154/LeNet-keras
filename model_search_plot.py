import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import ast
import sys
import os


class TxtPlot():
    def __init__(self, filename) -> None:
        self.result = []
        self.filename = filename
        self.y_label = ''
    
    def txt_parser(self):
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir,  self.filename)
        print(file_path)

        if 'accuracy' in self.filename:
            self.y_label = 'Accuracy'
        elif 'loss' in self.filename:
            self.y_label = 'Loss'
        print(self.y_label)
        
        with open(file_path, 'r') as file:
            for row in file:
                row = ast.literal_eval(row)
                self.result.append(row)
        # print(self.result)
        return self.result

    def plot_figure(self, parsed_txt_data, figure_idx):
        ax1 = plt.figure(figure_idx).gca()
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title(f"Gradually Freezing {self.y_label}, brute force search each freezing degree every 2 epochs")
        plt.ylabel(f"{self.y_label}")  # y label
        plt.xlabel("Epochs")  # x label
        
        for idx, data in enumerate(parsed_txt_data):
            if idx <= 3:
                plt.plot(data, label=f'Freeze {idx+1} layer',marker="o", markersize=4)
            elif idx == 4:
                plt.plot(data[2:], label='Primary',marker="o", markersize=4)
            elif idx == 5:
                plt.plot(data[2:], label='Secondary',marker="o", markersize=4)
        plt.legend()

        script_dir = os.path.dirname(__file__)
        image_path = os.path.join(script_dir, 'results/', self.filename.replace('.txt', '.png'))
        plt.savefig(image_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('no argument')
        sys.exit()
    
    font = {'size'   : 14}
    plt.rc('font', **font)

    filename = sys.argv[1]
    plot_obj = TxtPlot(filename)
    plot_obj.txt_parser()
    plot_obj.plot_figure(plot_obj.result, 1)

    filename2 = ''
    if 'accuracy' in filename:
        filename2 = filename.replace('accuracy', 'loss', 1)
    elif 'loss' in filename:
        filename2 = filename.replace('loss', 'accuracy', 1)
    
    plot_obj2 = TxtPlot(filename2)
    plot_obj2.txt_parser()
    plot_obj2.plot_figure(plot_obj2.result, 2)

    plt.show()