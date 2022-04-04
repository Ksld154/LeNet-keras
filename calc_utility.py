from ctypes import util
from model_search_plot import TxtPlot
import matplotlib.pyplot as plt
import sys
from constants import *


class CalcUtility():
    def __init__(self, loss_data) -> None:
        self.result = []
        self.all_loss = loss_data
        self.all_utility = []

    # get utility for all freezing degree
    def get_utility(self):
        self.all_utility.clear()
        for idx, model_loss in enumerate(self.all_loss):
            if idx <= 3:
                model_utility = self.calc_utility(model_loss, idx+1, False)
                # self.all_utility.append(model_utility)
            elif idx == 4:
                model_utility = self.calc_utility(model_loss[2:], -1, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3])
            elif idx == 5:
                model_utility = self.calc_utility(model_loss[2:], -1, [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4])
            self.all_utility.append(model_utility)

                
                
        print(self.all_utility)

    # Calcalate utility for a single model
    def calc_utility(self, single_model_loss, freeze_degree, freeze_degree_list):
        model_utility = []
        for idx, loss in enumerate(single_model_loss):
            if not loss:
                model_utility.append(None)
                continue
           
            model_loss_satisfied = True if loss <= LOSS_THRESHOLD * LOSS_THRESHOLD_ALPHA else False
            print(model_loss_satisfied) 
            
            utility = 0
            if freeze_degree_list:
                if not model_loss_satisfied:
                    utility = (loss-LOSS_THRESHOLD * LOSS_THRESHOLD_ALPHA) ** (1+freeze_degree_list[idx]*MAGIC_ALPHA)
                else:
                    utility = 1e-7 / (freeze_degree_list[idx]+1)
            
            else:
                if not model_loss_satisfied:
                    if freeze_degree != 0:
                        utility = (loss-LOSS_THRESHOLD * LOSS_THRESHOLD_ALPHA) ** (1+freeze_degree*MAGIC_ALPHA)
                    else: 
                        utility = (loss-LOSS_THRESHOLD * LOSS_THRESHOLD_ALPHA)
                else:
                    utility = 1e-7 / freeze_degree
            model_utility.append(utility)

        return model_utility

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('no argument')
        sys.exit()
    font = {'size'   : 14}
    plt.rc('font', **font)

    filename = sys.argv[1]
    plotterObj = TxtPlot(filename)
    loss_data = plotterObj.txt_parser()
    # print(loss_data)

    utilityObj = CalcUtility(loss_data)
    utilityObj.get_utility()
    plotterObj.y_label = 'Utility'
    plotterObj.plot_figure(utilityObj.all_utility, 1)
    plt.show()

