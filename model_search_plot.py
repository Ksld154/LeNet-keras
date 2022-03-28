import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import ast
import sys
import os


loss1 = [1.483873724937439, None, 1.3767938613891602, None, 1.3251079320907593, None, 1.2892528772354126, None, 1.2670772075653076, None, 1.2584422826766968, None, 1.2597230672836304, None, 1.2623604536056519, None, 1.276349663734436, None, 1.304131031036377, None]
loss2 = [1.485884189605713, None, 1.3795236349105835, None, 1.3244383335113525, None, 1.2840849161148071, None, 1.2619096040725708, None, 1.253419280052185, None, 1.2584288120269775, None, 1.261690378189087, None, 1.2759724855422974, None, 1.3038824796676636, None]
loss3 = [1.5096826553344727, None, 1.3802255392074585, None, 1.313450574874878, None, 1.2733384370803833, None, 1.250435471534729, None, 1.241100549697876, None, 1.2416936159133911, None, 1.2403268814086914, None, 1.2503414154052734, None, 1.269147515296936, None]
loss4= [1.5570533275604248, None, 1.4014657735824585, None, 1.3223589658737183, None, 1.276046872138977, None, 1.2419145107269287, None, 1.2234687805175781, None, 1.2180399894714355, None, 1.2185882329940796, None, 1.2255938053131104, None, 1.2396193742752075, None]
loss_primary = [1.568447232246399, 1.444635033607483, 1.382554531097412, 1.3506004810333252, 1.328365683555603, 1.3073208332061768, 1.2871878147125244, 1.2733041048049927, 1.2598755359649658, 1.2514395713806152, 1.2486062049865723, 1.2490280866622925, 1.2579289674758911, 1.256803035736084, 1.2556647062301636, 1.259216547012329, 1.262956142425537, 1.272085428237915, 1.2773239612579346, 1.279876708984375]
loss_secondary = [ 1.4862383604049683, 1.4248169660568237, 1.3818506002426147, 1.3533120155334473, 1.3306816816329956, 1.3179988861083984, 1.3018163442611694, 1.2900973558425903, 1.2797398567199707, 1.2707397937774658, 1.265718936920166, 1.2611618041992188, 1.2528785467147827, 1.2444547414779663, 1.2390484809875488, 1.235079288482666, 1.2319046258926392, 1.2299736738204956, 1.228243350982666, 1.211164116859436]

acc_1 = [0.4767000079154968, None, 0.5151000022888184, None, 0.5386000275611877, None, 0.5476999878883362, None, 0.5616999864578247, None, 0.5687999725341797, None, 0.5753999948501587, None, 0.5787000060081482, None, 0.5741999745368958, None, 0.5745000243186951, None]
acc_2 = [0.47099998593330383, None, 0.5138999819755554, None, 0.5371999740600586, None, 0.550599992275238, None, 0.5598999857902527, None, 0.5673999786376953, None, 0.5723000168800354, None, 0.5774999856948853, None, 0.574400007724762, None, 0.576200008392334, None]
acc_3 = [0.4603999853134155, None, 0.5101000070571899, None, 0.5414000153541565, None, 0.5558000206947327, None, 0.5667999982833862, None, 0.5710999965667725, None, 0.5767999887466431, None, 0.5759000182151794, None, 0.5792999863624573, None, 0.578499972820282, None]
acc_4 = [0.4499000012874603, None, 0.5054000020027161, None, 0.536899983882904, None, 0.5550000071525574, None, 0.5716000199317932, None, 0.579200029373169, None, 0.5827000141143799, None, 0.5831999778747559, None, 0.5860999822616577, None, 0.5863999724388123, None]
acc_primary = [0.4503999948501587, 0.4900999963283539, 0.5116999745368958, 0.5263000130653381, 0.5360999703407288, 0.5401999950408936, 0.5511999726295471, 0.5554999709129333, 0.5634999871253967, 0.5677000284194946, 0.5712000131607056, 0.5741999745368958, 0.5738000273704529, 0.5759000182151794, 0.5774000287055969, 0.5777999758720398, 0.5777000188827515, 0.5784000158309937, 0.5795999765396118, 0.578000009059906]
acc_secondary = [0.47540000081062317, 0.49459999799728394, 0.5076000094413757, 0.5239999890327454, 0.5360000133514404, 0.5414999723434448, 0.5473999977111816, 0.5501999855041504, 0.5551999807357788, 0.5576000213623047, 0.5622000098228455, 0.564300000667572, 0.5656999945640564, 0.5673999786376953, 0.5699999928474426, 0.5715000033378601, 0.5726000070571899, 0.573199987411499, 0.5752000212669373, 0.5806999802589417]


def plot_loss():
    plt.title("Gradually Freezing Loss, brute force search each freezing degree every 2 epochs")
    plt.ylabel("Loss")  # y label
    plt.xlabel("Epochs")  # x label


    plt.plot(loss_primary, label='Primary Model',marker="o", markersize=4)
    plt.plot(loss_secondary, label='Secondary Model',marker="o", markersize=4)
    plt.plot(loss1, label='Freeze 1 layer',marker="o", markersize=4)
    plt.plot(loss2, label='Freeze 2 layer',marker="o", markersize=4)
    plt.plot(loss3, label='Freeze 3 layer',marker="o", markersize=4)
    plt.plot(loss4, label='Freeze 4 layer',marker="o", markersize=4)

    plt.legend()


def plot_accuracy():
    plt.title("Gradually Freezing Accuracy, brute force search each freezing degree every 2 epochs")
    plt.ylabel("Accuracy")  # y label
    plt.xlabel("Epochs")  # x label


    plt.plot(acc_primary, label='Primary Model',marker="o", markersize=4)
    plt.plot(acc_secondary, label='Secondary Model',marker="o", markersize=4)
    plt.plot(acc_1, label='Freeze 1 layer',marker="o", markersize=4)
    plt.plot(acc_2, label='Freeze 2 layer',marker="o", markersize=4)
    plt.plot(acc_3, label='Freeze 3 layer',marker="o", markersize=4)
    plt.plot(acc_4, label='Freeze 4 layer',marker="o", markersize=4)

    plt.legend()


def offline_plot():
    ax1 = plt.figure(1).gca()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plot_loss()

    ax2 = plt.figure(2).gca()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    plot_accuracy()

    plt.show()

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
        
        return self.y_label

    def plot_figure(self, figure_idx):
        ax1 = plt.figure(figure_idx).gca()
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title(f"Gradually Freezing {self.y_label}, brute force search each freezing degree every 2 epochs")
        plt.ylabel(f"{self.y_label}")  # y label
        plt.xlabel("Epochs")  # x label
        
        for idx, data in enumerate( self.result):
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
    # offline_plot()

    if len(sys.argv) < 2:
        print('no argument')
        sys.exit()
    
    font = {
            'size'   : 14}
    plt.rc('font', **font)

    filename = sys.argv[1]
    plot_obj = TxtPlot(filename)
    plot_obj.txt_parser()
    plot_obj.plot_figure(1)


    filename2 = ''
    if 'accuracy' in filename:
        filename2 = filename.replace('accuracy', 'loss', 1)
    elif 'loss' in filename:
        filename2 = filename.replace('loss', 'accuracy', 1)
    
    plot_obj2 = TxtPlot(filename2)
    plot_obj2.txt_parser()
    plot_obj2.plot_figure(2)

    plt.show()