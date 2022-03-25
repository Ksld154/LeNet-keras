import matplotlib.pyplot as plt
import datetime
import os


def old_plot():
    no_freeze_acc = [
        0.3589000105857849, 0.444599986076355, 0.476500004529953,
        0.49779999256134033, 0.5239999890327454, 0.5419999957084656,
        0.550599992275238, 0.5609999895095825, 0.571399986743927,
        0.5735999941825867, 0.5776000022888184, 0.5795999765396118
    ]

    overlap_acc = [
        0.3537999987602234, 0.4399999976158142, 0.47450000047683716,
        0.49959999322891235, 0.5170000195503235, 0.5224999785423279,
        0.5302000045776367, 0.53329998254776, 0.5376999974250793,
        0.5440000295639038, 0.546500027179718, 0.5551000237464905
    ]
    non_overlap_acc = [
        0.33500000834465027, 0.39469999074935913, 0.43720000982284546,
        0.4796999990940094, 0.5044000148773193, 0.5178999900817871,
        0.5267999768257141, 0.5340999960899353, 0.5383999943733215,
        0.541700005531311, 0.5475999712944031, 0.5561000108718872
    ]

    freezeout_acc = [
        0.28400000000000003, 0.2288, 0.3884000000000001, 0.505, 0.5522,
        0.5307999999999999, 0.5806, 0.5938000000000001, 0.6022000000000001,
        0.6018, 0.5984, 0.6
    ]

    plt.figure(1)
    plt.title('Train LeNet-5 Model on CIFAR10 dataset (pre-train 2 epoches)')
    plt.ylabel("accuracy")  # y label
    plt.xlabel("Epochs")  # x label
    plt.plot(no_freeze_acc,
             label='No freeze (Baseline)',
             marker="o",
             linestyle="-")
    plt.plot(overlap_acc, label='Overlap', marker="o", linestyle="-")
    # plt.plot(non_overlap_acc, label='Non-overlap', marker="o", linestyle="-")
    plt.plot(freezeout_acc, label='FreezeOut', marker="o", linestyle="-")
    plt.legend()

    # offline_loss = [
    #     1.9939610958099365, 1.7598310708999634, 1.6062557697296143,
    #     1.50370454788208, 1.4394506216049194, 1.3845477104187012,
    #     1.3422168493270874, 1.3101344108581543, 1.2837029695510864,
    #     1.269901990890503, 1.251236915588379, 1.243031620979309
    # ]

    # base_loss = [
    #     1.8298, 1.6129, 1.5260257720947266, 1.4595435857772827, 1.3959473371505737,
    #     1.3744759559631348, 1.3438524007797241, 1.3227285146713257,
    #     1.3179458379745483, 1.3047133684158325, 1.2965121269226074,
    #     1.2780370712280273
    # ]

    # target_loss = [
    #     None, None, 1.4997633695602417, 1.4372648000717163, 1.401206612586975,
    #     1.3714802265167236, 1.3524084091186523, 1.3326568603515625,
    #     1.3041801452636719, 1.291694164276123, 1.284850001335144, 1.262576937675476
    # ]

    # plt.figure(2)
    # plt.title('Train LeNet-5 Model on CIFAR10 dataset (pre-train 2 epoches)')
    # plt.ylabel("Loss")  # y label
    # plt.xlabel("Epochs")  # x label
    # plt.plot(offline_loss, label='offline', marker="o", linestyle="-")
    # plt.plot(base_loss, label='base', marker="o", linestyle="-")
    # # plt.plot(target_loss, label='target', marker="o", linestyle="-")
    # plt.legend()

    plt.show()


def plot_diff(base_value, next_value, figure_idx, metric):
    plt.figure(figure_idx)
    plt.title(
        f'Gradually Freezing on LeNet-5 with CIFAR10 dataset metric: {metric}')
    plt.ylabel(metric)  # y label
    plt.xlabel("Epochs")  # x label
    plt.plot(base_value,
             label='Base (Current state)',
             marker="o",
             linestyle="-")
    plt.plot(next_value,
             label='Next (Freeze 1 more)',
             marker="o",
             linestyle="-")
    plt.legend()


baseline_1 = [0.3612000048160553, 0.41749998927116394, 0.4674000144004822, 0.5058000087738037, 0.5267999768257141, 0.5368000268936157, 0.5467000007629395, 0.5541999936103821, 0.5623000264167786, 0.567300021648407, 0.5751000046730042, 0.5777999758720398, 0.5839999914169312, 0.5861999988555908, 0.589900016784668, 0.5931000113487244, 0.5932999849319458]
baseline_2 = [None, None, 0.45739999413490295, 0.47099998593330383, 0.4828999936580658, 0.4896000027656555, 0.4950999915599823, 0.49900001287460327, 0.5019000172615051, 0.5055999755859375, 0.5084999799728394, 0.5116000175476074, 0.5133000016212463, 0.5181999802589417, 0.5198000073432922, 0.521399974822998, 0.5214999914169312]
switch_primary = [0.3172000050544739, 0.3977000117301941, 0.4487999975681305, 0.4747999906539917, 0.4909999966621399, 0.5065000057220459, 0.5253999829292297, 0.5370000004768372, 0.5444999933242798, 0.5497999787330627, 0.557200014591217, 0.5652999877929688, 0.5715000033378601, 0.5723999738693237, 0.578499972820282, 0.5784000158309937, 0.5792999863624573]
switch_secondary = [None, None, 0.43549999594688416, 0.4481000006198883, 0.4578000009059906, 0.46549999713897705, 0.46970000863075256, 0.475600004196167, 0.4790000021457672, 0.5461000204086304, 0.5565000176429749, 0.5638999938964844, 0.5723000168800354, 0.5759999752044678, 0.5796999931335449, 0.5853999853134155, 0.5874000191688538]

baseline_primary_utility = [0.26768045425415044, 0.21448819637298588, 0.17180137634277348, 0.13456492424011235, 0.09925751686096196, 0.07878153324127202, 0.05484144687652592, 0.036115932464599654, 0.02240507602691655, 0.012001681327819869, 0.0016428709030151811, 1e-06, 1e-06, 1e-06, 1e-06]
baseline_secondary_utility = [0.06941218376159669, 0.060230624675750744, 0.05437732934951783, 0.05000625252723695, 0.04614434838294984, 0.043193531036376964, 0.04075388312339784, 0.038535577058792125, 0.03673754334449769, 0.034964632987976085, 0.033387672901153576, 0.031994086503982555, 0.03073103427886964, 0.029638183116912853, 0.028450828790664684]

def offline_plot(idx, title, y_title):
    plt.figure(idx)
    plt.title(title)
    plt.ylabel(y_title)  # y label
    plt.xlabel("Epochs")  # x label

    if idx == 1:
        plt.plot(baseline_1,
                 label='Baseline: No freeze',
                 marker="o",
                 linestyle="--")
        plt.plot(baseline_2, label='Baseline: Freeze first 3 layers',
                 marker="o", linestyle="--")

        plt.plot(switch_primary, label='Switch: Primary',
                 marker="o", linestyle="-")
        plt.plot(switch_secondary, label='Switch: Secondary',
                 marker="o", linestyle="-")
    plt.legend()



def plot(data1, data2, y_label, title, idx):

    plt.figure(idx)
    plt.title(title)
    plt.ylabel(y_label)  # y label
    plt.xlabel("Epochs")  # x label
    
    if idx == 1:
        plt.plot(baseline_1,
                label='Baseline: No freeze',
                marker="o",
                linestyle="--")
        plt.plot(baseline_2, label='Baseline: Freeze first 3 layers',marker="o", linestyle="--")
    # elif idx == 3:
    #     plt.plot(baseline_primary_utility,
    #         label='Baseline: No freeze',
    #         marker="o",
    #         linestyle="--")
    #     plt.plot(baseline_secondary_utility, label='Baseline: Freeze first 3 layers',marker="o", linestyle="--")
    
    plt.plot(data1, label='Primary', marker="o", linestyle="-")
    plt.plot(data2, label='Secondary', marker="o", linestyle="-")
    plt.legend()


def save_figure(title):    
    now = datetime.datetime.now()
    dt_string = now.strftime("%m-%d-%Y_%H:%M:%S")

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'results/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    image_filename = title + "_" + dt_string + ".png"
    plt.savefig(results_dir+image_filename)
    # plt.savefig(f'{title}_{dt_string}.png')


def show():
    plt.show()


if __name__ == '__main__':
    offline_plot(1, "Gradually Freezing w/ Model Switching Accuracy", "Accuracy")

    plt.show()
