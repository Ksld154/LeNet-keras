import matplotlib.pyplot as plt


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


b_loss_diff = [
    None, -0.04492843151092529, -0.28560566902160645, -0.059226274490356445,
    -0.05096626281738281, -0.028076648712158203, -0.024088501930236816,
    -0.021425366401672363, -0.016093015670776367, -0.008991479873657227,
    -0.011537909507751465, -0.012024760246276855
]
b_layer_weight = [
    None, None, 0.013144027921888563, 0.014174765480889214,
    0.015112193425496419, 0.004682443936665853, 0.004876480897267659,
    0.005085810422897339, 0.000799169381459554, 0.0008280698458353679,
    0.0008582903544108073, 0.0002289440125875232
]
n_loss_diff = [
    None, None, None, -0.05121946334838867, -0.0351560115814209,
    -0.04049372673034668, -0.022573232650756836, -0.01764512062072754,
    -0.021270751953125, -0.010033488273620605, -0.00813746452331543,
    -0.019650578498840332
]
n_layer_weight = [
    None, None, 0.011980995602077908, 0.011980995602077908,
    0.011980995602077908, 0.004502254327138265, 0.004502254327138265,
    0.004502254327138265, 0.0007711529731750488, 0.0007711529731750488,
    0.0007711529731750488, 0.00022612115156176322
]


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


def plot(data1, data2, title, idx):

    plt.figure(idx)
    plt.title(title)
    plt.ylabel("accuracy")  # y label
    plt.xlabel("Epochs")  # x label
    plt.plot(data1, label='t1', marker="o", linestyle="-")
    plt.plot(data2, label='t2', marker="o", linestyle="-")
    plt.legend()


def show():
    plt.show()


if __name__ == '__main__':
    plot_diff(b_loss_diff, n_loss_diff, 1, 'Delta Loss')
    plot_diff(b_layer_weight, n_layer_weight, 2,
              'Avg. Current Layer Squared Weight')
    plt.show()
