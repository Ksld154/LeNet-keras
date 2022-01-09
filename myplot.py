import matplotlib.pyplot as plt

offline_acc = [
    0.35679998993873596, 0.43630000948905945, 0.476500004529953,
    0.5013999938964844, 0.5218999981880188, 0.5357999801635742,
    0.5454999804496765, 0.5562999844551086, 0.5647000074386597,
    0.574400007724762, 0.5792999863624573, 0.5827000141143799
]

base_acc = [
    0.3706, 0.4386, 0.46560001373291016, 0.48829999566078186,
    0.510699987411499, 0.5133000016212463, 0.5228000283241272,
    0.534600019454956, 0.5360000133514404, 0.5407999753952026,
    0.5454999804496765, 0.5532000064849854
]

target_acc = [
    None, None, 0.47099998593330383, 0.487199991941452, 0.5006999969482422,
    0.5162000060081482, 0.5231000185012817, 0.5311999917030334,
    0.5389999747276306, 0.544700026512146, 0.5486000180244446,
    0.555899977684021
]

plt.figure(1)
plt.title('Train LeNet-5 Model on CIFAR10 dataset (pre-train 2 epoches)')
plt.ylabel("accuracy")  # y label
plt.xlabel("Epochs")  # x label
plt.plot(offline_acc, label='offline', marker="o", linestyle="-")
plt.plot(base_acc, label='base', marker="o", linestyle="-")
plt.plot(target_acc, label='target', marker="o", linestyle="-")
plt.legend()

offline_loss = [
    1.9939610958099365, 1.7598310708999634, 1.6062557697296143,
    1.50370454788208, 1.4394506216049194, 1.3845477104187012,
    1.3422168493270874, 1.3101344108581543, 1.2837029695510864,
    1.269901990890503, 1.251236915588379, 1.243031620979309
]

base_loss = [
    1.8298, 1.6129, 1.5260257720947266, 1.4595435857772827, 1.3959473371505737,
    1.3744759559631348, 1.3438524007797241, 1.3227285146713257,
    1.3179458379745483, 1.3047133684158325, 1.2965121269226074,
    1.2780370712280273
]

target_loss = [
    None, None, 1.4997633695602417, 1.4372648000717163, 1.401206612586975,
    1.3714802265167236, 1.3524084091186523, 1.3326568603515625,
    1.3041801452636719, 1.291694164276123, 1.284850001335144, 1.262576937675476
]
plt.figure(2)
plt.title('Train LeNet-5 Model on CIFAR10 dataset (pre-train 2 epoches)')
plt.ylabel("Loss")  # y label
plt.xlabel("Epochs")  # x label
plt.plot(offline_loss, label='offline', marker="o", linestyle="-")
plt.plot(base_loss, label='base', marker="o", linestyle="-")
plt.plot(target_loss, label='target', marker="o", linestyle="-")
plt.legend()

plt.show()