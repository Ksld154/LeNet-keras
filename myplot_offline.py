import matplotlib.pyplot as plt

all_loss = [[
    1.8523476123809814, 1.631091833114624, 1.5088235139846802,
    1.4291595220565796, 1.35450279712677, 1.3098115921020508,
    1.273697853088379, 1.2460038661956787, 1.229794979095459,
    1.2171369791030884
],
            [
                1.8670097589492798, 1.6381103992462158, 1.5078295469284058,
                1.4611468315124512, 1.4262396097183228, 1.3999686241149902,
                1.3774938583374023, 1.3576470613479614, 1.339961051940918,
                1.3256330490112305
            ],
            [
                1.9296618700027466, 1.713958978652954, 1.4992014169692993,
                1.4499967098236084, 1.413470983505249, 1.388793706893921,
                1.368943214416504, 1.3495736122131348, 1.3341377973556519,
                1.3212769031524658
            ],
            [
                1.8397067785263062, 1.6096609830856323, 1.4817785024642944,
                1.4471687078475952, 1.4230316877365112, 1.4044910669326782,
                1.3887920379638672, 1.3766429424285889, 1.3658781051635742,
                1.3560066223144531
            ],
            [
                1.8725837469100952, 1.6218891143798828, 1.5520951747894287,
                1.5457165241241455, 1.5417664051055908, 1.538739800453186,
                1.5362632274627686, 1.5341098308563232, 1.5322201251983643,
                1.530527949333191
            ]]
all_acc = [[
    0.3578999936580658, 0.41929998993873596, 0.4625999927520752,
    0.4943999946117401, 0.5177000164985657, 0.5322999954223633,
    0.5461999773979187, 0.557699978351593, 0.5641000270843506,
    0.5687000155448914
],
           [
               0.351500004529953, 0.4214000105857849, 0.461899995803833,
               0.48249998688697815, 0.4927999973297119, 0.5030999779701233,
               0.5110999941825867, 0.5205000042915344, 0.527999997138977,
               0.5346999764442444
           ],
           [
               0.32420000433921814, 0.4018999934196472, 0.46239998936653137,
               0.4828000068664551, 0.49900001287460327, 0.5095999836921692,
               0.5174000263214111, 0.5267000198364258, 0.5314000248908997,
               0.5349000096321106
           ],
           [
               0.34850001335144043, 0.4334000051021576, 0.47269999980926514,
               0.4878000020980835, 0.4964999854564667, 0.5027999877929688,
               0.5080999732017517, 0.5127000212669373, 0.5170000195503235,
               0.5218999981880188
           ],
           [
               0.3560999929904938, 0.42899999022483826, 0.4526999890804291,
               0.45239999890327454, 0.4514000117778778, 0.45320001244544983,
               0.45570001006126404, 0.45680001378059387, 0.45739999413490295,
               0.4584999978542328
           ]]
# fig = plt.figure(figsize=10)
# ax = fig.add_subplot(1, 1, 1)

for idx, (loss, acc) in enumerate(zip(all_loss, all_acc)):
    print(loss)
    print(acc)
    label = ''

    if idx == 0:
        label = 'Base model'
    else:
        label = 'Freeze front layers {}'.format(idx)

    plt.figure(1)
    plt.plot(acc, label=label)

    plt.figure(2)
    plt.plot(loss, label=label)

plt.figure(1)
plt.title('Train LeNet-5 Model on CIFAR10 dataset (freeze at epoch 2)')
plt.legend()
plt.ylabel("accuracy")  # y label
plt.xlabel("Epochs")  # x label

plt.figure(2)
plt.title('Train LeNet-5 Model on CIFAR10 dataset (freeze at epoch 2)')
plt.legend()
plt.ylabel("loss")  # y label
plt.xlabel("Epochs")  # x label

plt.show()