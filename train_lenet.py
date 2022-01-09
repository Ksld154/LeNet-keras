from data import CIFAR10
from lenet import LeNet
import logging

import tensorflow as tf

RESULT_PATH = 'result.txt'

BATCH_SIZE = 64
EPOCHS = 10
LOSS_THRESHOLD = 0.1
PRE_EPOCHS = 2
FREEZE_OPTIONS = [0, 2, 4, 6, 7]
# TENSOR_TRANSMISSION_TIME = 30


def train_lenet():
    tf.get_logger().setLevel('INFO')

    # data = Batch_DATA(BATCH_SIZE)
    data = CIFAR10(BATCH_SIZE)

    all_loss = []
    all_acc = []

    for freeze_layers in FREEZE_OPTIONS:
        loss_history = []
        accuracy_history = []
        freezed = False

        model = LeNet(data.input_shape, data.num_classes)
        model.summary()

        # In each epochs
        for e in range(EPOCHS):
            print(f'Epoch {e}:')

            # In each batch
            for x, y in zip(data.x_train_batch, data.y_train_batch):
                model.train_on_batch(x, y)
            score = model.evaluate(data.x_test,
                                   data.y_test,
                                   batch_size=BATCH_SIZE)
            loss_history.append(score[0])
            accuracy_history.append(score[1])

            if e >= 1 and not freezed:
                freezed = True

                # freeze 1st conv2D layers
                for i in range(freeze_layers):
                    print(f'Freeze layer: {model.layers[i].name}')
                    model.layers[i].trainable = False

                model.compile(loss=tf.keras.losses.categorical_crossentropy,
                              optimizer='SGD',
                              metrics=['accuracy'])

        print(loss_history)
        print(accuracy_history)
        all_loss.append(loss_history)
        all_acc.append(accuracy_history)

    print(all_loss)
    print(all_acc)

    with open(RESULT_PATH, 'w+', encoding='utf-8') as f:
        f.write(', '.join(str(e) for e in all_loss))
        f.write(', '.join(str(e) for e in all_acc))
        f.write('********************')