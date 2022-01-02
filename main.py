# from concurrent import futures
# import threading
from data import CIFAR10
from lenet import LeNet
from transmitter import Transmitter
# from threading import Thread
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import tensorflow as tf
import tensorflow.keras as keras

RESULT_PATH = 'result.txt'

BATCH_SIZE = 64
EPOCHS = 10
LOSS_THRESHOLD = 0.1
PRE_EPOCHS = 2
TENSOR_TRANSMISSION_TIME = 30
FREEZE_OPTIONS = [0, 2, 4, 6, 7]


class Client():
    def __init__(self) -> None:
        self.base_freeze_idx = 0
        self.next_freeze_idx = 1
        self.loss_history = []
        self.accuracy_history = []

    def train_process(self):
        data = CIFAR10(BATCH_SIZE)

        base_model = LeNet(data.input_shape, data.num_classes, "Base")
        base_trainer = Trainer(base_model, data, self.base_freeze_idx, True)
        # next_model = LeNet(data.input_shape, data.num_classes, "Next")
        # next_trainer = Trainer(next_model, data, self.base_freeze_idx, True)

        # do some pre-training before freezing
        for e in range(PRE_EPOCHS):
            print(f'[Pre-Training Epoch {e}]')
            base_trainer.train_epoch()
            # next_trainer.train_epoch()
        # for l in range(self.next_freeze_idx):
        #     next_trainer.get_model().layers[l].trainable = False
        # next_trainer.get_model().summary()
        base_weights = base_trainer.get_model().get_weights()
        next_model = keras.models.clone_model(base_trainer.get_model())
        next_model._name = "Next"
        next_model.set_weights(base_weights)
        next_trainer = Trainer(next_model, data, self.next_freeze_idx, True)

        # next_model = LeNet(data.input_shape, data.num_classes, "Next")
        # base_trainer = Trainer(base_model, data, self.base_freeze_idx, True)
        # next_trainer = Trainer(next_model, data, self.next_freeze_idx, True)
        # base_model.summary()

        # In each epochs
        for e in range(EPOCHS):
            print(f'[Epoch {e}] Base freeze layers: {self.base_freeze_idx}')
            print(f'[Epoch {e}] Next freeze layers: {self.next_freeze_idx}')
            # base_trainer.recompile = False
            # next_trainer.recompile = False

            ## [TODO] suppose to pass base_loss to central server here?
            loss_1, acc_1 = base_trainer.train_epoch()
            print(f'Starting Transmitting tensors...')
            t1 = Transmitter(TENSOR_TRANSMISSION_TIME)
            t1.start()

            # loss_2, acc_2 = next_trainer.train_epoch(False)
            future = ''
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(next_trainer.train_epoch)

            ## [TODO] suppose to receive model update from central around here?
            t1.join()
            print(f'Tensor transmission done !')

            # check if t2 is finish
            if future.done():
                print(future.result())
                loss_2, acc_2 = future.result()
            else:
                # next trainer is not ready, so we will not wait for it and discard it's result
                continue

            # Switch to new model
            if e >= 0 and abs(loss_2 - loss_1) <= LOSS_THRESHOLD:
                print(f'Loss Diff.:{loss_2-loss_1}, use new model')
                self.base_freeze_idx += 1
                self.next_freeze_idx += 1
                if self.next_freeze_idx >= len(
                        FREEZE_OPTIONS) or self.base_freeze_idx >= len(
                            FREEZE_OPTIONS):
                    continue

                next_trainer.get_model()._name = "Base"
                base_trainer = Trainer(
                    model=next_trainer.get_model(),
                    data=data,
                    freeze_layers=FREEZE_OPTIONS[self.base_freeze_idx],
                    recompile=True)

                # [TODO] the weight is not preserved!!!
                base_weights = next_trainer.get_model().get_weights()
                new_next_model = keras.models.clone_model(
                    next_trainer.get_model())
                new_next_model.set_weights(base_weights)
                new_next_model._name = "Next"
                next_trainer = Trainer(
                    model=new_next_model,
                    data=data,
                    freeze_layers=FREEZE_OPTIONS[self.next_freeze_idx],
                    recompile=True)
                # base_trainer.get_model().summary()
                # next_trainer.get_model().summary()

        print(self.loss_history)
        print(self.accuracy_history)


class Trainer():
    def __init__(self, model, data, freeze_layers, recompile) -> None:
        self.model = model
        self.data = data
        self.freeze_layers = freeze_layers
        self.recompile = recompile

        self.loss_history = []
        self.accuracy_history = []

        if self.recompile:
            for l in range(self.freeze_layers):
                self.model.layers[l].trainable = False
            self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                               optimizer='SGD',
                               metrics=['accuracy'])
            self.model.summary()

    def train_epoch(self):
        # train each batch
        for x, y in zip(self.data.x_train_batch, self.data.y_train_batch):
            self.model.train_on_batch(x, y)

        # evaluate the model result for this epoch
        score = self.model.evaluate(self.data.x_test,
                                    self.data.y_test,
                                    batch_size=BATCH_SIZE)
        self.loss_history.append(score[0])
        self.accuracy_history.append(score[1])

        return score[0], score[1]

    def get_model(self):
        return self.model


def train_lenet():
    pass
    # tf.get_logger().setLevel('INFO')

    # # data = Batch_DATA(BATCH_SIZE)
    # data = CIFAR10(BATCH_SIZE)

    # all_loss = []
    # all_acc = []

    # for freeze_layers in FREEZE_OPTIONS:
    #     loss_history = []
    #     accuracy_history = []
    #     freezed = False

    #     model = LeNet(data.input_shape, data.num_classes)
    #     model.summary()

    #     # In each epochs
    #     for e in range(EPOCHS):
    #         print(f'Epoch {e}:')

    #         # In each batch
    #         for x, y in zip(data.x_train_batch, data.y_train_batch):
    #             model.train_on_batch(x, y)
    #         score = model.evaluate(data.x_test,
    #                                data.y_test,
    #                                batch_size=BATCH_SIZE)
    #         loss_history.append(score[0])
    #         accuracy_history.append(score[1])

    #         if e >= 1 and not freezed:
    #             freezed = True

    #             # freeze 1st conv2D layers
    #             for i in range(freeze_layers):
    #                 print(f'Freeze layer: {model.layers[i].name}')
    #                 model.layers[i].trainable = False

    #             model.compile(loss=tf.keras.losses.categorical_crossentropy,
    #                           optimizer='SGD',
    #                           metrics=['accuracy'])

    #     print(loss_history)
    #     print(accuracy_history)
    #     all_loss.append(loss_history)
    #     all_acc.append(accuracy_history)

    # print(all_loss)
    # print(all_acc)

    # with open(RESULT_PATH, 'w+', encoding='utf-8') as f:
    #     f.write(', '.join(str(e) for e in all_loss))
    #     f.write(', '.join(str(e) for e in all_acc))
    #     f.write('********************')


if __name__ == '__main__':
    # train_lenet()

    tf.get_logger().setLevel(logging.WARNING)
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
    tf.autograph.set_verbosity(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').disabled = True

    client_1 = Client()
    client_1.train_process()
