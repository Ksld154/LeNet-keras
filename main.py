from numpy import gradient
from data import CIFAR10
from lenet import LeNet
from transmitter import Transmitter
import logging, os
import datetime, time
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as k

# RESULT_PATH = 'result.txt'

BATCH_SIZE = 64
EPOCHS = 5

PRE_EPOCHS = 2
LOSS_THRESHOLD = 0.1
CONSECUTIVE_EPOCH_THRESHOLD = 3

TENSOR_TRANSMISSION_TIME = 30
FREEZE_OPTIONS = [0, 2, 4, 6, 7]


def opt_parser():
    usage = 'Trains and tests a Gradual layer freezing LeNet-5 model with CIFAR10.'
    parser = argparse.ArgumentParser(description=usage)

    parser.add_argument(
        '-o',
        '--overlap',
        default=True,
        dest='transmission_overlap',
        help=
        'Transmission overlap with next model training (default: %(default)s)',
        action=argparse.BooleanOptionalAction)
    parser.add_argument('-d',
                        '--dryrun',
                        default=False,
                        dest='dry_run',
                        help='Only do pre-training (default: %(default)s)',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-a',
                        '--all',
                        default=False,
                        dest='all_experiments',
                        help='Do 3 experiments at once (default: %(default)s)',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument(
        '-t',
        '--transmission-time',
        default=30,
        type=int,
        dest='transmission_time',
        help='Mock tensor transmission time (default: %(default)s)')
    parser.add_argument('-e',
                        '--epochs',
                        default=10,
                        type=int,
                        help='Training epoches (default: %(default))')
    return parser.parse_args()


class Client():
    def __init__(self) -> None:
        self.base_freeze_idx = 0
        self.next_freeze_idx = 1
        self.consecutive_frozen_epochs = 0

        self.base_accuracy = []
        self.base_loss = []
        self.target_accuracy = []
        self.target_loss = []
        self.layer_dicisions = []
        self.data = CIFAR10(BATCH_SIZE)

        self.base_loss_delta = []
        self.target_loss_delta = []
        self.base_weights = []
        self.target_weights = []
        self.base_grads = []
        self.target_grads = []

    def train_process(self, transmission_overlap, dry_run, transmission_time,
                      epochs):
        self.base_freeze_idx = 0
        self.next_freeze_idx = 1
        self.consecutive_frozen_epochs = 0
        self.base_accuracy.clear()
        self.base_loss.clear()
        self.target_accuracy.clear()
        self.target_loss.clear()
        self.layer_dicisions.clear()
        self.base_loss_delta.clear()
        self.target_loss_delta.clear()
        self.base_weights.clear()
        self.target_weights.clear()
        self.base_grads.clear()
        self.target_grads.clear()

        print(f'Overlap: {transmission_overlap}')
        print(f'Dry_run: {dry_run}')
        print(f'Total epochs: {epochs}')

        pre_epochs = PRE_EPOCHS
        if dry_run:
            pre_epochs = 12

        # do some pre-training before freezing
        base_model = LeNet(self.data.input_shape, self.data.num_classes,
                           "Base")
        base_trainer = Trainer(base_model, self.data, self.base_freeze_idx,
                               True)
        for e in range(pre_epochs):
            print(f'[Pre-Training Epoch {e}]')
            loss_1, acc_1 = base_trainer.train_epoch()

            if e == 0:
                self.base_loss_delta.append(None)
            else:
                b_delta_loss = loss_1 - self.base_loss[-1]
            self.base_loss.append(loss_1)
            self.base_accuracy.append(acc_1)
            self.target_loss.append(None)
            self.target_accuracy.append(None)
            self.target_loss_delta.append(None)
            self.base_weights.append(None)
            self.target_weights.append(None)
            # self.base_grads.append(base_trainer.gradients)
            print(base_trainer.gradients)

        if dry_run:
            print(self.base_accuracy)
            print(self.base_loss)
            return

        # Initailize target_model with all-layers pre-trained base model
        base_weights = base_trainer.get_model().get_weights()
        next_model = keras.models.clone_model(base_trainer.get_model())
        next_model._name = "Next"
        next_model.set_weights(base_weights)
        next_trainer = Trainer(next_model, self.data, self.next_freeze_idx,
                               True)

        # In each training epochs
        for e in range(epochs):
            print(f'[Epoch {e}] Base freeze layers: {self.base_freeze_idx}')
            print(f'[Epoch {e}] Next freeze layers: {self.next_freeze_idx}')

            ## pass base_loss to central server?
            loss_1, acc_1 = base_trainer.train_epoch()

            if transmission_overlap:
                print(f'[BG] Starting Transmitting tensors...')
                t1 = Transmitter(transmission_time)
                t1.start()

                # train target_model on background thread
                future = ''
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(next_trainer.train_epoch)

                ## Receive model update from central around here?
                t1.join()
                print(f'Tensor transmission done !')

                # check if t2 is finish
                if future.done():
                    print(future.result())
                    loss_2, acc_2 = future.result()
                else:
                    # next_trainer is not ready, so we will not wait for it and discard it's result
                    continue

            # tranmission is done after next_trainer is trained
            else:
                # train target model first
                loss_2, acc_2 = next_trainer.train_epoch()

                # simulate transmission only after target model training is finished
                print(f'[FG] Starting Transmitting tensors...')
                t1 = Transmitter(transmission_time)
                t1.start()
                t1.join()
                print(f'Tensor transmission done !')

            if self.base_loss[-1] != None:
                b_delta_loss = loss_1 - self.base_loss[-1]
            else:
                t_delta_loss = None

            if self.target_loss[-1] != None:
                t_delta_loss = loss_2 - self.target_loss[-1]
            else:
                t_delta_loss = None

            self.base_loss.append(loss_1)
            self.base_accuracy.append(acc_1)
            self.target_loss.append(loss_2)
            self.target_accuracy.append(acc_2)
            self.layer_dicisions.append(self.base_freeze_idx)
            self.base_loss_delta.append(b_delta_loss)
            self.target_loss_delta.append(t_delta_loss)

            self.save_layer_weights(base_trainer, next_trainer)
            # self.save_gradients(base_trainer, next_trainer, loss_1, loss_2)

            # Switch to new model
            # if e >= 0 and abs(loss_2 - loss_1) <= LOSS_THRESHOLD:
            if e >= 0 and loss_2 - loss_1 <= LOSS_THRESHOLD:

                print(f'Loss Diff.:{loss_2-loss_1}, is lower than threshold')
                self.consecutive_frozen_epochs += 1

                if self.consecutive_frozen_epochs >= 3:
                    print(
                        f'Loss Diff.:{loss_2-loss_1}, threshold satisfied and use new model'
                    )
                    self.consecutive_frozen_epochs = 0

                    if self.next_freeze_idx >= len(
                            FREEZE_OPTIONS) - 1 or self.base_freeze_idx >= len(
                                FREEZE_OPTIONS) - 1:
                        continue

                    self.base_freeze_idx += 1
                    self.next_freeze_idx += 1

                    next_trainer.get_model()._name = "Base"
                    base_trainer = Trainer(
                        model=next_trainer.get_model(),
                        data=self.data,
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
                        data=self.data,
                        freeze_layers=FREEZE_OPTIONS[self.next_freeze_idx],
                        recompile=True)
                    # base_trainer.get_model().summary()
                    # next_trainer.get_model().summary()

        print(self.base_accuracy)
        print(self.target_accuracy)
        print(self.base_loss)
        print(self.target_loss)
        print(self.layer_dicisions)

        print(self.base_loss_delta)
        print(self.target_loss_delta)
        print(self.base_weights)
        print(self.target_weights)
        print(self.base_grads)
        print(self.target_grads)

    def save_layer_weights(self, base_trainer, next_trainer):
        base_layer_weights = base_trainer.get_model().get_weights()[
            FREEZE_OPTIONS[self.base_freeze_idx]]
        next_layer_weights = next_trainer.get_model().get_weights()[
            FREEZE_OPTIONS[self.base_freeze_idx]]

        b_layer_sum = np.sum(base_layer_weights**2)
        t_layer_sum = np.sum(next_layer_weights**2)
        self.base_weights.append(b_layer_sum)
        self.target_weights.append(t_layer_sum)


class Trainer():
    def __init__(self, model, data, freeze_layers, recompile) -> None:
        self.model = model
        self.data = data
        self.freeze_layers = freeze_layers
        self.recompile = recompile

        self.loss_history = []
        self.accuracy_history = []
        self.gradients = []

        if self.recompile:
            for l in range(self.freeze_layers):
                self.model.layers[l].trainable = False
            self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                               optimizer='SGD',
                               metrics=['accuracy'])
            # self.model.summary()

    def train_epoch(self):
        # train each batch
        for x, y in zip(self.data.x_train_batch, self.data.y_train_batch):
            self.model.train_on_batch(x, y)

        score = self.model.evaluate(self.data.x_test,
                                    self.data.y_test,
                                    batch_size=BATCH_SIZE)

        # model.get
        self.loss_history.append(score[0])
        self.accuracy_history.append(score[1])

        return score[0], score[1]

    def get_model(self):
        return self.model


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = opt_parser()

    print(f'*** Start Training! ***')
    start = time.time()
    client_1 = Client()

    if args.all_experiments:
        client_1.train_process(transmission_overlap=True,
                               dry_run=True,
                               transmission_time=args.transmission_time,
                               epochs=args.epochs)  # dry run (baseline)
        client_1.train_process(transmission_overlap=True,
                               dry_run=False,
                               transmission_time=args.transmission_time,
                               epochs=args.epochs)  # Overlap
        client_1.train_process(transmission_overlap=False,
                               dry_run=False,
                               transmission_time=args.transmission_time,
                               epochs=args.epochs)  # Non-Overlap
    else:
        client_1.train_process(transmission_overlap=args.transmission_overlap,
                               dry_run=args.dry_run,
                               transmission_time=args.transmission_time,
                               epochs=args.epochs)

    end = time.time()
    print(f'Total training time: {datetime.timedelta(seconds= end-start)}')
