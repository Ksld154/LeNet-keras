from email.mime import base
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

import myplot
# RESULT_PATH = 'result.txt'

BATCH_SIZE = 64
EPOCHS = 5

PRE_EPOCHS = 2
CONSECUTIVE_EPOCH_THRESHOLD = 3

LOSS_THRESHOLD = 0.05
MOVING_AVERAGE_WINDOW_SIZE = 3

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
        self.data = CIFAR10(BATCH_SIZE)
        self.base_freeze_idx = 0
        self.next_freeze_idx = 1
        self.layer_dicisions = []
        self.loss_diff = []

        self.t1 = ''
        self.t2 = ''

    def train_process(self, transmission_overlap, dry_run, transmission_time,
                      epochs):
        self.base_freeze_idx = 0
        self.next_freeze_idx = 1

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
                               True, False)
        for e in range(pre_epochs):
            print(f'[Pre-Training Epoch {e+1}/{pre_epochs}]')
            loss_1, acc_1 = base_trainer.train_epoch()
            base_trainer.cur_layer_weight.append(None)
        if dry_run:
            print(base_trainer.accuracy)
            print(base_trainer.loss)
            return

        # Initailize target_model with all-layers pre-trained base model
        base_weights = base_trainer.get_model().get_weights()
        next_model = keras.models.clone_model(base_trainer.get_model())
        next_model._name = "Next"
        next_model.set_weights(base_weights)
        next_trainer = Trainer(next_model, self.data, self.next_freeze_idx,
                               True, False)

        for e in range(pre_epochs):
            next_trainer.accuracy.append(None)
            next_trainer.loss.append(None)
            next_trainer.loss_delta.append(None)
            next_trainer.cur_layer_weight.append(None)

        # In each training epochs
        for e in range(epochs):
            print(
                f'[Epoch {(e+1)}/{epochs}] Base freeze layers: {self.base_freeze_idx}'
            )
            print(
                f'[Epoch {(e+1)}/{epochs}] Next freeze layers: {self.next_freeze_idx}'
            )

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

            self.layer_dicisions.append(self.base_freeze_idx)
            base_trainer.save_layer_weights(self.base_freeze_idx)
            next_trainer.save_layer_weights(self.base_freeze_idx)

            self.loss_diff.append(loss_2 - loss_1)
            diff_ma = self.get_moving_average()
            print(diff_ma)

            # Switch to new model
            if not np.isinf(diff_ma) and diff_ma <= LOSS_THRESHOLD:
                self.loss_diff.clear()

                print(f'Loss Diff.:{loss_2-loss_1}, is lower than threshold')
                print(
                    f'Loss Diff.:{loss_2-loss_1}, threshold satisfied and use new model'
                )

                if self.next_freeze_idx >= len(
                        FREEZE_OPTIONS) - 1 or self.base_freeze_idx >= len(
                            FREEZE_OPTIONS) - 1:
                    continue

                self.base_freeze_idx += 1
                self.next_freeze_idx += 1

                # New Base model == current "next model"
                next_trainer.get_model()._name = "Base"
                base_trainer = Trainer(
                    model=next_trainer.get_model(),
                    data=self.data,
                    freeze_layers=FREEZE_OPTIONS[self.base_freeze_idx],
                    recompile=True,
                    old_obj=base_trainer)

                # Setup New "Next model"
                base_weights = next_trainer.get_model().get_weights()
                new_next_model = keras.models.clone_model(
                    next_trainer.get_model())
                new_next_model.set_weights(base_weights)
                new_next_model._name = "Next"
                next_trainer = Trainer(
                    model=new_next_model,
                    data=self.data,
                    freeze_layers=FREEZE_OPTIONS[self.next_freeze_idx],
                    recompile=True,
                    old_obj=next_trainer)

        print(self.layer_dicisions)

        print(base_trainer.accuracy)
        print(base_trainer.loss)
        print(next_trainer.accuracy)
        print(next_trainer.loss)

        self.t1 = base_trainer
        self.t2 = next_trainer

    def get_moving_average(self):
        W = MOVING_AVERAGE_WINDOW_SIZE
        if len(self.loss_diff) < W:
            return np.inf

        return sum(self.loss_diff[-W:]) / W

    def plot_figure(self):
        myplot.plot(self.t1.accuracy, self.t2.accuracy,
                    f"Gradually Freezing Accuracy", 1)
        myplot.plot(self.t1.loss, self.t2.loss, f"Gradually Freezing Loss", 2)
        myplot.show()


class Trainer():
    def __init__(self, model, data, freeze_layers, recompile, old_obj) -> None:
        self.model = model
        self.data = data
        self.freeze_layers = freeze_layers
        self.recompile = recompile

        self.loss = []
        self.loss_delta = []
        self.accuracy = []
        self.cur_layer_weight = []

        if self.recompile:
            for l in range(self.freeze_layers):
                self.model.layers[l].trainable = False
            self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                               optimizer='SGD',
                               metrics=['accuracy'])
            # self.model.summary()
        if old_obj:
            self.loss = old_obj.loss
            self.loss_delta = old_obj.loss_delta
            self.accuracy = old_obj.accuracy
            self.cur_layer_weight = old_obj.cur_layer_weight

    def train_epoch(self):
        # train each batch
        for x, y in zip(self.data.x_train_batch, self.data.y_train_batch):
            self.model.train_on_batch(x, y)

        loss, accuracy = self.model.evaluate(self.data.x_test,
                                             self.data.y_test,
                                             batch_size=BATCH_SIZE)
        self.save_loss_delta(loss)
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        return loss, accuracy

    def save_layer_weights(self, current_layer_idx):
        current_layer_weights = self.model.get_weights()[
            FREEZE_OPTIONS[current_layer_idx]]

        layer_sum = np.sum(current_layer_weights**
                           2) / current_layer_weights.size
        layer_sum2 = np.sum(current_layer_weights) / np.sum(
            current_layer_weights**2)
        self.cur_layer_weight.append(layer_sum2)

    def save_loss_delta(self, cur_loss):
        if not self.loss:
            self.loss_delta.append(None)
            return

        if self.loss[-1] != None:
            loss_delta = cur_loss - self.loss[-1]
        else:
            loss_delta = None
        self.loss_delta.append(loss_delta)

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

    client_1.plot_figure()
