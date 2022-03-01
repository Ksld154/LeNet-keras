# from email.mime import base
# from numpy import gradient
from gc import callbacks
from data import CIFAR10
from lenet import LeNet
from transmitter import Transmitter
# import logging
import os
import datetime
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

import myplot

# RESULT_PATH = 'result.txt'

BATCH_SIZE = 64
PRE_EPOCHS = 2

TENSOR_TRANSMISSION_TIME = 30
FREEZE_OPTIONS = [0, 2, 4, 6, 7]
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


MOVING_AVERAGE_WINDOW_SIZE = 7
LOSS_COVERGED_THRESHOLD = 0.01
LOSS_DIFF_THRESHOLD = 0.05

INITIAL_FREEZE_LAYERS = 3


def opt_parser():
    usage = 'Trains and tests a Gradual layer freezing LeNet-5 model with CIFAR10.'
    parser = argparse.ArgumentParser(description=usage)

    parser.add_argument(
        '-o',
        '--overlap',
        default=True,
        dest='transmission_overlap',
        help='Transmission overlap with next model training (default: %(default)s)',
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
    parser.add_argument('-s',
                        '--switch',
                        default=True,
                        dest='switch_model',
                        help='Enable model switching (default: %(default)s)',
                        action=argparse.BooleanOptionalAction)

    return parser.parse_args()


class Client():
    def __init__(self) -> None:
        self.data = CIFAR10(BATCH_SIZE)
        self.base_freeze_idx = 0
        self.next_freeze_idx = 1
        self.layer_dicisions = []
        self.loss_diff = []

        self.epochs = 10
        self.pre_epochs = PRE_EPOCHS

        self.t1 = ''
        self.t2 = ''

    def train_process(self, transmission_overlap, dry_run, transmission_time,
                      epochs, switch_model):

        print(f'Overlap: {transmission_overlap}')
        print(f'Dry_run: {dry_run}')
        print(f'Total epochs: {epochs}')
        print(f'Switch model: {switch_model}')

        self.epochs = epochs
        self.base_freeze_idx = 0
        self.next_freeze_idx = INITIAL_FREEZE_LAYERS
        both_converged = False

        primary_trainer, secondary_trainer = self.setup_and_pretrain(
            dry_run, switch_model)

        # In each training epochs
        for e in range(self.epochs):
            print(
                f'[Epoch {(e+1)}/{epochs}] Base freeze layers: {self.base_freeze_idx}'
            )
            print(
                f'[Epoch {(e+1)}/{epochs}] Next freeze layers: {self.next_freeze_idx}'
            )

            # pass base_loss to central server?
            loss_1, _ = primary_trainer.train_epoch()

            if transmission_overlap:
                print(f'[BG] Starting Transmitting tensors...')
                t1 = Transmitter(transmission_time)
                t1.start()

                # train target_model on background thread
                future = ''
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(secondary_trainer.train_epoch)

                # Receive model update from central around here?
                t1.join()
                print(f'Tensor transmission done !')

                # check if t2 is finish
                if future.done():
                    print(future.result())
                    loss_2, _ = future.result()
                else:
                    # secondary_trainer is not ready, so we will not wait for it and discard it's result
                    continue

            # tranmission is done after secondary_trainer is trained
            else:
                # train target model first
                loss_2, _ = secondary_trainer.train_epoch()

                # simulate transmission only after target model training is finished
                print(f'[FG] Starting Transmitting tensors...')
                t1 = Transmitter(transmission_time)
                t1.start()
                t1.join()
                print(f'Tensor transmission done !')

            # primary_trainer.save_layer_weights(self.base_freeze_idx)
            # secondary_trainer.save_layer_weights(self.base_freeze_idx)

            self.layer_dicisions.append(self.base_freeze_idx)
            self.loss_diff.append(loss_2 - loss_1)
            diff_ma = self.moving_average(self.loss_diff)
            print(diff_ma)

            if self.is_coverged(primary_trainer) and self.is_coverged(secondary_trainer):
                print('*** Both model are converged! ***')
                both_converged = True

            # Switch to new model
            if switch_model and both_converged and not np.isnan(diff_ma) and diff_ma >= LOSS_DIFF_THRESHOLD:
                self.loss_diff.clear()

                print(
                    f'Loss Diff.:{loss_2-loss_1}, is bigger than threshold, which means model#2 is bad')
                print(
                    f'Loss Diff.: {loss_2-loss_1}, we will copy model\# 1 to model#2')

                if self.next_freeze_idx >= len(FREEZE_OPTIONS) - 1 or self.base_freeze_idx >= len(FREEZE_OPTIONS) - 1:
                    continue

                self.next_freeze_idx  = self.base_freeze_idx

                # Assign New "Secondary model" to current primary model
                primary_weights = primary_trainer.get_model().get_weights()
                new_secondary_model = keras.models.clone_model(primary_trainer.get_model())
                new_secondary_model.set_weights(primary_weights)
                new_secondary_model._name = "Secondary"
                secondary_trainer = Trainer(
                    model=new_secondary_model,
                    data=self.data,
                    freeze_layers=FREEZE_OPTIONS[self.next_freeze_idx],
                    recompile=True,
                    old_obj=secondary_trainer)

                # New primary model model == current "primary model" + 1 more frozen layer
                self.base_freeze_idx += 1
                primary_trainer.get_model()._name = "Primary"
                primary_trainer = Trainer(
                    model=primary_trainer.get_model(),
                    data=self.data,
                    freeze_layers=FREEZE_OPTIONS[self.base_freeze_idx],
                    recompile=True,
                    old_obj=primary_trainer)


        print(self.layer_dicisions)

        print(primary_trainer.accuracy)
        print(primary_trainer.loss)
        print(secondary_trainer.accuracy)
        print(secondary_trainer.loss)

        self.t1 = primary_trainer
        self.t2 = secondary_trainer

    def setup_and_pretrain(self, dry_run, switch_model):
        if dry_run:
            self.pre_epochs = self.epochs

        # do some pre-training before freezing
        primary_model = LeNet(self.data.input_shape, self.data.num_classes,
                              "Primary")
        primary_trainer = Trainer(primary_model, self.data, FREEZE_OPTIONS[self.base_freeze_idx],
                                  True, False)
        for e in range(self.pre_epochs):
            print(f'[Pre-Training Epoch {e+1}/{self.pre_epochs}]')
            primary_trainer.train_epoch()
            primary_trainer.cur_layer_weight.append(None)
        if dry_run:
            print(primary_trainer.accuracy)
            print(primary_trainer.loss)
            return

        # Initailize target_model with all-layers pre-trained base model
        primary_model_weights = primary_trainer.get_model().get_weights()
        secondary_model = keras.models.clone_model(primary_trainer.get_model())
        secondary_model._name = "Secondary"
        secondary_model.set_weights(primary_model_weights)
        secondary_trainer = Trainer(secondary_model, self.data, FREEZE_OPTIONS[self.next_freeze_idx],
                                    True, False)

        for e in range(self.pre_epochs):
            secondary_trainer.accuracy.append(None)
            secondary_trainer.loss.append(None)
            secondary_trainer.loss_delta.append(None)
            secondary_trainer.cur_layer_weight.append(None)

        # if not switch_model:
        #     return primary_trainer, secondary_trainer
        # else:
        #     return secondary_trainer, primary_trainer
        return primary_trainer, secondary_trainer


    def moving_average(self, data):
        W = MOVING_AVERAGE_WINDOW_SIZE
        if len(data) < W:
            return np.nan
        return sum(filter(None, data[-W:])) / W

    def is_coverged(self, trainer):
        # print(trainer.loss_delta[self.pre_epochs+1:])
        delta_ma = self.moving_average(trainer.loss_delta[self.pre_epochs:])
        if not np.isnan(delta_ma) and delta_ma <= LOSS_COVERGED_THRESHOLD:
            return True
        else:
            return False

    def plot_figure(self):
        myplot.plot(self.t1.accuracy, self.t2.accuracy, "Accuracy",
                    f"Gradually Freezing Accuracy", 1)
        myplot.plot(self.t1.loss, self.t2.loss, "Loss",
                    f"Gradually Freezing Loss", 2)
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

            sgd = keras.optimizers.SGD(learning_rate=0.01,
                                       momentum=0.0,
                                       decay=1e-4,
                                       nesterov=False)
            self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                               optimizer=sgd,
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
                                             batch_size=BATCH_SIZE,
                                             )
        lr = K.get_value(self.model.optimizer._decayed_lr(tf.float32))
        # print(f"Learning rate: {lr:.6f}")
        print(f"Learning rate: {lr}")

        self.save_loss_delta(loss)
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        return loss, accuracy

    def save_layer_weights(self, current_layer_idx):
        current_layer_weights = self.model.get_weights()[
            FREEZE_OPTIONS[current_layer_idx]]

        layer_sum = np.sum(current_layer_weights **
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

    print('*** Start Training! ***')
    start = time.time()
    client_1 = Client()

    # if args.all_experiments:
    #     client_1.train_process(transmission_overlap=True,
    #                            dry_run=True,
    #                            transmission_time=args.transmission_time,
    #                            epochs=args.epochs)  # dry run (baseline)
    #     client_1.train_process(transmission_overlap=True,
    #                            dry_run=False,
    #                            transmission_time=args.transmission_time,
    #                            epochs=args.epochs)  # Overlap
    #     client_1.train_process(transmission_overlap=False,
    #                            dry_run=False,
    #                            transmission_time=args.transmission_time,
    #                            epochs=args.epochs)  # Non-Overlap
    # else:
    client_1.train_process(transmission_overlap=args.transmission_overlap,
                           dry_run=args.dry_run,
                           transmission_time=args.transmission_time,
                           epochs=args.epochs,
                           switch_model=args.switch_model)
    end = time.time()
    print(f'Total training time: {datetime.timedelta(seconds= end-start)}')

    client_1.plot_figure()
