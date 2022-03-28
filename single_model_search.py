from data import CIFAR10, CIFAR10_RAW
from lenet import LeNet
from transmitter import Transmitter
from trainer import Trainer
import utils
import myplot
from constants import *

import os
import datetime
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow.keras as keras
import tabulate 


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
    parser.add_argument('-u',
                        '--utility',
                        default=True,
                        dest='utility_flag',
                        help='Use model utility to decide switch model or not (default: %(default)s)',
                        action=argparse.BooleanOptionalAction)

    parser.add_argument('-g',
                        '--gpu',
                        default=3,
                        dest='gpu_device',
                        help='Specify which gpu device to use (default: %(default)s)',
                        type=int)
    parser.add_argument('-f',
                        '--force-switch',
                        default=0,
                        dest='force_switch_epoch',
                        type=int,
                        help='Force to switch model at specific epoch (default: %(default)s)')

    return parser.parse_args()


class Client():
    def __init__(self) -> None:
        self.data = CIFAR10(BATCH_SIZE)
        # self.data = CIFAR10_RAW()

        self.layer_dicisions = []
        self.loss_delta = []

        self.epochs = 10
        self.pre_epochs = PRE_EPOCHS

        self.intermediate_trainer_loss = []
        self.intermediate_trainer_accuracy = []

        self.t1 = ''
        self.t2 = ''

    def train_process(self, transmission_overlap, dry_run, transmission_time,
                      epochs, switch_model_flag, utility_flag, force_switch_epoch):

        table_header = ['Flags', 'Status']
        table_data = [
            ('Overlap', transmission_overlap),
            ('Dry_run', dry_run),
            ('Total epochs', epochs),
            ('Switch model', switch_model_flag),
            ('Utility', utility_flag),
            ('Force switch epoch', force_switch_epoch)
        ]
        print(tabulate.tabulate(table_data, headers=table_header, tablefmt='grid'))

        self.epochs = epochs
        both_converged = False

        primary_trainer = self.setup_and_pretrain(dry_run, switch_model_flag)

        # In each training epochs
        for e in range(self.epochs):
            print(f'[Epoch {(e+1)}/{epochs}] Base freeze layers: {primary_trainer.freeze_idx}')
            
            primary_trainer.train_epoch()            

            # tranmission is done after secondary_trainer is trained
            self.generate_and_train_intermediate_trainer(primary_trainer, e)


        print(primary_trainer.accuracy)
        print(primary_trainer.loss)
        # print(primary_trainer.utility)
        
        print(primary_trainer.total_training_time)
        print(primary_trainer.total_trainable_weights)
        
        for d in range(4):
            print(f"Intermediate model results: degree = {d+1}")
            print(self.intermediate_trainer_loss[d])
            print(self.intermediate_trainer_accuracy[d])

    def setup_and_pretrain(self, dry_run, switch_model_flag):

        # do some pre-training before freezing
        primary_model = LeNet(self.data.input_shape, self.data.num_classes, "Primary")
        primary_trainer = Trainer(primary_model, self.data, 0, True, False)
        for e in range(self.pre_epochs):
            print(f'[Pre-Training Epoch {e+1}/{self.pre_epochs}]')
            primary_trainer.train_epoch()
            primary_trainer.cur_layer_weight.append(None)
        
        for _ in range(4):
            self.intermediate_trainer_loss.append([])
            self.intermediate_trainer_accuracy.append([])

        return primary_trainer


    def print_metrics(self):
        table_header = ['Metric', 'Training time', 'Transmission parameter volume']
        table_data = [
            (self.t1.name, self.t1.total_training_time, self.t1.total_trainable_weights),
            (self.t2.name, self.t2.total_training_time, self.t2.total_trainable_weights),
        ]
        print(tabulate.tabulate(table_data, headers=table_header, tablefmt='grid'))

    def plot_figure(self, utility_flag):
        myplot.plot(self.t1.accuracy, self.t2.accuracy, "Accuracy",
                    f"Gradually Freezing Accuracy", 1)
        myplot.save_figure("Gradually Freezing Accuracy")

        myplot.plot(self.t1.loss, self.t2.loss, "Loss",
                    f"Gradually Freezing Loss", 2)
        myplot.save_figure(f"Gradually Freezing Loss")
        
        if utility_flag:
            myplot.plot(self.t1.utility, self.t2.utility, "Utility", "Gradually Freezing Utility", 3)
            myplot.save_figure(f"Gradually Freezing Utility")
        
        myplot.show()

    def generate_and_train_intermediate_trainer(self, primary_trainer, epochs):
        # intermediate_trainer = []
        for degree in range(4):
            frozen_degree = degree+1
            old_weights = primary_trainer.get_model().get_weights()
            new_model = keras.models.clone_model(primary_trainer.get_model())
            new_model.set_weights(old_weights)

            new_trainer = Trainer(new_model, self.data, frozen_degree, True, False)
            new_trainer.set_model_name(f"Frozen_degree_{frozen_degree}")
            print(f'[Epoch {(epochs+1)}/{self.epochs}] Training intermediate model, degree == {frozen_degree}')
            loss, accuracy = new_trainer.train_epoch()
            self.intermediate_trainer_loss[degree].append(loss)
            self.intermediate_trainer_accuracy[degree].append(accuracy)

            # intermediate_trainer.append(new_trainer)
        return


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = opt_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)

    print('*** Start Training! ***')
    print(f'GPU Device: {args.gpu_device}')
    start = time.time()
    client_1 = Client()

    client_1.train_process(transmission_overlap=args.transmission_overlap,
                           dry_run=args.dry_run,
                           transmission_time=args.transmission_time,
                           epochs=args.epochs,
                           switch_model_flag=args.switch_model,
                           utility_flag=args.utility_flag,
                           force_switch_epoch=args.force_switch_epoch)
    end = time.time()
    print(f'Total training time: {datetime.timedelta(seconds= end-start)}')

    # client_1.plot_figure(args.utility_flag)
    # client_1.print_metrics()
