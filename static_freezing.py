from xxlimited import new
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
import copy

import numpy as np
import tensorflow.keras as keras
import tabulate 

import tools.new_plotter

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
    parser.add_argument('--brute-force',
                        default=False,
                        dest='list_all_candidate_models',
                        action=argparse.BooleanOptionalAction,
                        help='List and train every candidate models with different freezing degree (default: %(default)s)')
    parser.add_argument('--freeze-idx',
                        default=0,
                        dest='freeze_idx',
                        type=int,
                        help='Freeze degree (default: %(default)s)')
    
    return parser.parse_args()


class Client():
    def __init__(self) -> None:
        self.data = CIFAR10(BATCH_SIZE)
        self.loss_delta = []

        self.epochs = 10
        self.pre_epochs = 3

        self.all_trainers = []
        self.all_results = []


    def train_process(self, cmd_args):
        self.epochs = cmd_args.epochs

        pretrained_trainer = self.setup_and_pretrain()
        self.train_all_static_freeze_model(pretrained_trainer)
        

    def setup_and_pretrain(self):

        # do pre-training before freezing
        primary_model = LeNet(self.data.input_shape, self.data.num_classes,"Primary")
        primary_trainer = Trainer(primary_model, self.data, 0, True, False)
        primary_trainer.freeze_idx = 0
        
        for e in range(self.pre_epochs):
            print(f'[Pre-Training Epoch {e+1}/{self.pre_epochs}]')
            primary_trainer.train_epoch()
        return primary_trainer


    def train_all_static_freeze_model(self, pretrained_trainer):
        # pretrained_acc = copy.deepcopy(pretrained_trainer.accuracy)
        for i in range(4):

            new_trainer = pretrained_trainer.static_further_freeze(i)
            new_trainer.name = f'Static Freeze: {i} layers'
            new_trainer.accuracy = copy.deepcopy(pretrained_trainer.accuracy)
            new_trainer.model.summary()
            
            self.all_trainers.append(new_trainer)

        all_acc = []
        for t in self.all_trainers:
            print(f'Static freeze degree: {t.freeze_idx}')
            for e in range(self.epochs):
                print(f'[Training Epoch {e+1}/{self.epochs}]')
                t.train_epoch()
            print(t.accuracy)
            all_acc.append(t.accuracy)

        for t in self.all_trainers:
            d = dict(name=t.name, acc=t.accuracy)
            self.all_results.append(d)

        print(all_acc)
        print(self.all_results)


    def print_metrics(self):
        table_header = ['Metric', 'Training time', 'Transmission parameter volume']
        table_data = [
            (self.t1.name, self.t1.total_training_time, self.t1.total_trainable_weights),
        ]
        print(tabulate.tabulate(table_data, headers=table_header, tablefmt='grid'))


    def plot_figure(self):
        tools.new_plotter.multiplot(all_data=self.all_results, 
                y_label='Accuracy', 
                title= f'Static Freeze Accuracy',
                figure_idx=1
            )
        tools.new_plotter.show()


if __name__ == '__main__':
    args = opt_parser()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)

    print('*** Start Training! ***')
    print(f'GPU Device: {args.gpu_device}')
    
    start = time.time()
    client_1 = Client()
    client_1.train_process(cmd_args=args)
    end = time.time()
    print(f'Total training time: {datetime.timedelta(seconds= end-start)}')

    client_1.plot_figure()
