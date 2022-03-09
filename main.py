from data import CIFAR10
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
# import tensorflow as tf
# from tensorflow.keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# BATCH_SIZE = 64
# PRE_EPOCHS = 2
# TENSOR_TRANSMISSION_TIME = 30
# FREEZE_OPTIONS = [0, 2, 4, 6, 7]
# MOVING_AVERAGE_WINDOW_SIZE = 7

# LOSS_DIFF_THRESHOLD = 0.05
# INITIAL_FREEZE_LAYERS = 3


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
                        default=False,
                        dest='utility_flag',
                        help='Use model utility to decide switch model or not (default: %(default)s)',
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
                      epochs, switch_model_flag, utility_flag):

        print(f'Overlap: {transmission_overlap}')
        print(f'Dry_run: {dry_run}')
        print(f'Total epochs: {epochs}')
        print(f'Switch model: {switch_model_flag}')
        print(f'Utility: {utility_flag}')


        self.epochs = epochs
        self.base_freeze_idx = 0
        self.next_freeze_idx = INITIAL_FREEZE_LAYERS
        both_converged = False

        primary_trainer, secondary_trainer = self.setup_and_pretrain(
            dry_run, switch_model_flag)

        # In each training epochs
        for e in range(self.epochs):
            print(f'[Epoch {(e+1)}/{epochs}] Base freeze layers: {self.base_freeze_idx}')
            print(f'[Epoch {(e+1)}/{epochs}] Next freeze layers: {self.next_freeze_idx}')

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
            
            self.layer_dicisions.append(self.base_freeze_idx)
            self.loss_diff.append(loss_2 - loss_1)
            diff_ma = utils.moving_average(self.loss_diff, MOVING_AVERAGE_WINDOW_SIZE)
            print(f'Loss Difference: {diff_ma}')

            # primary_frozen_params, primary_frozen_ratio = primary_trainer.get_frozen_ratio()
            # secondary_frozen_params, secondary_frozen_ratio = secondary_trainer.get_frozen_ratio()
            primary_utility = primary_trainer.get_model_utility(self.base_freeze_idx+1)
            secondary_utility = secondary_trainer.get_model_utility(self.next_freeze_idx+1)
            primary_trainer.utility.append(primary_utility)
            secondary_trainer.utility.append(secondary_utility)
            print(f'Primary Utility: {primary_utility}')
            print(f'Secondary Utility: {secondary_utility}')
            
            secondary_utility_avg = utils.moving_average(primary_trainer.utility, MOVING_AVERAGE_WINDOW_SIZE)
            secondary_utility_avg = utils.moving_average(secondary_utility, MOVING_AVERAGE_WINDOW_SIZE)


            if primary_trainer.is_coverged(self.pre_epochs) and secondary_trainer.is_coverged(self.pre_epochs):
                print('*** Both model are converged! ***')
                both_converged = True
            
            # Switch to new model
            if switch_model_flag and both_converged :
                # boundary check
                if self.next_freeze_idx >= len(FREEZE_OPTIONS) - 1 or self.base_freeze_idx >= len(FREEZE_OPTIONS) - 1:
                    continue
                
                # Use utility function to decide switch model or not
                if utility_flag and secondary_utility_avg < secondary_utility_avg:
                    print(f'Secondary model has better avg. utility: {secondary_utility_avg}')
                    print('copy model#2 to model#1')

                    self.base_freeze_idx  = self.next_freeze_idx
                    primary_trainer, secondary_trainer = self.switch_model_reverse(secondary_trainer, primary_trainer)

                if not utility_flag and not np.isnan(diff_ma) and diff_ma >= LOSS_DIFF_THRESHOLD:
                    print(f'Loss Diff.: {loss_2-loss_1}, is bigger than threshold, which means model#2 is bad')
                    print(f'Loss Diff.: {loss_2-loss_1}, we will copy model#1 to model#2')

                    self.loss_diff.clear()
                    if self.next_freeze_idx >= len(FREEZE_OPTIONS) - 1 or self.base_freeze_idx >= len(FREEZE_OPTIONS) - 1:
                        continue

                    self.next_freeze_idx  = self.base_freeze_idx
                    primary_trainer, secondary_trainer = self.switch_model(primary_trainer, secondary_trainer)

        print(self.layer_dicisions)
        print(primary_trainer.accuracy)
        print(primary_trainer.loss)
        print(secondary_trainer.accuracy)
        print(secondary_trainer.loss)

        self.t1 = primary_trainer
        self.t2 = secondary_trainer

    def setup_and_pretrain(self, dry_run, switch_model_flag):
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
        
        return primary_trainer, secondary_trainer

    

    def switch_model(self, primary_trainer, secondary_trainer):
        # Assign New "Secondary model" to current primary model
        primary_weights = primary_trainer.get_model().get_weights()
        new_secondary_model = keras.models.clone_model(primary_trainer.get_model())
        new_secondary_model.set_weights(primary_weights)
        new_secondary_model._name = "Secondary"
        new_secondary_trainer = Trainer(
            model=new_secondary_model,
            data=self.data,
            freeze_layers=FREEZE_OPTIONS[self.next_freeze_idx],
            recompile=True,
            old_obj=secondary_trainer)

        # New primary model == current "primary model" + 1 more frozen layer
        self.base_freeze_idx += 1
        primary_trainer.get_model()._name = "Primary"
        new_primary_trainer = Trainer(
            model=primary_trainer.get_model(),
            data=self.data,
            freeze_layers=FREEZE_OPTIONS[self.base_freeze_idx],
            recompile=True,
            old_obj=primary_trainer)
        
        return new_primary_trainer, new_secondary_trainer

    
    # copy src_trainer to dst_trainer, since src is better than dst
    def switch_model_reverse(self, src_trainer, dst_trainer):
        
        # New "src model" = current dst_model
        primary_weights = src_trainer.get_model().get_weights()
        new_dst_model = keras.models.clone_model(src_trainer.get_model())
        new_dst_model.set_weights(primary_weights)
        new_dst_model._name = "Primary"
        new_dst_trainer = Trainer(
            model=new_dst_model,
            data=self.data,
            freeze_layers=FREEZE_OPTIONS[self.next_freeze_idx],
            recompile=True,
            old_obj=dst_trainer)

        # New src_model == current "src_model"
        src_trainer.get_model()._name = "Secondary"
        new_src_trainer = Trainer(
            model=src_trainer.get_model(),
            data=self.data,
            freeze_layers=FREEZE_OPTIONS[self.base_freeze_idx],
            recompile=True,
            old_obj=src_trainer)
        
        return new_dst_trainer, new_src_trainer

    def plot_figure(self):
        myplot.plot(self.t1.accuracy, self.t2.accuracy, "Accuracy",
                    f"Gradually Freezing Accuracy", 1)
        myplot.plot(self.t1.loss, self.t2.loss, "Loss",
                    f"Gradually Freezing Loss", 2)
        myplot.show()



if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = opt_parser()

    print('*** Start Training! ***')
    start = time.time()
    client_1 = Client()

    client_1.train_process(transmission_overlap=args.transmission_overlap,
                           dry_run=args.dry_run,
                           transmission_time=args.transmission_time,
                           epochs=args.epochs,
                           switch_model_flag=args.switch_model,
                           utility_flag=args.utility_flag)
    end = time.time()
    print(f'Total training time: {datetime.timedelta(seconds= end-start)}')

    client_1.plot_figure()
