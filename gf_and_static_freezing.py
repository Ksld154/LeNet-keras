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
import tools.csv_exporter

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
    parser.add_argument(
        '--window_size', type=int, default=5,
        help='Moving average window size for model loss difference (default: %(default)s)')

    parser.add_argument('--static-freeze',
                    default=True,
                    dest='static_freeze',
                    action=argparse.BooleanOptionalAction,
                    help='Train all Static Freeze Models (default: %(default)s)')
    parser.add_argument('--gradually-freeze',
                    default=True,
                    dest='gradually_freeze',
                    action=argparse.BooleanOptionalAction,
                    help='Train Gradually Freeze Models (default: %(default)s)')

    parser.add_argument('--static-freeze-candidates',
                    default=5,
                    dest='static_freeze_candidates',
                    type=int,
                    help='Candidate Static Freeze Degree (default: %(default)s)')
    return parser.parse_args()


class Client():
    def __init__(self) -> None:
        self.data = CIFAR10(BATCH_SIZE)
        self.loss_delta = []

        self.epochs = 10
        self.pre_epochs = 3

        self.all_trainers = []
        self.all_results = []
        self.all_metrics = []

        self.results_dir = None


    def train_process(self, cmd_args):
        self.epochs = cmd_args.epochs
        self.setup_folders()

        # start_time = time.time()
        pretrained_trainer = self.setup_and_pretrain()
        # pretrain_end_time = time.time()
        # pretraining_elapse_time = datetime.timedelta(seconds=pretrain_end_time-start_time)
        # print(pretraining_elapse_time)
        
        gf1 = None
        if cmd_args.gradually_freeze:
            gf1 = GraduallyFreezing(pretrained_trainer=pretrained_trainer)
            primary_results, _ = gf1.train_process(cmd_args)
            self.all_results.append(primary_results)

            # gf_elapse_time = gf1.primary_trainer.total_training_time
            # gf1.total_time = pretraining_elapse_time + gf_elapse_time
            # self.all_results.append(secondary_results)
            # print(gf_elapse_time)
            # gf1.print_metrics()

        if cmd_args.static_freeze:
            self.train_all_static_freeze_model(pretrained_trainer)
            self.all_metrics = self.print_metrics(gf_primary_trainer=gf1.primary_trainer)
        
        # gf1.print_metrics()


    def setup_folders(self):
        base_dir = os.path.dirname(__file__)
        now = datetime.datetime.now()
        dt_string = now.strftime("%m-%d-%Y_%H%M%S")
        results_dir = os.path.join(base_dir, 'results/', dt_string)
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        print(results_dir)

        self.results_dir = results_dir


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
        for i in range(5):

            new_trainer = pretrained_trainer.static_further_freeze(i)
            new_trainer.name = f'Static Freeze: {i} layers'
            if i == 0:
                new_trainer.name = 'Baseline: No Freeze'
            new_trainer.accuracy = copy.deepcopy(pretrained_trainer.accuracy)
            new_trainer.model.summary()
            
            self.all_trainers.append(new_trainer)

        all_acc = []
        for t in self.all_trainers:
            print(f'Static freeze degree: {t.freeze_idx}')
            for e in range(self.epochs):
                print(f'[Training Epoch {e+1}/{self.epochs}]')
                t.train_epoch()
                t.get_frozen_ratio()
            print(t.accuracy)
            all_acc.append(t.accuracy)

        for t in self.all_trainers:
            d = dict(name=t.name, acc=t.accuracy)
            self.all_results.append(d)

        print(all_acc)
        print(self.all_results)

    
    def plot_figure(self):
        tools.new_plotter.multiplot(all_data=self.all_results, 
            y_label='Accuracy', 
            title= f'Static Freeze Accuracy',
            figure_idx=1
        )
        
        png_file = os.path.join(self.results_dir, "Single-Machine Gradually Freezing Accuracy.png")
        print(png_file)
        tools.new_plotter.save_figure(png_file)
        tools.new_plotter.show()

    def print_metrics(self, gf_primary_trainer):
        table_header = ['Model', 'Training time', 'Transmission parameter volume', 'Save Transmission Ratio']
        table_data = []
        total_transmitted_params = self.all_trainers[0].total_trainable_weights
        
        all_metrics = [] # for output csv

        table_data.append(
            (gf_primary_trainer.name, gf_primary_trainer.total_training_time, gf_primary_trainer.total_trainable_weights, 
            f'{ (1 - (gf_primary_trainer.total_trainable_weights/total_transmitted_params) ) * 100} %')
        )
        all_metrics.append(
            dict(
                name=gf_primary_trainer.name, 
                total_training_time=gf_primary_trainer.total_training_time, 
                total_trainable_weights=gf_primary_trainer.total_trainable_weights, 
                save_transmission_ratio=f'{ (1 - (gf_primary_trainer.total_trainable_weights/total_transmitted_params) ) * 100} %'
            )
        )

        # Static Freezing Metrics
        for trainer in self.all_trainers:
            table_data.append(
                (trainer.name, trainer.total_training_time, trainer.total_trainable_weights, 
                f'{ (1-(trainer.total_trainable_weights/total_transmitted_params)) *100 } %')
            )

            all_metrics.append(
                dict(
                    name=trainer.name, 
                    total_training_time=trainer.total_training_time, 
                    total_trainable_weights=trainer.total_trainable_weights, 
                    save_transmission_ratio=f'{ (1 - (trainer.total_trainable_weights/total_transmitted_params) ) * 100} %'
                )
            )

        print(tabulate.tabulate(table_data, headers=table_header, tablefmt='grid'))

        return all_metrics


    def output_csv(self, data, filename, fields):
        csv_file = os.path.join(self.results_dir, filename)
        print(csv_file)
        tools.csv_exporter.export_csv(data=data, filepath=csv_file, fields=fields)



class GraduallyFreezing():
    def __init__(self, pretrained_trainer=None) -> None:
        self.data = CIFAR10(BATCH_SIZE)

        self.layer_dicisions = []
        self.loss_delta = []

        self.epochs = 10
        self.pre_epochs = PRE_EPOCHS

        self.primary_trainer, self.secondary_trainer = self.setup_and_pretrain(pretrained_trainer=pretrained_trainer)
        self.total_time = None



    def train_process(self, cmd_args=None, transmission_overlap=False, dry_run=False, transmission_time=0, switch_model_flag=True):

        table_header = ['Flags', 'Status']
        table_data = [
            ('Overlap', transmission_overlap),
            ('Dry_run', dry_run),
            ('Switch model', switch_model_flag),
            ('Total epochs', cmd_args.epochs),
            ('Window size', cmd_args.window_size),
        ]
        print(tabulate.tabulate(table_data, headers=table_header, tablefmt='grid'))

        self.epochs = cmd_args.epochs
        both_converged = False

        primary_trainer, secondary_trainer = self.primary_trainer, self.secondary_trainer

        # In each training epochs
        for e in range(self.epochs):
        
            secondary_trainer = primary_trainer.generate_further_freeze_trainer(secondary_trainer)
        
            print(f'[Epoch {(e+1)}/{self.epochs}] Base freeze layers: {primary_trainer.freeze_idx}')
            print(f'[Epoch {(e+1)}/{self.epochs}] Next freeze layers: {secondary_trainer.freeze_idx}')
            
            loss_1, _ = primary_trainer.train_epoch()

            # train target model first,
            # simulate transmission after target model training is finished
            loss_2, _ = secondary_trainer.train_epoch()
            # print(f'[FG] Starting Transmitting tensors...')
            # t1 = Transmitter(transmission_time)
            # t1.start()
            # t1.join()
            # print(f'Tensor transmission done !')
            
            self.layer_dicisions.append(primary_trainer.freeze_idx)
            self.loss_delta.append(loss_2 - loss_1)
            models_loss_diff = utils.moving_average(self.loss_delta, cmd_args.window_size)
            print(f'Avg Loss Difference: {models_loss_diff}')

            # Update transmitted parameter amount
            primary_trainer.get_frozen_ratio()
            secondary_trainer.get_frozen_ratio()
        

            if primary_trainer.is_converged(self.pre_epochs, cmd_args.window_size) and secondary_trainer.is_converged(self.pre_epochs, cmd_args.window_size):
                print('*** Both model are converged! ***')
                both_converged = True
        

            # Switch to new model
            if switch_model_flag and both_converged:
                
                # boundary check
                if secondary_trainer.freeze_idx > len(FREEZE_OPTIONS) -1 or primary_trainer.freeze_idx >= len(FREEZE_OPTIONS) -1:
                    continue
                
                # Switch model using Loss difference
                if not np.isnan(models_loss_diff) and models_loss_diff <= LOSS_DIFF_THRESHOLD:
                    print(f'Loss Diff.: {models_loss_diff}, is smaller than threshold, which means model#2 is better')
                    print(f'Loss Diff.: {models_loss_diff}, we will copy model#2 to model#1')

                    self.loss_delta.clear()
                    # Approach 1
                    # secondary_trainer.freeze_idx = primary_trainer.freeze_idx
                    # primary_trainer, _ = self.switch_model_old(primary_trainer, secondary_trainer)
                    
                    # Approach 2
                    primary_trainer = self.switch_model_new(primary_trainer, secondary_trainer)

                    # Approach 3
                    # primary_trainer = primary_trainer.further_freeze(self.pre_epochs, True)
                    # 


        print(self.layer_dicisions)
        print(primary_trainer.accuracy)
        print(secondary_trainer.accuracy)
        print(primary_trainer.loss)
        print(secondary_trainer.loss)
        print(primary_trainer.layer_history[self.pre_epochs:])
        print(secondary_trainer.layer_history)
        
        print(primary_trainer.total_training_time)
        print(secondary_trainer.total_training_time)



        self.primary_trainer = primary_trainer
        self.secondary_trainer = secondary_trainer

        primary_trainer.name = 'Gradually Freezing: Primary Model'
        secondary_trainer.name = "Gradually Freezing: Secondary Model"
        primary_results = dict(name=primary_trainer.name, acc=primary_trainer.accuracy)
        secondary_results = dict(name=secondary_trainer.name, acc=secondary_trainer.accuracy)

        return primary_results, secondary_results


    def setup_and_pretrain(self, dry_run=False, pretrained_trainer=False):
        if dry_run:
            self.pre_epochs = self.epochs
        
        primary_trainer = None
        secondary_trainer = None
        if pretrained_trainer:
            primary_trainer = pretrained_trainer.static_further_freeze(0)
            primary_trainer.name = f'Gradually Freezing: Primary Model'
            primary_trainer.accuracy = copy.deepcopy(pretrained_trainer.accuracy)
            primary_trainer.model.summary()
            
            # Initailize target_model with all-layers pre-trained base model
            secondary_trainer = pretrained_trainer.static_further_freeze(1)
            secondary_trainer.name = "Gradually Freezing: Secondary Model"
            secondary_trainer.accuracy = copy.deepcopy(pretrained_trainer.accuracy)
            secondary_trainer.model.summary()
        
        else:
            # do some pre-training before freezing
            primary_model = LeNet(self.data.input_shape, self.data.num_classes,"Primary")
            primary_trainer = Trainer(primary_model, self.data, 0, True, False)
            for e in range(self.pre_epochs):
                print(f'[Pre-Training Epoch {e+1}/{self.pre_epochs}]')
                primary_trainer.train_epoch()
                # primary_trainer.train_epoch_fit()

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
            secondary_trainer = Trainer(secondary_model, self.data, INITIAL_FREEZE_LAYERS, True, False)

            for e in range(self.pre_epochs):
                secondary_trainer.accuracy.append(None)
                secondary_trainer.loss.append(None)
                secondary_trainer.loss_delta.append(None)
                secondary_trainer.cur_layer_weight.append(None)
            
        self.primary_trainer = primary_trainer
        self.secondary_trainer = secondary_trainer

        # secondary_trainer act as a dummy trainer here
        return primary_trainer, secondary_trainer


    def switch_model_new(self, primary_trainer, secondary_trainer):

        # New primary model == current "secondary model", since secondary_model is better
        # New secondary model will be generated by primary_trainer.generate_further_freeze_trainer() at next epoch
        primary_trainer.freeze_idx = secondary_trainer.freeze_idx

        primary_weights = secondary_trainer.get_model().get_weights()
        new_primary_model = keras.models.clone_model(secondary_trainer.get_model())
        new_primary_model.set_weights(primary_weights)
        new_primary_trainer = Trainer(
            model=new_primary_model,
            data=self.data,
            freeze_idx=primary_trainer.freeze_idx,
            recompile=True,
            old_obj=primary_trainer)
        new_primary_trainer.set_model_name = "Gradually Freezing: Primary Model"
        
        return new_primary_trainer


    def print_metrics(self):
        table_header = ['Model', 'Training time', 'Transmission parameter volume']
        table_data = [
            (self.primary_trainer.name, self.primary_trainer.total_training_time, self.primary_trainer.total_trainable_weights),
            (self.secondary_trainer.name, self.secondary_trainer.total_training_time, self.secondary_trainer.total_trainable_weights),
        ]
        print(tabulate.tabulate(table_data, headers=table_header, tablefmt='grid'))




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

    client_1.output_csv(client_1.all_results, "results.csv", ["name", "acc"])
    client_1.output_csv(client_1.all_metrics, "metrics.csv", ["name", "total_training_time", "total_trainable_weights", "save_transmission_ratio"])

    client_1.plot_figure()
