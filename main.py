from data import CIFAR10
from lenet import LeNet
from transmitter import Transmitter
import logging, os
import datetime, time
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import tensorflow.keras as keras

# RESULT_PATH = 'result.txt'

BATCH_SIZE = 64
EPOCHS = 10

PRE_EPOCHS = 12
LOSS_THRESHOLD = 0.1
CONSECUTIVE_EPOCH_THRESHOLD = 3

TENSOR_TRANSMISSION_TIME = 30
FREEZE_OPTIONS = [0, 2, 4, 6, 7]


class Client():
    def __init__(self) -> None:
        self.base_freeze_idx = 0
        self.next_freeze_idx = 1
        self.consecutive_frozen_epochs = 0
        # self.loss_history = []
        # self.accuracy_history = []

        self.base_accuracy = []
        self.base_loss = []
        self.target_accuracy = []
        self.target_loss = []
        self.layer_dicisions = []

    def train_process(self, dry_run=False, parallel_transmit=False):
        data = CIFAR10(BATCH_SIZE)

        # do some pre-training before freezing
        base_model = LeNet(data.input_shape, data.num_classes, "Base")
        base_trainer = Trainer(base_model, data, self.base_freeze_idx, True)
        for e in range(PRE_EPOCHS):
            print(f'[Pre-Training Epoch {e}]')
            loss_1, acc_1 = base_trainer.train_epoch()
            self.base_loss.append(loss_1)
            self.base_accuracy.append(acc_1)
            self.target_loss.append(None)
            self.target_accuracy.append(None)

        if dry_run:
            print(self.base_accuracy)
            print(self.base_loss)
            return

        # Initailize target_model with all-layers pre-trained base model
        base_weights = base_trainer.get_model().get_weights()
        next_model = keras.models.clone_model(base_trainer.get_model())
        next_model._name = "Next"
        next_model.set_weights(base_weights)
        next_trainer = Trainer(next_model, data, self.next_freeze_idx, True)

        # In each training epochs
        for e in range(EPOCHS):
            print(f'[Epoch {e}] Base freeze layers: {self.base_freeze_idx}')
            print(f'[Epoch {e}] Next freeze layers: {self.next_freeze_idx}')

            ## pass base_loss to central server?
            loss_1, acc_1 = base_trainer.train_epoch()
            print(f'Starting Transmitting tensors...')
            t1 = Transmitter(TENSOR_TRANSMISSION_TIME)
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

            self.base_loss.append(loss_1)
            self.base_accuracy.append(acc_1)
            self.target_loss.append(loss_2)
            self.target_accuracy.append(acc_2)
            self.layer_dicisions.append(self.base_freeze_idx)

            # Switch to new model
            if e >= 0 and abs(loss_2 - loss_1) <= LOSS_THRESHOLD:
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

        # print(self.loss_history)
        # print(self.accuracy_history)
        print(self.base_accuracy)
        print(self.target_accuracy)
        print(self.base_loss)
        print(self.target_loss)
        print(self.layer_dicisions)


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


if __name__ == '__main__':
    # train_lenet()

    tf.get_logger().setLevel(logging.WARNING)
    tf.autograph.set_verbosity(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').disabled = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print(f'*** Start Training! ***')
    start = time.time()
    client_1 = Client()
    # client_1.train_process()
    client_1.train_process(True)

    end = time.time()
    print(f'Total training time: {datetime.timedelta(seconds= end-start)}')
