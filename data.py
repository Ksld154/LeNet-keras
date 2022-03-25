# Data preprocessing

import tensorflow as tf
# import keras
from keras import backend
# from keras import datasets。


class DATA():
    def __init__(self):
        num_classes = 10

        (x_train, y_train), (x_test,
                             y_test) = tf.keras.datasets.mnist.load_data()
        img_rows, img_cols = x_train.shape[1:]

        if backend.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test


class Batch_DATA():
    def __init__(self, batch_size):
        num_classes = 10

        (x_train, y_train), (x_test,
                             y_test) = tf.keras.datasets.mnist.load_data()
        img_rows, img_cols = x_train.shape[1:]

        if backend.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        x_train_batch = []
        y_train_batch = []
        for batch_idx in range(0, int(x_train.shape[0] / batch_size)):
            # print(batch_idx)
            new_batch_x = x_train[(batch_idx *
                                   batch_size):((batch_idx + 1) *
                                                batch_size), :, :, :]
            new_batch_y = y_train[(batch_idx * batch_size):((batch_idx + 1) *
                                                            batch_size), :]
            x_train_batch.append(new_batch_x)
            y_train_batch.append(new_batch_y)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        self.x_train_batch, self.y_train_batch = x_train_batch, y_train_batch


class CIFAR10():
    def __init__(self, batch_size):
        num_classes = 10

        (x_train, y_train), (x_test,
                             y_test) = tf.keras.datasets.cifar10.load_data()
        print(x_train.shape)
        img_rows, img_cols, img_depth = x_train.shape[1:]

        if backend.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], img_depth, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], img_depth, img_rows, img_cols)
            input_shape = (img_depth, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_depth)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_depth)
            input_shape = (img_rows, img_cols, img_depth)

        x_train = x_train.astype('float32')/255.0
        x_test = x_test.astype('float32')/255.0
        # x_train /= 255
        # x_test /= 255

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        x_train_batch = []
        y_train_batch = []
        for batch_idx in range(0, int(x_train.shape[0] / batch_size)):
            # print(batch_idx)
            new_batch_x = x_train[(batch_idx *
                                   batch_size):((batch_idx + 1) *
                                                batch_size), :, :, :]
            new_batch_y = y_train[(batch_idx * batch_size):((batch_idx + 1) *
                                                            batch_size), :]
            x_train_batch.append(new_batch_x)
            y_train_batch.append(new_batch_y)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        self.x_train_batch, self.y_train_batch = x_train_batch, y_train_batch


class CIFAR10_RAW():
    def __init__(self):
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        print(x_train.shape)
        img_rows, img_cols, img_depth = x_train.shape[1:]

        if backend.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], img_depth, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], img_depth, img_rows, img_cols)
            input_shape = (img_depth, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_depth)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_depth)
            input_shape = (img_rows, img_cols, img_depth)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test