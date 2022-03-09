from gc import callbacks
import tensorflow.keras as keras
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras import backend as K

# from keras import models, layers, optimizers
# import keras


class LeNet(models.Sequential):
    def __init__(self, input_shape, nb_classes, name):
        super().__init__(name=name)
        # self.name = name

        self.add(
            layers.Conv2D(6,
                          kernel_size=(5, 5),
                          strides=(1, 1),
                          activation='ReLU',
                          input_shape=input_shape,
                          padding="same"))
        self.add(
            layers.AveragePooling2D(pool_size=(2, 2),
                                    strides=(1, 1),
                                    padding='valid',
                                    trainable=False))
        self.add(
            layers.Conv2D(16,
                          kernel_size=(5, 5),
                          strides=(1, 1),
                          activation='ReLU',
                          padding='valid'))
        self.add(
            layers.AveragePooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='valid',
                                    trainable=False))
        self.add(
            layers.Conv2D(120,
                          kernel_size=(5, 5),
                          strides=(1, 1),
                          activation='ReLU',
                          padding='valid'))
        self.add(layers.Flatten(trainable=False))
        self.add(layers.Dense(84, activation='ReLU'))
        self.add(layers.Dense(nb_classes, activation='softmax'))

        sgd = optimizers.SGD(learning_rate=0.01,
                             momentum=0.0,
                             decay=1e-4,
                             nesterov=False)

        self.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=sgd,
                     metrics=['accuracy'])


