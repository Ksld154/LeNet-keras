import tensorflow as tf


class MobileNetV2(tf.keras.models.Sequential):
    def __init__(self):
        super().__init__()

        self.add(
            tf.keras.applications.MobileNetV2(
                input_shape=None,
                alpha=1.0,
                include_top=True,
                weights="imagenet",
                input_tensor=None,
                pooling=None,
                classes=1000,
                classifier_activation="softmax"))

        self.compile(loss=tf.keras.losses.categorical_crossentropy,
                     optimizer=tf.keras.optimizers.Adam(lr=0.00002),
                     metrics=['accuracy'])

        # self.model = mobile_model
