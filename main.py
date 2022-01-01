from keras.utils.layer_utils import print_summary
from data import CIFAR10, DATA, Batch_DATA
from lenet import LeNet
from plot import plot_loss
from mobilenet import MobileNetV2
import tensorflow as tf

# tensorboard = tf.keras.callbacks.TensorBoard(log_dir='/tmp/my_tf_logs',
#                                              histogram_freq=0,
#                                              batch_size=128,
#                                              write_graph=True,
#                                              write_grads=True)

RESULT_PATH = 'result.txt'

BATCH_SIZE = 32
EPOCHS = 10
LOSS_THRESHOLD = 0.3
FREEZE_LAYERS = [0,2,4,6,7]


def main():
    tf.get_logger().setLevel('INFO')

    # data = Batch_DATA(BATCH_SIZE)
    data = CIFAR10(BATCH_SIZE)


    # for l in model.layers:
    #     print("{} {}".format(l.name, l.trainable))

    # for l in model.trainable_weights:
    #     print(l.name)
    # print(data.x_train.shape)
    # print(data.y_train.shape)
    # print(len(data.x_train_batch))

    # loss_history =[]
    # accuracy_history = []
    # freezed = False

    all_loss = []
    all_acc = []

    for freeze_layers in FREEZE_LAYERS:
        loss_history =[]
        accuracy_history = []
        freezed = False
        
        model = LeNet(data.input_shape, data.num_classes)
        model.summary()
        
        # In each epochs
        for e in range(EPOCHS):
            print('Epoch {}:'.format(e))
            
            # In each batch
            for x, y in zip(data.x_train_batch, data.y_train_batch):
                model.train_on_batch(x, y)
            score = model.evaluate(data.x_test, data.y_test, batch_size=BATCH_SIZE)
            loss_history.append(score[0])
            accuracy_history.append(score[1])


            if e>=1 and not freezed:
                freezed = True
                
                # freeze 1st conv2D layers
                for i in range(freeze_layers):
                    print('Freeze layer: {}'.format(model.layers[i].name))
                    model.layers[i].trainable = False
                    
                model.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer='SGD',
                    metrics=['accuracy'])

    
        print(loss_history)
        print(accuracy_history)
        all_loss.append(loss_history)
        all_acc.append(accuracy_history)
    
    print(all_loss)
    print(all_acc)

    with open(RESULT_PATH, 'w+') as f:
        f.write(', '.join(str(e) for e in all_loss))
        f.write(', '.join(str(e) for e in all_acc))
        f.write('********************')


def complex_model():
    tf.get_logger().setLevel('INFO')

    data = CIFAR10(BATCH_SIZE)
    model = tf.keras.applications.MobileNetV2(
                    input_shape=None,
                    alpha=1.0,
                    include_top=True,
                    weights="imagenet",
                    input_tensor=None,
                    pooling=None,
                    classes=1000,
                    classifier_activation="softmax")

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=tf.keras.optimizers.Adam(lr=0.00002),
                    metrics=['accuracy'])
    model.summary()

    loss_history =[]
    accuracy_history = []

    for e in range(EPOCHS):
        print('Epoch {}:'.format(e))
        
        # In each batch
        for x, y in zip(data.x_train_batch, data.y_train_batch):
            model.train_on_batch(x, y)

        score = model.evaluate(data.x_test, data.y_test, batch_size=BATCH_SIZE)
        loss_history.append(score[0])
        accuracy_history.append(score[1])

        # loss = score[0]
        # if loss <= LOSS_THRESHOLD:
        #     # freeze 1st conv2D layers
        #     for i in range(FREEZE_LAYERS*2):
        #         print('Freeze layer: {}'.format(model.layers[i].name))
        #         model.layers[i].trainable = False

    
    print(loss_history)
    print(accuracy_history)

if __name__ == '__main__':
    main()
    # complex_model()