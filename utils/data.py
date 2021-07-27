import numpy as np
from tensorflow import keras

def get_data(dataset, num_classes):
    if dataset == 'mnist':
        if num_classes != 10:
            raise Exception(f'Num classes ({num_classes}) not considered for this dataset ({dataset})')
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.reshape(60000, 28, 28, 1)
        x_test = x_test.reshape(10000, 28, 28, 1)
        
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = x_train.reshape(60000, 28, 28, 1)
        x_test = x_test.reshape(10000, 28, 28, 1)
        if num_classes == 2:
            train_filter = np.where((y_train == 0 ) | (y_train == 1))
            test_filter = np.where((y_test == 0) | (y_test == 1))
            y_train = np.where(y_train == 1, 1, y_train)
            y_test = np.where(y_test == 1, 1, y_test)
            x_train, y_train = x_train[train_filter], y_train[train_filter]
            x_test, y_test = x_test[test_filter], y_test[test_filter]
        elif num_classes != 10:
            raise Exception(f'Num classes ({num_classes}) not considered for this dataset ({dataset})')
        
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        if num_classes == 2:
            train_filter = np.where((y_train == 0 ) | (y_train == 8))[0] ### IMPORTANT
            test_filter = np.where((y_test == 0) | (y_test == 8))[0] ### IMPORTANT
            y_train = np.where(y_train == 8, 1, y_train) ### IMPORTANT
            y_test = np.where(y_test == 8, 1, y_test) ### IMPORTANT
            x_train, y_train = x_train[train_filter], y_train[train_filter]
            x_test, y_test = x_test[test_filter], y_test[test_filter]
        elif num_classes != 10:
            raise Exception(f'Num classes ({num_classes}) not considered for this dataset ({dataset})')
    else:
        raise Exception("Unknown dataset: ", dataset)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, x_test, y_train, y_test
    