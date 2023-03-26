import tensorflow as tf
from skimage import transform

# This module loads and transforms data from MNIST and Fashion MNIST.

def get_mnist_data(num_train = 60000, num_test = 10000, border = True, size = None):
    
    # This function retrieves the MNIST dataset at the specified size,
    # with or without the black border.

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    if not border:
        x_train = x_train[:, 4:24, 4:24]
        x_test = x_test[:, 4:24, 4:24]
    if size is not None:
        x_train = transform.resize(x_train, (60000,) + size) - 0.5
        x_test = transform.resize(x_test, (10000,) + size) - 0.5
    x_train = tf.reshape(x_train[:num_train], [num_train, -1]) 
    x_test = tf.reshape(x_test[:num_test], [num_test, -1])
    y_train = tf.constant(tf.keras.utils.to_categorical(y_train[:num_train], 10))
    y_test = tf.constant(tf.keras.utils.to_categorical(y_test[:num_test], 10))
    return ((x_train, y_train), (x_test, y_test))

def get_fashion_data(num_train = 60000, num_test = 10000, border = True, size = None, shuffle = False, **kwds):
    
    # This function retrieves the Fashion MNIST dataset at the specified size,
    # with or without the black border.
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    if not border:
        x_train = x_train[:, 4:24, 4:24]
        x_test = x_test[:, 4:24, 4:24]
    if size is not None:
        x_train = transform.resize(x_train, (60000,) + size) - 0.5
        x_test = transform.resize(x_test, (10000,) + size) - 0.5
    x_train = tf.reshape(x_train[:num_train], [num_train, -1]) 
    x_test = tf.reshape(x_test[:num_test], [num_test, -1])
    y_train = tf.constant(tf.keras.utils.to_categorical(y_train[:num_train], 10))
    y_test = tf.constant(tf.keras.utils.to_categorical(y_test[:num_test], 10))
    return ((x_train, y_train), (x_test, y_test))

def get_dataset(name, **kwrds):

    # This function loads the desired dataset and transformations.

    if name == "mnist":
        (train, test) = get_mnist_data(**kwrds)
        num_classes = 10
    elif name == "fashion":
        (train, test) = get_fashion_data(**kwrds)
        num_classes = 10
    else:
        raise ValueError(f"Dataset '{name}' not recognized.")
    return (train, test, num_classes)
